"""Group-reliability actions enter the preserved backbone BEFORE its core.

No forecast cache or prediction-side correction branch is used. Hooks exist
only during the synchronous official forward and are always removed afterward.
"""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import json, math
import torch
from torch import nn
from .boot import BASE
from benchmark_suite.contracts import FinancialBatch, GROUP_ORDER, GROUP_DIMENSIONS, group_slices
from benchmark_suite.official import load_official_class, source_record, _BroadcastAttentionMaskCache, _RegularDailyChannelTokenForecast
from mef_rank.str_model.network import MEFRankSTR, _GroupProjection, _GateMLP, _masked_query_pool, TorchSTRForward
from mef_rank.str_model.day_batch import DayBatch

SOURCES = dict(zip(('iTransformer','TimeMixer','ModernTCN','SOFTS','TimeXer','TimeBridge','TimeFilter','SRSNet','ChannelTokenFormer'),('m01_itransformer','m02_timemixer','m03_moderntcn','m04_softs','m05_timexer','m06_timebridge','m07_timefilter','m09_srsnet','m10_channeltokenformer')))
HOOKS = {'STR':'native FSA -> Temporal -> cross-sectional', 'iTransformer':'enc_embedding output -> encoder', 'SOFTS':'enc_embedding output -> STAR encoder', 'ModernTCN':'model.stages[0] input after stem/RevIN -> all stages', 'TimeMixer':'each normalize_layers[i] norm output -> pre_enc/PDM/FMM', 'TimeXer':'en_embedding/ex_embedding output -> endogenous/exogenous encoder', 'TimeBridge':'embedding input after normalization -> bridges', 'TimeFilter':'norm output -> patch_embed -> graph backbone', 'SRSNet':'revin norm output -> selection/reorganization/fusion', 'ChannelTokenFormer':'encoder input after patch/normalization -> token interactions'}

@dataclass
class Batch:
    financial: FinancialBatch
    shock: torch.Tensor
    shock_mask: torch.Tensor
    date: str = 'synthetic'
    codes: tuple = ()

    def slice(self, start, end):
        f=self.financial
        sub=FinancialBatch(**{k:(v[:,start:end] if v is not None else None) for k,v in vars(f).items()})
        return Batch(sub,self.shock,self.shock_mask,self.date,self.codes[start:end])

    def native(self):
        f=self.financial; B,N=f.x.shape[:2]
        xs=dict(zip(GROUP_ORDER,torch.split(f.x,GROUP_DIMENSIONS,dim=-1)))
        masks=dict(zip(GROUP_ORDER,torch.split(f.feature_mask,GROUP_DIMENSIONS,dim=-1)))
        lag=torch.stack([masks[g].all(-1)&f.history_mask for g in GROUP_ORDER],-1)
        empty=f.x.new_zeros((B,0)); em=empty.bool()
        return DayBatch([self.date]*B,[list(self.codes or range(N))]*B,xs,masks,f.history_mask,lag,f.stock_mask,f.stock_mask,lag.any(2),empty,{5:empty},em,{5:em},self.shock,self.shock_mask,label_return=f.label_return,label_rank=f.label_rank,label_valid_mask=f.label_valid)

@dataclass
class ExternalCache:
    batch: Batch
    pooled: torch.Tensor
    base_gate_logits: torch.Tensor
    group_available_mask: torch.Tensor

    @property
    def signal_universe_mask(self):
        return self.batch.financial.stock_mask

    def detached(self):
        return replace(self, pooled=self.pooled.detach(), base_gate_logits=self.base_gate_logits.detach())

@dataclass(frozen=True)
class Prediction(TorchSTRForward):
    auxiliary_loss: torch.Tensor | None = None

class GroupEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections=nn.ModuleList([_GroupProjection(n,96) for n in GROUP_DIMENSIONS])
        self.gates=nn.ModuleList([_GateMLP(96,48) for _ in range(7)])
        self.position=nn.Parameter(torch.randn(60,96)*96**-.5)
        self.queries=nn.Parameter(torch.randn(7,96)*96**-.5)

    def forward(self,batch):
        f=batch.financial; pooled=[]; logits=[]; available=[]
        for g,s in enumerate(group_slices()):
            mask=f.feature_mask[...,s].all(-1)&f.history_mask
            value=f.x[...,s].masked_fill(~f.feature_mask[...,s],0)
            tokens=self.projections[g](value)+self.position[None,None]
            tokens=tokens.masked_fill(~mask[...,None],0)
            h=_masked_query_pool(tokens,mask,self.queries[g]); avail=mask.any(-1)
            h=h.masked_fill(~avail[...,None],0)
            z=(2.5*torch.tanh(self.gates[g](h)/2.5)).masked_fill(~avail,0)
            pooled.append(h); logits.append(z); available.append(avail)
        return ExternalCache(batch,torch.stack(pooled,2),torch.stack(logits,2),torch.stack(available,2))

class UnifiedRanker(nn.Module):
    def __init__(self,name,profile=None):
        super().__init__(); self.name=name
        if name=='STR':
            self.native=MEFRankSTR(); self.profile=vars(self.native.profile)
            return
        self.profile=profile or json.loads((BASE/'config/training_v001.json').read_text())['profiles'][name]
        sid=SOURCES[name]; self.source=source_record(sid)
        self.core=load_official_class(sid)(SimpleNamespace(**self.profile))
        self.input=GroupEncoder()
        self.channels=1 if name=='TimeXer' else 125
        self.rank_head=nn.Linear(self.channels,20); self.residual_head=nn.Linear(self.channels,1)
        self.register_buffer('rank_support',torch.linspace(0,1,20))
        self.register_buffer('feature_group',torch.repeat_interleave(torch.arange(7),torch.tensor(GROUP_DIMENSIONS)))
        if name=='ChannelTokenFormer':
            for layer in self.core.encoder.layers:
                layer.build_attention_mask=_BroadcastAttentionMaskCache(layer.build_attention_mask)
            self.core.forecast_multi=_RegularDailyChannelTokenForecast(self.core)
        if name=='TimeFilter':
            n=125*(60//self.profile['patch_len']); q=torch.arange(n)[:,None]; k=torch.arange(n)[None,:]; p=60//self.profile['patch_len']
            spatial=(k%p==q%p)&(k!=q); temporal=k//p==q//p
            self.register_buffer('structural',torch.stack((spatial,temporal,~(spatial|temporal)),1).float(),persistent=False)

    def encode_action_independent(self,batch):
        if self.name=='STR': return self.native.encode_once(batch.native(),horizon=5)
        return self.input(batch)

    def predict_with_action(self,cache,beta):
        if self.name=='STR': return self.native.rank_from_cache(cache,beta_logit=beta)
        if beta.shape!=(cache.base_gate_logits.shape[0],7) or not torch.isfinite(beta).all():
            raise ValueError('beta must be finite [B,7]')
        f=cache.batch.financial; B,N,L,C=f.x.shape
        r=torch.sigmoid(cache.base_gate_logits+beta[:,None]).masked_fill(~cache.group_available_mask,0)
        scale=r[...,self.feature_group].reshape(B*N,125)
        x=f.x.masked_fill(~f.feature_mask,0).reshape(B*N,L,C)
        handles=[]
        def post(module,func): handles.append(module.register_forward_hook(func))
        def pre(module,func): handles.append(module.register_forward_pre_hook(func))
        def norm_hook(mod,args,out):
            return out*scale[:,None,:] if args[1]=='norm' else out
        name=self.name; core=self.core
        if name in ('iTransformer','SOFTS'):
            post(core.enc_embedding,lambda m,a,o:o*scale[:,:,None])
        elif name=='ModernTCN':
            pre(core.model.stages[0],lambda m,a:(a[0]*scale[:,:,None,None],*a[1:]))
        elif name=='TimeMixer':
            for layer in core.normalize_layers: post(layer,norm_hook)
        elif name=='TimeXer':
            idx=int(self.profile['endogenous_feature_index']); order=[i for i in range(125) if i!=idx]+[idx]; x=x[...,order]
            post(core.en_embedding,lambda m,a,o:(o[0]*scale[:,idx,None,None],o[1]))
            post(core.ex_embedding,lambda m,a,o:o*scale[:,order[:-1],None])
        elif name=='TimeBridge':
            pre(core.embedding,lambda m,a:(a[0]*scale[:,None,:],*a[1:]))
        elif name=='TimeFilter': post(core.norm,norm_hook)
        elif name=='SRSNet': post(core.revin,norm_hook)
        elif name=='ChannelTokenFormer':
            pre(core.encoder,lambda m,a:(a[0]*scale.repeat_interleave(7,dim=1)[:,:,None],*a[1:]))
        try:
            if name=='SRSNet': out=core(x)
            elif name=='ModernTCN': out=core(x,te=None)
            elif name=='TimeFilter': out=core(x,masks=self.structural,is_training=self.training)
            else: out=core(x,None,None,None)
        finally:
            for h in handles: h.remove()
        aux=None
        if isinstance(out,tuple): out,aux=out
        if out.shape!=(B*N,1,self.channels): raise ValueError(f'bad upstream output {out.shape}')
        rep=out[:,-1].reshape(B,N,self.channels)
        logits=self.rank_head(rep); probability=logits.softmax(-1)
        score=(probability*self.rank_support).sum(-1)+.1*self.residual_head(rep).squeeze(-1).tanh()
        entropy=-(probability*probability.clamp_min(1e-12).log()).sum(-1)/math.log(20)
        invalid=~f.stock_mask
        result=Prediction(score.masked_fill(invalid,0),logits.masked_fill(invalid[...,None],0),probability.masked_fill(invalid[...,None],0),entropy.masked_fill(invalid,0),r,r[...,None]*cache.pooled,aux)
        return result

    def forward(self,batch,beta=None):
        cache=self.encode_action_independent(batch)
        if beta is None: beta=cache.base_gate_logits.new_zeros((cache.base_gate_logits.shape[0],7))
        return self.predict_with_action(cache,beta)

    def with_memory_action(self,batch,beta=None):
        cache=self.encode_action_independent(batch)
        zero=cache.base_gate_logits.new_zeros((cache.base_gate_logits.shape[0],7))
        first=self.predict_with_action(cache,zero)
        return first,first if beta is None else self.predict_with_action(cache,beta)


# Writer adapter: preserves the complete supervision universe while splitting
# only independent external backbone computations. Native STR is never split.
def _rank_from_cache(self, cache, *, beta_logit):
    if self.name == 'STR':
        return self.predict_with_action(cache, beta_logit)
    from torch.utils.checkpoint import checkpoint
    outputs = []
    n = cache.base_gate_logits.shape[1]
    cap = int(self.profile['stock_batch_cap'])
    for start in range(0, n, cap):
        stop = min(n, start + cap)
        sub = replace(cache, batch=cache.batch.slice(start, stop), pooled=cache.pooled[:,start:stop], base_gate_logits=cache.base_gate_logits[:,start:stop], group_available_mask=cache.group_available_mask[:,start:stop])
        def forward(beta, sub=sub):
            out = self.predict_with_action(sub, beta)
            return tuple(getattr(out, name) for name in ('score','rank_logits','rank_probability','uncertainty','reliability','weighted_group_pooled'))
        values = checkpoint(forward, beta_logit, use_reentrant=False) if torch.is_grad_enabled() and beta_logit.requires_grad else forward(beta_logit)
        outputs.append(values)
    return Prediction(*(torch.cat([v[i] for v in outputs], dim=1) for i in range(6)))

UnifiedRanker.rank_from_cache = _rank_from_cache
