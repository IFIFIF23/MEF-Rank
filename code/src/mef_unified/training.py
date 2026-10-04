"""Main-predictor training. Memory/On remain separately recorded downstream stages."""
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,math,os,random,time,uuid
import numpy as np
import pandas as pd
import torch
from torch.utils.checkpoint import checkpoint
from .boot import BASE,ROOT
from .adapters import UnifiedRanker,Batch,SOURCES
from benchmark_suite.official import source_record
from benchmark_suite.data import load_panel
from mef_rank.str_model.training import base_str_loss
from mef_rank.shock.builder import fit_shock_scaler
from src.evaluation.metrics import _spearman

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temp,path)

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def verify_shards(binding):
    for rel,expected in binding['sha256'].items():
        if sha(ROOT/rel)!=expected:raise ValueError(f'manifest changed: {rel}')
    root=ROOT/binding['main_training_view'];manifest=json.loads((root/'manifest.json').read_text())
    entries={}
    for entry in manifest.get('file_inventory',[]):entries[entry['path']]=entry['sha256']
    for record in manifest.get('annual_outputs',{}).values():
        for entry in record.values():
            if isinstance(entry,dict) and 'path' in entry and 'sha256' in entry:entries[entry['path']]=entry['sha256']
    verified={}
    for year in range(2018,2023):
        for kind in ('stock_daily','market_daily'):
            rel=f'year={year}/{kind}.parquet'
            if rel not in entries:raise ValueError(f'unregistered shard: {root}/{rel}')
            digest=sha(root/rel)
            if digest!=entries[rel]:raise ValueError(f'changed shard: {root}/{rel}')
            verified[str((root/rel).relative_to(ROOT))]=digest
    return root,verified

class TrainingPanel:
    def __init__(self,root,dataset):
        self.panel=load_panel(name=dataset,root=root,years=list(range(2018,2023)))
        frames=[pd.read_parquet(root/f'year={year}/market_daily.parquet') for year in range(2018,2023)]
        market=pd.concat(frames).set_index('feature_date');market.index=market.index.astype(str);market=market.reindex(self.panel.dates)
        names=[x for x in market if x.startswith('shock_causal__')]
        masks=[x.replace('shock_causal__','shock_causal_available__') for x in names]
        if len(names)!=5:raise ValueError('shock schema')
        self.shock=market[names].to_numpy(np.float64)
        self.shock_mask=market[masks].fillna(False).to_numpy(bool)
        self.shock[~self.shock_mask]=0
        if not np.isfinite(self.shock).all():raise ValueError('nonfinite observed shock')

    def scaler(self,end):
        s=fit_shock_scaler(tuple(self.panel.dates),self.shock,self.shock_mask,fit_date_max=end,fit_date_min=self.panel.dates[0])
        return {'center':s.center.tolist(),'scale':s.scale.tolist(),'supported':s.supported.tolist(),'fit_date_max':end,'fit_date_min':s.fit_date_min,'sha256':s.scaler_hash,'source':'shock_causal, checkpoint-local median/MAD; reference shock_network not used'}

    def batch(self,i,device,cutoff,scaler):
        p=self.panel;stocks=p.eligible_indices(i,require_label=False)
        f=p.make_batch(i,stocks,device,maturity_cutoff=cutoff)
        mask=self.shock_mask[i]&np.asarray(scaler['supported'])
        value=(self.shock[i]-np.asarray(scaler['center']))/np.asarray(scaler['scale']);value[~mask]=0
        return Batch(f,torch.tensor(value[None],dtype=torch.float32,device=device),torch.tensor(mask[None],device=device),p.dates[i],tuple(p.codes[j] for j in stocks))

@contextmanager
def preserve_bn_buffers(model):
    saved=[(m,m.running_mean.clone(),m.running_var.clone(),m.num_batches_tracked.clone()) for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
    try:yield
    finally:
        with torch.no_grad():
            for m,mean,var,num in saved:m.running_mean.copy_(mean);m.running_var.copy_(var);m.num_batches_tracked.copy_(num)

def whole_cross_section(model,batch):
    """One full-date loss; chunk only stock-independent external cores.

    Activation recomputation preserves RNG and prevents duplicate BN updates.
    Native STR's cross-stock attention is NEVER split into stock minibatches.
    """
    if model.name=='STR':return model(batch)
    n=batch.financial.x.shape[1];cap=int(model.profile['stock_batch_cap']);scores=[];logits=[];auxs=[]
    for start in range(0,n,cap):
        sub=batch.slice(start,min(n,start+cap))
        def run(x,sub=sub):
            sub=replace(sub,financial=replace(sub.financial,x=x));o=model(sub)
            aux=getattr(o,'auxiliary_loss',None)
            if aux is None:aux=o.score.new_zeros(())
            return o.score,o.rank_logits,aux.mean()
        if model.training:
            vals=checkpoint(run,sub.financial.x,use_reentrant=False,preserve_rng_state=True,context_fn=lambda:(nullcontext(),preserve_bn_buffers(model)))
        else:vals=run(sub.financial.x)
        scores.append(vals[0]);logits.append(vals[1]);auxs.append(vals[2]*sub.financial.x.shape[1]/n)
    return SimpleNamespace(score=torch.cat(scores,1),rank_logits=torch.cat(logits,1),auxiliary_loss=torch.stack(auxs).sum())

def loss_value(output,batch,namespace,cfg):
    train=cfg['training'];loss=base_str_loss(output,batch.native(),pair_loss_weight=train['pair_weight'],maximum_pairs=train['maximum_pairs'],minimum_rank_gap=train['minimum_rank_gap'],namespace=namespace).total
    aux=getattr(output,'auxiliary_loss',None)
    return loss if aux is None else loss+train['auxiliary_weight']*aux

def indices(panel,cutoff):
    return [i for i in range(59,len(panel.dates)) if panel.dates[i]<=cutoff and len(panel.eligible_indices(i,require_label=True,maturity_cutoff=cutoff))>=2]

def evaluate(model,panel,days,device,cutoff,scaler,retain=False):
    model.eval();daily=[];frames=[]
    with torch.no_grad():
        for i in days:
            b=panel.batch(i,device,cutoff,scaler);o=whole_cross_section(model,b)
            valid=b.financial.label_valid[0].cpu().numpy();score=o.score[0].cpu().numpy();label=b.financial.label_return[0].cpu().numpy()
            ic=_spearman(score[valid],label[valid]);daily.append({'trade_date':b.date,'rankic':float(ic) if np.isfinite(ic) else None,'n':int(valid.sum())})
            if retain:frames.append(pd.DataFrame({'trade_date':b.date,'ts_code':np.array(b.codes)[valid],'arm':'off','y_pred':score[valid],'label_H5__primary_return':label[valid]}))
    finite=[r['rankic'] for r in daily if r['rankic'] is not None]
    if not finite:raise ValueError('no evaluable internal RankIC dates')
    return float(np.mean(finite)),daily,pd.concat(frames,ignore_index=True) if frames else None
