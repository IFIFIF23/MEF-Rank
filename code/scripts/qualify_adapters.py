#!/usr/bin/env python3
"""Bounded synthetic GPU interface checks; no market labels or effects read."""
from pathlib import Path
import sys,json,time,argparse,hashlib
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'src'))
import torch
from mef_unified.adapters import UnifiedRanker,Batch,HOOKS
from benchmark_suite.contracts import FinancialBatch
from mef_rank.str_model.training import base_str_loss

def make_batch(device,n=6):
    x=torch.randn(1,n,60,125,device=device);fm=torch.ones_like(x,dtype=torch.bool)
    mask=torch.ones((1,n),device=device,dtype=torch.bool)
    ranks=torch.linspace(0,1,n,device=device)[None]
    return Batch(FinancialBatch(x,fm,torch.ones((1,n,60),device=device,dtype=torch.bool),mask,ranks,ranks,mask),torch.zeros(1,5,device=device),torch.ones(1,5,device=device,dtype=torch.bool))

def check(name,device):
    torch.manual_seed(1221); batch=make_batch(device);model=UnifiedRanker(name).to(device);model.eval()
    beta=torch.tensor([[.15,-.12,.18,-.2,.11,-.13,.17]],device=device,requires_grad=True)
    c=model.encode_action_independent(batch)
    zero=model.predict_with_action(c,torch.zeros_like(beta)); changed=model.predict_with_action(c,beta)
    grad=torch.autograd.grad(changed.score.sum(),beta)[0]
    assert torch.isfinite(grad).all() and (grad.abs()>1e-10).all(),(name,grad)
    assert not torch.equal(zero.score,changed.score),name
    with torch.no_grad():
        direct=model(batch);cached=model.predict_with_action(model.encode_action_independent(batch),torch.zeros_like(beta))
        assert torch.equal(direct.score,cached.score),(name,'cache')
        first,fallback=model.with_memory_action(batch,None);assert first is fallback
        repeat=model(batch);assert torch.equal(direct.score,repeat.score),(name,'eval nondeterministic')
        reloaded=UnifiedRanker(name).to(device).eval();reloaded.load_state_dict(model.state_dict());assert torch.equal(direct.score,reloaded(batch).score)
        del reloaded
        # Missing groups must have zero reliability and cannot acquire support from beta.
        batch.financial.feature_mask[...,24:32]=False;batch.financial.x[...,24:32]=0
        absent=model(batch,beta.detach());assert torch.count_nonzero(absent.reliability[...,3])==0
    model.train();optim=torch.optim.AdamW(model.parameters(),lr=3e-4)
    out=model(batch);loss=base_str_loss(out,batch.native(),pair_loss_weight=.2,maximum_pairs=4096,minimum_rank_gap=.1,namespace='qualification').total
    if getattr(out,'auxiliary_loss',None) is not None:loss=loss+.01*out.auxiliary_loss.mean()
    loss.backward();finite=all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);assert finite
    updated=[p for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad)>0]
    assert updated, 'no trainable gradient'
    before=updated[-1].detach().clone();optim.step();assert not torch.equal(before,updated[-1])
    return {'status':'PASS','hook':HOOKS[name],'beta_gradient':grad.detach().cpu().tolist(),'score_action_max_delta':float((changed.score-zero.score).abs().max().detach()),'training_loss':float(loss.detach()),'parameters':sum(p.numel() for p in model.parameters()),'cuda_peak_mib':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None}

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);a.add_argument('--device',default='cuda');args=a.parse_args()
    torch.set_num_threads(2);dev=torch.device(args.device)
    if dev.type=='cuda':torch.cuda.set_per_process_memory_fraction(.15)
    start=time.time();result=check(args.model,dev);result.update(model=args.model,device=str(dev),seconds=time.time()-start)
    result['source_sha256']={str(p.relative_to(BASE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'src/mef_unified').glob('*.py')}
    out=BASE/'reports/audits/adapters_v001';out.mkdir(parents=True,exist_ok=True)
    (out/f'{args.model}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
