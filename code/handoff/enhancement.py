"""The same nested Writer, CAL1 and bounded action execution for every backbone."""
from dataclasses import replace
import hashlib
import json
from collections import Counter
import numpy as np
import torch
from mef_rank.mbs.observable import build_mbs
from mef_rank.v36.jacf_cal0 import JointState
from mef_rank.inverse import invert_reliability_shift
from mef_delivery.contracts import V371WriterConfig
from mef_delivery.writer import solve_nested_writer
from mef_delivery.memory import build_v37_memory_record, V37MemoryRecord
from mef_delivery.cal1 import fit_cal1, run_cal1, _weighted_quantile


PROFILES = {'gem': (200,100,40,50), 'main': (200,100,40,50), 'sp500': (200,100,40,50), 'nasdaq100': (60,29,13,18), 'crypto': (15,7,3,5)}


def writer_config(family):
    n,d,m,o=PROFILES[family]
    return V371WriterConfig(selection_k=50, minimum_stocks=n, direction_minimum_stocks=d, magnitude_minimum_stocks=m, outer_minimum_stocks=o)


def query_state(panel,index,scaler,first,cache):
    minimum = 5 if panel.reader.family == 'crypto' else 20
    b=build_mbs(first.detached_numpy(), cache.signal_universe_mask.cpu().numpy(), cache.group_available_mask.cpu().numpy(), group_min_stocks=minimum,pair_min_common_stocks=minimum)
    j,jmask=panel.joint_state(index,scaler)
    return JointState(j,jmask,b.full_values[0],b.full_mask[0])


def first_forward(model,batch):
    model.eval()
    with torch.no_grad():
        cache=model.encode_action_independent(batch)
        first=model.rank_from_cache(cache,beta_logit=cache.base_gate_logits.new_zeros((1,7)))
    return cache,first


def solve_record(model,panel,index,scaler,device,cutoff,episode,identity,checkpoint_hash,source_lineage):
    batch=panel.batch(index,device,cutoff,scaler)
    cache,first=first_forward(model,batch)
    state=query_state(panel,index,scaler,first,cache)
    mask=batch.financial.label_valid[0] & batch.financial.stock_mask[0]
    config=writer_config(panel.reader.family)
    if int(mask.sum())<config.minimum_stocks:
        return None, {'reason':'WRITER_INSUFFICIENT_STOCKS','n':int(mask.sum())}
    p=panel.panel; positions=p.eligible_indices(index,require_label=False)
    maturities=p.label_maturity[index,positions][mask.cpu().numpy()]
    if len(maturities)==0 or np.any(maturities=='') or np.any(maturities>cutoff):raise ValueError('Writer labels not matured')
    # Proxies use current input observations only. Missing fundamentals collapse strata.
    proxies=panel.proxies(index,batch.codes)
    result=solve_nested_writer(model,cache,first,raw_returns=batch.financial.label_return[0].cpu().numpy(),writer_mask=mask,cap_proxy=proxies[0],liquidity_proxy=proxies[1],stock_ids=np.asarray(batch.codes),namespace=f'{identity}|{episode}|{batch.date}',config=config)
    audit={'reason':result.reason_code,'qualified':result.qualified,'valid_repeats':result.valid_repeat_count,'n':int(mask.sum()),'outer_summary':result.outer_summary,'repeats':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in vars(r).items()} for r in result.repeats]}
    # Rejected records are kept in the audit, never passed as usable Memory.
    if not result.qualified: return None,audit
    matured=str(max(maturities))
    record=build_v37_memory_record(writer=result,anchor_id=f'{identity}:{episode}:{batch.date}',feature_date=batch.date,episode_id=episode,checkpoint_id=episode,label_maturity_timestamp=matured,memory_eligible_timestamp=matured,state=state,oob_gain_reference=1.,checkpoint_identity={'chain_identity':identity,'checkpoint_sha256':checkpoint_hash,'adapter':'unified_handoff_v1'},input_and_scaler_lineage={'scaler_sha256':scaler['sha256'],'target_date':batch.date,**source_lineage},lifecycle_dates={'feature_date':batch.date,'label_maturity_date':matured,'memory_eligible_date':matured,'writer_completion_policy':'simulated_target_maturity_close; strict subsequent-date queries only','source_time_independent':False},selection_k=50)
    return record,audit


def calibrate(records):
    positive=[r for r in records if r.qualified and r.outer_primary_gain_median>0]
    if not positive:return records,None,{'status':'fallback','reason':'NO_QUALIFIED_MEMORY','episodes':0}
    counts=Counter(r.episode_id for r in positive)
    reference=_weighted_quantile([r.outer_primary_gain_median for r in positive],[1/counts[r.episode_id] for r in positive],.5)
    corrected=[]
    for r in records:
        gain=float(np.clip(r.outer_primary_gain_median/reference,0,1))
        stability=float(np.clip(2*((r.outer_positive_primary_gain_count+1)/(r.valid_repeat_count+2)-.5),0,1))
        corrected.append(replace(r,oob_gain_reference=reference,oob_quality_gain=gain,oob_quality_stability=stability,q_anchor=float(np.sqrt(gain*stability))))
    if len(counts)<20:return corrected,None,{'status':'fallback','reason':'CAL1_REFERENCE_SHORTFALL','episodes':len(counts),'required':20,'gain_reference':reference}
    try:
        cal,loeo=fit_cal1(corrected)
    except ValueError as exc:
        return corrected,None,{'status':'fallback','reason':'CAL1_REFERENCE_INVALID','detail':str(exc),'episodes':len(counts)}
    return corrected,cal,{'status':'ready','episodes':len(counts),'gain_reference':reference,'loeo':loeo}


def execute_shift(model,cache,first,target):
    target=np.asarray(target,dtype=np.float64)
    valid=cache.group_available_mask[0].cpu().numpy() & cache.signal_universe_mask[0,:,None].cpu().numpy()
    logits=cache.base_gate_logits[0].detach().cpu().numpy().astype(np.float64)
    if target.shape!=(7,) or not np.isfinite(target).all():return first,{'apply':False,'reason':'NONFINITE_ACTION'}
    inverse=invert_reliability_shift(logits,valid,target,action_cap=.5,iterations=24,tolerance=1e-5)
    supported=valid.any(axis=0)
    if np.any(~supported & (np.abs(target)>1e-12)) or inverse.per_group_clipped.any() or np.max(np.abs(inverse.projection_error))>1e-6 or not inverse.converged[supported].all() or np.max(np.abs(inverse.solver_residual))>1e-5:
        return first,{'apply':False,'reason':'ACTION_INVERSE_REJECTED'}
    beta=torch.as_tensor(inverse.delta_beta.copy(),device=cache.base_gate_logits.device,dtype=cache.base_gate_logits.dtype)[None]
    with torch.no_grad():final=model.rank_from_cache(cache,beta_logit=beta)
    if not torch.isfinite(final.score).all():return first,{'apply':False,'reason':'NONFINITE_SECOND_FORWARD'}
    return final,{'apply':True,'reason':'APPLY','beta':inverse.delta_beta.tolist(),'target_shift':target.tolist(),'inverse_residual_max':float(np.max(np.abs(inverse.solver_residual)))}


def infer_on(model,cache,first,state,records,calibration,date,identity):
    if any(r.checkpoint_identity.get('chain_identity')!=identity for r in records):raise ValueError('Cross-model/dataset/seed Memory rejected')
    if calibration is None:return first,{'apply':False,'reason':'CAL1_UNAVAILABLE'}
    result=run_cal1(state,records,calibration,query_date=date)
    if not result.apply:return first,{'apply':False,'reason':result.reason_code}
    final,trace=execute_shift(model,cache,first,result.target_shift)
    trace.update(episodes=list(result.representative_episode_ids),weights=result.weights.tolist(),effective_sample_size=result.effective_sample_size,coverage=result.coverage,quality=result.quality,confidence=result.confidence)
    return final,trace
