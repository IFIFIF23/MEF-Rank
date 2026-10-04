"""Portable main -> quarterly history -> Writer -> Memory -> CAL1 -> paired prediction."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import numpy as np
import pandas as pd
import torch
from mef_unified.boot import BASE
from src.evaluation.metrics import _spearman, _nw_tstat
from handoff.secure_data import EncryptedDataset
from handoff.panel import Panel
from handoff.trainer import fit_predictor, load_predictor
from handoff.enhancement import first_forward, query_state, solve_record, calibrate, infer_on, PROFILES
from mef_delivery.memory import V37MemoryRecord
from mef_delivery.cal1 import CAL1Values

FAMILIES={'gem':'gem1','main':'main1','nasdaq100':'us1_nasdaq100','sp500':'us1_sp500','crypto':'crypto1'}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not math.isfinite(value):return None
    return value


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n');temp.replace(path)


def quarters(dates,cutoff):
    dates=np.asarray(dates)
    first_year=int(dates[0][:4])+5
    result=[]
    for year in range(first_year,int(cutoff[:4])+1):
        for month in (1,4,7,10):
            start=f'{year:04d}{month:02d}01'
            following=f'{year+1:04d}0101' if month==10 else f'{year:04d}{month+3:02d}01'
            anchors=np.flatnonzero((dates>=start)&(dates<following)&(dates<=cutoff))
            prior=np.flatnonzero(dates<start)
            if not len(anchors) or not len(prior):continue
            fit_start=f'{year-5:04d}{month:02d}01';fit_end=str(dates[prior[-1]])
            result.append({'episode':f'{year}Q{(month-1)//3+1}','fit_start':fit_start,'fit_end':fit_end,'anchors':anchors.tolist()})
    return result


def build_history(reader,args,cfg,run,device,identity,cutoff):
    bank_path=run/'memory.json';cal_path=run/'calibration.json'
    if bank_path.exists() and cal_path.exists():
        payload=json.loads(bank_path.read_text());cal=json.loads(cal_path.read_text())
        if payload['chain_identity']!=identity or cal['chain_identity']!=identity:raise ValueError('Incompatible Memory bank')
        if digest(bank_path)!=cal['memory_sha256']:raise ValueError('Memory changed after calibration')
        return [V37MemoryRecord.from_dict(v) for v in payload['records']],CAL1Values.from_dict(cal['values']) if cal['values'] else None
    panel=Panel(reader,'1_memory')
    plan=quarters(panel.panel.dates,cutoff)
    if args.smoke:plan=plan[:2]
    write(run/'history_plan.json',plan)
    mappings=reader.lineage('1_memory')
    records=[]
    for slot in plan:
        ep=slot['episode'];destination=run/'history'/ep
        print(f'history {ep}: {slot["fit_start"]}..{slot["fit_end"]}',flush=True)
        model,scaler=fit_predictor(panel,args.model,args.seed,slot['fit_start'],slot['fit_end'],cfg,destination,device,args.smoke)
        checkpoint_hash=digest(destination/'model.pt')
        anchors=slot['anchors'][:1] if args.smoke else slot['anchors']
        for index in anchors:
            date=panel.panel.dates[index]
            record_path=destination/'writer'/(date+'.json')
            if record_path.exists():
                row=json.loads(record_path.read_text())
                if row.get('chain_identity')!=identity or row.get('checkpoint_sha256')!=checkpoint_hash:raise ValueError('Writer checkpoint identity changed')
                if row['record']:records.append(V37MemoryRecord.from_dict(row['record']))
                continue
            if date not in mappings:raise ValueError('Missing source/target lineage')
            started=time.monotonic()
            record,audit=solve_record(model,panel,index,scaler,device,cutoff,ep,identity,checkpoint_hash,{'source_mapping':mappings[date],'source_time_independent':False})
            row={'chain_identity':identity,'checkpoint_sha256':checkpoint_hash,'target_date':date,'built_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,'record':record.to_dict() if record else None,'audit':audit}
            write(record_path,row)
            if record:records.append(record)
            print(f'writer {ep}/{date}: {audit["reason"]}',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    del panel
    records,calibration,report=calibrate(records)
    write(bank_path,{'chain_identity':identity,'training_cutoff':cutoff,'records':[r.to_dict() for r in records]})
    write(cal_path,{'chain_identity':identity,'values':calibration.to_dict() if calibration else None,'report':report,'memory_sha256':digest(bank_path),'fit_target_cutoff':cutoff})
    return records,calibration


def paired_evaluate(reader,args,binding,run,device,model,scaler,records,calibration,identity):
    panel=Panel(reader,'evaluation')
    start=binding['validation_range'][0];end=binding['test_range'][1]
    eligible=[i for i,d in enumerate(panel.panel.dates) if i>=59 and start<=d<=end and len(panel.panel.eligible_indices(i,require_label=True,maturity_cutoff=end))>=2]
    if args.smoke:eligible=eligible[:2]
    parts=run/'prediction_parts';parts.mkdir(exist_ok=True)
    daily=[]
    for i in eligible:
        date=panel.panel.dates[i];part=parts/(date+'.parquet');trace_path=parts/(date+'.json')
        if part.exists() and trace_path.exists():
            row=json.loads(trace_path.read_text())
            if row['chain_identity']!=identity:raise ValueError('Prediction identity mismatch')
            if args.mode=='off' or row['reason']!='OFF_REQUESTED':
                daily.append(row);continue
        batch=panel.batch(i,device,end,scaler)
        cache,first=first_forward(model,batch)
        if args.mode=='off':final=first;trace={'apply':False,'reason':'OFF_REQUESTED'}
        else:
            state=query_state(panel,i,scaler,first,cache)
            final,trace=infer_on(model,cache,first,state,records,calibration,date,identity)
        valid=batch.financial.label_valid[0].cpu().numpy()
        off=first.score[0].cpu().numpy();on=final.score[0].cpu().numpy();labels=batch.financial.label_return[0].cpu().numpy()
        valid=valid & np.isfinite(off) & np.isfinite(on) & np.isfinite(labels)
        if not trace['apply'] and not np.array_equal(off,on):raise AssertionError('Fallback must be exact')
        common={'trade_date':date,'ts_code':np.asarray(batch.codes)[valid],'label_H5__primary_return':labels[valid]}
        frames=[pd.DataFrame({**common,'arm':'off','y_pred':off[valid]})]
        if args.mode!='off':frames.append(pd.DataFrame({**common,'arm':'on','y_pred':on[valid]}))
        frame=pd.concat(frames,ignore_index=True)
        temp=part.with_suffix('.tmp');frame.to_parquet(temp,index=False);temp.replace(part)
        row={'chain_identity':identity,'trade_date':date,'split':'validation' if date<=binding['validation_range'][1] else 'test','n':int(valid.sum()),'rankic_off':float(_spearman(off[valid],labels[valid])),'rankic_on':float(_spearman(on[valid],labels[valid])),**trace}
        write(trace_path,row);daily.append(json_safe(row))
        print(f'evaluate {date}: {trace["reason"]}',flush=True)
    if not daily:raise ValueError('No evaluation dates')
    import pyarrow.parquet as pq
    output=run/'predictions.parquet';temp=output.with_suffix('.tmp')
    writer=None
    try:
        for row in daily:
            table=pq.ParquetFile(parts/(row['trade_date']+'.parquet')).read()
            if writer is None:writer=pq.ParquetWriter(temp,table.schema,compression='zstd')
            if args.mode=='off':
                import pyarrow.compute as pc
                table=table.filter(pc.equal(table['arm'],'off'))
            writer.write_table(table)
    finally:
        if writer:writer.close()
    temp.replace(output)
    pd.DataFrame(daily).to_csv(run/'daily_metrics.csv',index=False)
    summaries={}
    for split in ('validation','test'):
        rows=[r for r in daily if r['split']==split and r['rankic_off'] is not None and r['rankic_on'] is not None]
        if not rows:
            summaries[split]={'n_days':0};continue
        off=np.asarray([r['rankic_off'] for r in rows]);on=np.asarray([r['rankic_on'] for r in rows]);delta=on-off
        summaries[split]={'n_days':len(rows),'rankic_off':off.mean(),'rankic_on':on.mean() if args.mode!='off' else None,'delta_rankic':delta.mean() if args.mode!='off' else None,'delta_nw_t_lag10':_nw_tstat(delta,lags=10) if len(rows)>11 and args.mode!='off' else None,'apply_days':sum(r['apply'] for r in rows),'reasons':dict(Counter(r['reason'] for r in rows))}
    write(run/'metrics.json',{'scope':'engineering_smoke' if args.smoke else 'source_exposed_diagnostic','primary':'daily H5 RankIC; no executable net-return backtest','splits':summaries})


def run_pipeline(args):
    cfg=json.loads((BASE/'config/training_v001.json').read_text())
    if cfg['do_not_execute']:raise ValueError('Non-executable training configuration')
    if args.epochs is not None:
        if args.epochs<1:raise ValueError('epochs must be positive')
        cfg['training']['max_epochs']=args.epochs
        cfg['training']['min_epochs']=min(cfg['training']['min_epochs'],args.epochs)
    binding=next(d for d in cfg['datasets'] if d['dataset_id']==FAMILIES[args.dataset])
    reader=EncryptedDataset(args.data_dir,args.dataset,args.key_file)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; install compatible PyTorch or use --device cpu')
    torch.set_num_threads(args.threads)
    code_hashes={str(f.relative_to(BASE)):digest(f) for folder in ('handoff','src','vendor') for f in (BASE/folder).rglob('*.py')}
    code_hashes['train.py']=digest(BASE/'train.py')
    code_hashes['third_party/source_hashes.json']=digest(BASE/'third_party/source_hashes.json')
    immutable={'format':'mef_off_on_handoff_v1','model':args.model,'dataset':args.dataset,'variant':1,'seed':args.seed,'smoke':args.smoke,'config':cfg,'code_hashes':code_hashes,'data_manifest_sha256':digest(reader.root/'manifest.enc'),'writer_profile':PROFILES[args.dataset]}
    identity=hashlib.sha256(json.dumps(immutable,sort_keys=True).encode()).hexdigest()
    run=Path(args.resume).resolve() if args.resume else Path(args.output_dir)/args.dataset/args.model/f'seed_{args.seed}'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    if args.resume:
        old=json.loads((run/'run_manifest.json').read_text())
        if old['chain_identity']!=identity:raise ValueError('Resume model/seed/data/code/config identity mismatch; restore original invocation')
    else:
        run.mkdir(parents=True,exist_ok=False)
        write(run/'run_manifest.json',{'chain_identity':identity,**immutable,'created_at_utc':datetime.now(timezone.utc).isoformat(),'device':str(device),'torch_version':torch.__version__,'data_privacy':'pseudonymous numeric data; authorized local users can extract it','source_time_independent':False,'target_time_causal':True,'shared_off_on_checkpoint':True})
    with (run/'requests.jsonl').open('a') as handle:
        handle.write(json.dumps({'mode':args.mode,'utc':datetime.now(timezone.utc).isoformat()})+'\n')
    print(f'run directory: {run.resolve()}',flush=True)
    def state(stage,status='running',**extra):write(run/'status.json',{'stage':stage,'status':status,'chain_identity':identity,**extra})
    try:
        state('main_predictor')
        panel=Panel(reader,'1_str')
        model,scaler=fit_predictor(panel,args.model,args.seed,*binding['panel_range'],cfg,run/'main',device,args.smoke)
        del panel,model
        records=[];calibration=None
        if args.mode!='off':
            state('historical_chain_writer_memory_cal1')
            records,calibration=build_history(reader,args,cfg,run,device,identity,binding['panel_range'][1])
        model,scaler=load_predictor(run/'main/model.pt',device)
        state('paired_evaluation')
        paired_evaluate(reader,args,binding,run,device,model,scaler,records,calibration,identity)
        artifacts={str(f.relative_to(run)):digest(f) for f in run.rglob('*') if f.is_file() and f.name in ('model.pt','memory.json','calibration.json','predictions.parquet','metrics.json')}
        write(run/'artifact_hashes.json',artifacts)
        state('completed','completed',memory_records=len(records),calibration_ready=calibration is not None)
    except BaseException as exc:
        state('failed','failed',error=f'{type(exc).__name__}: {exc}')
        raise
    return run
