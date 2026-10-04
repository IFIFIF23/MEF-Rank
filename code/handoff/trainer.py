"""Shared zero-action fitting for final and rolling historical predictors."""
import json
import hashlib
from pathlib import Path
import random
import time
import numpy as np
import torch
from mef_unified.adapters import UnifiedRanker
from mef_unified.training import seed_all, whole_cross_section, loss_value, evaluate, write


def eligible(panel, start, end):
    p = panel.panel
    begin = next((i for i,d in enumerate(p.dates) if d >= start), len(p.dates))
    return [i for i in range(begin+59,len(p.dates)) if p.dates[i] <= end and len(p.eligible_indices(i, require_label=True, maturity_cutoff=end)) >= 2]


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp); temp.replace(path)


def fit_predictor(panel, name, seed, start, end, cfg, destination, device, smoke=False):
    destination = Path(destination)
    final_path = destination/'model.pt'
    if final_path.exists() and (destination/'complete.json').exists():
        expected=json.loads((destination/'complete.json').read_text())['checkpoint_sha256']
        if hashlib.sha256(final_path.read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint changed after completion')
        return load_predictor(final_path, device)
    destination.mkdir(parents=True, exist_ok=True)
    days = eligible(panel,start,end)
    valid_count = 2 if smoke else int(cfg['training']['inner_validation_days'])
    if len(days) <= valid_count+1: raise ValueError('Insufficient mature dates for fit/validation')
    val = days[-valid_count:]
    fit_end = panel.panel.dates[val[0]-1]
    fit_days = [i for i in eligible(panel,start,fit_end) if i < val[0]][::cfg['training']['date_stride']]
    final_days = days[::cfg['training']['date_stride']]
    if smoke: fit_days=fit_days[-2:];final_days=final_days[-2:]
    if not fit_days or not final_days: raise ValueError('No fit dates after maturity purge')
    write(destination/'split.json', {'fit_start':start,'fit_end':fit_end,'final_end':end,'fit_dates':[panel.panel.dates[i] for i in fit_days], 'validation_dates':[panel.panel.dates[i] for i in val], 'final_dates':[panel.panel.dates[i] for i in final_days], 'engineering_smoke':smoke})
    def phase(which, selected, max_epochs, scaler, cutoff, validation):
        seed_all(seed)
        model=UnifiedRanker(name).to(device)
        o=cfg['optimizer']; opt=torch.optim.AdamW(model.parameters(),lr=o['learning_rate'],weight_decay=o['weight_decay'])
        best=-float('inf'); best_epoch=1; stale=0; first_epoch=1; history=[]
        resume=destination/(which+'_resume.pt')
        if resume.exists():
            saved=torch.load(resume,map_location='cpu',weights_only=False)
            model.load_state_dict(saved['state_dict']);opt.load_state_dict(saved['optimizer'])
            for state in opt.state.values():
                for key,val_tensor in state.items():
                    if torch.is_tensor(val_tensor): state[key]=val_tensor.to(device)
            first_epoch=saved['epoch']+1;best=saved['best'];best_epoch=saved['best_epoch'];stale=saved['stale'];history=saved['history']
            torch.set_rng_state(saved['torch_rng'])
            if device.type=='cuda' and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        for epoch in range(first_epoch,max_epochs+1):
            if validation and first_epoch>1 and epoch-1>=cfg['training']['min_epochs'] and stale>=cfg['training']['patience']:break
            model.train();losses=[];started=time.monotonic()
            for i in np.random.default_rng(seed+epoch).permutation(selected):
                batch=panel.batch(int(i),device,cutoff,scaler)
                opt.zero_grad(set_to_none=True)
                out=whole_cross_section(model,batch)
                loss=loss_value(out,batch,f'{panel.panel.name}|{which}|{panel.panel.dates[i]}',cfg)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),o['gradient_clip'],error_if_nonfinite=True);opt.step()
                losses.append(float(loss.detach()))
            row={'phase':which,'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.monotonic()-started}
            if validation:
                score,daily,pred=evaluate(model,panel,val,device,end,scaler,True)
                row['rankic']=score
                if score>best:
                    best=score;best_epoch=epoch;stale=0
                    pred.to_parquet(destination/'predictions.parquet',index=False)
                    write(destination/'metrics.json',{'scope':'inner_validation_only','rankic':score,'best_epoch':epoch,'daily':daily})
                    save_checkpoint(destination/'selection_best.pt',{'state_dict':model.state_dict(),'profile':model.profile,'scaler':scaler,'name':name,'epoch':epoch})
                else:stale+=1
            history.append(row);print(json.dumps(row),flush=True)
            save_checkpoint(resume,{'state_dict':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
            if validation and epoch>=cfg['training']['min_epochs'] and stale>=cfg['training']['patience']:break
        return model,best_epoch
    selection_scaler=panel.scaler(fit_end,start)
    selection_model,selected_epochs=phase('selection',fit_days,1 if smoke else cfg['training']['max_epochs'],selection_scaler,fit_end,True)
    del selection_model
    if device.type=='cuda':torch.cuda.empty_cache()
    scaler=panel.scaler(end,start)
    model,_=phase('final',final_days,selected_epochs,scaler,end,False)
    save_checkpoint(final_path,{'state_dict':model.state_dict(),'profile':model.profile,'scaler':scaler,'name':name,'seed':seed,'start':start,'end':end,'epochs':selected_epochs})
    write(destination/'complete.json',{'epochs':selected_epochs,'fit_start':start,'fit_end':end,'smoke':smoke,'checkpoint_sha256':hashlib.sha256(final_path.read_bytes()).hexdigest()})
    return model.eval(),scaler


def load_predictor(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    model=UnifiedRanker(saved['name']).to(device)
    if model.profile != saved['profile']:raise ValueError('checkpoint profile changed')
    model.load_state_dict(saved['state_dict'])
    return model.eval(),saved['scaler']
