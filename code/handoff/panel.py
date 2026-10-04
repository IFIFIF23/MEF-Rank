"""Encrypted financial views and checkpoint-local shock/FEC scalers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from benchmark_suite.data import load_panel
from mef_unified.training import TrainingPanel
from mef_rank.shock.builder import fit_shock_scaler
from mef_rank.fec.builder import _expand_msjd, _fit_columns


class Panel(TrainingPanel):
    def __init__(self, reader, view, years=None):
        self.reader = reader
        self.view = view
        years = years or reader.years(view)
        self.panel = load_panel(name={'gem':'gem1','main':'main1','nasdaq100':'us1_nasdaq100','sp500':'us1_sp500','crypto':'crypto1'}[reader.family], root=Path(view), years=years, reader=reader, view=view)
        frames = [reader.read_frame(view, y, 'market') for y in years]
        market = pd.concat(frames).set_index('feature_date')
        market.index = market.index.astype(str)
        if market.index.duplicated().any(): raise ValueError('duplicate market dates')
        market = market.reindex(self.panel.dates)
        names = [n for n in market if n.startswith('shock_causal__')]
        masks = [n.replace('shock_causal__','shock_causal_available__') for n in names]
        if len(names) != 5: raise ValueError('shock schema')
        self.shock = market[names].to_numpy(np.float64)
        self.shock_mask = market[masks].fillna(False).to_numpy(bool)
        self.shock[~self.shock_mask] = 0
        self.fec_names = [n for prefix in ('fec_common_raw__', 'fec_H5_raw__') for n in market if n.startswith(prefix)]
        if len(self.fec_names) != 51: raise ValueError('FEC schema')
        self.fec = market[self.fec_names].to_numpy(np.float64)
        mask_names = [n.replace('_raw__', '_raw_available__') for n in self.fec_names]
        self.fec_mask = market[mask_names].fillna(False).to_numpy(bool)
        self.fec[~self.fec_mask] = 0
        if not np.isfinite(self.shock).all() or not np.isfinite(self.fec).all(): raise ValueError('nonfinite observed context')
        self._joint_cache = {}

    def scaler(self, end, start=None):
        start = start or self.panel.dates[0]
        shock = fit_shock_scaler(tuple(self.panel.dates), self.shock, self.shock_mask, fit_date_max=end, fit_date_min=start)
        dates = np.asarray(self.panel.dates)
        fit = (dates >= start) & (dates <= end)
        center, scale, supported = _fit_columns(self.fec, self.fec_mask, fit)
        value = {'center':shock.center.tolist(), 'scale':shock.scale.tolist(), 'supported':shock.supported.tolist(), 'fit_date_min':start, 'fit_date_max':end, 'fec_center':center.tolist(), 'fec_scale':scale.tolist(), 'fec_supported':supported.tolist(), 'fec_columns':self.fec_names}
        value['sha256'] = hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
        return value

    def joint_state(self, index, scaler):
        if scaler['fec_columns'] != self.fec_names: raise ValueError('FEC column identity changed')
        key = scaler['sha256']
        if key not in self._joint_cache:
            mask = self.fec_mask & np.asarray(scaler['fec_supported'])
            level = (self.fec-np.asarray(scaler['fec_center']))/np.asarray(scaler['fec_scale'])
            self._joint_cache = {key: _expand_msjd(np.where(mask,level,0),mask)}
        values, masks = self._joint_cache[key]
        return values[index], masks[index]


    def proxies(self, index, codes):
        date = self.panel.dates[index]
        year = int(date[:4])
        if getattr(self, '_proxy_year', None) != year:
            columns=['feature_date','ts_code','context__log_circ_mv_lag1','context__log_circ_mv_lag1__available','context__amount']
            frame=self.reader.read_frame(self.view,year,'stock',columns)
            frame.loc[~frame['context__log_circ_mv_lag1__available'].fillna(False),'context__log_circ_mv_lag1']=np.nan
            frame['feature_date']=frame['feature_date'].astype(str)
            self._proxy_frame=frame.set_index(['feature_date','ts_code'])
            self._proxy_year=year
        index=pd.MultiIndex.from_tuples([(date,c) for c in codes],names=['feature_date','ts_code'])
        frame=self._proxy_frame.reindex(index)
        return frame['context__log_circ_mv_lag1'].to_numpy(float),frame['context__amount'].to_numpy(float)
