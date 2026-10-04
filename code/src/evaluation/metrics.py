"""Evaluation harness: ranking + portfolio metrics for daily stock prediction.

All metrics are computed per trade_date over the valid universe and then
aggregated, matching the protocol used in A-conf stock-prediction papers
(MASTER, HIST, FactorVAE, etc.):

  ranking : IC, RankIC, ICIR, RankICIR, %positive days, Newey-West t-stat
  portfolio: Top-K long-only excess, long-short spread, ann. return/vol/Sharpe,
             max drawdown, turnover, after-cost return.

`y_eval` for ranking/IC is the raw next-day return `label_cc_1`; portfolio
excess returns are measured against the daily cross-sectional mean (market
neutral), i.e. `label_excess_cs_1`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 242


# --------------------------------------------------------------------------- #
# daily cross-sectional helpers
# --------------------------------------------------------------------------- #
def _spearman(a, b):
    if len(a) < 5:
        return np.nan
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return _pearson(ra, rb)


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    da = np.sqrt((a * a).sum()); db = np.sqrt((b * b).sum())
    if da < 1e-12 or db < 1e-12:
        return np.nan
    return float((a * b).sum() / (da * db))


def daily_ic(df: pd.DataFrame, score="score", y="y"):
    """Return per-date IC and RankIC series."""
    ics, rics, dates = [], [], []
    for d, g in df.groupby("trade_date"):
        s = g[score].to_numpy(); yy = g[y].to_numpy()
        m = np.isfinite(s) & np.isfinite(yy)
        if m.sum() < 5:
            continue
        ics.append(_pearson(s[m], yy[m]))
        rics.append(_spearman(s[m], yy[m]))
        dates.append(d)
    return pd.DataFrame({"trade_date": dates, "ic": ics, "rankic": rics})


def _nw_tstat(x, lags=5):
    """Newey-West t-stat of the mean of a daily series."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan
    mu = x.mean(); e = x - mu
    gamma0 = (e @ e) / n
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        cov = (e[k:] @ e[:-k]) / n
        var += 2 * w * cov
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else np.nan


def ranking_summary(ic_df: pd.DataFrame) -> dict:
    ic = ic_df["ic"].to_numpy(); ric = ic_df["rankic"].to_numpy()
    def stats(x, tag):
        x = x[np.isfinite(x)]
        return {
            f"{tag}_mean": float(np.mean(x)),
            f"{tag}_std": float(np.std(x)),
            f"{tag}_ir": float(np.mean(x) / (np.std(x) + 1e-12)),
            f"{tag}_pos": float(np.mean(x > 0)),
            f"{tag}_t": _nw_tstat(x),
        }
    out = {}
    out.update(stats(ic, "ic"))
    out.update(stats(ric, "rankic"))
    out["n_days"] = int(len(ic_df))
    return out


# --------------------------------------------------------------------------- #
# portfolio backtest
# --------------------------------------------------------------------------- #
def topk_backtest(df: pd.DataFrame, score="score", ret="ret_fwd",
                  excess="ret_excess", k=20, cost_bps=15.0):
    """Equal-weight Top-K long-only (excess) and Top-K minus Bottom-K long-short.

    df needs columns: trade_date, ts_code, score, ret(next-day cc return),
    excess(next-day cross-sectional demeaned return). Turnover and cost are
    computed from day-to-day holdings overlap.
    """
    long_ex, ls, turn = [], [], []
    prev_long = set()
    # single groupby (date-sorted) instead of an O(N) filter per date
    for d, g in df.groupby("trade_date", sort=True):
        g = g[np.isfinite(g[score]) & np.isfinite(g[ret])]
        if len(g) < 2 * k:
            continue
        g = g.sort_values(score, ascending=False)
        topk = g.head(k); botk = g.tail(k)
        long_ex.append(topk[excess].mean())
        ls.append(topk[ret].mean() - botk[ret].mean())
        # turnover of the long book
        cur = set(topk["ts_code"])
        if prev_long:
            turn.append(1.0 - len(cur & prev_long) / k)
        prev_long = cur
    long_ex = np.array(long_ex); ls = np.array(ls); turn = np.array(turn)
    cost = cost_bps / 1e4
    avg_turn = float(np.mean(turn)) if len(turn) else 0.0
    # cost charged on the long book (two-sided turnover ~ 2*name turnover)
    long_ex_net = long_ex - 2 * avg_turn * cost
    return {
        "k": k,
        "long_excess_ann": _ann_ret(long_ex),
        "long_excess_ann_net": _ann_ret(long_ex_net),
        "long_excess_sharpe": _sharpe(long_ex),
        "ls_ann": _ann_ret(ls),
        "ls_sharpe": _sharpe(ls),
        "ls_mdd": _mdd(ls),
        "long_excess_t": _nw_tstat(long_ex),
        "ls_t": _nw_tstat(ls),
        "avg_turnover": avg_turn,
        "n_days": int(len(long_ex)),
    }


def _ann_ret(daily):
    daily = daily[np.isfinite(daily)]
    if len(daily) == 0:
        return np.nan
    return float(np.mean(daily) * TRADING_DAYS)


def _sharpe(daily):
    daily = daily[np.isfinite(daily)]
    if len(daily) < 5 or np.std(daily) < 1e-12:
        return np.nan
    return float(np.mean(daily) / np.std(daily) * np.sqrt(TRADING_DAYS))


def _mdd(daily):
    daily = daily[np.isfinite(daily)]
    if len(daily) == 0:
        return np.nan
    eq = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(eq)
    return float(np.min(eq / peak - 1.0))
