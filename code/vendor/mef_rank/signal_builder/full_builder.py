"""Real-data G1--G7 materialization for MEF-Rank v3.3.

The builder is deliberately independent of labels and execution outcomes.  It
accepts an explicit exchange-date grid and stock set, computes the frozen raw
coordinates, and only then applies the per-date average-tie rank transform.
Invalid coordinates retain the numeric placeholder zero and are accompanied by
an explicit boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .g7_builder import build_g7_77
from .g7_chen import cross_section_average_tie_rank


ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]
EPSILON = 1.0e-12
GROUP_ORDER = (
    "G1_return_trend",
    "G2_volume_liquidity",
    "G3_volatility_downside",
    "G4_valuation_size",
    "G5_style_reversal",
    "G6_market_relative",
    "G7_ohlc_ope",
)
GROUP_DIMENSIONS = (8, 8, 8, 8, 8, 8, 77)


@dataclass(frozen=True)
class SignalPanel:
    """A normalized signal panel on one explicit ``[T,N]`` grid."""

    dates: tuple[str, ...]
    codes: tuple[str, ...]
    values: Mapping[str, npt.NDArray[np.float32]]
    feature_masks: Mapping[str, ArrayB]
    raw_values: Mapping[str, ArrayF]
    raw_masks: Mapping[str, ArrayB]
    stock_present: ArrayB
    signal_universe: ArrayB
    transition_return: ArrayF
    transition_mask: ArrayB
    market_return: ArrayF
    market_return_mask: ArrayB
    context: Mapping[str, npt.NDArray]
    metadata: Mapping[str, object]

    def validate(self) -> None:
        time_count, stock_count = len(self.dates), len(self.codes)
        if not self.dates or not self.codes:
            raise ValueError("signal panel cannot be empty")
        if tuple(self.values) != GROUP_ORDER or tuple(self.feature_masks) != GROUP_ORDER:
            raise ValueError("signal groups must use the frozen group order")
        if tuple(self.raw_values) != GROUP_ORDER or tuple(self.raw_masks) != GROUP_ORDER:
            raise ValueError("raw signal groups must use the frozen group order")
        for name, dimension in zip(GROUP_ORDER, GROUP_DIMENSIONS):
            shape = (time_count, stock_count, dimension)
            if self.values[name].shape != shape or self.raw_values[name].shape != shape:
                raise ValueError(f"{name} has an invalid value shape")
            if self.feature_masks[name].shape != shape or self.raw_masks[name].shape != shape:
                raise ValueError(f"{name} has an invalid mask shape")
            if self.feature_masks[name].dtype != np.bool_ or self.raw_masks[name].dtype != np.bool_:
                raise TypeError(f"{name} masks must be boolean")
            if not np.isfinite(self.values[name]).all():
                raise ValueError(f"{name} contains a non-finite normalized placeholder")
            if not np.isfinite(self.raw_values[name][self.raw_masks[name]]).all():
                raise ValueError(f"{name} contains a non-finite available raw value")
            if np.any(self.values[name][~self.feature_masks[name]] != 0.0):
                raise ValueError(f"{name} invalid normalized values must be zero")
        for value, label in (
            (self.stock_present, "stock_present"),
            (self.signal_universe, "signal_universe"),
            (self.transition_mask, "transition_mask"),
        ):
            if value.shape != (time_count, stock_count) or value.dtype != np.bool_:
                raise ValueError(f"{label} must have shape [T,N] and bool dtype")
        if self.market_return.shape != (time_count,) or self.market_return_mask.shape != (time_count,):
            raise ValueError("market return arrays must have shape [T]")
        if np.any(self.signal_universe & ~self.stock_present):
            raise ValueError("signal universe must be a subset of stock-present rows")


def _date_text(frame: pd.DataFrame, column: str = "trade_date") -> pd.DataFrame:
    result = frame.copy()
    result[column] = (
        result[column]
        .astype("string")
        .str.replace("-", "", regex=False)
        .str[:8]
    )
    return result


def _grid_frame(
    frame: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
    fields: Sequence[str],
) -> pd.DataFrame:
    required = {"trade_date", "ts_code", *fields}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source lacks required fields: {missing}")
    source = _date_text(frame[list(required)])
    source["ts_code"] = source["ts_code"].astype(str)
    source = source[
        source["trade_date"].isin(dates) & source["ts_code"].isin(codes)
    ]
    if source.duplicated(["trade_date", "ts_code"]).any():
        raise ValueError("source has duplicate date/stock keys")
    grid = pd.MultiIndex.from_product(
        [list(dates), list(codes)], names=["trade_date", "ts_code"]
    )
    return source.set_index(["trade_date", "ts_code"]).reindex(grid)


def _matrix(frame: pd.DataFrame, field: str, shape: tuple[int, int]) -> ArrayF:
    return pd.to_numeric(frame[field], errors="coerce").to_numpy(np.float64).reshape(shape)


def _rolling_sum(values: ArrayF, valid: ArrayB, window: int) -> tuple[ArrayF, ArrayB]:
    if values.shape != valid.shape or values.ndim != 2:
        raise ValueError("rolling inputs must be aligned [T,N] arrays")
    time_count = values.shape[0]
    result = np.zeros_like(values, dtype=np.float64)
    mask = np.zeros_like(valid, dtype=bool)
    if window > time_count:
        return result, mask
    safe = np.where(valid, values, 0.0)
    sums = np.concatenate(
        [np.zeros((1, values.shape[1])), np.cumsum(safe, axis=0)], axis=0
    )
    counts = np.concatenate(
        [np.zeros((1, values.shape[1]), dtype=np.int64), np.cumsum(valid, axis=0)],
        axis=0,
    )
    result[window - 1 :] = sums[window:] - sums[:-window]
    count = counts[window:] - counts[:-window]
    mask[window - 1 :] = count == window
    result[~mask] = 0.0
    return result, mask


def _rolling_mean(values: ArrayF, valid: ArrayB, window: int) -> tuple[ArrayF, ArrayB]:
    result, mask = _rolling_sum(values, valid, window)
    result[mask] /= float(window)
    return result, mask


def _windows(values: ArrayF, window: int) -> npt.NDArray[np.float64]:
    return np.lib.stride_tricks.sliding_window_view(values, window, axis=0)


def _complete_windows(valid: ArrayB, window: int) -> ArrayB:
    output = np.zeros_like(valid, dtype=bool)
    if window <= valid.shape[0]:
        output[window - 1 :] = _windows(valid.astype(np.int8), window).sum(axis=-1) == window
    return output


def _rolling_median(values: ArrayF, valid: ArrayB, window: int) -> tuple[ArrayF, ArrayB]:
    output = np.zeros_like(values)
    mask = _complete_windows(valid, window)
    if window <= values.shape[0]:
        medians = np.median(_windows(np.where(valid, values, 0.0), window), axis=-1)
        output[window - 1 :][mask[window - 1 :]] = medians[mask[window - 1 :]]
    return output, mask


def _rolling_rms(values: ArrayF, valid: ArrayB, window: int) -> tuple[ArrayF, ArrayB]:
    mean_square, mask = _rolling_mean(np.square(values), valid, window)
    mean_square[mask] = np.sqrt(np.maximum(mean_square[mask], 0.0))
    return mean_square, mask


def _rolling_drawdown(values: ArrayF, valid: ArrayB, window: int) -> tuple[ArrayF, ArrayB]:
    output = np.zeros_like(values)
    mask = _complete_windows(valid, window)
    if window <= values.shape[0]:
        rolling = _windows(np.where(valid, values, 0.0), window)
        cumulative = np.cumsum(rolling, axis=-1)
        path = np.concatenate([np.zeros((*cumulative.shape[:-1], 1)), cumulative], axis=-1)
        log_drawdown = np.min(path - np.maximum.accumulate(path, axis=-1), axis=-1)
        result = 1.0 - np.exp(log_drawdown)
        output[window - 1 :][mask[window - 1 :]] = result[mask[window - 1 :]]
    return output, mask


def _rolling_quantile(
    values: ArrayF, valid: ArrayB, window: int, probability: float
) -> tuple[ArrayF, ArrayB]:
    output = np.zeros_like(values)
    mask = _complete_windows(valid, window)
    if window <= values.shape[0]:
        quantile = np.quantile(
            _windows(np.where(valid, values, 0.0), window),
            probability,
            axis=-1,
            method="linear",
        )
        output[window - 1 :][mask[window - 1 :]] = quantile[mask[window - 1 :]]
    return output, mask


def _rolling_ols_cumulative_slope(
    values: ArrayF, valid: ArrayB, window: int = 20
) -> tuple[ArrayF, ArrayB]:
    output = np.zeros_like(values)
    mask = _complete_windows(valid, window)
    if window <= values.shape[0]:
        cumulative = np.cumsum(_windows(np.where(valid, values, 0.0), window), axis=-1)
        x = np.arange(window, dtype=np.float64)
        centered = x - x.mean()
        slope = np.sum(cumulative * centered, axis=-1) / np.sum(np.square(centered))
        output[window - 1 :][mask[window - 1 :]] = slope[mask[window - 1 :]]
    return output, mask


def _rolling_covariance_coordinates(
    stock: ArrayF,
    stock_valid: ArrayB,
    market: ArrayF,
    market_valid: ArrayB,
    window: int,
) -> tuple[ArrayF, ArrayB, ArrayF, ArrayB]:
    common = stock_valid & market_valid[:, None]
    market_matrix = np.broadcast_to(market[:, None], stock.shape)
    stock_mean, complete = _rolling_mean(stock, common, window)
    market_mean, _ = _rolling_mean(market_matrix, common, window)
    product_mean, _ = _rolling_mean(stock * market_matrix, common, window)
    stock_square, _ = _rolling_mean(np.square(stock), common, window)
    market_square, _ = _rolling_mean(np.square(market_matrix), common, window)
    covariance = product_mean - stock_mean * market_mean
    stock_variance = np.maximum(stock_square - np.square(stock_mean), 0.0)
    market_variance = np.maximum(market_square - np.square(market_mean), 0.0)
    beta_mask = complete & (market_variance > EPSILON)
    corr_mask = beta_mask & (stock_variance > EPSILON)
    beta = np.zeros_like(stock)
    corr = np.zeros_like(stock)
    beta[beta_mask] = covariance[beta_mask] / market_variance[beta_mask]
    denominator = np.sqrt(stock_variance * market_variance)
    corr[corr_mask] = covariance[corr_mask] / denominator[corr_mask]
    corr[corr_mask] = np.clip(corr[corr_mask], -1.0, 1.0)
    return beta, beta_mask, corr, corr_mask


def _rank_panel(raw: ArrayF, mask: ArrayB, universe: ArrayB) -> tuple[npt.NDArray[np.float32], ArrayB]:
    ranked = np.zeros_like(raw, dtype=np.float32)
    ranked_mask = np.zeros_like(mask, dtype=bool)
    for time_index in range(raw.shape[0]):
        values, available = cross_section_average_tie_rank(
            raw[time_index], mask[time_index] & universe[time_index, :, None]
        )
        ranked[time_index] = values.astype(np.float32)
        ranked_mask[time_index] = available
    return ranked, ranked_mask


def _stack(entries: Sequence[tuple[ArrayF, ArrayB]]) -> tuple[ArrayF, ArrayB]:
    return (
        np.stack([entry[0] for entry in entries], axis=-1),
        np.stack([entry[1] for entry in entries], axis=-1),
    )


def build_signal_panel(
    daily_facts: pd.DataFrame,
    daily_basic: pd.DataFrame,
    adj_factor: pd.DataFrame,
    factor_transition_evidence: pd.DataFrame,
    index_daily: pd.DataFrame,
    dates: Sequence[str],
    codes: Sequence[str],
) -> SignalPanel:
    """Build all seven real signal groups on an explicit engineering grid."""

    ordered_dates = tuple(str(value) for value in dates)
    ordered_codes = tuple(str(value) for value in codes)
    if tuple(sorted(set(ordered_dates))) != ordered_dates:
        raise ValueError("dates must be unique and increasing")
    if len(set(ordered_codes)) != len(ordered_codes):
        raise ValueError("codes must be unique")
    shape = (len(ordered_dates), len(ordered_codes))
    fact_fields = (
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "listing_exchange_open_day_number",
        "historical_st",
        "historical_st_status_known",
        "suspend_event_count",
        "suspend_status_known",
    )
    fact = _grid_frame(daily_facts, ordered_dates, ordered_codes, fact_fields)
    stock_present = fact["close"].notna().to_numpy().reshape(shape)
    open_ = _matrix(fact, "open", shape)
    high = _matrix(fact, "high", shape)
    low = _matrix(fact, "low", shape)
    close = _matrix(fact, "close", shape)
    volume = _matrix(fact, "vol", shape)
    amount = _matrix(fact, "amount", shape)
    listing_day = _matrix(fact, "listing_exchange_open_day_number", shape)
    historical_st = (
        fact["historical_st"].astype("boolean").fillna(True).to_numpy(dtype=bool).reshape(shape)
    )
    st_known = (
        fact["historical_st_status_known"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
        .reshape(shape)
    )
    signal_universe = (
        stock_present
        & st_known
        & ~historical_st
        & (listing_day >= 20)
        & np.isfinite(close)
        & (close > 0)
    )

    factor = _grid_frame(adj_factor, ordered_dates, ordered_codes, ("adj_factor",))
    adj = _matrix(factor, "adj_factor", shape)
    previous_close = np.roll(close, 1, axis=0)
    previous_factor = np.roll(adj, 1, axis=0)
    previous_close[0] = np.nan
    previous_factor[0] = np.nan
    factor_pair = (
        np.isfinite(adj)
        & (adj > 0)
        & np.isfinite(previous_factor)
        & (previous_factor > 0)
    )
    raw_pair = (
        np.isfinite(close)
        & (close > 0)
        & np.isfinite(previous_close)
        & (previous_close > 0)
        & stock_present
        & np.roll(stock_present, 1, axis=0)
    )
    raw_pair[0] = False
    ratio = np.ones(shape, dtype=np.float64)
    ratio[factor_pair] = adj[factor_pair] / previous_factor[factor_pair]
    nonunit = factor_pair & ~np.isclose(ratio, 1.0, atol=EPSILON, rtol=EPSILON)
    evidence = _grid_frame(
        factor_transition_evidence,
        ordered_dates,
        ordered_codes,
        ("strict_transition_verified",),
    )
    strict_verified = (
        evidence["strict_transition_verified"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
        .reshape(shape)
    )
    transition_mask = raw_pair & factor_pair & (~nonunit | strict_verified)
    transition_return = np.zeros(shape, dtype=np.float64)
    transition_return[transition_mask] = np.log(
        (close[transition_mask] * adj[transition_mask])
        / (previous_close[transition_mask] * previous_factor[transition_mask])
    )

    market_source = _date_text(index_daily)
    market_source = market_source[market_source["index_code"].astype(str).eq("399006.SZ")]
    if market_source.duplicated("trade_date").any():
        raise ValueError("399006.SZ index source has duplicate dates")
    market_close = (
        pd.to_numeric(market_source.set_index("trade_date")["close"], errors="coerce")
        .reindex(ordered_dates)
        .to_numpy(np.float64)
    )
    market_previous = np.roll(market_close, 1)
    market_previous[0] = np.nan
    market_return_mask = (
        np.isfinite(market_close)
        & (market_close > 0)
        & np.isfinite(market_previous)
        & (market_previous > 0)
    )
    market_return = np.zeros(len(ordered_dates), dtype=np.float64)
    market_return[market_return_mask] = np.log(
        market_close[market_return_mask] / market_previous[market_return_mask]
    )

    ret: dict[int, tuple[ArrayF, ArrayB]] = {
        window: _rolling_sum(transition_return, transition_mask, window)
        for window in (1, 5, 20, 60)
    }
    positive20, positive20_mask = _rolling_mean(
        (transition_return > 0).astype(np.float64), transition_mask, 20
    )
    slope20, slope20_mask = _rolling_ols_cumulative_slope(
        transition_return, transition_mask, 20
    )
    g1 = _stack(
        [
            ret[1],
            ret[5],
            ret[20],
            ret[60],
            (ret[5][0] / 5.0 - ret[20][0] / 20.0, ret[5][1] & ret[20][1]),
            (ret[20][0] / 20.0 - ret[60][0] / 60.0, ret[20][1] & ret[60][1]),
            (positive20, positive20_mask),
            (slope20, slope20_mask),
        ]
    )

    log_volume = np.zeros(shape)
    volume_mask = np.isfinite(volume) & (volume >= 0)
    log_volume[volume_mask] = np.log1p(volume[volume_mask])
    log_amount = np.zeros(shape)
    amount_mask = np.isfinite(amount) & (amount >= 0)
    log_amount[amount_mask] = np.log1p(amount[amount_mask])
    mean_log_volume20 = _rolling_mean(log_volume, volume_mask, 20)
    mean_log_amount20 = _rolling_mean(log_amount, amount_mask, 20)
    log_volume_rel = log_volume - mean_log_volume20[0]
    log_amount_rel = log_amount - mean_log_amount20[0]

    basic_fields = (
        "turnover_rate_f",
        "volume_ratio",
        "total_mv",
        "circ_mv",
        "free_share",
        "total_share",
        "pb",
        "pe_ttm",
        "ps_ttm",
        "dv_ttm",
    )
    basic = _grid_frame(daily_basic, ordered_dates, ordered_codes, basic_fields)
    lagged: dict[str, ArrayF] = {}
    for field in basic_fields:
        current = _matrix(basic, field, shape)
        lagged[field] = np.roll(current, 1, axis=0)
        lagged[field][0] = np.nan
    turnover_mask = np.isfinite(lagged["turnover_rate_f"])
    volume_ratio_mask = np.isfinite(lagged["volume_ratio"])
    turnover_median = _rolling_median(
        np.where(turnover_mask, lagged["turnover_rate_f"], 0.0), turnover_mask, 20
    )
    positive_amount = np.isfinite(amount) & (amount > 0)
    amihud = _rolling_mean(
        np.divide(
            np.abs(transition_return),
            amount,
            out=np.zeros_like(amount),
            where=positive_amount,
        ),
        transition_mask & positive_amount,
        20,
    )
    g2 = _stack(
        [
            (log_volume, volume_mask),
            (log_amount, amount_mask),
            (log_volume_rel, volume_mask & mean_log_volume20[1]),
            (log_amount_rel, amount_mask & mean_log_amount20[1]),
            (np.where(turnover_mask, lagged["turnover_rate_f"], 0.0), turnover_mask),
            (np.where(volume_ratio_mask, lagged["volume_ratio"], 0.0), volume_ratio_mask),
            turnover_median,
            amihud,
        ]
    )

    realized5 = _rolling_rms(transition_return, transition_mask, 5)
    realized20 = _rolling_rms(transition_return, transition_mask, 20)
    realized60 = _rolling_rms(transition_return, transition_mask, 60)
    downside = np.minimum(transition_return, 0.0)
    downside20 = _rolling_rms(downside, transition_mask, 20)
    downside60 = _rolling_rms(downside, transition_mask, 60)
    drawdown20 = _rolling_drawdown(transition_return, transition_mask, 20)
    drawdown60 = _rolling_drawdown(transition_return, transition_mask, 60)
    tail20, tail20_mask = _rolling_quantile(transition_return, transition_mask, 20, 0.05)
    tail20 = -tail20
    g3 = _stack(
        [
            realized5,
            realized20,
            realized60,
            downside20,
            downside60,
            drawdown20,
            drawdown60,
            (tail20, tail20_mask),
        ]
    )

    total_mv = lagged["total_mv"]
    circ_mv = lagged["circ_mv"]
    free_share = lagged["free_share"]
    total_share = lagged["total_share"]
    positive_total = np.isfinite(total_mv) & (total_mv > 0)
    positive_circ = np.isfinite(circ_mv) & (circ_mv > 0)
    positive_free = np.isfinite(free_share) & (free_share > 0)
    positive_share = np.isfinite(total_share) & (total_share > 0)
    reciprocal_entries: list[tuple[ArrayF, ArrayB]] = []
    for field in ("pb", "pe_ttm", "ps_ttm"):
        source = lagged[field]
        valid = np.isfinite(source) & (source != 0)
        reciprocal_entries.append((np.divide(1.0, source, out=np.zeros_like(source), where=valid), valid))
    dividend = lagged["dv_ttm"]
    dividend_mask = np.isfinite(dividend)
    free_float = np.divide(
        free_share, total_share, out=np.zeros_like(free_share), where=positive_share
    )
    free_float_mask = positive_free & positive_share & (free_float >= 0) & (free_float <= 1)
    g4 = _stack(
        [
            (np.where(positive_total, np.log(np.where(positive_total, total_mv, 1.0)), 0.0), positive_total),
            (np.where(positive_circ, np.log(np.where(positive_circ, circ_mv, 1.0)), 0.0), positive_circ),
            (np.where(positive_free, np.log(np.where(positive_free, free_share, 1.0)), 0.0), positive_free),
            *reciprocal_entries,
            (np.where(dividend_mask, dividend / 100.0, 0.0), dividend_mask),
            (free_float, free_float_mask),
        ]
    )

    g5 = _stack(
        [
            (-ret[1][0], ret[1][1]),
            (-ret[5][0], ret[5][1]),
            (ret[20][0] - ret[5][0], ret[20][1] & ret[5][1]),
            (ret[60][0] - ret[20][0], ret[60][1] & ret[20][1]),
            (-g4[0][..., 1], g4[1][..., 1]),
            (g4[0][..., 3], g4[1][..., 3]),
            (-realized20[0], realized20[1]),
            turnover_median,
        ]
    )

    market_ret: dict[int, tuple[ArrayF, ArrayB]] = {}
    for window in (1, 5, 20, 60):
        value, valid = _rolling_sum(
            market_return[:, None], market_return_mask[:, None], window
        )
        market_ret[window] = (value[:, 0], valid[:, 0])
    beta20, beta20_mask, corr20, corr20_mask = _rolling_covariance_coordinates(
        transition_return, transition_mask, market_return, market_return_mask, 20
    )
    beta60, beta60_mask, corr60, corr60_mask = _rolling_covariance_coordinates(
        transition_return, transition_mask, market_return, market_return_mask, 60
    )
    excess_entries: list[tuple[ArrayF, ArrayB]] = []
    for window in (1, 5, 20, 60):
        valid = ret[window][1] & market_ret[window][1][:, None]
        excess_entries.append((ret[window][0] - market_ret[window][0][:, None], valid))
    g6 = _stack(
        [
            *excess_entries,
            (beta20, beta20_mask),
            (beta60, beta60_mask),
            (corr20, corr20_mask),
            (corr60, corr60_mask),
        ]
    )

    suspend_known = (
        fact["suspend_status_known"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
        .reshape(shape)
    )
    suspend_count = _matrix(fact, "suspend_event_count", shape)
    no_suspend = suspend_known & np.isfinite(suspend_count) & (suspend_count == 0)
    full_market_dates = set(_date_text(daily_facts)["trade_date"].dropna().astype(str))
    market_day_available = np.asarray(
        [date in full_market_dates for date in ordered_dates], dtype=bool
    )
    g7_result = build_g7_77(
        open_, high, low, close, volume, signal_universe & no_suspend, market_day_available
    )
    g7_raw = np.asarray(g7_result.values, dtype=np.float64)
    g7_mask = np.asarray(g7_result.feature_mask, dtype=bool)

    raw_groups: dict[str, ArrayF] = {
        GROUP_ORDER[0]: g1[0],
        GROUP_ORDER[1]: g2[0],
        GROUP_ORDER[2]: g3[0],
        GROUP_ORDER[3]: g4[0],
        GROUP_ORDER[4]: g5[0],
        GROUP_ORDER[5]: g6[0],
        # G7's public builder returns its already-normalized tensor.  Retaining
        # the same array here is intentional; raw D3 tensors are not a Query input.
        GROUP_ORDER[6]: g7_raw,
    }
    raw_masks: dict[str, ArrayB] = {
        GROUP_ORDER[0]: g1[1],
        GROUP_ORDER[1]: g2[1],
        GROUP_ORDER[2]: g3[1],
        GROUP_ORDER[3]: g4[1],
        GROUP_ORDER[4]: g5[1],
        GROUP_ORDER[5]: g6[1],
        GROUP_ORDER[6]: g7_mask,
    }
    normalized: dict[str, npt.NDArray[np.float32]] = {}
    normalized_masks: dict[str, ArrayB] = {}
    for name in GROUP_ORDER[:6]:
        normalized[name], normalized_masks[name] = _rank_panel(
            raw_groups[name], raw_masks[name], signal_universe
        )
    normalized[GROUP_ORDER[6]] = np.array(g7_result.values, dtype=np.float32, copy=True)
    normalized_masks[GROUP_ORDER[6]] = np.array(g7_mask, dtype=bool, copy=True)
    for name in GROUP_ORDER:
        raw_groups[name] = np.where(raw_masks[name], raw_groups[name], 0.0)
        normalized[name][~normalized_masks[name]] = 0.0

    context: dict[str, npt.NDArray] = {
        "amount": amount,
        "amount_mask": amount_mask,
        "volume": volume,
        "turnover_rate_f_lag1": g2[0][..., 4],
        "turnover_rate_f_lag1_mask": g2[1][..., 4],
        "log_circ_mv_lag1": g4[0][..., 1],
        "log_circ_mv_lag1_mask": g4[1][..., 1],
        "book_to_price_lag1": g4[0][..., 3],
        "book_to_price_lag1_mask": g4[1][..., 3],
        "ret_5": ret[5][0],
        "ret_5_mask": ret[5][1],
        "ret_20": ret[20][0],
        "ret_20_mask": ret[20][1],
        "realized_vol_20": realized20[0],
        "realized_vol_20_mask": realized20[1],
        "total_return_log_level": np.cumsum(
            np.where(transition_mask, transition_return, 0.0), axis=0
        ),
        "no_suspend": no_suspend,
        "market_day_available": market_day_available,
    }
    panel = SignalPanel(
        dates=ordered_dates,
        codes=ordered_codes,
        values=normalized,
        feature_masks=normalized_masks,
        raw_values=raw_groups,
        raw_masks=raw_masks,
        stock_present=stock_present,
        signal_universe=signal_universe,
        transition_return=transition_return,
        transition_mask=transition_mask,
        market_return=market_return,
        market_return_mask=market_return_mask,
        context=context,
        metadata={
            "transition_formula": "log((raw_close_t*factor_t)/(raw_close_t-1*factor_t-1))",
            "normalization": "per_date_average_tie_2r_over_n_minus_1",
            "date_count": len(ordered_dates),
            "stock_count": len(ordered_codes),
            "cross_section_scope": "explicit_input_codes",
        },
    )
    panel.validate()
    return panel
