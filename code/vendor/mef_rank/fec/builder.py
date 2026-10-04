"""Causal FEC C1--C8 and MSJD builders for MEF-Rank v3.3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..signal_builder.full_builder import (
    EPSILON,
    GROUP_ORDER,
    SignalPanel,
    _rolling_drawdown,
    _rolling_mean,
    _rolling_rms,
    _rolling_sum,
)


ArrayF = npt.NDArray[np.float64]
ArrayB = npt.NDArray[np.bool_]
COMMON_DIMENSION = 26
HORIZON_DIMENSION = 25


@dataclass(frozen=True)
class FECPanel:
    dates: tuple[str, ...]
    common_raw: ArrayF
    common_mask: ArrayB
    horizon_raw: Mapping[int, ArrayF]
    horizon_mask: Mapping[int, ArrayB]
    maturity_audit: pd.DataFrame
    drivers: Mapping[str, npt.NDArray]

    def validate(self) -> None:
        time_count = len(self.dates)
        if self.common_raw.shape != (time_count, COMMON_DIMENSION):
            raise ValueError("common FEC must have shape [T,26]")
        if self.common_mask.shape != self.common_raw.shape or self.common_mask.dtype != np.bool_:
            raise ValueError("common FEC mask mismatch")
        if set(self.horizon_raw) != {1, 5} or set(self.horizon_mask) != {1, 5}:
            raise ValueError("FEC must have independent H1 and H5 views")
        for horizon in (1, 5):
            if self.horizon_raw[horizon].shape != (time_count, HORIZON_DIMENSION):
                raise ValueError(f"H{horizon} FEC must have shape [T,25]")
            if self.horizon_mask[horizon].shape != self.horizon_raw[horizon].shape:
                raise ValueError(f"H{horizon} FEC mask mismatch")
            if self.horizon_mask[horizon].dtype != np.bool_:
                raise TypeError("FEC masks must be boolean")
            if not np.isfinite(self.horizon_raw[horizon]).all():
                raise ValueError("FEC placeholders must be finite")
        if not np.isfinite(self.common_raw).all():
            raise ValueError("common FEC placeholders must be finite")
        if np.any(self.common_raw[~self.common_mask] != 0.0):
            raise ValueError("invalid common FEC placeholders must be zero")


@dataclass(frozen=True)
class FECScaler:
    fit_date_max: str
    common_center: ArrayF
    common_scale: ArrayF
    common_mask: ArrayB
    horizon_center: Mapping[int, ArrayF]
    horizon_scale: Mapping[int, ArrayF]
    horizon_mask: Mapping[int, ArrayB]
    scaler_hash: str
    fit_date_min: str | None = None


@dataclass(frozen=True)
class MSJDPanel:
    dates: tuple[str, ...]
    values: Mapping[int, ArrayF]
    masks: Mapping[int, ArrayB]
    scaler_hash: str

    def validate(self) -> None:
        expected = (len(self.dates), (COMMON_DIMENSION + HORIZON_DIMENSION) * 6)
        for horizon in (1, 5):
            if self.values[horizon].shape != expected or self.masks[horizon].shape != expected:
                raise ValueError("MSJD must have shape [T,306] for each horizon")
            if self.masks[horizon].dtype != np.bool_:
                raise TypeError("MSJD mask must be boolean")
            if not np.isfinite(self.values[horizon]).all():
                raise ValueError("MSJD placeholders must be finite")


def _market_coordinates(panel: SignalPanel) -> tuple[ArrayF, ArrayB]:
    market = panel.market_return[:, None]
    market_mask = panel.market_return_mask[:, None]
    ret20 = _rolling_sum(market, market_mask, 20)
    drawdown60 = _rolling_drawdown(market, market_mask, 60)
    vol20 = _rolling_rms(market, market_mask, 20)
    downside20 = _rolling_rms(np.minimum(market, 0.0), market_mask, 20)
    values = np.concatenate([ret20[0], drawdown60[0], vol20[0], downside20[0]], axis=1)
    masks = np.concatenate([ret20[1], drawdown60[1], vol20[1], downside20[1]], axis=1)
    return values, masks


def _common_fec(panel: SignalPanel, minimum_stocks: int) -> tuple[ArrayF, ArrayB, dict[str, npt.NDArray]]:
    time_count, stock_count = panel.signal_universe.shape
    common = np.zeros((time_count, COMMON_DIMENSION), dtype=np.float64)
    mask = np.zeros_like(common, dtype=bool)
    c1, c1_mask = _market_coordinates(panel)
    common[:, :4], mask[:, :4] = c1, c1_mask

    amount = np.asarray(panel.context["amount"], dtype=np.float64)
    amount_mask = np.asarray(panel.context["amount_mask"], dtype=bool)
    total_amount = np.zeros(time_count)
    total_amount_mask = np.zeros(time_count, dtype=bool)
    turnover = np.asarray(panel.context["turnover_rate_f_lag1"], dtype=np.float64)
    turnover_mask = np.asarray(panel.context["turnover_rate_f_lag1_mask"], dtype=bool)
    leaders: list[frozenset[int] | None] = [None] * time_count
    transition = panel.transition_return
    transition_mask = panel.transition_mask
    level = np.asarray(panel.context["total_return_log_level"], dtype=np.float64)
    level_mean20, level_mean20_mask = _rolling_mean(level, transition_mask, 20)

    for time_index in range(time_count):
        universe = panel.signal_universe[time_index]
        amount_available = universe & amount_mask[time_index]
        if int(amount_available.sum()) >= minimum_stocks:
            total_amount[time_index] = amount[time_index, amount_available].sum()
            total_amount_mask[time_index] = np.isfinite(total_amount[time_index]) & (
                total_amount[time_index] >= 0
            )
            if total_amount_mask[time_index] and total_amount[time_index] > EPSILON:
                indices = np.flatnonzero(amount_available)
                top_count = max(1, int(np.ceil(0.10 * len(indices))))
                order = np.lexsort(
                    (
                        np.asarray(panel.codes, dtype=object)[indices],
                        -amount[time_index, indices],
                    )
                )
                top = indices[order[:top_count]]
                common[time_index, 6] = amount[time_index, top].sum() / total_amount[time_index]
                mask[time_index, 6] = True

        turnover_available = universe & turnover_mask[time_index]
        if int(turnover_available.sum()) >= minimum_stocks:
            common[time_index, 5] = np.median(turnover[time_index, turnover_available])
            mask[time_index, 5] = True

        amihud_available = (
            universe
            & transition_mask[time_index]
            & amount_mask[time_index]
            & (amount[time_index] > 0)
        )
        if int(amihud_available.sum()) >= minimum_stocks:
            common[time_index, 7] = np.median(
                np.abs(transition[time_index, amihud_available])
                / amount[time_index, amihud_available]
            )
            mask[time_index, 7] = True

        return_available = universe & transition_mask[time_index]
        if int(return_available.sum()) >= minimum_stocks:
            returns = transition[time_index, return_available]
            common[time_index, 8] = np.mean(returns > 0)
            common[time_index, 10] = np.std(returns, ddof=0)
            mask[time_index, 8] = mask[time_index, 10] = True
            median = np.median(returns)
            mad = np.median(np.abs(returns - median))
            if mad > EPSILON:
                common[time_index, 11] = np.mean(
                    np.abs(returns - median) > 2.5 * 1.4826 * mad
                )
                mask[time_index, 11] = True

        above_available = universe & level_mean20_mask[time_index]
        if int(above_available.sum()) >= minimum_stocks:
            common[time_index, 9] = np.mean(
                level[time_index, above_available]
                > level_mean20[time_index, above_available]
            )
            mask[time_index, 9] = True

        contribution_available = (
            universe
            & transition_mask[time_index]
            & amount_mask[time_index]
            & (amount[time_index] >= 0)
        )
        if int(contribution_available.sum()) >= max(minimum_stocks, 5):
            indices = np.flatnonzero(contribution_available)
            contribution = np.maximum(
                amount[time_index, indices] * transition[time_index, indices], 0.0
            )
            mass = contribution.sum()
            if mass > EPSILON:
                order = np.lexsort(
                    (np.asarray(panel.codes, dtype=object)[indices], -contribution)
                )
                leader = indices[order[:5]]
                leaders[time_index] = frozenset(int(value) for value in leader)
                weights = contribution / mass
                common[time_index, 12] = contribution[order[:5]].sum() / mass
                common[time_index, 14] = (
                    transition[time_index, leader].mean()
                    - np.median(transition[time_index, indices])
                )
                common[time_index, 15] = np.sum(np.square(weights))
                mask[time_index, [12, 14, 15]] = True
        if time_index >= 20 and leaders[time_index] is not None and leaders[time_index - 20] is not None:
            current, previous = leaders[time_index], leaders[time_index - 20]
            union = current | previous
            common[time_index, 13] = 1.0 - len(current & previous) / len(union)
            mask[time_index, 13] = True

    total_log = np.zeros(time_count)
    total_log[total_amount_mask] = np.log1p(total_amount[total_amount_mask])
    total_log_mean20, total_log_mean20_mask = _rolling_mean(
        total_log[:, None], total_amount_mask[:, None], 20
    )
    c2_first_mask = total_amount_mask & total_log_mean20_mask[:, 0]
    common[c2_first_mask, 4] = (
        total_log[c2_first_mask] - total_log_mean20[c2_first_mask, 0]
    )
    mask[:, 4] = c2_first_mask

    style_sources = (
        (
            np.asarray(panel.context["log_circ_mv_lag1"], dtype=np.float64),
            np.asarray(panel.context["log_circ_mv_lag1_mask"], dtype=bool),
            False,
        ),
        (
            np.asarray(panel.context["book_to_price_lag1"], dtype=np.float64),
            np.asarray(panel.context["book_to_price_lag1_mask"], dtype=bool),
            False,
        ),
        (
            np.asarray(panel.context["ret_20"], dtype=np.float64),
            np.asarray(panel.context["ret_20_mask"], dtype=bool),
            True,
        ),
        (
            -np.asarray(panel.context["ret_5"], dtype=np.float64),
            np.asarray(panel.context["ret_5_mask"], dtype=bool),
            True,
        ),
        (
            panel.raw_values[GROUP_ORDER[1]][..., 6],
            panel.raw_masks[GROUP_ORDER[1]][..., 6],
            False,
        ),
        (
            np.asarray(panel.context["realized_vol_20"], dtype=np.float64),
            np.asarray(panel.context["realized_vol_20_mask"], dtype=bool),
            True,
        ),
    )
    for coordinate, (source, source_mask, shift_one) in enumerate(style_sources):
        exposure = np.roll(source, 1, axis=0) if shift_one else source
        exposure_mask = np.roll(source_mask, 1, axis=0) if shift_one else source_mask
        if shift_one:
            exposure_mask[0] = False
        for time_index in range(1, time_count):
            available = (
                panel.signal_universe[time_index]
                & transition_mask[time_index]
                & exposure_mask[time_index]
            )
            indices = np.flatnonzero(available)
            if len(indices) < minimum_stocks:
                continue
            values = exposure[time_index, indices]
            ranks = pd.Series(values).rank(method="average").to_numpy(np.float64)
            bottom = indices[ranks <= 0.20 * len(indices)]
            top = indices[ranks > 0.80 * len(indices)]
            if len(bottom) < 4 or len(top) < 4:
                continue
            common[time_index, 16 + coordinate] = (
                transition[time_index, top].mean() - transition[time_index, bottom].mean()
            )
            mask[time_index, 16 + coordinate] = True

    for time_index in range(59, time_count):
        window60_mask = panel.transition_mask[time_index - 59 : time_index + 1].all(axis=0)
        candidates = panel.signal_universe[time_index] & window60_mask
        indices = np.flatnonzero(candidates)
        if len(indices) < minimum_stocks:
            continue
        returns60 = transition[time_index - 59 : time_index + 1, indices]
        returns20 = returns60[-20:]
        variable = (returns60.var(axis=0) > EPSILON) & (returns20.var(axis=0) > EPSILON)
        indices = indices[variable]
        returns60 = returns60[:, variable]
        returns20 = returns20[:, variable]
        if len(indices) < minimum_stocks:
            continue
        corr60 = np.corrcoef(returns60, rowvar=False)
        corr20 = np.corrcoef(returns20, rowvar=False)
        if not np.isfinite(corr60).all() or not np.isfinite(corr20).all():
            continue
        upper = corr20[np.triu_indices(len(indices), 1)]
        eigenvalues = np.linalg.eigvalsh(corr60)
        trace = np.trace(corr60)
        if upper.size == 0 or trace <= EPSILON:
            continue
        common[time_index, 22] = upper.mean()
        common[time_index, 23] = max(float(eigenvalues[-1]), 0.0) / trace
        common[time_index, 24] = upper.std(ddof=0)
        common[time_index, 25] = np.linalg.norm(corr20 - corr60, ord="fro") / len(indices)
        mask[time_index, 22:26] = True

    common[~mask] = 0.0
    drivers = {
        "market_return": panel.market_return.copy(),
        "market_return_mask": panel.market_return_mask.copy(),
        "log_total_amount": total_log,
        "log_total_amount_mask": total_amount_mask,
    }
    return common, mask, drivers


def _average_ranks(values: ArrayF) -> ArrayF:
    return pd.Series(values).rank(method="average").to_numpy(np.float64)


def _spearman(left: ArrayF, right: ArrayF) -> float:
    if len(left) < 2:
        return np.nan
    left_rank, right_rank = _average_ranks(left), _average_ranks(right)
    left_center = left_rank - left_rank.mean()
    right_center = right_rank - right_rank.mean()
    denominator = np.linalg.norm(left_center) * np.linalg.norm(right_center)
    if denominator <= EPSILON:
        return np.nan
    return float(np.dot(left_center, right_center) / denominator)


def _horizon_fec(
    panel: SignalPanel,
    labels: pd.DataFrame,
    horizon: int,
    *,
    matured_source_window: int,
    minimum_matured_dates: int,
    minimum_common_stocks: int,
) -> tuple[ArrayF, ArrayB, list[dict[str, object]]]:
    time_count = len(panel.dates)
    date_to_index = {date: index for index, date in enumerate(panel.dates)}
    code_to_index = {code: index for index, code in enumerate(panel.codes)}
    frame = labels[labels["horizon"].astype(str).eq(f"H{horizon}")].copy()
    frame["feature_date"] = frame["feature_date"].astype(str)
    frame["ts_code"] = frame["ts_code"].astype(str)
    frame = frame[
        frame["feature_date"].isin(panel.dates) & frame["ts_code"].isin(panel.codes)
    ]
    by_date = {str(date): group for date, group in frame.groupby("feature_date", sort=False)}
    source_rankic = np.zeros((time_count, 7), dtype=np.float64)
    source_support = np.zeros((time_count, 7), dtype=np.float64)
    source_common_count = np.zeros((time_count, 7), dtype=np.int32)
    source_group_mask = np.zeros((time_count, 7), dtype=bool)
    source_c8 = np.zeros((time_count, 4), dtype=np.float64)
    source_c8_mask = np.zeros((time_count, 4), dtype=bool)
    source_c8_support = np.zeros((time_count, 4), dtype=np.int32)
    maturity = np.full(time_count, "", dtype=object)
    source_label_maps: list[dict[str, tuple[float, float]]] = [dict() for _ in range(time_count)]

    for time_index, date in enumerate(panel.dates):
        daily = by_date.get(date)
        if daily is None:
            continue
        valid = daily["label_valid"].fillna(False).astype(bool)
        valid &= pd.to_numeric(daily["primary_rank"], errors="coerce").notna()
        valid &= pd.to_numeric(daily["primary_return"], errors="coerce").notna()
        daily = daily.loc[valid]
        if len(daily) < minimum_common_stocks:
            continue
        maturity_values = daily["label_matured_date"].dropna().astype(str)
        if maturity_values.empty:
            continue
        maturity[time_index] = str(maturity_values.max())
        label_by_code = {
            str(row.ts_code): (float(row.primary_rank), float(row.primary_return))
            for row in daily.itertuples(index=False)
        }
        source_label_maps[time_index] = label_by_code
        label_count = len(label_by_code)
        for group_index, group_name in enumerate(GROUP_ORDER):
            group_mask = panel.feature_masks[group_name][time_index].all(axis=-1)
            proxy = panel.values[group_name][time_index].mean(axis=-1)
            common_codes = [
                code
                for code in label_by_code
                if group_mask[code_to_index[code]]
            ]
            if len(common_codes) < minimum_common_stocks:
                continue
            proxy_values = np.asarray(
                [proxy[code_to_index[code]] for code in common_codes], dtype=np.float64
            )
            rank_values = np.asarray(
                [label_by_code[code][0] for code in common_codes], dtype=np.float64
            )
            correlation = _spearman(proxy_values, rank_values)
            if not np.isfinite(correlation):
                continue
            source_rankic[time_index, group_index] = correlation
            source_support[time_index, group_index] = len(common_codes) / label_count
            source_common_count[time_index, group_index] = len(common_codes)
            source_group_mask[time_index, group_index] = True

        ranks = np.asarray([value[0] for value in label_by_code.values()], dtype=np.float64)
        returns = np.asarray([value[1] for value in label_by_code.values()], dtype=np.float64)
        top = returns[ranks >= 0.90]
        bottom = returns[ranks <= 0.10]
        if len(top) and len(bottom):
            source_c8[time_index, 1] = top.mean() - bottom.mean()
            source_c8_mask[time_index, 1] = True
            source_c8_support[time_index, 1] = label_count
        source_c8[time_index, 2] = returns.std(ddof=0)
        source_c8_mask[time_index, 2] = True
        source_c8_support[time_index, 2] = label_count
        median = np.median(returns)
        q25, q75 = np.quantile(returns, [0.25, 0.75], method="linear")
        iqr = q75 - q25
        source_c8[time_index, 3] = (
            1.0 if iqr <= EPSILON else np.mean(np.abs(returns - median) <= 0.10 * iqr)
        )
        source_c8_mask[time_index, 3] = True
        source_c8_support[time_index, 3] = label_count
        if time_index > 0 and source_label_maps[time_index - 1]:
            previous = source_label_maps[time_index - 1]
            common_codes = sorted(set(label_by_code) & set(previous))
            if len(common_codes) >= minimum_common_stocks:
                correlation = _spearman(
                    np.asarray([label_by_code[code][0] for code in common_codes]),
                    np.asarray([previous[code][0] for code in common_codes]),
                )
                if np.isfinite(correlation):
                    source_c8[time_index, 0] = 1.0 - correlation
                    source_c8_mask[time_index, 0] = True
                    source_c8_support[time_index, 0] = len(common_codes)

    output = np.zeros((time_count, HORIZON_DIMENSION), dtype=np.float64)
    output_mask = np.zeros_like(output, dtype=bool)
    audit: list[dict[str, object]] = []
    for asof_index, asof_date in enumerate(panel.dates):
        matured = np.flatnonzero(
            (maturity != "")
            & (np.asarray(maturity, dtype=str) <= asof_date)
            & (np.arange(time_count) <= asof_index)
        )
        for group_index in range(7):
            eligible = matured[source_group_mask[matured, group_index]]
            eligible = eligible[-matured_source_window:]
            start = group_index * 3
            if len(eligible) >= minimum_matured_dates:
                rankic = source_rankic[eligible, group_index]
                support = source_support[eligible, group_index]
                mean = rankic.mean()
                output[asof_index, start : start + 3] = (
                    mean,
                    mean / max(rankic.std(ddof=0), EPSILON),
                    support.mean(),
                )
                output_mask[asof_index, start : start + 3] = True
                audit.append(
                    {
                        "as_of_date": asof_date,
                        "horizon": f"H{horizon}",
                        "block": f"C7_G{group_index + 1}",
                        "source_anchor_date_max": panel.dates[int(eligible[-1])],
                        "label_matured_date_max": str(max(maturity[eligible])),
                        "aggregation_window": matured_source_window,
                        "matured_source_date_count": int(len(eligible)),
                        "common_stock_support_count": int(
                            source_common_count[eligible[-1], group_index]
                        ),
                    }
                )
        for coordinate in range(4):
            eligible = matured[source_c8_mask[matured, coordinate]]
            eligible = eligible[-matured_source_window:]
            if len(eligible) >= minimum_matured_dates:
                output[asof_index, 21 + coordinate] = source_c8[eligible, coordinate].mean()
                output_mask[asof_index, 21 + coordinate] = True
                audit.append(
                    {
                        "as_of_date": asof_date,
                        "horizon": f"H{horizon}",
                        "block": f"C8_{coordinate}",
                        "source_anchor_date_max": panel.dates[int(eligible[-1])],
                        "label_matured_date_max": str(max(maturity[eligible])),
                        "aggregation_window": matured_source_window,
                        "matured_source_date_count": int(len(eligible)),
                        "common_stock_support_count": int(
                            source_c8_support[eligible[-1], coordinate]
                        ),
                    }
                )
    output[~output_mask] = 0.0
    return output, output_mask, audit


def build_fec_panel(
    panel: SignalPanel,
    labels: pd.DataFrame,
    *,
    minimum_stocks: int = 20,
    matured_source_window: int = 60,
    minimum_matured_dates: int = 20,
    minimum_common_stocks: int = 20,
) -> FECPanel:
    """Build C1--C8 without allowing current or future labels into Query."""

    common, common_mask, drivers = _common_fec(panel, minimum_stocks)
    horizon_raw: dict[int, ArrayF] = {}
    horizon_mask: dict[int, ArrayB] = {}
    audit_rows: list[dict[str, object]] = []
    for horizon in (1, 5):
        value, mask, audit = _horizon_fec(
            panel,
            labels,
            horizon,
            matured_source_window=matured_source_window,
            minimum_matured_dates=minimum_matured_dates,
            minimum_common_stocks=minimum_common_stocks,
        )
        horizon_raw[horizon] = value
        horizon_mask[horizon] = mask
        audit_rows.extend(audit)
    result = FECPanel(
        dates=panel.dates,
        common_raw=common,
        common_mask=common_mask,
        horizon_raw=horizon_raw,
        horizon_mask=horizon_mask,
        maturity_audit=pd.DataFrame.from_records(audit_rows),
        drivers=drivers,
    )
    result.validate()
    return result


def _fit_columns(values: ArrayF, mask: ArrayB, rows: ArrayB) -> tuple[ArrayF, ArrayF, ArrayB]:
    dimension = values.shape[1]
    center = np.zeros(dimension, dtype=np.float64)
    scale = np.ones(dimension, dtype=np.float64)
    supported = np.zeros(dimension, dtype=bool)
    for coordinate in range(dimension):
        selected = rows & mask[:, coordinate]
        if not selected.any():
            continue
        data = values[selected, coordinate]
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        center[coordinate] = median
        scale[coordinate] = max(1.4826 * mad, EPSILON)
        supported[coordinate] = True
    return center, scale, supported


def _fit_fec_scaler_interval(
    fec: FECPanel, *, fit_date_min: str | None, fit_date_max: str
) -> FECScaler:
    dates = np.asarray(fec.dates)
    rows = dates <= str(fit_date_max)
    if fit_date_min is not None:
        if str(fit_date_min) > str(fit_date_max):
            raise ValueError("FEC scaler fit interval is reversed")
        rows &= dates >= str(fit_date_min)
    if not rows.any():
        raise ValueError("FEC scaler fit interval is empty")
    common_center, common_scale, common_supported = _fit_columns(
        fec.common_raw, fec.common_mask, rows
    )
    horizon_center: dict[int, ArrayF] = {}
    horizon_scale: dict[int, ArrayF] = {}
    horizon_supported: dict[int, ArrayB] = {}
    for horizon in (1, 5):
        center, scale, supported = _fit_columns(
            fec.horizon_raw[horizon], fec.horizon_mask[horizon], rows
        )
        horizon_center[horizon], horizon_scale[horizon] = center, scale
        horizon_supported[horizon] = supported
    payload = {
        "fit_date_max": str(fit_date_max),
        "common_center": common_center.tolist(),
        "common_scale": common_scale.tolist(),
        "common_mask": common_supported.tolist(),
        "horizon_center": {str(key): value.tolist() for key, value in horizon_center.items()},
        "horizon_scale": {str(key): value.tolist() for key, value in horizon_scale.items()},
        "horizon_mask": {str(key): value.tolist() for key, value in horizon_supported.items()},
    }
    if fit_date_min is not None:
        payload["fit_date_min"] = str(fit_date_min)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return FECScaler(
        fit_date_max=str(fit_date_max),
        common_center=common_center,
        common_scale=common_scale,
        common_mask=common_supported,
        horizon_center=horizon_center,
        horizon_scale=horizon_scale,
        horizon_mask=horizon_supported,
        scaler_hash=digest,
        fit_date_min=None if fit_date_min is None else str(fit_date_min),
    )


def fit_fec_scaler(fec: FECPanel, *, fit_date_max: str) -> FECScaler:
    """Fit the legacy expanding-prefix scaler used by engineering artifacts."""

    return _fit_fec_scaler_interval(
        fec, fit_date_min=None, fit_date_max=fit_date_max
    )


def fit_fec_scaler_interval(
    fec: FECPanel, *, fit_date_min: str, fit_date_max: str
) -> FECScaler:
    """Fit a checkpoint-local scaler on exactly its training interval."""

    return _fit_fec_scaler_interval(
        fec, fit_date_min=fit_date_min, fit_date_max=fit_date_max
    )


def _expand_msjd(level: ArrayF, level_mask: ArrayB) -> tuple[ArrayF, ArrayB]:
    time_count, dimension = level.shape
    output = np.zeros((time_count, dimension, 6), dtype=np.float64)
    mask = np.zeros_like(output, dtype=bool)
    output[..., 0] = level
    mask[..., 0] = level_mask
    velocities: dict[int, tuple[ArrayF, ArrayB]] = {}
    for slot, lag in enumerate((5, 20, 60), start=1):
        value = np.zeros_like(level)
        valid = np.zeros_like(level_mask)
        value[lag:] = (level[lag:] - level[:-lag]) / lag
        valid[lag:] = level_mask[lag:] & level_mask[:-lag]
        value[~valid] = 0.0
        output[..., slot], mask[..., slot] = value, valid
        velocities[lag] = (value, valid)
    for slot, (left, right) in enumerate(((5, 20), (20, 60)), start=4):
        valid = velocities[left][1] & velocities[right][1]
        value = velocities[left][0] - velocities[right][0]
        output[..., slot] = np.where(valid, value, 0.0)
        mask[..., slot] = valid
    return output.reshape(time_count, dimension * 6), mask.reshape(time_count, dimension * 6)


def build_msjd_panel(fec: FECPanel, scaler: FECScaler) -> MSJDPanel:
    values: dict[int, ArrayF] = {}
    masks: dict[int, ArrayB] = {}
    common = (fec.common_raw - scaler.common_center) / scaler.common_scale
    common_mask = fec.common_mask & scaler.common_mask[None, :]
    common[~common_mask] = 0.0
    for horizon in (1, 5):
        specific = (
            fec.horizon_raw[horizon] - scaler.horizon_center[horizon]
        ) / scaler.horizon_scale[horizon]
        specific_mask = fec.horizon_mask[horizon] & scaler.horizon_mask[horizon][None, :]
        specific[~specific_mask] = 0.0
        level = np.concatenate([common, specific], axis=1)
        level_mask = np.concatenate([common_mask, specific_mask], axis=1)
        values[horizon], masks[horizon] = _expand_msjd(level, level_mask)
    result = MSJDPanel(fec.dates, values, masks, scaler.scaler_hash)
    result.validate()
    return result
