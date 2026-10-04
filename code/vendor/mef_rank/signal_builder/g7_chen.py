"""Level6-native G7 daily path and raw third-level Chen primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


SELECTED_D3_INDICES = np.asarray(
    [1, 3, 4, 5, 7, 9, 10, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 25],
    dtype=np.int64,
)


@dataclass(frozen=True)
class DailyD3:
    d3mean: npt.NDArray[np.float64]
    d3ord: npt.NDArray[np.float64]
    hl_signature_flat: npt.NDArray[np.float64]
    lh_signature_flat: npt.NDArray[np.float64]


def chen_signature_level3(path: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return the raw third-level signature tensor for a piecewise-linear path."""

    points = np.asarray(path, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("path must have shape [points>=2,3]")
    if not np.isfinite(points).all():
        raise ValueError("path must be finite")

    first = np.zeros(3, dtype=np.float64)
    second = np.zeros((3, 3), dtype=np.float64)
    third = np.zeros((3, 3, 3), dtype=np.float64)
    for increment in np.diff(points, axis=0):
        # All right-hand-side lower levels are pre-update values.
        third = (
            third
            + np.einsum("ab,c->abc", second, increment)
            + 0.5 * np.einsum("a,b,c->abc", first, increment, increment)
            + (1.0 / 6.0) * np.einsum("a,b,c->abc", increment, increment, increment)
        )
        second = second + np.outer(first, increment) + 0.5 * np.outer(increment, increment)
        first = first + increment
    return third


def _validate_ohlc(open_: float, high: float, low: float, close: float) -> None:
    values = np.asarray([open_, high, low, close], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("OHLC must be finite and strictly positive")
    if high < max(open_, close) or low > min(open_, close) or high < low:
        raise ValueError("illegal OHLC ordering")


def candidate_paths(
    open_: float, high: float, low: float, close: float, volume_shock: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Construct semantic HL and LH paths in [time, log-price, volume] order."""

    _validate_ohlc(open_, high, low, close)
    if not np.isfinite(volume_shock):
        raise ValueError("volume_shock must be finite")
    log_open, log_high, log_low, log_close = np.log([open_, high, low, close])
    hl = np.asarray(
        [
            [0.0, log_open, 0.0],
            [1.0, log_high, volume_shock / 3.0],
            [2.0, log_low, 2.0 * volume_shock / 3.0],
            [3.0, log_close, volume_shock],
        ],
        dtype=np.float64,
    )
    lh = np.asarray(
        [
            [0.0, log_open, 0.0],
            [1.0, log_low, volume_shock / 3.0],
            [2.0, log_high, 2.0 * volume_shock / 3.0],
            [3.0, log_close, volume_shock],
        ],
        dtype=np.float64,
    )
    return hl, lh


def daily_d3(
    open_: float, high: float, low: float, close: float, volume_shock: float
) -> DailyD3:
    """Compute selected daily D3mean and HL-minus-LH full difference."""

    hl, lh = candidate_paths(open_, high, low, close, volume_shock)
    hl_flat = chen_signature_level3(hl).reshape(-1, order="C")
    lh_flat = chen_signature_level3(lh).reshape(-1, order="C")
    mean = 0.5 * (hl_flat + lh_flat)
    order_contrast = hl_flat - lh_flat
    return DailyD3(
        d3mean=mean[SELECTED_D3_INDICES],
        d3ord=order_contrast[SELECTED_D3_INDICES],
        hl_signature_flat=hl_flat,
        lh_signature_flat=lh_flat,
    )


def trailing_complete_means(
    daily_values: npt.ArrayLike,
    daily_valid: npt.ArrayLike,
    *,
    windows: tuple[int, ...] = (5, 20),
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Aggregate daily D3 coordinates using only complete trailing windows.

    The returned value shape is ``[T, len(windows), F]`` and the mask shape is
    identical.  Invalid coordinates retain the fixed numeric placeholder zero.
    """

    values = np.asarray(daily_values, dtype=np.float64)
    valid = np.asarray(daily_valid, dtype=bool)
    if values.ndim != 2:
        raise ValueError("daily_values must have shape [T,F]")
    if valid.shape not in {(values.shape[0],), values.shape}:
        raise ValueError("daily_valid must have shape [T] or [T,F]")
    if valid.ndim == 1:
        valid = np.broadcast_to(valid[:, None], values.shape)
    if not np.isfinite(values[valid]).all():
        raise ValueError("valid daily values must be finite")
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("windows must contain positive integers")

    output = np.zeros((values.shape[0], len(windows), values.shape[1]), dtype=np.float64)
    output_mask = np.zeros_like(output, dtype=bool)
    for window_index, window in enumerate(windows):
        for endpoint in range(window - 1, values.shape[0]):
            start = endpoint - window + 1
            coordinate_valid = valid[start : endpoint + 1].all(axis=0)
            if coordinate_valid.any():
                output[endpoint, window_index, coordinate_valid] = values[
                    start : endpoint + 1, coordinate_valid
                ].mean(axis=0)
                output_mask[endpoint, window_index, coordinate_valid] = True
    return output, output_mask


def cross_section_average_tie_rank(
    values: npt.ArrayLike,
    available_mask: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Apply the frozen per-date ``2*average_rank/n-1`` transform by coordinate."""

    raw = np.asarray(values, dtype=np.float64)
    mask = np.asarray(available_mask, dtype=bool)
    if raw.ndim != 2 or mask.shape != raw.shape:
        raise ValueError("values and available_mask must have shape [N,F]")
    if not np.isfinite(raw[mask]).all():
        raise ValueError("available values must be finite")
    ranked = np.zeros_like(raw, dtype=np.float64)
    ranked_mask = np.zeros_like(mask, dtype=bool)
    for coordinate in range(raw.shape[1]):
        stock_indices = np.flatnonzero(mask[:, coordinate])
        count = stock_indices.size
        if count < 2:
            continue
        coordinate_values = raw[stock_indices, coordinate]
        order = np.argsort(coordinate_values, kind="mergesort")
        sorted_values = coordinate_values[order]
        ranks = np.empty(count, dtype=np.float64)
        start = 0
        while start < count:
            end = start + 1
            while end < count and sorted_values[end] == sorted_values[start]:
                end += 1
            average_rank = ((start + 1) + end) / 2.0
            ranks[order[start:end]] = average_rank
            start = end
        ranked[stock_indices, coordinate] = 2.0 * ranks / count - 1.0
        ranked_mask[stock_indices, coordinate] = True
    return ranked, ranked_mask
