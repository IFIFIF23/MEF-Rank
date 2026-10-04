"""Complete deterministic 77-coordinate G7 materialization.

This module composes the audited daily Chen primitive into the frozen
5 ordinary candle + 36 D3mean + 36 D3ord dictionary.  It performs no file I/O
and never consults a label universe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .g7_chen import cross_section_average_tie_rank, daily_d3, trailing_complete_means


@dataclass(frozen=True)
class G7BuildResult:
    values: npt.NDArray[np.float32]
    feature_mask: npt.NDArray[np.bool_]
    group_lag_mask: npt.NDArray[np.bool_]
    volume_coordinate: npt.NDArray[np.float64]
    volume_coordinate_mask: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name in (
            "values",
            "feature_mask",
            "group_lag_mask",
            "volume_coordinate",
            "volume_coordinate_mask",
        ):
            owned = np.array(getattr(self, name), copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


def _as_float_matrix(name: str, value: npt.ArrayLike) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [T,N]")
    return array


def _volume_coordinate(
    volume: npt.NDArray[np.float64],
    market_day_available: npt.NDArray[np.bool_],
    signal_universe: npt.NDArray[np.bool_],
    *,
    maximum_observations: int = 60,
    minimum_observations: int = 10,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    valid = (
        np.isfinite(volume)
        & (volume >= 0)
        & market_day_available[:, None]
        & signal_universe
    )
    log_volume = np.zeros_like(volume, dtype=np.float64)
    log_volume[valid] = np.log1p(volume[valid])
    result = np.zeros_like(volume, dtype=np.float64)
    result_mask = np.zeros_like(valid, dtype=bool)
    for stock in range(volume.shape[1]):
        available_indices: list[int] = []
        for time_index in range(volume.shape[0]):
            if not valid[time_index, stock]:
                continue
            available_indices.append(time_index)
            reference = available_indices[-maximum_observations:]
            if len(reference) < minimum_observations:
                continue
            result[time_index, stock] = log_volume[time_index, stock] - float(
                np.mean(log_volume[reference, stock])
            )
            result_mask[time_index, stock] = True
    return result, result_mask


def build_g7_77(
    open_: npt.ArrayLike,
    high: npt.ArrayLike,
    low: npt.ArrayLike,
    close: npt.ArrayLike,
    volume: npt.ArrayLike,
    signal_universe_mask: npt.ArrayLike,
    market_day_available_mask: npt.ArrayLike,
) -> G7BuildResult:
    """Build ranked G7 values on an explicit exchange-date grid.

    All OHLCV arrays and ``signal_universe_mask`` have shape ``[T,N]``.
    ``market_day_available_mask`` has shape ``[T]`` and must keep unresolved
    full-market gaps on the grid as ``False``; those gaps invalidate every D3
    complete window that crosses them.
    """

    open_array = _as_float_matrix("open", open_)
    arrays = {
        "high": _as_float_matrix("high", high),
        "low": _as_float_matrix("low", low),
        "close": _as_float_matrix("close", close),
        "volume": _as_float_matrix("volume", volume),
    }
    if any(array.shape != open_array.shape for array in arrays.values()):
        raise ValueError("all OHLCV inputs must share shape [T,N]")
    signal_universe = np.asarray(signal_universe_mask, dtype=bool)
    market_days = np.asarray(market_day_available_mask, dtype=bool)
    if signal_universe.shape != open_array.shape:
        raise ValueError("signal_universe_mask must have shape [T,N]")
    if market_days.shape != (open_array.shape[0],):
        raise ValueError("market_day_available_mask must have shape [T]")

    high_array = arrays["high"]
    low_array = arrays["low"]
    close_array = arrays["close"]
    volume_array = arrays["volume"]
    finite_positive = (
        np.isfinite(open_array)
        & np.isfinite(high_array)
        & np.isfinite(low_array)
        & np.isfinite(close_array)
        & (open_array > 0)
        & (high_array > 0)
        & (low_array > 0)
        & (close_array > 0)
    )
    legal = (
        finite_positive
        & (high_array >= np.maximum(open_array, close_array))
        & (low_array <= np.minimum(open_array, close_array))
        & (high_array > low_array)
        & signal_universe
        & market_days[:, None]
    )

    volume_coordinate, volume_mask = _volume_coordinate(
        volume_array, market_days, signal_universe
    )
    daily_valid = legal & volume_mask
    ordinary = np.zeros((*open_array.shape, 5), dtype=np.float64)
    ordinary_mask = np.broadcast_to(legal[..., None], ordinary.shape).copy()
    ordinary[..., 0][legal] = np.log(close_array[legal] / open_array[legal])
    ordinary[..., 1][legal] = np.log(high_array[legal] / low_array[legal])
    price_range = high_array - low_array
    ordinary[..., 2][legal] = (
        high_array[legal] - np.maximum(open_array[legal], close_array[legal])
    ) / price_range[legal]
    ordinary[..., 3][legal] = (
        np.minimum(open_array[legal], close_array[legal]) - low_array[legal]
    ) / price_range[legal]
    ordinary[..., 4][legal] = (close_array[legal] - low_array[legal]) / price_range[legal]

    daily_mean = np.zeros((*open_array.shape, 18), dtype=np.float64)
    daily_order = np.zeros((*open_array.shape, 18), dtype=np.float64)
    daily_coordinate_mask = np.broadcast_to(daily_valid[..., None], daily_mean.shape).copy()
    for time_index, stock_index in zip(*np.nonzero(daily_valid), strict=True):
        value = daily_d3(
            open_array[time_index, stock_index],
            high_array[time_index, stock_index],
            low_array[time_index, stock_index],
            close_array[time_index, stock_index],
            volume_coordinate[time_index, stock_index],
        )
        daily_mean[time_index, stock_index] = value.d3mean
        daily_order[time_index, stock_index] = value.d3ord

    time_count, stock_count = open_array.shape
    aggregated_mean = np.zeros((time_count, stock_count, 2, 18), dtype=np.float64)
    aggregated_order = np.zeros_like(aggregated_mean)
    aggregated_mean_mask = np.zeros_like(aggregated_mean, dtype=bool)
    aggregated_order_mask = np.zeros_like(aggregated_mean, dtype=bool)
    for stock_index in range(stock_count):
        mean_value, mean_mask = trailing_complete_means(
            daily_mean[:, stock_index], daily_coordinate_mask[:, stock_index]
        )
        order_value, order_mask = trailing_complete_means(
            daily_order[:, stock_index], daily_coordinate_mask[:, stock_index]
        )
        aggregated_mean[:, stock_index] = mean_value
        aggregated_order[:, stock_index] = order_value
        aggregated_mean_mask[:, stock_index] = mean_mask
        aggregated_order_mask[:, stock_index] = order_mask

    raw = np.concatenate(
        [
            ordinary,
            aggregated_mean.reshape(time_count, stock_count, 36),
            aggregated_order.reshape(time_count, stock_count, 36),
        ],
        axis=-1,
    )
    raw_mask = np.concatenate(
        [
            ordinary_mask,
            aggregated_mean_mask.reshape(time_count, stock_count, 36),
            aggregated_order_mask.reshape(time_count, stock_count, 36),
        ],
        axis=-1,
    )
    ranked = np.zeros_like(raw, dtype=np.float64)
    ranked_mask = np.zeros_like(raw_mask, dtype=bool)
    for time_index in range(time_count):
        ranked[time_index], ranked_mask[time_index] = cross_section_average_tie_rank(
            raw[time_index], raw_mask[time_index]
        )
    ranked[~ranked_mask] = 0.0
    group_lag_mask = ranked_mask.all(axis=-1)
    return G7BuildResult(
        values=ranked.astype(np.float32),
        feature_mask=ranked_mask,
        group_lag_mask=group_lag_mask,
        volume_coordinate=volume_coordinate,
        volume_coordinate_mask=volume_mask,
    )
