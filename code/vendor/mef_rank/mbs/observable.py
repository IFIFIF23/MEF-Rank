"""Deterministic MBS-Full/MBS-Compact construction from STRForward only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..str_model.contracts import STRForward


@dataclass(frozen=True)
class MBSResult:
    full_values: npt.NDArray[np.float64]
    full_mask: npt.NDArray[np.bool_]
    compact_values: npt.NDArray[np.float64]
    compact_mask: npt.NDArray[np.bool_]
    pair_common_support: npt.NDArray[np.int64]
    pair_mask: npt.NDArray[np.bool_]
    R_block_valid: npt.NDArray[np.bool_]
    U_block_valid: npt.NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            owned = np.array(getattr(self, name), copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)


def build_mbs(
    forward: STRForward,
    signal_universe_mask: npt.ArrayLike,
    group_available_mask: npt.ArrayLike,
    *,
    matured_performance: npt.ArrayLike | None = None,
    matured_performance_mask: npt.ArrayLike | None = None,
    group_min_stocks: int = 20,
    pair_min_common_stocks: int = 20,
    exposure_mass_epsilon: float = 1.0e-12,
    centered_norm_epsilon: float = 1.0e-12,
) -> MBSResult:
    """Build MBS without reading arbitrary hidden state or filling missing blocks."""

    forward.validate()
    signal = np.asarray(signal_universe_mask, dtype=bool)
    groups = np.asarray(group_available_mask, dtype=bool)
    if signal.shape != forward.score.shape:
        raise ValueError("signal_universe_mask must have shape [B,N]")
    if groups.shape != (*forward.score.shape, 7):
        raise ValueError("group_available_mask must have shape [B,N,7]")
    if group_min_stocks <= 0 or pair_min_common_stocks <= 0:
        raise ValueError("minimum supports must be positive")

    batch, stocks = forward.score.shape
    r_mean = np.zeros((batch, 7), dtype=np.float64)
    r_std = np.zeros((batch, 7), dtype=np.float64)
    r_mask = np.zeros((batch, 7), dtype=bool)
    exposure = np.zeros((batch, 7), dtype=np.float64)
    exposure_mask = np.zeros((batch, 7), dtype=bool)
    disagreement = np.zeros((batch, 7), dtype=np.float64)
    disagreement_mask = np.zeros((batch, 7), dtype=bool)
    pair_support = np.zeros((batch, 7, 7), dtype=np.int64)
    pair_mask = np.zeros((batch, 7, 7), dtype=bool)
    u_value = np.zeros((batch, 5), dtype=np.float64)
    u_mask = np.zeros((batch, 5), dtype=bool)

    for batch_index in range(batch):
        valid_groups: list[int] = []
        raw_exposure = np.zeros(7, dtype=np.float64)
        for group in range(7):
            stock_mask = signal[batch_index] & groups[batch_index, :, group]
            if int(stock_mask.sum()) < group_min_stocks:
                continue
            reliability = forward.reliability[batch_index, stock_mask, group]
            if not np.isfinite(reliability).all():
                continue
            r_mean[batch_index, group] = float(np.mean(reliability))
            r_std[batch_index, group] = float(np.std(reliability, ddof=0))
            r_mask[batch_index, group] = True
            vectors = forward.weighted_group_pooled[batch_index, stock_mask, group]
            if np.isfinite(vectors).all():
                raw_exposure[group] = float(np.mean(np.linalg.norm(vectors, axis=-1)))
                valid_groups.append(group)
        mass = float(raw_exposure[valid_groups].sum()) if valid_groups else 0.0
        if mass > exposure_mass_epsilon:
            exposure[batch_index, valid_groups] = raw_exposure[valid_groups] / mass
            exposure_mask[batch_index, valid_groups] = True

        pair_distances: list[list[float]] = [[] for _ in range(7)]
        for left in range(7):
            for right in range(left + 1, 7):
                common = (
                    signal[batch_index]
                    & groups[batch_index, :, left]
                    & groups[batch_index, :, right]
                )
                support = int(common.sum())
                pair_support[batch_index, left, right] = support
                pair_support[batch_index, right, left] = support
                if support < pair_min_common_stocks:
                    continue
                left_value = forward.weighted_group_pooled[batch_index, common, left]
                right_value = forward.weighted_group_pooled[batch_index, common, right]
                if not np.isfinite(left_value).all() or not np.isfinite(right_value).all():
                    continue
                left_centered = left_value - left_value.mean(axis=0, keepdims=True)
                right_centered = right_value - right_value.mean(axis=0, keepdims=True)
                left_norm = float(np.linalg.norm(left_centered))
                right_norm = float(np.linalg.norm(right_centered))
                if left_norm <= centered_norm_epsilon or right_norm <= centered_norm_epsilon:
                    continue
                cosine = float(np.sum(left_centered * right_centered) / (left_norm * right_norm))
                distance = (1.0 - float(np.clip(cosine, -1.0, 1.0))) / 2.0
                pair_mask[batch_index, left, right] = True
                pair_mask[batch_index, right, left] = True
                pair_distances[left].append(distance)
                pair_distances[right].append(distance)
        for group, values in enumerate(pair_distances):
            if values:
                disagreement[batch_index, group] = float(np.mean(values))
                disagreement_mask[batch_index, group] = True

        stock_mask = signal[batch_index]
        count = int(stock_mask.sum())
        if count >= 2:
            uncertainty = forward.uncertainty[batch_index, stock_mask]
            score = forward.score[batch_index, stock_mask]
            if np.isfinite(uncertainty).all() and np.isfinite(score).all():
                median = float(np.median(score))
                iqr = float(np.quantile(score, 0.75) - np.quantile(score, 0.25))
                top_count = max(1, int(np.ceil(0.10 * count)))
                top_mean = float(np.mean(np.sort(score)[-top_count:]))
                u_value[batch_index] = [
                    float(np.mean(uncertainty)),
                    float(np.quantile(uncertainty, 0.90)),
                    float(np.std(score, ddof=0)),
                    top_mean - median,
                    1.0 if iqr <= 1.0e-12 else float(np.mean(np.abs(score - median) <= 0.10 * iqr)),
                ]
                u_mask[batch_index] = True

    p_value = np.zeros((batch, 4), dtype=np.float64)
    p_mask = np.zeros((batch, 4), dtype=bool)
    if matured_performance is not None or matured_performance_mask is not None:
        if matured_performance is None or matured_performance_mask is None:
            raise ValueError("matured performance values and mask must be supplied together")
        p_value_input = np.asarray(matured_performance, dtype=np.float64)
        p_mask_input = np.asarray(matured_performance_mask, dtype=bool)
        if p_value_input.shape != (batch, 4) or p_mask_input.shape != (batch, 4):
            raise ValueError("matured performance values and mask must have shape [B,4]")
        if not np.isfinite(p_value_input[p_mask_input]).all():
            raise ValueError("available matured performance values must be finite")
        p_value[p_mask_input] = p_value_input[p_mask_input]
        p_mask = p_mask_input.copy()

    full = np.concatenate([r_mean, r_std, exposure, disagreement, u_value, p_value], axis=-1)
    full_mask = np.concatenate([r_mask, r_mask, exposure_mask, disagreement_mask, u_mask, p_mask], axis=-1)
    compact = np.concatenate([r_mean, r_std, u_value, p_value], axis=-1)
    compact_mask = np.concatenate([r_mask, r_mask, u_mask, p_mask], axis=-1)
    full[~full_mask] = 0.0
    compact[~compact_mask] = 0.0
    return MBSResult(
        full_values=full,
        full_mask=full_mask,
        compact_values=compact,
        compact_mask=compact_mask,
        pair_common_support=pair_support,
        pair_mask=pair_mask,
        R_block_valid=r_mask.any(axis=-1),
        U_block_valid=u_mask.all(axis=-1),
    )
