"""Deterministic design decisions for the repaired-label v3.6 rebaseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ALPHAS = (0.50, 0.75, 1.00)
BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED = 271_828


def median_bootstrap_se(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("Episode bootstrap requires a finite vector")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(
        0, len(array), size=(BOOTSTRAP_REPEATS, len(array)), endpoint=False
    )
    return float(np.std(np.median(array[draws], axis=1), ddof=1))


@dataclass(frozen=True)
class ResamplingDecision:
    selected_method: str | None
    extend_b40_both: bool
    paired_gain_median: float
    paired_gain_bootstrap_se: float
    r1_win_count: int
    paired_cosine_median: float
    reason: str


def choose_resampling(
    r0_gain: Sequence[float],
    r1_gain: Sequence[float],
    r0_cosine: Sequence[float],
    r1_cosine: Sequence[float],
    *,
    stage: str,
    support_stable: bool,
) -> ResamplingDecision:
    vectors = [
        np.asarray(values, dtype=np.float64)
        for values in (r0_gain, r1_gain, r0_cosine, r1_cosine)
    ]
    if any(len(values) != 20 or not np.isfinite(values).all() for values in vectors):
        raise ValueError("R selection requires 20 finite paired Episodes")
    gain_difference = vectors[1] - vectors[0]
    cosine_difference = vectors[3] - vectors[2]
    gain_median = float(np.median(gain_difference))
    gain_se = median_bootstrap_se(gain_difference)
    wins = int(np.count_nonzero(gain_difference > 0.0))
    cosine_median = float(np.median(cosine_difference))
    if (
        gain_median > gain_se
        and wins >= 12
        and cosine_median >= 0.0
        and support_stable
    ):
        return ResamplingDecision(
            "R1", False, gain_median, gain_se, wins, cosine_median, "CLEAR_R1"
        )
    if (
        gain_median < -gain_se
        and wins <= 8
        and cosine_median <= 0.0
        and support_stable
    ):
        return ResamplingDecision(
            "R0", False, gain_median, gain_se, wins, cosine_median, "CLEAR_R0"
        )
    if stage == "B20":
        return ResamplingDecision(
            None,
            True,
            gain_median,
            gain_se,
            wins,
            cosine_median,
            "B20_AMBIGUOUS_EXTEND_BOTH",
        )
    if gain_median >= 0.0 and cosine_median >= 0.0 and support_stable:
        return ResamplingDecision(
            "R1",
            False,
            gain_median,
            gain_se,
            wins,
            cosine_median,
            "B40_R1_NONINFERIOR_TIEBREAK",
        )
    return ResamplingDecision(
        None,
        False,
        gain_median,
        gain_se,
        wins,
        cosine_median,
        "B40_UNRESOLVED",
    )


@dataclass(frozen=True)
class BootstrapDecision:
    selected_repeats: int
    cosine_median: float
    cosine_q10: float
    covariance_trace_relative_difference_median: float
    minimum_valid_fraction: float


def choose_bootstrap_repeats(
    mean_action_cosines: Sequence[float],
    covariance_trace_relative_differences: Sequence[float],
    valid_fractions_b40: Sequence[float],
) -> BootstrapDecision:
    cosine = np.asarray(mean_action_cosines, dtype=np.float64)
    covariance = np.asarray(
        covariance_trace_relative_differences, dtype=np.float64
    )
    valid = np.asarray(valid_fractions_b40, dtype=np.float64)
    if any(
        len(values) != 20 or not np.isfinite(values).all()
        for values in (cosine, covariance, valid)
    ):
        raise ValueError("B selection requires 20 finite Episodes")
    median = float(np.median(cosine))
    q10 = float(np.quantile(cosine, 0.10))
    covariance_median = float(np.median(covariance))
    minimum_valid = float(np.min(valid))
    selected = (
        20
        if median >= 0.95
        and q10 >= 0.90
        and covariance_median <= 0.15
        and minimum_valid >= 0.80
        else 40
    )
    return BootstrapDecision(
        selected, median, q10, covariance_median, minimum_valid
    )


@dataclass(frozen=True)
class AlphaDecision:
    selected_alpha: float | None
    best_alpha: float | None
    best_median: float
    bootstrap_se: float
    acceptance_floor: float
    acceptable_alphas: tuple[float, ...]


def choose_alpha(
    episode_medians: Mapping[float, Sequence[float]],
) -> AlphaDecision:
    if tuple(sorted(float(value) for value in episode_medians)) != ALPHAS:
        raise PermissionError("alpha candidate set changed")
    vectors = {
        float(alpha): np.asarray(values, dtype=np.float64)
        for alpha, values in episode_medians.items()
    }
    if any(
        len(values) != 20 or not np.isfinite(values).all()
        for values in vectors.values()
    ):
        raise ValueError("alpha selection requires 20 finite Episode medians")
    aggregate = {
        alpha: float(np.median(values)) for alpha, values in vectors.items()
    }
    positive = {alpha: value for alpha, value in aggregate.items() if value > 0.0}
    if not positive:
        return AlphaDecision(None, None, max(aggregate.values()), 0.0, 0.0, ())
    best = max(positive, key=lambda alpha: (positive[alpha], -alpha))
    best_median = positive[best]
    se = median_bootstrap_se(vectors[best])
    floor = best_median - se
    acceptable = tuple(
        alpha
        for alpha in ALPHAS
        if aggregate[alpha] > 0.0 and aggregate[alpha] >= floor
    )
    return AlphaDecision(min(acceptable), best, best_median, se, floor, acceptable)


def quality_values(
    *,
    gain_median: float,
    positive_count: int,
    valid_count: int,
    gain_reference: float,
) -> dict[str, float]:
    if (
        not np.isfinite(gain_median)
        or not np.isfinite(gain_reference)
        or gain_reference <= 0.0
        or valid_count <= 0
        or not 0 <= positive_count <= valid_count
    ):
        raise ValueError("continuous quality inputs are invalid")
    q_gain = float(np.clip(gain_median / gain_reference, 0.0, 1.0))
    p_hat = float((positive_count + 1) / (valid_count + 2))
    q_stability = float(np.clip(2.0 * (p_hat - 0.5), 0.0, 1.0))
    return {
        "q_gain": q_gain,
        "p_hat": p_hat,
        "q_stab": q_stability,
        "q_anchor": float(np.sqrt(q_gain * q_stability)),
    }


def quality_diagnostics(values: Sequence[float]) -> dict[str, float | int]:
    quality = np.asarray(values, dtype=np.float64)
    if (
        quality.ndim != 1
        or not len(quality)
        or not np.isfinite(quality).all()
        or np.any(quality < 0.0)
    ):
        raise ValueError("quality vector is invalid")
    mass = float(quality.sum())
    if mass <= 0.0:
        return {
            "quality_N_eff": 0.0,
            "maximum_episode_weight": 0.0,
            "top2_weight": 0.0,
            "zero_quality_count": int(len(quality)),
            "equal_quality_N_eff": float(len(quality)),
            "equal_quality_maximum_weight": float(1.0 / len(quality)),
        }
    weights = quality / mass
    return {
        "quality_N_eff": float(1.0 / np.square(weights).sum()),
        "maximum_episode_weight": float(np.max(weights)),
        "top2_weight": float(np.sort(weights)[-2:].sum()),
        "zero_quality_count": int(np.count_nonzero(quality == 0.0)),
        "equal_quality_N_eff": float(len(quality)),
        "equal_quality_maximum_weight": float(1.0 / len(quality)),
    }
