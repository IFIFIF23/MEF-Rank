"""Frozen v3.6 JACF/CAL0 joint retrieval and execution primitives.

This module is deliberately independent of the historical v3.3 group-wise
ACF implementation.  It implements one joint seven-dimensional neighborhood,
one quality factor in the unnormalised mass, and all-or-none rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ArrayF = NDArray[np.float64]
ArrayB = NDArray[np.bool_]
J_BLOCKS = {
    "level": np.asarray([row * 6 for row in range(51)], dtype=np.int64),
    "velocity": np.asarray(
        [row * 6 + column for row in range(51) for column in (1, 2, 3)],
        dtype=np.int64,
    ),
    "curvature": np.asarray(
        [row * 6 + column for row in range(51) for column in (4, 5)],
        dtype=np.int64,
    ),
}
B_BLOCKS = {
    "R": np.arange(0, 14, dtype=np.int64),
    "E": np.arange(14, 21, dtype=np.int64),
    "D": np.arange(21, 28, dtype=np.int64),
    "U": np.arange(28, 33, dtype=np.int64),
    "P": np.arange(33, 37, dtype=np.int64),
}


def _symmetric_psd(value: ArrayF, floor: float = 1.0e-10) -> ArrayF:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square")
    matrix = (matrix + matrix.T) / 2.0
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite")
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


@dataclass(frozen=True)
class BlockCalibration:
    median: ArrayF
    mad: ArrayF
    covariance: ArrayF
    minimum_common_fraction: float = 0.50

    def __post_init__(self) -> None:
        median = np.asarray(self.median, dtype=np.float64)
        mad = np.asarray(self.mad, dtype=np.float64)
        covariance = _symmetric_psd(np.asarray(self.covariance, dtype=np.float64))
        if (
            median.ndim != 1
            or mad.shape != median.shape
            or covariance.shape != (len(median), len(median))
            or not 0.0 < float(self.minimum_common_fraction) <= 1.0
        ):
            raise ValueError("block calibration axes are invalid")
        object.__setattr__(self, "median", median)
        object.__setattr__(self, "mad", np.maximum(mad, 1.0e-8))
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "_inverse_cache", {})

    def distance(
        self, left: ArrayF, left_mask: ArrayB, right: ArrayF, right_mask: ArrayB
    ) -> float | None:
        x = np.asarray(left, dtype=np.float64)
        y = np.asarray(right, dtype=np.float64)
        mask = np.asarray(left_mask, dtype=bool) & np.asarray(right_mask, dtype=bool)
        if x.shape != self.median.shape or y.shape != x.shape or mask.shape != x.shape:
            raise ValueError("block query axes changed")
        support = np.flatnonzero(mask)
        if len(support) / len(mask) < self.minimum_common_fraction:
            return None
        difference = (x[support] - y[support]) / self.mad[support]
        key = mask.tobytes()
        inverse = self._inverse_cache.get(key)
        if inverse is None:
            covariance = self.covariance[np.ix_(support, support)]
            inverse = np.linalg.pinv(covariance, rcond=1.0e-10)
            self._inverse_cache[key] = inverse
        squared = float(difference @ inverse @ difference)
        return float(np.sqrt(max(0.0, len(mask) / len(support) * squared)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median.tolist(),
            "mad": self.mad.tolist(),
            "covariance": self.covariance.tolist(),
            "minimum_common_fraction": self.minimum_common_fraction,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BlockCalibration":
        return cls(
            np.asarray(value["median"], dtype=np.float64),
            np.asarray(value["mad"], dtype=np.float64),
            np.asarray(value["covariance"], dtype=np.float64),
            float(value["minimum_common_fraction"]),
        )


def fit_block_calibration(
    values: ArrayF, masks: ArrayB, *, minimum_common_fraction: float = 0.50
) -> BlockCalibration:
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool)
    if data.ndim != 2 or valid.shape != data.shape or len(data) < 2:
        raise ValueError("block calibration sample is invalid")
    median = np.zeros(data.shape[1], dtype=np.float64)
    mad = np.ones(data.shape[1], dtype=np.float64)
    filled = np.zeros_like(data)
    for column in range(data.shape[1]):
        observed = data[valid[:, column], column]
        if len(observed):
            median[column] = float(np.median(observed))
            mad[column] = max(
                1.0e-8, float(np.median(np.abs(observed - median[column])))
            )
        filled[:, column] = np.where(
            valid[:, column], (data[:, column] - median[column]) / mad[column], 0.0
        )
    centered = filled - filled.mean(axis=0, keepdims=True)
    sample = centered.T @ centered / max(1, len(filled) - 1)
    target = np.diag(np.diag(sample))
    shrinkage = min(1.0, data.shape[1] / max(float(len(data)), 1.0))
    covariance = (1.0 - shrinkage) * sample + shrinkage * target
    return BlockCalibration(median, mad, covariance, minimum_common_fraction)


@dataclass(frozen=True)
class JointState:
    j: ArrayF
    j_mask: ArrayB
    b: ArrayF
    b_mask: ArrayB

    def __post_init__(self) -> None:
        j = np.asarray(self.j, dtype=np.float64)
        jm = np.asarray(self.j_mask, dtype=bool)
        b = np.asarray(self.b, dtype=np.float64)
        bm = np.asarray(self.b_mask, dtype=bool)
        if j.shape != (306,) or jm.shape != j.shape or b.shape != (37,) or bm.shape != b.shape:
            raise ValueError("v3.6 J/B state schema changed")
        if not np.isfinite(j[jm]).all() or not np.isfinite(b[bm]).all():
            raise ValueError("v3.6 J/B available state is non-finite")
        object.__setattr__(self, "j", np.where(jm, j, 0.0))
        object.__setattr__(self, "j_mask", jm)
        object.__setattr__(self, "b", np.where(bm, b, 0.0))
        object.__setattr__(self, "b_mask", bm)


@dataclass(frozen=True)
class DistanceCalibration:
    j_blocks: Mapping[str, BlockCalibration]
    b_blocks: Mapping[str, BlockCalibration]

    def distances(self, left: JointState, right: JointState) -> tuple[float, float] | None:
        j_values: list[float] = []
        for name, indices in J_BLOCKS.items():
            value = self.j_blocks[name].distance(
                left.j[indices], left.j_mask[indices], right.j[indices], right.j_mask[indices]
            )
            if value is None:
                return None
            j_values.append(value)
        b_values: list[float] = []
        for name, indices in B_BLOCKS.items():
            value = self.b_blocks[name].distance(
                left.b[indices], left.b_mask[indices], right.b[indices], right.b_mask[indices]
            )
            if value is not None:
                b_values.append(value)
            elif name in {"R", "U"}:
                return None
        if not b_values:
            return None
        return float(sum(j_values)), float(np.mean(b_values))

    def to_dict(self) -> dict[str, Any]:
        return {
            "j_blocks": {key: value.to_dict() for key, value in self.j_blocks.items()},
            "b_blocks": {key: value.to_dict() for key, value in self.b_blocks.items()},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DistanceCalibration":
        return cls(
            {key: BlockCalibration.from_dict(row) for key, row in value["j_blocks"].items()},
            {key: BlockCalibration.from_dict(row) for key, row in value["b_blocks"].items()},
        )


def fit_distance_calibration(states: Sequence[JointState]) -> DistanceCalibration:
    if len(states) < 20:
        raise ValueError("CAL0 distance calibration requires at least 20 states")
    j = np.stack([row.j for row in states])
    jm = np.stack([row.j_mask for row in states])
    b = np.stack([row.b for row in states])
    bm = np.stack([row.b_mask for row in states])
    return DistanceCalibration(
        {
            name: fit_block_calibration(j[:, indices], jm[:, indices])
            for name, indices in J_BLOCKS.items()
        },
        {
            name: fit_block_calibration(b[:, indices], bm[:, indices])
            for name, indices in B_BLOCKS.items()
        },
    )


@dataclass(frozen=True)
class CAL0Values:
    distances: DistanceCalibration
    market_ood_threshold: float
    belief_ood_threshold: float
    tau: float
    support_reference: float
    precision_floor: float
    precision_cap: float
    kernel_mass_reference: float
    runtime_gain_reference: float
    runtime_gain_cap: float
    disagreement_trace_threshold: float
    disagreement_max_eigen_threshold: float
    positive_quality_unique_episode_count: int
    positive_loeo_mass_reference_count: int
    positive_gain_reference_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance_calibration": self.distances.to_dict(),
            **{
                key: value
                for key, value in vars(self).items()
                if key != "distances"
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CAL0Values":
        payload = dict(value)
        distance = DistanceCalibration.from_dict(payload.pop("distance_calibration"))
        return cls(distances=distance, **payload)


@dataclass(frozen=True)
class JointMemoryRecord:
    anchor_id: str
    anchor_date: str
    episode_id: str
    memory_eligible_date: str
    state: JointState
    action: ArrayF
    covariance: ArrayF
    q_anchor: float
    writer_stock_count: int
    gain: float

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float64)
        covariance = _symmetric_psd(np.asarray(self.covariance, dtype=np.float64))
        if (
            action.shape != (7,)
            or not np.isfinite(action).all()
            or not np.isfinite(float(self.q_anchor))
            or float(self.q_anchor) < 0.0
            or int(self.writer_stock_count) < 100
        ):
            raise ValueError("joint Memory record is invalid")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "covariance", covariance)


@dataclass(frozen=True)
class JACFResult:
    apply: bool
    reason_code: str
    target_shift: ArrayF
    mean_action: ArrayF
    total_covariance: ArrayF
    mean_covariance: ArrayF
    neighbor_count: int
    effective_sample_size: float
    coverage: float
    quality: float
    confidence: float
    kernel_mass: float
    maximum_episode_weight: float
    representative_episode_ids: tuple[str, ...]
    weights: ArrayF
    market_distance_minimum: float
    belief_distance_minimum: float


def _reject(reason: str, *, market: float = np.nan, belief: float = np.nan) -> JACFResult:
    return JACFResult(
        False,
        reason,
        np.zeros(7),
        np.zeros(7),
        np.zeros((7, 7)),
        np.zeros((7, 7)),
        0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        (),
        np.empty(0),
        float(market),
        float(belief),
    )


def _eligible_distances(
    query: JointState,
    records: Iterable[JointMemoryRecord],
    calibration: CAL0Values,
) -> list[tuple[JointMemoryRecord, float, float, float]]:
    rows: list[tuple[JointMemoryRecord, float, float, float]] = []
    for record in records:
        if record.q_anchor <= 0.0:
            continue
        distance = calibration.distances.distances(query, record.state)
        if distance is None:
            continue
        market, belief = distance
        joint = 0.60 * market + 0.40 * belief
        rows.append((record, market, belief, joint))
    return rows


def run_joint_cal0(
    query: JointState,
    records: Iterable[JointMemoryRecord],
    calibration: CAL0Values,
    *,
    query_date: str,
) -> JACFResult:
    if calibration.positive_quality_unique_episode_count < 20:
        return _reject("ABSTAIN_CALIBRATION_REFERENCE_SHORTFALL")
    asof = [row for row in records if row.memory_eligible_date < str(query_date)]
    rows = _eligible_distances(query, asof, calibration)
    if not rows:
        return _reject("ABSTAIN_NO_COMPATIBLE_MEMORY")
    market_min = min(row[1] for row in rows)
    market_rows = [row for row in rows if row[1] <= calibration.market_ood_threshold]
    if not market_rows:
        return _reject("ABSTAIN_MARKET_OOD", market=market_min)
    belief_min = min(row[2] for row in market_rows)
    belief_rows = [row for row in market_rows if row[2] <= calibration.belief_ood_threshold]
    if not belief_rows:
        return _reject("ABSTAIN_BELIEF_OOD", market=market_min, belief=belief_min)
    representatives: dict[str, tuple[JointMemoryRecord, float, float, float]] = {}
    for row in belief_rows:
        previous = representatives.get(row[0].episode_id)
        identity = (row[3], row[0].anchor_date, row[0].anchor_id)
        if previous is None or identity < (
            previous[3], previous[0].anchor_date, previous[0].anchor_id
        ):
            representatives[row[0].episode_id] = row
    selected = sorted(representatives.values(), key=lambda row: (row[3], row[0].episode_id))[:5]
    if len(selected) < 2:
        return _reject("ABSTAIN_LOW_NEIGHBOR_SUPPORT", market=market_min, belief=belief_min)
    raw_precision = np.asarray(
        [1.0 / (float(np.trace(row[0].covariance)) + calibration.precision_floor) for row in selected]
    )
    precision = np.minimum(raw_precision, calibration.precision_cap)
    support = np.minimum(
        1.0,
        np.asarray([row[0].writer_stock_count for row in selected], dtype=np.float64)
        / calibration.support_reference,
    )
    quality_anchor = np.asarray([row[0].q_anchor for row in selected], dtype=np.float64)
    kernel = np.exp(-np.asarray([row[3] for row in selected]) / calibration.tau)
    unnormalised = kernel * support * quality_anchor * precision
    mass = float(unnormalised.sum())
    if not np.isfinite(mass) or mass <= 1.0e-12:
        return _reject("ABSTAIN_NO_KERNEL_MASS", market=market_min, belief=belief_min)
    weights = unnormalised / mass
    effective = float(1.0 / np.square(weights).sum())
    if effective < 2.0:
        return _reject("ABSTAIN_LOW_EFFECTIVE_SUPPORT", market=market_min, belief=belief_min)
    actions = np.stack([row[0].action for row in selected])
    mean = np.sum(weights[:, None] * actions, axis=0)
    total = np.zeros((7, 7), dtype=np.float64)
    mean_covariance = np.zeros((7, 7), dtype=np.float64)
    for weight, action, row in zip(weights, actions, selected, strict=True):
        component = np.outer(action - mean, action - mean) + row[0].covariance
        total += weight * component
        mean_covariance += weight * weight * component
    total = (total + total.T) / 2.0
    mean_covariance = (mean_covariance + mean_covariance.T) / 2.0
    trace = float(np.trace(total))
    maximum_eigen = float(np.linalg.eigvalsh(total).max())
    if (
        trace > calibration.disagreement_trace_threshold
        or maximum_eigen > calibration.disagreement_max_eigen_threshold
    ):
        return _reject("ABSTAIN_JOINT_DISAGREEMENT", market=market_min, belief=belief_min)
    coverage = float(np.clip(mass / calibration.kernel_mass_reference, 0.0, 1.0))
    gains = np.asarray([row[0].gain for row in selected], dtype=np.float64)
    raw_quality = float(
        np.sum(weights * np.clip(gains, 0.0, calibration.runtime_gain_cap))
    )
    quality = float(np.clip(raw_quality / calibration.runtime_gain_reference, 0.0, 1.0))
    squared_norm = float(np.square(mean).sum())
    confidence = float(
        squared_norm / (squared_norm + float(np.trace(mean_covariance)) + 1.0e-12)
    )
    target = coverage * quality * confidence * mean
    if not np.isfinite(target).all():
        return _reject("ABSTAIN_NONFINITE_TARGET", market=market_min, belief=belief_min)
    return JACFResult(
        True,
        "APPLY",
        target,
        mean,
        total,
        mean_covariance,
        len(selected),
        effective,
        coverage,
        quality,
        confidence,
        mass,
        float(weights.max()),
        tuple(row[0].episode_id for row in selected),
        weights,
        market_min,
        belief_min,
    )


def fit_joint_cal0(
    records: Sequence[JointMemoryRecord],
) -> tuple[CAL0Values, list[dict[str, Any]]]:
    """Fit the one frozen CAL0 bundle from pre-effect Episode-LOEO records."""

    rows = tuple(records)
    positive = tuple(row for row in rows if row.q_anchor > 0.0)
    episodes = sorted({row.episode_id for row in positive})
    gains = np.asarray([row.gain for row in positive if row.gain > 0.0])
    if len(episodes) < 20 or len(gains) < 20:
        raise ValueError("CAL0 positive quality/gain Episode reference gate failed")
    distances = fit_distance_calibration([row.state for row in positive])
    market: list[float] = []
    belief: list[float] = []
    joint: list[float] = []
    for left_index, left in enumerate(positive):
        for right in positive[left_index + 1 :]:
            if left.episode_id == right.episode_id:
                continue
            value = distances.distances(left.state, right.state)
            if value is None:
                continue
            market.append(value[0])
            belief.append(value[1])
            joint.append(0.60 * value[0] + 0.40 * value[1])
    positive_joint = np.asarray([value for value in joint if value > 0.0])
    if len(positive_joint) < 20 or len(market) < 20 or len(belief) < 20:
        raise ValueError("CAL0 positive LOEO distance reference gate failed")
    traces = np.asarray([float(np.trace(row.covariance)) for row in positive])
    positive_traces = traces[traces > 0.0]
    precision_floor = max(1.0e-8, 0.10 * float(np.median(positive_traces)))
    raw_precision = 1.0 / (traces + precision_floor)
    provisional = CAL0Values(
        distances=distances,
        market_ood_threshold=float(np.quantile(market, 0.95)),
        belief_ood_threshold=float(np.quantile(belief, 0.95)),
        tau=float(np.median(positive_joint)),
        support_reference=float(np.median([row.writer_stock_count for row in positive])),
        precision_floor=precision_floor,
        precision_cap=float(np.quantile(raw_precision, 0.95)),
        kernel_mass_reference=1.0,
        runtime_gain_reference=float(np.median(gains)),
        runtime_gain_cap=float(np.quantile(gains, 0.95)),
        disagreement_trace_threshold=float("inf"),
        disagreement_max_eigen_threshold=float("inf"),
        positive_quality_unique_episode_count=len(episodes),
        positive_loeo_mass_reference_count=0,
        positive_gain_reference_count=len(gains),
    )
    representatives = {
        episode: min(
            (row for row in positive if row.episode_id == episode),
            key=lambda row: (row.anchor_date, row.anchor_id),
        )
        for episode in episodes
    }
    loeo: list[dict[str, Any]] = []
    for episode in episodes:
        query = representatives[episode]
        eligible = [row for row in positive if row.episode_id != episode]
        result = run_joint_cal0(
            query.state, eligible, provisional, query_date="99991231"
        )
        loeo.append(
            {
                "query_episode_id": episode,
                "query_anchor_id": query.anchor_id,
                "apply": result.apply,
                "reason_code": result.reason_code,
                "neighbor_count": result.neighbor_count,
                "kernel_mass": result.kernel_mass,
                "N_eff": result.effective_sample_size,
                "trace_total": float(np.trace(result.total_covariance)),
                "maximum_eigen_total": (
                    float(np.linalg.eigvalsh(result.total_covariance).max())
                    if result.apply
                    else 0.0
                ),
                "maximum_episode_weight": result.maximum_episode_weight,
                "market_distance_minimum": result.market_distance_minimum,
                "belief_distance_minimum": result.belief_distance_minimum,
            }
        )
    valid = [row for row in loeo if row["apply"] and row["kernel_mass"] > 1.0e-12]
    if len(valid) < 20:
        raise ValueError("CAL0 positive LOEO mass reference gate failed")
    final = CAL0Values(
        **{
            key: value
            for key, value in vars(provisional).items()
            if key
            not in {
                "kernel_mass_reference",
                "disagreement_trace_threshold",
                "disagreement_max_eigen_threshold",
                "positive_loeo_mass_reference_count",
            }
        },
        kernel_mass_reference=float(np.median([row["kernel_mass"] for row in valid])),
        disagreement_trace_threshold=float(
            np.quantile([row["trace_total"] for row in valid], 0.95)
        ),
        disagreement_max_eigen_threshold=float(
            np.quantile([row["maximum_eigen_total"] for row in valid], 0.95)
        ),
        positive_loeo_mass_reference_count=len(valid),
    )
    # Re-evaluate the exact frozen bundle so the persisted LOEO table reflects
    # the same disagreement gates used at runtime.
    final_rows: list[dict[str, Any]] = []
    for episode in episodes:
        query = representatives[episode]
        result = run_joint_cal0(
            query.state,
            [row for row in positive if row.episode_id != episode],
            final,
            query_date="99991231",
        )
        final_rows.append(
            {
                "query_episode_id": episode,
                "query_anchor_id": query.anchor_id,
                "apply": result.apply,
                "reason_code": result.reason_code,
                "neighbor_count": result.neighbor_count,
                "kernel_mass": result.kernel_mass,
                "N_eff": result.effective_sample_size,
                "trace_total": float(np.trace(result.total_covariance)),
                "maximum_eigen_total": (
                    float(np.linalg.eigvalsh(result.total_covariance).max())
                    if result.apply
                    else 0.0
                ),
                "maximum_episode_weight": result.maximum_episode_weight,
                "market_distance_minimum": result.market_distance_minimum,
                "belief_distance_minimum": result.belief_distance_minimum,
            }
        )
    return final, final_rows
