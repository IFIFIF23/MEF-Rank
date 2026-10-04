"""CAL1 contract and fail-closed runtime adapter for v3.7.

CAL1 is fitted only after a v006 Memory bank exists.  It may reuse the stable
J+B/JACF numeric primitives, but it cannot load a v3.6 CAL0 artifact: the
bundle identity, K/objective compatibility and episode-balanced provenance
are mandatory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..v36.jacf_cal0 import (
    B_BLOCKS,
    J_BLOCKS,
    BlockCalibration,
    CAL0Values,
    DistanceCalibration,
    JACFResult,
    JointState,
    run_joint_cal0,
)
from .contracts import CALIBRATION_BUNDLE_ID, OBJECTIVE_ID, VALUE_SCHEMA
from .memory import V37MemoryRecord


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0)
    data = data[valid]
    mass = mass[valid]
    if not len(data) or not 0 <= quantile <= 1:
        raise ValueError("weighted quantile reference is invalid")
    order = np.argsort(data, kind="mergesort")
    data = data[order]
    mass = mass[order]
    centers = (np.cumsum(mass) - 0.5 * mass) / mass.sum()
    return float(
        np.interp(quantile, centers, data, left=data[0], right=data[-1])
    )


def _episode_balanced_weights(records: Sequence[V37MemoryRecord]) -> np.ndarray:
    episodes: dict[str, int] = {}
    for record in records:
        episodes[record.episode_id] = episodes.get(record.episode_id, 0) + 1
    if not episodes:
        raise ValueError("CAL1 has no records")
    episode_mass = 1.0 / len(episodes)
    return np.asarray(
        [episode_mass / episodes[record.episode_id] for record in records],
        dtype=np.float64,
    )


def _fit_weighted_block(
    values: np.ndarray, masks: np.ndarray, weights: np.ndarray
) -> BlockCalibration:
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(masks, dtype=bool)
    mass = np.asarray(weights, dtype=np.float64)
    if (
        data.ndim != 2
        or valid.shape != data.shape
        or mass.shape != (len(data),)
        or len(data) < 2
    ):
        raise ValueError("CAL1 block calibration sample is invalid")
    median = np.zeros(data.shape[1], dtype=np.float64)
    mad = np.ones(data.shape[1], dtype=np.float64)
    filled = np.zeros_like(data)
    for column in range(data.shape[1]):
        observed = valid[:, column]
        if observed.any():
            median[column] = _weighted_quantile(
                data[observed, column], mass[observed], 0.5
            )
            mad[column] = max(
                1.0e-8,
                _weighted_quantile(
                    np.abs(data[observed, column] - median[column]),
                    mass[observed],
                    0.5,
                ),
            )
        filled[:, column] = np.where(
            observed, (data[:, column] - median[column]) / mad[column], 0.0
        )
    normalised = mass / mass.sum()
    mean = np.sum(normalised[:, None] * filled, axis=0)
    centered = filled - mean
    covariance = (centered * normalised[:, None]).T @ centered
    target = np.diag(np.diag(covariance))
    effective = 1.0 / float(np.square(normalised).sum())
    shrinkage = min(1.0, data.shape[1] / max(effective, 1.0))
    covariance = (1.0 - shrinkage) * covariance + shrinkage * target
    return BlockCalibration(median, mad, covariance, 0.50)


def _fit_weighted_distances(
    records: Sequence[V37MemoryRecord],
) -> DistanceCalibration:
    states = [row.state for row in records]
    weights = _episode_balanced_weights(records)
    j = np.stack([row.j for row in states])
    jm = np.stack([row.j_mask for row in states])
    b = np.stack([row.b for row in states])
    bm = np.stack([row.b_mask for row in states])
    return DistanceCalibration(
        {
            name: _fit_weighted_block(j[:, indices], jm[:, indices], weights)
            for name, indices in J_BLOCKS.items()
        },
        {
            name: _fit_weighted_block(b[:, indices], bm[:, indices], weights)
            for name, indices in B_BLOCKS.items()
        },
    )


def _memory_hash(records: Sequence[V37MemoryRecord]) -> str:
    payload = json.dumps(
        [row.to_dict() for row in records],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CAL1Values:
    numeric: CAL0Values
    calibration_bundle_id: str
    value_schema: str
    objective_id: str
    selection_k: int
    episode_balanced: bool
    source_memory_hash: str
    source_unique_episode_count: int

    def __post_init__(self) -> None:
        if self.calibration_bundle_id != CALIBRATION_BUNDLE_ID:
            raise ValueError("CAL1 bundle ID mismatch")
        if (
            self.value_schema != VALUE_SCHEMA
            or self.objective_id != OBJECTIVE_ID
            or self.selection_k != 50
        ):
            raise ValueError("CAL1 Memory objective/K compatibility mismatch")
        if not self.episode_balanced:
            raise ValueError("CAL1 must be episode-balanced")
        if self.source_unique_episode_count < 20:
            raise ValueError("CAL1 requires at least 20 source Episodes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_bundle_id": self.calibration_bundle_id,
            "value_schema": self.value_schema,
            "objective_id": self.objective_id,
            "selection_k": self.selection_k,
            "episode_balanced": self.episode_balanced,
            "source_memory_hash": self.source_memory_hash,
            "source_unique_episode_count": self.source_unique_episode_count,
            "numeric": self.numeric.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CAL1Values":
        payload = dict(value)
        numeric = CAL0Values.from_dict(payload.pop("numeric"))
        return cls(numeric=numeric, **payload)


def build_cal1_values(
    numeric: CAL0Values,
    *,
    records: Sequence[V37MemoryRecord],
    source_memory_hash: str,
    episode_balanced: bool,
) -> CAL1Values:
    """Seal newly fitted numeric values as CAL1.

    The fitting pipeline must provide episode-balanced pre-effect/LOEO
    provenance.  This constructor deliberately refuses a bare CAL0 JSON.
    """

    episodes = {row.episode_id for row in records if row.qualified}
    return CAL1Values(
        numeric=numeric,
        calibration_bundle_id=CALIBRATION_BUNDLE_ID,
        value_schema=VALUE_SCHEMA,
        objective_id=OBJECTIVE_ID,
        selection_k=50,
        episode_balanced=bool(episode_balanced),
        source_memory_hash=str(source_memory_hash),
        source_unique_episode_count=len(episodes),
    )


def fit_cal1(
    records: Sequence[V37MemoryRecord],
) -> tuple[CAL1Values, list[dict[str, Any]]]:
    """Fit the frozen Episode-balanced CAL1 bundle from pre-effect v006 records.

    Every Episode receives equal total mass.  During LOEO, all daily queries of
    a held-out Episode are evaluated, their weights sum to one Episode, and
    every candidate Episode contributes at most its nearest representative.
    """

    rows = tuple(
        row
        for row in records
        if row.qualified and row.q_anchor > 0 and row.outer_primary_gain_median > 0
    )
    by_episode: dict[str, list[V37MemoryRecord]] = {}
    for row in rows:
        by_episode.setdefault(row.episode_id, []).append(row)
    episodes = sorted(by_episode)
    if len(episodes) < 20:
        raise ValueError("CAL1 positive quality Episode reference gate failed")
    distances = _fit_weighted_distances(rows)

    market_reference: list[float] = []
    belief_reference: list[float] = []
    joint_reference: list[float] = []
    distance_weights: list[float] = []
    episode_mass = 1.0 / len(episodes)
    for held_episode in episodes:
        queries = by_episode[held_episode]
        query_mass = episode_mass / len(queries)
        candidate_episodes = [value for value in episodes if value != held_episode]
        candidate_mass = query_mass / len(candidate_episodes)
        for query in queries:
            for candidate_episode in candidate_episodes:
                choices: list[tuple[float, float, float]] = []
                for candidate in by_episode[candidate_episode]:
                    value = distances.distances(query.state, candidate.state)
                    if value is None:
                        continue
                    market, belief = float(value[0]), float(value[1])
                    choices.append(
                        (0.60 * market + 0.40 * belief, market, belief)
                    )
                if not choices:
                    continue
                joint, market, belief = min(choices)
                market_reference.append(market)
                belief_reference.append(belief)
                joint_reference.append(joint)
                distance_weights.append(candidate_mass)
    positive_joint = np.asarray(joint_reference) > 0
    if (
        len(market_reference) < 20
        or len(belief_reference) < 20
        or int(positive_joint.sum()) < 20
    ):
        raise ValueError("CAL1 Episode-balanced LOEO distance reference gate failed")

    record_weights = _episode_balanced_weights(rows)
    traces = np.asarray(
        [float(np.trace(row.repeat_action_covariance)) for row in rows]
    )
    positive_trace = traces > 0
    trace_median = (
        _weighted_quantile(
            traces[positive_trace], record_weights[positive_trace], 0.5
        )
        if positive_trace.any()
        else 0.0
    )
    precision_floor = max(1.0e-8, 0.10 * trace_median)
    raw_precision = 1.0 / (traces + precision_floor)
    gains = np.asarray([row.outer_primary_gain_median for row in rows])
    provisional = CAL0Values(
        distances=distances,
        market_ood_threshold=_weighted_quantile(
            market_reference, distance_weights, 0.95
        ),
        belief_ood_threshold=_weighted_quantile(
            belief_reference, distance_weights, 0.95
        ),
        tau=_weighted_quantile(
            np.asarray(joint_reference)[positive_joint],
            np.asarray(distance_weights)[positive_joint],
            0.5,
        ),
        support_reference=_weighted_quantile(
            [row.writer_stock_count for row in rows], record_weights, 0.5
        ),
        precision_floor=precision_floor,
        precision_cap=_weighted_quantile(raw_precision, record_weights, 0.95),
        kernel_mass_reference=1.0,
        runtime_gain_reference=_weighted_quantile(gains, record_weights, 0.5),
        runtime_gain_cap=_weighted_quantile(gains, record_weights, 0.95),
        disagreement_trace_threshold=float("inf"),
        disagreement_max_eigen_threshold=float("inf"),
        positive_quality_unique_episode_count=len(episodes),
        positive_loeo_mass_reference_count=0,
        positive_gain_reference_count=len(rows),
    )
    runtime_rows = [row.to_joint_record() for row in rows]

    def run_loeo(calibration: CAL0Values) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for held_episode in episodes:
            queries = by_episode[held_episode]
            query_weight = episode_mass / len(queries)
            candidates = [
                row
                for row in runtime_rows
                if row.episode_id != held_episode
            ]
            for query in queries:
                result = run_joint_cal0(
                    query.state,
                    candidates,
                    calibration,
                    query_date="99991231",
                )
                output.append(
                    {
                        "held_out_episode_id": held_episode,
                        "query_anchor_id": query.anchor_id,
                        "query_anchor_date": query.anchor_date,
                        "episode_balanced_query_weight": query_weight,
                        "held_out_episode_daily_query_count": len(queries),
                        "apply": result.apply,
                        "reason_code": result.reason_code,
                        "neighbor_count": result.neighbor_count,
                        "representative_episode_ids": ";".join(
                            result.representative_episode_ids
                        ),
                        "kernel_mass": result.kernel_mass,
                        "N_eff": result.effective_sample_size,
                        "trace_total": float(np.trace(result.total_covariance)),
                        "maximum_eigen_total": (
                            float(
                                np.linalg.eigvalsh(
                                    result.total_covariance
                                ).max()
                            )
                            if result.apply
                            else 0.0
                        ),
                        "maximum_episode_weight": (
                            result.maximum_episode_weight
                        ),
                        "market_distance_minimum": (
                            result.market_distance_minimum
                        ),
                        "belief_distance_minimum": (
                            result.belief_distance_minimum
                        ),
                    }
                )
        return output

    provisional_rows = run_loeo(provisional)
    valid = [
        row for row in provisional_rows if row["apply"] and row["kernel_mass"] > 1.0e-12
    ]
    valid_episodes = {row["held_out_episode_id"] for row in valid}
    if len(valid_episodes) < 20:
        raise ValueError("CAL1 positive LOEO Episode mass reference gate failed")
    valid_weights = [row["episode_balanced_query_weight"] for row in valid]
    final_numeric = CAL0Values(
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
        kernel_mass_reference=_weighted_quantile(
            [row["kernel_mass"] for row in valid], valid_weights, 0.5
        ),
        disagreement_trace_threshold=_weighted_quantile(
            [row["trace_total"] for row in valid], valid_weights, 0.95
        ),
        disagreement_max_eigen_threshold=_weighted_quantile(
            [row["maximum_eigen_total"] for row in valid],
            valid_weights,
            0.95,
        ),
        positive_loeo_mass_reference_count=len(valid),
    )
    final_rows = run_loeo(final_numeric)
    bundle = build_cal1_values(
        final_numeric,
        records=rows,
        source_memory_hash=_memory_hash(rows),
        episode_balanced=True,
    )
    return bundle, final_rows


def run_cal1(
    query: JointState,
    records: Sequence[V37MemoryRecord],
    calibration: CAL1Values,
    *,
    query_date: str,
) -> JACFResult:
    compatible = [
        row.to_joint_record()
        for row in records
        if row.qualified
        and row.value_schema == calibration.value_schema
        and row.objective_id == calibration.objective_id
        and row.selection_k == calibration.selection_k
    ]
    return run_joint_cal0(
        query, compatible, calibration.numeric, query_date=str(query_date)
    )
