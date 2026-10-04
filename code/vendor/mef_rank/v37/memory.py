"""MEF-Rank v3.7 Memory value schema v006.

The v006 record is intentionally richer than the runtime JACF view.  It keeps
the complete Writer construction audit while exposing a lossless adapter to
the unchanged seven-dimensional deployment primitive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..v36.jacf_cal0 import JointMemoryRecord, JointState
from .contracts import (
    ACTION_COORDINATE_VERSION,
    ACTION_GENERATION_VERSION,
    OBJECTIVE_ID,
    VALUE_SCHEMA,
)


ArrayF = NDArray[np.float64]


def _vector(value: Any, size: int, name: str) -> ArrayF:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    result = np.array(result, copy=True)
    result.setflags(write=False)
    return result


def _covariance(value: Any) -> ArrayF:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (7, 7) or not np.isfinite(matrix).all():
        raise ValueError("covariance must be finite with shape (7,7)")
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    repaired = (eigenvectors * np.maximum(eigenvalues, 1.0e-12)) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0
    repaired.setflags(write=False)
    return repaired


@dataclass(frozen=True)
class V37MemoryRecord:
    anchor_id: str
    anchor_date: str
    episode_id: str
    checkpoint_id: str
    memory_eligible_date: str
    state: JointState
    final_reliability_shift: ArrayF
    repeat_action_covariance: ArrayF
    writer_stock_count: int
    valid_repeat_count: int
    calibrated_equivalent_multiplier: float
    full_repeat_mean_cosine: float
    outer_primary_gain_median: float
    outer_rank_ic_delta_median: float
    outer_precision_delta_median: float
    outer_top_k_mean_return_delta_median: float
    q_anchor: float
    qualified: bool
    reason_code: str
    split_registry_hash: str
    raw_full_solve_reliability_shift: ArrayF
    raw_full_solve_transfer_logit: ArrayF
    direction_unit_reliability_shift: ArrayF
    calibrated_dose_fraction_of_source_max: float
    full_max_no_clip_action_l2: float
    joint_final_transfer_logit: ArrayF
    repeat_calibrated_shift_mean: ArrayF
    repeat_consensus_direction: ArrayF
    repeat_projected_magnitude_median: float
    repeat_projected_magnitude_iqr: float
    repeat_negative_projection_count: int
    repeat_selected_dose_fraction_median: float
    repeat_selected_dose_fraction_iqr: float
    repeat_equivalent_multiplier_median: float
    repeat_equivalent_multiplier_iqr: float
    repeat_zero_dose_count: int
    outer_primary_gain_mean: float
    outer_primary_gain_mad: float
    outer_positive_primary_gain_count: int
    outer_positive_primary_gain_fraction: float
    outer_spread_delta_median: float
    full_anchor_ndcg_at_k_delta_audit: float | None
    full_anchor_fractional_precision_at_k_delta_audit: float | None
    full_anchor_top_k_mean_return_delta_audit: float | None
    full_anchor_spread_delta_audit: float | None
    oob_gain_reference: float
    oob_quality_gain: float
    oob_quality_stability: float
    magnitude_stability_score_diagnostic: float
    feature_date: str
    label_maturity_timestamp: str
    memory_eligible_timestamp: str
    checkpoint_identity: Mapping[str, Any]
    input_and_scaler_lineage: Mapping[str, Any]
    lifecycle_dates: Mapping[str, Any]
    selection_k: int = 50
    anchor_frequency: str = "daily"
    value_schema: str = VALUE_SCHEMA
    objective_id: str = OBJECTIVE_ID
    action_generation_version: str = ACTION_GENERATION_VERSION
    action_coordinate_version: str = ACTION_COORDINATE_VERSION

    def __post_init__(self) -> None:
        if self.value_schema != VALUE_SCHEMA:
            raise ValueError("Memory value schema is not v006")
        if self.objective_id != OBJECTIVE_ID or self.selection_k != 50:
            raise ValueError("Memory objective/K is incompatible with v3.7")
        if self.action_generation_version != ACTION_GENERATION_VERSION:
            raise ValueError("Memory action-generation version is incompatible")
        if self.action_coordinate_version != ACTION_COORDINATE_VERSION:
            raise ValueError("Memory action coordinates are incompatible")
        if self.anchor_frequency != "daily":
            raise ValueError("v3.7 Memory anchor frequency must be daily")
        if int(self.writer_stock_count) < 200 or not 0 <= int(
            self.valid_repeat_count
        ) <= 20:
            raise ValueError("Memory Writer support is invalid")
        if not np.isfinite(
            [
                self.calibrated_equivalent_multiplier,
                self.full_repeat_mean_cosine,
                self.outer_primary_gain_median,
                self.outer_rank_ic_delta_median,
                self.outer_precision_delta_median,
                self.outer_top_k_mean_return_delta_median,
                self.q_anchor,
                self.calibrated_dose_fraction_of_source_max,
                self.full_max_no_clip_action_l2,
                self.repeat_projected_magnitude_median,
                self.repeat_projected_magnitude_iqr,
                self.repeat_selected_dose_fraction_median,
                self.repeat_selected_dose_fraction_iqr,
                self.repeat_equivalent_multiplier_median,
                self.repeat_equivalent_multiplier_iqr,
                self.outer_primary_gain_mean,
                self.outer_primary_gain_mad,
                self.outer_positive_primary_gain_fraction,
                self.outer_spread_delta_median,
                self.oob_gain_reference,
                self.oob_quality_gain,
                self.oob_quality_stability,
                self.magnitude_stability_score_diagnostic,
            ]
        ).all():
            raise ValueError("Memory scalar audit fields must be finite")
        if (
            self.q_anchor < 0
            or self.oob_gain_reference <= 0
            or not 0 <= self.oob_quality_gain <= 1
            or not 0 <= self.oob_quality_stability <= 1
            or not 0 <= self.magnitude_stability_score_diagnostic <= 1
            or not 0 <= self.outer_positive_primary_gain_fraction <= 1
            or self.repeat_negative_projection_count < 0
            or self.repeat_zero_dose_count < 0
            or self.outer_positive_primary_gain_count < 0
        ):
            raise ValueError("Memory quality/count audit fields are invalid")
        expected_quality = float(
            np.sqrt(self.oob_quality_gain * self.oob_quality_stability)
        )
        if not np.isclose(self.q_anchor, expected_quality, atol=1.0e-12, rtol=0.0):
            raise ValueError("q_anchor does not equal sqrt(q_gain*q_stab)")
        expected_magnitude_stability = 1.0 / (
            1.0
            + self.repeat_projected_magnitude_iqr
            / (self.repeat_projected_magnitude_median + 1.0e-12)
        )
        if not np.isclose(
            self.magnitude_stability_score_diagnostic,
            expected_magnitude_stability,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("magnitude stability diagnostic is inconsistent")
        object.__setattr__(
            self,
            "final_reliability_shift",
            _vector(self.final_reliability_shift, 7, "final_reliability_shift"),
        )
        object.__setattr__(
            self, "repeat_action_covariance", _covariance(self.repeat_action_covariance)
        )
        for name in (
            "raw_full_solve_reliability_shift",
            "raw_full_solve_transfer_logit",
            "direction_unit_reliability_shift",
            "joint_final_transfer_logit",
            "repeat_calibrated_shift_mean",
            "repeat_consensus_direction",
        ):
            object.__setattr__(self, name, _vector(getattr(self, name), 7, name))

    @property
    def joint_shift_mean(self) -> ArrayF:
        return self.final_reliability_shift

    @property
    def calibrated_action_l2(self) -> float:
        return float(np.linalg.norm(self.final_reliability_shift))

    def to_joint_record(self) -> JointMemoryRecord:
        """Return the unchanged seven-dimensional JACF runtime view."""

        return JointMemoryRecord(
            anchor_id=self.anchor_id,
            anchor_date=self.anchor_date,
            episode_id=self.episode_id,
            memory_eligible_date=self.memory_eligible_date,
            state=self.state,
            action=self.final_reliability_shift,
            covariance=self.repeat_action_covariance,
            q_anchor=float(self.q_anchor if self.qualified else 0.0),
            writer_stock_count=int(self.writer_stock_count),
            gain=float(max(self.outer_primary_gain_median, 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_date": self.anchor_date,
            "episode_id": self.episode_id,
            "checkpoint_id": self.checkpoint_id,
            "memory_eligible_date": self.memory_eligible_date,
            "state_j": self.state.j.tolist(),
            "state_j_mask": self.state.j_mask.tolist(),
            "state_b": self.state.b.tolist(),
            "state_b_mask": self.state.b_mask.tolist(),
            "joint_shift_mean": self.joint_shift_mean.tolist(),
            "raw_full_solve_reliability_shift": (
                self.raw_full_solve_reliability_shift.tolist()
            ),
            "raw_full_solve_transfer_logit": (
                self.raw_full_solve_transfer_logit.tolist()
            ),
            "direction_unit_reliability_shift": (
                self.direction_unit_reliability_shift.tolist()
            ),
            "calibrated_action_l2": self.calibrated_action_l2,
            "calibrated_dose_fraction_of_source_max": float(
                self.calibrated_dose_fraction_of_source_max
            ),
            "full_max_no_clip_action_l2": float(
                self.full_max_no_clip_action_l2
            ),
            "joint_final_transfer_logit": self.joint_final_transfer_logit.tolist(),
            "repeat_calibrated_shift_mean": (
                self.repeat_calibrated_shift_mean.tolist()
            ),
            "repeat_consensus_direction": self.repeat_consensus_direction.tolist(),
            "repeat_projected_magnitude_median": float(
                self.repeat_projected_magnitude_median
            ),
            "repeat_projected_magnitude_iqr": float(
                self.repeat_projected_magnitude_iqr
            ),
            "repeat_negative_projection_count": int(
                self.repeat_negative_projection_count
            ),
            "repeat_selected_dose_fraction_median": float(
                self.repeat_selected_dose_fraction_median
            ),
            "repeat_selected_dose_fraction_iqr": float(
                self.repeat_selected_dose_fraction_iqr
            ),
            "repeat_equivalent_multiplier_median": float(
                self.repeat_equivalent_multiplier_median
            ),
            "repeat_equivalent_multiplier_iqr": float(
                self.repeat_equivalent_multiplier_iqr
            ),
            "repeat_zero_dose_count": int(self.repeat_zero_dose_count),
            "joint_measurement_covariance": self.repeat_action_covariance.tolist(),
            "full_vs_repeat_direction_cosine": float(
                self.full_repeat_mean_cosine
            ),
            "writer_resampling_method": (
                "R1N_outer70_30_inner70_30_without_replacement"
            ),
            "writer_holdout_repeats": 20,
            "writer_direction_fraction_overall": 0.49,
            "writer_magnitude_fraction_overall": 0.21,
            "writer_outer_oob_fraction": 0.30,
            "writer_repeat_label_schema_version": (
                "nested_split_local_average_tie_rank_and_local_k_v001"
            ),
            "writer_stock_count": int(self.writer_stock_count),
            "valid_repeat_count": int(self.valid_repeat_count),
            "calibrated_equivalent_multiplier": float(
                self.calibrated_equivalent_multiplier
            ),
            "full_repeat_mean_cosine": float(self.full_repeat_mean_cosine),
            "outer_primary_gain_median": float(self.outer_primary_gain_median),
            "outer_primary_gain_mean": float(self.outer_primary_gain_mean),
            "outer_primary_gain_mad": float(self.outer_primary_gain_mad),
            "outer_positive_primary_gain_count": int(
                self.outer_positive_primary_gain_count
            ),
            "outer_positive_primary_gain_fraction": float(
                self.outer_positive_primary_gain_fraction
            ),
            "outer_rank_ic_delta_median": float(self.outer_rank_ic_delta_median),
            "outer_precision_delta_median": float(self.outer_precision_delta_median),
            "outer_top_k_mean_return_delta_median": float(
                self.outer_top_k_mean_return_delta_median
            ),
            "outer_spread_delta_median": float(self.outer_spread_delta_median),
            "full_anchor_ndcg_at_k_delta_audit": (
                self.full_anchor_ndcg_at_k_delta_audit
            ),
            "full_anchor_fractional_precision_at_k_delta_audit": (
                self.full_anchor_fractional_precision_at_k_delta_audit
            ),
            "full_anchor_top_k_mean_return_delta_audit": (
                self.full_anchor_top_k_mean_return_delta_audit
            ),
            "full_anchor_spread_delta_audit": self.full_anchor_spread_delta_audit,
            "oob_gain_reference": float(self.oob_gain_reference),
            "oob_quality_gain": float(self.oob_quality_gain),
            "oob_quality_stability": float(self.oob_quality_stability),
            "magnitude_stability_score_diagnostic": float(
                self.magnitude_stability_score_diagnostic
            ),
            "q_anchor": float(self.q_anchor),
            "qualified": bool(self.qualified),
            "reason_code": self.reason_code,
            "split_registry_hash": self.split_registry_hash,
            "anchor_frequency": self.anchor_frequency,
            "feature_date": self.feature_date,
            "label_maturity_timestamp": self.label_maturity_timestamp,
            "memory_eligible_timestamp": self.memory_eligible_timestamp,
            "checkpoint_identity": dict(self.checkpoint_identity),
            "input_and_scaler_lineage": dict(self.input_and_scaler_lineage),
            "lifecycle_dates": dict(self.lifecycle_dates),
            "selection_k": int(self.selection_k),
            "value_schema": self.value_schema,
            "objective_id": self.objective_id,
            "action_generation_version": self.action_generation_version,
            "action_coordinate_version": self.action_coordinate_version,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "V37MemoryRecord":
        row = dict(value)
        row["final_reliability_shift"] = row.pop("joint_shift_mean")
        row["repeat_action_covariance"] = row.pop(
            "joint_measurement_covariance"
        )
        row["full_repeat_mean_cosine"] = row.pop(
            "full_vs_repeat_direction_cosine"
        )
        for name in (
            "calibrated_action_l2",
            "writer_resampling_method",
            "writer_holdout_repeats",
            "writer_direction_fraction_overall",
            "writer_magnitude_fraction_overall",
            "writer_outer_oob_fraction",
            "writer_repeat_label_schema_version",
        ):
            row.pop(name, None)
        state = JointState(
            np.asarray(row.pop("state_j"), dtype=np.float64),
            np.asarray(row.pop("state_j_mask"), dtype=bool),
            np.asarray(row.pop("state_b"), dtype=np.float64),
            np.asarray(row.pop("state_b_mask"), dtype=bool),
        )
        return cls(state=state, **row)


def runtime_compatible(records: list[V37MemoryRecord]) -> list[JointMemoryRecord]:
    """Filter fail-closed and expose only qualified v3.7 records."""

    return [record.to_joint_record() for record in records if record.qualified]


def build_v37_memory_record(
    *,
    writer: Any,
    anchor_id: str,
    feature_date: str,
    episode_id: str,
    checkpoint_id: str,
    label_maturity_timestamp: str,
    memory_eligible_timestamp: str,
    state: JointState,
    oob_gain_reference: float,
    checkpoint_identity: Mapping[str, Any],
    input_and_scaler_lineage: Mapping[str, Any],
    lifecycle_dates: Mapping[str, Any],
    full_anchor_audit: Mapping[str, float | None] | None = None,
) -> V37MemoryRecord:
    """Materialise the complete v006 value from a frozen Writer result."""

    if not np.isfinite(oob_gain_reference) or oob_gain_reference <= 0:
        raise ValueError("v3.7 oob_gain_reference must be positive and frozen")
    primary_median = float(writer.outer_summary["primary_gain_median"])
    q_gain = float(np.clip(primary_median / oob_gain_reference, 0.0, 1.0))
    positive_count = int(writer.outer_summary["primary_gain_positive_count"])
    valid_count = int(writer.valid_repeat_count)
    p_hat = (positive_count + 1.0) / (valid_count + 2.0)
    q_stab = float(np.clip(2.0 * (p_hat - 0.5), 0.0, 1.0))
    q_anchor = float(np.sqrt(q_gain * q_stab))
    magnitude_stability = float(
        1.0
        / (
            1.0
            + writer.repeat_projected_magnitude_iqr
            / (writer.repeat_projected_magnitude_median + 1.0e-12)
        )
    )
    split_payload = [
        {
            "repeat": int(row.repeat),
            "valid": bool(row.valid),
            "split_hash": row.split_hash,
            "pair_registry_hash": row.pair_registry_hash,
        }
        for row in writer.repeats
    ]
    split_hash = hashlib.sha256(
        json.dumps(
            split_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    audit = dict(full_anchor_audit or {})
    eligible_date = str(memory_eligible_timestamp)[:10].replace("-", "")
    return V37MemoryRecord(
        anchor_id=str(anchor_id),
        anchor_date=str(feature_date).replace("-", ""),
        episode_id=str(episode_id),
        checkpoint_id=str(checkpoint_id),
        memory_eligible_date=eligible_date,
        state=state,
        final_reliability_shift=writer.final_action,
        repeat_action_covariance=writer.covariance,
        writer_stock_count=int(writer.writer_stock_count),
        valid_repeat_count=valid_count,
        calibrated_equivalent_multiplier=float(
            writer.calibrated_equivalent_multiplier
        ),
        full_repeat_mean_cosine=float(writer.full_repeat_cosine),
        outer_primary_gain_median=primary_median,
        outer_rank_ic_delta_median=float(
            writer.outer_summary["rank_ic_delta_median"]
        ),
        outer_precision_delta_median=float(
            writer.outer_summary["precision_delta_median"]
        ),
        outer_top_k_mean_return_delta_median=float(
            writer.outer_summary["top_k_mean_return_delta_median"]
        ),
        q_anchor=q_anchor,
        qualified=bool(writer.qualified),
        reason_code=str(writer.reason_code),
        split_registry_hash=split_hash,
        raw_full_solve_reliability_shift=writer.full_raw_reliability_shift,
        raw_full_solve_transfer_logit=writer.full_raw_transfer_logit,
        direction_unit_reliability_shift=writer.full_direction_unit,
        calibrated_dose_fraction_of_source_max=float(
            writer.final_magnitude
            / (writer.full_maximum_no_clip_dose + 1.0e-12)
        ),
        full_max_no_clip_action_l2=float(writer.full_maximum_no_clip_dose),
        joint_final_transfer_logit=writer.final_transfer_logit,
        repeat_calibrated_shift_mean=writer.repeat_calibrated_shift_mean,
        repeat_consensus_direction=writer.repeat_consensus_direction,
        repeat_projected_magnitude_median=float(
            writer.repeat_projected_magnitude_median
        ),
        repeat_projected_magnitude_iqr=float(
            writer.repeat_projected_magnitude_iqr
        ),
        repeat_negative_projection_count=int(
            writer.repeat_negative_projection_count
        ),
        repeat_selected_dose_fraction_median=float(
            writer.repeat_selected_dose_fraction_median
        ),
        repeat_selected_dose_fraction_iqr=float(
            writer.repeat_selected_dose_fraction_iqr
        ),
        repeat_equivalent_multiplier_median=float(
            writer.repeat_equivalent_multiplier_median
        ),
        repeat_equivalent_multiplier_iqr=float(
            writer.repeat_equivalent_multiplier_iqr
        ),
        repeat_zero_dose_count=int(writer.repeat_zero_dose_count),
        outer_primary_gain_mean=float(
            writer.outer_summary["primary_gain_mean"]
        ),
        outer_primary_gain_mad=float(
            writer.outer_summary["primary_gain_mad"]
        ),
        outer_positive_primary_gain_count=positive_count,
        outer_positive_primary_gain_fraction=float(
            writer.outer_summary["primary_gain_positive_fraction"]
        ),
        outer_spread_delta_median=float(
            writer.outer_summary["spread_delta_median"]
        ),
        full_anchor_ndcg_at_k_delta_audit=audit.get("ndcg_at_k_delta"),
        full_anchor_fractional_precision_at_k_delta_audit=audit.get(
            "fractional_precision_at_k_delta"
        ),
        full_anchor_top_k_mean_return_delta_audit=audit.get(
            "top_k_mean_return_delta"
        ),
        full_anchor_spread_delta_audit=audit.get("spread_delta"),
        oob_gain_reference=float(oob_gain_reference),
        oob_quality_gain=q_gain,
        oob_quality_stability=q_stab,
        magnitude_stability_score_diagnostic=magnitude_stability,
        feature_date=str(feature_date),
        label_maturity_timestamp=str(label_maturity_timestamp),
        memory_eligible_timestamp=str(memory_eligible_timestamp),
        checkpoint_identity=dict(checkpoint_identity),
        input_and_scaler_lineage=dict(input_and_scaler_lineage),
        lifecycle_dates=dict(lifecycle_dates),
    )
