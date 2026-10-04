"""Pure contracts and aggregation helpers for the v3.6 formal Writer run."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .rebaseline import quality_diagnostics, quality_values


FORMAL_WRITER_SEMANTICS = {
    "repeat_label_semantics": "repeat_local_average_tie_rank_v001",
    "resampling_method": "R1_70_30_without_replacement",
    "repeats_B": 20,
    "joint_global_alpha": 1.0,
    "oob_gain_reference": 1.1389308241681104e-05,
    "minimum_valid_repeats": 16,
    "full_repeat_cosine_minimum": 0.50,
    "full_cross_section_feature_forward": True,
    "feature_context_split": False,
    "supervision_stock_set_split": True,
    "supervision_rank_scope_split_local": True,
    "repeat_global_primary_rank_forbidden": True,
}

FINAL_MEMORY_COLUMNS = (
    "anchor_index",
    "anchor_id",
    "anchor_date",
    "episode_id",
    "checkpoint_id",
    "memory_eligible_date",
    "input_identity_hash",
    "checkpoint_sha256",
    "joint_full_solve_transfer_logit",
    "joint_final_reliability_shift",
    "bootstrap_reliability_shift_mean",
    "joint_measurement_covariance",
    "full_vs_repeat_mean_cosine",
    "joint_global_alpha",
    "writer_resampling_method",
    "writer_train_fraction",
    "writer_oob_fraction",
    "writer_holdout_repeats",
    "writer_repeat_label_schema_version",
    "oob_relative_gain_median",
    "oob_relative_gain_mean",
    "oob_relative_gain_mad",
    "oob_positive_gain_count",
    "oob_positive_gain_fraction",
    "oob_valid_repeat_count",
    "oob_gain_reference",
    "oob_quality_gain",
    "oob_quality_stability",
    "joint_anchor_oob_quality",
    "writer_stock_count",
    "value_schema",
    "action_coordinate_version",
    "memory_namespace",
    "mef_system_semantics_hash",
)


def validate_anchor_registry(frame: pd.DataFrame) -> None:
    required = {
        "anchor_index",
        "anchor_id",
        "anchor_date",
        "episode_id",
        "checkpoint_id",
        "checkpoint_refresh_date",
        "label_maturity_date",
        "writer_scheduled_date",
        "memory_eligible_date",
        "input_identity_hash",
        "writer_stock_count_expected",
        "status",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"formal anchor registry fields missing: {sorted(missing)}")
    if len(frame) != 301 or frame["anchor_id"].nunique() != 301:
        raise ValueError("formal Writer authority requires exactly 301 fixed anchors")
    if frame["anchor_index"].tolist() != list(range(1, 302)):
        raise ValueError("formal anchor registry index is not stable 1..301")
    if not frame["anchor_date"].astype(str).is_monotonic_increasing:
        raise ValueError("formal anchor registry is not ordered by anchor date")
    if frame["anchor_date"].astype(str).max() > "20230331":
        raise PermissionError("formal anchor registry escapes the freeze boundary")
    if not (
        frame["label_maturity_date"].astype(str)
        < frame["writer_scheduled_date"].astype(str)
    ).all():
        raise ValueError("Writer scheduling does not follow label maturity")
    if not (
        frame["writer_scheduled_date"].astype(str)
        == frame["memory_eligible_date"].astype(str)
    ).all():
        raise ValueError("Memory eligibility lifecycle changed")


def hard_gate_reason(
    *,
    stock_count: int,
    valid_repeats: int,
    gains: np.ndarray,
    actions: np.ndarray,
) -> str | None:
    if stock_count < 100:
        return "WRITER_INSUFFICIENT_STOCKS"
    if valid_repeats < 16:
        return "WRITER_INSUFFICIENT_VALID_REPEATS"
    if not np.isfinite(gains).all() or not np.isfinite(actions).all():
        return "WRITER_NONFINITE_REPEAT"
    if float(np.median(gains)) <= 0.0:
        return "WRITER_NONPOSITIVE_MEDIAN_OOB_GAIN"
    return None


def final_gate_reason(
    *,
    full_action: np.ndarray,
    cosine: float,
    covariance: np.ndarray,
) -> str | None:
    if full_action.shape != (7,) or not np.isfinite(full_action).all():
        return "WRITER_FULL_SOLVE_NONFINITE"
    if not np.isfinite(cosine) or cosine < 0.50:
        return "WRITER_FULL_REPEAT_COSINE_FAILURE"
    if covariance.shape != (7, 7) or not np.isfinite(covariance).all():
        return "WRITER_COVARIANCE_NONFINITE"
    if float(np.linalg.eigvalsh(covariance).min()) < -1.0e-12:
        return "WRITER_COVARIANCE_NOT_PSD"
    return None


def anchor_quality(gains: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(tuple(gains), dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("formal Writer gains are invalid")
    positive = int(np.count_nonzero(values > 0.0))
    result: dict[str, float | int] = {
        "oob_relative_gain_median": float(np.median(values)),
        "oob_relative_gain_mean": float(np.mean(values)),
        "oob_relative_gain_mad": float(
            np.median(np.abs(values - np.median(values)))
        ),
        "oob_positive_gain_count": positive,
        "oob_positive_gain_fraction": float(positive / len(values)),
        "oob_valid_repeat_count": int(len(values)),
    }
    result.update(
        quality_values(
            gain_median=float(result["oob_relative_gain_median"]),
            positive_count=positive,
            valid_count=len(values),
            gain_reference=FORMAL_WRITER_SEMANTICS["oob_gain_reference"],
        )
    )
    return result


def strict_asof_supply(
    candidates: pd.DataFrame,
    query_dates: Iterable[str],
) -> pd.DataFrame:
    eligible = candidates[["episode_id", "memory_eligible_date"]].copy()
    eligible["memory_eligible_date"] = eligible["memory_eligible_date"].astype(str)
    rows: list[dict[str, Any]] = []
    for query in map(str, query_dates):
        available = eligible.loc[eligible["memory_eligible_date"] < query]
        rows.append(
            {
                "query_date": query,
                "eligible_anchor_count": int(len(available)),
                "eligible_unique_episode_count": int(
                    available["episode_id"].nunique()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["REF0_reached"] = result["eligible_unique_episode_count"].ge(20)
    return result


def candidate_quality_diagnostics(candidates: pd.DataFrame) -> dict[str, float | int]:
    return quality_diagnostics(
        candidates["joint_anchor_oob_quality"].astype(float).to_numpy()
    )
