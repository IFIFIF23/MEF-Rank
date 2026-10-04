"""MEF-Rank v3.7.1: required-K continuum daily corrective Memory."""

from .cal1 import CAL1Values, build_cal1_values, fit_cal1, run_cal1
from .contracts import (
    ACTION_COORDINATE_VERSION, ACTION_GENERATION_VERSION,
    BASE_STR_SEMANTICS_HASH, CALIBRATION_BUNDLE_ID, OBJECTIVE_ID,
    PRIMARY_UTILITY_VERSION, VALUE_SCHEMA, V371WriterConfig,
    candidate_projection, memory_namespace,
)
from .identity import canonical_hash, frozen_candidate_projection
from .lifecycle import DailyAnchor, MemoryLifecycle, assert_transition, daily_mature_anchors
from .memory import V37MemoryRecord, build_v37_memory_record, runtime_compatible
from .metrics import (
    KContinuumMetrics, LambdaPairRegistry, TopKMetrics, effective_k,
    k_continuum_alpha, k_continuum_pairs, local_k, ranking_movement,
    rank_ic, topk_metrics,
)
from .writer import (
    ActionEvaluation, DirectionResult, NestedWriterResult, R1NSplit,
    RepeatResult, dose_grid, evaluate_shift, maximum_no_clip_dose,
    r1n_split, select_safe_dose, solve_k_direction, solve_nested_writer,
)

V371MemoryRecord = V37MemoryRecord
build_v371_memory_record = build_v37_memory_record

__all__ = [
    "ACTION_COORDINATE_VERSION", "ACTION_GENERATION_VERSION",
    "ActionEvaluation", "BASE_STR_SEMANTICS_HASH", "CAL1Values",
    "CALIBRATION_BUNDLE_ID", "DailyAnchor", "DirectionResult",
    "KContinuumMetrics", "LambdaPairRegistry", "MemoryLifecycle",
    "NestedWriterResult", "OBJECTIVE_ID", "PRIMARY_UTILITY_VERSION",
    "R1NSplit", "RepeatResult", "TopKMetrics", "V371MemoryRecord",
    "V371WriterConfig", "VALUE_SCHEMA", "assert_transition",
    "build_cal1_values", "build_v371_memory_record", "candidate_projection",
    "canonical_hash", "daily_mature_anchors", "dose_grid", "effective_k",
    "evaluate_shift", "fit_cal1", "frozen_candidate_projection",
    "k_continuum_alpha", "k_continuum_pairs", "local_k",
    "maximum_no_clip_dose", "memory_namespace", "r1n_split", "rank_ic",
    "ranking_movement", "run_cal1", "runtime_compatible",
    "select_safe_dose", "solve_k_direction", "solve_nested_writer",
    "topk_metrics",
]
