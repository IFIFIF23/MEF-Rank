"""Frozen candidate contracts for the MEF-Rank v3.7.1 Writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DOSE_FRACTIONS = (
    0.0, 1.0 / 256.0, 1.0 / 128.0, 1.0 / 64.0, 1.0 / 32.0,
    1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0, 3.0 / 4.0, 1.0,
)
BASE_STR_SEMANTICS_HASH = (
    "326b7831a6f4b3302f77952b00545ce7340532db524d0eccf4ddf275101eb5c5"
)
PRIMARY_UTILITY_VERSION = "K_CONTINUUM_NDCG_RANKIC_V001"
OBJECTIVE_ID = "k_continuum_lambda_rank_v001"
ACTION_GENERATION_VERSION = "handoff_kcontinuum_nested_oob_v001"
ACTION_COORDINATE_VERSION = "joint_reliability_shift_v004"
VALUE_SCHEMA = "handoff_joint_reliability_shift_v001"
CALIBRATION_BUNDLE_ID = "CAL1_HANDOFF_K50_V001"


def memory_namespace(selection_k: int) -> str:
    return f"MEF_RANK_H5_V371_K{selection_k}_DAILY_KCU_DM_R1N_JCAL1_V001"


@dataclass(frozen=True)
class V371WriterConfig:
    selection_k: int
    repeats: int = 20
    minimum_valid_repeats: int = 16
    minimum_stocks: int = 200
    direction_minimum_stocks: int = 100
    magnitude_minimum_stocks: int = 40
    outer_minimum_stocks: int = 50
    outer_development_fraction: float = 0.70
    inner_direction_fraction: float = 0.70
    pair_count: int = 4096
    global_pair_minimum_rank_gap: float = 0.10
    action_l2: float = 0.05
    steps: int = 5
    initial_step_size: float = 0.15
    backtracking_trials: int = 4
    backtracking_factor: float = 0.5
    raw_logit_cap: float = 0.75
    transfer_logit_cap: float = 0.50
    inverse_cap: float = 0.50
    inverse_margin: float = 1.0e-6
    inverse_iterations: int = 24
    inverse_tolerance: float = 1.0e-5
    near_optimal_ratio: float = 0.95
    dose_fractions: tuple[float, ...] = DOSE_FRACTIONS
    full_repeat_cosine_minimum: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.selection_k, bool) or not isinstance(self.selection_k, int):
            raise TypeError("selection_k must be a positive integer")
        if self.selection_k <= 0:
            raise ValueError("selection_k must be a positive integer")
        if (self.repeats, self.minimum_valid_repeats) != (20, 16):
            raise PermissionError("v3.7.1 requires R1N/B20 with 16 valid repeats")
        if (
            self.minimum_stocks, self.direction_minimum_stocks,
            self.magnitude_minimum_stocks, self.outer_minimum_stocks,
            self.outer_development_fraction, self.inner_direction_fraction,
        ) not in ((200, 100, 40, 50, 0.70, 0.70), (15, 7, 3, 5, 0.70, 0.70), (60, 29, 13, 18, 0.70, 0.70)):
            raise PermissionError("v3.7.1 R1N split contract changed")
        if (
            self.pair_count, self.global_pair_minimum_rank_gap, self.action_l2,
            self.steps, self.initial_step_size, self.backtracking_trials,
            self.backtracking_factor, self.raw_logit_cap, self.transfer_logit_cap,
        ) != (4096, 0.10, 0.05, 5, 0.15, 4, 0.5, 0.75, 0.50):
            raise PermissionError("v3.7.1 Direction Writer controls changed")
        if (
            self.inverse_cap, self.inverse_margin, self.inverse_iterations,
            self.inverse_tolerance, self.near_optimal_ratio,
            self.dose_fractions, self.full_repeat_cosine_minimum,
        ) != (0.50, 1.0e-6, 24, 1.0e-5, 0.95, DOSE_FRACTIONS, 0.50):
            raise PermissionError("v3.7.1 dose/inverse contract changed")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dose_fractions"] = list(self.dose_fractions)
        return value


def candidate_projection(config: V371WriterConfig) -> dict[str, Any]:
    return {
        "projection_version": "v010_candidate_v001",
        "base_str_semantics_hash": BASE_STR_SEMANTICS_HASH,
        "mef_system_semantics_hash": None,
        "selection_k_required": True,
        "selection_k": config.selection_k,
        "effective_k_policy": "min_with_valid_stock_count",
        "primary_utility_version": PRIMARY_UTILITY_VERSION,
        "objective_id": OBJECTIVE_ID,
        "anchor_frequency": "daily_every_eligible_mature_H5_feature_date",
        "writer_resampling": "R1N_outer70_30_inner70_30_without_replacement",
        "action_generation_version": ACTION_GENERATION_VERSION,
        "action_coordinate_version": ACTION_COORDINATE_VERSION,
        "value_schema": VALUE_SCHEMA,
        "calibration_bundle_id": CALIBRATION_BUNDLE_ID,
        "memory_namespace": memory_namespace(config.selection_k),
        "writer": config.to_dict(),
        "cal1_data_derived_values": "PENDING_PRE_EFFECT_EPISODE_BALANCED_CALIBRATION",
    }
