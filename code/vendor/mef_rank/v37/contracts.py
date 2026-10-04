"""Frozen candidate contracts for the MEF-Rank v3.7 offline Writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


DOSE_FRACTIONS = (
    0.0,
    1.0 / 256.0,
    1.0 / 128.0,
    1.0 / 64.0,
    1.0 / 32.0,
    1.0 / 16.0,
    1.0 / 8.0,
    1.0 / 4.0,
    1.0 / 2.0,
    3.0 / 4.0,
    1.0,
)

BASE_STR_SEMANTICS_HASH = (
    "326b7831a6f4b3302f77952b00545ce7340532db524d0eccf4ddf275101eb5c5"
)
OBJECTIVE_ID = "base_position_lambda_ndcg_at_k_v001"
ACTION_GENERATION_VERSION = "kaware_direction_magnitude_nested_oob_v001"
ACTION_COORDINATE_VERSION = "joint_reliability_shift_v004"
VALUE_SCHEMA = "joint_reliability_shift_v006"
CALIBRATION_BUNDLE_ID = "CAL1_JOINT_V37_K50_DAILY_V001"
MEMORY_NAMESPACE = "MEF_RANK_H5_V37_K50_DAILY_DM_R1N_JCAL1_V001"


@dataclass(frozen=True)
class V37WriterConfig:
    selection_k: int = 50
    repeats: int = 20
    minimum_valid_repeats: int = 16
    minimum_stocks: int = 200
    direction_minimum_stocks: int = 100
    magnitude_minimum_stocks: int = 40
    outer_minimum_stocks: int = 50
    outer_development_fraction: float = 0.70
    inner_direction_fraction: float = 0.70
    pair_count: int = 4096
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
        if self.selection_k <= 0:
            raise ValueError("v3.7 K mode requires a positive selection_k")
        if (self.repeats, self.minimum_valid_repeats) != (20, 16):
            raise PermissionError("v3.7 requires R1N/B20 with 16 valid repeats")
        if (
            self.minimum_stocks != 200
            or self.direction_minimum_stocks != 100
            or self.magnitude_minimum_stocks != 40
            or self.outer_minimum_stocks != 50
            or self.outer_development_fraction != 0.70
            or self.inner_direction_fraction != 0.70
        ):
            raise PermissionError("v3.7 R1N split contract changed")
        if (
            self.pair_count != 4096
            or self.action_l2 != 0.05
            or self.steps != 5
            or self.initial_step_size != 0.15
            or self.backtracking_trials != 4
            or self.backtracking_factor != 0.5
            or self.raw_logit_cap != 0.75
            or self.transfer_logit_cap != 0.50
        ):
            raise PermissionError("v3.7 Direction Writer controls changed")
        if (
            self.inverse_cap != 0.50
            or self.inverse_margin != 1.0e-6
            or self.inverse_iterations != 24
            or self.inverse_tolerance != 1.0e-5
            or self.near_optimal_ratio != 0.95
            or self.dose_fractions != DOSE_FRACTIONS
            or self.full_repeat_cosine_minimum != 0.50
        ):
            raise PermissionError("v3.7 dose/inverse contract changed")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["dose_fractions"] = list(self.dose_fractions)
        return value


def candidate_projection(config: V37WriterConfig = V37WriterConfig()) -> dict[str, Any]:
    """Return the machine-readable candidate projection without fabricating CAL1."""

    return {
        "projection_version": "v009_candidate_v003",
        "base_str_semantics_hash": BASE_STR_SEMANTICS_HASH,
        "mef_system_semantics_hash": None,
        "selection_k": config.selection_k,
        "objective_id": OBJECTIVE_ID,
        "anchor_frequency": "daily_every_eligible_mature_H5_feature_date",
        "writer_resampling": "R1N_outer70_30_inner70_30_without_replacement",
        "action_generation_version": ACTION_GENERATION_VERSION,
        "action_coordinate_version": ACTION_COORDINATE_VERSION,
        "value_schema": VALUE_SCHEMA,
        "calibration_bundle_id": CALIBRATION_BUNDLE_ID,
        "memory_namespace": MEMORY_NAMESPACE,
        "writer": config.to_dict(),
        "cal1_data_derived_values": "PENDING_PRE_EFFECT_EPISODE_BALANCED_CALIBRATION",
    }
