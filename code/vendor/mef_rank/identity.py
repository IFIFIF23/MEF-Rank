"""MEF-Rank v3.7.1 candidate / machine-v010 identity."""

from __future__ import annotations

from dataclasses import dataclass

HUMAN_CANONICAL = "MEF-Rank_v3.7.1_统一模型设计与实现闭合规格_完整候选版_v001.md"
HUMAN_CANONICAL_SHA256 = "38cc05d7639461056fb15ac8db2c5808b205630fc58b5cc98b4ac4a4a5010825"
BASE_STR_SEMANTICS_HASH = "326b7831a6f4b3302f77952b00545ce7340532db524d0eccf4ddf275101eb5c5"
MEF_SYSTEM_SEMANTICS_HASH = "ec0a1e3e590c3a08074f287cb6c5622d3340e4d82b84d0cded561ef854f99c49"
MODEL_SEMANTICS_PROJECTION_VERSION = "v010_candidate_v001"
CALIBRATION_BUNDLE_ID = "CAL1_JOINT_V371_K50_DAILY_KCU_V001"
ACTION_COORDINATE_VERSION = "joint_reliability_shift_v004"
ACTION_GENERATION_VERSION = "kcontinuum_direction_magnitude_nested_oob_v002"
VALUE_SCHEMA = "joint_reliability_shift_v007"
MEMORY_NAMESPACE = "MEF_RANK_H5_V371_K50_DAILY_KCU_DM_R1N_JCAL1_V001"
OBJECTIVE_ID = "k_continuum_lambda_rank_v001"
SELECTION_K = 50


@dataclass(frozen=True)
class FrozenV371Identity:
    human_canonical: str = HUMAN_CANONICAL
    human_canonical_sha256: str = HUMAN_CANONICAL_SHA256
    base_str_semantics_hash: str = BASE_STR_SEMANTICS_HASH
    mef_system_semantics_hash: str = MEF_SYSTEM_SEMANTICS_HASH
    model_semantics_projection_version: str = MODEL_SEMANTICS_PROJECTION_VERSION
    calibration_bundle_id: str = CALIBRATION_BUNDLE_ID
    action_coordinate_version: str = ACTION_COORDINATE_VERSION
    action_generation_version: str = ACTION_GENERATION_VERSION
    value_schema: str = VALUE_SCHEMA
    memory_namespace: str = MEMORY_NAMESPACE
    objective_id: str = OBJECTIVE_ID
    selection_k: int = SELECTION_K


FROZEN_V371_IDENTITY = FrozenV371Identity()
