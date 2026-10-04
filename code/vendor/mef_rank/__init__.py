"""Portable MEF-Rank v3.7.1 candidate package."""

from .identity import *
from .model import MEFRank, MEFRankResult
from .str_model import (
    DayBatch, MEFRankSTR, STRCache, STRProfile, TorchSTRForward,
    base_str_loss, load_str_checkpoint, save_str_checkpoint,
)
from .v36 import (
    CAL0Values, FORMAL_WRITER_SEMANTICS, JointMemoryRecord, JointState,
    RepeatLabelIsolationSession, WriterConfig, WriterResult, fit_joint_cal0,
    run_joint_cal0, solve_joint_writer,
)
from .v371 import (
    CAL1Values, DailyAnchor, MemoryLifecycle, NestedWriterResult,
    V371MemoryRecord, V371WriterConfig, build_cal1_values,
    build_v371_memory_record, daily_mature_anchors, fit_cal1,
    frozen_candidate_projection, run_cal1, solve_nested_writer,
)

__all__ = [name for name in globals() if not name.startswith("_")]
