"""Final v3.6 Writer isolation, Value-v005 and joint CAL0 implementation."""

from .formal_writer import (
    FINAL_MEMORY_COLUMNS,
    FORMAL_WRITER_SEMANTICS,
    anchor_quality,
    candidate_quality_diagnostics,
    final_gate_reason,
    hard_gate_reason,
    strict_asof_supply,
    validate_anchor_registry,
)
from .jacf_cal0 import (
    CAL0Values,
    JACFResult,
    JointMemoryRecord,
    JointState,
    fit_joint_cal0,
    run_joint_cal0,
)
from .label_isolation import (
    FrozenActionToken,
    FrozenPairRegistry,
    FrozenSplitRegistry,
    LocalLabelView,
    RepeatLabelIsolationSession,
    SplitRawReturnVault,
    evaluate_frozen_action,
    solve_isolated_train_action,
)
from .writer_runtime import (
    WriterConfig,
    WriterRepeat,
    WriterResult,
    solve_joint_writer,
)

__all__ = [
    "CAL0Values",
    "FINAL_MEMORY_COLUMNS",
    "FORMAL_WRITER_SEMANTICS",
    "FrozenActionToken",
    "FrozenPairRegistry",
    "FrozenSplitRegistry",
    "JACFResult",
    "JointMemoryRecord",
    "JointState",
    "LocalLabelView",
    "RepeatLabelIsolationSession",
    "SplitRawReturnVault",
    "WriterConfig",
    "WriterRepeat",
    "WriterResult",
    "anchor_quality",
    "candidate_quality_diagnostics",
    "evaluate_frozen_action",
    "final_gate_reason",
    "fit_joint_cal0",
    "hard_gate_reason",
    "run_joint_cal0",
    "solve_isolated_train_action",
    "solve_joint_writer",
    "strict_asof_supply",
    "validate_anchor_registry",
]
