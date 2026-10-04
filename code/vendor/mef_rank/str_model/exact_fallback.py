"""Exact global-ABSTAIN branch carried into MEF-Rank v3.3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .contracts import STRForward


GLOBAL_ABSTAIN_REASONS = frozenset(
    {
        "ABSTAIN_INSUFFICIENT_NEIGHBORS",
        "ABSTAIN_LOW_EFFECTIVE_SUPPORT",
        "ABSTAIN_MARKET_OOD",
        "ABSTAIN_BELIEF_OOD",
        "ABSTAIN_QUALITY_REFERENCE_UNDEFINED",
        "ABSTAIN_LOW_QUALITY",
        "ABSTAIN_ACTION_DISAGREEMENT",
        "ABSTAIN_TARGET_SHIFT_UNREACHABLE",
        "ABSTAIN_ALL_GROUPS_REJECTED",
    }
)


@dataclass(frozen=True)
class FeedbackResult:
    first: STRForward
    final: STRForward
    delta_beta: npt.NDArray[np.float64]
    global_reason: str
    group_reasons: tuple[str, ...]
    second_forward_executed: bool

    def __post_init__(self) -> None:
        action = np.asarray(self.delta_beta, dtype=np.float64)
        owned = np.array(action, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "delta_beta", owned)


def exact_global_abstain(
    first: STRForward,
    global_reason: str,
    *,
    batch_size: int | None = None,
    group_reasons: tuple[str, ...] = (),
) -> FeedbackResult:
    """Return the first forward object itself and an exactly zero action.

    Reusing the same immutable interface object is stronger than a numerical
    closeness check: no second forward is called and every output field has
    object identity with the first-pass result.
    """

    first.validate()
    if global_reason not in GLOBAL_ABSTAIN_REASONS:
        raise ValueError(f"not a registered global ABSTAIN reason: {global_reason}")
    inferred_batch = first.score.shape[0]
    if batch_size is not None and batch_size != inferred_batch:
        raise ValueError("batch_size does not match first forward")
    delta_beta = np.zeros((inferred_batch, 7), dtype=np.float64)
    return FeedbackResult(
        first=first,
        final=first,
        delta_beta=delta_beta,
        global_reason=global_reason,
        group_reasons=group_reasons,
        second_forward_executed=False,
    )
