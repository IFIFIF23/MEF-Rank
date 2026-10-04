"""Explicit observable output contract for an STR forward pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class STRForward:
    """Only fields that the v3.3 contract allows MBS to observe.

    The class deliberately carries no arbitrary hidden state.  It is a minimal
    NumPy-facing engineering contract; the future torch implementation must
    expose the same named fields without widening the MBS boundary.
    """

    score: FloatArray
    rank_logits: FloatArray
    rank_probability: FloatArray
    uncertainty: FloatArray
    reliability: FloatArray
    weighted_group_pooled: FloatArray

    def __post_init__(self) -> None:
        # ``frozen=True`` protects attributes, not NumPy buffers.  Copy every
        # observable field and make the owned buffer read-only so exact fallback
        # can safely reuse this object without later in-place mutation corrupting
        # both the base output and its audit record.
        for name in (
            "score",
            "rank_logits",
            "rank_probability",
            "uncertainty",
            "reliability",
            "weighted_group_pooled",
        ):
            value = np.asarray(getattr(self, name))
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(f"{name} must have a floating dtype")
            owned = np.array(value, copy=True)
            owned.setflags(write=False)
            object.__setattr__(self, name, owned)

    def validate(self) -> None:
        if self.score.ndim != 2:
            raise ValueError("score must have shape [B,N]")
        batch, stocks = self.score.shape
        if self.rank_logits.ndim != 3 or self.rank_logits.shape[:2] != (batch, stocks):
            raise ValueError("rank_logits must have shape [B,N,Q]")
        if self.rank_probability.shape != self.rank_logits.shape:
            raise ValueError("rank_probability must match rank_logits")
        if self.uncertainty.shape != (batch, stocks):
            raise ValueError("uncertainty must have shape [B,N]")
        if self.reliability.shape != (batch, stocks, 7):
            raise ValueError("reliability must have shape [B,N,7]")
        if self.weighted_group_pooled.ndim != 4 or self.weighted_group_pooled.shape[:3] != (
            batch,
            stocks,
            7,
        ):
            raise ValueError("weighted_group_pooled must have shape [B,N,7,D]")
