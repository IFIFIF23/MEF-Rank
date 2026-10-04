"""High-level MEF-Rank v3.7.1 STR and one-pass CAL1 joint feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .identity import FROZEN_V371_IDENTITY, FrozenV371Identity
from .inverse import InverseResult, invert_reliability_shift
from .str_model import DayBatch, MEFRankSTR, STRCache, STRProfile, TorchSTRForward
from .v36 import JACFResult, JointState
from .v371 import CAL1Values, V371MemoryRecord, run_cal1


@dataclass(frozen=True)
class MEFRankResult:
    first: TorchSTRForward
    final: TorchSTRForward
    correction: JACFResult
    inverse: InverseResult | None
    delta_beta: np.ndarray
    reason_code: str
    second_forward_executed: bool

    def __post_init__(self) -> None:
        delta = np.asarray(self.delta_beta, dtype=np.float64)
        if delta.shape != (7,) or not np.isfinite(delta).all():
            raise ValueError("delta_beta must be finite [7]")
        owned = np.array(delta, copy=True)
        owned.setflags(write=False)
        object.__setattr__(self, "delta_beta", owned)
        if not self.second_forward_executed and self.final is not self.first:
            raise ValueError("v3.7.1 fallback must reuse the first forward object")

    @property
    def applied(self) -> bool:
        return self.second_forward_executed


class MEFRank(nn.Module):
    """Base STR plus the all-or-none v3.7.1 JACF/CAL1 correction."""

    def __init__(self, profile: STRProfile = STRProfile()) -> None:
        super().__init__()
        self.str = MEFRankSTR(profile)
        self.identity: FrozenV371Identity = FROZEN_V371_IDENTITY

    @property
    def architecture_id(self) -> str:
        return self.str.architecture_id

    def base_forward(
        self, batch: DayBatch, *, horizon: int = 5
    ) -> tuple[STRCache, TorchSTRForward]:
        cache = self.str.encode_once(batch, horizon=horizon)
        zero = torch.zeros(
            (cache.base_gate_logits.shape[0], 7),
            dtype=cache.base_gate_logits.dtype,
            device=cache.base_gate_logits.device,
        )
        first = self.str.rank_from_cache(cache, beta_logit=zero)
        return cache, first

    def corrected_from_cache(
        self,
        cache: STRCache,
        first: TorchSTRForward,
        *,
        state: JointState,
        records: Sequence[V371MemoryRecord],
        calibration: CAL1Values,
        query_date: str,
    ) -> MEFRankResult:
        """Apply the exact v3.7.1 joint correction to one cross-sectional day."""

        if first.score.shape[0] != 1 or cache.base_gate_logits.shape[0] != 1:
            raise ValueError("v3.7.1 online correction expects one query date per call")
        correction = run_cal1(
            state, records, calibration, query_date=str(query_date)
        )
        zero = np.zeros(7, dtype=np.float64)
        if not correction.apply:
            return MEFRankResult(
                first, first, correction, None, zero, correction.reason_code, False
            )
        inverse = invert_reliability_shift(
            cache.base_gate_logits[0].detach().cpu().numpy().astype(np.float64),
            cache.group_available_mask[0].detach().cpu().numpy().astype(bool),
            correction.target_shift,
            action_cap=0.50,
            iterations=24,
            tolerance=1.0e-5,
        )
        residual = float(np.max(np.abs(inverse.solver_residual)))
        if (
            not bool(inverse.converged.all())
            or residual > 1.0e-5
            or not np.isfinite(inverse.delta_beta).all()
        ):
            return MEFRankResult(
                first,
                first,
                correction,
                inverse,
                zero,
                "ABSTAIN_INVERSE_ALL_OR_NONE",
                False,
            )
        delta = np.asarray(inverse.delta_beta, dtype=np.float64)
        with torch.no_grad():
            final = self.str.rank_from_cache(
                cache,
                beta_logit=torch.as_tensor(
                    delta[None],
                    dtype=cache.base_gate_logits.dtype,
                    device=cache.base_gate_logits.device,
                ),
            )
        return MEFRankResult(
            first, final, correction, inverse, delta, "APPLY", True
        )

    def corrected_forward(
        self,
        batch: DayBatch,
        *,
        state: JointState,
        records: Sequence[V371MemoryRecord],
        calibration: CAL1Values,
        query_date: str,
        horizon: int = 5,
    ) -> MEFRankResult:
        cache, first = self.base_forward(batch, horizon=horizon)
        return self.corrected_from_cache(
            cache,
            first,
            state=state,
            records=records,
            calibration=calibration,
            query_date=query_date,
        )

    def forward(self, batch: DayBatch, *, horizon: int = 5) -> TorchSTRForward:
        _, first = self.base_forward(batch, horizon=horizon)
        return first
