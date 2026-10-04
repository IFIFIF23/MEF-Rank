"""Strict tensor contracts shared by all benchmark backbones."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


GROUP_ORDER = (
    "G1_return_trend",
    "G2_volume_liquidity",
    "G3_volatility_downside",
    "G4_valuation_size",
    "G5_style_reversal",
    "G6_market_relative",
    "G7_ohlc_ope",
)
GROUP_DIMENSIONS = (8, 8, 8, 8, 8, 8, 77)
FEATURE_DIMENSION = sum(GROUP_DIMENSIONS)


def group_slices() -> tuple[slice, ...]:
    start = 0
    result: list[slice] = []
    for width in GROUP_DIMENSIONS:
        result.append(slice(start, start + width))
        start += width
    return tuple(result)


@dataclass(frozen=True)
class FinancialBatch:
    """One or more cross-sectional dates with per-stock lookback paths."""

    x: Tensor
    feature_mask: Tensor
    history_mask: Tensor
    stock_mask: Tensor
    label_rank: Tensor | None = None
    label_return: Tensor | None = None
    label_valid: Tensor | None = None

    def validate(self, *, lookback: int = 60) -> tuple[int, int]:
        if self.x.ndim != 4 or self.x.shape[2:] != (lookback, FEATURE_DIMENSION):
            raise ValueError(
                f"x must have shape [B,N,{lookback},{FEATURE_DIMENSION}], got {tuple(self.x.shape)}"
            )
        if self.feature_mask.shape != self.x.shape or self.feature_mask.dtype is not torch.bool:
            raise ValueError("feature_mask must be bool and match x")
        batch, stocks = self.x.shape[:2]
        if self.history_mask.shape != (batch, stocks, lookback):
            raise ValueError("history_mask must have shape [B,N,L]")
        if self.history_mask.dtype is not torch.bool:
            raise TypeError("history_mask must be bool")
        if self.stock_mask.shape != (batch, stocks) or self.stock_mask.dtype is not torch.bool:
            raise ValueError("stock_mask must be bool [B,N]")
        for value, name in (
            (self.label_rank, "label_rank"),
            (self.label_return, "label_return"),
            (self.label_valid, "label_valid"),
        ):
            if value is not None and value.shape != (batch, stocks):
                raise ValueError(f"{name} must have shape [B,N]")
        if self.label_valid is not None and self.label_valid.dtype is not torch.bool:
            raise TypeError("label_valid must be bool")
        if not torch.isfinite(self.x).all():
            raise ValueError("x must be finite; unavailable coordinates must be zero-masked")
        if bool((self.x.masked_select(~self.feature_mask) != 0).any()):
            raise ValueError("unavailable feature coordinates must be exactly zero")
        return batch, stocks


@dataclass(frozen=True)
class RankOutput:
    score: Tensor
    rank_logits: Tensor
    uncertainty: Tensor
    forecast: Tensor
    reliability: Tensor | None = None
    weighted_group_pooled: Tensor | None = None
    auxiliary_loss: Tensor | None = None

    def validate(self) -> tuple[int, int]:
        if self.score.ndim != 2:
            raise ValueError("score must have shape [B,N]")
        batch, stocks = self.score.shape
        if self.rank_logits.ndim != 3 or self.rank_logits.shape[:2] != (batch, stocks):
            raise ValueError("rank_logits must have shape [B,N,Q]")
        if self.uncertainty.shape != (batch, stocks):
            raise ValueError("uncertainty must have shape [B,N]")
        if self.forecast.ndim != 4 or self.forecast.shape[:2] != (batch, stocks):
            raise ValueError("forecast must have shape [B,N,H,C]")
        if self.reliability is not None and self.reliability.shape != (batch, stocks, 7):
            raise ValueError("reliability must have shape [B,N,7]")
        if self.weighted_group_pooled is not None:
            if self.weighted_group_pooled.ndim != 4:
                raise ValueError("weighted_group_pooled must have shape [B,N,7,D]")
            if self.weighted_group_pooled.shape[:3] != (batch, stocks, 7):
                raise ValueError("weighted_group_pooled has incompatible leading dimensions")
        tensors = [self.score, self.rank_logits, self.uncertainty, self.forecast]
        if self.reliability is not None:
            tensors.append(self.reliability)
        if self.weighted_group_pooled is not None:
            tensors.append(self.weighted_group_pooled)
        if not all(bool(torch.isfinite(value).all()) for value in tensors):
            raise ValueError("model output contains non-finite values")
        return batch, stocks

