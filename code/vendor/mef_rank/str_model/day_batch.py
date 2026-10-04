"""Torch DayBatch contract for the v3.3 STR implementation.

This module is imported only by the deep-training environment.  The repository's
NumPy data pipeline remains torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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


@dataclass
class DayBatch:
    trade_date: list[str]
    ts_code: list[list[str]]
    x_group: dict[str, Tensor]
    feature_mask: dict[str, Tensor]
    history_mask: Tensor
    group_lag_mask: Tensor
    stock_present_mask: Tensor
    signal_universe_mask: Tensor
    group_available_mask: Tensor
    fec_common_raw: Tensor
    fec_horizon_raw: dict[int, Tensor]
    fec_common_mask: Tensor
    fec_horizon_mask: dict[int, Tensor]
    shock_raw: Tensor
    shock_mask: Tensor
    label_return: Tensor | None = None
    label_rank: Tensor | None = None
    label_valid_mask: Tensor | None = None
    label_invalid_reason: Any | None = None
    label_maturity_date: Any | None = None
    auxiliary_entry_executable_mask: Tensor | None = None
    auxiliary_exit_resolved_mask: Tensor | None = None
    auxiliary_exit_type: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, lookback: int, horizon: int) -> tuple[int, int]:
        if set(self.x_group) != set(GROUP_ORDER) or set(self.feature_mask) != set(GROUP_ORDER):
            raise ValueError(f"x_group and feature_mask must contain exactly {GROUP_ORDER}")
        if self.history_mask.dtype is not torch.bool:
            raise TypeError("history_mask must be bool")
        if self.history_mask.ndim != 3 or self.history_mask.shape[2] != lookback:
            raise ValueError("history_mask must have shape [B,N,L]")
        batch, stocks, _ = self.history_mask.shape
        expected_group_shape = (batch, stocks, lookback, 7)
        if self.group_lag_mask.shape != expected_group_shape or self.group_lag_mask.dtype is not torch.bool:
            raise ValueError("group_lag_mask must have shape [B,N,L,7] and bool dtype")
        for tensor, name in (
            (self.stock_present_mask, "stock_present_mask"),
            (self.signal_universe_mask, "signal_universe_mask"),
        ):
            if tensor.shape != (batch, stocks) or tensor.dtype is not torch.bool:
                raise ValueError(f"{name} must have shape [B,N] and bool dtype")
        if self.group_available_mask.shape != (batch, stocks, 7):
            raise ValueError("group_available_mask must have shape [B,N,7]")
        if self.group_available_mask.dtype is not torch.bool:
            raise TypeError("group_available_mask must be bool")
        if len(self.trade_date) != batch or len(self.ts_code) != batch:
            raise ValueError("trade_date and ts_code must match batch dimension")
        if any(len(codes) != stocks for codes in self.ts_code):
            raise ValueError("every ts_code row must match stock dimension")
        for group_index, (group_name, feature_dim) in enumerate(
            zip(GROUP_ORDER, GROUP_DIMENSIONS)
        ):
            values = self.x_group[group_name]
            mask = self.feature_mask[group_name]
            if values.shape != (batch, stocks, lookback, feature_dim):
                raise ValueError(f"{group_name} must have shape [B,N,L,{feature_dim}]")
            if mask.shape != values.shape or mask.dtype is not torch.bool:
                raise ValueError(f"{group_name} feature_mask must match values and be bool")
            expected_lag = mask.all(dim=-1) & self.history_mask
            if not torch.equal(expected_lag, self.group_lag_mask[..., group_index]):
                raise ValueError(f"{group_name} group_lag_mask is inconsistent with feature_mask")
        expected_available = self.group_lag_mask.any(dim=2)
        if not torch.equal(expected_available, self.group_available_mask):
            raise ValueError("group_available_mask must equal any(group_lag_mask over lag)")
        if torch.any(self.signal_universe_mask & ~self.stock_present_mask):
            raise ValueError("signal_universe_mask must be a subset of stock_present_mask")
        if self.shock_raw.shape != (batch, 5) or self.shock_mask.shape != (batch, 5):
            raise ValueError("shock_raw and shock_mask must have shape [B,5]")
        if self.shock_mask.dtype is not torch.bool:
            raise TypeError("shock_mask must be bool")
        if horizon not in self.fec_horizon_raw or horizon not in self.fec_horizon_mask:
            raise ValueError(f"missing horizon-specific FEC for H{horizon}")
        if self.fec_common_raw.shape[:1] != (batch,) or self.fec_common_mask.shape != self.fec_common_raw.shape:
            raise ValueError("common FEC value/mask shape mismatch")
        if self.fec_common_mask.dtype is not torch.bool:
            raise TypeError("fec_common_mask must be bool")
        if self.fec_horizon_raw[horizon].shape[:1] != (batch,):
            raise ValueError("horizon FEC must match batch dimension")
        if self.fec_horizon_mask[horizon].shape != self.fec_horizon_raw[horizon].shape:
            raise ValueError("horizon FEC value/mask shape mismatch")
        if self.fec_horizon_mask[horizon].dtype is not torch.bool:
            raise TypeError("fec_horizon_mask must be bool")
        for tensor, name in (
            (self.label_return, "label_return"),
            (self.label_rank, "label_rank"),
            (self.label_valid_mask, "label_valid_mask"),
        ):
            if tensor is not None and tensor.shape != (batch, stocks):
                raise ValueError(f"{name} must have shape [B,N]")
        if self.label_valid_mask is not None and self.label_valid_mask.dtype is not torch.bool:
            raise TypeError("label_valid_mask must be bool")
        return batch, stocks
