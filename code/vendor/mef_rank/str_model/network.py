"""Canonical v3.3 PyTorch STR with one reusable pre-gate cache."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .day_batch import DayBatch, GROUP_DIMENSIONS, GROUP_ORDER


@dataclass(frozen=True)
class STRProfile:
    lookback: int = 60
    d_model: int = 96
    temporal_layers: int = 2
    cross_sectional_layers: int = 1
    ffn_dim: int = 192
    heads: int = 4
    dropout: float = 0.1
    base_gate_hidden_dim: int = 48
    base_gate_logit_max: float = 2.5
    rank_bins: int = 20
    score_residual_lambda: float = 0.1

    def validate(self) -> None:
        if self.lookback <= 0 or self.d_model <= 0 or self.ffn_dim <= 0:
            raise ValueError("profile dimensions must be positive")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if self.temporal_layers <= 0 or self.cross_sectional_layers <= 0:
            raise ValueError("transformer layer counts must be positive")
        if self.base_gate_logit_max <= 0 or self.rank_bins < 2:
            raise ValueError("gate bound and rank_bins are invalid")


@dataclass(frozen=True)
class STRCache:
    group_tokens: Tensor
    group_lag_mask: Tensor
    group_pooled: Tensor
    base_gate_logits: Tensor
    group_available_mask: Tensor
    history_mask: Tensor
    signal_universe_mask: Tensor
    shock_token: Tensor
    shock_token_mask: Tensor
    horizon: int


@dataclass(frozen=True)
class TorchSTRForward:
    score: Tensor
    rank_logits: Tensor
    rank_probability: Tensor
    uncertainty: Tensor
    reliability: Tensor
    weighted_group_pooled: Tensor

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

    def detached_numpy(self):
        """Convert only the six observable fields to the NumPy MBS boundary."""

        from .contracts import STRForward

        return STRForward(
            score=self.score.detach().cpu().numpy(),
            rank_logits=self.rank_logits.detach().cpu().numpy(),
            rank_probability=self.rank_probability.detach().cpu().numpy(),
            uncertainty=self.uncertainty.detach().cpu().numpy(),
            reliability=self.reliability.detach().cpu().numpy(),
            weighted_group_pooled=self.weighted_group_pooled.detach().cpu().numpy(),
        )


class _GroupProjection(nn.Module):
    def __init__(self, feature_dim: int, d_model: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.linear = nn.Linear(feature_dim, d_model)

    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(self.linear(self.layer_norm(value)))


class _GateMLP(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.first = nn.Linear(d_model, hidden_dim)
        self.second = nn.Linear(hidden_dim, 1)

    def forward(self, value: Tensor) -> Tensor:
        return self.second(F.gelu(self.first(value))).squeeze(-1)


def _masked_query_pool(tokens: Tensor, mask: Tensor, query: Tensor) -> Tensor:
    logits = torch.einsum("bnld,d->bnl", tokens, query) / math.sqrt(tokens.shape[-1])
    negative = torch.finfo(logits.dtype).min
    logits = torch.where(mask, logits, torch.full_like(logits, negative))
    maximum = logits.max(dim=2, keepdim=True).values
    maximum = torch.where(mask.any(dim=2, keepdim=True), maximum, torch.zeros_like(maximum))
    weights = torch.where(mask, torch.exp(logits - maximum), torch.zeros_like(logits))
    weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    return torch.einsum("bnl,bnld->bnd", weights, tokens)


def _safe_transformer(
    encoder: nn.TransformerEncoder, value: Tensor, valid: Tensor
) -> Tensor:
    """Avoid all-key-masked NaNs while preserving exact zero invalid rows."""

    safe_valid = valid.clone()
    all_invalid = ~safe_valid.any(dim=1)
    if all_invalid.any():
        safe_valid[all_invalid, 0] = True
        value = value.clone()
        value[all_invalid, 0] = 0.0
    output = encoder(value, src_key_padding_mask=~safe_valid, mask=None, is_causal=False)
    output = output.masked_fill(~valid.unsqueeze(-1), 0.0)
    output = output.masked_fill(all_invalid[:, None, None], 0.0)
    return output


class MEFRankSTR(nn.Module):
    """Seven-group short-term ranker with an explicit reliability interface."""

    def __init__(self, profile: STRProfile = STRProfile()) -> None:
        super().__init__()
        profile.validate()
        self.profile = profile
        self.group_projection = nn.ModuleList(
            [_GroupProjection(feature_dim, profile.d_model) for feature_dim in GROUP_DIMENSIONS]
        )
        self.group_pool_query = nn.Parameter(torch.empty(7, profile.d_model))
        nn.init.normal_(self.group_pool_query, mean=0.0, std=profile.d_model**-0.5)
        self.base_gate = nn.ModuleList(
            [_GateMLP(profile.d_model, profile.base_gate_hidden_dim) for _ in range(7)]
        )
        self.position = nn.Parameter(torch.empty(profile.lookback + 1, profile.d_model))
        nn.init.normal_(self.position, mean=0.0, std=profile.d_model**-0.5)
        self.shock_projection = nn.Sequential(
            nn.Linear(10, profile.d_model),
            nn.LayerNorm(profile.d_model),
            nn.GELU(),
        )
        self.fsa_norm = nn.LayerNorm(profile.d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=profile.d_model,
            nhead=profile.heads,
            dim_feedforward=profile.ffn_dim,
            dropout=profile.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            temporal_layer, num_layers=profile.temporal_layers, enable_nested_tensor=False
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=profile.d_model,
            nhead=profile.heads,
            dim_feedforward=profile.ffn_dim,
            dropout=profile.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_sectional = nn.TransformerEncoder(
            cross_layer,
            num_layers=profile.cross_sectional_layers,
            enable_nested_tensor=False,
        )
        self.rank_head = nn.Linear(profile.d_model, profile.rank_bins)
        self.residual_head = nn.Linear(profile.d_model, 1)
        support = torch.linspace(0.0, 1.0, profile.rank_bins)
        self.register_buffer("rank_support", support, persistent=True)

    @property
    def architecture_id(self) -> str:
        p = self.profile
        return (
            f"mef_rank_str_v3_3_L{p.lookback}_D{p.d_model}_T{p.temporal_layers}"
            f"_X{p.cross_sectional_layers}_H{p.heads}_Q{p.rank_bins}"
        )

    def encode_once(self, batch: DayBatch, *, horizon: int) -> STRCache:
        batch_size, stocks = batch.validate(lookback=self.profile.lookback, horizon=horizon)
        group_content: list[Tensor] = []
        group_pooled: list[Tensor] = []
        base_logits: list[Tensor] = []
        positions = self.position[: self.profile.lookback].view(
            1, 1, self.profile.lookback, self.profile.d_model
        )
        for group_index, group_name in enumerate(GROUP_ORDER):
            values = batch.x_group[group_name]
            feature_mask = batch.feature_mask[group_name]
            safe_values = torch.where(feature_mask, values, torch.zeros_like(values))
            content = self.group_projection[group_index](safe_values)
            lag_mask = batch.group_lag_mask[..., group_index]
            content = content.masked_fill(~lag_mask.unsqueeze(-1), 0.0)
            positioned = (content + positions).masked_fill(~lag_mask.unsqueeze(-1), 0.0)
            pooled = _masked_query_pool(
                positioned, lag_mask, self.group_pool_query[group_index]
            )
            available = batch.group_available_mask[..., group_index]
            pooled = pooled.masked_fill(~available.unsqueeze(-1), 0.0)
            raw_logit = self.base_gate[group_index](pooled)
            bound = self.profile.base_gate_logit_max
            bounded = bound * torch.tanh(raw_logit / bound)
            bounded = bounded.masked_fill(~available, 0.0)
            group_content.append(positioned)
            group_pooled.append(pooled)
            base_logits.append(bounded)
        tokens = torch.stack(group_content, dim=3)
        pooled = torch.stack(group_pooled, dim=2)
        logits = torch.stack(base_logits, dim=2)
        shock_value = torch.where(batch.shock_mask, batch.shock_raw, torch.zeros_like(batch.shock_raw))
        shock_input = torch.cat([shock_value, batch.shock_mask.to(shock_value.dtype)], dim=-1)
        shock_token = self.shock_projection(shock_input)
        shock_token_mask = batch.shock_mask.any(dim=-1)
        if tokens.shape != (
            batch_size,
            stocks,
            self.profile.lookback,
            7,
            self.profile.d_model,
        ):
            raise RuntimeError("internal group token shape mismatch")
        return STRCache(
            group_tokens=tokens,
            group_lag_mask=batch.group_lag_mask,
            group_pooled=pooled,
            base_gate_logits=logits,
            group_available_mask=batch.group_available_mask,
            history_mask=batch.history_mask,
            signal_universe_mask=batch.signal_universe_mask,
            shock_token=shock_token,
            shock_token_mask=shock_token_mask,
            horizon=horizon,
        )

    def rank_from_cache(self, cache: STRCache, *, beta_logit: Tensor) -> TorchSTRForward:
        batch, stocks, lookback, groups, d_model = cache.group_tokens.shape
        if lookback != self.profile.lookback or groups != 7 or d_model != self.profile.d_model:
            raise ValueError("cache is incompatible with this STR profile")
        if beta_logit.shape != (batch, 7):
            raise ValueError("beta_logit must have shape [B,7]")
        if not torch.isfinite(beta_logit).all():
            raise ValueError("beta_logit must be finite")
        reliability = torch.sigmoid(cache.base_gate_logits + beta_logit[:, None, :])
        reliability = reliability.masked_fill(~cache.group_available_mask, 0.0)
        weighted_group_pooled = reliability.unsqueeze(-1) * cache.group_pooled
        gated = (
            cache.group_tokens
            * reliability[:, :, None, :, None]
            * cache.group_lag_mask.unsqueeze(-1)
        ).sum(dim=3)
        gated = self.fsa_norm(gated)
        gated = gated.masked_fill(~cache.history_mask.unsqueeze(-1), 0.0)
        shock = cache.shock_token + self.position[self.profile.lookback]
        shock = shock[:, None, None, :].expand(batch, stocks, 1, d_model)
        temporal_input = torch.cat([gated, shock], dim=2).reshape(
            batch * stocks, lookback + 1, d_model
        )
        shock_mask = cache.shock_token_mask[:, None].expand(batch, stocks)
        temporal_valid = torch.cat([cache.history_mask, shock_mask.unsqueeze(-1)], dim=2).reshape(
            batch * stocks, lookback + 1
        )
        temporal_output = _safe_transformer(self.temporal, temporal_input, temporal_valid)
        readout = temporal_output[:, lookback - 1].reshape(batch, stocks, d_model)
        readout_valid = cache.history_mask[:, :, lookback - 1] & cache.signal_universe_mask
        readout = readout.masked_fill(~readout_valid.unsqueeze(-1), 0.0)
        cross = _safe_transformer(
            self.cross_sectional, readout, cache.signal_universe_mask
        )
        cross = cross.masked_fill(~cache.signal_universe_mask.unsqueeze(-1), 0.0)
        rank_logits = self.rank_head(cross)
        probability = torch.softmax(rank_logits, dim=-1)
        residual = self.residual_head(cross).squeeze(-1)
        score = torch.sum(probability * self.rank_support, dim=-1) + (
            self.profile.score_residual_lambda * torch.tanh(residual)
        )
        entropy = -torch.sum(
            probability * torch.log(probability.clamp_min(torch.finfo(probability.dtype).tiny)),
            dim=-1,
        ) / math.log(self.profile.rank_bins)
        invalid = ~cache.signal_universe_mask
        score = score.masked_fill(invalid, 0.0)
        rank_logits = rank_logits.masked_fill(invalid.unsqueeze(-1), 0.0)
        probability = probability.masked_fill(invalid.unsqueeze(-1), 0.0)
        entropy = entropy.masked_fill(invalid, 0.0)
        output = TorchSTRForward(
            score=score,
            rank_logits=rank_logits,
            rank_probability=probability,
            uncertainty=entropy,
            reliability=reliability,
            weighted_group_pooled=weighted_group_pooled,
        )
        output.validate()
        return output

    def forward(self, batch: DayBatch, *, horizon: int, beta_logit: Tensor | None = None):
        cache = self.encode_once(batch, horizon=horizon)
        if beta_logit is None:
            beta_logit = torch.zeros(
                (batch.history_mask.shape[0], 7),
                dtype=batch.history_mask.dtype if batch.history_mask.is_floating_point() else batch.x_group[GROUP_ORDER[0]].dtype,
                device=batch.history_mask.device,
            )
        return self.rank_from_cache(cache, beta_logit=beta_logit)
