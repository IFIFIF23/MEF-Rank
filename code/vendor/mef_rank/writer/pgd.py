"""Canonical fixed-budget Writer PGD and stratified bootstrap quality filter."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

if TYPE_CHECKING:
    from ..str_model.network import MEFRankSTR, STRCache, TorchSTRForward


def _seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:8], "big")


def fixed_pair_indices(
    label_rank: Tensor,
    sample_indices: Tensor,
    *,
    minimum_rank_gap: float,
    maximum_pairs: int,
    namespace: str,
) -> Tensor:
    """Create one deterministic pair set, preserving repeated bootstrap rows."""

    if sample_indices.ndim != 1 or sample_indices.dtype != torch.long:
        raise ValueError("sample_indices must be a one-dimensional long tensor")
    if maximum_pairs <= 0 or minimum_rank_gap < 0:
        raise ValueError("pair controls are invalid")
    local_labels = label_rank[sample_indices].detach().cpu().numpy().astype(np.float64)
    count = len(local_labels)
    if count < 2:
        return torch.empty((0, 2), dtype=torch.long, device=sample_indices.device)
    rng = np.random.default_rng(_seed(namespace))
    pairs: set[tuple[int, int]] = set()
    # Enumerate small fixtures exactly; sample large anchors with a deterministic
    # bounded loop so the Writer never materializes an O(N^2) table.
    if count <= 256:
        candidates = [
            (left, right)
            for left in range(count)
            for right in range(left + 1, count)
            if abs(local_labels[left] - local_labels[right]) >= minimum_rank_gap
        ]
        if len(candidates) > maximum_pairs:
            selected = rng.choice(len(candidates), size=maximum_pairs, replace=False)
            candidates = [candidates[index] for index in selected]
        pairs.update(candidates)
    else:
        attempts = 0
        maximum_attempts = max(20 * maximum_pairs, 10000)
        while len(pairs) < maximum_pairs and attempts < maximum_attempts:
            batch = min(4 * (maximum_pairs - len(pairs)), 16384)
            left = rng.integers(0, count, size=batch)
            right = rng.integers(0, count, size=batch)
            for i, j in zip(left, right):
                attempts += 1
                if i == j or abs(local_labels[i] - local_labels[j]) < minimum_rank_gap:
                    continue
                pair = (int(min(i, j)), int(max(i, j)))
                pairs.add(pair)
                if len(pairs) >= maximum_pairs or attempts >= maximum_attempts:
                    break
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long, device=sample_indices.device)
    local = torch.as_tensor(sorted(pairs), dtype=torch.long, device=sample_indices.device)
    return sample_indices[local]


def _normalized_weighted_mean(values: Tensor, weights: Tensor | None) -> Tensor:
    if weights is None:
        return values.mean()
    if weights.ndim != 1 or weights.shape != values.shape:
        raise ValueError("loss weights must align with the unreduced loss")
    normalized = weights.to(device=values.device, dtype=values.dtype)
    if (
        not bool(torch.isfinite(normalized).all())
        or bool((normalized < 0).any())
        or float(normalized.sum()) <= 0.0
    ):
        raise ValueError("loss weights must be finite, nonnegative and nonempty")
    return (values * normalized).sum() / normalized.sum()


def soft_ordinal_loss(
    rank_logits: Tensor,
    label_rank: Tensor,
    sample_weights: Tensor | None = None,
) -> Tensor:
    if rank_logits.ndim != 2 or label_rank.shape != rank_logits.shape[:1]:
        raise ValueError("ordinal logits/label shape mismatch")
    bins = rank_logits.shape[-1]
    center = torch.arange(bins, dtype=rank_logits.dtype, device=rank_logits.device)
    target_center = label_rank.clamp(0.0, 1.0).unsqueeze(-1) * (bins - 1)
    target = torch.exp(-0.5 * (center - target_center).square())
    target = target / target.sum(dim=-1, keepdim=True)
    per_stock = -(target * F.log_softmax(rank_logits, dim=-1)).sum(dim=-1)
    return _normalized_weighted_mean(per_stock, sample_weights)


def pairwise_loss(
    score: Tensor,
    label_rank: Tensor,
    pairs: Tensor,
    pair_weights: Tensor | None = None,
) -> Tensor:
    if pairs.numel() == 0:
        return score.sum() * 0.0
    left, right = pairs[:, 0], pairs[:, 1]
    direction = torch.sign(label_rank[left] - label_rank[right])
    per_pair = F.softplus(-direction * (score[left] - score[right]))
    return _normalized_weighted_mean(per_pair, pair_weights)


@dataclass(frozen=True)
class PredictionLossComponents:
    softordinal: Tensor
    pair: Tensor
    combined: Tensor
    pair_count: int


def prediction_loss_components(
    output: "TorchSTRForward",
    label_rank: Tensor,
    sample_indices: Tensor,
    *,
    sample_weights: Tensor | None = None,
    pair_loss_weight: float,
    pair_count: int,
    pair_min_rank_gap: float,
    namespace: str,
) -> PredictionLossComponents:
    if output.score.shape[0] != 1 or label_rank.ndim != 1:
        raise ValueError("Writer prediction loss requires one date and one-dimensional labels")
    if sample_indices.numel() == 0:
        raise ValueError("Writer sample is empty")
    if sample_weights is not None:
        if sample_weights.shape != sample_indices.shape:
            raise ValueError("Writer sample weights must align with sample indices")
        if sample_indices.unique().numel() != sample_indices.numel():
            raise ValueError(
                "weighted Writer supervision requires unique stock nodes; "
                "bootstrap duplicates belong in multiplicity weights"
            )
    logits = output.rank_logits[0, sample_indices]
    labels = label_rank[sample_indices]
    ordinal = soft_ordinal_loss(logits, labels, sample_weights)
    pairs = fixed_pair_indices(
        label_rank,
        sample_indices,
        minimum_rank_gap=pair_min_rank_gap,
        maximum_pairs=pair_count,
        namespace=namespace,
    )
    pair_weights = None
    if sample_weights is not None and pairs.numel():
        stock_weights = torch.zeros(
            label_rank.shape[0],
            dtype=sample_weights.dtype,
            device=sample_weights.device,
        )
        stock_weights[sample_indices] = sample_weights
        pair_weights = stock_weights[pairs[:, 0]] * stock_weights[pairs[:, 1]]
    pair = pairwise_loss(output.score[0], label_rank, pairs, pair_weights)
    return PredictionLossComponents(
        softordinal=ordinal,
        pair=pair,
        combined=ordinal + pair_loss_weight * pair,
        pair_count=int(pairs.shape[0]),
    )


def prediction_loss(
    output: "TorchSTRForward",
    label_rank: Tensor,
    sample_indices: Tensor,
    *,
    sample_weights: Tensor | None = None,
    pair_loss_weight: float,
    pair_count: int,
    pair_min_rank_gap: float,
    namespace: str,
) -> Tensor:
    return prediction_loss_components(
        output,
        label_rank,
        sample_indices,
        sample_weights=sample_weights,
        pair_loss_weight=pair_loss_weight,
        pair_count=pair_count,
        pair_min_rank_gap=pair_min_rank_gap,
        namespace=namespace,
    ).combined


@dataclass(frozen=True)
class WriterSolveResult:
    # Candidate before the explicit zero-action comparison.  These fields are
    # diagnostic only and never enter OOB replay or Memory.
    full_solve_raw_action: Tensor
    full_solve_transfer_action: Tensor
    # Selected action after the zero-action candidate is allowed to win.
    raw_action: Tensor
    transfer_action: Tensor
    zero_prediction_loss: float
    final_prediction_loss: float
    final_objective: float
    accepted_steps: int
    reason: str
    zero_action_winner: bool


@dataclass(frozen=True)
class WriterBootstrapResult:
    full_solve_raw_action: Tensor
    # Unclipped full-solve action is retained for cap-collision diagnostics only.
    # It is never persisted as the Memory Value or replayed by the checkpoint.
    raw_action: Tensor
    # Selected transfer action before the groupwise W0 mask.
    pre_mask_transfer_action: Tensor
    # ``transfer_action`` remains the checkpoint-local logit displacement used
    # for OOB replay.  Memory must persist ``action_reliability_shift`` instead.
    transfer_action: Tensor
    pre_mask_reliability_shift: Tensor
    action_reliability_shift: Tensor
    action_reliability_variance: Tensor
    action_reliability_mask: Tensor
    bootstrap_actions: Tensor
    bootstrap_reliability_shifts: Tensor
    bootstrap_valid_mask: Tensor
    bootstrap_positive_gain_mask: Tensor
    bootstrap_sign_count: Tensor
    bootstrap_sign_rate: Tensor
    bootstrap_reliability_variance: Tensor
    bootstrap_solve_reason: tuple[str, ...]
    oob_relative_gains: Tensor
    oob_relative_gain_median: float
    oob_relative_gain_mad: float
    oob_valid_repeat_count: int
    writer_stock_count: int
    group_failure_reason: tuple[str, ...]
    zero_action_winner: bool
    action_cap: float
    reason: str
    memory_eligible: bool


def _writer_context(model: "MEFRankSTR"):
    class Context:
        def __enter__(self):
            self.training = model.training
            self.requires = [parameter.requires_grad for parameter in model.parameters()]
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            return model

        def __exit__(self, exc_type, exc, traceback):
            for parameter, requires in zip(model.parameters(), self.requires):
                parameter.requires_grad_(requires)
            model.train(self.training)
            return False

    return Context()


def _detached_cache(cache: "STRCache") -> "STRCache":
    if hasattr(cache, 'detached'):
        return cache.detached()
    from ..str_model.network import STRCache

    return STRCache(
        group_tokens=cache.group_tokens.detach(),
        group_lag_mask=cache.group_lag_mask.detach(),
        group_pooled=cache.group_pooled.detach(),
        base_gate_logits=cache.base_gate_logits.detach(),
        group_available_mask=cache.group_available_mask.detach(),
        history_mask=cache.history_mask.detach(),
        signal_universe_mask=cache.signal_universe_mask.detach(),
        shock_token=cache.shock_token.detach(),
        shock_token_mask=cache.shock_token_mask.detach(),
        horizon=cache.horizon,
    )


def solve_writer_action(
    model: "MEFRankSTR",
    cache: "STRCache",
    label_rank: Tensor,
    sample_indices: Tensor,
    *,
    sample_weights: Tensor | None = None,
    namespace: str,
    pair_loss_weight: float,
    transfer_logit_cap: float,
    action_l2: float = 0.05,
    steps: int = 5,
    initial_step_size: float = 0.15,
    backtracking_trials: int = 4,
    backtracking_factor: float = 0.5,
    raw_logit_cap: float = 0.75,
    pair_count: int = 4096,
    pair_min_rank_gap: float = 0.10,
    active_coordinate_mask: Tensor | None = None,
) -> WriterSolveResult:
    """Optimize only one seven-dimensional action against a frozen checkpoint."""

    if cache.base_gate_logits.shape[0] != 1 or label_rank.ndim != 1:
        raise ValueError("Writer requires one anchor date")
    if sample_indices.numel() < 2:
        zero = torch.zeros((1, 7), dtype=cache.base_gate_logits.dtype, device=cache.base_gate_logits.device)
        return WriterSolveResult(
            zero,
            zero,
            zero,
            zero,
            math.nan,
            math.nan,
            math.nan,
            0,
            "WRITER_INVALID_LOW_SUPPORT",
            True,
        )
    if min(
        pair_loss_weight,
        action_l2,
        initial_step_size,
        backtracking_factor,
        raw_logit_cap,
        transfer_logit_cap,
    ) < 0 or steps <= 0 or backtracking_trials <= 0:
        raise ValueError("Writer solver controls are invalid")
    dtype = cache.base_gate_logits.dtype
    device = cache.base_gate_logits.device
    if active_coordinate_mask is None:
        active = torch.ones((1, 7), dtype=torch.bool, device=device)
    else:
        active = active_coordinate_mask.to(device=device, dtype=torch.bool)
        if active.shape == (7,):
            active = active.unsqueeze(0)
        if active.shape != (1, 7) or not bool(active.any()):
            raise ValueError("active_coordinate_mask must select a nonempty subset of seven groups")
    cache = _detached_cache(cache)
    labels = label_rank.to(device=device, dtype=dtype)
    indices = sample_indices.to(device=device, dtype=torch.long)
    weights = (
        None
        if sample_weights is None
        else sample_weights.to(device=device, dtype=dtype)
    )
    with _writer_context(model):
        zero = torch.zeros((1, 7), dtype=dtype, device=device)
        with torch.no_grad():
            zero_forward = model.rank_from_cache(cache, beta_logit=zero)
            zero_prediction = prediction_loss(
                zero_forward,
                labels,
                indices,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
                pair_count=pair_count,
                pair_min_rank_gap=pair_min_rank_gap,
                namespace=f"{namespace}|pairs",
            )
        delta = zero.clone().requires_grad_(True)
        accepted = 0
        for step_index in range(steps):
            output = model.rank_from_cache(cache, beta_logit=delta)
            pred = prediction_loss(
                output,
                labels,
                indices,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
                pair_count=pair_count,
                pair_min_rank_gap=pair_min_rank_gap,
                namespace=f"{namespace}|pairs",
            )
            objective = pred + action_l2 * delta.square().sum()
            (gradient,) = torch.autograd.grad(objective, delta, create_graph=False)
            gradient = gradient.masked_fill(~active, 0.0)
            accepted_candidate: Tensor | None = None
            for trial in range(backtracking_trials):
                step_size = initial_step_size * (backtracking_factor**trial)
                candidate = (delta - step_size * gradient).clamp(-raw_logit_cap, raw_logit_cap)
                candidate = candidate.masked_fill(~active, 0.0)
                with torch.no_grad():
                    candidate_output = model.rank_from_cache(cache, beta_logit=candidate)
                    candidate_pred = prediction_loss(
                        candidate_output,
                        labels,
                        indices,
                        sample_weights=weights,
                        pair_loss_weight=pair_loss_weight,
                        pair_count=pair_count,
                        pair_min_rank_gap=pair_min_rank_gap,
                        namespace=f"{namespace}|pairs",
                    )
                    candidate_objective = candidate_pred + action_l2 * candidate.square().sum()
                if float(candidate_objective) < float(objective.detach()):
                    accepted_candidate = candidate.detach()
                    break
            if accepted_candidate is None:
                delta = delta.detach().requires_grad_(True)
                continue
            delta = accepted_candidate.requires_grad_(True)
            accepted += 1
        full_solve_raw = delta.detach().masked_fill(~active, 0.0)
        full_solve_transfer = full_solve_raw.clamp(
            -transfer_logit_cap, transfer_logit_cap
        )
        with torch.no_grad():
            final_output = model.rank_from_cache(
                cache, beta_logit=full_solve_transfer
            )
            final_prediction = prediction_loss(
                final_output,
                labels,
                indices,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
                pair_count=pair_count,
                pair_min_rank_gap=pair_min_rank_gap,
                namespace=f"{namespace}|pairs",
            )
            final_objective = (
                final_prediction + action_l2 * full_solve_transfer.square().sum()
            )
        if float(final_objective) >= float(zero_prediction):
            raw = zero
            transfer = zero
            final_prediction = zero_prediction
            final_objective = zero_prediction
            reason = "WRITER_INVALID_NO_IMPROVEMENT"
            zero_action_winner = True
        else:
            raw = full_solve_raw
            transfer = full_solve_transfer
            reason = "VALID"
            zero_action_winner = False
    return WriterSolveResult(
        full_solve_raw_action=full_solve_raw.detach(),
        full_solve_transfer_action=full_solve_transfer.detach(),
        raw_action=raw.detach(),
        transfer_action=transfer.detach(),
        zero_prediction_loss=float(zero_prediction),
        final_prediction_loss=float(final_prediction),
        final_objective=float(final_objective),
        accepted_steps=accepted,
        reason=reason,
        zero_action_winner=zero_action_winner,
    )


def _stratified_bootstrap(
    valid_indices: np.ndarray, strata: np.ndarray, *, namespace: str
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(_seed(namespace))
    selected: list[np.ndarray] = []
    for value in sorted(set(strata[valid_indices].tolist())):
        members = valid_indices[strata[valid_indices] == value]
        if len(members):
            selected.append(rng.choice(members, size=len(members), replace=True))
    inbag = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    oob = np.setdiff1d(valid_indices, np.unique(inbag), assume_unique=False)
    return inbag, oob


def _mean_reliability_shift(
    cache: "STRCache",
    delta_logit: Tensor,
    writer_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map a checkpoint-local logit action to canonical mean reliability shift."""

    if cache.base_gate_logits.shape[0] != 1 or delta_logit.shape != (1, 7):
        raise ValueError("reliability-shift mapping requires [1,N,7] logits and [1,7] action")
    if writer_mask.ndim != 1 or writer_mask.shape[0] != cache.base_gate_logits.shape[1]:
        raise ValueError("writer_mask must align with the cache stock axis")
    valid = (
        writer_mask.to(device=cache.base_gate_logits.device, dtype=torch.bool)[:, None]
        & cache.signal_universe_mask[0, :, None]
        & cache.group_available_mask[0]
    )
    base = torch.sigmoid(cache.base_gate_logits[0].detach())
    moved = torch.sigmoid(cache.base_gate_logits[0].detach() + delta_logit.detach())
    difference = moved - base
    support = valid.sum(dim=0)
    shift = torch.zeros(7, dtype=difference.dtype, device=difference.device)
    supported = support > 0
    shift[supported] = (
        (difference * valid.to(difference.dtype)).sum(dim=0)[supported]
        / support[supported].to(difference.dtype)
    )
    return shift, support


def solve_writer_with_bootstrap(
    model: "MEFRankSTR",
    cache: "STRCache",
    label_rank: Tensor,
    writer_mask: Tensor,
    strata: Tensor,
    group_stock_support: Tensor,
    *,
    namespace: str,
    pair_loss_weight: float,
    action_cap: float,
    bootstrap_repeats: int = 5,
    min_nonempty_oob_repeats: int = 3,
    sign_agreement_min: float = 0.80,
    action_nonzero_epsilon: float = 0.005,
    action_coordinate_min_stocks: int = 20,
    oob_denominator_epsilon: float = 1.0e-12,
    **solver_kwargs,
) -> WriterBootstrapResult:
    if writer_mask.ndim != 1 or label_rank.shape != writer_mask.shape or strata.shape != writer_mask.shape:
        raise ValueError("label_rank, writer_mask and strata must be one-dimensional and aligned")
    if group_stock_support.shape != (7,):
        raise ValueError("group_stock_support must have shape [7]")
    if bootstrap_repeats < 2 or min_nonempty_oob_repeats <= 0 or action_cap <= 0:
        raise ValueError("bootstrap controls are invalid")
    valid = torch.nonzero(writer_mask, as_tuple=False).flatten()
    full = solve_writer_action(
        model,
        cache,
        label_rank,
        valid,
        namespace=f"{namespace}|full",
        pair_loss_weight=pair_loss_weight,
        transfer_logit_cap=action_cap,
        **solver_kwargs,
    )
    device = cache.base_gate_logits.device
    dtype = cache.base_gate_logits.dtype
    actions = torch.zeros((bootstrap_repeats, 7), dtype=dtype, device=device)
    reliability_shifts = torch.zeros(
        (bootstrap_repeats, 7), dtype=dtype, device=device
    )
    gains = torch.full((bootstrap_repeats,), float("nan"), dtype=dtype, device=device)
    repeat_valid = torch.zeros(bootstrap_repeats, dtype=torch.bool, device=device)
    solve_reasons: list[str] = ["WRITER_INVALID_OOB"] * bootstrap_repeats
    valid_np = valid.detach().cpu().numpy().astype(np.int64)
    strata_np = strata.detach().cpu().numpy()
    labels = label_rank.to(device=device, dtype=dtype)
    full_reliability_shift, canonical_support = _mean_reliability_shift(
        cache, full.transfer_action, writer_mask
    )
    declared_support = group_stock_support.to(
        device=device, dtype=canonical_support.dtype
    )
    if not torch.equal(declared_support, canonical_support):
        raise ValueError(
            "group_stock_support must equal writer_mask & signal_universe & group availability"
        )
    for repeat in range(bootstrap_repeats):
        inbag_np, oob_np = _stratified_bootstrap(
            valid_np, strata_np, namespace=f"{namespace}|bootstrap|{repeat}"
        )
        if len(inbag_np) < 2 or len(oob_np) < 2:
            continue
        inbag = torch.as_tensor(inbag_np, dtype=torch.long, device=device)
        oob = torch.as_tensor(oob_np, dtype=torch.long, device=device)
        solved = solve_writer_action(
            model,
            cache,
            labels,
            inbag,
            namespace=f"{namespace}|bootstrap|{repeat}",
            pair_loss_weight=pair_loss_weight,
            transfer_logit_cap=action_cap,
            **solver_kwargs,
        )
        solve_reasons[repeat] = solved.reason
        actions[repeat] = solved.transfer_action[0]
        reliability_shifts[repeat], _ = _mean_reliability_shift(
            cache, solved.transfer_action, writer_mask
        )
        with _writer_context(model), torch.no_grad():
            zero_output = model.rank_from_cache(cache, beta_logit=torch.zeros((1, 7), dtype=dtype, device=device))
            action_output = model.rank_from_cache(cache, beta_logit=solved.transfer_action)
            zero_loss = prediction_loss(
                zero_output,
                labels,
                oob,
                pair_loss_weight=pair_loss_weight,
                pair_count=int(solver_kwargs.get("pair_count", 4096)),
                pair_min_rank_gap=float(solver_kwargs.get("pair_min_rank_gap", 0.10)),
                namespace=f"{namespace}|bootstrap|{repeat}|oob",
            )
            action_loss = prediction_loss(
                action_output,
                labels,
                oob,
                pair_loss_weight=pair_loss_weight,
                pair_count=int(solver_kwargs.get("pair_count", 4096)),
                pair_min_rank_gap=float(solver_kwargs.get("pair_min_rank_gap", 0.10)),
                namespace=f"{namespace}|bootstrap|{repeat}|oob",
            )
            gains[repeat] = (zero_loss - action_loss) / (
                zero_loss.abs() + oob_denominator_epsilon
            )
        repeat_valid[repeat] = True
    valid_count = int(repeat_valid.sum())
    action_mask = torch.zeros(7, dtype=torch.bool, device=device)
    variance = torch.zeros(7, dtype=dtype, device=device)
    diagnostic_variance = torch.zeros(7, dtype=dtype, device=device)
    sign_count = torch.zeros(7, dtype=torch.int64, device=device)
    sign_rate = torch.zeros(7, dtype=dtype, device=device)
    if valid_count >= min_nonempty_oob_repeats and full.reason == "VALID":
        valid_actions = reliability_shifts[repeat_valid]
        required_sign = math.ceil(sign_agreement_min * bootstrap_repeats)
        if len(valid_actions) >= 2:
            diagnostic_variance = valid_actions.var(dim=0, unbiased=True)
        for group in range(7):
            full_value = full_reliability_shift[group]
            same_sign = torch.sign(valid_actions[:, group]).eq(torch.sign(full_value))
            sign_count[group] = int(same_sign.sum())
            sign_rate[group] = sign_count[group].to(dtype) / max(valid_count, 1)
            stable = int(sign_count[group]) >= required_sign
            supported = int(group_stock_support[group]) >= action_coordinate_min_stocks
            # The frozen epsilon is defined on the Writer's transfer-logit
            # action.  The persisted coordinate is A^rho, but changing the
            # epsilon's domain here would silently tighten the canonical W0.
            nonzero = (
                abs(float(full.transfer_action[0, group]))
                > action_nonzero_epsilon
            )
            if stable and supported and nonzero and len(valid_actions) >= 2:
                action_mask[group] = True
                variance[group] = diagnostic_variance[group]
    filtered_logit_action = full.transfer_action[0].clone()
    filtered_logit_action[~action_mask] = 0.0
    filtered_reliability_shift = full_reliability_shift.clone()
    filtered_reliability_shift[~action_mask] = 0.0
    finite_gain = gains[repeat_valid]
    if len(finite_gain):
        gain_median = float(finite_gain.median())
        gain_mad = float((finite_gain - finite_gain.median()).abs().median())
    else:
        gain_median = math.nan
        gain_mad = math.nan
    memory_eligible = bool(
        valid_count >= min_nonempty_oob_repeats
        and np.isfinite(gain_median)
        and gain_median > 0.0
        and action_mask.any()
    )
    if full.reason != "VALID":
        reason = full.reason
    elif valid_count < min_nonempty_oob_repeats:
        reason = "WRITER_INVALID_OOB"
    elif not np.isfinite(gain_median) or gain_median <= 0.0:
        reason = "WRITER_INVALID_NO_IMPROVEMENT"
    elif not action_mask.any():
        reason = "WRITER_LOW_CONFIDENCE"
    else:
        reason = "VALID"
    group_reasons: list[str] = []
    required_sign = math.ceil(sign_agreement_min * bootstrap_repeats)
    for group in range(7):
        if int(group_stock_support[group]) < action_coordinate_min_stocks:
            group_reasons.append("LOW_STOCK_SUPPORT")
        elif full.zero_action_winner:
            group_reasons.append("FULL_SOLVE_ZERO_ACTION_WINNER")
        elif valid_count < min_nonempty_oob_repeats:
            group_reasons.append("INSUFFICIENT_OOB_FOLDS")
        elif (
            abs(float(full.transfer_action[0, group]))
            <= action_nonzero_epsilon
        ):
            group_reasons.append("TRANSFER_BELOW_0_005")
        elif int(sign_count[group]) < required_sign:
            group_reasons.append("BOOTSTRAP_SIGN_UNSTABLE")
        elif bool(action_mask[group]):
            group_reasons.append("MASK_PASS")
        else:  # Defensive: every W0 condition above must explain the mask.
            raise RuntimeError("Writer action mask lacks a unique W0 reason")
    return WriterBootstrapResult(
        full_solve_raw_action=full.full_solve_raw_action[0].detach(),
        raw_action=full.raw_action[0].detach(),
        pre_mask_transfer_action=full.transfer_action[0].detach(),
        transfer_action=filtered_logit_action.detach(),
        pre_mask_reliability_shift=full_reliability_shift.detach(),
        action_reliability_shift=filtered_reliability_shift.detach(),
        action_reliability_variance=variance.detach(),
        action_reliability_mask=action_mask.detach(),
        bootstrap_actions=actions.detach(),
        bootstrap_reliability_shifts=reliability_shifts.detach(),
        bootstrap_valid_mask=repeat_valid.detach(),
        bootstrap_positive_gain_mask=(repeat_valid & (gains > 0)).detach(),
        bootstrap_sign_count=sign_count.detach(),
        bootstrap_sign_rate=sign_rate.detach(),
        bootstrap_reliability_variance=diagnostic_variance.detach(),
        bootstrap_solve_reason=tuple(solve_reasons),
        oob_relative_gains=gains.detach(),
        oob_relative_gain_median=gain_median,
        oob_relative_gain_mad=gain_mad,
        oob_valid_repeat_count=valid_count,
        writer_stock_count=int(writer_mask.sum()),
        group_failure_reason=tuple(group_reasons),
        zero_action_winner=full.zero_action_winner,
        action_cap=float(action_cap),
        reason=reason,
        memory_eligible=memory_eligible,
    )
