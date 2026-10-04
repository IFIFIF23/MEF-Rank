"""K-aware Direction Writer, R1N nested dose calibration, and outer OOB audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ..inverse import InverseResult, invert_reliability_shift
from ..writer.pgd import _detached_cache, _mean_reliability_shift, _writer_context
from ..v36.label_isolation import subset_average_tie_rank
from .contracts import V37WriterConfig
from .metrics import (
    LambdaPairRegistry,
    TopKMetrics,
    array_hash,
    lambda_ndcg_pairs,
    local_k,
    ranking_movement,
    topk_metrics,
)


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


def _seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _stable_tercile(values: ArrayF, valid: ArrayI) -> ArrayI:
    selected = np.asarray(values, dtype=np.float64)[valid]
    if not np.isfinite(selected).all():
        raise ValueError("PIT split variables must be finite on Writer stocks")
    order = np.lexsort((valid, selected))
    bins = np.empty(len(valid), dtype=np.int64)
    bins[order] = np.minimum(2, np.floor(3 * np.arange(len(valid)) / len(valid))).astype(
        np.int64
    )
    result = np.full(len(values), -1, dtype=np.int64)
    result[valid] = bins
    return result


@dataclass(frozen=True)
class R1NSplit:
    direction: ArrayI
    magnitude: ArrayI
    outer_oob: ArrayI
    split_hash: str


def r1n_split(
    valid: ArrayI,
    cap_proxy: ArrayF,
    liquidity_proxy: ArrayF,
    *,
    namespace: str,
    config: V37WriterConfig,
) -> R1NSplit:
    valid = np.asarray(valid, dtype=np.int64)
    cap = _stable_tercile(cap_proxy, valid)
    liquidity = _stable_tercile(liquidity_proxy, valid)
    strata = 3 * cap + liquidity
    rng = np.random.default_rng(_seed(namespace))
    direction_parts: list[ArrayI] = []
    magnitude_parts: list[ArrayI] = []
    outer_parts: list[ArrayI] = []
    for value in sorted(set(strata[valid].tolist())):
        members = rng.permutation(valid[strata[valid] == value]).astype(np.int64)
        if len(members) < 3:
            raise ValueError("WRITER_INSUFFICIENT_SPLIT_SUPPORT")
        development_count = int(
            np.clip(
                round(config.outer_development_fraction * len(members)),
                2,
                len(members) - 1,
            )
        )
        development = members[:development_count]
        outer_parts.append(members[development_count:])
        development = rng.permutation(development).astype(np.int64)
        direction_count = int(
            np.clip(
                round(config.inner_direction_fraction * len(development)),
                1,
                len(development) - 1,
            )
        )
        direction_parts.append(development[:direction_count])
        magnitude_parts.append(development[direction_count:])
    direction = np.sort(np.concatenate(direction_parts)).astype(np.int64)
    magnitude = np.sort(np.concatenate(magnitude_parts)).astype(np.int64)
    outer = np.sort(np.concatenate(outer_parts)).astype(np.int64)
    if (
        len(direction) < config.direction_minimum_stocks
        or len(magnitude) < config.magnitude_minimum_stocks
        or len(outer) < config.outer_minimum_stocks
    ):
        raise ValueError("WRITER_INSUFFICIENT_SPLIT_SUPPORT")
    if (
        np.intersect1d(direction, magnitude).size
        or np.intersect1d(direction, outer).size
        or np.intersect1d(magnitude, outer).size
        or not np.array_equal(
            np.sort(np.concatenate([direction, magnitude, outer])), np.sort(valid)
        )
    ):
        raise AssertionError("R1N is not a disjoint exhaustive partition")
    payload = {
        "namespace": namespace,
        "direction": direction.tolist(),
        "magnitude": magnitude.tolist(),
        "outer_oob": outer.tolist(),
    }
    return R1NSplit(direction, magnitude, outer, _hash(payload))


@dataclass(frozen=True)
class DirectionResult:
    transfer_logit: ArrayF
    reliability_shift: ArrayF
    pair_registry: LambdaPairRegistry
    zero_loss: float
    final_loss: float
    accepted_steps: int
    reason_code: str


def solve_k_direction(
    model: Any,
    cache: Any,
    first: Any,
    raw_returns: ArrayF,
    indices: ArrayI,
    stock_ids: NDArray[np.str_],
    writer_mask: Tensor,
    *,
    full_stock_count: int,
    namespace: str,
    config: V37WriterConfig,
) -> DirectionResult:
    indices = np.asarray(indices, dtype=np.int64)
    returns = np.asarray(raw_returns, dtype=np.float64)[indices]
    ranks = subset_average_tie_rank(returns)
    k = local_k(config.selection_k, len(indices), full_stock_count)
    base_scores = first.score[0].detach().cpu().numpy().astype(np.float64)[indices]
    registry = lambda_ndcg_pairs(
        base_scores,
        ranks,
        indices,
        np.asarray(stock_ids)[indices],
        k=k,
        maximum_pairs=config.pair_count,
        namespace=f"{namespace}|direction-pairs",
    )
    if not len(registry.pairs):
        zero = np.zeros(7, dtype=np.float64)
        return DirectionResult(
            zero,
            zero,
            registry,
            float("nan"),
            float("nan"),
            0,
            "WRITER_DIRECTION_PAIR_REGISTRY_INVALID",
        )
    device = cache.base_gate_logits.device
    dtype = cache.base_gate_logits.dtype
    pairs = torch.as_tensor(registry.pairs, dtype=torch.long, device=device)
    weights = torch.as_tensor(registry.weights, dtype=dtype, device=device)
    detached = _detached_cache(cache)

    def prediction_loss(beta: Tensor) -> Tensor:
        output = model.rank_from_cache(detached, beta_logit=beta)
        difference = output.score[0, pairs[:, 0]] - output.score[0, pairs[:, 1]]
        return (F.softplus(-difference) * weights).sum() / weights.sum()

    with _writer_context(model):
        zero = torch.zeros((1, 7), dtype=dtype, device=device)
        with torch.no_grad():
            zero_loss = prediction_loss(zero)
        delta = zero.clone().requires_grad_(True)
        accepted = 0
        for _ in range(config.steps):
            loss = prediction_loss(delta)
            objective = loss + config.action_l2 * delta.square().sum()
            (gradient,) = torch.autograd.grad(objective, delta)
            chosen: Tensor | None = None
            for trial in range(config.backtracking_trials):
                step = config.initial_step_size * config.backtracking_factor**trial
                candidate = (delta - step * gradient).clamp(
                    -config.raw_logit_cap, config.raw_logit_cap
                )
                with torch.no_grad():
                    candidate_objective = prediction_loss(candidate) + (
                        config.action_l2 * candidate.square().sum()
                    )
                if float(candidate_objective) < float(objective.detach()):
                    chosen = candidate.detach()
                    break
            if chosen is not None:
                delta = chosen.requires_grad_(True)
                accepted += 1
            else:
                delta = delta.detach().requires_grad_(True)
        transfer = delta.detach().clamp(
            -config.transfer_logit_cap, config.transfer_logit_cap
        )
        with torch.no_grad():
            final_loss_tensor = prediction_loss(transfer)
            final_objective = final_loss_tensor + config.action_l2 * transfer.square().sum()
        if float(final_objective) >= float(zero_loss):
            transfer = zero
            reason = "WRITER_DIRECTION_ZERO"
            final_loss_tensor = zero_loss
        else:
            reason = "WRITER_DIRECTION_VALID"
    shift, _ = _mean_reliability_shift(cache, transfer, writer_mask)
    return DirectionResult(
        transfer.detach().cpu().numpy().astype(np.float64).reshape(7),
        shift.detach().cpu().numpy().astype(np.float64).reshape(7),
        registry,
        float(zero_loss),
        float(final_loss_tensor),
        accepted,
        reason,
    )


def maximum_no_clip_dose(
    cache: Any,
    unit_shift: ArrayF,
    *,
    config: V37WriterConfig,
) -> float:
    unit = np.asarray(unit_shift, dtype=np.float64).reshape(7)
    logits = cache.base_gate_logits[0].detach().cpu().numpy().astype(np.float64)
    mask = cache.group_available_mask[0].detach().cpu().numpy().astype(bool)
    bound = config.inverse_cap - config.inverse_margin
    candidates: list[float] = []
    for group in range(7):
        if abs(unit[group]) <= 1.0e-12:
            continue
        supported = logits[mask[:, group], group]
        if not len(supported):
            return 0.0
        base = 1.0 / (1.0 + np.exp(-supported))
        moved = 1.0 / (1.0 + np.exp(-(supported + np.sign(unit[group]) * bound)))
        reachable = float(np.mean(moved - base))
        candidates.append(reachable / unit[group])
    if not candidates:
        return 0.0
    value = float(min(candidates))
    return value if np.isfinite(value) and value > 0 else 0.0


@dataclass(frozen=True)
class ActionEvaluation:
    action: ArrayF
    dose: float
    dose_fraction: float
    equivalent_multiplier: float
    inverse: InverseResult | None
    feasible: bool
    score: ArrayF
    metrics: TopKMetrics
    deltas: dict[str, float]
    movement: dict[str, float]


def evaluate_shift(
    model: Any,
    cache: Any,
    first: Any,
    action: ArrayF,
    raw_returns: ArrayF,
    indices: ArrayI,
    stock_ids: NDArray[np.str_],
    *,
    full_stock_count: int,
    base_metrics: TopKMetrics,
    dose_fraction: float,
    raw_shift_norm: float,
    config: V37WriterConfig,
) -> ActionEvaluation:
    action = np.asarray(action, dtype=np.float64).reshape(7)
    dose = float(np.linalg.norm(action))
    if dose <= 1.0e-12:
        score = first.score[0].detach().cpu().numpy().astype(np.float64)
        return ActionEvaluation(
            action,
            0.0,
            dose_fraction,
            0.0,
            None,
            True,
            score,
            base_metrics,
            base_metrics.delta(base_metrics),
            {
                "mean_abs_rank_move": 0.0,
                "median_abs_rank_move": 0.0,
                "maximum_abs_rank_move": 0.0,
                "top10_member_change_fraction": 0.0,
                "top20_member_change_fraction": 0.0,
            },
        )
    inverse = invert_reliability_shift(
        cache.base_gate_logits[0].detach().cpu().numpy().astype(np.float64),
        cache.group_available_mask[0].detach().cpu().numpy().astype(bool),
        action,
        action_cap=config.inverse_cap,
        iterations=config.inverse_iterations,
        tolerance=config.inverse_tolerance,
    )
    residual = float(np.max(np.abs(inverse.solver_residual)))
    feasible = bool(
        inverse.converged.all()
        and not inverse.per_group_clipped.any()
        and residual <= config.inverse_tolerance
        and np.max(np.abs(inverse.delta_beta))
        <= config.inverse_cap - config.inverse_margin + 2.0e-6
    )
    if feasible:
        with torch.no_grad():
            final = model.rank_from_cache(
                cache,
                beta_logit=torch.as_tensor(
                    np.array(inverse.delta_beta[None], copy=True),
                    dtype=cache.base_gate_logits.dtype,
                    device=cache.base_gate_logits.device,
                ),
            )
        score = final.score[0].detach().cpu().numpy().astype(np.float64)
    else:
        score = first.score[0].detach().cpu().numpy().astype(np.float64)
    subset = np.asarray(indices, dtype=np.int64)
    k = local_k(config.selection_k, len(subset), full_stock_count)
    metrics = topk_metrics(
        score[subset],
        np.asarray(raw_returns, dtype=np.float64)[subset],
        np.asarray(stock_ids)[subset],
        k=k,
    )
    base_score = first.score[0].detach().cpu().numpy().astype(np.float64)
    movement = ranking_movement(
        base_score[subset], score[subset], np.asarray(stock_ids)[subset]
    )
    return ActionEvaluation(
        action,
        dose,
        dose_fraction,
        dose / (raw_shift_norm + 1.0e-12),
        inverse,
        feasible,
        score,
        metrics,
        metrics.delta(base_metrics),
        movement,
    )


def dose_grid(
    unit: ArrayF,
    maximum_dose: float,
    raw_shift_norm: float,
    *,
    config: V37WriterConfig,
) -> list[tuple[float, ArrayF]]:
    values = [(fraction, fraction * maximum_dose * unit) for fraction in config.dose_fractions]
    if 1.0e-12 < raw_shift_norm <= maximum_dose + 1.0e-12:
        fraction = raw_shift_norm / maximum_dose
        if all(abs(fraction - existing) > 1.0e-10 for existing, _ in values):
            values.append((float(fraction), raw_shift_norm * unit))
    return sorted(values, key=lambda item: item[0])


def select_safe_dose(evaluations: list[ActionEvaluation], ratio: float) -> ActionEvaluation:
    eligible = [
        row
        for row in evaluations
        if row.feasible
        and row.deltas["primary_gain"] > 0
        and row.deltas["rank_ic_delta"] >= -1.0e-12
        and row.deltas["precision_delta"] >= -1.0e-12
        and row.deltas["top_k_mean_return_delta"] >= -1.0e-12
    ]
    if not eligible:
        return min(evaluations, key=lambda row: row.dose)
    best_gain = max(row.deltas["primary_gain"] for row in eligible)
    near = [row for row in eligible if row.deltas["primary_gain"] >= ratio * best_gain]
    return sorted(
        near,
        key=lambda row: (
            row.dose,
            -row.deltas["precision_delta"],
            -row.deltas["spread_delta"],
            row.equivalent_multiplier,
        ),
    )[0]


@dataclass(frozen=True)
class RepeatResult:
    repeat: int
    valid: bool
    reason_code: str
    split_hash: str
    pair_registry_hash: str
    direction_count: int
    magnitude_count: int
    outer_count: int
    raw_transfer_logit: ArrayF
    raw_reliability_shift: ArrayF
    selected_action: ArrayF
    selected_dose: float
    selected_dose_fraction: float
    equivalent_multiplier: float
    maximum_no_clip_dose: float
    magnitude_deltas: dict[str, float]
    outer_deltas: dict[str, float]
    outer_movement: dict[str, float]
    dose_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class NestedWriterResult:
    qualified: bool
    reason_code: str
    writer_stock_count: int
    valid_repeat_count: int
    repeats: tuple[RepeatResult, ...]
    full_raw_transfer_logit: ArrayF
    full_raw_reliability_shift: ArrayF
    full_direction_unit: ArrayF
    repeat_calibrated_shift_mean: ArrayF
    repeat_consensus_direction: ArrayF
    repeat_projected_magnitude_median: float
    repeat_projected_magnitude_iqr: float
    repeat_negative_projection_count: int
    repeat_selected_dose_fraction_median: float
    repeat_selected_dose_fraction_iqr: float
    repeat_equivalent_multiplier_median: float
    repeat_equivalent_multiplier_iqr: float
    repeat_zero_dose_count: int
    final_magnitude: float
    full_maximum_no_clip_dose: float
    calibrated_equivalent_multiplier: float
    final_action: ArrayF
    final_transfer_logit: ArrayF
    covariance: ArrayF
    full_repeat_cosine: float
    outer_summary: dict[str, float]


def _covariance(actions: ArrayF) -> ArrayF:
    matrix = np.asarray(actions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 7 or len(matrix) < 2:
        return np.eye(7, dtype=np.float64) * 1.0e-12
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    empirical = centered.T @ centered / len(matrix)
    target = np.eye(7) * max(float(np.trace(empirical) / 7.0), 1.0e-12)
    shrinkage = min(1.0, 7.0 / len(matrix))
    result = (1.0 - shrinkage) * empirical + shrinkage * target
    result = (result + result.T) / 2.0
    values, vectors = np.linalg.eigh(result)
    repaired = (vectors * np.maximum(values, 1.0e-12)) @ vectors.T
    return (repaired + repaired.T) / 2.0


def _cosine(left: ArrayF, right: ArrayF) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right) + 1.0e-12)
    return float(np.dot(left, right) / denominator)


def solve_nested_writer(
    model: Any,
    cache: Any,
    first: Any,
    *,
    raw_returns: ArrayF,
    writer_mask: Tensor,
    cap_proxy: ArrayF,
    liquidity_proxy: ArrayF,
    stock_ids: NDArray[np.str_],
    namespace: str,
    config: V37WriterConfig = V37WriterConfig(),
) -> NestedWriterResult:
    valid = (
        torch.nonzero(writer_mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    returns = np.asarray(raw_returns, dtype=np.float64)
    if len(valid) < config.minimum_stocks or returns.shape != (writer_mask.numel(),):
        raise ValueError("WRITER_INSUFFICIENT_STOCKS")
    full_count = len(valid)
    base_score = first.score[0].detach().cpu().numpy().astype(np.float64)
    repeats: list[RepeatResult] = []
    for repeat in range(config.repeats):
        repeat_namespace = f"{namespace}|R1N|repeat={repeat}"
        try:
            split = r1n_split(
                valid,
                cap_proxy,
                liquidity_proxy,
                namespace=repeat_namespace,
                config=config,
            )
            direction = solve_k_direction(
                model,
                cache,
                first,
                returns,
                split.direction,
                stock_ids,
                writer_mask,
                full_stock_count=full_count,
                namespace=repeat_namespace,
                config=config,
            )
            raw_norm = float(np.linalg.norm(direction.reliability_shift))
            unit = (
                direction.reliability_shift / raw_norm
                if raw_norm > 1.0e-12
                else np.zeros(7, dtype=np.float64)
            )
            maximum = maximum_no_clip_dose(cache, unit, config=config)
            magnitude_k = local_k(
                config.selection_k, len(split.magnitude), full_count
            )
            magnitude_base = topk_metrics(
                base_score[split.magnitude],
                returns[split.magnitude],
                np.asarray(stock_ids)[split.magnitude],
                k=magnitude_k,
            )
            evaluations = [
                evaluate_shift(
                    model,
                    cache,
                    first,
                    action,
                    returns,
                    split.magnitude,
                    stock_ids,
                    full_stock_count=full_count,
                    base_metrics=magnitude_base,
                    dose_fraction=fraction,
                    raw_shift_norm=raw_norm,
                    config=config,
                )
                for fraction, action in dose_grid(
                    unit, maximum, raw_norm, config=config
                )
            ]
            selected = select_safe_dose(evaluations, config.near_optimal_ratio)
            outer_k = local_k(config.selection_k, len(split.outer_oob), full_count)
            outer_base = topk_metrics(
                base_score[split.outer_oob],
                returns[split.outer_oob],
                np.asarray(stock_ids)[split.outer_oob],
                k=outer_k,
            )
            outer = evaluate_shift(
                model,
                cache,
                first,
                selected.action,
                returns,
                split.outer_oob,
                stock_ids,
                full_stock_count=full_count,
                base_metrics=outer_base,
                dose_fraction=selected.dose_fraction,
                raw_shift_norm=raw_norm,
                config=config,
            )
            dose_rows = tuple(
                {
                    "dose_fraction": row.dose_fraction,
                    "dose": row.dose,
                    "equivalent_multiplier": row.equivalent_multiplier,
                    "feasible": row.feasible,
                    **row.deltas,
                    **row.movement,
                }
                for row in evaluations
            )
            repeats.append(
                RepeatResult(
                    repeat,
                    True,
                    "WRITER_REPEAT_VALID",
                    split.split_hash,
                    direction.pair_registry.registry_hash,
                    len(split.direction),
                    len(split.magnitude),
                    len(split.outer_oob),
                    direction.transfer_logit,
                    direction.reliability_shift,
                    selected.action,
                    selected.dose,
                    selected.dose_fraction,
                    selected.equivalent_multiplier,
                    maximum,
                    selected.deltas,
                    outer.deltas,
                    outer.movement,
                    dose_rows,
                )
            )
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            zero = np.zeros(7, dtype=np.float64)
            repeats.append(
                RepeatResult(
                    repeat,
                    False,
                    str(exc),
                    "",
                    "",
                    0,
                    0,
                    0,
                    zero,
                    zero,
                    zero,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    {},
                    {},
                    {},
                    (),
                )
            )
    valid_repeats = [row for row in repeats if row.valid]
    actions = (
        np.stack([row.selected_action for row in valid_repeats])
        if valid_repeats
        else np.zeros((0, 7), dtype=np.float64)
    )
    mean_action = actions.mean(axis=0) if len(actions) else np.zeros(7)
    consensus_norm = float(np.linalg.norm(mean_action))
    consensus = mean_action / (consensus_norm + 1.0e-12)
    projections = (
        np.maximum(0.0, actions @ consensus) if len(actions) else np.zeros(0)
    )
    negative_count = (
        int((actions @ consensus < 0).sum()) if len(actions) else 0
    )
    median_magnitude = float(np.median(projections)) if len(projections) else 0.0
    iqr = (
        float(np.quantile(projections, 0.75) - np.quantile(projections, 0.25))
        if len(projections)
        else 0.0
    )
    full = solve_k_direction(
        model,
        cache,
        first,
        returns,
        valid,
        stock_ids,
        writer_mask,
        full_stock_count=full_count,
        namespace=f"{namespace}|full-direction",
        config=config,
    )
    full_raw_norm = float(np.linalg.norm(full.reliability_shift))
    full_unit = (
        full.reliability_shift / full_raw_norm
        if full_raw_norm > 1.0e-12
        else np.zeros(7, dtype=np.float64)
    )
    cosine = _cosine(full_unit, consensus)
    full_maximum = maximum_no_clip_dose(cache, full_unit, config=config)
    final_magnitude = min(median_magnitude, full_maximum)
    final_action = final_magnitude * full_unit
    final_inverse = invert_reliability_shift(
        cache.base_gate_logits[0].detach().cpu().numpy().astype(np.float64),
        cache.group_available_mask[0].detach().cpu().numpy().astype(bool),
        final_action,
        action_cap=config.inverse_cap,
        iterations=config.inverse_iterations,
        tolerance=config.inverse_tolerance,
    )
    inverse_valid = bool(
        final_magnitude <= 1.0e-12
        or (
            final_inverse.converged.all()
            and not final_inverse.per_group_clipped.any()
            and np.max(np.abs(final_inverse.solver_residual))
            <= config.inverse_tolerance
        )
    )
    outer_keys = (
        "primary_gain",
        "rank_ic_delta",
        "precision_delta",
        "top_k_mean_return_delta",
        "spread_delta",
    )
    outer_summary: dict[str, float] = {}
    for key in outer_keys:
        values = np.asarray(
            [row.outer_deltas.get(key, np.nan) for row in valid_repeats],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        outer_summary[f"{key}_mean"] = float(np.mean(values)) if len(values) else np.nan
        outer_summary[f"{key}_median"] = (
            float(np.median(values)) if len(values) else np.nan
        )
        outer_summary[f"{key}_positive_fraction"] = (
            float(np.mean(values > 0)) if len(values) else np.nan
        )
        outer_summary[f"{key}_positive_count"] = (
            int(np.sum(values > 0)) if len(values) else 0
        )
        outer_summary[f"{key}_mad"] = (
            float(np.median(np.abs(values - np.median(values))))
            if len(values)
            else np.nan
        )
    dose_fractions = np.asarray(
        [row.selected_dose_fraction for row in valid_repeats], dtype=np.float64
    )
    multipliers = np.asarray(
        [row.equivalent_multiplier for row in valid_repeats], dtype=np.float64
    )
    dose_fraction_median = (
        float(np.median(dose_fractions)) if len(dose_fractions) else 0.0
    )
    dose_fraction_iqr = (
        float(np.quantile(dose_fractions, 0.75) - np.quantile(dose_fractions, 0.25))
        if len(dose_fractions)
        else 0.0
    )
    multiplier_median = (
        float(np.median(multipliers)) if len(multipliers) else 0.0
    )
    multiplier_iqr = (
        float(np.quantile(multipliers, 0.75) - np.quantile(multipliers, 0.25))
        if len(multipliers)
        else 0.0
    )
    reason = "WRITER_VALID"
    if len(valid_repeats) < config.minimum_valid_repeats:
        reason = "WRITER_VALID_REPEAT_SHORTFALL"
    elif not all(
        np.isfinite(outer_summary[f"{key}_median"]) for key in outer_keys
    ):
        reason = "WRITER_OUTER_METRIC_NONFINITE"
    elif outer_summary["primary_gain_median"] <= 0:
        reason = "WRITER_NONPOSITIVE_MEDIAN_PRIMARY_GAIN"
    elif outer_summary["rank_ic_delta_median"] < 0:
        reason = "WRITER_GLOBAL_RANKIC_GUARD_FAILED"
    elif outer_summary["precision_delta_median"] < 0:
        reason = "WRITER_TOPK_PRECISION_GUARD_FAILED"
    elif outer_summary["top_k_mean_return_delta_median"] < 0:
        reason = "WRITER_TOPK_RETURN_GUARD_FAILED"
    elif full_raw_norm <= 1.0e-12:
        reason = "WRITER_FULL_DIRECTION_ZERO"
    elif median_magnitude <= 1.0e-12:
        reason = "WRITER_MAGNITUDE_NO_POSITIVE_CANDIDATE"
    elif cosine < config.full_repeat_cosine_minimum:
        reason = "WRITER_JOINT_DIRECTION_UNSTABLE"
    elif not inverse_valid:
        reason = "WRITER_CALIBRATED_ACTION_INVERSE_FAILED"
    final_transfer = np.asarray(final_inverse.delta_beta, dtype=np.float64)
    return NestedWriterResult(
        reason == "WRITER_VALID",
        reason,
        full_count,
        len(valid_repeats),
        tuple(repeats),
        full.transfer_logit,
        full.reliability_shift,
        full_unit,
        mean_action,
        consensus,
        median_magnitude,
        iqr,
        negative_count,
        dose_fraction_median,
        dose_fraction_iqr,
        multiplier_median,
        multiplier_iqr,
        int(np.sum(np.asarray([row.selected_dose for row in valid_repeats]) <= 1.0e-12)),
        final_magnitude,
        full_maximum,
        final_magnitude / (full_raw_norm + 1.0e-12),
        final_action,
        final_transfer,
        _covariance(actions),
        cosine,
        outer_summary,
    )
