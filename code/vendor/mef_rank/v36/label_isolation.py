"""Repeat-local H5 label isolation for the v3.6 Writer repair probe.

This module is deliberately narrow.  It does not change the Writer objective,
PGD controls, action cap, or the full-solve target.  It makes the train/OOB
label boundary physical: the train solver receives only a train-local rank
view and a frozen train pair registry, while OOB returns stay sealed until the
action has been frozen.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
from torch.nn import functional as F

from ..writer.pgd import (
    PredictionLossComponents,
    WriterSolveResult,
    _detached_cache,
    _normalized_weighted_mean,
    _writer_context,
    fixed_pair_indices,
    soft_ordinal_loss,
)


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    descriptor = (
        str(array.dtype).encode()
        + b"|"
        + json.dumps(array.shape).encode()
        + b"|"
        + array.tobytes()
    )
    return hashlib.sha256(descriptor).hexdigest()


def subset_average_tie_rank(raw_returns: NDArray[np.floating]) -> NDArray[np.float64]:
    """Rank finite returns inside one split, matching fixed_horizon semantics."""

    values = np.asarray(raw_returns, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("repeat-local raw returns must be one-dimensional")
    finite = np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    count = int(finite.sum())
    if count < 2:
        return result
    order = values[finite]
    sorter = np.argsort(order, kind="mergesort")
    sorted_values = order[sorter]
    ranks = np.empty(count, dtype=np.float64)
    start = 0
    while start < count:
        stop = start + 1
        while stop < count and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[sorter[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    result[finite] = (ranks - 1.0) / float(count - 1)
    return result


@dataclass(frozen=True)
class FrozenSplitRegistry:
    stock_count: int
    train_indices: NDArray[np.int64]
    train_weights: NDArray[np.float64]
    oob_indices: NDArray[np.int64]
    namespace: str
    registry_hash: str

    @classmethod
    def create(
        cls,
        *,
        stock_count: int,
        train_indices: NDArray[np.integer],
        train_weights: NDArray[np.floating],
        oob_indices: NDArray[np.integer],
        namespace: str,
    ) -> "FrozenSplitRegistry":
        train = np.asarray(train_indices, dtype=np.int64)
        weights = np.asarray(train_weights, dtype=np.float64)
        oob = np.asarray(oob_indices, dtype=np.int64)
        if len(train) != len(weights) or len(train) < 2 or len(oob) < 2:
            raise ValueError("split registry has invalid support")
        if np.intersect1d(train, oob).size or len(np.unique(train)) != len(train):
            raise ValueError("split registry violates disjoint unique stock nodes")
        if (
            train.min(initial=0) < 0
            or oob.min(initial=0) < 0
            or train.max(initial=-1) >= stock_count
            or oob.max(initial=-1) >= stock_count
        ):
            raise ValueError("split registry index escapes the stock axis")
        payload = {
            "stock_count": int(stock_count),
            "train_indices": train.tolist(),
            "train_weights": weights.tolist(),
            "oob_indices": oob.tolist(),
            "namespace": namespace,
        }
        return cls(
            int(stock_count), train, weights, oob, namespace, _hash(payload)
        )


@dataclass(frozen=True)
class LocalLabelView:
    side: str
    indices: NDArray[np.int64]
    ranks: NDArray[np.float64]
    raw_return_hash: str
    rank_hash: str

    @classmethod
    def materialize(
        cls,
        side: str,
        indices: NDArray[np.int64],
        raw_returns: NDArray[np.floating],
    ) -> "LocalLabelView":
        index = np.asarray(indices, dtype=np.int64)
        raw = np.asarray(raw_returns, dtype=np.float64)
        if side not in {"train", "oob"} or raw.shape != index.shape:
            raise ValueError("local label view is malformed")
        ranks = subset_average_tie_rank(raw)
        if int(np.isfinite(ranks).sum()) < 2:
            raise ValueError(f"{side} local rank has insufficient finite support")
        return cls(side, index, ranks, _array_hash(raw), _array_hash(ranks))

    def dense(self, stock_count: int) -> NDArray[np.float64]:
        values = np.full(stock_count, np.nan, dtype=np.float64)
        values[self.indices] = self.ranks
        return values


@dataclass(frozen=True)
class FrozenPairRegistry:
    side: str
    pairs: NDArray[np.int64]
    directions: NDArray[np.float64]
    namespace: str
    registry_hash: str

    @classmethod
    def create(
        cls,
        view: LocalLabelView,
        *,
        stock_count: int,
        minimum_rank_gap: float,
        maximum_pairs: int,
        namespace: str,
    ) -> "FrozenPairRegistry":
        dense = torch.as_tensor(view.dense(stock_count), dtype=torch.float64)
        indices = torch.as_tensor(view.indices, dtype=torch.long)
        pairs = (
            fixed_pair_indices(
                dense,
                indices,
                minimum_rank_gap=minimum_rank_gap,
                maximum_pairs=maximum_pairs,
                namespace=namespace,
            )
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        direction = np.sign(
            view.dense(stock_count)[pairs[:, 0]]
            - view.dense(stock_count)[pairs[:, 1]]
        ).astype(np.float64)
        if np.any(direction == 0) or np.any(~np.isfinite(direction)):
            raise ValueError("pair registry contains invalid directions")
        allowed = set(view.indices.tolist())
        if any(int(left) not in allowed or int(right) not in allowed for left, right in pairs):
            raise ValueError("pair registry crosses the frozen split")
        payload = {
            "side": view.side,
            "pairs": pairs.tolist(),
            "directions": direction.tolist(),
            "namespace": namespace,
            "label_rank_hash": view.rank_hash,
        }
        return cls(view.side, pairs, direction, namespace, _hash(payload))


@dataclass(frozen=True)
class FrozenActionToken:
    action: NDArray[np.float64]
    action_hash: str
    split_hash: str
    train_label_hash: str
    train_pair_hash: str


class RepeatState(str, Enum):
    SPLIT_FROZEN = "SPLIT_FROZEN"
    TRAIN_LABEL_VIEW_MATERIALIZED = "TRAIN_LABEL_VIEW_MATERIALIZED"
    TRAIN_PAIR_REGISTRY_FROZEN = "TRAIN_PAIR_REGISTRY_FROZEN"
    RAW_ACTION_SOLVED = "RAW_ACTION_SOLVED"
    ACTION_FROZEN = "ACTION_FROZEN"
    OOB_LABEL_VIEW_OPENED = "OOB_LABEL_VIEW_OPENED"
    OOB_PAIR_REGISTRY_FROZEN = "OOB_PAIR_REGISTRY_FROZEN"
    OOB_EVALUATED = "OOB_EVALUATED"


class SplitRawReturnVault:
    """Separate return compartments; OOB has a cryptographic action barrier."""

    def __init__(
        self,
        split: FrozenSplitRegistry,
        full_raw_returns: NDArray[np.floating],
    ) -> None:
        raw = np.asarray(full_raw_returns, dtype=np.float64)
        if raw.shape != (split.stock_count,):
            raise ValueError("raw return source does not align with stock axis")
        self._split = split
        self._train = raw[split.train_indices].copy()
        self._oob = raw[split.oob_indices].copy()
        self._oob_opened = False

    def train(self) -> NDArray[np.float64]:
        return self._train.copy()

    def oob(self, token: FrozenActionToken) -> NDArray[np.float64]:
        if token.split_hash != self._split.registry_hash:
            raise PermissionError("action token does not bind the frozen split")
        if not token.action_hash or not token.train_label_hash or not token.train_pair_hash:
            raise PermissionError("action token is incomplete")
        self._oob_opened = True
        return self._oob.copy()

    @property
    def oob_opened(self) -> bool:
        return self._oob_opened


def loss_with_frozen_pairs(
    output: Any,
    view: LocalLabelView,
    pairs: FrozenPairRegistry,
    *,
    sample_weights: Tensor | None,
    pair_loss_weight: float,
) -> PredictionLossComponents:
    if pairs.side != view.side:
        raise PermissionError("label view and pair registry side differ")
    device = output.score.device
    dtype = output.score.dtype
    indices = torch.as_tensor(view.indices, dtype=torch.long, device=device)
    ranks = torch.as_tensor(view.ranks, dtype=dtype, device=device)
    ordinal = soft_ordinal_loss(output.rank_logits[0, indices], ranks, sample_weights)
    pair_index = torch.as_tensor(pairs.pairs, dtype=torch.long, device=device)
    direction = torch.as_tensor(pairs.directions, dtype=dtype, device=device)
    if pair_index.numel():
        per_pair = F.softplus(
            -direction
            * (
                output.score[0, pair_index[:, 0]]
                - output.score[0, pair_index[:, 1]]
            )
        )
        pair_weights = None
        if sample_weights is not None:
            stock_weights = torch.zeros(
                output.score.shape[1], dtype=dtype, device=device
            )
            stock_weights[indices] = sample_weights
            pair_weights = (
                stock_weights[pair_index[:, 0]] * stock_weights[pair_index[:, 1]]
            )
        pair = _normalized_weighted_mean(per_pair, pair_weights)
    else:
        pair = output.score.sum() * 0.0
    return PredictionLossComponents(
        ordinal, pair, ordinal + pair_loss_weight * pair, int(len(pairs.pairs))
    )


@dataclass(frozen=True)
class IsolatedSolveResult:
    writer: WriterSolveResult
    initial_gradient: NDArray[np.float64]
    train_zero_components: tuple[float, float, float]
    train_action_components: tuple[float, float, float]


def solve_isolated_train_action(
    model: Any,
    cache: Any,
    view: LocalLabelView,
    pairs: FrozenPairRegistry,
    *,
    sample_weights: NDArray[np.floating],
    pair_loss_weight: float = 1.0,
    transfer_logit_cap: float = 0.50,
    action_l2: float = 0.05,
    steps: int = 5,
    initial_step_size: float = 0.15,
    backtracking_trials: int = 4,
    backtracking_factor: float = 0.5,
    raw_logit_cap: float = 0.75,
) -> IsolatedSolveResult:
    """Solve from a train-local view; no full label or raw-return object enters."""

    if view.side != "train" or pairs.side != "train":
        raise PermissionError("repeat solve accepts only frozen train-local artifacts")
    if min(
        pair_loss_weight,
        action_l2,
        initial_step_size,
        backtracking_factor,
        raw_logit_cap,
        transfer_logit_cap,
    ) < 0 or steps <= 0 or backtracking_trials <= 0:
        raise ValueError("Writer solver controls are invalid")
    cache = _detached_cache(cache)
    dtype = cache.base_gate_logits.dtype
    device = cache.base_gate_logits.device
    weights = torch.as_tensor(sample_weights, dtype=dtype, device=device)
    if weights.shape != (len(view.indices),):
        raise ValueError("train multiplicity weights do not align")
    with _writer_context(model):
        zero = torch.zeros((1, 7), dtype=dtype, device=device)
        with torch.no_grad():
            zero_parts = loss_with_frozen_pairs(
                model.rank_from_cache(cache, beta_logit=zero),
                view,
                pairs,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
            )
        delta = zero.clone().requires_grad_(True)
        initial_gradient: Tensor | None = None
        accepted = 0
        for _ in range(steps):
            parts = loss_with_frozen_pairs(
                model.rank_from_cache(cache, beta_logit=delta),
                view,
                pairs,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
            )
            objective = parts.combined + action_l2 * delta.square().sum()
            (gradient,) = torch.autograd.grad(objective, delta, create_graph=False)
            if initial_gradient is None:
                initial_gradient = gradient.detach().clone()
            accepted_candidate: Tensor | None = None
            for trial in range(backtracking_trials):
                step_size = initial_step_size * (backtracking_factor**trial)
                candidate = (delta - step_size * gradient).clamp(
                    -raw_logit_cap, raw_logit_cap
                )
                with torch.no_grad():
                    candidate_parts = loss_with_frozen_pairs(
                        model.rank_from_cache(cache, beta_logit=candidate),
                        view,
                        pairs,
                        sample_weights=weights,
                        pair_loss_weight=pair_loss_weight,
                    )
                    candidate_objective = (
                        candidate_parts.combined + action_l2 * candidate.square().sum()
                    )
                if float(candidate_objective) < float(objective.detach()):
                    accepted_candidate = candidate.detach()
                    break
            if accepted_candidate is None:
                delta = delta.detach().requires_grad_(True)
                continue
            delta = accepted_candidate.requires_grad_(True)
            accepted += 1
        raw_candidate = delta.detach()
        transfer_candidate = raw_candidate.clamp(
            -transfer_logit_cap, transfer_logit_cap
        )
        with torch.no_grad():
            action_parts = loss_with_frozen_pairs(
                model.rank_from_cache(cache, beta_logit=transfer_candidate),
                view,
                pairs,
                sample_weights=weights,
                pair_loss_weight=pair_loss_weight,
            )
            final_objective = (
                action_parts.combined + action_l2 * transfer_candidate.square().sum()
            )
        if float(final_objective) >= float(zero_parts.combined):
            raw = zero
            transfer = zero
            final_parts = zero_parts
            final_objective = zero_parts.combined
            reason = "WRITER_INVALID_NO_IMPROVEMENT"
            zero_winner = True
        else:
            raw = raw_candidate
            transfer = transfer_candidate
            final_parts = action_parts
            reason = "VALID"
            zero_winner = False
    if initial_gradient is None:
        raise RuntimeError("Writer did not produce an initial gradient")
    writer = WriterSolveResult(
        full_solve_raw_action=raw_candidate.detach(),
        full_solve_transfer_action=transfer_candidate.detach(),
        raw_action=raw.detach(),
        transfer_action=transfer.detach(),
        zero_prediction_loss=float(zero_parts.combined),
        final_prediction_loss=float(final_parts.combined),
        final_objective=float(final_objective),
        accepted_steps=accepted,
        reason=reason,
        zero_action_winner=zero_winner,
    )
    return IsolatedSolveResult(
        writer=writer,
        initial_gradient=initial_gradient.cpu().numpy().astype(np.float64)[0],
        train_zero_components=(
            float(zero_parts.softordinal),
            float(zero_parts.pair),
            float(zero_parts.combined),
        ),
        train_action_components=(
            float(final_parts.softordinal),
            float(final_parts.pair),
            float(final_parts.combined),
        ),
    )


def evaluate_frozen_action(
    model: Any,
    cache: Any,
    view: LocalLabelView,
    pairs: FrozenPairRegistry,
    action: NDArray[np.floating],
    *,
    pair_loss_weight: float = 1.0,
) -> dict[str, float]:
    if view.side != "oob" or pairs.side != "oob":
        raise PermissionError("OOB evaluation requires OOB-local frozen artifacts")
    dtype = cache.base_gate_logits.dtype
    device = cache.base_gate_logits.device
    zero = torch.zeros((1, 7), dtype=dtype, device=device)
    changed = torch.as_tensor(action, dtype=dtype, device=device).reshape(1, 7)
    weights = torch.ones(len(view.indices), dtype=dtype, device=device)
    with _writer_context(model), torch.no_grad():
        zero_parts = loss_with_frozen_pairs(
            model.rank_from_cache(_detached_cache(cache), beta_logit=zero),
            view,
            pairs,
            sample_weights=weights,
            pair_loss_weight=pair_loss_weight,
        )
        action_parts = loss_with_frozen_pairs(
            model.rank_from_cache(_detached_cache(cache), beta_logit=changed),
            view,
            pairs,
            sample_weights=weights,
            pair_loss_weight=pair_loss_weight,
        )
    denominator = max(abs(float(zero_parts.combined)), 1.0e-12)
    return {
        "zero_softordinal_loss": float(zero_parts.softordinal),
        "zero_pair_loss": float(zero_parts.pair),
        "zero_combined_loss": float(zero_parts.combined),
        "action_softordinal_loss": float(action_parts.softordinal),
        "action_pair_loss": float(action_parts.pair),
        "action_combined_loss": float(action_parts.combined),
        "relative_gain": (
            float(zero_parts.combined) - float(action_parts.combined)
        )
        / denominator,
    }


class RepeatLabelIsolationSession:
    """Fail-closed state controller for one repeat."""

    def __init__(
        self,
        split: FrozenSplitRegistry,
        vault: SplitRawReturnVault,
    ) -> None:
        self.split = split
        self.vault = vault
        self.state = RepeatState.SPLIT_FROZEN
        self.trace = [self.state.value]
        self.train_view: LocalLabelView | None = None
        self.train_pairs: FrozenPairRegistry | None = None
        self.solve_result: IsolatedSolveResult | None = None
        self.action_token: FrozenActionToken | None = None
        self.oob_view: LocalLabelView | None = None
        self.oob_pairs: FrozenPairRegistry | None = None
        self.oob_metrics: dict[str, float] | None = None

    def _require(self, state: RepeatState) -> None:
        if self.state is not state:
            raise PermissionError(
                f"illegal label-isolation transition: {self.state.value} -> expected {state.value}"
            )

    def _advance(self, state: RepeatState) -> None:
        self.state = state
        self.trace.append(state.value)

    def materialize_train(self) -> LocalLabelView:
        self._require(RepeatState.SPLIT_FROZEN)
        self.train_view = LocalLabelView.materialize(
            "train", self.split.train_indices, self.vault.train()
        )
        self._advance(RepeatState.TRAIN_LABEL_VIEW_MATERIALIZED)
        return self.train_view

    def freeze_train_pairs(
        self, *, minimum_rank_gap: float, maximum_pairs: int, namespace: str
    ) -> FrozenPairRegistry:
        self._require(RepeatState.TRAIN_LABEL_VIEW_MATERIALIZED)
        assert self.train_view is not None
        self.train_pairs = FrozenPairRegistry.create(
            self.train_view,
            stock_count=self.split.stock_count,
            minimum_rank_gap=minimum_rank_gap,
            maximum_pairs=maximum_pairs,
            namespace=namespace,
        )
        self._advance(RepeatState.TRAIN_PAIR_REGISTRY_FROZEN)
        return self.train_pairs

    def solve(self, model: Any, cache: Any, **kwargs: Any) -> IsolatedSolveResult:
        self._require(RepeatState.TRAIN_PAIR_REGISTRY_FROZEN)
        assert self.train_view is not None and self.train_pairs is not None
        self.solve_result = solve_isolated_train_action(
            model,
            cache,
            self.train_view,
            self.train_pairs,
            sample_weights=self.split.train_weights,
            **kwargs,
        )
        self._advance(RepeatState.RAW_ACTION_SOLVED)
        return self.solve_result

    def freeze_action(self) -> FrozenActionToken:
        self._require(RepeatState.RAW_ACTION_SOLVED)
        assert (
            self.solve_result is not None
            and self.train_view is not None
            and self.train_pairs is not None
        )
        action = (
            self.solve_result.writer.transfer_action.detach()
            .cpu()
            .numpy()
            .astype(np.float64)[0]
        )
        self.action_token = FrozenActionToken(
            action=action,
            action_hash=_array_hash(action),
            split_hash=self.split.registry_hash,
            train_label_hash=self.train_view.rank_hash,
            train_pair_hash=self.train_pairs.registry_hash,
        )
        self._advance(RepeatState.ACTION_FROZEN)
        return self.action_token

    def open_oob(self) -> LocalLabelView:
        self._require(RepeatState.ACTION_FROZEN)
        assert self.action_token is not None
        self.oob_view = LocalLabelView.materialize(
            "oob", self.split.oob_indices, self.vault.oob(self.action_token)
        )
        self._advance(RepeatState.OOB_LABEL_VIEW_OPENED)
        return self.oob_view

    def freeze_oob_pairs(
        self, *, minimum_rank_gap: float, maximum_pairs: int, namespace: str
    ) -> FrozenPairRegistry:
        self._require(RepeatState.OOB_LABEL_VIEW_OPENED)
        assert self.oob_view is not None
        self.oob_pairs = FrozenPairRegistry.create(
            self.oob_view,
            stock_count=self.split.stock_count,
            minimum_rank_gap=minimum_rank_gap,
            maximum_pairs=maximum_pairs,
            namespace=namespace,
        )
        self._advance(RepeatState.OOB_PAIR_REGISTRY_FROZEN)
        return self.oob_pairs

    def evaluate_oob(self, model: Any, cache: Any) -> dict[str, float]:
        self._require(RepeatState.OOB_PAIR_REGISTRY_FROZEN)
        assert (
            self.oob_view is not None
            and self.oob_pairs is not None
            and self.action_token is not None
        )
        self.oob_metrics = evaluate_frozen_action(
            model,
            cache,
            self.oob_view,
            self.oob_pairs,
            self.action_token.action,
        )
        self._advance(RepeatState.OOB_EVALUATED)
        return self.oob_metrics
