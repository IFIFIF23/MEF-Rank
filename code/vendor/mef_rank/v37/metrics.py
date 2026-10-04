"""Split-local Top-K metrics and frozen base-position LambdaNDCG pairs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..v36.label_isolation import subset_average_tie_rank


ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = str(array.dtype).encode() + b"|" + json.dumps(array.shape).encode() + b"|" + array.tobytes()
    return hashlib.sha256(payload).hexdigest()


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def local_k(selection_k: int, subset_count: int, full_count: int) -> int:
    if selection_k <= 0 or subset_count <= 0 or full_count <= 0:
        raise ValueError("local-K inputs must be positive")
    return min(
        subset_count,
        max(1, int(np.floor(selection_k * subset_count / full_count + 0.5))),
    )


def stable_order(scores: ArrayF, stock_ids: NDArray[np.str_], *, descending: bool) -> ArrayI:
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(stock_ids).astype(str)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("stable order inputs are invalid")
    primary = -values if descending else values
    return np.lexsort((ids, primary)).astype(np.int64)


def rank_ic(scores: ArrayF, ranks: ArrayF) -> float:
    x = subset_average_tie_rank(np.asarray(scores, dtype=np.float64))
    y = np.asarray(ranks, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def ranking_movement(
    base_scores: ArrayF,
    final_scores: ArrayF,
    stock_ids: NDArray[np.str_],
) -> dict[str, float]:
    base = np.asarray(base_scores, dtype=np.float64)
    final = np.asarray(final_scores, dtype=np.float64)
    ids = np.asarray(stock_ids).astype(str)
    if base.shape != final.shape or ids.shape != base.shape or len(base) < 2:
        raise ValueError("ranking movement inputs are invalid")
    base_order = stable_order(base, ids, descending=True)
    final_order = stable_order(final, ids, descending=True)
    base_position = np.empty(len(base), dtype=np.int64)
    final_position = np.empty(len(base), dtype=np.int64)
    base_position[base_order] = np.arange(len(base))
    final_position[final_order] = np.arange(len(base))
    movement = np.abs(final_position - base_position)

    def membership_change(fraction: float) -> float:
        count = max(1, int(np.ceil(fraction * len(base))))
        overlap = len(
            set(base_order[:count].tolist()) & set(final_order[:count].tolist())
        )
        return float(1.0 - overlap / count)

    return {
        "mean_abs_rank_move": float(np.mean(movement)),
        "median_abs_rank_move": float(np.median(movement)),
        "maximum_abs_rank_move": float(np.max(movement)),
        "top10_member_change_fraction": membership_change(0.10),
        "top20_member_change_fraction": membership_change(0.20),
    }


@dataclass(frozen=True)
class TopKMetrics:
    ndcg: float
    rank_ic: float
    fractional_precision: float
    top_k_mean_return: float
    spread: float

    def delta(self, base: "TopKMetrics") -> dict[str, float]:
        return {
            "primary_gain": self.ndcg - base.ndcg,
            "rank_ic_delta": self.rank_ic - base.rank_ic,
            "precision_delta": self.fractional_precision - base.fractional_precision,
            "top_k_mean_return_delta": self.top_k_mean_return - base.top_k_mean_return,
            "spread_delta": self.spread - base.spread,
        }


def topk_metrics(
    scores: ArrayF,
    raw_returns: ArrayF,
    stock_ids: NDArray[np.str_],
    *,
    k: int,
) -> TopKMetrics:
    scores = np.asarray(scores, dtype=np.float64)
    returns = np.asarray(raw_returns, dtype=np.float64)
    ids = np.asarray(stock_ids).astype(str)
    if scores.shape != returns.shape or ids.shape != scores.shape or k < 1 or k > len(scores):
        raise ValueError("Top-K metric inputs are invalid")
    if not np.isfinite(scores).all() or not np.isfinite(returns).all():
        raise ValueError("Top-K metric inputs must be finite")
    ranks = subset_average_tie_rank(returns)
    order = stable_order(scores, ids, descending=True)
    top = order[:k]
    bottom = order[-k:]
    gain = np.exp2(ranks) - 1.0
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gain[top] * discount))
    ideal = stable_order(ranks, ids, descending=True)[:k]
    idcg = float(np.sum(gain[ideal] * discount))
    cutoff = np.partition(returns, len(returns) - k)[len(returns) - k]
    greater = int((returns > cutoff).sum())
    equal = int((returns == cutoff).sum())
    membership = np.where(
        returns > cutoff,
        1.0,
        np.where(returns == cutoff, (k - greater) / max(equal, 1), 0.0),
    )
    top_return = float(np.mean(returns[top]))
    return TopKMetrics(
        ndcg=dcg / (idcg + 1.0e-12),
        rank_ic=rank_ic(scores, ranks),
        fractional_precision=float(np.mean(membership[top])),
        top_k_mean_return=top_return,
        spread=top_return - float(np.mean(returns[bottom])),
    )


@dataclass(frozen=True)
class LambdaPairRegistry:
    pairs: ArrayI
    weights: ArrayF
    local_k: int
    registry_hash: str


def lambda_ndcg_pairs(
    base_scores: ArrayF,
    label_ranks: ArrayF,
    global_indices: ArrayI,
    stock_ids: NDArray[np.str_],
    *,
    k: int,
    maximum_pairs: int,
    namespace: str,
) -> LambdaPairRegistry:
    scores = np.asarray(base_scores, dtype=np.float64)
    ranks = np.asarray(label_ranks, dtype=np.float64)
    indices = np.asarray(global_indices, dtype=np.int64)
    ids = np.asarray(stock_ids).astype(str)
    if not (scores.shape == ranks.shape == indices.shape == ids.shape):
        raise ValueError("LambdaNDCG pair inputs do not align")
    order = stable_order(scores, ids, descending=True)
    positions = np.empty(len(order), dtype=np.int64)
    positions[order] = np.arange(len(order), dtype=np.int64)
    gain = np.exp2(ranks) - 1.0
    ideal = stable_order(ranks, ids, descending=True)[:k]
    discount = np.zeros(len(order), dtype=np.float64)
    discount[:k] = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    idcg = float(np.sum(gain[ideal] * discount[:k])) + 1.0e-12
    left: list[int] = []
    right: list[int] = []
    weight: list[float] = []
    for i in range(len(order)):
        for j in range(len(order)):
            if ranks[i] <= ranks[j] or min(positions[i], positions[j]) >= k:
                continue
            value = abs(
                (gain[i] - gain[j])
                * (discount[positions[i]] - discount[positions[j]])
            ) / idcg
            if value > 0:
                left.append(i)
                right.append(j)
                weight.append(float(value))
    if not weight:
        empty = np.empty((0, 2), dtype=np.int64)
        return LambdaPairRegistry(empty, np.empty(0), k, array_hash(empty))
    weights = np.asarray(weight, dtype=np.float64)
    local_pairs = np.column_stack([left, right]).astype(np.int64)
    if len(weights) > maximum_pairs:
        rng = np.random.default_rng(seed(namespace))
        chosen = rng.choice(
            len(weights),
            size=maximum_pairs,
            replace=False,
            p=weights / weights.sum(),
        )
        local_pairs = local_pairs[chosen]
        weights = weights[chosen]
    pairs = indices[local_pairs]
    weights = weights / weights.mean()
    payload = np.column_stack([pairs, weights]).astype(np.float64)
    return LambdaPairRegistry(pairs, weights, k, array_hash(payload))
