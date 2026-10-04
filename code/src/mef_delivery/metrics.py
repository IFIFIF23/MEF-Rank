"""Endpoint-exact split-local K-continuum metrics and pair registry."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mef_rank.v36.label_isolation import subset_average_tie_rank

ArrayF = NDArray[np.float64]
ArrayI = NDArray[np.int64]


def array_hash(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    payload = str(a.dtype).encode() + b"|" + json.dumps(a.shape).encode() + b"|" + a.tobytes()
    return hashlib.sha256(payload).hexdigest()


def seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def effective_k(selection_k: int, count: int) -> int:
    if isinstance(selection_k, bool) or not isinstance(selection_k, int):
        raise TypeError("selection_k must be a positive integer")
    if selection_k <= 0 or count <= 0:
        raise ValueError("selection_k and count must be positive")
    return min(selection_k, count)


def local_k(selection_k: int, subset_count: int, full_count: int) -> int:
    if subset_count <= 0 or full_count <= 0:
        raise ValueError("split-local K counts must be positive")
    fraction = effective_k(selection_k, full_count) / full_count
    return min(subset_count, max(1, int(np.floor(fraction * subset_count + 0.5))))


def k_continuum_alpha(count: int, k: int) -> float:
    if count <= 0 or k < 1 or k > count:
        raise ValueError("invalid K-continuum endpoint")
    return 0.0 if count == 1 else float((count - k) / (count - 1))


def stable_order(scores: ArrayF, stock_ids: NDArray[np.str_], *, descending: bool) -> ArrayI:
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(stock_ids).astype(str)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("stable order inputs are invalid")
    return np.lexsort((ids, -values if descending else values)).astype(np.int64)


def rank_ic(scores: ArrayF, ranks: ArrayF) -> float:
    x = subset_average_tie_rank(np.asarray(scores, dtype=np.float64))
    y = np.asarray(ranks, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3 or np.std(x[valid]) <= 0 or np.std(y[valid]) <= 0:
        return 0.0
    return float(np.corrcoef(x[valid], y[valid])[0, 1])


def ranking_movement(base_scores: ArrayF, final_scores: ArrayF, stock_ids: NDArray[np.str_]) -> dict[str, float]:
    base, final, ids = np.asarray(base_scores), np.asarray(final_scores), np.asarray(stock_ids).astype(str)
    if base.shape != final.shape or ids.shape != base.shape or len(base) < 2:
        raise ValueError("ranking movement inputs are invalid")
    bo, fo = stable_order(base, ids, descending=True), stable_order(final, ids, descending=True)
    bp, fp = np.empty(len(base), dtype=np.int64), np.empty(len(base), dtype=np.int64)
    bp[bo], fp[fo] = np.arange(len(base)), np.arange(len(base))
    move = np.abs(fp - bp)
    def change(fraction: float) -> float:
        n = max(1, int(np.ceil(fraction * len(base))))
        return float(1 - len(set(bo[:n]) & set(fo[:n])) / n)
    return {
        "mean_abs_rank_move": float(move.mean()),
        "median_abs_rank_move": float(np.median(move)),
        "maximum_abs_rank_move": float(move.max()),
        "top10_member_change_fraction": change(0.10),
        "top20_member_change_fraction": change(0.20),
    }


@dataclass(frozen=True)
class KContinuumMetrics:
    ndcg: float
    rank_ic: float
    fractional_precision: float
    top_k_mean_return: float
    spread: float
    k: int
    alpha: float
    utility: float

    def delta(self, base: "KContinuumMetrics") -> dict[str, float]:
        return {
            "primary_gain": self.utility - base.utility,
            "utility_delta": self.utility - base.utility,
            "ndcg_delta": self.ndcg - base.ndcg,
            "rank_ic_delta": self.rank_ic - base.rank_ic,
            "precision_delta": self.fractional_precision - base.fractional_precision,
            "top_k_mean_return_delta": self.top_k_mean_return - base.top_k_mean_return,
            "spread_delta": self.spread - base.spread,
        }


TopKMetrics = KContinuumMetrics


def topk_metrics(scores: ArrayF, raw_returns: ArrayF, stock_ids: NDArray[np.str_], *, k: int) -> KContinuumMetrics:
    scores, returns, ids = np.asarray(scores, dtype=np.float64), np.asarray(raw_returns, dtype=np.float64), np.asarray(stock_ids).astype(str)
    if scores.shape != returns.shape or ids.shape != scores.shape or k < 1 or k > len(scores):
        raise ValueError("K-continuum metric inputs are invalid")
    if not np.isfinite(scores).all() or not np.isfinite(returns).all():
        raise ValueError("K-continuum metric inputs must be finite")
    ranks = subset_average_tie_rank(returns)
    order = stable_order(scores, ids, descending=True)
    top, bottom = order[:k], order[-k:]
    gain = np.exp2(ranks) - 1.0
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gain[top] * discount))
    ideal = stable_order(ranks, ids, descending=True)[:k]
    idcg = float(np.sum(gain[ideal] * discount))
    cutoff = np.partition(returns, len(returns) - k)[len(returns) - k]
    greater, equal = int((returns > cutoff).sum()), int((returns == cutoff).sum())
    membership = np.where(returns > cutoff, 1.0, np.where(returns == cutoff, (k - greater) / max(equal, 1), 0.0))
    top_return = float(np.mean(returns[top]))
    ric = rank_ic(scores, ranks)
    alpha = k_continuum_alpha(len(scores), k)
    ndcg = dcg / (idcg + 1.0e-12)
    return KContinuumMetrics(
        ndcg, ric, float(np.mean(membership[top])), top_return,
        top_return - float(np.mean(returns[bottom])), k, alpha,
        alpha * ndcg + (1.0 - alpha) * (1.0 + ric) / 2.0,
    )


@dataclass(frozen=True)
class LambdaPairRegistry:
    pairs: ArrayI
    weights: ArrayF
    local_k: int
    alpha: float
    registry_hash: str


def k_continuum_pairs(
    base_scores: ArrayF, label_ranks: ArrayF, global_indices: ArrayI,
    stock_ids: NDArray[np.str_], *, k: int, maximum_pairs: int,
    minimum_global_rank_gap: float, namespace: str,
) -> LambdaPairRegistry:
    scores, ranks = np.asarray(base_scores, dtype=np.float64), np.asarray(label_ranks, dtype=np.float64)
    indices, ids = np.asarray(global_indices, dtype=np.int64), np.asarray(stock_ids).astype(str)
    if not (scores.shape == ranks.shape == indices.shape == ids.shape):
        raise ValueError("K-continuum pair inputs do not align")
    alpha = k_continuum_alpha(len(scores), k)
    order = stable_order(scores, ids, descending=True)
    positions = np.empty(len(order), dtype=np.int64); positions[order] = np.arange(len(order))
    gains = np.exp2(ranks) - 1.0
    ideal = stable_order(ranks, ids, descending=True)[:k]
    discount = np.zeros(len(order)); discount[:k] = 1.0 / np.log2(np.arange(2, k + 2))
    idcg = float(np.sum(gains[ideal] * discount[:k])) + 1.0e-12
    better = ranks[:, None] > ranks[None, :]
    left, right = np.nonzero(better)
    top = np.abs(
        (gains[left] - gains[right])
        * (discount[positions[left]] - discount[positions[right]])
    ) / idcg
    glob = (
        np.abs(ranks[left] - ranks[right]) >= minimum_global_rank_gap
    ).astype(np.float64)
    candidate = (top > 0) | (glob > 0)
    left, right, top, glob = (
        left[candidate], right[candidate], top[candidate], glob[candidate]
    )
    if not len(left):
        empty = np.empty((0, 2), dtype=np.int64)
        return LambdaPairRegistry(empty, np.empty(0), k, alpha, array_hash(empty))
    if np.any(top > 0):
        top = top / top[top > 0].mean()
    weights = alpha * top + (1.0 - alpha) * glob
    keep = weights > 0
    local_pairs = np.column_stack([left, right]).astype(np.int64)[keep]
    weights = weights[keep]
    if len(weights) > maximum_pairs:
        chosen = np.random.default_rng(seed(namespace)).choice(
            len(weights), maximum_pairs, replace=False, p=weights / weights.sum()
        )
        local_pairs, weights = local_pairs[chosen], weights[chosen]
    global_pairs = indices[local_pairs]
    weights = weights / weights.mean()
    payload = np.column_stack([global_pairs, weights, np.full(len(weights), alpha)])
    return LambdaPairRegistry(global_pairs, weights, k, alpha, array_hash(payload))


lambda_ndcg_pairs = k_continuum_pairs
