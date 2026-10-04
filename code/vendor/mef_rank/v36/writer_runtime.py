"""Portable orchestration of the frozen v3.6 R1/B20 joint Writer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..writer.pgd import _mean_reliability_shift, solve_writer_action
from .formal_writer import (
    FORMAL_WRITER_SEMANTICS,
    anchor_quality,
    final_gate_reason,
    hard_gate_reason,
)
from .jacf_cal0 import JointMemoryRecord, JointState
from .label_isolation import (
    FrozenSplitRegistry,
    RepeatLabelIsolationSession,
    SplitRawReturnVault,
    evaluate_frozen_action,
)


ArrayF = NDArray[np.float64]


def _seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")


def _r1_split(
    valid: NDArray[np.int64],
    strata: NDArray[np.integer],
    *,
    namespace: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.default_rng(_seed(namespace))
    train_parts: list[NDArray[np.int64]] = []
    oob_parts: list[NDArray[np.int64]] = []
    for value in sorted(set(strata[valid].tolist())):
        members = valid[strata[valid] == value]
        shuffled = rng.permutation(members)
        if len(shuffled) == 1:
            train_count = 1
        else:
            train_count = int(
                np.clip(round(0.70 * len(shuffled)), 1, len(shuffled) - 1)
            )
        train_parts.append(shuffled[:train_count])
        oob_parts.append(shuffled[train_count:])
    train = np.sort(np.concatenate(train_parts)).astype(np.int64)
    oob = np.sort(np.concatenate(oob_parts)).astype(np.int64)
    if len(np.intersect1d(train, oob)) or len(np.union1d(train, oob)) != len(valid):
        raise AssertionError("R1 split is not a disjoint partition of valid stocks")
    return train, oob


def _covariance(values: ArrayF) -> ArrayF:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != 7:
        raise ValueError("Writer covariance requires at least two [7] repeats")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    empirical = centered.T @ centered / float(len(matrix))
    mu = float(np.trace(empirical) / 7.0)
    target = np.eye(7) * max(mu, 1.0e-12)
    delta = float(np.square(empirical - target).sum())
    beta = sum(
        float(np.square(np.outer(row, row) - empirical).sum())
        for row in centered
    ) / float(len(matrix) ** 2)
    shrinkage = (
        1.0 if delta <= 1.0e-24 else float(np.clip(beta / delta, 0.0, 1.0))
    )
    result = (1.0 - shrinkage) * empirical + shrinkage * target
    result = (result + result.T) / 2.0
    values_, vectors = np.linalg.eigh(result)
    result = (vectors * np.maximum(values_, 1.0e-12)) @ vectors.T
    return (result + result.T) / 2.0


def _cosine(left: ArrayF, right: ArrayF) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right) + 1.0e-12)
    return float(np.dot(left, right) / denominator)


@dataclass(frozen=True)
class WriterConfig:
    repeats: int = 20
    minimum_valid_repeats: int = 16
    pair_loss_weight: float = 1.0
    action_cap: float = 0.50
    action_l2: float = 0.05
    steps: int = 5
    initial_step_size: float = 0.15
    backtracking_trials: int = 4
    backtracking_factor: float = 0.5
    raw_logit_cap: float = 0.75
    pair_count: int = 4096
    pair_min_rank_gap: float = 0.10
    minimum_stocks: int = 100

    def __post_init__(self) -> None:
        if self.repeats != 20 or self.minimum_valid_repeats != 16:
            raise PermissionError("v3.6 Writer requires R1/B20 and 16 valid repeats")
        if (
            self.action_cap != 0.50
            or self.action_l2 != 0.05
            or self.steps != 5
            or self.initial_step_size != 0.15
            or self.backtracking_trials != 4
            or self.backtracking_factor != 0.5
            or self.raw_logit_cap != 0.75
            or self.pair_count != 4096
            or self.pair_min_rank_gap != 0.10
            or self.minimum_stocks != 100
        ):
            raise PermissionError("frozen v3.6 Writer controls changed")


@dataclass(frozen=True)
class WriterRepeat:
    repeat: int
    valid: bool
    transfer_action: ArrayF
    reliability_shift: ArrayF
    relative_gain: float
    split_hash: str


@dataclass(frozen=True)
class WriterResult:
    qualified: bool
    reason_code: str | None
    repeats: tuple[WriterRepeat, ...]
    full_solve_transfer_action: ArrayF
    final_reliability_shift: ArrayF
    repeat_reliability_shift_mean: ArrayF
    covariance: ArrayF
    full_repeat_mean_cosine: float
    quality: dict[str, float | int]
    writer_stock_count: int

    def to_memory_record(
        self,
        *,
        anchor_id: str,
        anchor_date: str,
        episode_id: str,
        memory_eligible_date: str,
        state: JointState,
    ) -> JointMemoryRecord:
        if not self.qualified:
            raise RuntimeError(f"Writer is not Memory eligible: {self.reason_code}")
        return JointMemoryRecord(
            anchor_id=anchor_id,
            anchor_date=anchor_date,
            episode_id=episode_id,
            memory_eligible_date=memory_eligible_date,
            state=state,
            action=self.final_reliability_shift,
            covariance=self.covariance,
            q_anchor=float(self.quality["q_anchor"]),
            writer_stock_count=self.writer_stock_count,
            gain=float(self.quality["oob_relative_gain_median"]),
        )


def solve_joint_writer(
    model: object,
    cache: object,
    *,
    raw_returns: NDArray[np.floating],
    global_primary_rank: Tensor | NDArray[np.floating],
    writer_mask: Tensor,
    strata: NDArray[np.integer],
    namespace: str,
    config: WriterConfig = WriterConfig(),
) -> WriterResult:
    """Run the frozen 20-repeat local-rank Writer plus one global full solve."""

    valid = (
        torch.nonzero(writer_mask, as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    returns = np.asarray(raw_returns, dtype=np.float64)
    strata_array = np.asarray(strata)
    if (
        returns.shape != (writer_mask.numel(),)
        or strata_array.shape != returns.shape
        or len(valid) < config.minimum_stocks
    ):
        raise ValueError("Writer inputs do not align or contain fewer than 100 stocks")

    repeat_rows: list[WriterRepeat] = []
    for repeat in range(config.repeats):
        train, oob = _r1_split(
            valid,
            strata_array,
            namespace=f"{namespace}|R1|repeat={repeat}",
        )
        split = FrozenSplitRegistry.create(
            stock_count=len(returns),
            train_indices=train,
            train_weights=np.ones(len(train), dtype=np.float64),
            oob_indices=oob,
            namespace=f"{namespace}|R1|repeat={repeat}",
        )
        session = RepeatLabelIsolationSession(
            split, SplitRawReturnVault(split, returns)
        )
        session.materialize_train()
        session.freeze_train_pairs(
            minimum_rank_gap=config.pair_min_rank_gap,
            maximum_pairs=config.pair_count,
            namespace=f"{namespace}|R1|repeat={repeat}|train|pairs",
        )
        session.solve(
            model,
            cache,
            pair_loss_weight=config.pair_loss_weight,
            transfer_logit_cap=config.action_cap,
            action_l2=config.action_l2,
            steps=config.steps,
            initial_step_size=config.initial_step_size,
            backtracking_trials=config.backtracking_trials,
            backtracking_factor=config.backtracking_factor,
            raw_logit_cap=config.raw_logit_cap,
        )
        token = session.freeze_action()
        session.open_oob()
        session.freeze_oob_pairs(
            minimum_rank_gap=config.pair_min_rank_gap,
            maximum_pairs=config.pair_count,
            namespace=f"{namespace}|R1|repeat={repeat}|oob|pairs",
        )
        assert session.oob_view is not None and session.oob_pairs is not None
        metrics = evaluate_frozen_action(
            model, cache, session.oob_view, session.oob_pairs, token.action
        )
        action_tensor = torch.as_tensor(
            token.action[None],
            dtype=cache.base_gate_logits.dtype,
            device=cache.base_gate_logits.device,
        )
        shift, _ = _mean_reliability_shift(cache, action_tensor, writer_mask)
        action = np.asarray(token.action, dtype=np.float64)
        shift_array = shift.detach().cpu().numpy().astype(np.float64).reshape(7)
        gain = float(metrics["relative_gain"])
        repeat_rows.append(
            WriterRepeat(
                repeat,
                bool(
                    np.isfinite(action).all()
                    and np.isfinite(shift_array).all()
                    and np.isfinite(gain)
                ),
                action,
                shift_array,
                gain,
                split.registry_hash,
            )
        )

    valid_rows = tuple(row for row in repeat_rows if row.valid)
    gains = np.asarray([row.relative_gain for row in valid_rows], dtype=np.float64)
    actions = (
        np.stack([row.transfer_action for row in valid_rows])
        if valid_rows
        else np.empty((0, 7))
    )
    reason = hard_gate_reason(
        stock_count=len(valid),
        valid_repeats=len(valid_rows),
        gains=gains,
        actions=actions,
    )
    empty_vector = np.full(7, np.nan)
    empty_covariance = np.full((7, 7), np.nan)
    if reason is not None:
        quality = anchor_quality(gains) if len(gains) else {}
        return WriterResult(
            False,
            reason,
            tuple(repeat_rows),
            empty_vector,
            empty_vector,
            empty_vector,
            empty_covariance,
            float("nan"),
            quality,
            len(valid),
        )

    device = cache.base_gate_logits.device
    dtype = cache.base_gate_logits.dtype
    global_rank = torch.as_tensor(
        global_primary_rank, dtype=dtype, device=device
    ).reshape(-1)
    indices = torch.as_tensor(valid, dtype=torch.long, device=device)
    full = solve_writer_action(
        model,
        cache,
        global_rank,
        indices,
        sample_weights=torch.ones(len(indices), dtype=dtype, device=device),
        namespace=f"{namespace}|full",
        pair_loss_weight=config.pair_loss_weight,
        transfer_logit_cap=config.action_cap,
        action_l2=config.action_l2,
        steps=config.steps,
        initial_step_size=config.initial_step_size,
        backtracking_trials=config.backtracking_trials,
        backtracking_factor=config.backtracking_factor,
        raw_logit_cap=config.raw_logit_cap,
        pair_count=config.pair_count,
        pair_min_rank_gap=config.pair_min_rank_gap,
        active_coordinate_mask=None,
    )
    full_action = (
        full.transfer_action.detach().cpu().numpy().astype(np.float64).reshape(7)
    )
    shifts = np.stack([row.reliability_shift for row in valid_rows])
    covariance = _covariance(shifts)
    cosine = _cosine(full_action, actions.mean(axis=0))
    reason = final_gate_reason(
        full_action=full_action, cosine=cosine, covariance=covariance
    )
    full_tensor = torch.as_tensor(
        full_action[None], dtype=dtype, device=device
    )
    final_shift, _ = _mean_reliability_shift(
        cache, full_tensor, writer_mask
    )
    quality = anchor_quality(gains)
    return WriterResult(
        reason is None,
        reason,
        tuple(repeat_rows),
        full_action,
        final_shift.detach().cpu().numpy().astype(np.float64).reshape(7),
        shifts.mean(axis=0),
        covariance,
        cosine,
        quality,
        len(valid),
    )
