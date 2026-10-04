"""Frozen base-STR loss and checkpoint compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ..writer.pgd import fixed_pair_indices, pairwise_loss, soft_ordinal_loss
from .day_batch import DayBatch
from .network import MEFRankSTR, TorchSTRForward


@dataclass(frozen=True)
class BaseLoss:
    total: Tensor
    soft_ordinal: Tensor
    pairwise: Tensor
    valid_stock_count: int
    pair_count: int


def base_str_loss(
    output: TorchSTRForward,
    batch: DayBatch,
    *,
    pair_loss_weight: float,
    maximum_pairs: int,
    minimum_rank_gap: float,
    namespace: str,
) -> BaseLoss:
    """Compute only the canonical soft-ordinal plus fixed-pair objective."""

    if output.score.shape[0] != 1 or batch.label_rank is None or batch.label_valid_mask is None:
        raise ValueError("base STR loss requires one labeled date")
    valid = batch.label_valid_mask[0] & batch.signal_universe_mask[0]
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() < 2:
        raise ValueError("base STR loss has insufficient valid stocks")
    labels = batch.label_rank[0]
    ordinal = soft_ordinal_loss(output.rank_logits[0, indices], labels[indices])
    pairs = fixed_pair_indices(
        labels,
        indices,
        minimum_rank_gap=minimum_rank_gap,
        maximum_pairs=maximum_pairs,
        namespace=namespace,
    )
    pair = pairwise_loss(output.score[0], labels, pairs)
    return BaseLoss(
        total=ordinal + float(pair_loss_weight) * pair,
        soft_ordinal=ordinal,
        pairwise=pair,
        valid_stock_count=int(indices.numel()),
        pair_count=int(pairs.shape[0]),
    )


def save_str_checkpoint(
    model: MEFRankSTR,
    path: Path,
    *,
    metadata: dict[str, Any],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    required = {
        "horizon",
        "checkpoint_id",
        "model_architecture_id",
        "model_semantics_hash",
        "run_manifest_hash",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata lacks fields: {missing}")
    if str(metadata["model_architecture_id"]) != model.architecture_id:
        raise ValueError("checkpoint architecture metadata disagrees with model")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "state_dict": model.state_dict(),
            "profile": vars(model.profile),
            "metadata": metadata,
        },
        path,
    )


def load_str_checkpoint(
    path: Path,
    *,
    expected_horizon: int,
    expected_model_semantics_hash: str,
    map_location: str | torch.device = "cpu",
) -> tuple[MEFRankSTR, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported STR checkpoint format")
    metadata = dict(payload["metadata"])
    if int(metadata["horizon"]) != int(expected_horizon):
        raise ValueError("checkpoint horizon mismatch")
    if str(metadata["model_semantics_hash"]) != str(expected_model_semantics_hash):
        raise ValueError("checkpoint model semantics mismatch")
    from .network import STRProfile

    model = MEFRankSTR(STRProfile(**payload["profile"]))
    if metadata["model_architecture_id"] != model.architecture_id:
        raise ValueError("checkpoint architecture id is not reproducible")
    model.load_state_dict(payload["state_dict"], strict=True)
    return model, metadata
