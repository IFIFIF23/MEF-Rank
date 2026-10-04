"""Canonical machine projection for the v3.7.1 candidate."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .contracts import V371WriterConfig, candidate_projection


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frozen_candidate_projection(config: V371WriterConfig) -> dict[str, Any]:
    projection = candidate_projection(config)
    identity_payload = {
        key: value for key, value in projection.items()
        if key not in {"mef_system_semantics_hash", "cal1_data_derived_values"}
    }
    projection["mef_system_semantics_hash"] = canonical_hash(identity_payload)
    projection["candidate_projection_hash"] = canonical_hash(projection)
    return projection
