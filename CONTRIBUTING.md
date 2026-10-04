# Contributing

Start with the [training guide](code/README.md) and [implementation notes](code/IMPLEMENTATION.md). Use the locked dependencies and retain a record of any configuration changes when comparing experiments.

## Where to make changes

| Area | Location |
| --- | --- |
| Backbone interface | `code/src/mef_unified/adapters.py` |
| Writer, Memory, and CAL1 | `code/src/mef_delivery/` |
| Data loading and training orchestration | `code/handoff/` |
| Training configuration | `code/config/` |
| Shared evaluation | `code/src/evaluation/` |

Keep Off/On comparisons paired through the same final predictor weights. Fit preprocessing on the applicable training window, respect label maturity, and retain fallback and rejection records. Do not use evaluation labels to choose actions or silently change dataset variants.

## Describing a change

Explain the problem, resulting behavior, affected models or datasets, and any changes to experimental semantics. State which checks or experiments were actually run; distinguish interface checks from full training validation. For experimental results, include the relevant configuration, seed, data identity, and aggregate metrics without exposing keys, private paths, raw data, or per-asset labels.

Upstream components retain their own licenses. Consult [third-party notices](THIRD_PARTY_NOTICES.md) before redistributing upstream code. The project's own license remains pending author selection.
