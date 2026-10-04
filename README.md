<p align="center">
  <img src="docs/assets/banner.svg" alt="MEF-Rank — Model-Aware Action Memory for Financial Ranking" width="100%">
</p>

<h3 align="center">Model-Aware Action Memory for Financial Ranking</h3>

<p align="center">
  Historical experience · Model-aware retrieval · Selective feedback
</p>

<p align="center">
  <a href="README.zh-CN.md">中文</a> &nbsp; / &nbsp;
  <a href="#quick-start">Quick start</a> &nbsp; / &nbsp;
  <a href="code/README.md">Training guide</a> &nbsp; / &nbsp;
  <a href="docs/DATA.md">Data</a> &nbsp; / &nbsp;
  <a href="CITATION.cff">Citation</a>
</p>

---

## Overview

**MEF-Rank** is a model-aware action memory framework for financial ranking. The relevance of historical experience depends on both the market environment and how the current predictor uses financial signals. MEF-Rank brings these two perspectives together, turning matured prediction outcomes into reusable adjustments to signal utilization.

An offline **Writer** searches and screens historical actions, storing their utilization shifts alongside market and model states. At inference, joint retrieval identifies compatible experience, calibration controls the strength of feedback, and bounded inversion translates the selected shift into the current predictor's gate coordinates. The predictor then recomputes its representations with fixed weights, forming an explicit connection between historical feedback and the current ranking decision.

**Designed for portability across forecasting backbones and markets.** A shared seven-group signal interface separates the feedback mechanism from each backbone's internal representation. Backbone-specific adapters connect this interface to sequence, variable, or patch representations, allowing Writer, retrieval, calibration, and action inversion to follow a common workflow. The repository includes ten backbone adaptations and five market configurations. Extending to a new setting involves adapting the signal interface and data configuration, then building the corresponding history, memory, and calibration for that backbone, market, and seed.

| Design advantage | What it enables |
| :--- | :--- |
| **Portability across architectures and markets** | Reuses the feedback workflow through a common action interface and backbone-specific adapters. Separate histories, memories, and calibration retain the context of each model and market. |
| **Market–model joint retrieval** | Matches historical experience using both market conditions and the predictor's decision state, making relevance specific to the model receiving the feedback. |
| **Action transport across compatible model states** | Stores the intended utilization change across seven financial signal groups and maps it through the current gates using bounded inversion. This accounts for gate saturation when realizing a retrieved action and makes the adjustment inspectable at the group level. |
| **Screened historical experience** | Separates direction search, dose selection, and held-out screening within the Writer. Memory records retain support, quality, and uncertainty information for subsequent selection. |
| **Feedback with frozen weights** | Applies an accepted action through one additional forward pass, without test-time parameter updates. Paired Off/On evaluation uses the same final predictor to isolate the effect of historical feedback. |
| **Selective execution and explicit fallback** | Uses support, uncertainty, and feasibility checks to decide whether to apply an action. Unsupported queries retain the original Off prediction, with fallback reasons recorded for analysis. |

## How it works

<p align="center">
  <img src="docs/assets/framework.svg" alt="MEF-Rank pipeline: offline historical predictors and Writer build action memory and CAL1; inference retrieves, calibrates, inverts, and recomputes with fixed weights." width="100%">
</p>

**Off** is the first prediction. **On** uses historical experience to adjust seven signal groups and performs a second forward pass with the same weights. Both settings use the same reordered **variant 1** training data. See the [implementation notes](code/IMPLEMENTATION.md) for the detailed pipeline and adaptations.

## Quick start

**Environment:** Linux · Python 3.10 / 3.11 with `venv` · Git. First-time setup requires network access; training supports a compatible CUDA environment or CPU.

Obtain an authorized encrypted dataset and its matching key. Place the dataset in `data_encrypted/<dataset>/` and keep the key outside the repository. See [data setup](docs/DATA.md).

From the repository root, launch the paired pipeline:

```bash
bash code/run.sh --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

Replace the example key path with your own. The launcher prepares the local environment, installs locked upstream sources, and starts final predictor training, historical training, Writer, Memory, calibration, and paired evaluation. Use `--device cpu` for CPU execution.

<details>
<summary><b>Use an existing Python / CUDA environment</b></summary>

Install the dependencies in `code/requirements.txt` into your environment, then run:

```bash
MEF_PYTHON=/path/to/python bash code/run.sh --dataset nasdaq100 --model iTransformer --key-file /secure/keys/nasdaq100.key --device cuda
```

Manual installation, independent model entries, configuration options, and resume instructions are in the [training guide](code/README.md).

</details>

## Backbones and markets

**Ten backbones:** STR · iTransformer · TimeMixer · ModernTCN · SOFTS · TimeXer · TimeBridge · TimeFilter · SRSNet · ChannelTokenFormer.

| Dataset | Market | CLI argument |
| :--- | :--- | :--- |
| ChiNext | Chinese equities | `--dataset gem` |
| Main board | Chinese equities | `--dataset main` |
| NASDAQ-100 | U.S. equities | `--dataset nasdaq100` |
| S&P 500 | U.S. equities | `--dataset sp500` |
| Crypto-15 | Cryptocurrencies | `--dataset crypto` |

Choose the backbone with `--model`, for example `--model TimeMixer`. Use **`--mode both`** (the default) for paired Off/On evaluation; `off` and `on` are also available. Each backbone has dedicated entries in `code/models/`. Independent invocations create separate runs unless the same run is resumed.

## Repository layout

<details>
<summary><b>Explore the code and documentation</b></summary>

```text
.
├── README.md                  English project overview
├── README.zh-CN.md             Chinese project overview
├── CITATION.cff               Citation metadata
├── CONTRIBUTING.md            Guidance for extending the code
├── THIRD_PARTY_NOTICES.md      Upstream sources and notices
├── docs/
│   ├── assets/                Banner and framework diagram
│   └── DATA.md                Dataset access and layout
├── code/
│   ├── train.py               Unified training entry point
│   ├── run.sh                 Environment setup and launch
│   ├── requirements.txt       Python dependencies
│   ├── config/                Training and backbone configurations
│   ├── models/                Ten backbones × Off/On entry points
│   ├── handoff/               Data loading and orchestration
│   ├── src/                   Adapters, training, and evaluation
│   ├── vendor/                Shared MEF components
│   ├── third_party/           Upstream revision and file-hash locks
│   └── tests/                 Existing focused tests
└── data_encrypted/            Authorized local data; excluded from Git
```

Upstream sources are installed by `code/setup_sources.py` and excluded from Git. Local environments, encrypted data, keys, and run outputs are also excluded.

</details>

## Run artifacts

Outputs are written to `code/outputs/<dataset>/<model>/seed_<seed>/<run>/`.

| Artifact | Contents |
| :--- | :--- |
| `run_manifest.json` | Configuration, code/data identity, and device |
| `main/model.pt` | Shared final predictor weights and training scaler |
| `history/` | Historical checkpoints, Writer records, and rejection reasons |
| `memory.json` / `calibration.json` | Experience memory and calibration |
| `predictions.parquet` | Predictions and labels; paired outputs in `both/on` mode |
| `metrics.json` / `daily_metrics.csv` | Ranking metrics, paired differences, and fallback records |

See the [training guide](code/README.md#4-输出与改进位置) for output details and mode-specific behavior.

## Documentation

| Start here | Purpose |
| :--- | :--- |
| [Training and configuration](code/README.md) | Installation, command options, and resuming a run |
| [Data guide](docs/DATA.md) | Encrypted packages, views, and access |
| [Implementation notes](code/IMPLEMENTATION.md) | Components, provenance, and engineering details |
| [Contributing](CONTRIBUTING.md) | Where and how to extend the implementation |
| [Third-party notices](THIRD_PARTY_NOTICES.md) | Upstream source attribution |

---

<p align="center">
  <b>Code &amp; data inquiries</b><br>
  <a href="mailto:yifany@bupt.edu.cn">yifany@bupt.edu.cn</a>
</p>
