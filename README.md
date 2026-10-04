# MEF-Rank

**MEF-Rank: Model-Aware Action Memory for Financial Ranking**

Yifan Yuan · Yuanjian Zhang · Xiang Xu · Chuanchang Liu  
School of Computer Science, Beijing University of Posts and Telecommunications

[Paper](main.pdf) · [中文说明](README.zh-CN.md) · [Training guide](code/README.md) · [Data access](docs/DATA.md) · [Implementation notes](code/IMPLEMENTATION.md)

## Overview

MEF-Rank uses historical prediction experience to adjust how a forecasting backbone uses its input signals. This repository provides a shared training pipeline and paired **Off / On** evaluation across ten backbones and five market datasets.

- **Off:** predict with the final trained backbone.
- **On:** retrieve historical actions and perform a second forward pass through the **same backbone weights**; retain the Off prediction when support is insufficient.

Both settings use the same reordered **variant 1** training data. The pipeline connects final predictor training, quarterly historical predictors, Writer, Memory, CAL1 calibration, and paired evaluation.

## Quick start

**Requirements:** Linux, Python 3.10 or 3.11 with `venv`, Git, and network access for the first dependency installation. A compatible CUDA environment is recommended for training; CPU execution is also supported.

Obtain an authorized encrypted dataset and its matching key from the data owner. Place the dataset under `data_encrypted/<dataset>/` and store the key outside this repository. See [data setup](docs/DATA.md).

From the repository root:

```bash
bash code/run.sh --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

Replace the example key path with your own. The command prepares `code/.venv`, installs the pinned upstream sources, and starts the full Off/On pipeline. Historical training and Writer add substantial work beyond training the final predictor alone.

For an existing environment with the required dependencies installed:

```bash
MEF_PYTHON=/path/to/python bash code/run.sh --dataset nasdaq100 --model iTransformer --key-file /secure/keys/nasdaq100.key --device cuda
```

For manual installation, CPU execution, individual model entry points, and resuming a run, see the [training and configuration guide](code/README.md).

## Models and datasets

| Setting | Supported values |
| --- | --- |
| Backbone | `STR`, `iTransformer`, `TimeMixer`, `ModernTCN`, `SOFTS`, `TimeXer`, `TimeBridge`, `TimeFilter`, `SRSNet`, `ChannelTokenFormer` |
| Dataset | `gem`, `main`, `nasdaq100`, `sp500`, `crypto` |
| Mode | `both` (default), `off`, `on` |

Use `--mode both` for paired comparisons. Each backbone also has `off.py` and `on.py` entry points in `code/models/`; these share the implementation. Running the entries independently creates separate runs unless the same run is resumed.

## Repository layout

```text
.
├── README.md                  Project overview and quick start
├── README.zh-CN.md             Chinese overview
├── main.pdf                   Submitted manuscript
├── CITATION.cff               Machine-readable citation
├── CONTRIBUTING.md            Guidance for model and pipeline changes
├── THIRD_PARTY_NOTICES.md      Upstream sources and licensing status
├── docs/
│   └── DATA.md                Dataset access, layout, and semantics
├── code/
│   ├── train.py               Unified training entry point
│   ├── run.sh                 Environment setup and launch
│   ├── requirements.txt       Python dependencies
│   ├── config/                Training and backbone configurations
│   ├── models/                Ten backbones × Off/On entry points
│   ├── handoff/               Encrypted data loading and orchestration
│   ├── src/                   Adapters, training, and evaluation modules
│   ├── vendor/                Shared MEF mathematical components
│   ├── third_party/           Upstream revision and file-hash locks
│   └── tests/                 Existing focused tests
└── data_encrypted/            Local authorized data; excluded from Git
```

Upstream source trees are installed locally by `code/setup_sources.py` and excluded from Git. The local preparation copy already contains six permitted upstream subsets with their license files.

## Outputs and implementation status

Runs are written to `code/outputs/<dataset>/<model>/seed_<seed>/<run>/`. They include the run manifest, model weights, historical records, Memory and calibration artifacts, paired predictions, and ranking metrics. See [output details](code/README.md#4-输出与改进位置).

The newly connected full enhancement pipeline has **not yet undergone end-to-end training validation**. This distribution contains no verified reproduction of the submitted paper's numerical results. Engineering adaptations and their validation status are documented in [IMPLEMENTATION.md](code/IMPLEMENTATION.md).

The reordered training data contain some observations sourced from later periods. Results under this setup are diagnostic and must not be presented as independent out-of-sample performance. Reported ranking metrics are not a transaction-cost-adjusted trading backtest.

## Paper and citation

This repository accompanies **MEF-Rank: Model-Aware Action Memory for Financial Ranking** by Yifan Yuan, Yuanjian Zhang, Xiang Xu, and Chuanchang Liu. Read the [submitted manuscript](main.pdf) for the method, experimental protocol, and reported results.

Citation metadata are provided in [CITATION.cff](CITATION.cff). The manuscript is listed as submitted work; venue, DOI, and publication year will be added when available.

## License and acknowledgements

The project license is pending author selection. Third-party components retain their respective licenses; this repository does not grant additional rights to them. See [third-party notices](THIRD_PARTY_NOTICES.md) for the locked sources and redistribution status.
