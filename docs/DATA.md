# Data access and setup

[Back to project](../README.md) · [中文训练指南](../code/README.md)

## Access

The five datasets are distributed separately to authorized recipients. Dataset files and decryption keys are not included in the Git repository. Obtain the selected encrypted package and its matching binary `.key` file directly from the data owner; each market uses a separate key.

Place the package alongside `code/`, preserving all file names:

```text
data_encrypted/
└── gem/
    ├── manifest.enc
    ├── 1_str_mapping.enc
    ├── 1_memory_mapping.enc
    ├── summary.json
    └── … numbered encrypted shards
```

Keep the key outside the repository. Then run from the repository root:

```bash
bash code/run.sh --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

For data on a separate disk, pass the directory containing the market subdirectories:

```bash
bash code/run.sh --data-dir /mnt/mef-data --dataset gem --model STR --key-file /secure/keys/gem.key --device cuda
```

## Dataset identifiers

| Identifier | Market |
| --- | --- |
| `gem` | China A-share ChiNext |
| `main` | China A-share main board |
| `nasdaq100` | NASDAQ-100 |
| `sp500` | S&P 500 |
| `crypto` | Crypto-15 |

## Views and time semantics

| View | Purpose |
| --- | --- |
| `1_str` | Reordered variant 1 for final predictor training, 2018–2022 |
| `1_memory` | Reordered variant 1 for five-year rolling historical predictors and Memory construction |
| `evaluation` | Original observations: 2022 window context, 2023 validation, and 2024–2026-07-08 test |

Off and On use the same variant and final predictor weights. Do not substitute variant 0 or a sequential variant 1 dataset. Target/source date mappings are included in encrypted form.

The reordered views contain some later-period source observations. Their results are diagnostic rather than independent out-of-sample evidence. See the [implementation notes](../code/IMPLEMENTATION.md) for availability clocks, label maturity, and calibration support requirements.

## Confidentiality

Data are de-identified, serialized as compressed Arrow IPC, and encrypted with AES-256-GCM. The loader decrypts data in memory without creating plaintext dataset caches. Asset aliases are consistent across views within each market.

Encryption protects stored and transferred files. An authorized recipient who can train locally can also extract decrypted numeric values; de-identification does not guarantee that re-identification is impossible. Predictions and other training artifacts can contain sensitive values and labels. Keep data, keys, and run artifacts out of public commits and issue attachments.
