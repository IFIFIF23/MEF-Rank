#!/usr/bin/env bash
set -euo pipefail
MEF_CODE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$MEF_CODE_DIR"
MEF_PYTHON="${MEF_PYTHON:-$MEF_CODE_DIR/.venv/bin/python}"
if [[ ! -x "$MEF_PYTHON" ]]; then
  python3 -m venv "$MEF_CODE_DIR/.venv"
fi
if [[ "$MEF_PYTHON" == "$MEF_CODE_DIR/.venv/bin/python" && ! -f "$MEF_CODE_DIR/.venv/.requirements-installed" ]]; then
  "$MEF_PYTHON" -m pip install -r requirements.txt
  touch "$MEF_CODE_DIR/.venv/.requirements-installed"
fi
"$MEF_PYTHON" setup_sources.py
exec "$MEF_PYTHON" train.py "$@"
