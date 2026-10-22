#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${RENORM_PYTHON:-python3.12}"
VENV="${RENORM_VENV:-.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install torch==2.11.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  "$VENV/bin/python" -m pip install -e .
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"$VENV/bin/python" -m renormcell.assets --output "${RENORM_ASSETS:-.assets}"
exec "$VENV/bin/python" -u -m renormcell.predict --assets "${RENORM_ASSETS:-.assets}" "$@"
