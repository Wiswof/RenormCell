#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON="${RENORM_PYTHON:-python3.12}"
VENV="${RENORM_VENV:-.venv}"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv --without-pip "$VENV"
fi
if ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
  "$VENV/bin/python" -c 'import sys,urllib.request; urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py",sys.argv[1])' "$VENV/get-pip.py"
  "$VENV/bin/python" "$VENV/get-pip.py"
fi
if [[ ! -f "$VENV/.renormcell-ready" ]]; then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install torch==2.11.0 --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  "$VENV/bin/python" -m pip install -e .
  touch "$VENV/.renormcell-ready"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
"$VENV/bin/python" -m renormcell.assets --output "${RENORM_ASSETS:-.assets}"
exec "$VENV/bin/python" -u -m renormcell.predict --assets "${RENORM_ASSETS:-.assets}" "$@"
