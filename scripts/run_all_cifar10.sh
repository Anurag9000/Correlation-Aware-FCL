#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${CAFCL_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi
exec "$PYTHON_BIN" -m cafcl.cli suite --dataset cifar10 --suite all-supported --download \
  --data-root "${DATA_ROOT:-$ROOT/data}" \
  --output-dir "${RESULTS_ROOT:-$ROOT/results}" "$@"
