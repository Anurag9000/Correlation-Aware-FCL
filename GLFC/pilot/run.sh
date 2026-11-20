#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -n "${CAFCL_PYTHON:-}" ]]; then
  PYTHON_BIN="$CAFCL_PYTHON"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 interpreter not found; set CAFCL_PYTHON or create $ROOT/.venv" >&2
  exit 127
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
exec "$PYTHON_BIN" -m cafcl.native_primary --suite glfc --gpu 0 "$@"
