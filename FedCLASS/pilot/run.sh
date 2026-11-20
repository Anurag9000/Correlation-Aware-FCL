#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
if [[ -n "${CAFCL_PYTHON:-}" ]]; then
  PYTHON_BIN="$CAFCL_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 interpreter not found; set CAFCL_PYTHON or create $PROJECT_ROOT/.venv" >&2
  exit 127
fi
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
exec "$PYTHON_BIN" -m cafcl.fedclass_primary --gpu 0 "$@"
