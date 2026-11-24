#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CAFCL_PYTHON_BIN="${CAFCL_PYTHON_BIN:-}"
if [[ -z "$CAFCL_PYTHON_BIN" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    CAFCL_PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python >/dev/null 2>&1; then
    CAFCL_PYTHON_BIN=python
  elif command -v python3 >/dev/null 2>&1; then
    CAFCL_PYTHON_BIN=python3
  else
    printf '%s\n' 'No Python interpreter found (checked .venv/bin/python, python, python3).' >&2
    exit 127
  fi
fi
"$CAFCL_PYTHON_BIN" -m cafcl.primary_probe "$@"
