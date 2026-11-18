#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
python -m cafcl.primary_agem --data-root "${DATA_ROOT:-$ROOT/data}" \
  --output-dir "${RESULTS_ROOT:-$ROOT/results/primary_agem_cifar10}" "$@"
