#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# Default: original targeted jobs plus full finite-axis factorial sweep.
python -m cafcl.full_matrix --dataset cifar100 --suite full --download \
  --data-root "${DATA_ROOT:-$ROOT/data}" \
  --output-dir "${RESULTS_ROOT:-$ROOT/results}" "$@"
