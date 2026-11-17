#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$ROOT/scripts/run_all_cifar10.sh" "$@"
bash "$ROOT/scripts/run_all_cifar100.sh" "$@"
