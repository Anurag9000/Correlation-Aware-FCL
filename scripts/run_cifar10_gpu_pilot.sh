#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"
mkdir -p "$ROOT/logs" "$ROOT/results/cifar10_gpu_pilot_v1"
exec > >(tee -a "$ROOT/logs/cifar10_gpu_pilot_v1.log") 2>&1

while true; do
  active="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader)"
  if [[ -z "$active" ]]; then
    break
  fi
  printf '[WAIT] CUDA is occupied by another process:\n%s\n' "$active"
  sleep 30
done

printf '[PREFLIGHT] Running interrupted CUDA resume regression.\n'
.venv/bin/python -m pytest -q \
  tests/test_advanced.py::test_cuda_run_and_interrupted_resume_are_identical
.venv/bin/python scripts/record_cifar10_environment.py \
  --output results/cifar10_gpu_pilot_v1/environment.json --data-root data

printf '[PILOT] Starting the sequential 71-job CIFAR-10 pilot.\n'
exec .venv/bin/python scripts/run_cifar10_pilot.py \
  --plan results/cifar10_gpu_pilot_v1/job_plan.json \
  --output-dir results/cifar10_gpu_pilot_v1 --data-root data
