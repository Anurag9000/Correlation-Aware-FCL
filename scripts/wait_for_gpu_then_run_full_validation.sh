#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/cifar10_full_validation_v2_launcher.log"
mkdir -p "$ROOT/logs" "$ROOT/results/cifar10_full_validation_v2"
exec > >(tee -a "$LOG") 2>&1

printf '[WAIT] Waiting for VRAM and RAM headroom; concurrent CUDA work is authorized.\n'
stable=0
while (( stable < 1 )); do
  reason=""
  free_vram_mib="$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)"
  if [[ ! "$free_vram_mib" =~ ^[0-9]+$ ]]; then
    reason="cannot read free VRAM on GPU 0"
  elif (( free_vram_mib < 4096 )); then
    reason="only $((free_vram_mib/1024)) GiB free VRAM"
  fi

  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_kib < 6*1024*1024 )); then
    reason="only $((available_kib/1024/1024)) GiB MemAvailable"
  fi

  if [[ -z "$reason" ]]; then
    stable=$((stable+1))
    gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr '\n' ' ')"
    printf '[READY] Free VRAM %d MiB; MemAvailable %d GiB; concurrent CUDA PIDs: %s\n' \
      "$free_vram_mib" "$((available_kib/1024/1024))" "${gpu_pids:-none}"
  else
    stable=0
    printf '[WAIT] %s.\n' "$reason"
  fi
  if (( stable < 1 )); then sleep 30; fi
done

printf '[PREFLIGHT] Running the complete test suite with CUDA hidden.\n'
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/pytest" -q
printf '[PREFLIGHT] Validating committed CIFAR plans.\n'
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" scripts/validate_plans.py
CUDA_VISIBLE_DEVICES='' bash scripts/run_all_cifar10.sh --plan-only \
  --stop-on-error --wait-for-exclusive-gpu --minimum-memory-gib 6
CUDA_VISIBLE_DEVICES='' "$ROOT/.venv/bin/python" scripts/run_cifar10_tuning.py \
  --plan results/cifar10_full_validation_v2/job_plan.json \
  --output-dir results/cifar10_full_validation_v2 --plan-only

printf '[START] Launching the resumable six-candidate full-data validation.\n'
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PATH="$ROOT/.venv/bin:$PATH"
exec "$ROOT/.venv/bin/python" -u scripts/run_cifar10_tuning.py \
  --plan results/cifar10_full_validation_v2/job_plan.json \
  --output-dir results/cifar10_full_validation_v2 \
  --gpu-poll-seconds 30 --minimum-memory-gib 6 \
  --allow-concurrent-gpu --minimum-free-vram-gib 4
