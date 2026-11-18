#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG="$ROOT/logs/cifar10_full_validation_v2_launcher.log"
mkdir -p "$ROOT/logs" "$ROOT/results/cifar10_full_validation_v2"
exec > >(tee -a "$LOG") 2>&1

printf '[WAIT] Waiting for other CUDA suite controllers, GPU use, and RAM pressure to clear.\n'
stable=0
while (( stable < 4 )); do
  reason=""
  if pgrep -f '[s]cripts/essential_67.py run --device cuda' >/dev/null; then
    reason="essential_67 CUDA controller is active"
  fi

  allowed=""
  cpu_cmd=""
  if [[ -r /proc/15190/cmdline ]]; then
    cpu_cmd="$(tr '\0' ' ' </proc/15190/cmdline)"
  fi
  if [[ "$cpu_cmd" == *"/tmp/opf_cpu_resume_20261105.py"* ]] &&
     grep -Fq '"--cpu-only"' /tmp/opf_cpu_resume_20261105_args.json 2>/dev/null; then
    allowed="15190"
  fi

  gpu_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr '\n' ' ')"
  for pid in $gpu_pids; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if [[ " $allowed " != *" $pid "* ]]; then
      reason="CUDA compute process $pid is active"
    fi
  done

  available_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
  if (( available_kib < 8*1024*1024 )); then
    reason="only $((available_kib/1024/1024)) GiB MemAvailable"
  fi

  if [[ -z "$reason" ]]; then
    stable=$((stable+1))
    printf '[IDLE] Stable exclusive-GPU/RAM sample %d/4.\n' "$stable"
  else
    stable=0
    printf '[WAIT] %s.\n' "$reason"
  fi
  sleep 30
done

printf '[START] Launching the resumable six-candidate full-data validation.\n'
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export CAFCL_GPU_CPU_ONLY_PIDS="${allowed:-}"
export PATH="$ROOT/.venv/bin:$PATH"
exec "$ROOT/.venv/bin/python" -u scripts/run_cifar10_tuning.py \
  --plan results/cifar10_full_validation_v2/job_plan.json \
  --output-dir results/cifar10_full_validation_v2 \
  --gpu-poll-seconds 30 --minimum-memory-gib 8
