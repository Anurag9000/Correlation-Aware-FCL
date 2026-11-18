#!/usr/bin/env python3
"""Run a CIFAR-10 validation-tuning plan sequentially and resumably on CUDA."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cafcl.config import RunConfig
from cafcl.simulator import atomic_json, run_experiment


def wait_for_exclusive_gpu(current_pid: int, poll_seconds: int) -> None:
    while True:
        probe = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        pids = {int(x.strip()) for x in probe.stdout.splitlines() if x.strip().isdigit()}
        pids.discard(current_pid)
        approved_cpu_only = {int(value) for value in
                             os.environ.get("CAFCL_GPU_CPU_ONLY_PIDS", "").split()
                             if value.isdigit()}
        blockers = []
        ignored = []
        for pid in sorted(pids):
            if pid in approved_cpu_only:
                ignored.append(pid)
                continue
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    errors="replace").split("\0")
            except OSError:
                continue
            if "--cpu-only" not in argv:
                blockers.append(pid)
        if not blockers:
            if ignored:
                print(f"Ignoring explicitly verified CPU-only contexts: {ignored}", flush=True)
            return
        print(f"Waiting for other GPU compute processes to exit: {blockers}; "
              f"CPU-only contexts ignored: {ignored}", flush=True)
        time.sleep(max(5, poll_seconds))


def gpu_snapshot() -> tuple[int, list[int]]:
    memory = subprocess.run(
        ["nvidia-smi", "-i", "0", "--query-gpu=memory.free",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    values = [int(line.strip()) for line in memory.stdout.splitlines()
              if line.strip().isdigit()]
    if not values:
        raise RuntimeError("Cannot read free VRAM for CUDA device 0")
    processes = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    pids = sorted({int(line.strip()) for line in processes.stdout.splitlines()
                   if line.strip().isdigit()})
    return values[0], pids


def wait_for_minimum_free_vram(minimum_gib: float, poll_seconds: int) -> None:
    minimum_mib = int(minimum_gib * 1024)
    while True:
        free_mib, _ = gpu_snapshot()
        if free_mib >= minimum_mib:
            return
        print(f"Waiting for VRAM headroom: {free_mib/1024:.1f} GiB free; "
              f"minimum is {minimum_gib:.1f} GiB", flush=True)
        time.sleep(max(5, poll_seconds))


def record_gpu_context(output: Path, cfg: RunConfig, event: str,
                       free_mib: int, pids: list[int], current_pid: int,
                       result_status: str | None = None) -> None:
    available_kib = None
    with open("/proc/meminfo", encoding="ascii") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                available_kib = int(line.split()[1])
                break
    row = {
        "job_id": cfg.job_id, "event": event, "unix_time": time.time(),
        "runner_pid": current_pid, "free_vram_mib": free_mib,
        "observed_compute_pids": [pid for pid in pids if pid != current_pid],
        "cpu_only_allowlisted_pids": sorted(
            int(value) for value in os.environ.get("CAFCL_GPU_CPU_ONLY_PIDS", "").split()
            if value.isdigit()),
        "mem_available_kib": available_kib,
    }
    if result_status is not None:
        row["result_status"] = result_status
    with (output / "execution_context.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
        fh.flush()


def wait_for_host_memory(minimum_gib: float, poll_seconds: int) -> None:
    """Keep a shared server from entering memory pressure between jobs."""
    minimum_kib = int(minimum_gib * 1024 * 1024)
    while True:
        available_kib = None
        with open("/proc/meminfo", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                    break
        if available_kib is None:
            raise RuntimeError("Cannot read MemAvailable for the tuning resource gate")
        if available_kib >= minimum_kib:
            return
        print(f"Waiting for host memory headroom: {available_kib/1024/1024:.1f} GiB available; "
              f"minimum is {minimum_gib:.1f} GiB", flush=True)
        time.sleep(max(5, poll_seconds))


def release_job_resources(torch, device) -> None:
    """Return per-job tensor and allocator memory before the next resource gate."""
    if torch.cuda.is_initialized():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    gc.collect()
    # CPython's allocator may retain freed NumPy/PyTorch backing arenas. Mtech
    # uses glibc; keep this optional so the runner remains portable elsewhere.
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def load_plan(path: Path) -> tuple[dict, list[RunConfig]]:
    payload = json.loads(path.read_text())
    if payload.get("dataset") != "cifar10" or payload.get("suite") != "validation_hyperparameter_tuning":
        raise ValueError(f"Not a CIFAR-10 validation tuning plan: {path}")
    jobs = [RunConfig(**item) for item in payload.get("jobs", [])]
    if not jobs or len({job.job_id for job in jobs}) != len(jobs):
        raise ValueError("Tuning plan must contain uniquely named jobs")
    for job in jobs:
        job.validate()
        if not job.job_id.startswith("TUNE_C10_") or job.batch_size != 512 or not job.require_cuda:
            raise ValueError(f"Tuning execution invariant failed: {job.job_id}")
        if job.fork_source_job:
            raise ValueError("Hyperparameter candidates may not depend on causal fork checkpoints")
    return payload, jobs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/cifar10_tuning_v1/job_plan.json")
    parser.add_argument("--output-dir", default="results/cifar10_tuning_v1")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    parser.add_argument("--minimum-memory-gib", type=float, default=8.0)
    parser.add_argument("--allow-concurrent-gpu", action="store_true",
                        help="Run alongside other CUDA processes when VRAM headroom permits")
    parser.add_argument("--minimum-free-vram-gib", type=float, default=4.0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    payload, jobs = load_plan(Path(args.plan))
    if args.plan_only:
        print(f"Validated {len(jobs)} sequential tuning jobs")
        return 0

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for tuning; no jobs were started")
    device = None
    device_name = None
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "tuning_status.json"
    completed = already_completed = 0
    failures = []
    for index, cfg in enumerate(jobs, 1):
        wait_for_host_memory(args.minimum_memory_gib, args.gpu_poll_seconds)
        if args.allow_concurrent_gpu:
            wait_for_minimum_free_vram(args.minimum_free_vram_gib, args.gpu_poll_seconds)
        else:
            wait_for_exclusive_gpu(os.getpid(), args.gpu_poll_seconds)
            wait_for_minimum_free_vram(args.minimum_free_vram_gib, args.gpu_poll_seconds)
        if device is None:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device)
        print(f"\n=== TUNING [{index}/{len(jobs)}] {cfg.job_id} (batch 512) ===", flush=True)
        free_mib, compute_pids = gpu_snapshot()
        record_gpu_context(output, cfg, "start", free_mib, compute_pids, os.getpid())
        try:
            result = run_experiment(cfg, str(output / "runs"), args.data_root, download=False)
            if result.get("status") == "already_completed":
                already_completed += 1
            else:
                completed += 1
            record_gpu_context(output, cfg, "end", *gpu_snapshot(), os.getpid(),
                               result_status=str(result.get("status", "completed")))
            del result
            release_job_resources(torch, device)
        except Exception as exc:
            end_free_mib, end_pids = gpu_snapshot()
            record_gpu_context(output, cfg, "failed", end_free_mib, end_pids,
                               os.getpid(), result_status=repr(exc))
            failures.append({"job_id": cfg.job_id, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            atomic_json(status_path, {
                "suite": payload["suite"], "planned": len(jobs), "attempted": index,
                "completed_this_invocation": completed, "already_completed": already_completed,
                "failed": failures, "cuda_device": device_name,
                "updated_unix": time.time(),
            })
            raise
        atomic_json(status_path, {
            "suite": payload["suite"], "planned": len(jobs), "attempted": index,
            "completed_this_invocation": completed, "already_completed": already_completed,
            "failed": failures, "cuda_device": device_name,
            "updated_unix": time.time(),
        })
    print(f"Tuning finished: completed={completed}, already_completed={already_completed}, failed=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
