#!/usr/bin/env python3
"""Run a generated CIFAR-10 GPU pilot plan sequentially and resumably."""
from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cafcl.config import RunConfig
from cafcl.simulator import atomic_json, run_experiment


def load_plan(path: Path) -> tuple[dict, list[RunConfig]]:
    payload = json.loads(path.read_text())
    if payload.get("dataset") != "cifar10" or payload.get("suite") != "representative_gpu_pilot":
        raise ValueError(f"Not a CIFAR-10 pilot plan: {path}")
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("Pilot plan must contain at least one job")
    jobs = [RunConfig(**item) for item in raw_jobs]
    ids = [job.job_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Pilot job identifiers are not unique")
    seen: set[str] = set()
    for job in jobs:
        job.validate()
        if not job.job_id.startswith("PILOT_"):
            raise ValueError(f"Pilot job ID lacks PILOT_ prefix: {job.job_id}")
        if job.dataset != "cifar10" or job.batch_size != 64 or not job.require_cuda:
            raise ValueError(f"Pilot GPU/data invariant failed: {job.job_id}")
        if job.train_fraction != 0.15 or job.rounds_per_stage != 2 or job.epochs != 1:
            raise ValueError(f"Pilot quick-run invariant failed: {job.job_id}")
        if job.fork_source_job:
            if job.fork_source_job not in seen:
                raise ValueError(f"Fork source must precede dependent job: {job.job_id}")
            if job.fork_checkpoint_round != 2:
                raise ValueError(f"Pilot fork checkpoint must be round 2: {job.job_id}")
        seen.add(job.job_id)
    return payload, jobs


def wait_for_exclusive_gpu(current_pid: int, status_path: Path, *,
                           planned: int, attempted: int, completed: int,
                           already_completed: int, poll_seconds: int = 30) -> None:
    """Wait between jobs while another process owns a CUDA compute context."""
    while True:
        probe = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True)
        pids = sorted({int(value.strip()) for value in probe.stdout.splitlines()
                       if value.strip().isdigit()} - {current_pid})
        approved_cpu_only = {int(value) for value in
                             os.environ.get("CAFCL_GPU_CPU_ONLY_PIDS", "").split()
                             if value.isdigit()}
        gpu_pids = []
        cpu_only_pids = []
        for pid in pids:
            if pid in approved_cpu_only:
                cpu_only_pids.append(pid)
                continue
            try:
                argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                    errors="replace").split("\0")
            except OSError:
                argv = []
            if "--cpu-only" in argv:
                cpu_only_pids.append(pid)
            else:
                gpu_pids.append(pid)
        if not gpu_pids:
            if cpu_only_pids:
                print(f"Ignoring explicitly CPU-only CUDA contexts: {cpu_only_pids}", flush=True)
            return
        atomic_json(status_path, {
            "suite": "representative_gpu_pilot", "status": "waiting_for_exclusive_gpu",
            "planned": planned, "attempted": attempted,
            "completed_this_invocation": completed,
            "already_completed": already_completed,
            "foreign_compute_pids": gpu_pids,
            "ignored_cpu_only_context_pids": cpu_only_pids,
            "updated_unix": time.time(),
        })
        print(f"Waiting for other GPU compute processes to exit: {gpu_pids}; "
              f"CPU-only contexts ignored: {cpu_only_pids}", flush=True)
        time.sleep(max(5, poll_seconds))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/cifar10_gpu_pilot_v1/job_plan.json")
    parser.add_argument("--output-dir", default="results/cifar10_gpu_pilot_v1")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--gpu-poll-seconds", type=int, default=30)
    args = parser.parse_args()

    plan_path = Path(args.plan)
    output_root = Path(args.output_dir)
    payload, jobs = load_plan(plan_path)
    if args.plan_only:
        print(f"Validated {len(jobs)} sequential pilot jobs from {plan_path}")
        return 0

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this pilot; no jobs were started")
    device = torch.device("cuda")
    print(f"CUDA pilot GPU: {torch.cuda.get_device_name(device)}", flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "pilot_status.json"
    failures: list[dict] = []
    completed = already_completed = 0

    for index, config in enumerate(jobs, start=1):
        wait_for_exclusive_gpu(os.getpid(), status_path, planned=len(jobs),
                               attempted=index-1, completed=completed,
                               already_completed=already_completed,
                               poll_seconds=args.gpu_poll_seconds)
        print(f"\n=== PILOT [{index}/{len(jobs)}] {config.job_id} (batch 64) ===", flush=True)
        source = None
        if config.fork_source_job:
            source = (output_root / "runs" / config.fork_source_job /
                      f"checkpoint_round_{config.fork_checkpoint_round:04d}.pt")
            if not source.is_file():
                error = f"Required pilot fork checkpoint missing: {source}"
                failures.append({"job_id": config.job_id, "error": error})
                atomic_json(status_path, {
                    "suite": payload["suite"], "planned": len(jobs), "attempted": index,
                    "completed_this_invocation": completed,
                    "already_completed": already_completed, "failures": failures,
                    "batch_size": 64, "cuda_device": torch.cuda.get_device_name(device),
                })
                print(error, file=sys.stderr, flush=True)
                return 1
        try:
            result = run_experiment(config, str(output_root / "runs"), args.data_root,
                                    download=False, fork_from_checkpoint=source)
            if result.get("status") == "already_completed":
                already_completed += 1
            else:
                completed += 1
            state = "completed" if result.get("status") != "already_completed" else "already_completed"
            print(f"{state}: {config.job_id}", flush=True)
        except Exception as exc:
            failures.append({"job_id": config.job_id, "error": repr(exc),
                             "traceback": traceback.format_exc()})
            atomic_json(status_path, {
                "suite": payload["suite"], "planned": len(jobs), "attempted": index,
                "completed_this_invocation": completed,
                "already_completed": already_completed, "failures": failures,
                "batch_size": 64, "cuda_device": torch.cuda.get_device_name(device),
                "updated_unix": time.time(),
            })
            print(f"FAILED {config.job_id}: {exc}", file=sys.stderr, flush=True)
            return 1
        atomic_json(status_path, {
            "suite": payload["suite"], "planned": len(jobs), "attempted": index,
            "completed_this_invocation": completed,
            "already_completed": already_completed, "failures": failures,
            "batch_size": 64, "cuda_device": torch.cuda.get_device_name(device),
            "updated_unix": time.time(),
        })

    print(f"Pilot finished: completed={completed}, already_completed={already_completed}, "
          f"failed={len(failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
