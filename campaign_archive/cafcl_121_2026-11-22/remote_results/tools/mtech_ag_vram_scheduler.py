#!/usr/bin/env python3
"""Dynamically admit the prevalidated Mtech A-G retries by measured VRAM."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PREFLIGHT_ONLY = "--preflight-only" in sys.argv[1:]
ARGS = [arg for arg in sys.argv[1:] if arg != "--preflight-only"]
QUEUE = Path(ARGS[0]) if ARGS else Path(
    "/dev/shm/cafcl-kakarot-results/mtech/independent_ablation_A_G/recovery/"
    "AG_retry_dispatch_queue_20261122_2135.json"
)
REPO = Path(
    "/home/anurag24092/Projects/Correlation-Aware-FCL-independent-ablations-20261122"
)
RUNNER = REPO / "compiler/cafcl_independent_ablation_runner.py"
BROKER_HOME = Path("/dev/shm/cafcl-broker-home")
STATE_DIR = Path("/dev/shm/cafcl-ag-recovery")
PEAK_MIB = 6186
MARGIN_MIB = 1024
POLL_SECONDS = 20


def atomic(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def gpu_free() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [int(x.strip()) for x in out.splitlines() if x.strip()]
    if len(values) != 1:
        raise RuntimeError(f"expected one GPU, got free-memory rows {values}")
    return values[0]


def gpu_by_pid() -> dict[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
    result: dict[int, int] = {}
    for line in out.splitlines():
        try:
            pid, mib = (int(x.strip()) for x in line.split(",", 1))
            result[pid] = mib
        except (ValueError, TypeError):
            continue
    return result


def descendants(pid: int) -> set[int]:
    rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
    children: dict[int, list[int]] = {}
    for row in rows:
        try:
            child, parent = map(int, row.split())
            children.setdefault(parent, []).append(child)
        except ValueError:
            continue
    found, todo = {pid}, [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


def validate(job: dict) -> tuple[Path, Path, Path, str]:
    if int(job.get("expected_peak_vram_mib", 0)) != PEAK_MIB:
        raise RuntimeError(f"unexpected VRAM estimate for {job.get('cell_id')}")
    if int(job.get("margin_mib", 0)) != MARGIN_MIB:
        raise RuntimeError(f"unexpected VRAM margin for {job.get('cell_id')}")
    spec_dir = Path(job["spec_dir"])
    output = Path(job["output_dir"])
    config = json.loads((spec_dir / "config.json").read_text())
    plan = json.loads((spec_dir / "plan_validation.json").read_text())
    runtime = spec_dir.parent / "runtime_plan_validation.json"
    digest = hashlib.sha256(runtime.read_bytes()).hexdigest()
    if digest != job["runtime_plan_validation_sha256"]:
        raise RuntimeError(f"runtime plan checksum mismatch: {job['cell_id']}")
    if plan.get("run_id") != job["run_id"] or plan.get("output_dir") != str(output):
        raise RuntimeError(f"plan identity/output mismatch: {job['cell_id']}")
    if config.get("seed") != 0:
        raise RuntimeError(f"seed mismatch: {job['cell_id']}")
    if not RUNNER.is_file():
        raise RuntimeError(f"campaign worker missing: {RUNNER}")
    if output.exists():
        entries = list(output.iterdir())
        if entries:
            status = output / "worker_status.json"
            if status.exists():
                state = json.loads(status.read_text()).get("status")
                raise RuntimeError(f"attempt output already has status {state}: {output}")
            raise RuntimeError(f"attempt output is nonempty; refusing overwrite: {output}")
    variant = "taskil" if config.get("learning_setting", "task_il") == "task_il" else "classil"
    runtime_root = REPO / f"runtime_baseline_{variant}"
    python = runtime_root / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"runtime interpreter missing: {python}")
    env = os.environ.copy()
    env.update({
        "HOME": "/dev/shm/cafcl-home",
        "TMPDIR": "/dev/shm/cafcl-tmp",
        "XDG_CACHE_HOME": "/dev/shm/cafcl-runtime-cache",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(runtime_root / "src"),
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CAFCL_VRAM_BROKER_ADMITTED": "1",
    })
    return spec_dir, output, python, env


def main() -> int:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    BROKER_HOME.mkdir(parents=True, exist_ok=True)
    queue = json.loads(QUEUE.read_text())
    jobs = queue.get("jobs", [])
    if len(jobs) != 16:
        raise RuntimeError(f"expected 16 validated retry specs, found {len(jobs)}")
    prepared = []
    for job in jobs:
        spec, output, python, env = validate(job)
        prepared.append((job, spec, output, python, env))
    if PREFLIGHT_ONLY:
        print(json.dumps({"status": "preflight_ok", "jobs": len(prepared),
                          "output_paths": [str(item[2]) for item in prepared]}, indent=2))
        return 0
    state_path = STATE_DIR / "mtech_ag_vram_scheduler_state.json"
    state = {
        "campaign": "independent_ablation_A_G",
        "queue": str(QUEUE),
        "status": "running",
        "peak_estimate_mib": PEAK_MIB,
        "margin_mib": MARGIN_MIB,
        "admission_rule": "measured free VRAM minus outstanding reservations >= one worker peak plus one aggregate margin",
        "validated_jobs": len(prepared),
        "jobs": [],
        "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    active: dict[int, tuple[subprocess.Popen, dict, Path]] = {}
    pending = list(prepared)
    atomic(state_path, state)
    while pending or active:
        for pid, (proc, job, output) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            status_path = output / "worker_status.json"
            worker_status = None
            if status_path.is_file():
                try:
                    worker_status = json.loads(status_path.read_text()).get("status")
                except (OSError, json.JSONDecodeError):
                    pass
            rec = next(x for x in state["jobs"] if x["pid"] == pid)
            rec.update({"broker_exit_code": code, "worker_status": worker_status,
                        "finished_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            rec["state"] = "completed" if code == 0 and worker_status in {"completed", "finished"} else "needs_audit"
            del active[pid]
            atomic(state_path, state)

        free = gpu_free()
        mem = gpu_by_pid()
        outstanding = 0
        for pid, (_, _, _) in active.items():
            observed = sum(mem.get(q, 0) for q in descendants(pid))
            outstanding += max(0, PEAK_MIB - observed)
        launched = 0
        while pending and free - outstanding >= PEAK_MIB + MARGIN_MIB:
            job, spec, output, python, env = pending.pop(0)
            output.mkdir(parents=True, exist_ok=True)
            log = output / "worker.log"
            args = [str(python), str(RUNNER), "--worker-spec", str(spec)]
            with log.open("ab") as log_file:
                proc = subprocess.Popen(args, cwd=str(REPO), env=env,
                                        stdin=subprocess.DEVNULL, stdout=log_file,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            rec = {"cell_id": job["cell_id"], "attempt_id": job["attempt_id"],
                   "run_id": job["run_id"], "pid": proc.pid, "state": "running",
                   "output_dir": str(output), "spec_dir": str(spec),
                   "prelaunch_free_vram_mib": free, "peak_estimate_mib": PEAK_MIB,
                   "aggregate_safety_margin_mib": MARGIN_MIB,
                   "started_local": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            state["jobs"].append(rec)
            active[proc.pid] = (proc, job, output)
            launched += 1
            outstanding += PEAK_MIB
            atomic(state_path, state)
            free = gpu_free()
            mem = gpu_by_pid()
            observed = sum(mem.get(q, 0) for q in descendants(proc.pid))
            outstanding = max(0, outstanding - PEAK_MIB) + max(0, PEAK_MIB - observed)
        state["last_sample"] = {
            "local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "free_vram_mib": free,
            "outstanding_reserved_mib": outstanding,
            "active_pids": sorted(active),
            "pending_jobs": len(pending),
            "launched_this_pass": launched,
        }
        atomic(state_path, state)
        if pending or active:
            time.sleep(POLL_SECONDS)
    state["status"] = "finished_with_audit_items" if any(
        j["state"] != "completed" for j in state["jobs"]
    ) else "completed"
    state["finished_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    atomic(state_path, state)
    return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
