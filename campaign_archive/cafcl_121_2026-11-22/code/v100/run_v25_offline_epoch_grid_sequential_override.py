#!/usr/bin/env python3
"""Resume the corrected V100 epoch grid serially without a VRAM gate.

Uses the already validated V2 run plans and their output directories. The
separate override manifest preserves this launch history without racing the
older controller's manifest writes. Each worker uses the campaign-wide lock
in run_v25_offline_epoch_grid.py, which also makes an old controller child
idempotent if it wakes and reaches one of these same run directories.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "results" / "v25_offline_epoch_grid_20261122_v2"
SOURCE_ROOT = REPO / "source_variants"
WORKER_SCRIPT = Path(__file__).with_name("run_v25_offline_epoch_grid.py")
OVERRIDE = CAMPAIGN / "sequential_override"
MANIFEST = OVERRIDE / "manifest.json"


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> int:
    original_path = CAMPAIGN / "manifest.json"
    if not original_path.is_file():
        raise SystemExit("Corrected V2 manifest not found: {}".format(original_path))
    original = json.loads(original_path.read_text())
    rows = original.get("runs", [])
    if len(rows) != 48 or len({row["job_id"] for row in rows}) != 48:
        raise SystemExit("V2 manifest must contain 48 unique planned runs")

    OVERRIDE.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        state = json.loads(MANIFEST.read_text())
        if state.get("status") == "completed":
            print("Sequential override already completed", flush=True)
            return 0
    else:
        state = {
            "campaign": original.get("campaign"),
            "mode": "one worker at a time; no free-VRAM admission threshold",
            "source_manifest": str(original_path),
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "queued",
            "run_count": len(rows),
            "runs": [{"job_id": row["job_id"], "status": "queued",
                      "output_dir": row["output_dir"]} for row in rows],
        }
    by_id = {row["job_id"]: row for row in state["runs"]}
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1"})

    state["status"] = "running"
    state["started_utc"] = state.get("started_utc") or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(MANIFEST, state)
    for row in rows:
        run = by_id[row["job_id"]]
        if run.get("status") == "completed":
            continue
        output = Path(row["output_dir"])
        worker_status = output / "worker_status.json"
        summary = output / "result_summary.json"
        if worker_status.exists():
            previous = json.loads(worker_status.read_text())
            if previous.get("status") == "completed" and summary.exists():
                run["status"] = "completed"
                run["completion_source"] = "existing worker artifacts"
                atomic_json(MANIFEST, state)
                continue
            if previous.get("status") in {"failed", "running"}:
                run["status"] = "blocked_existing_worker_state"
                run["detail"] = previous
                state["status"] = "blocked"
                atomic_json(MANIFEST, state)
                raise SystemExit("Refusing to overwrite {} worker state for {}".format(
                    previous.get("status"), row["job_id"]))

        setting = row["setting"]
        source = SOURCE_ROOT / ("taskil_src" if setting == "task_il" else "classil_src")
        worker_env = env.copy()
        worker_env["PYTHONPATH"] = str(source / "src")
        worker_env["CAFCL_SOURCE_DIR"] = str(source)
        cmd = [sys.executable, str(WORKER_SCRIPT), "--worker",
               "--setting", setting,
               "--curriculum", row["curriculum"],
               "--offline-count", str(row["offline_count"]),
               "--epochs", str(row["epochs_per_task"]),
               "--job-id", row["job_id"],
               "--block-id", row["block_id"],
               "--output-dir", row["output_dir"]]
        log_path = OVERRIDE / "logs" / (row["job_id"] + ".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        run.update({"status": "running", "started_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pid": None,
            "log": str(log_path)})
        state["active_job"] = row["job_id"]
        atomic_json(MANIFEST, state)
        with log_path.open("ab") as log:
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=worker_env,
                                    stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            run["pid"] = proc.pid
            atomic_json(MANIFEST, state)
            code = proc.wait()
        run["return_code"] = code
        run["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        run["status"] = "completed" if code == 0 else "failed"
        atomic_json(MANIFEST, state)
        print("{}: {} (exit {})".format(row["job_id"], run["status"], code),
              flush=True)

    failed = [run["job_id"] for run in state["runs"]
              if run.get("status") != "completed"]
    state.pop("active_job", None)
    state["failed_runs"] = failed
    state["status"] = "completed_with_failures" if failed else "completed"
    state["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(MANIFEST, state)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
