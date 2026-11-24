#!/usr/bin/env python3
"""Create fresh local-disk A-G worker specs and rerun schedule-only validation."""
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path("/home/anurag24092/Projects/Correlation-Aware-FCL-independent-ablations-20261122")
OLD = Path("/dev/shm/cafcl-ag-recovery-v4")
NEW = Path("/dev/shm/cafcl-ag-recovery-v5")
OUT = REPO / "results" / "recovery_local_20261123" / "independent_ablation_A_G"
RUNNER = REPO / "compiler" / "cafcl_independent_ablation_runner.py"
FAILED = {
    "B_recur_gap_5_seed0": ("attempt_06", "attempt_07"),
    "B_recur_gap_10_seed0": ("attempt_05", "attempt_06"),
    "C_hidden_2_seed0": ("attempt_05", "attempt_06"),
    "D_cross_step_2_seed0": ("attempt_03", "attempt_04"),
    "D_cross_step_3_seed0": ("attempt_03", "attempt_04"),
}


def main():
    state = json.loads((OLD / "mtech_ag_vram_scheduler_state.json").read_text())
    for old_job in state.get("jobs", []):
        pid = old_job.get("pid")
        if not pid:
            continue
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            continue
        raise RuntimeError("Refusing to prepare while prior A-G worker PID is alive: " + str(pid))

    original = json.loads((OLD / "queue.json").read_text())
    NEW.mkdir(parents=True, exist_ok=True)
    jobs = []
    for item in original["jobs"]:
        cell = item["cell_id"]
        if cell in FAILED:
            prior, attempt = FAILED[cell]
            prior_attempts = list(item.get("prior_attempts", []))
            if prior not in prior_attempts:
                prior_attempts.append(prior)
        else:
            attempt = item["attempt_id"]
            prior_attempts = list(item.get("prior_attempts", []))

        run_id = cell + "__" + attempt
        output = OUT / cell / attempt / "run"
        spec_dir = NEW / "specs" / cell / attempt / "spec"
        spec_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(item["spec_dir"], spec_dir)
        plan_path = spec_dir / "plan_validation.json"
        plan = json.loads(plan_path.read_text())
        plan["run_id"] = run_id
        plan["output_dir"] = str(output)
        plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")

        config = json.loads((spec_dir / "config.json").read_text())
        variant = "taskil" if config.get("learning_setting", "task_il") == "task_il" else "classil"
        runtime = REPO / ("runtime_baseline_" + variant)
        python = runtime / ".venv" / "bin" / "python"
        tmp = Path("/dev/shm/cafcl-tmp")
        for d in (tmp, Path("/dev/shm/cafcl-home"), Path("/dev/shm/cafcl-runtime-cache")):
            d.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update({
            "HOME": "/dev/shm/cafcl-home",
            "TMPDIR": str(tmp),
            "XDG_CACHE_HOME": "/dev/shm/cafcl-runtime-cache",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(runtime / "src"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "CAFCL_VRAM_BROKER_ADMITTED": "1",
        })
        proc = subprocess.run(
            [str(python), str(RUNNER), "--worker-spec", str(spec_dir), "--plan-only"],
            cwd=str(REPO), env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=True,
        )
        validation = json.loads(proc.stdout)
        if validation.get("run_id") != run_id:
            raise RuntimeError("plan-only run_id mismatch: " + cell)
        runtime_path = spec_dir.parent / "runtime_plan_validation.json"
        runtime_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
        entry = dict(item)
        entry.update({
            "attempt_id": attempt,
            "prior_attempts": prior_attempts,
            "run_id": run_id,
            "output_dir": str(output),
            "spec_dir": str(spec_dir),
            "runtime_plan_validation_file": "runtime_plan_validation.json",
            "runtime_plan_validation_sha256": hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "output_storage": "Mtech local filesystem; no FUSE output mount",
            "plan_validation_status": "runner_plan_validated",
        })
        jobs.append(entry)

    queue = dict(original)
    queue.update({
        "created_at_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "jobs": jobs,
        "storage_mount": str(OUT),
        "recovery_note": "Fresh attempt IDs for five EIO failures first; nine not-yet-admitted jobs follow. New unique local-disk outputs; original FUSE attempts retained.",
        "queue_version": "mtech_ag_local_disk_retry_v1",
    })
    qpath = NEW / "queue.json"
    qpath.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "prepared",
        "jobs": len(jobs),
        "priority_failed_cells": list(FAILED),
        "output_root": str(OUT),
        "queue": str(qpath),
        "queue_sha256": hashlib.sha256(qpath.read_bytes()).hexdigest(),
        "plan_validation_count": len(jobs),
        "failed_cells_first": [x["cell_id"] for x in jobs[:5]],
        "remaining_jobs": [x["cell_id"] for x in jobs[5:]],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
