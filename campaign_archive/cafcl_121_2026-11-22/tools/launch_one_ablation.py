#!/usr/bin/env python3
"""Safely admit one validated A-G worker on the shared Mtech V100.

This helper deliberately launches only one worker. It does not alter or stop
other processes, and it stores launcher metadata outside the result folder so
the experiment runner can create its own files there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "compiler" / "cafcl_independent_ablation_runner.py"
DEFAULT_PLAN_ROOT = ROOT / "plans" / "server_compiled_v4"
MIN_FREE_MIB = 8192


def gpu_free_mib() -> int:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(output.splitlines()[0].strip())


def active_workers() -> list[str]:
    output = subprocess.check_output(["ps", "-eo", "args="], text=True)
    needle = f"{RUNNER} --worker-spec"
    return [line.strip() for line in output.splitlines() if needle in line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="validated run ID, for example A_clients_2_task_il_seed0")
    parser.add_argument("--plan-root", type=Path, default=DEFAULT_PLAN_ROOT)
    parser.add_argument("--minimum-free-vram-mib", type=int, default=MIN_FREE_MIB)
    args = parser.parse_args()

    if not re.fullmatch(r"[A-G]_[a-z0-9_]+_seed0", args.run_id):
        parser.error("run ID must match the one-seed A-G campaign format")
    family, suffix = args.run_id.split("_", 1)
    spec = args.plan_root / family / suffix
    for name in ("config.json", "schedule.json", "plan_validation.json"):
        if not (spec / name).is_file():
            raise SystemExit(f"missing plan input: {spec / name}")
    plan = json.loads((spec / "plan_validation.json").read_text())
    config = json.loads((spec / "config.json").read_text())
    if plan.get("run_id") != args.run_id or plan.get("schedule_validated") is not True:
        raise SystemExit("plan validation does not match the requested run ID")
    if config.get("seed") != 0:
        raise SystemExit("refusing a run that does not use the campaign seed 0")

    output_dir = ROOT / "results" / family / args.run_id
    status_path = output_dir / "worker_status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        raise SystemExit(f"run already has status {status.get('status')}: {status_path}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"refusing nonempty output directory without worker status: {output_dir}")
    record_dir = ROOT / "queue" / "manual_worker_launches"
    record_dir.mkdir(parents=True, exist_ok=True)
    record_path = record_dir / f"{args.run_id}.json"
    if record_path.exists():
        raise SystemExit(f"refusing to replace launch record: {record_path}")

    live = active_workers()
    if live:
        raise SystemExit("refusing concurrent A-G worker; active worker command(s):\n" + "\n".join(live))
    free = gpu_free_mib()
    if free < args.minimum_free_vram_mib:
        raise SystemExit(
            f"refusing launch: {free} MiB free VRAM is below "
            f"{args.minimum_free_vram_mib} MiB"
        )

    setting = config.get("learning_setting", "task_il")
    variant = "taskil" if setting == "task_il" else "classil"
    python = ROOT / f"runtime_baseline_{variant}" / ".venv" / "bin" / "python"
    if not python.is_file():
        raise SystemExit(f"missing runtime interpreter: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "worker.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / f"runtime_baseline_{variant}" / "src"),
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
        }
    )
    record = {
        "run_id": args.run_id,
        "status": "launching",
        "started_at_epoch": time.time(),
        "prelaunch_gpu_free_mib": free,
        "minimum_free_vram_mib": args.minimum_free_vram_mib,
        "plan_path": str(spec),
        "result_path": str(output_dir),
        "learning_setting": setting,
        "seed": config["seed"],
        "method": config.get("method"),
        "aggregator": config.get("aggregator"),
    }
    # Exclusive create prevents two helper invocations from admitting the
    # same run at once. The worker itself owns result metadata/status files.
    with record_path.open("x", encoding="utf-8") as record_file:
        record_file.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(python), str(RUNNER), "--worker-spec", str(spec)],
            cwd=ROOT / f"runtime_baseline_{variant}",
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    record.update({"status": "launched", "pid": process.pid})
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "launched", **record}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
