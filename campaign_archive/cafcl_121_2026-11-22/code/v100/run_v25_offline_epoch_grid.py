#!/usr/bin/env python3
"""V25 Replay/FedAvg participation x IL x curriculum x local-epoch grid.

48 runs = 4 participation levels x 2 learning settings x 2 curricula x 3
local epoch values. Runs are launched one at a time, with no free-VRAM
admission threshold. A global worker lock also serializes children from the
earlier four-at-a-time controller if it later wakes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO / "results" / "v25_offline_epoch_grid_20261122_v2"
DATA = REPO / "data"
SOURCE_ROOT = REPO / "source_variants"

def source_for_setting(setting):
    return SOURCE_ROOT / ("taskil_src" if setting == "task_il" else "classil_src")

PYTHON = Path(sys.executable)
SEED = 0
OFFLINE_COUNTS = (0, 1, 2, 3)
EPOCH_VALUES = (1, 2, 5)
SETTINGS = ("task_il", "class_il")
CURRICULA = ("disjoint", "joint")
MAX_CONCURRENT_PER_BLOCK = 1
MIN_FREE_VRAM_MIB = 0
GPU_WAIT_INTERVAL_SEC = 300
STATUS_INTERVAL_SEC = 10


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def slug(setting: str, curriculum: str, offline: int, epochs: int) -> str:
    mode = "TASKIL" if setting == "task_il" else "CLASSIL"
    stream = "JOINT_CHAIN20" if curriculum == "joint" else "DISJOINT10"
    off = "ALLSYNC" if offline == 0 else "OFFLINE{}".format(offline)
    return "V25_{}_{}_{}_EPOCHS{}_REPLAY_FEDAVG_SEED{}".format(
        mode, stream, off, epochs, SEED)


def make_blocks():
    blocks = []
    index = 0
    for epochs in EPOCH_VALUES:
        for setting in SETTINGS:
            for curriculum in CURRICULA:
                for offline in OFFLINE_COUNTS:
                    index += 1
                    block_id = "block_{:02d}_{}_{}_offline{}_ep{}".format(
                        index, setting.upper(), curriculum.upper(), offline, epochs)
                    run = {
                        "job_id": slug(setting, curriculum, offline, epochs),
                        "setting": setting,
                        "curriculum": curriculum,
                        "offline_count": offline,
                        "participation": (5 - offline) / 5.0,
                        "epochs_per_task": epochs,
                        "seed": SEED,
                        "block_id": block_id,
                    }
                    blocks.append({"block_id": block_id, "runs": [run]})
    return blocks


def make_config(spec):
    from cafcl.full_matrix import base

    setting = spec["setting"]
    curriculum = spec["curriculum"]
    chain = curriculum == "joint"
    offline = int(spec["offline_count"])
    participation = (5 - offline) / 5.0
    cfg = replace(
        base(seed=SEED, model="cnn"),
        job_id=spec["job_id"],
        seed=SEED,
        dataset="cifar100",
        num_clients=5,
        batch_size=64,
        learning_setting=setting,
        class_pair_protocol="strict",
        classwise_head_merge=False,
        method="replay",
        replay_reservoir_fraction=1.0 / 9.0,
        memory_per_class=50,
        share_continual_state=True,
        aggregator="fedavg",
        scenario="disjoint",
        stream_overlap_kind="none",
        stream_overlap_fraction=0.0,
        sample_overlap_fraction=0.0,
        stream_client_chain=chain,
        classes_per_local_task=2,
        local_tasks_per_round=1,
        stream_recurrence_rounds=0,
        stream_terminal_flush=True,
        participation=participation,
        selector="random",
        stream_offline_policy="random",
        evaluate_final_only=True,
        lr=0.03,
        epochs=int(spec["epochs_per_task"]),
        optimizer_state_policy="persist",
        momentum=0.9,
        weight_decay=0.0005,
        client_training_concurrency=5,
        training_augmentation="cifar_crop_flip",
        replay_ratio=1.0,
        replay_update_policy="sampled_mixed",
        stream_sync_every_batch=False,
    )
    cfg.validate()
    return cfg


def prepare_design(spec, out_dir: Path):
    import numpy as np
    from cafcl.models import resolve_backbone
    from cafcl.selection import participation_budget, select_clients
    from cafcl.streaming import make_task_schedule, select_offline_cohort

    cfg = make_config(spec)
    offline = int(spec["offline_count"])
    online = 5 - offline
    if participation_budget(5, cfg.participation) != online:
        raise RuntimeError("Participation does not select the exact requested cohort size")
    schedule = make_task_schedule(100, cfg)
    learning_syncs = 20 if spec["curriculum"] == "joint" else 10
    if len(schedule) != 5 * learning_syncs:
        raise RuntimeError("Unexpected task-stream size: {}".format(len(schedule)))
    if 1 + max(row[0] for row in schedule) != learning_syncs:
        raise RuntimeError("Unexpected number of learning syncs")
    occurrences = {
        c: sum(c in labels for _, _, _, labels, _ in schedule)
        for c in range(100)
    }
    expected_occurrences = {1, 2, 3} if spec["curriculum"] == "joint" else {1}
    if set(occurrences.values()) != expected_occurrences:
        raise RuntimeError("Unexpected class-overlap profile: {}".format(
            sorted(set(occurrences.values()))))

    rng = np.random.default_rng(SEED + 888777)
    state = {cid: None for cid in range(5)}
    cohorts = []
    for round_id in range(learning_syncs):
        standard = select_clients(state, cfg, round_id, rng)
        selected = select_offline_cohort(
            cfg, round_id, learning_syncs + 1, standard)
        offline_clients = sorted(set(range(5)) - set(selected))
        if len(offline_clients) != offline or len(selected) != online:
            raise RuntimeError("Random cohort has incorrect size at sync {}".format(
                round_id + 1))
        cohorts.append({
            "sync": round_id + 1,
            "selected_clients": selected,
            "offline_clients": offline_clients,
        })

    design = {
        **spec,
        "online_count_per_learning_sync": online,
        "participation": cfg.participation,
        "learning_syncs": learning_syncs,
        "terminal_flush": True,
        "total_server_syncs_including_flush": learning_syncs + 1,
        "task_instances": len(schedule),
        "class_occurrence_values": sorted(set(occurrences.values())),
        "randomized_online_cohorts": cohorts,
        "config": cfg.to_dict(),
        "backbone": resolve_backbone("cnn"),
    }
    atomic_json(out_dir / "design.json", design)
    return cfg, design


def run_worker(args):
    spec = {
        "job_id": args.job_id,
        "setting": args.setting,
        "curriculum": args.curriculum,
        "offline_count": args.offline_count,
        "participation": (5 - args.offline_count) / 5.0,
        "epochs_per_task": args.epochs,
        "seed": SEED,
        "block_id": args.block_id,
        "source_dir": os.environ.get("CAFCL_SOURCE_DIR"),
    }
    out_dir = Path(args.output_dir)
    if args.plan_only:
        cfg, design = prepare_design(spec, out_dir)
        print(json.dumps({
            "job_id": args.job_id,
            "block_id": args.block_id,
            "learning_syncs": design["learning_syncs"],
            "total_syncs_including_flush": design["total_server_syncs_including_flush"],
            "offline_count": args.offline_count,
            "epochs_per_task": args.epochs,
            "cohorts": design["randomized_online_cohorts"],
        }, indent=2), flush=True)
        return 0

    # The original waiting controller has a four-worker block in memory.
    # Lock all workers campaign-wide, so both this sequential launcher and
    # any later child from that controller can train at most one model.
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    lock_path = CAMPAIGN / ".single_training_worker.lock"
    status_path = out_dir / "worker_status.json"
    summary_path = out_dir / "result_summary.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if status_path.exists() and summary_path.exists():
            prior = json.loads(status_path.read_text())
            if prior.get("status") == "completed":
                print(json.dumps({"job_id": args.job_id,
                                  "status": "already_completed"}), flush=True)
                return 0
        if status_path.exists():
            prior = json.loads(status_path.read_text())
            if prior.get("status") == "running":
                raise RuntimeError("Prior worker ended without a terminal status: {}".format(
                    out_dir))
            if prior.get("status") == "failed":
                raise RuntimeError("Refusing to overwrite failed run: {}".format(out_dir))

        atomic_json(status_path, {
            "job_id": args.job_id,
            "status": "running",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "worker_lock": str(lock_path),
        })
        try:
            cfg, design = prepare_design(spec, out_dir)
            from cafcl.simulator import run_experiment
            result = run_experiment(cfg, str(out_dir / "runs"), str(DATA), download=False)
            atomic_json(summary_path, result)
            atomic_json(status_path, {
                "job_id": args.job_id,
                "status": "completed",
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "learning_syncs": design["learning_syncs"],
                "result_summary": str(summary_path),
            })
            print(json.dumps({"job_id": args.job_id, "result": result},
                             default=str), flush=True)
            return 0
        except BaseException as exc:
            atomic_json(status_path, {
                "job_id": args.job_id,
                "status": "failed",
                "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "pid": os.getpid(),
                "error": repr(exc),
            })
            raise


def gpu_free_mib():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True)
    values = [int(line.strip()) for line in result.stdout.splitlines()
              if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi returned no GPU memory readings")
    return max(values)


def persist_manifest(path, manifest):
    manifest["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(path, manifest)


def run_campaign():
    if (CAMPAIGN / "manifest.json").exists():
        raise SystemExit("Campaign manifest already exists; refusing to overwrite it")
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    (CAMPAIGN / "logs").mkdir()
    blocks = make_blocks()
    runs = [run for block in blocks for run in block["runs"]]
    manifest = {
        "campaign": "V25 CIFAR-100 Replay/FedAvg offline and local-epoch grid",
        "status": "planning",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(REPO),
        "data_root": str(DATA),
        "seed": SEED,
        "participation_levels": [
            {"offline_count": n, "online_count": 5 - n,
             "participation": (5 - n) / 5.0}
            for n in OFFLINE_COUNTS
        ],
        "learning_settings": list(SETTINGS),
        "curricula": list(CURRICULA),
        "epochs_per_task": list(EPOCH_VALUES),
        "run_count": len(runs),
        "block_count": len(blocks),
        "max_concurrent_per_block": MAX_CONCURRENT_PER_BLOCK,
        "min_free_vram_mib_before_block": MIN_FREE_VRAM_MIB,
        "gpu_wait_interval_sec": GPU_WAIT_INTERVAL_SEC,
        "blocks": [],
        "runs": [],
    }
    run_rows = {}
    for block in blocks:
        run_ids = []
        for spec in block["runs"]:
            out_dir = CAMPAIGN / "runs" / spec["job_id"]
            log = CAMPAIGN / "logs" / (spec["job_id"] + ".log")
            row = {
                **spec,
                "output_dir": str(out_dir),
                "source_dir": str(source_for_setting(spec["setting"])),
                "log": str(log),
                "status": "planned",
                "pid": None,
                "return_code": None,
            }
            run_rows[spec["job_id"]] = row
            run_ids.append(spec["job_id"])
        manifest["blocks"].append({
            "block_id": block["block_id"],
            "job_ids": run_ids,
            "status": "planned",
        })
    manifest["runs"] = list(run_rows.values())
    mp = CAMPAIGN / "manifest.json"
    persist_manifest(mp, manifest)

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(REPO / "src"),
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })

    # Preflight all 48 exact designs without allocating GPU models.
    for row in manifest["runs"]:
        worker_env = env.copy()
        worker_env["PYTHONPATH"] = str(Path(row["source_dir"]) / "src")
        worker_env["CAFCL_SOURCE_DIR"] = row["source_dir"]
        cmd = [
            str(PYTHON), str(Path(__file__).resolve()), "--worker", "--plan-only",
            "--setting", row["setting"],
            "--curriculum", row["curriculum"],
            "--offline-count", str(row["offline_count"]),
            "--epochs", str(row["epochs_per_task"]),
            "--job-id", row["job_id"],
            "--block-id", row["block_id"],
            "--output-dir", row["output_dir"],
        ]
        with open(row["log"], "ab") as log:
            result = subprocess.run(
                cmd, cwd=str(REPO), env=worker_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            row["status"] = "plan_failed"
            row["return_code"] = result.returncode
            manifest["status"] = "plan_failed"
            persist_manifest(mp, manifest)
            raise SystemExit("Plan validation failed for " + row["job_id"])
        row["status"] = "queued"
        persist_manifest(mp, manifest)
    manifest["status"] = "waiting_for_gpu"
    persist_manifest(mp, manifest)

    for block_index, block in enumerate(blocks):
        block_row = manifest["blocks"][block_index]
        while True:
            free = gpu_free_mib()
            manifest["last_gpu_free_mib"] = free
            if free >= MIN_FREE_VRAM_MIB:
                break
            manifest["status"] = "waiting_for_gpu"
            manifest["waiting_for_block"] = block["block_id"]
            persist_manifest(mp, manifest)
            print("[{}] waiting: free GPU VRAM {} MiB; need >= {} MiB".format(
                time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                free, MIN_FREE_VRAM_MIB), flush=True)
            time.sleep(GPU_WAIT_INTERVAL_SEC)

        block_row["status"] = "running"
        block_row["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        block_row["gpu_free_mib_at_start"] = free
        manifest["status"] = "running"
        manifest["active_block"] = block["block_id"]
        procs = []
        for spec in block["runs"]:
            row = run_rows[spec["job_id"]]
            out_dir = Path(row["output_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            row["status"] = "running"
            row["started_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            worker_env = env.copy()
            worker_env["PYTHONPATH"] = str(Path(row["source_dir"]) / "src")
            worker_env["CAFCL_SOURCE_DIR"] = row["source_dir"]
            cmd = [
                str(PYTHON), str(Path(__file__).resolve()), "--worker",
                "--setting", row["setting"],
                "--curriculum", row["curriculum"],
                "--offline-count", str(row["offline_count"]),
                "--epochs", str(row["epochs_per_task"]),
                "--job-id", row["job_id"],
                "--block-id", row["block_id"],
                "--output-dir", row["output_dir"],
            ]
            with open(row["log"], "ab") as log:
                proc = subprocess.Popen(
                    cmd, cwd=str(REPO), env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            row["pid"] = proc.pid
            procs.append((proc, row))
            persist_manifest(mp, manifest)
            time.sleep(0.5)

        remaining = list(procs)
        while remaining:
            for proc, row in list(remaining):
                code = proc.poll()
                if code is None:
                    continue
                row["return_code"] = code
                row["status"] = "completed" if code == 0 else "failed"
                row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                remaining.remove((proc, row))
                persist_manifest(mp, manifest)
            if remaining:
                time.sleep(STATUS_INTERVAL_SEC)

        block_row["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        block_row["status"] = (
            "completed" if all(run_rows[j]["status"] == "completed"
                               for j in block_row["job_ids"])
            else "completed_with_failures")
        manifest.pop("active_block", None)
        persist_manifest(mp, manifest)

    failed = [row["job_id"] for row in manifest["runs"]
              if row["status"] != "completed"]
    manifest["status"] = "completed_with_failures" if failed else "completed"
    manifest["failed_runs"] = failed
    manifest["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    persist_manifest(mp, manifest)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--setting", choices=SETTINGS)
    parser.add_argument("--curriculum", choices=CURRICULA)
    parser.add_argument("--offline-count", type=int, choices=OFFLINE_COUNTS)
    parser.add_argument("--epochs", type=int, choices=EPOCH_VALUES)
    parser.add_argument("--job-id")
    parser.add_argument("--block-id")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.worker:
        run_campaign()
        return
    required = ("setting", "curriculum", "offline_count", "epochs",
                "job_id", "block_id", "output_dir")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit("Missing worker arguments: " + ", ".join(missing))
    raise SystemExit(run_worker(args))


if __name__ == "__main__":
    main()

