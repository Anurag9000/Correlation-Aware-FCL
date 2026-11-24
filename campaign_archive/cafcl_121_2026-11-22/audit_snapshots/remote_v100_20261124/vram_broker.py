#!/usr/bin/env python3
"""VRAM-only admission wrapper for concurrent CIFAR-100 experiment workers."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STATE = Path.home() / ".cache" / "cafcl-vram-broker"
RESERVATIONS = STATE / "reservations"
LOCK = STATE / "admission.lock"
TERMINAL = STATE / "terminal"
POLL_SECONDS = 30


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def free_vram_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [int(line.strip()) for line in out.splitlines() if line.strip()]
    if len(values) != 1:
        raise RuntimeError(f"Expected one CUDA device, got free-VRAM rows: {values}")
    return values[0]


def process_tree(root_pid: int) -> set[int]:
    try:
        rows = subprocess.check_output(["ps", "-eo", "pid=,ppid="], text=True).splitlines()
    except subprocess.CalledProcessError:
        return {root_pid}
    children: dict[int, list[int]] = {}
    for row in rows:
        parts = row.split()
        if len(parts) == 2:
            pid, ppid = map(int, parts)
            children.setdefault(ppid, []).append(pid)
    result = {root_pid}
    stack = [root_pid]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child not in result:
                result.add(child)
                stack.append(child)
    return result


def gpu_memory_by_pid() -> dict[int, int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return {}
    result: dict[int, int] = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            result[int(parts[0])] = int(parts[1])
        except ValueError:
            continue
    return result


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def load_reservations() -> list[tuple[Path, dict]]:
    RESERVATIONS.mkdir(parents=True, exist_ok=True)
    live = []
    for path in RESERVATIONS.glob("*.reservation.json"):
        try:
            item = json.loads(path.read_text())
            pid = int(item["worker_pid"])
            if not alive(pid):
                path.unlink(missing_ok=True)
                continue
            live.append((path, item))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            # Preserve unreadable reservation evidence; fail closed.
            raise RuntimeError(f"Unreadable VRAM reservation; refusing admission: {path}")
    return live


def outstanding_reserved_mib(live: list[tuple[Path, dict]], memory: dict[int, int]) -> int:
    outstanding = 0
    for _, item in live:
        pids = process_tree(int(item["worker_pid"]))
        observed = sum(memory.get(pid, 0) for pid in pids)
        outstanding += max(0, int(item["estimate_mib"]) - observed)
    return outstanding


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--estimate-mib", required=True, type=int)
    parser.add_argument("--margin-mib", type=int, default=1024)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide worker command after --")
    if args.estimate_mib <= 0 or args.margin_mib < 0:
        parser.error("VRAM estimate must be positive and margin nonnegative")

    STATE.mkdir(parents=True, exist_ok=True)
    RESERVATIONS.mkdir(parents=True, exist_ok=True)
    TERMINAL.mkdir(parents=True, exist_ok=True)
    reservation = RESERVATIONS / f"{args.run_id}.{os.getpid()}.reservation.json"
    receipt = args.receipt
    while True:
        with LOCK.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            live = load_reservations()
            if any(item.get("run_id") == args.run_id for _, item in live):
                raise RuntimeError(f"Run already has a live VRAM reservation: {args.run_id}")
            free = free_vram_mib()
            memory = gpu_memory_by_pid()
            outstanding = outstanding_reserved_mib(live, memory)
            available = free - outstanding
            required = args.estimate_mib + args.margin_mib
            if available >= required:
                before = free
                child_env = os.environ.copy()
                child_env["CAFCL_VRAM_BROKER_ADMITTED"] = "1"
                child = subprocess.Popen(command, env=child_env, start_new_session=True)
                record = {
                    "run_id": args.run_id,
                    "worker_pid": child.pid,
                    "estimate_mib": args.estimate_mib,
                    "margin_mib": args.margin_mib,
                    "prelaunch_free_vram_mib": before,
                    "other_outstanding_reservations_mib": outstanding,
                    "admitted_at_epoch": time.time(),
                    "command": command,
                    "reservation_file": str(reservation),
                    "status": "running",
                }
                atomic_json(reservation, record)
                if receipt:
                    atomic_json(receipt, record)
                print(json.dumps({"event": "admitted", **record}), flush=True)
                break
            print(json.dumps({
                "event": "waiting_for_vram",
                "run_id": args.run_id,
                "free_vram_mib": free,
                "outstanding_reservations_mib": outstanding,
                "available_after_reservations_mib": available,
                "required_mib": required,
            }), flush=True)
        time.sleep(POLL_SECONDS)

    peak = 0
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        while child.poll() is None:
            memory = gpu_memory_by_pid()
            observed = sum(memory.get(pid, 0) for pid in process_tree(child.pid))
            peak = max(peak, observed)
            time.sleep(10)
        code = child.returncode
    except KeyboardInterrupt:
        # Do not kill a worker implicitly; propagate interruption state and leave
        # its own lifecycle to the owning campaign controller.
        raise
    finally:
        with LOCK.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                current = json.loads(reservation.read_text())
                current.update({"status": "terminal", "finished_at_epoch": time.time(), "peak_observed_mib": peak})
                atomic_json(TERMINAL / f"{args.run_id}.{child.pid}.json", current)
                if receipt:
                    atomic_json(receipt, current)
            finally:
                reservation.unlink(missing_ok=True)
    print(json.dumps({"event": "finished", "run_id": args.run_id, "exit_code": code, "peak_observed_mib": peak}), flush=True)
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
