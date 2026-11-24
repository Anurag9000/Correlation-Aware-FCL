#!/usr/bin/env python3
"""Inventory/delete only *.pt files in this campaign's allowlisted output roots."""
import argparse
import hashlib
import json
import os
import socket
import stat
import sys
import time
from pathlib import Path


def roots_for(host):
    roots = []
    if host == "mtech":
        roots.extend([
            Path("/home/anurag24092/Projects/Correlation-Aware-FCL-tasksize45-disjoint-ablation-20261122/results/results"),
            Path("/home/anurag24092/Projects/Correlation-Aware-FCL-intratask-sync-ablation-20261122/results/results"),
            Path("/home/anurag24092/Projects/Correlation-Aware-FCL-independent-ablations-20261122/results/results"),
            Path("/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/results/results"),
        ])
        for parent in (Path("/dev/shm"), Path("/run/user")):
            if parent.exists():
                roots.extend(p for p in parent.glob("**/mtech") if p.is_dir() and p.parent.name.startswith("cafcl-"))
    elif host == "v100":
        roots.append(Path("/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/results/v25_offline_epoch_grid_20261122_v2"))
        for parent, pattern in ((Path("/dev/shm"), "**/v100/v25_offline_epoch_grid_20261122_v2"),
                                (Path("/run/user"), "**/v100/v25_offline_epoch_grid_20261122_v2")):
            if parent.exists():
                roots.extend(p for p in parent.glob(pattern) if p.is_dir())
    else:
        raise ValueError(host)
    return sorted(set(roots))


def iter_pt_files(roots):
    seen = set()
    for root in roots:
        if not root.is_dir():
            continue
        for base, dirs, files in os.walk(str(root), followlinks=False):
            dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(base, d))]
            for name in files:
                if not name.endswith(".pt"):
                    continue
                path = os.path.join(base, name)
                try:
                    st = os.lstat(path)
                    if not stat.S_ISREG(st.st_mode):
                        continue
                except OSError:
                    continue
                if path not in seen:
                    seen.add(path)
                    yield root, path, st.st_size


def active_campaign_processes():
    matches = []
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if any(token in cmd for token in (
            "cafcl_independent_ablation_runner.py",
            "run_v25_offline_epoch_grid.py",
            "vram_scheduler_v6",
            "mtech_ag_vram_scheduler",
        )):
            matches.append({"pid": int(entry.name), "cmd": cmd[:500]})
    return matches


def open_pids(path):
    result = []
    target = os.path.realpath(path)
    for entry in Path("/proc").glob("[0-9]*"):
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                link = os.path.realpath(str(fd))
            except OSError:
                continue
            if link == target:
                result.append(int(entry.name))
                break
    return sorted(set(result))


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, choices=("mtech", "v100"))
    ap.add_argument("--delete", action="store_true")
    args = ap.parse_args()
    roots = roots_for(args.host)
    active = active_campaign_processes()
    files = []
    errors = []
    if args.delete and active:
        raise SystemExit("Refusing deletion while campaign processes are active: " + json.dumps(active))
    for root, path, size in iter_pt_files(roots):
        entry = {"path": path, "allowlisted_root": str(root), "bytes": size, "sha256": None, "open_pids": [], "deleted": False}
        try:
            entry["sha256"] = digest(path)
            entry["open_pids"] = open_pids(path)
            if args.delete and not entry["open_pids"]:
                os.unlink(path)
                entry["deleted"] = True
        except OSError as exc:
            entry["error"] = repr(exc)
            errors.append({"path": path, "error": repr(exc)})
        files.append(entry)
    report = {
        "created_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "scope": args.host,
        "operation": "delete_pt_files" if args.delete else "inventory_pt_files",
        "allowlisted_roots": [str(p) for p in roots],
        "active_campaign_processes": active,
        "file_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "deleted_count": sum(x["deleted"] for x in files),
        "skipped_open_count": sum(bool(x["open_pids"]) for x in files),
        "error_count": len(errors),
        "files": files,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
