#!/usr/bin/env python3
"""Stage live A-G run evidence from Mtech without deleting destination files.

Model checkpoints stay on Mtech. This helper is safe to rerun: it refreshes
the same provisional run directory with rsync checksum comparison and records
local hashes plus a remote post-transfer inventory in transfer_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path


HOST = "anurag24092@192.168.3.30"
REMOTE_ROOT = "/home/anurag24092/Projects/Correlation-Aware-FCL-independent-ablations-20261122"
ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ARCHIVE_ROOT / "results"
CHECKPOINT_NAMES = {"model_final.pt", "checkpoint_latest.pt"}


def ssh_json(code: str, *args: str):
    remote = "python3 -c " + shlex.quote(code) + " " + " ".join(map(shlex.quote, args))
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, remote],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_runs():
    code = r'''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); out=[]
for p in sorted(root.glob("results/*/*/worker_status.json")):
 try: status=json.loads(p.read_text())
 except Exception: continue
 if status.get("status") != "running": continue
 run=p.parent; log=run/"worker.log"
 checkpoints=[]
 for f in run.rglob("*.pt"):
  if f.name.startswith("model_stage_") or f.name in {"model_final.pt","checkpoint_latest.pt"}:
   checkpoints.append({"path":str(f.relative_to(root)),"bytes":f.stat().st_size})
 out.append({"run_id":status.get("run_id",run.name),"relative_path":str(run.relative_to(root)),
  "pid":status.get("pid"),"started_at":status.get("started_at"),
  "log_tail":log.read_text(errors="replace").splitlines()[-12:] if log.exists() else [],
  "checkpoints":checkpoints})
print(json.dumps(out,separators=(",",":")))'''
    return ssh_json(code, REMOTE_ROOT)


def remote_file_manifest(relative_path: str):
    code = r'''import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); run=root/sys.argv[2]; rows=[]
for f in sorted(run.rglob("*")):
 if not f.is_file() or f.name.startswith("model_stage_") or f.name in {"model_final.pt","checkpoint_latest.pt"}: continue
 h=hashlib.sha256()
 with f.open("rb") as s:
  for b in iter(lambda:s.read(1024*1024),b""): h.update(b)
 rows.append({"path":str(f.relative_to(root)),"bytes":f.stat().st_size,"sha256":h.hexdigest()})
print(json.dumps(rows,separators=(",",":")))'''
    return ssh_json(code, REMOTE_ROOT, relative_path)


def sync_run(run: dict) -> dict:
    relative = run["relative_path"]
    destination = LOCAL_RESULTS / relative.removeprefix("results/")
    destination.mkdir(parents=True, exist_ok=True)
    source = f"{HOST}:{REMOTE_ROOT}/{relative}/"
    command = ["rsync", "-a", "--checksum", "--protect-args"]
    for pattern in ("model_stage_*.pt", "model_final.pt", "checkpoint_latest.pt"):
        command.extend(["--exclude", pattern])
    command.extend([source, str(destination) + "/"])
    subprocess.run(command, check=True)

    relative_files = remote_file_manifest(relative)
    local_files = []
    for row in relative_files:
        path = ARCHIVE_ROOT / row["path"]
        if not path.is_file():
            raise RuntimeError(f"remote file missing after rsync: {row['path']}")
        local = {"path": row["path"], "bytes": path.stat().st_size,
                 "sha256": sha256_file(path)}
        if local != row:
            raise RuntimeError(f"post-transfer size/hash mismatch: {row['path']}")
        local_files.append(local)
    return {"run_id": run["run_id"], "status": "running",
            "source": relative, "local_path": str(destination.relative_to(ARCHIVE_ROOT)),
            "observed_at_epoch": time.time(), "worker_pid": run["pid"],
            "worker_started_at_epoch": run["started_at"], "worker_log_tail": run["log_tail"],
            "copied_files": local_files, "server_only_checkpoints": run["checkpoints"],
            "full_file_manifest_verified": True}


def main():
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_RESULTS / "transfer_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "host": HOST, "remote_root": REMOTE_ROOT, "runs": {}, "provisional_runs": {}}
    active = remote_runs()
    for run in active:
        manifest.setdefault("provisional_runs", {})[run["run_id"]] = sync_run(run)
    manifest["last_active_sync_epoch"] = time.time()
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)
    print(json.dumps({"active_runs_found": len(active),
                      "runs_staged": [row["run_id"] for row in active],
                      "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
