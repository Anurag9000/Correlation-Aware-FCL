#!/usr/bin/env python3
"""Pull stable A-G result evidence from Mtech to the Kakarot archive.

Copies only completed/failed run directories and never deletes destination
files. Large model checkpoints stay on Mtech; their paths and byte sizes are
recorded in the local transfer manifest.
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
WEIGHT_NAMES = ("model_stage_*.pt", "model_final.pt", "checkpoint_latest.pt")


def remote_finished_runs():
    code = r'''import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);out=[]
for status_path in sorted(root.glob("results/*/*/worker_status.json")):
 try: status=json.loads(status_path.read_text())
 except Exception: continue
 state=status.get("status")
 if state not in {"completed","finished","failed"}: continue
 run=status_path.parent
 checkpoints=[]
 for p in run.rglob("*"):
  if p.is_file() and (p.name.startswith("model_stage_") and p.suffix==".pt" or p.name in {"model_final.pt","checkpoint_latest.pt"}):
   checkpoints.append({"path":str(p.relative_to(root)),"bytes":p.stat().st_size})
 out.append({"run_id":status.get("run_id",run.name),"status":state,
  "relative_path":str(run.relative_to(root)),"checkpoints":checkpoints})
print(json.dumps(out,separators=(",",":")))'''
    remote_cmd = "python3 -c " + shlex.quote(code) + " " + shlex.quote(REMOTE_ROOT)
    completed = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", HOST, remote_cmd],
        check=True, text=True, capture_output=True,
    )
    return json.loads(completed.stdout)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_run(run: dict) -> dict:
    remote_rel = run["relative_path"]
    remote_path = f"{HOST}:{REMOTE_ROOT}/{remote_rel}/"
    local_path = LOCAL_RESULTS / remote_rel.removeprefix("results/")
    local_path.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--checksum", "--protect-args"]
    for pattern in WEIGHT_NAMES:
        command.extend(["--exclude", pattern])
    command.extend([remote_path, str(local_path) + "/"])
    subprocess.run(command, check=True)
    files = []
    for path in sorted(local_path.rglob("*")):
        if path.is_file():
            files.append({"path":str(path.relative_to(ARCHIVE_ROOT)),
                          "bytes":path.stat().st_size,"sha256":sha256_file(path)})
    return {"run_id":run["run_id"],"status":run["status"],
            "source":remote_rel,"local_path":str(local_path.relative_to(ARCHIVE_ROOT)),
            "copied_files":files,"server_only_checkpoints":run["checkpoints"]}


def main():
    LOCAL_RESULTS.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_RESULTS / "transfer_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"host":HOST,"remote_root":REMOTE_ROOT,"runs":{}}
    runs = remote_finished_runs()
    for run in runs:
        manifest["runs"][run["run_id"]] = sync_run(run)
        # A terminal refresh supersedes any earlier active-run snapshot.
        manifest.setdefault("provisional_runs", {}).pop(run["run_id"], None)
    manifest["last_sync_epoch"] = time.time()
    manifest["checkpoint_policy"] = (
        "model_stage_*.pt, model_final.pt, checkpoint_latest.pt remain on Mtech; "
        "their server paths and byte sizes are recorded. All other files under "
        "completed/failed run directories are copied and hashed locally."
    )
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(manifest_path)
    print(json.dumps({"finished_runs_found":len(runs),
                      "runs_synced":len(runs),
                      "manifest":str(manifest_path)},indent=2))


if __name__ == "__main__":
    main()
