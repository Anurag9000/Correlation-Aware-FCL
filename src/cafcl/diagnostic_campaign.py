"""Single ledger-driven entry point for the unified diagnostic campaign."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT / "configs" / "diagnostic_campaign" / "job_ledger.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def load_jobs(path=LEDGER):
    jobs = json.loads(Path(path).read_text())["jobs"]
    assert len(jobs) == 152 and len({j["job_id"] for j in jobs}) == 152
    assert Counter(j["family"] for j in jobs) == {
        "FL":11,"CC":27,"MC":36,"CM":12,"JR":8,"OF":8,"ST":12,
        "MO":12,"JC":16,"CF":10}
    assert sum(bool(j.get("reproduction")) for j in jobs) == 14
    return jobs


def signature(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def inspect_job(job, root):
    if job["status"] == "reserved":
        return "reserved"
    if job.get("binding") not in ("stationary_reference","matched_reference_v1","historical_reconstruction_v1"):
        return "blocked"
    base = root / job["family"] / job["job_id"]
    for p in sorted(base.glob("attempt_*/status.json"), reverse=True):
        try:
            status = json.loads(p.read_text())
            if status["signature"] != signature(job):
                continue
            if status["state"] == "completed" and (p.parent / "result_summary.json").is_file() and (p.parent / "metrics.csv").is_file() and (p.parent / "partition.json").is_file():
                return "completed"
            return status["state"]
        except (ValueError, KeyError, OSError):
            continue
    return "ready"


def run_one(job, args, root):
    base = root / job["family"] / job["job_id"]
    base.mkdir(parents=True, exist_ok=True)
    lock = base / ".active"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(f"Active lock exists: {lock}; verify no worker is running before clearing")
    try:
        if inspect_job(job, root) == "completed":
            return
        attempts = [int(p.name.split("_")[1]) for p in base.glob("attempt_*") if p.is_dir()]
        path = base / f"attempt_{max(attempts, default=0)+1:04d}"
        path.mkdir()
        atomic_json(path / "job.json", job)
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"], cwd=ROOT, text=True).strip())
        atomic_json(path / "provenance.json", {"git_commit":commit,"tracked_dirty":dirty,"python":sys.version,"device":args.device})
        status = {"state":"running","signature":signature(job),"start":time.time()}
        atomic_json(path / "status.json", status)
        module = {"stationary_reference":"cafcl.stationary_diagnostic","matched_reference_v1":"cafcl.matched_diagnostic","historical_reconstruction_v1":"cafcl.matched_diagnostic"}[job["binding"]]
        cmd = [sys.executable,"-m",module,"--job",str(path/"job.json"),
               "--output",str(path),"--data-root",str(Path(args.data_root).resolve()),"--device",args.device]
        if args.download:
            cmd.append("--download")
        if job["binding"]!="stationary_reference" and args.subset_probes:
            cmd.append("--subset-probes")
        with (path/"stdout.log").open("w") as out, (path/"stderr.log").open("w") as err:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=out, stderr=err).returncode
        complete = rc == 0 and all((path/name).is_file() for name in ("result_summary.json","metrics.csv","partition.json"))
        status.update(state="completed" if complete else "failed",returncode=rc,end=time.time())
        atomic_json(path/"status.json",status)
        if not complete:
            raise RuntimeError(f"{job['job_id']} failed; see {path}")
    finally:
        lock.rmdir()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","status","preflight","run","resume"])
    p.add_argument("--family")
    p.add_argument("--wave",type=int)
    p.add_argument("--job-id")
    p.add_argument("--all",action="store_true")
    p.add_argument("--ready-only",action="store_true",help="Explicitly allow skipping blocked/reserved jobs")
    p.add_argument("--output",default="runs/diagnostic_campaign")
    p.add_argument("--data-root",default="data")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--download",action="store_true")
    p.add_argument("--subset-probes",action="store_true")
    p.add_argument("--allow-reconstruction",action="store_true",help="Explicitly opt into non-exact historical reference jobs")
    args=p.parse_args(argv)
    root=Path(args.output).resolve()
    all_jobs=load_jobs()
    if args.allow_reconstruction:
        all_jobs=[{**j,"binding":j["reconstruction_binding"],"protocol_revision":j["reconstruction_binding"],"execution_fidelity":"reconstructed_not_exact"} if j.get("reconstruction_binding") else j for j in all_jobs]
    jobs=[j for j in all_jobs if (not args.family or j["family"]==args.family) and
          (args.wave is None or j["wave"]==args.wave) and
          (not args.job_id or j["job_id"]==args.job_id)]
    if not jobs:
        p.error("No matching jobs")
    states=[{"job_id":j["job_id"],"state":inspect_job(j,root),"blocker":j.get("blocker")} for j in jobs]
    print(json.dumps({"counts":dict(Counter(s["state"] for s in states)),"jobs":states},indent=2))
    if args.command in ("plan","status"):
        return 0
    blocked=[s for s in states if s["state"]=="blocked"]
    if blocked and not args.ready_only:
        print("Selected suite contains unresolved jobs; no training launched. Use --ready-only to run implemented subset. Reserved slots are always skipped.",file=sys.stderr)
        return 2
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing CPU fallback")
    if args.command=="preflight":
        from .data import get_data_bank
        from .diagnostic_designs import compile_matched,compile_reconstructed
        if any(j.get("binding")=="stationary_reference" for j in jobs):
            from .stationary_diagnostic import check_data
            check_data(args.data_root,args.download)
        if any(j.get("binding") in ("matched_reference_v1","historical_reconstruction_v1") for j in jobs):
            bank=get_data_bank("cifar100",args.data_root,download=args.download,seed=0)
            for j in jobs:
                if j.get("binding") in ("matched_reference_v1","historical_reconstruction_v1"):
                    compiler=compile_matched if j["binding"]=="matched_reference_v1" else compile_reconstructed
                    compiler(j,bank.train.targets,bank.num_classes)
        return 0
    failures=[]
    for job in jobs:
        if inspect_job(job,root) in ("blocked","reserved","completed"):
            continue
        try:
            run_one(job,args,root)
        except Exception as exc:
            failures.append({"job_id":job["job_id"],"error":str(exc)})
    atomic_json(root/"suite_status.json",{"failures":failures,"jobs":[{"job_id":j["job_id"],"state":inspect_job(j,root)} for j in all_jobs]})
    return int(bool(failures))


if __name__=="__main__":
    sys.exit(main())
