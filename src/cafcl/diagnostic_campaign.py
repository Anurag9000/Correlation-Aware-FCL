"""Single ledger-driven suite with explicit historical-fidelity boundary."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
LEDGER=ROOT/"configs"/"diagnostic_campaign"/"job_ledger.json"
MODULES={"stationary_reference":"cafcl.stationary_diagnostic",
         "matched_reference_v1":"cafcl.matched_diagnostic",
         "historical_reconstruction_v1":"cafcl.matched_diagnostic"}


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)


def load_jobs(path=LEDGER):
    jobs=json.loads(Path(path).read_text())["jobs"]
    assert len(jobs)==152 and len({j["job_id"] for j in jobs})==152
    assert Counter(j["family"] for j in jobs)=={
        "FL":11,"CC":27,"MC":36,"CM":12,"JR":8,"OF":8,"ST":12,"MO":12,"JC":16,"CF":10}
    assert sum(bool(j.get("reproduction")) for j in jobs)==14
    return jobs


def effective_jobs(allow_reconstruction=False,subset_probes=False):
    result=[]
    for raw in load_jobs():
        j=dict(raw)
        if allow_reconstruction and j.get("reconstruction_binding"):
            j.update(binding=j["reconstruction_binding"],protocol_revision=j["reconstruction_binding"],
                     execution_fidelity="reconstructed_not_exact")
        if j.get("binding") and j["binding"]!="stationary_reference":
            j["subset_probes"]=bool(subset_probes)
        result.append(j)
    return result


def signature(job):
    return hashlib.sha256(json.dumps(job,sort_keys=True).encode()).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def expected_rounds(job):
    if job["binding"]=="stationary_reference":
        return job["rounds_per_stage"] or 100,0
    if job["binding"]=="matched_reference_v1":
        return 10*job["rounds_per_stage"],int(job["participation"]<1)
    stages=20 if job["curriculum"]=="joint" else math.ceil(50/job["clients"])
    return stages*job["rounds_per_stage"],1


def validate_artifacts(job,path):
    """Validate terminal records before granting completed status."""
    path=Path(path)
    required=["job.json","provenance.json","partition.json","metrics.csv","result_summary.json"]
    for name in required:
        if not (path/name).is_file():raise ValueError(f"Missing {name}")
    if json.loads((path/"job.json").read_text())!=job:raise ValueError("Job config mismatch")
    summary=json.loads((path/"result_summary.json").read_text())
    if summary["job_id"]!=job["job_id"] or summary["protocol"]!=job["binding"]:
        raise ValueError("Summary identity mismatch")
    with (path/"metrics.csv").open(newline="") as f:rows=list(csv.DictReader(f))
    n,flush=expected_rounds(job)
    if len(rows)!=n+flush:raise ValueError(f"Expected {n+flush} rows, found {len(rows)}")
    key="round" if job["binding"]=="stationary_reference" else "sync"
    if [int(x[key]) for x in rows]!=list(range(1,n+flush+1)):
        raise ValueError("Missing/duplicated round numbers")
    if job["binding"]=="stationary_reference":
        value=summary["final"]["test_accuracy"]
        if summary["rounds"]!=n:raise ValueError("Stationary round count mismatch")
        if not (path/"checkpoints"/f"round_{n:03d}.pt").is_file():raise ValueError("Missing final checkpoint")
    else:
        value=summary["final_accuracy"]
        if summary["learning_syncs"]!=n or bool(summary["terminal_flush"])!=bool(flush):
            raise ValueError("FCL summary horizon mismatch")
        if job["binding"]=="historical_reconstruction_v1" and summary.get("historical_reproduction_verified") is not False:
            raise ValueError("Historical reconstruction must not be labelled exact")
        for s in range(1,n+flush+1):
            if not (path/"per_round_metrics"/f"sync_{s:04d}.json").is_file():
                raise ValueError(f"Missing per-round diagnostic {s}")
        if not (path/"checkpoints"/"latest.pt").is_file():raise ValueError("Missing final checkpoint")
        required.append("schedule.json")
    if not isinstance(value,(float,int)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError("Invalid final accuracy")
    return {"job_id":job["job_id"],"rows":len(rows),"final_accuracy":value,
            "sha256":{name:sha_file(path/name) for name in required}}


def inspect_job(job,root):
    if job["status"]=="reserved":return "reserved"
    if job.get("binding") not in MODULES:return "blocked"
    base=Path(root)/job["family"]/job["job_id"]
    for p in sorted(base.glob("attempt_*/status.json"),reverse=True):
        try:
            status=json.loads(p.read_text())
            if status["signature"]!=signature(job):continue
            if status["state"]=="completed":
                validate_artifacts(job,p.parent)
                return "completed"
            return status["state"]
        except (ValueError,KeyError,OSError,TypeError):
            return "invalid"
    return "ready"


def source_identity():
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    dirty=bool(subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=ROOT,text=True).strip())
    return head,dirty


def run_one(job,args,root,device,expected_head):
    base=root/job["family"]/job["job_id"];base.mkdir(parents=True,exist_ok=True)
    lock=base/".active"
    try:lock.mkdir()
    except FileExistsError:raise RuntimeError(f"Active job lock: {lock}; verify worker termination before removal")
    try:
        if inspect_job(job,root)=="completed":return
        commit,dirty=source_identity()
        if commit!=expected_head or dirty:raise RuntimeError("Tracked source changed or is dirty")
        attempts=[int(p.name.split("_")[1]) for p in base.glob("attempt_*") if p.is_dir()]
        path=base/f"attempt_{max(attempts,default=0)+1:04d}";path.mkdir()
        atomic_json(path/"job.json",job)
        atomic_json(path/"provenance.json",{"git_commit":commit,"tracked_dirty":dirty,"python":sys.version,"device":device})
        status={"state":"running","signature":signature(job),"start":time.time()}
        atomic_json(path/"status.json",status)
        cmd=[sys.executable,"-m",MODULES[job["binding"]],"--job",str(path/"job.json"),
             "--output",str(path),"--data-root",str(Path(args.data_root).resolve()),"--device",device]
        if job.get("subset_probes"):cmd.append("--subset-probes")
        try:
            with (path/"stdout.log").open("w") as out,(path/"stderr.log").open("w") as err:
                rc=subprocess.run(cmd,cwd=ROOT,stdout=out,stderr=err).returncode
            if rc:raise RuntimeError(f"Worker exit code {rc}")
            receipt=validate_artifacts(job,path)
            atomic_json(path/"artifact_validation.json",receipt)
            status.update(state="completed",returncode=rc,end=time.time())
        except BaseException as exc:
            status.update(state="failed",error=repr(exc),end=time.time())
            raise
        finally:atomic_json(path/"status.json",status)
    finally:lock.rmdir()


def preflight(jobs,args):
    # One dataset initialization before workers start; prevents download races.
    from .data import get_data_bank
    from .diagnostic_designs import compile_matched,compile_reconstructed
    if any(j.get("binding")=="stationary_reference" for j in jobs):
        from .stationary_diagnostic import check_data
        check_data(args.data_root,args.download)
    if any(j.get("binding") in ("matched_reference_v1","historical_reconstruction_v1") for j in jobs):
        bank=get_data_bank("cifar100",args.data_root,download=args.download,seed=0)
        for j in jobs:
            if j.get("binding")=="matched_reference_v1":
                compile_matched(j,bank.train.targets,bank.num_classes)
            elif j.get("binding")=="historical_reconstruction_v1":
                compile_reconstructed(j,bank.train.targets,bank.num_classes)


def states(jobs,root):
    return [{"job_id":j["job_id"],"family":j["family"],"state":inspect_job(j,root),
             "binding":j.get("binding"),"blocker":j.get("blocker") if j.get("binding") not in MODULES else None}
            for j in jobs]


def write_status(root,jobs,failures):
    rows=states(jobs,root)
    atomic_json(root/"suite_status.json",{"counts":dict(Counter(r["state"] for r in rows)),
                                         "failures":failures,"jobs":rows})
    tmp=root/"suite_status.csv.tmp"
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["job_id","family","state","binding","blocker"])
        writer.writeheader();writer.writerows(rows)
    tmp.replace(root/"suite_status.csv")


def device_list(args):
    import torch
    devices=[f"cuda:{x}" for x in args.gpus] if args.gpus else [args.device]
    if len(set(devices))!=len(devices):raise ValueError("Duplicate devices")
    for device in devices:
        d=torch.device(device)
        if d.type=="cuda":
            if not torch.cuda.is_available() or (d.index or 0)>=torch.cuda.device_count():
                raise RuntimeError(f"Unavailable GPU {device}; refusing CPU fallback")
        elif d.type!="cpu":raise ValueError(f"Unsupported device {device}")
    return devices


def run_suite(jobs,all_jobs,args,root,devices):
    root.mkdir(parents=True,exist_ok=True)
    lock=root/".suite_active"
    try:lock.mkdir()
    except FileExistsError:raise RuntimeError(f"Active suite lock {lock}; verify workers stopped before removal")
    failures=[]
    try:
        head,dirty=source_identity()
        if dirty:raise RuntimeError("Use a clean tracked checkout")
        if shutil.disk_usage(root).free<args.min_disk_gib*1024**3:
            raise RuntimeError("Insufficient free result disk space")
        preflight(jobs,args)
        atomic_json(root/"last_launch.json",{"git_commit":head,"devices":devices,"selected":[j["job_id"] for j in jobs],
            "allow_reconstruction":args.allow_reconstruction,"subset_probes":args.subset_probes})
        # One worker process per GPU; the executor allocates the next job to
        # whichever device becomes free. All worker writes are job-local.
        pending=iter(j for j in jobs if inspect_job(j,root) not in ("blocked","reserved","completed"))
        def execute(job,device):
            for attempt in range(args.retries+1):
                try:
                    if shutil.disk_usage(root).free<args.min_disk_gib*1024**3:
                        raise RuntimeError("Result disk below minimum")
                    run_one(job,args,root,device,head);return None
                except Exception as exc:
                    error=repr(exc)
            return {"job_id":job["job_id"],"device":device,"error":error}
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            active={}
            def schedule(device):
                j=next(pending,None)
                if j is not None:active[pool.submit(execute,j,device)]=device
            for device in devices:schedule(device)
            write_status(root,all_jobs,failures)
            while active:
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    device=active.pop(future)
                    result=future.result()
                    if result:failures.append(result)
                    write_status(root,all_jobs,failures)
                    schedule(device)
        write_status(root,all_jobs,failures)
        from .diagnostic_analysis import analyze
        analyze(root,all_jobs)
        return int(bool(failures))
    finally:lock.rmdir()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=["plan","status","preflight","run","resume"])
    p.add_argument("--family");p.add_argument("--wave",type=int);p.add_argument("--job-id")
    p.add_argument("--all",action="store_true")
    p.add_argument("--ready-only",action="store_true")
    p.add_argument("--allow-reconstruction",action="store_true",help="Explicitly allow non-exact historical reference jobs")
    p.add_argument("--subset-probes",action="store_true")
    p.add_argument("--output",default="runs/diagnostic_campaign")
    p.add_argument("--data-root",default="data")
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--gpus",nargs="+",type=int)
    p.add_argument("--download",action="store_true")
    p.add_argument("--retries",type=int,default=0)
    p.add_argument("--min-disk-gib",type=float,default=10.)
    args=p.parse_args(argv)
    if args.retries<0 or args.min_disk_gib<0:p.error("Invalid resource limits")
    if args.all and (args.family or args.wave is not None or args.job_id):p.error("--all cannot be combined with filters")
    root=Path(args.output).resolve()
    all_jobs=effective_jobs(args.allow_reconstruction,args.subset_probes)
    jobs=[j for j in all_jobs if (not args.family or j["family"]==args.family) and
          (args.wave is None or j["wave"]==args.wave) and (not args.job_id or j["job_id"]==args.job_id)]
    if not jobs:p.error("No matching jobs")
    rows=states(jobs,root)
    print(json.dumps({"counts":dict(Counter(s["state"] for s in rows)),"jobs":rows},indent=2),flush=True)
    if args.command in ("plan","status"):return 0
    if any(s["state"]=="blocked" for s in rows) and not args.ready_only:
        print("Unbound historical-fidelity jobs; no training launched. Use --allow-reconstruction or --ready-only.",file=sys.stderr)
        return 2
    jobs=[j for j in jobs if j.get("binding") in MODULES]
    if not jobs:return 0
    if args.command=="preflight":
        preflight(jobs,args);return 0
    devices=device_list(args)
    return run_suite(jobs,all_jobs,args,root,devices)


if __name__=="__main__":sys.exit(main())
