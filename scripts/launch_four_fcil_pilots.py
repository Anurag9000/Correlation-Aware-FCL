#!/usr/bin/env python3
"""Fail-fast supervisor for the four paired native/FedCLASS CIFAR-100 pilots.

Run this process inside a persistent tmux session. It starts all four paired
suite controllers together; each controller starts its FULL and OFFLINE workers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("glfc", "lga", "feat", "fedclass")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_json(command: list[str], *, cwd: Path, env: dict[str, str]) -> dict:
    result = subprocess.run(command, cwd=cwd, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(f"Preflight failed ({result.returncode}): {' '.join(command)}\n"
                           f"{result.stderr[-4000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Expected JSON from {' '.join(command)}: {result.stdout[-1000:]}") from exc


def gpu_snapshot(gpu: int) -> dict:
    query = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=name,memory.total,memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if query.returncode:
        raise RuntimeError(f"nvidia-smi query failed: {query.stderr.strip()}")
    fields = [value.strip() for value in query.stdout.splitlines()[0].split(",")]
    apps = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if apps.returncode:
        raise RuntimeError(f"nvidia-smi process query failed: {apps.stderr.strip()}")
    processes = []
    for line in apps.stdout.splitlines():
        cols = [v.strip() for v in line.split(",")]
        if len(cols) >= 2:
            processes.append({"pid": int(cols[0]), "used_mib": int(cols[1])})
    return {"name": fields[0], "total_mib": int(fields[1]),
            "free_mib": int(fields[2]), "utilization_pct": int(fields[3]),
            "compute_processes": processes, "sampled_at_utc": utc_now()}


def hardware_preflight(args, env: dict[str, str]) -> dict:
    if args.gpu != 0:
        raise RuntimeError("This campaign's four launchers are pinned to physical GPU 0")
    cuda = run_json([sys.executable, "-c",
        "import json,torch; print(json.dumps({'available':torch.cuda.is_available(),"
        "'count':torch.cuda.device_count(), 'name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"],
        cwd=ROOT, env=env)
    if not cuda["available"] or cuda["count"] < 1:
        raise RuntimeError(f"CUDA device 0 unavailable in {sys.executable}: {cuda}")
    gpu = gpu_snapshot(args.gpu)
    if gpu["compute_processes"]:
        raise RuntimeError("GPU 0 has unrelated compute processes; preserving them and refusing launch: "
                           + json.dumps(gpu["compute_processes"]))
    if gpu["free_mib"] < args.min_free_mib:
        raise RuntimeError(f"GPU 0 has {gpu['free_mib']} MiB free; campaign gate requires "
                           f"{args.min_free_mib} MiB free")
    return {"cuda": cuda, "gpu": gpu}


def preflight(args, env: dict[str, str], *, plans_only: bool = False) -> tuple[list[dict], dict]:
    if plans_only:
        hardware = {"cuda": {"skipped": "--plans-only"}, "gpu": {"skipped": "--plans-only"}}
    else:
        hardware = hardware_preflight(args, env)
    plans = []
    with tempfile.TemporaryDirectory(prefix="cafcl_four_pilot_plan_") as plan_tmp:
        for suite in SUITES:
            module = "cafcl.fedclass_primary" if suite == "fedclass" else "cafcl.native_primary"
            command = [sys.executable, "-m", module, "--seed", str(args.seed),
                       "--lr", str(args.lr), "--epochs", str(args.epochs),
                       "--gpu", "0", "--data-root", str(args.data_root),
                       "--output-dir", str(Path(plan_tmp) / suite), "--plan-only"]
            if suite != "fedclass":
                command += ["--suite", suite]
            else:
                command += ["--memory-mode", "reservoir", "--replay-ratio", "1.0"]
            plan = run_json(command, cwd=ROOT, env=env)
            if plan.get("total_jobs") != 2 or plan.get("clients") != 5:
                raise RuntimeError(f"{suite} plan did not produce the required paired five-client jobs")
            if plan.get("private_reservoir_per_class") != 50:
                raise RuntimeError(f"{suite} plan does not retain exactly 50 examples/class")
            if plan.get("batch_size") != 64 or plan.get("sgd_momentum") != .9 or plan.get("sgd_weight_decay") != .0005:
                raise RuntimeError(f"{suite} plan has an unexpected physical batch or SGD setting")
            plans.append({"suite": suite, "module": module, "plan": plan})
    return plans, hardware


def atomic_manifest(path: Path, data: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def metrics_progress(output_dir: Path) -> dict:
    progress = {}
    for path in sorted(output_dir.rglob("metrics.json")):
        try:
            rows = json.loads(path.read_text())
            last = rows[-1] if rows else {}
            progress[str(path.relative_to(output_dir))] = {
                "rows": len(rows), "last_round": last.get("round"),
                "test_accuracy": last.get("accuracy"),
                "terminal_flush": last.get("terminal_communication_only_flush"),
                "bytes": path.stat().st_size, "mtime": path.stat().st_mtime}
        except (OSError, ValueError, TypeError) as exc:
            progress[str(path.relative_to(output_dir))] = {"read_error": str(exc)}
    return progress


def verify_results(output_dir: Path) -> dict:
    progress = metrics_progress(output_dir)
    if len(progress) != 8:
        raise RuntimeError(f"Expected eight independent metrics.json files, found {len(progress)}")
    incomplete = {name: row for name, row in progress.items()
                  if row.get("rows") != 10 or not row.get("terminal_flush")}
    if incomplete:
        raise RuntimeError(f"Runs missing nine training syncs plus terminal U10 flush: {incomplete}")
    comparisons = sorted(output_dir.rglob("comparison/summary.json"))
    if len(comparisons) != 4:
        raise RuntimeError(f"Expected four paired comparison summaries, found {len(comparisons)}")
    return {"metrics": progress, "comparison_summaries":
            [str(path.relative_to(output_dir)) for path in comparisons]}


def terminate_group(proc: subprocess.Popen, grace: float = 20.) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def worker_pids(log_path: Path) -> list[int]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="replace")
    return [int(value) for value in re.findall(r"(?:PID=|pid=)(\d+)", text)]


def run_gpu_smoke(args, env: dict[str, str], smoke_dir: Path) -> dict:
    command = [sys.executable, str(ROOT / "scripts" / "gpu_memory_smoke_four_fcil.py"),
               "--gpu", "0", "--reserve-mib", str(args.smoke_reserve_mib),
               "--output-dir", str(smoke_dir)]
    result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (smoke_dir / "supervisor.log").write_text(result.stdout)
    report_path = smoke_dir / "gpu_smoke.json"
    if result.returncode != 0 or not report_path.is_file():
        failure_path = smoke_dir / "gpu_smoke_failed.json"
        failure = failure_path.read_text() if failure_path.is_file() else result.stdout[-4000:]
        raise RuntimeError(f"Eight-process CUDA memory smoke failed (exit {result.returncode}): {failure}")
    report = json.loads(report_path.read_text())
    if report.get("status") != "passed":
        raise RuntimeError(f"Eight-process CUDA memory smoke lacks the required VRAM margin: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.03)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-free-mib", type=int, default=28672,
                        help="minimum free GPU memory before launch; default reserves 4 GiB on a 32-GiB card")
    parser.add_argument("--smoke-reserve-mib", type=int, default=4096,
                        help="minimum measured free VRAM after the eight-process smoke")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--plans-only", action="store_true", help="validate all four job plans without GPU checks")
    parser.add_argument("--wait-for-gpu", action="store_true",
                        help="inside tmux, check GPU admission every --poll-seconds until it is available")
    parser.add_argument("--resume", action="store_true",
                        help="resume a failed campaign from its per-round checkpoints")
    args = parser.parse_args()
    if args.output_dir is None:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output_dir = ROOT / "results" / f"eight_arm_{stamp}"
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.data_root.resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    if args.wait_for_gpu and (args.preflight_only or args.plans_only):
        print("PREFLIGHT_BLOCKED: --wait-for-gpu cannot be combined with a preflight-only mode", file=sys.stderr)
        return 2
    if args.wait_for_gpu and not os.environ.get("TMUX"):
        print("PREFLIGHT_BLOCKED: --wait-for-gpu must run inside a persistent tmux session", file=sys.stderr)
        return 2
    try:
        if args.wait_for_gpu:
            plans, _ = preflight(args, env, plans_only=True)
            print(f"All four paired plans passed. Waiting for GPU 0 admission; rechecking every {args.poll_seconds}s.",
                  flush=True)
            while True:
                try:
                    hardware = hardware_preflight(args, env)
                    break
                except Exception as exc:
                    print(f"GPU_ADMISSION_WAIT: {exc}; next check in {args.poll_seconds}s", flush=True)
                    time.sleep(args.poll_seconds)
        else:
            plans, hardware = preflight(args, env, plans_only=args.plans_only)
    except Exception as exc:
        print(f"PREFLIGHT_BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 2
    if args.preflight_only or args.plans_only:
        print(json.dumps({"plans": plans, "hardware": hardware}, indent=2))
        return 0
    if not os.environ.get("TMUX"):
        print("PREFLIGHT_BLOCKED: run the campaign supervisor inside a persistent tmux session", file=sys.stderr)
        return 2
    try:
        bank_check = run_json([sys.executable, "-c",
            "from cafcl.data import get_data_bank; import json,sys; b=get_data_bank('cifar100',sys.argv[1],download=False,seed=0); print(json.dumps({'train':len(b.train),'validation':len(b.validation),'test':len(b.test),'classes':b.num_classes}))",
            str(args.data_root)], cwd=ROOT, env=env)
        if (bank_check["classes"] != 100 or bank_check["train"] != 45000
                or bank_check["validation"] != 5000 or bank_check["test"] != 10000):
            raise RuntimeError(f"Unexpected CIFAR-100 partition: {bank_check}")
    except Exception as exc:
        print(f"PREFLIGHT_BLOCKED: CIFAR-100 load failed: {exc}", file=sys.stderr)
        return 2
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest_path = args.output_dir / "campaign.json"
    previous = None
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume or not manifest_path.is_file():
            print(f"PREFLIGHT_BLOCKED: output directory is not empty (use --resume only for a failed campaign): {args.output_dir}", file=sys.stderr)
            return 2
        previous = json.loads(manifest_path.read_text())
        same_identity = (previous.get("commit") == commit
                         and previous.get("seed") == args.seed
                         and previous.get("lr") == args.lr
                         and previous.get("epochs") == args.epochs
                         and previous.get("gpu") == args.gpu
                         and previous.get("data_root") == str(args.data_root))
        resumable = {"failed", "preflight_failed", "preflight",
                     "waiting_for_gpu_smoke_admission", "waiting_for_gpu_after_smoke"}
        if previous.get("status") not in resumable or not same_identity:
            print("PREFLIGHT_BLOCKED: --resume requires a failed or safely waiting campaign with identical commit/config/data root", file=sys.stderr)
            return 2
        prior_pids = [previous.get("supervisor_pid")]
        prior_pids += [item.get("pid") for item in previous.get("controllers", {}).values()]
        prior_pids += [pid for pids in previous.get("worker_pids", {}).values() for pid in pids]
        active = []
        for pid in set(filter(None, prior_pids)):
            try:
                os.kill(int(pid), 0)
                cmdline = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ")
                if b"cafcl." in cmdline or b"pilot/run.sh" in cmdline or b"launch_four_fcil_pilots" in cmdline:
                    active.append(int(pid))
            except (ProcessLookupError, FileNotFoundError, PermissionError):
                pass
        if active:
            print(f"PREFLIGHT_BLOCKED: prior campaign processes still exist: {active}", file=sys.stderr)
            return 2
    else:
        if args.resume:
            print("PREFLIGHT_BLOCKED: --resume requires an existing failed campaign directory", file=sys.stderr)
            return 2
        args.output_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = int(previous.get("attempt_number", 1)) + 1 if previous else 1
    logs = args.output_dir / "logs" / f"attempt_{attempt_number:02d}"
    logs.mkdir(parents=True, exist_ok=False)
    smoke_attempts = []
    manifest_path = args.output_dir / "campaign.json"
    preflight_manifest = {"status": "preflight", "commit": commit,
                          "started_at_utc": utc_now(), "seed": args.seed,
                          "lr": args.lr, "epochs": args.epochs, "batch_size": 64,
                          "gpu": args.gpu, "data_root": str(args.data_root),
                          "dataset_partition": bank_check, "plans": plans,
                          "hardware": hardware, "output_dir": str(args.output_dir),
                          "logs_dir": str(logs), "attempt_number": attempt_number,
                          "supervisor_pid": os.getpid(),
                          "attempt_history": list(previous.get("attempt_history", [])) if previous else [],
                          "gpu_smoke_attempts": smoke_attempts,
                          "controllers": {}, "worker_pids": {}}
    if previous:
        preflight_manifest["attempt_history"].append({key: previous.get(key) for key in
            ("status", "started_at_utc", "finished_at_utc", "controller_exit_codes",
             "worker_pids", "failure", "logs_dir", "gpu_smoke_attempts")})
    atomic_manifest(manifest_path, preflight_manifest)

    def save_preflight_failure(message: str) -> int:
        preflight_manifest["status"] = "preflight_failed"
        preflight_manifest["failure"] = message
        preflight_manifest["finished_at_utc"] = utc_now()
        preflight_manifest["gpu_smoke_attempts"] = smoke_attempts
        atomic_manifest(manifest_path, preflight_manifest)
        print(f"PREFLIGHT_BLOCKED: {message}", file=sys.stderr, flush=True)
        return 2

    while True:
        try:
            pre_smoke_hardware = hardware_preflight(args, env)
        except Exception as exc:
            if not args.wait_for_gpu:
                return save_preflight_failure(f"before GPU smoke: {exc}")
            preflight_manifest["status"] = "waiting_for_gpu_smoke_admission"
            atomic_manifest(manifest_path, preflight_manifest)
            print(f"GPU_ADMISSION_WAIT before smoke: {exc}; next check in {args.poll_seconds}s", flush=True)
            time.sleep(args.poll_seconds)
            continue
        smoke_number = len(smoke_attempts) + 1
        smoke_dir = logs / f"gpu_memory_smoke_attempt_{smoke_number:02d}"
        try:
            smoke_report = run_gpu_smoke(args, env, smoke_dir)
        except Exception as exc:
            smoke_attempts.append({"directory": str(smoke_dir), "status": "failed", "failure": str(exc)})
            if args.wait_for_gpu and "Unrelated GPU compute processes appeared" in str(exc):
                preflight_manifest["status"] = "waiting_for_gpu_smoke_admission"
                preflight_manifest["gpu_smoke_attempts"] = smoke_attempts
                atomic_manifest(manifest_path, preflight_manifest)
                print(f"GPU_ADMISSION_WAIT during smoke: {exc}; next check in {args.poll_seconds}s", flush=True)
                time.sleep(args.poll_seconds)
                continue
            return save_preflight_failure(f"{exc}; smoke evidence: {smoke_dir}")
        smoke_attempts.append({"directory": str(smoke_dir), "status": "passed",
                               "peak_device_used_mib": smoke_report["peak_device_used_mib"],
                               "measured_free_at_peak_mib": smoke_report["measured_free_at_peak_mib"]})
        try:
            hardware = hardware_preflight(args, env)
            hardware["gpu_memory_smoke"] = smoke_report
            break
        except Exception as exc:
            if not args.wait_for_gpu:
                return save_preflight_failure(f"after GPU smoke: {exc}")
            preflight_manifest["status"] = "waiting_for_gpu_after_smoke"
            atomic_manifest(manifest_path, preflight_manifest)
            print(f"GPU_ADMISSION_WAIT after smoke: {exc}; next check in {args.poll_seconds}s", flush=True)
            time.sleep(args.poll_seconds)
    history = list(previous.get("attempt_history", [])) if previous else []
    if previous:
        history.append({key: previous.get(key) for key in
                        ("status", "started_at_utc", "finished_at_utc", "controller_exit_codes",
                         "worker_pids", "failure", "logs_dir")})
    manifest = {"status": "starting", "commit": commit, "started_at_utc": utc_now(),
                "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
                "batch_size": 64, "gpu": args.gpu, "data_root": str(args.data_root),
                "dataset_partition": bank_check, "plans": plans, "hardware": hardware,
                "controllers": {}, "worker_pids": {}, "snapshots": [],
                "output_dir": str(args.output_dir), "logs_dir": str(logs),
                "attempt_history": history, "attempt_number": attempt_number,
                "gpu_smoke_attempts": smoke_attempts, "supervisor_pid": os.getpid()}
    atomic_manifest(manifest_path, manifest)
    processes: dict[str, tuple[subprocess.Popen, object]] = {}
    failed = None
    try:
        for suite in SUITES:
            launcher_dir = {"glfc": "GLFC", "lga": "LGA", "feat": "FEAT", "fedclass": "FedCLASS"}[suite]
            launcher = ROOT / launcher_dir / "pilot" / "run.sh"
            if suite == "fedclass":
                command = ["bash", str(launcher), "--seed", str(args.seed), "--lr", str(args.lr),
                           "--epochs", str(args.epochs), "--gpu", "0", "--memory-mode", "reservoir",
                           "--replay-ratio", "1.0", "--data-root", str(args.data_root), "--no-download",
                           "--output-dir", str(args.output_dir / "fedclass")]
            else:
                command = ["bash", str(launcher), "--seed", str(args.seed), "--lr", str(args.lr),
                           "--epochs", str(args.epochs), "--gpu", "0", "--data-root", str(args.data_root),
                           "--no-download", "--output-dir", str(args.output_dir / "native")]
            log = (logs / f"{suite}.log").open("w")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            processes[suite] = (proc, log)
            manifest["controllers"][suite] = {"pid": proc.pid, "command": command,
                                               "started_at_utc": utc_now(), "exit_code": None}
            atomic_manifest(manifest_path, manifest)
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            if any(proc.poll() not in (None, 0) for proc, _ in processes.values()):
                failed = "a suite controller exited nonzero during worker startup"
                break
            for suite, (proc, _) in processes.items():
                log_path = logs / f"{suite}.log"
                manifest["worker_pids"][suite] = worker_pids(log_path)
            if sum(len(v) for v in manifest["worker_pids"].values()) >= 8:
                break
            time.sleep(1)
        if failed is None and sum(len(v) for v in manifest["worker_pids"].values()) < 8:
            failed = "did not observe all eight worker PIDs before startup timeout"
        if failed:
            raise RuntimeError(failed)
        workers = [pid for pids in manifest["worker_pids"].values() for pid in pids]
        if len(set(workers)) != 8:
            raise RuntimeError(f"Expected eight unique worker PIDs, got {workers}")
        for pid in workers:
            os.kill(pid, 0)
        startup_deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < startup_deadline:
            gpu_now = gpu_snapshot(args.gpu)
            cuda_pids = {item["pid"] for item in gpu_now["compute_processes"]}
            if set(workers) <= cuda_pids:
                manifest["all_eight_cuda_contexts_observed_at_utc"] = utc_now()
                break
            if any(proc.poll() not in (None, 0) for proc, _ in processes.values()):
                raise RuntimeError("a suite controller failed before all eight workers acquired CUDA")
            time.sleep(1)
        else:
            raise RuntimeError(f"All workers were spawned, but not all eight appeared on CUDA: "
                               f"workers={workers}, cuda_pids={sorted(cuda_pids)}")
        manifest["status"] = "running"
        manifest["all_eight_workers_observed_at_utc"] = utc_now()
        atomic_manifest(manifest_path, manifest)
        while any(proc.poll() is None for proc, _ in processes.values()):
            snapshot = gpu_snapshot(args.gpu)
            manifest["snapshots"].append(snapshot)
            manifest["metrics_progress"] = metrics_progress(args.output_dir)
            for suite, (proc, _) in processes.items():
                manifest["controllers"][suite]["exit_code"] = proc.poll()
                manifest["worker_pids"][suite] = worker_pids(logs / f"{suite}.log")
            nonzero = [suite for suite, (proc, _) in processes.items()
                       if proc.poll() not in (None, 0)]
            atomic_manifest(manifest_path, manifest)
            if nonzero:
                failed = f"suite controller failure: {nonzero}"
                raise RuntimeError(failed)
            expected_pids = set(workers)
            unexpected_gpu_pids = {item["pid"] for item in snapshot["compute_processes"]} - expected_pids
            if unexpected_gpu_pids:
                failed = f"unrelated GPU compute process appeared during campaign: {sorted(unexpected_gpu_pids)}"
                raise RuntimeError(failed)
            time.sleep(args.poll_seconds)
        codes = {suite: proc.wait() for suite, (proc, _) in processes.items()}
        manifest["controller_exit_codes"] = codes
        if all(code == 0 for code in codes.values()):
            manifest["result_verification"] = verify_results(args.output_dir)
            manifest["status"] = "completed"
        else:
            manifest["status"] = "failed"
        manifest["finished_at_utc"] = utc_now()
        atomic_manifest(manifest_path, manifest)
        return 0 if manifest["status"] == "completed" else 1
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failure"] = str(exc)
        manifest["finished_at_utc"] = utc_now()
        for suite, (proc, _) in processes.items():
            manifest["controllers"].setdefault(suite, {})["exit_code"] = proc.poll()
            terminate_group(proc)
        atomic_manifest(manifest_path, manifest)
        print(f"CAMPAIGN_FAILED: {exc}; see {manifest_path}", file=sys.stderr, flush=True)
        return 1
    finally:
        for _, log in processes.values():
            log.close()


if __name__ == "__main__":
    raise SystemExit(main())
