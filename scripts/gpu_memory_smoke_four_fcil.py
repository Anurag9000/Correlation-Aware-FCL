#!/usr/bin/env python3
"""Concurrent eight-process CUDA memory smoke for the four CIFAR-100 pilots.

Uses random 32x32 inputs, exact physical batch 64, each suite's production
backbone/loss, five client models plus a server model, and native teachers.
It creates no experiment checkpoints and uses no CIFAR labels or test data.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SUITES = (
    ("glfc", "native_glfc", "native_glfc_cbam"),
    ("lga", "native_lga", "native_lga_cbam"),
    ("feat", "native_feat", "native_feat_resnet"),
    ("fedclass", "replay", "fedclass_resnet8"),
)


def smoke_worker_specs():
    return tuple((index * 2 + arm, suite, method, model_id, arm_name)
                 for index, (suite, method, model_id) in enumerate(SUITES)
                 for arm, arm_name in enumerate(("full", "offline")))


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def gpu_usage(gpu=0):
    result = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.total,memory.used,memory.free",
         "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    total, used, free = map(int, result.stdout.strip().split(","))
    apps = subprocess.run(
        ["nvidia-smi", f"--id={gpu}", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if apps.returncode:
        raise RuntimeError(apps.stderr.strip())
    processes = []
    for line in apps.stdout.splitlines():
        cols = [value.strip() for value in line.split(",")]
        if len(cols) >= 2:
            processes.append({"pid": int(cols[0]), "used_mib": int(cols[1])})
    return {"total_mib": total, "used_mib": used, "free_mib": free,
            "compute_processes": processes, "at_utc": utc_now()}


def worker(suite, method, model_id, start_dir, timeout):
    import copy
    import torch
    import torch.nn.functional as F
    from cafcl.models import create_model

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(901 + SUITES.index((suite, method, model_id)))
    is_native = suite != "fedclass"
    server = create_model(model_id, 100).to(device)
    clients = [create_model(model_id, 100).to(device) for _ in range(5)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=.03, momentum=.9, weight_decay=.0005)
                  for m in clients]
    teachers = []
    if is_native:
        for model in clients:
            teacher = copy.deepcopy(model).to(device).eval()
            for parameter in teacher.parameters():
                parameter.requires_grad_(False)
            teachers.append(teacher)
    else:
        teachers = [None] * 5
    streams = [torch.cuda.Stream(device=device) for _ in clients]
    inputs = [torch.randn(64, 3, 32, 32, device=device) for _ in clients]
    labels = [torch.randint(0, 4, (64,), device=device) for _ in clients]
    torch.cuda.synchronize(device)
    start = Path(start_dir) / "start"
    ready = Path(start_dir) / f"ready_{os.getpid()}"
    ready.write_text(json.dumps({"pid": os.getpid(), "suite": suite,
                                 "model_id": model_id, "native_teacher_models": len(teachers) if is_native else 0}))
    deadline = time.monotonic() + timeout
    while not start.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for all eight workers to be ready")
        time.sleep(.05)
    def train_one(index):
        model, optimizer, stream = clients[index], optimizers[index], streams[index]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            model.train()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                x, y = inputs[index], labels[index]
                if is_native:
                    from cafcl.native_methods import native_loss
                    loss, _ = native_loss(method, model, x, y,
                                          known_classes={0, 1, 2, 3},
                                          teacher=teachers[index], old_classes={0, 1})
                else:
                    loss = F.cross_entropy(model(x)[:, :4], y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite {suite} smoke loss")
                loss.backward()
                optimizer.step()
        return float(loss.detach().cpu())

    with ThreadPoolExecutor(max_workers=5, thread_name_prefix=f"{suite}-smoke") as pool:
        losses = list(pool.map(train_one, range(5)))
    torch.cuda.synchronize(device)
    result = {
        "suite": suite, "method": method, "model_id": model_id,
        "pid": os.getpid(), "physical_batch_size": 64,
        "client_models": len(clients), "server_models": 1,
        "teacher_models": len(teachers) if is_native else 0,
        "steps_per_client": 2, "losses": losses,
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        "finished_at_utc": utc_now(),
    }
    print("GPU_SMOKE_RESULT " + json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--reserve-mib", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(8), default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--start-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.gpu != 0:
        parser.error("The four suite pilots are pinned to GPU 0")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.worker is not None:
        if args.start_dir is None:
            parser.error("--start-dir is required for a worker")
        suite, method, model_id = SUITES[args.worker // 2]
        worker(suite, method, model_id, args.start_dir, args.timeout)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=False)
    start_dir = args.output_dir / "barrier"
    start_dir.mkdir()
    workers = []
    logs = []
    initial_usage = gpu_usage(args.gpu)
    if initial_usage["compute_processes"]:
        raise RuntimeError(f"Refusing memory smoke while GPU has existing compute PIDs: {initial_usage['compute_processes']}")
    peak = initial_usage["used_mib"]
    baseline_used = initial_usage["used_mib"]
    sampled = [initial_usage]
    failed = None
    try:
        for worker_index, suite, method, model_id, arm in smoke_worker_specs():
            log_path = args.output_dir / f"{suite}_{arm}.log"
            handle = log_path.open("w")
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--worker", str(worker_index),
                       "--start-dir", str(start_dir), "--timeout", str(args.timeout)]
            proc = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(),
                                    stdout=handle, stderr=subprocess.STDOUT)
            workers.append((proc, suite, arm))
            logs.append(handle)
        deadline = time.monotonic() + args.timeout
        ready_expected = 8
        while time.monotonic() < deadline:
            ready_files = list(start_dir.glob("ready_*"))
            for proc, suite, arm in workers:
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f"{suite}/{arm} worker failed during initialization; "
                                       f"inspect its smoke log")
            if len(ready_files) == ready_expected:
                break
            time.sleep(.1)
        else:
            raise TimeoutError(f"Only {len(list(start_dir.glob('ready_*')))} of eight smoke workers initialized")
        expected_pids = {int(json.loads(path.read_text())["pid"]) for path in ready_files}
        ready_snapshot = gpu_usage(args.gpu)
        if {item["pid"] for item in ready_snapshot["compute_processes"]} != expected_pids:
            raise RuntimeError("GPU compute PID set did not match exactly the eight isolated smoke workers: "
                               f"expected={sorted(expected_pids)} actual={ready_snapshot['compute_processes']}")
        (start_dir / "start").write_text(utc_now())
        while any(proc.poll() is None for proc, _, _ in workers):
            snapshot = gpu_usage(args.gpu)
            sampled.append(snapshot)
            peak = max(peak, snapshot["used_mib"])
            unexpected = {item["pid"] for item in snapshot["compute_processes"]} - expected_pids
            if unexpected:
                raise RuntimeError(f"Unrelated GPU compute processes appeared during smoke: {sorted(unexpected)}")
            for proc, suite, arm in workers:
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f"{suite}/{arm} worker failed during training; inspect its smoke log")
            time.sleep(.2)
        codes = [(suite, arm, proc.wait()) for proc, suite, arm in workers]
        bad = [item for item in codes if item[2] != 0]
        if bad:
            raise RuntimeError(f"CUDA smoke subprocesses failed: {bad}")
        usage = gpu_usage(args.gpu)
        peak = max(peak, usage["used_mib"])
        results = []
        for path in sorted(args.output_dir.glob("*.log")):
            for line in path.read_text(errors="replace").splitlines():
                if line.startswith("GPU_SMOKE_RESULT "):
                    results.append(json.loads(line[len("GPU_SMOKE_RESULT "):]))
        if len(results) != 8:
            raise RuntimeError(f"Expected eight worker smoke results, found {len(results)}")
        allocator_estimate = baseline_used + sum(item["peak_reserved_mib"] for item in results)
        peak = max(peak, allocator_estimate)
        reserve_available = usage["total_mib"] - peak
        report = {
            "status": "passed" if reserve_available >= args.reserve_mib else "insufficient_margin",
            "started_at_utc": sampled[0]["at_utc"] if sampled else utc_now(),
            "finished_at_utc": utc_now(), "gpu": usage,
            "baseline_device_used_mib": baseline_used,
            "peak_device_used_mib": peak, "allocator_peak_estimate_mib": allocator_estimate,
            "measured_free_at_peak_mib": reserve_available,
            "required_free_margin_mib": args.reserve_mib,
            "workers": results, "samples": sampled,
        }
        (args.output_dir / "gpu_smoke.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "passed" else 2
    except Exception as exc:
        failed = str(exc)
        for proc, _, _ in workers:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    proc.terminate()
        for proc, _, _ in workers:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        report = {"status": "failed", "failure": failed,
                  "finished_at_utc": utc_now(), "peak_device_used_mib": peak,
                  "samples": sampled}
        (args.output_dir / "gpu_smoke_failed.json").write_text(json.dumps(report, indent=2))
        print(f"GPU_SMOKE_FAILED: {failed}", file=sys.stderr, flush=True)
        return 1
    finally:
        for handle in logs:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
