#!/usr/bin/env python3
"""Write a reproducible CUDA, software, Git, and CIFAR-10 environment receipt."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path


def command_output(args: list[str]) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    import torch
    import torchvision
    from torchvision.datasets import CIFAR10

    if not torch.cuda.is_available():
        raise SystemExit("CUDA preflight failed; environment receipt not written")
    archive = Path(args.data_root) / "cifar-10-python.tar.gz"
    if not archive.is_file():
        raise SystemExit(f"Official CIFAR-10 archive is missing: {archive}")
    archive_md5 = md5_file(archive)
    if archive_md5 != "c58f30108f718f92721af3b95e74349a":
        raise SystemExit(f"Unexpected CIFAR-10 archive MD5: {archive_md5}")
    train = CIFAR10(root=args.data_root, train=True, download=False)
    test = CIFAR10(root=args.data_root, train=False, download=False)
    if len(train) != 50000 or len(test) != 10000 or train.classes != test.classes:
        raise SystemExit("CIFAR-10 split size or class mapping is invalid")
    props = torch.cuda.get_device_properties(0)
    requirements = Path("requirements-cuda121.txt")
    git_commit = command_output(["git", "rev-parse", "HEAD"])
    git_branch = command_output(["git", "branch", "--show-current"])
    git_status = command_output(["git", "status", "--short"])
    payload = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "git": {"commit": git_commit, "branch": git_branch,
                "dirty_status": git_status},
        "python": {"version": sys.version, "executable": sys.executable,
                   "platform": platform.platform()},
        "frameworks": {"torch": torch.__version__,
                       "torchvision": torchvision.__version__,
                       "cuda_build": torch.version.cuda,
                       "cudnn": torch.backends.cudnn.version()},
        "gpu": {"name": torch.cuda.get_device_name(0),
                "device_count": torch.cuda.device_count(),
                "total_memory_bytes": props.total_memory,
                "driver": command_output(["nvidia-smi", "--query-gpu=driver_version",
                                           "--format=csv,noheader"])},
        "dependencies": sorted(
            [{"name": item.metadata["Name"], "version": item.version}
             for item in importlib.metadata.distributions()
             if item.metadata.get("Name")], key=lambda x: x["name"].lower()),
        "requirements_cuda121_sha256": sha256_file(requirements) if requirements.exists() else None,
        "dataset": {"name": "CIFAR-10", "archive": str(archive),
                    "archive_bytes": archive.stat().st_size,
                    "archive_md5": archive_md5,
                    "archive_sha256": sha256_file(archive),
                    "train_samples": len(train), "test_samples": len(test),
                    "classes": train.classes,
                    "train_class_counts": dict(sorted(Counter(train.targets).items())),
                    "test_class_counts": dict(sorted(Counter(test.targets).items()))},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "git_commit": git_commit,
                      "python": platform.python_version(), "torch": torch.__version__,
                      "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
                      "gpu": torch.cuda.get_device_name(0), "train": len(train),
                      "test": len(test), "archive_md5": archive_md5}, indent=2))


if __name__ == "__main__":
    main()
