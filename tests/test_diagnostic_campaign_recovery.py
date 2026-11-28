import json
from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cafcl import diagnostic_campaign as campaign


def job(job_id):
    return {
        "job_id": job_id,
        "family": "FL",
        "binding": "stationary_reference",
        "objective": "global",
        "seed": 42,
        "clients": 1,
        "partition": "all_data",
        "curriculum": "stationary_30_class",
        "rounds_per_stage": 1,
        "local_epochs": 1,
        "participation": 1.0,
        "il": "stationary",
        "checkpoint_rounds": [1],
        "status": "planned",
    }


def args(tmp_path):
    return SimpleNamespace(data_root=str(tmp_path / "data"), command="run")


def test_retry_uses_a_separate_attempt_directory(tmp_path):
    spec = job("RETRY-IDENTITY")
    root = tmp_path / "runs"
    codes = iter((17, 0))

    def worker(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        (output / "worker_marker").write_text("attempt")
        return SimpleNamespace(returncode=next(codes))

    def validate(expected, output):
        assert (output / "job.json").is_file()
        assert (output / "worker_marker").read_text() == "attempt"
        return {"job_id": expected["job_id"], "sha256": {}}

    with patch.object(campaign, "source_identity", return_value=("a" * 40, False)), \
         patch.object(campaign.subprocess, "run", side_effect=worker), \
         patch.object(campaign, "validate_artifacts", side_effect=validate):
        with pytest.raises(RuntimeError, match="Worker exit code 17"):
            campaign.run_one(spec, args(tmp_path), root, "cpu", "a" * 40)
        campaign.run_one(spec, args(tmp_path), root, "cpu", "a" * 40)

    base = root / "FL" / spec["job_id"]
    first, second = base / "attempt_0001", base / "attempt_0002"
    first_status = json.loads((first / "status.json").read_text())
    second_status = json.loads((second / "status.json").read_text())
    assert first_status["state"] == "failed"
    assert second_status["state"] == "completed"
    assert first_status["signature"] == second_status["signature"]
    assert (first / "stdout.log").is_file() and (second / "stdout.log").is_file()
    assert (second / "artifact_validation.json").is_file()


def test_zero_exit_without_artifacts_is_not_completed(tmp_path):
    spec = job("EXIT-WITHOUT-ARTIFACTS")
    root = tmp_path / "runs"
    with patch.object(campaign, "source_identity", return_value=("b" * 40, False)), \
         patch.object(campaign.subprocess, "run", return_value=SimpleNamespace(returncode=0)):
        with pytest.raises(ValueError, match="Missing partition.json"):
            campaign.run_one(spec, args(tmp_path), root, "cpu", "b" * 40)

    status = json.loads((root / "FL" / spec["job_id"] / "attempt_0001" / "status.json").read_text())
    assert status["state"] == "failed"
    assert "Missing partition.json" in status["error"]


def test_duplicate_claim_is_rejected_while_worker_is_running(tmp_path):
    spec = job("DUPLICATE-CLAIM")
    root = tmp_path / "runs"
    started, release = Event(), Event()
    thread_errors = []

    def worker(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        return SimpleNamespace(returncode=17)

    def first_claim():
        try:
            campaign.run_one(spec, args(tmp_path), root, "cpu", "c" * 40)
        except Exception as exc:
            thread_errors.append(exc)

    with patch.object(campaign, "source_identity", return_value=("c" * 40, False)), \
         patch.object(campaign.subprocess, "run", side_effect=worker):
        thread = Thread(target=first_claim)
        thread.start()
        assert started.wait(5)
        with pytest.raises(RuntimeError, match="Active job lock"):
            campaign.run_one(spec, args(tmp_path), root, "cpu", "c" * 40)
        release.set()
        thread.join(5)

    assert not thread.is_alive()
    assert len(thread_errors) == 1
    assert "Worker exit code 17" in str(thread_errors[0])
    attempts = sorted((root / "FL" / spec["job_id"]).glob("attempt_*"))
    assert [path.name for path in attempts] == ["attempt_0001"]
