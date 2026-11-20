import json
from pathlib import Path

import pytest

from scripts.launch_four_fcil_pilots import hardware_preflight, metrics_progress, verify_results
from scripts.gpu_memory_smoke_four_fcil import smoke_worker_specs


def _write_campaign(root: Path, *, complete=True):
    for index in range(8):
        path = root / f"suite_{index}" / "run" / "metrics.json"
        path.parent.mkdir(parents=True)
        rows = [{"round": round_id, "accuracy": 0.1,
                 "terminal_communication_only_flush": round_id == 9}
                for round_id in range(10)]
        if not complete and index == 7:
            rows[-1]["terminal_communication_only_flush"] = False
        path.write_text(json.dumps(rows))
    for index in range(4):
        path = root / f"suite_{index}" / "comparison" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")


def test_campaign_verifier_requires_eight_complete_ten_sync_runs_and_four_pairs(tmp_path):
    _write_campaign(tmp_path)
    evidence = verify_results(tmp_path)
    assert len(evidence["metrics"]) == 8
    assert len(evidence["comparison_summaries"]) == 4


def test_campaign_verifier_rejects_missing_terminal_flush(tmp_path):
    _write_campaign(tmp_path, complete=False)
    with pytest.raises(RuntimeError, match="terminal U10 flush"):
        verify_results(tmp_path)


def test_metrics_progress_records_terminal_and_accuracy(tmp_path):
    _write_campaign(tmp_path)
    result = metrics_progress(tmp_path)
    first = next(iter(result.values()))
    assert first["rows"] == 10
    assert first["terminal_flush"] is True
    assert first["test_accuracy"] == 0.1


def test_memory_smoke_covers_all_four_suites_in_both_arms():
    specs = smoke_worker_specs()
    assert len(specs) == 8
    assert {suite for _, suite, *_ in specs} == {"glfc", "lga", "feat", "fedclass"}
    assert {arm for *_, arm in specs} == {"full", "offline"}
    assert len({worker_id for worker_id, *_ in specs}) == 8


def test_shared_gpu_admission_requires_explicit_flag_and_memory_reserve(monkeypatch):
    import types
    import scripts.launch_four_fcil_pilots as campaign
    gpu = {"free_mib": 11000, "compute_processes": [{"pid": 77, "used_mib": 20000}]}
    monkeypatch.setattr(campaign, "gpu_snapshot", lambda _: gpu)
    monkeypatch.setattr(campaign, "run_json", lambda *a, **k: {"available": True, "count": 1})
    ordinary = types.SimpleNamespace(gpu=0, allow_shared_gpu=False, min_free_mib=10000)
    with pytest.raises(RuntimeError, match="unrelated compute processes"):
        hardware_preflight(ordinary, {})
    shared = types.SimpleNamespace(gpu=0, allow_shared_gpu=True, min_free_mib=10000)
    receipt = hardware_preflight(shared, {})
    assert receipt["gpu"]["compute_processes"][0]["pid"] == 77
    shared.min_free_mib = 12000
    with pytest.raises(RuntimeError, match="requires 12000 MiB"):
        hardware_preflight(shared, {})
