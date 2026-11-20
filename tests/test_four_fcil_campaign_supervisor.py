import json
from pathlib import Path

import pytest

from scripts.launch_four_fcil_pilots import metrics_progress, verify_results


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
