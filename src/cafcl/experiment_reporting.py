"""Shared extraction of final metrics from completed experiment runs."""
from __future__ import annotations
import json
from pathlib import Path

def final_metrics_from_result(result):
    metrics=result.get("final_metrics")
    if not metrics:
        run_dir=result.get("run_dir")
        if not run_dir:
            raise RuntimeError("Completed experiment result has no run directory")
        summary_path=Path(run_dir)/"summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"Completed experiment is missing {summary_path}")
        metrics=json.loads(summary_path.read_text()).get("final_metrics")
    if not metrics or metrics.get("accuracy") is None:
        raise RuntimeError("Completed experiment has no final accuracy metrics")
    return metrics

def validation_accuracy(metrics):
    return metrics.get("validation_accuracy",metrics.get("per_sync_validation_accuracy"))
