import json
import subprocess
import sys
from pathlib import Path


def test_tuning_selection_requires_old_and_new_accuracy_above_chance(tmp_path):
    plan = {"suite": "validation_hyperparameter_tuning", "jobs": []}
    candidates = [
        ("old_only", 0.90, 0.10),
        ("both_above", 0.30, 0.20),
        ("same_accuracy_but_faster", 0.30, 0.20),
        ("new_only", 0.99, 0.05),
    ]
    for job_id, old, new in candidates:
        config = {"job_id": job_id, "lr": 0.03, "momentum": 0.0,
                  "stages": 2, "rounds_per_stage": 1}
        plan["jobs"].append(config)
        run = tmp_path / "runs" / job_id
        run.mkdir(parents=True)
        (run / "config.json").write_text(json.dumps(config))
        (run / "status.json").write_text(json.dumps({"status": "completed"}))
        metrics = [
            {"round": 0, "stage": 0, "evaluation_split": "validation",
             "old_accuracy": old, "new_accuracy": new,
             "macro_accuracy": (old + new) / 2, "integrated_forgetting": 0.0,
             "globally_integrated_classes": list(range(10)),
             "seconds": 0.01 if job_id == "same_accuracy_but_faster" else 1.0},
            {"round": 1, "stage": 1, "evaluation_split": "validation",
             "old_accuracy": old, "new_accuracy": new,
             "macro_accuracy": (old + new) / 2, "integrated_forgetting": 0.0,
             "globally_integrated_classes": list(range(10)),
             "seconds": 0.01 if job_id == "same_accuracy_but_faster" else 1.0},
        ]
        (run / "metrics.json").write_text(json.dumps(metrics))

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    script = Path(__file__).resolve().parents[1] / "scripts" / "summarize_cifar10_tuning.py"
    subprocess.run([sys.executable, str(script), "--plan", str(plan_path),
                    "--output-dir", str(tmp_path), "--window", "1"],
                   check=True, capture_output=True, text=True)

    result = json.loads((tmp_path / "selection.json").read_text())
    assert result["selected_job_id"] == "both_above"
    assert result["official_test_used_for_selection"] is False
    assert result["runtime_used_for_selection"] is False
    by_id = {row["job_id"]: row for row in result["ranked_candidates"]}
    assert by_id["old_only"]["learning_gate_passed"] is False
    assert by_id["both_above"]["learning_gate_passed"] is True
    assert by_id["new_only"]["learning_gate_passed"] is False
