#!/usr/bin/env python3
"""Rank completed tuning runs using validation-only stability/plasticity metrics."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/cifar10_tuning_v1/job_plan.json")
    parser.add_argument("--output-dir", default="results/cifar10_tuning_v1")
    parser.add_argument("--window", type=int, default=4)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.output_dir)
    rows = []
    for raw in plan["jobs"]:
        job = root / "runs" / raw["job_id"]
        config = json.loads((job / "config.json").read_text())
        if config != raw:
            raise AssertionError(f"{raw['job_id']}: saved config differs from tuning plan")
        status = json.loads((job / "status.json").read_text())
        if status.get("status") != "completed":
            raise AssertionError(f"{raw['job_id']}: run is not complete")
        metrics = json.loads((job / "metrics.json").read_text())
        expected = raw["stages"] * raw["rounds_per_stage"]
        if len(metrics) != expected or any(r.get("evaluation_split") != "validation" for r in metrics):
            raise AssertionError(f"{raw['job_id']}: incomplete rounds or non-validation feedback")
        tail = metrics[-args.window:]
        if len(tail) != args.window or any(r.get("stage") != raw["stages"] - 1 for r in tail):
            raise AssertionError(f"{raw['job_id']}: selection window is not in the final stage")
        old = sum(float(r["old_accuracy"]) for r in tail) / len(tail)
        new = sum(float(r["new_accuracy"]) for r in tail) / len(tail)
        macro = sum(float(r["macro_accuracy"]) for r in tail) / len(tail)
        forgetting = sum(float(r["integrated_forgetting"]) for r in tail) / len(tail)
        coverage = len(metrics[-1]["globally_integrated_classes"]) / 10
        seconds = sum(float(r["seconds"]) for r in metrics)
        harmonic = 0.0 if old + new == 0 else 2 * old * new / (old + new)
        values = [old, new, macro, forgetting, coverage, seconds, harmonic]
        if not all(math.isfinite(x) for x in values):
            raise AssertionError(f"{raw['job_id']}: nonfinite selection metric")
        rows.append({
            "job_id": raw["job_id"], "learning_rate": raw["lr"], "momentum": raw["momentum"],
            "old_validation_accuracy": old, "new_validation_accuracy": new,
            "harmonic_old_new_validation_accuracy": harmonic,
            "all_observed_validation_macro_accuracy": macro,
            "integrated_class_coverage": coverage,
            "integrated_forgetting": forgetting, "training_seconds": seconds,
        })

    # Primary objective is the old/new harmonic mean; the remaining metrics
    # are prespecified tie-breakers in this order.
    ranked = sorted(rows, key=lambda r: (
        r["harmonic_old_new_validation_accuracy"],
        r["all_observed_validation_macro_accuracy"],
        r["integrated_class_coverage"], -r["integrated_forgetting"],
        -r["training_seconds"]), reverse=True)
    viable = [r for r in ranked if r["harmonic_old_new_validation_accuracy"] > 0]
    selected = viable[0] if viable else None
    result = {
        "suite": plan["suite"], "selection_split": "validation",
        "official_test_used_for_selection": False,
        "window_rounds": args.window,
        "selection_order": ["harmonic_old_new_validation_accuracy",
                            "all_observed_validation_macro_accuracy",
                            "integrated_class_coverage", "lower_integrated_forgetting",
                            "lower_training_seconds"],
        "decision": ("selected" if selected else "no_candidate_passed_old_new_learning_gate"),
        "selected_job_id": selected["job_id"] if selected else None,
        "selected_candidate": selected,
        "ranked_candidates": ranked,
        "limitations": (
            f"Training fraction {plan.get('training_fraction_of_eligible_data', 'per-job')}; "
            f"{plan.get('rounds_per_stage', 'per-job')} rounds per experience across "
            f"{len(plan['jobs'][0].get('stages', [])) if isinstance(plan['jobs'][0].get('stages'), list) else plan['jobs'][0].get('stages', 'unknown')} "
            "experiences. Selection uses validation only; this ranking does not establish "
            "statistical significance or replace production artifact verification."
        ),
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    with (root / "selection.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ranked)
    print(json.dumps({"decision": result["decision"], "selected_job_id": result["selected_job_id"],
                      "ranked_candidates": ranked}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
