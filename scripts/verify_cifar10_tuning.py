#!/usr/bin/env python3
"""Independently audit every CIFAR-10 tuning artifact and CUDA final model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_tree(value, where):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{where}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_tree(child, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise AssertionError(f"Nonfinite value at {where}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", default="results/cifar10_tuning_v1/job_plan.json")
    parser.add_argument("--output-dir", default="results/cifar10_tuning_v1")
    args = parser.parse_args()
    import torch
    from cafcl.config import RunConfig
    from cafcl.models import create_model, task_class_sets_from_state_dict

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to audit saved tuning models")
    plan = json.loads(Path(args.plan).read_text())
    if plan.get("suite") != "validation_hyperparameter_tuning":
        raise ValueError("Unexpected tuning suite identifier")
    jobs = [RunConfig(**raw) for raw in plan["jobs"]]
    root = Path(args.output_dir)
    checked = []
    for cfg in jobs:
        run = root / "runs" / cfg.job_id
        required = ["config.json", "status.json", "metrics.json", "summary.json",
                    "model_final.pt", "final_test_metrics.json", "data_partition.json",
                    "scenario.pt"]
        required.extend(f"model_stage_{stage+1:02d}.pt" for stage in range(cfg.stages))
        missing = [name for name in required if not (run/name).is_file()]
        if missing:
            raise AssertionError(f"{cfg.job_id}: missing artifacts {missing}")
        if json.loads((run/"config.json").read_text()) != cfg.to_dict():
            raise AssertionError(f"{cfg.job_id}: configuration differs from frozen plan")
        partition = json.loads((run/"data_partition.json").read_text())
        per_class_train = int(round(4500 * cfg.train_fraction))
        if (partition.get("train_samples") != per_class_train * 10 or
                partition.get("validation_samples") != 5000 or
                partition.get("official_test_samples") != 10000 or
                partition.get("train_class_counts") !=
                    {str(label): per_class_train for label in range(10)} or
                partition.get("validation_class_counts") !=
                    {str(label): 500 for label in range(10)} or
                partition.get("official_test_class_counts") !=
                    {str(label): 1000 for label in range(10)} or
                not partition.get("train_validation_source_indices_disjoint") or
                not partition.get("official_test_is_original_cifar_test_split")):
            raise AssertionError(f"{cfg.job_id}: training/validation/test split is invalid")
        status = json.loads((run/"status.json").read_text())
        if status.get("status") != "completed":
            raise AssertionError(f"{cfg.job_id}: run status is not completed")
        metrics = json.loads((run/"metrics.json").read_text())
        expected = cfg.stages * cfg.rounds_per_stage
        if len(metrics) != expected or [row.get("round") for row in metrics] != list(range(expected)):
            raise AssertionError(f"{cfg.job_id}: round count/order mismatch")
        for row in metrics:
            finite_tree(row, f"{cfg.job_id}.round{row.get('round')}")
            if row.get("evaluation_split") != "validation":
                raise AssertionError(f"{cfg.job_id}: a training round used non-validation feedback")
            if not str(row.get("model_execution_device", "")).startswith("cuda"):
                raise AssertionError(f"{cfg.job_id}: round was not executed on CUDA")
            if row.get("selected_count") != len(row.get("selected_clients", [])):
                raise AssertionError(f"{cfg.job_id}: selected-client count mismatch")
            weights = row.get("aggregation_weights", {})
            if len(weights) != row.get("uploaded_count") or not weights:
                raise AssertionError(f"{cfg.job_id}: aggregation weights do not match uploads")
            if any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(
                    sum(weights.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7):
                raise AssertionError(f"{cfg.job_id}: invalid aggregation weights")
            if row.get("communication_total_bytes") != row.get("update_transmitted_bytes", 0) + row.get("model_broadcast_bytes", 0):
                raise AssertionError(f"{cfg.job_id}: communication-byte accounting mismatch")
            if any(int(v) != row.get("model_version")
                   for v in row.get("client_training_versions", {}).values()):
                raise AssertionError(f"{cfg.job_id}: selected client missed model synchronization")
        final_test = json.loads((run/"final_test_metrics.json").read_text())
        finite_tree(final_test, f"{cfg.job_id}.final_test")
        if final_test.get("evaluation_split") != "test":
            raise AssertionError(f"{cfg.job_id}: final saved-model evaluation is not on test split")
        summary = json.loads((run/"summary.json").read_text())
        if summary.get("require_cuda") is not True or not str(
                summary.get("model_execution_device", "")).startswith("cuda"):
            raise AssertionError(f"{cfg.job_id}: CUDA execution requirement is not recorded")
        model_file = torch.load(run/"model_final.pt", map_location="cpu", weights_only=False)
        if model_file.get("configuration") != cfg.to_dict():
            raise AssertionError(f"{cfg.job_id}: final model config mismatch")
        state = model_file["model_state"]
        if any(not torch.isfinite(t).all() for t in state.values() if t.is_floating_point()):
            raise AssertionError(f"{cfg.job_id}: nonfinite final model tensor")
        model = create_model(cfg.model, 10,
            task_class_sets=task_class_sets_from_state_dict(state)).to("cuda")
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32, device="cuda"))
        if output.shape != (1, 10) or not torch.isfinite(output).all():
            raise AssertionError(f"{cfg.job_id}: reloaded CUDA inference failed")
        checked.append(cfg.job_id)
    receipt = {"suite": plan["suite"], "verified_jobs": len(checked),
               "job_ids": checked, "all_configured_batch_sizes": [cfg.batch_size for cfg in jobs],
               "all_cuda": True, "all_models_finite_and_reloadable": True}
    (root/"verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified {len(checked)}/{len(jobs)} tuning jobs: CUDA, finite, complete, reloadable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
