#!/usr/bin/env python3
"""Independently verify artifacts and execution evidence for a CIFAR-10 pilot."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cafcl.config import RunConfig
from cafcl.models import create_model, task_class_sets_from_state_dict


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
    parser.add_argument("--plan", default="results/cifar10_gpu_pilot_v1/job_plan.json")
    parser.add_argument("--output-dir", default="results/cifar10_gpu_pilot_v1")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to verify pilot model reload and inference")
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.output_dir)
    jobs = [RunConfig(**raw) for raw in plan["jobs"]]
    ids = [cfg.job_id for cfg in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("Duplicate pilot job IDs")
    checked = 0
    for cfg in jobs:
        run = root / "runs" / cfg.job_id
        source_run = (root / "runs" / cfg.fork_source_job
                      if cfg.fork_source_job else None)
        required = ["config.json", "status.json", "metrics.json", "summary.json",
                    "model_final.pt", "final_test_metrics.json", "data_partition.json",
                    "scenario.pt"]
        for stage in range(cfg.stages):
            stage_end_round = (stage + 1) * cfg.rounds_per_stage
            artifact_run = (source_run if source_run is not None and
                            stage_end_round <= cfg.fork_checkpoint_round else run)
            artifact = f"model_stage_{stage+1:02d}.pt"
            if artifact_run is None or artifact_run == run:
                required.append(artifact)
            elif not (artifact_run / artifact).is_file():
                raise AssertionError(f"{cfg.job_id}: source run lacks {artifact}")
        if cfg.checkpoint_every:
            for round_id in range(cfg.stages * cfg.rounds_per_stage):
                absolute_round = round_id + 1
                artifact_run = (source_run if source_run is not None and
                                absolute_round <= cfg.fork_checkpoint_round else run)
                artifact = f"checkpoint_round_{absolute_round:04d}.pt"
                if artifact_run is None or artifact_run == run:
                    required.append(artifact)
                elif not (artifact_run / artifact).is_file():
                    raise AssertionError(f"{cfg.job_id}: source run lacks {artifact}")
        missing = [name for name in required if not (run/name).is_file()]
        if missing:
            raise AssertionError(f"{cfg.job_id}: missing artifacts {missing}")
        if json.loads((run/"config.json").read_text()) != cfg.to_dict():
            raise AssertionError(f"{cfg.job_id}: saved configuration differs from plan")
        if source_run is not None:
            source_config_path = source_run / "config.json"
            source_checkpoint = source_run / (
                f"checkpoint_round_{cfg.fork_checkpoint_round:04d}.pt")
            if not source_config_path.is_file() or not source_checkpoint.is_file():
                raise AssertionError(f"{cfg.job_id}: causal source checkpoint is missing")
            source_config = json.loads(source_config_path.read_text())
            allowed = {"job_id", "force_target", "target_client", "intervention_round",
                       "causal_diagnostics", "gradient_diagnostics", "checkpoint_every",
                       "fork_source_job", "fork_checkpoint_round"}
            if ({k: v for k, v in source_config.items() if k not in allowed} !=
                    {k: v for k, v in cfg.to_dict().items() if k not in allowed}):
                raise AssertionError(f"{cfg.job_id}: source configuration is confounded")
            source_metrics = json.loads((source_run / "metrics.json").read_text())
            fork_metrics = json.loads((run / "metrics.json").read_text())
            if source_metrics[:cfg.fork_checkpoint_round] != fork_metrics[:cfg.fork_checkpoint_round]:
                raise AssertionError(f"{cfg.job_id}: pre-fork metric history differs from source")
        partition = json.loads((run/"data_partition.json").read_text())
        expected_train_counts = {str(label): 675 for label in range(10)}
        expected_validation_counts = {str(label): 500 for label in range(10)}
        expected_test_counts = {str(label): 1000 for label in range(10)}
        if (partition.get("train_samples") != 6750 or
                partition.get("validation_samples") != 5000 or
                partition.get("official_test_samples") != 10000 or
                partition.get("train_class_counts") != expected_train_counts or
                partition.get("validation_class_counts") != expected_validation_counts or
                partition.get("official_test_class_counts") != expected_test_counts or
                not partition.get("train_validation_source_indices_disjoint") or
                not partition.get("official_test_is_original_cifar_test_split")):
            raise AssertionError(f"{cfg.job_id}: train/validation/test partition is invalid")
        status = json.loads((run/"status.json").read_text())
        if status.get("status") != "completed":
            raise AssertionError(f"{cfg.job_id}: status is {status.get('status')}")
        metrics = json.loads((run/"metrics.json").read_text())
        expected_rounds = cfg.stages * cfg.rounds_per_stage
        if len(metrics) != expected_rounds or [x.get("round") for x in metrics] != list(range(expected_rounds)):
            raise AssertionError(f"{cfg.job_id}: wrong round count/order")
        for row in metrics:
            finite_tree(row, f"{cfg.job_id}.round{row.get('round')}")
            if row.get("evaluation_split") != "validation":
                raise AssertionError(f"{cfg.job_id}: non-validation round feedback")
            if not str(row.get("model_execution_device", "")).startswith("cuda"):
                raise AssertionError(f"{cfg.job_id}: non-CUDA training/evaluation")
            if row.get("selected_count") != len(row.get("selected_clients", [])):
                raise AssertionError(f"{cfg.job_id}: participation count mismatch")
            if row.get("uploaded_count") > row.get("selected_count"):
                raise AssertionError(f"{cfg.job_id}: uploaded more clients than selected")
            weights = row.get("aggregation_weights", {})
            if len(weights) != row.get("uploaded_count"):
                raise AssertionError(f"{cfg.job_id}: aggregation weights do not match uploads")
            if weights and (any(not math.isfinite(value) or value < 0 for value in weights.values()) or
                            not math.isclose(sum(weights.values()), 1., rel_tol=1e-7, abs_tol=1e-7)):
                raise AssertionError(f"{cfg.job_id}: invalid aggregation weights")
            if (row.get("communication_total_bytes") !=
                    row.get("update_transmitted_bytes", 0)+row.get("model_broadcast_bytes", 0)):
                raise AssertionError(f"{cfg.job_id}: communication accounting is inconsistent")
            if any(int(version) != row.get("model_version")
                   for version in row.get("client_training_versions", {}).values()):
                raise AssertionError(f"{cfg.job_id}: selected client missed post-aggregation sync")
        final_test = json.loads((run/"final_test_metrics.json").read_text())
        finite_tree(final_test, f"{cfg.job_id}.final_test")
        if final_test.get("evaluation_split") != "test":
            raise AssertionError(f"{cfg.job_id}: final benchmark split is not test")
        summary = json.loads((run/"summary.json").read_text())
        if summary.get("require_cuda") is not True or not str(
                summary.get("model_execution_device", "")).startswith("cuda"):
            raise AssertionError(f"{cfg.job_id}: CUDA requirement is not recorded")
        model_file = torch.load(run/"model_final.pt", map_location="cpu", weights_only=False)
        if model_file.get("configuration") != cfg.to_dict():
            raise AssertionError(f"{cfg.job_id}: final model configuration mismatch")
        state = model_file["model_state"]
        if any(not torch.isfinite(t).all() for t in state.values() if t.is_floating_point()):
            raise AssertionError(f"{cfg.job_id}: final model contains nonfinite tensors")
        model = create_model(cfg.model, 10,
            task_class_sets=task_class_sets_from_state_dict(state)).to("cuda")
        model.load_state_dict(state)
        model.eval()
        with torch.inference_mode():
            output = model(torch.zeros(1, 3, 32, 32, device="cuda"))
        if output.shape != (1, 10) or not torch.isfinite(output).all():
            raise AssertionError(f"{cfg.job_id}: reloaded CUDA inference failed")
        checked += 1
    print(f"Verified {checked}/{len(jobs)} pilot jobs, all CUDA, finite, complete and reloadable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
