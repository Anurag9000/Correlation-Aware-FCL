#!/usr/bin/env python3
"""Audit all registered CIFAR-10 production jobs and reload final CUDA models."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FORK_ALLOWED_CONFIG_FIELDS = {
    "job_id", "force_target", "target_client", "intervention_round",
    "causal_diagnostics", "gradient_diagnostics", "checkpoint_every",
    "fork_source_job", "fork_checkpoint_round",
}


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
    parser.add_argument("--plan", default="configs/plans/cifar10_jobs.json")
    parser.add_argument("--output-dir", default="results/cifar10")
    args = parser.parse_args()
    import torch
    from cafcl.config import RunConfig
    from cafcl.models import create_model
    from cafcl.planner import all_supported_suite

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to audit final production models")
    manifest = json.loads(Path(args.plan).read_text())
    raw_jobs = manifest.get("jobs", [])
    jobs = [RunConfig(**raw) for raw in raw_jobs]
    if manifest.get("dataset") != "cifar10" or manifest.get("batch_size") != 64:
        raise AssertionError("Production manifest dataset or batch-size header is invalid")
    expected = json.loads(json.dumps([cfg.to_dict() for cfg in all_supported_suite("cifar10", 0)]))
    if raw_jobs != expected:
        raise AssertionError("Committed CIFAR-10 manifest differs from the source planner")
    if len(jobs) != 423:
        raise AssertionError(f"Expected 423 registered CIFAR-10 jobs, got {len(jobs)}")
    ids = [cfg.job_id for cfg in jobs]
    if len(ids) != len(set(ids)):
        raise AssertionError("Production manifest contains duplicate job IDs")
    if any(cfg.dataset != "cifar10" or cfg.batch_size != 64 or not cfg.require_cuda for cfg in jobs):
        raise AssertionError("A production job violates CIFAR-10/batch-64/CUDA requirements")

    root = Path(args.output_dir)
    errors = []
    inspected = {}
    for cfg in jobs:
        run = root / "runs" / cfg.job_id
        required = ["config.json", "status.json", "metrics.json", "summary.json",
                    "model_final.pt", "final_test_metrics.json", "data_partition.json",
                    "scenario.pt"]
        source_run = root / "runs" / cfg.fork_source_job if cfg.fork_source_job else None
        for stage in range(cfg.stages):
            end_round = (stage + 1) * cfg.rounds_per_stage
            artifact_run = source_run if source_run is not None and end_round <= cfg.fork_checkpoint_round else run
            artifact = f"model_stage_{stage+1:02d}.pt"
            if artifact_run is None or artifact_run == run:
                required.append(artifact)
            elif not (artifact_run / artifact).is_file():
                errors.append(f"{cfg.job_id}: source missing {artifact}")
        if cfg.checkpoint_every:
            for round_number in range(1, cfg.stages * cfg.rounds_per_stage + 1):
                artifact_run = source_run if source_run is not None and round_number <= cfg.fork_checkpoint_round else run
                artifact = f"checkpoint_round_{round_number:04d}.pt"
                if artifact_run is None or artifact_run == run:
                    required.append(artifact)
                elif not (artifact_run / artifact).is_file():
                    errors.append(f"{cfg.job_id}: source missing {artifact}")
        missing = [name for name in required if not (run / name).is_file()]
        if missing:
            errors.append(f"{cfg.job_id}: missing artifacts {missing}")
            continue
        try:
            saved_config = json.loads((run / "config.json").read_text())
            if saved_config != cfg.to_dict():
                raise AssertionError("saved config differs from manifest")
            status = json.loads((run / "status.json").read_text())
            if status.get("status") != "completed":
                raise AssertionError(f"status is {status.get('status')}")
            partition = json.loads((run / "data_partition.json").read_text())
            if (partition.get("train_samples") != 45000 or
                    partition.get("validation_samples") != 5000 or
                    partition.get("official_test_samples") != 10000 or
                    partition.get("train_class_counts") != {str(c): 4500 for c in range(10)} or
                    partition.get("validation_class_counts") != {str(c): 500 for c in range(10)} or
                    partition.get("official_test_class_counts") != {str(c): 1000 for c in range(10)} or
                    not partition.get("train_validation_source_indices_disjoint") or
                    not partition.get("official_test_is_original_cifar_test_split")):
                raise AssertionError("train/validation/test partition is invalid")
            metrics = json.loads((run / "metrics.json").read_text())
            rounds = cfg.stages * cfg.rounds_per_stage
            if len(metrics) != rounds or [r.get("round") for r in metrics] != list(range(rounds)):
                raise AssertionError("round count or order differs from configuration")
            for row in metrics:
                finite_tree(row, f"{cfg.job_id}.round{row.get('round')}")
                if row.get("evaluation_split") != "validation":
                    raise AssertionError(f"round {row.get('round')} used non-validation feedback")
                if not str(row.get("model_execution_device", "")).startswith("cuda"):
                    raise AssertionError(f"round {row.get('round')} did not execute on CUDA")
                if row.get("global_evaluation_includes_inactive") is not True:
                    raise AssertionError(f"round {row.get('round')} omitted inactive holders")
                observed = set(row.get("observed_classes", []))
                old = set(row.get("old_classes", []))
                new = set(row.get("new_classes", []))
                integrated = set(row.get("globally_integrated_classes", []))
                never = set(row.get("never_integrated_classes", []))
                if old & new or old | new != observed:
                    raise AssertionError(f"round {row.get('round')} old/new classes do not partition observed classes")
                if not integrated <= observed or integrated & never or integrated | never != observed:
                    raise AssertionError(f"round {row.get('round')} integrated/never classes are inconsistent")
                expected_per_class = {str(label) for label in observed}
                if set(row.get("per_class", {})) != expected_per_class:
                    raise AssertionError(f"round {row.get('round')} global evaluation omitted an observed class")
                confusion = row.get("confusion_matrix", {})
                labels = confusion.get("labels", [])
                counts = confusion.get("counts", [])
                if (labels != list(range(10)) or len(counts) != 10 or
                        any(len(values) != 10 for values in counts) or
                        sum(sum(int(value) for value in values) for values in counts) !=
                        row.get("evaluation_samples")):
                    raise AssertionError(f"round {row.get('round')} confusion counts are incomplete")
                if row.get("selected_count") != len(row.get("selected_clients", [])):
                    raise AssertionError(f"round {row.get('round')} selected-client count mismatch")
                if row.get("uploaded_count", 0) > row.get("selected_count", 0):
                    raise AssertionError(f"round {row.get('round')} uploaded more clients than selected")
                weights = row.get("aggregation_weights", {})
                if len(weights) != row.get("uploaded_count"):
                    raise AssertionError(f"round {row.get('round')} aggregation weights do not match uploads")
                if weights and (any(not math.isfinite(v) or v < 0 for v in weights.values()) or
                                not math.isclose(sum(weights.values()), 1.0, rel_tol=1e-7, abs_tol=1e-7)):
                    raise AssertionError(f"round {row.get('round')} aggregation weights are invalid")
                if row.get("communication_total_bytes") != row.get("update_transmitted_bytes", 0) + row.get("model_broadcast_bytes", 0):
                    raise AssertionError(f"round {row.get('round')} communication accounting mismatch")
                if any(int(v) != row.get("model_version")
                       for v in row.get("client_training_versions", {}).values()):
                    raise AssertionError(f"round {row.get('round')} selected client sync version mismatch")
                if set(row.get("client_training_versions", {})) != {
                        str(client) for client in row.get("selected_clients", [])}:
                    raise AssertionError(f"round {row.get('round')} is missing selected-client sync evidence")
            summary = json.loads((run / "summary.json").read_text())
            if summary.get("require_cuda") is not True or not str(
                    summary.get("model_execution_device", "")).startswith("cuda"):
                raise AssertionError("summary does not confirm CUDA execution")
            final_test = json.loads((run / "final_test_metrics.json").read_text())
            finite_tree(final_test, f"{cfg.job_id}.final_test")
            if final_test.get("evaluation_split") != "test":
                raise AssertionError("final model benchmark evaluation is not the official test split")
            final_confusion = final_test.get("confusion_matrix", {})
            test_labels = final_confusion.get("labels", [])
            test_counts = final_confusion.get("counts", [])
            if (test_labels != list(range(10)) or len(test_counts) != 10 or
                    any(len(values) != 10 for values in test_counts) or
                    sum(sum(int(value) for value in values) for values in test_counts) !=
                    final_test.get("evaluation_samples")):
                raise AssertionError("final model confusion counts are incomplete")
            if cfg.fork_source_job:
                source = root / "runs" / cfg.fork_source_job
                source_config = json.loads((source / "config.json").read_text())
                checkpoint = source / f"checkpoint_round_{cfg.fork_checkpoint_round:04d}.pt"
                if not checkpoint.is_file():
                    raise AssertionError(f"missing causal source checkpoint {checkpoint.name}")
                if ({k: v for k, v in source_config.items() if k not in FORK_ALLOWED_CONFIG_FIELDS} !=
                        {k: v for k, v in cfg.to_dict().items() if k not in FORK_ALLOWED_CONFIG_FIELDS}):
                    raise AssertionError("causal branch configuration is confounded")
                source_metrics = json.loads((source / "metrics.json").read_text())
                if source_metrics[:cfg.fork_checkpoint_round] != metrics[:cfg.fork_checkpoint_round]:
                    raise AssertionError("pre-intervention branch metrics differ from source")
            model_file = torch.load(run / "model_final.pt", map_location="cpu", weights_only=False)
            if model_file.get("configuration") != cfg.to_dict():
                raise AssertionError("final model configuration mismatch")
            state = model_file["model_state"]
            if any(not torch.isfinite(t).all() for t in state.values() if t.is_floating_point()):
                raise AssertionError("final model contains a nonfinite tensor")
            model = create_model(cfg.model, 10).to("cuda")
            model.load_state_dict(state)
            model.eval()
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 32, 32, device="cuda"))
            if output.shape != (1, 10) or not torch.isfinite(output).all():
                raise AssertionError("reloaded final model CUDA inference failed")
            del model, model_file, state, output
            torch.cuda.empty_cache()
            inspected[cfg.job_id] = {"status": "completed", "rounds": rounds,
                                     "batch_size": cfg.batch_size,
                                     "model_execution_device": summary["model_execution_device"]}
        except Exception as exc:
            errors.append(f"{cfg.job_id}: {type(exc).__name__}: {exc}")

    receipt = {
        "dataset": "cifar10", "planned_jobs": len(jobs),
        "verified_jobs": len(inspected), "failed_or_invalid_jobs": len(errors),
        "batch_size_64_jobs": sum(cfg.batch_size == 64 for cfg in jobs),
        "cuda_required_jobs": sum(cfg.require_cuda for cfg in jobs),
        "jobs": inspected, "errors": errors,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "suite_verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: receipt[k] for k in (
        "planned_jobs", "verified_jobs", "failed_or_invalid_jobs",
        "batch_size_64_jobs", "cuda_required_jobs")}, indent=2))
    if errors or len(inspected) != len(jobs):
        print("See suite_verification.json for per-job failures")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
