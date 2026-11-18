#!/usr/bin/env python3
"""Create a paired CIFAR-10 validation-only baseline hyperparameter sweep."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from cafcl.planner import all_supported_suite

CIFAR10_CHANCE_ACCURACY = 0.1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/cifar10_tuning_v1/job_plan.json")
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--rounds-per-stage", type=int, default=8)
    parser.add_argument("--learning-rates", type=float, nargs="+",
                        default=[0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--momentum", type=float, nargs="+", default=[0.0, 0.9])
    args = parser.parse_args()

    baseline = next((cfg for cfg in all_supported_suite("cifar10", 0)
                     if cfg.job_id == "A01_disjoint_sgd_full"), None)
    if baseline is None:
        raise RuntimeError("The registered full-participation CIFAR-10 SGD baseline is missing")
    jobs = []
    for momentum in args.momentum:
        for lr in args.learning_rates:
            lr_tag = f"{lr:.4g}".replace(".", "p")
            momentum_tag = f"{momentum:.3g}".replace(".", "p")
            job_id = f"TUNE_C10_SGD_lr{lr_tag}_mom{momentum_tag}"
            cfg = replace(baseline, job_id=job_id, lr=lr, momentum=momentum,
                          train_fraction=args.train_fraction,
                          rounds_per_stage=args.rounds_per_stage,
                          require_cuda=True)
            cfg.validate()
            jobs.append(cfg.to_dict())
    ids = [job["job_id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Tuning candidate IDs are not unique")
    plan = {
        "dataset": "cifar10",
        "suite": "validation_hyperparameter_tuning",
        "seed": 0,
        "batch_size": 512,
        "validation_fraction": baseline.validation_fraction,
        "training_fraction_of_eligible_data": args.train_fraction,
        "rounds_per_stage": args.rounds_per_stage,
        "candidate_parameters": {
            "local_sgd_learning_rate": args.learning_rates,
            "local_sgd_momentum": args.momentum,
        },
        "selection_protocol": {
            "feedback_split": "validation",
            "primary_metric": "harmonic_mean_of_mean_old_and_new_validation_accuracy_over_last_stage",
            "learning_gate": {
                "old_and_new_validation_accuracy_must_exceed": CIFAR10_CHANCE_ACCURACY,
                "rationale": "10-class class-incremental chance accuracy",
            },
            "supplementary_metrics": ["all_observed_validation_macro_accuracy",
                                      "integrated_class_coverage",
                                      "integrated_forgetting"],
            "reported_but_not_used_for_selection": ["training_seconds"],
            "runtime_used_for_selection": False,
            "paired_controls": "same seed, partition, model, class schedule, participation, aggregation and compute schedule",
            "official_test_used_for_selection": False,
        },
        "jobs": jobs,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Wrote {len(jobs)} paired CUDA tuning candidates to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
