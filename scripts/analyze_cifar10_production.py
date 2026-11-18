#!/usr/bin/env python3
"""Export auditable tables and SVG figures from verified CIFAR-10 runs."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def flatten_numbers(value, prefix=""):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_numbers(child, f"{prefix}/{key}" if prefix else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_numbers(child, f"{prefix}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        yield prefix, float(value)


def svg_line_chart(path: Path, job_id: str, rows: list[dict]) -> None:
    width, height, left, right, top, bottom = 960, 520, 72, 28, 46, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    colors = {"accuracy": "#111827", "old_accuracy": "#2563eb",
              "new_accuracy": "#dc2626", "integrated_forgetting": "#7c3aed"}
    labels = {"accuracy": "All observed", "old_accuracy": "Old classes",
              "new_accuracy": "New classes", "integrated_forgetting": "Integrated forgetting"}
    count = max(1, len(rows)-1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="27" font-family="sans-serif" font-size="18">{html.escape(job_id)} — validation by round</text>']
    for tick in range(6):
        value = tick/5
        y = top + (1-value)*plot_h
        parts.append(f'<path d="M {left} {y:.1f} H {width-right}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for series, color in colors.items():
        points = []
        for index, row in enumerate(rows):
            value = row.get(series)
            if value is None or not math.isfinite(float(value)):
                continue
            x = left + index/count*plot_w
            y = top + (1-float(value))*plot_h
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>')
    parts.append(f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#374151"/>')
    parts.append(f'<text x="{left+plot_w/2:.1f}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="12">Communication round</text>')
    for index, (series, color) in enumerate(colors.items()):
        x = left + index*205
        parts.append(f'<path d="M {x} {height-42} h 22" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{x+28}" y="{height-38}" font-family="sans-serif" font-size="11">{labels[series]}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts)+"\n", encoding="utf-8")


def svg_heatmap(path: Path, title: str, matrix: list[list[float]], labels: list[str]) -> None:
    width, height, cell, left, top = 560, 520, 34, 118, 86
    maximum = max((float(v) for row in matrix for v in row), default=0.0)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="30" font-family="sans-serif" font-size="16">{html.escape(title)}</text>']
    for row, label in enumerate(labels):
        y = top+row*cell
        parts.append(f'<text x="{left-12}" y="{y+22}" text-anchor="end" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
        for col, value in enumerate(matrix[row]):
            ratio = float(value)/maximum if maximum else 0.0
            red, green, blue = 248-int(180*ratio), 250-int(105*ratio), 252-int(35*ratio)
            x = left+col*cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="rgb({red},{green},{blue})" stroke="white"/>')
            parts.append(f'<text x="{x+cell/2}" y="{y+21}" text-anchor="middle" font-family="sans-serif" font-size="9">{float(value):.0f}</text>')
    for col, label in enumerate(labels):
        x = left+col*cell+cell/2
        parts.append(f'<text x="{x}" y="{top-8}" text-anchor="middle" font-family="sans-serif" font-size="10">{html.escape(label)}</text>')
    parts.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts)+"\n", encoding="utf-8")


def svg_causal_curve(path: Path, source_id: str, branch_id: str,
                     source_rows: list[dict], branch_rows: list[dict]) -> None:
    width, height, left, right, top, bottom = 960, 520, 72, 28, 46, 70
    plot_w, plot_h = width-left-right, height-top-bottom
    count=max(1,min(len(source_rows),len(branch_rows))-1)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="{left}" y="27" font-family="sans-serif" font-size="16">Paired causal trajectory: {html.escape(source_id)} vs {html.escape(branch_id)}</text>']
    for tick in range(6):
        value=tick/5; y=top+(1-value)*plot_h
        parts.append(f'<path d="M {left} {y:.1f} H {width-right}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for rows,color in ((source_rows,"#2563eb"),(branch_rows,"#dc2626")):
        pts=[]
        for i,row in enumerate(rows[:count+1]):
            value=row.get("accuracy")
            if value is None: continue
            x=left+i/count*plot_w; y=top+(1-float(value))*plot_h
            pts.append(f"{x:.1f},{y:.1f}")
        if pts: parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>')
    parts.extend([f'<path d="M {left} {top} V {height-bottom} H {width-right}" fill="none" stroke="#374151"/>',
                  f'<path d="M {left+20} {height-38} h 22" stroke="#2563eb" stroke-width="3"/>',
                  f'<text x="{left+48}" y="{height-34}" font-family="sans-serif" font-size="11">Source</text>',
                  f'<path d="M {left+140} {height-38} h 22" stroke="#dc2626" stroke-width="3"/>',
                  f'<text x="{left+168}" y="{height-34}" font-family="sans-serif" font-size="11">Intervention</text>', '</svg>'])
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("\n".join(parts)+"\n",encoding="utf-8")


def mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/cifar10")
    parser.add_argument("--plan", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Export only completed jobs without requiring the 423-job receipt")
    args = parser.parse_args()
    root = Path(args.results_root)
    plan_path = Path(args.plan) if args.plan else root/"job_plan.json"
    out = Path(args.output_dir) if args.output_dir else root/"analysis"
    plan = json.loads(plan_path.read_text())
    jobs = plan.get("jobs", [])
    if plan.get("dataset") != "cifar10" or not jobs:
        raise SystemExit("Analysis requires a nonempty CIFAR-10 plan")
    if not args.allow_incomplete and len(jobs) != 423:
        raise SystemExit("Complete production analysis requires the canonical 423-job plan")
    if not args.allow_incomplete:
        receipt_path = root/"suite_verification.json"
        if not receipt_path.is_file():
            raise SystemExit(f"Missing production verification receipt: {receipt_path}")
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get("planned_jobs") != 423 or receipt.get("verified_jobs") != 423 or
                receipt.get("failed_or_invalid_jobs") != 0):
            raise SystemExit("Production receipt does not prove all 423 jobs valid")

    registry, rounds_out, classes_out = [], [], []
    confusion_out, participation_out, aggregation_out, diagnostics_out, gradient_pairs_out = [], [], [], [], []
    checkpoint_out, causal_out, causal_trajectory_out, causal_class_out, resource_out = [], [], [], [], []
    grouped = defaultdict(list)
    cache = {}
    errors = []
    for cfg in jobs:
        job_id = cfg["job_id"]
        run = root/"runs"/job_id
        try:
            saved = json.loads((run/"config.json").read_text())
            status = json.loads((run/"status.json").read_text())
            metrics = json.loads((run/"metrics.json").read_text())
            summary = json.loads((run/"summary.json").read_text())
            final_test = json.loads((run/"final_test_metrics.json").read_text())
            if saved != cfg or status.get("status") != "completed":
                raise ValueError("configuration mismatch or incomplete status")
            if not metrics:
                raise ValueError("no round metrics")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"job_id": job_id, "error": str(exc)})
            if not args.allow_incomplete:
                raise SystemExit(f"Cannot analyze {job_id}: {exc}")
            registry.append({"job_id": job_id, "block_id": cfg.get("block_id"),
                             "status": "missing_or_invalid", "error": str(exc),
                             "artifact_path": str(run)})
            continue

        cache[job_id] = (cfg, metrics)
        tail = metrics[-1]
        total_seconds = sum(float(row.get("seconds", 0) or 0) for row in metrics)
        tx_bytes = sum(int(row.get("update_transmitted_bytes", 0) or 0) for row in metrics)
        broadcast_bytes = sum(int(row.get("model_broadcast_bytes", 0) or 0) for row in metrics)
        files = list(run.glob("model_*.pt"))+list(run.glob("checkpoint_round_*.pt"))
        registry.append({
            "job_id": job_id, "block_id": cfg.get("block_id"), "method": cfg.get("method"),
            "aggregator": cfg.get("aggregator"), "selector": cfg.get("selector"),
            "participation": cfg.get("participation"), "scenario": cfg.get("scenario"),
            "relation": cfg.get("relation"), "model": cfg.get("model"),
            "lr": cfg.get("lr"), "momentum": cfg.get("momentum"),
            "batch_size": cfg.get("batch_size"), "train_fraction": cfg.get("train_fraction"),
            "rounds": len(metrics), "expected_rounds": cfg.get("stages", 0)*cfg.get("rounds_per_stage", 0),
            "status": status.get("status"), "device": summary.get("model_execution_device"),
            "final_validation_accuracy": tail.get("accuracy"),
            "final_validation_old_accuracy": tail.get("old_accuracy"),
            "final_validation_new_accuracy": tail.get("new_accuracy"),
            "final_test_accuracy": final_test.get("accuracy"),
            "training_seconds": total_seconds, "update_bytes": tx_bytes,
            "broadcast_bytes": broadcast_bytes, "total_communication_bytes": tx_bytes+broadcast_bytes,
            "model_checkpoint_files": len(files), "model_checkpoint_bytes": sum(p.stat().st_size for p in files),
            "artifact_path": str(run), "error": ""})

        for row in metrics:
            rounds_out.append({
                "job_id": job_id, "block_id": cfg.get("block_id"), "round": row.get("round"),
                "stage": row.get("stage"), "method": cfg.get("method"),
                "aggregator": cfg.get("aggregator"), "selector": cfg.get("selector"),
                "participation_configured": cfg.get("participation"),
                "accuracy": row.get("accuracy"), "macro_accuracy": row.get("macro_accuracy"),
                "old_accuracy": row.get("old_accuracy"), "new_accuracy": row.get("new_accuracy"),
                "integrated_forgetting": row.get("integrated_forgetting"),
                "all_observed_peak_drop": row.get("all_observed_peak_drop"),
                "never_integrated_accuracy": row.get("never_integrated_accuracy"),
                "observed_classes": encoded(row.get("observed_classes", [])),
                "integrated_classes": encoded(row.get("globally_integrated_classes", [])),
                "never_integrated_classes": encoded(row.get("never_integrated_classes", [])),
                "selected_clients": encoded(row.get("selected_clients", [])),
                "uploaded_count": row.get("uploaded_count"), "seconds": row.get("seconds"),
                "peak_cuda_allocated_bytes": row.get("peak_cuda_allocated_bytes"),
                "communication_bytes": row.get("communication_total_bytes")})
            for detail in row.get("local_details") or []:
                for label, samples in (detail.get("class_counts") or {}).items():
                    participation_out.append({
                        "job_id": job_id, "round": row.get("round"),
                        "client_id": detail.get("client_id"), "class_id": label,
                        "local_training_samples": samples, "examples": detail.get("examples"),
                        "method_used": detail.get("method_used"), "steps": detail.get("steps"),
                        "memory_samples": detail.get("memory_samples"),
                        "qp_calls": detail.get("qp_calls"),
                        "constraint_violations": detail.get("constraint_violations"),
                        "projection_norm_last": detail.get("projection_norm_last")})
            for client, weight in (row.get("aggregation_weights") or {}).items():
                aggregation_out.append({"job_id": job_id, "round": row.get("round"),
                                        "client_id": client, "aggregation_weight": weight,
                                        "aggregator": cfg.get("aggregator"),
                                        "update_bytes": row.get("update_transmitted_bytes"),
                                        "broadcast_bytes": row.get("model_broadcast_bytes")})
            diagnostics_out.append({
                "job_id": job_id, "round": row.get("round"),
                "gradient_relationships": encoded(row.get("gradient_relationships")),
                "old_gradient_vs_client_updates": encoded(row.get("old_gradient_vs_client_updates")),
                "client_update_cosines": encoded(row.get("client_update_cosines")),
                "subset_diagnostics": encoded(row.get("subset_diagnostics")),
                "representation_drift": encoded(row.get("representation_drift")),
                "causal_diagnostics": encoded(row.get("causal_diagnostics"))})
            for name in ("gradient_relationships", "client_update_cosines",
                         "old_gradient_vs_client_updates"):
                for pair, value in flatten_numbers(row.get(name)):
                    gradient_pairs_out.append({"job_id": job_id, "round": row.get("round"),
                                               "diagnostic": name, "pair_or_index": pair,
                                               "value": value})

        labels = [str(i) for i in range(10)]
        confusion = final_test.get("confusion_matrix", {})
        counts = confusion.get("counts", [])
        confusion_labels = confusion.get("labels", [])
        if counts and len(counts) == len(confusion_labels):
            for i, true_label in enumerate(confusion_labels):
                for j, predicted_label in enumerate(confusion_labels):
                    confusion_out.append({"job_id": job_id, "split": "official_test_final_model",
                                          "true_class": true_label, "predicted_class": predicted_label,
                                          "count": counts[i][j]})
        val_per_class = tail.get("per_class", {})
        final_integrated = set(tail.get("globally_integrated_classes", []))
        final_never = set(tail.get("never_integrated_classes", []))
        for label in set(val_per_class) | {str(c) for row in metrics for c in row.get("per_class", {})}:
            history = [(row.get("per_class", {}).get(label),
                        int(label) in set(row.get("globally_integrated_classes", [])))
                       for row in metrics if label in row.get("per_class", {})]
            learned_scores = [float(score) for score, integrated in history
                              if score is not None and integrated]
            final_score = val_per_class.get(label)
            peak = max(learned_scores) if learned_scores else None
            drop = (max(0.0, peak-float(final_score))
                    if peak is not None and final_score is not None and
                    int(label) in final_integrated and peak > 0.12 else None)
            classes_out.append({"job_id": job_id, "class_id": label,
                                "final_validation_accuracy": final_score,
                                "peak_accuracy_when_integrated": peak,
                                "integrated_accuracy_drop": drop,
                                "integrated_at_final": int(label) in final_integrated,
                                "never_integrated_at_final": int(label) in final_never,
                                "classification": ("never_integrated" if int(label) in final_never else
                                                   "acquired_then_declined" if drop is not None else
                                                   "integrated_or_not_yet_forgotten")})

        group_key = tuple(cfg.get(key) for key in
                          ("block_id", "method", "aggregator", "selector", "participation", "scenario", "relation"))
        grouped[group_key].append({"accuracy": tail.get("accuracy"),
                                   "old_accuracy": tail.get("old_accuracy"),
                                   "new_accuracy": tail.get("new_accuracy"),
                                   "integrated_forgetting": tail.get("integrated_forgetting"),
                                   "seconds": total_seconds,
                                   "communication_bytes": tx_bytes+broadcast_bytes})
        resource_out.append({"job_id": job_id, "device": summary.get("model_execution_device"),
                             "rounds": len(metrics), "total_training_seconds": total_seconds,
                             "mean_seconds_per_round": total_seconds/len(metrics),
                             "peak_cuda_allocated_bytes": max(
                                 (int(row.get("peak_cuda_allocated_bytes", 0) or 0) for row in metrics),
                                 default=0),
                             "update_bytes": tx_bytes, "broadcast_bytes": broadcast_bytes,
                             "total_communication_bytes": tx_bytes+broadcast_bytes,
                             "checkpoint_bytes": sum(p.stat().st_size for p in files)})
        svg_line_chart(out/"curves"/f"{job_id}.svg", job_id, metrics)

        # Local training sample incidence and simultaneous class co-presence
        # among participating clients, accumulated over communication rounds.
        class_client = [[0.0]*10 for _ in range(10)]
        class_holders = [[0.0]*10 for _ in range(10)]
        for row in metrics:
            holders = defaultdict(set)
            for detail in row.get("local_details") or []:
                client = int(detail.get("client_id", detail.get("client", -1)))
                for label, samples in (detail.get("class_counts") or {}).items():
                    label_id = int(label)
                    class_client[client][label_id] += float(samples)
                    if samples:
                        holders[label_id].add(client)
            for a in range(10):
                if holders[a]:
                    class_holders[a][a] += 1
                for b in range(10):
                    if holders[a] and holders[b] and holders[a].intersection(holders[b]):
                        class_holders[a][b] += 1
        svg_heatmap(out/"heatmaps"/f"{job_id}_client_class_samples.svg",
                    f"{job_id}: local training samples by client/class",
                    class_client, [str(i) for i in range(10)])
        svg_heatmap(out/"heatmaps"/f"{job_id}_simultaneous_class_overlap.svg",
                    f"{job_id}: rounds with a shared participating client",
                    class_holders, [str(i) for i in range(10)])

        if cfg.get("fork_source_job"):
            causal_out.append({"source_job_id": cfg["fork_source_job"], "branch_job_id": job_id,
                               "arm": cfg.get("force_target"), "target_client": cfg.get("target_client"),
                               "intervention_round": cfg.get("intervention_round"),
                               "source_final_validation_accuracy": None,
                               "branch_final_validation_accuracy": tail.get("accuracy"),
                               "branch_final_old_accuracy": tail.get("old_accuracy"),
                               "branch_final_new_accuracy": tail.get("new_accuracy"),
                               "paired_final_accuracy_delta": None})

        for checkpoint in files:
            checkpoint_out.append({"job_id": job_id, "artifact": checkpoint.name,
                                   "size_bytes": checkpoint.stat().st_size,
                                   "path": str(checkpoint)})

    if not args.allow_incomplete and (len(cache) != 423 or errors):
        raise SystemExit(f"Only {len(cache)}/423 jobs are readable; see {out/'analysis_errors.csv'}")

    # Resolve paired causal deltas after every source trajectory is loaded.
    for row in causal_out:
        source = cache.get(row["source_job_id"])
        branch = cache.get(row["branch_job_id"])
        if source and branch:
            source_final = source[1][-1].get("accuracy")
            branch_final = branch[1][-1].get("accuracy")
            row["source_final_validation_accuracy"] = source_final
            if source_final is not None and branch_final is not None:
                row["paired_final_accuracy_delta"] = float(branch_final)-float(source_final)
            cfg, source_metrics = source
            branch_cfg, branch_metrics = branch
            intervention = int(branch_cfg.get("intervention_round", -1))
            paired_rounds = min(len(source_metrics),len(branch_metrics))
            for index in range(paired_rounds):
                source_row, branch_row = source_metrics[index], branch_metrics[index]
                causal_trajectory_out.append({
                    "source_job_id": row["source_job_id"], "branch_job_id": row["branch_job_id"],
                    "round": index, "post_intervention": index >= intervention,
                    "source_accuracy": source_row.get("accuracy"),
                    "branch_accuracy": branch_row.get("accuracy"),
                    "accuracy_delta": (float(branch_row["accuracy"])-float(source_row["accuracy"])
                                       if source_row.get("accuracy") is not None and branch_row.get("accuracy") is not None else None),
                    "source_old_accuracy": source_row.get("old_accuracy"),
                    "branch_old_accuracy": branch_row.get("old_accuracy"),
                    "source_new_accuracy": source_row.get("new_accuracy"),
                    "branch_new_accuracy": branch_row.get("new_accuracy"),
                    "source_integrated_forgetting": source_row.get("integrated_forgetting"),
                    "branch_integrated_forgetting": branch_row.get("integrated_forgetting")})
                source_per_class = source_row.get("per_class", {})
                branch_per_class = branch_row.get("per_class", {})
                for label in sorted(set(source_per_class)|set(branch_per_class),key=int):
                    a,b=source_per_class.get(label),branch_per_class.get(label)
                    causal_class_out.append({"source_job_id":row["source_job_id"],
                        "branch_job_id":row["branch_job_id"],"round":index,
                        "class_id":label,"source_accuracy":a,"branch_accuracy":b,
                        "accuracy_delta":float(b)-float(a) if a is not None and b is not None else None,
                        "post_intervention":index>=intervention})
            svg_causal_curve(out/"causal_curves"/f"{row['branch_job_id']}.svg",
                             row["source_job_id"],row["branch_job_id"],source_metrics,branch_metrics)

    write_csv(out/"job_execution_registry.csv", list(registry[0]) if registry else ["job_id"], registry)
    write_csv(out/"per_round_metrics.csv", list(rounds_out[0]) if rounds_out else ["job_id"], rounds_out)
    write_csv(out/"per_class_forgetting.csv", list(classes_out[0]) if classes_out else ["job_id"], classes_out)
    write_csv(out/"final_confusion_matrix.csv", ["job_id", "split", "true_class", "predicted_class", "count"], confusion_out)
    write_csv(out/"client_class_training_samples.csv", list(participation_out[0]) if participation_out else ["job_id"], participation_out)
    write_csv(out/"aggregation_diagnostics.csv", list(aggregation_out[0]) if aggregation_out else ["job_id"], aggregation_out)
    write_csv(out/"gpu_runtime_communication.csv", list(resource_out[0]) if resource_out else ["job_id"], resource_out)
    write_csv(out/"gradient_and_causal_diagnostics.csv", list(diagnostics_out[0]) if diagnostics_out else ["job_id"], diagnostics_out)
    write_csv(out/"gradient_interference_pairs.csv", list(gradient_pairs_out[0]) if gradient_pairs_out else ["job_id", "round", "diagnostic", "pair_or_index", "value"], gradient_pairs_out)
    write_csv(out/"causal_comparisons.csv", list(causal_out[0]) if causal_out else ["source_job_id", "branch_job_id"], causal_out)
    write_csv(out/"causal_trajectory_deltas.csv", list(causal_trajectory_out[0]) if causal_trajectory_out else ["source_job_id", "branch_job_id", "round"], causal_trajectory_out)
    write_csv(out/"causal_per_class_deltas.csv", list(causal_class_out[0]) if causal_class_out else ["source_job_id", "branch_job_id", "round", "class_id"], causal_class_out)
    write_csv(out/"checkpoint_inventory.csv", list(checkpoint_out[0]) if checkpoint_out else ["job_id", "artifact", "size_bytes", "path"], checkpoint_out)
    write_csv(out/"analysis_errors.csv", ["job_id", "error"], errors)
    suite_status_path = root/"suite_status.json"
    suite_status = json.loads(suite_status_path.read_text()) if suite_status_path.is_file() else {}
    failure_rows = [{"job_id": item.get("job_id"), "error": item.get("error"),
                     "traceback": item.get("traceback", ""),
                     "fix_reference": "docs/STATUS.md or per-job recovery notes"}
                    for item in suite_status.get("failures", [])]
    write_csv(out/"failure_fix_log.csv", ["job_id", "error", "traceback", "fix_reference"], failure_rows)

    gem_rows=[]
    local_by_job=defaultdict(list)
    for row in participation_out:
        if row.get("method_used") in {"gem","agem"}:
            local_by_job[row["job_id"]].append(row)
    for job_id,(cfg,_) in cache.items():
        local=local_by_job.get(job_id,[])
        if local:
            gem_rows.append({"job_id":job_id,"method":cfg.get("method"),
                "local_update_records":len(local),
                "total_qp_calls":sum(int(row.get("qp_calls") or 0) for row in local),
                "total_constraint_violations":sum(int(row.get("constraint_violations") or 0) for row in local),
                "total_local_steps":sum(int(row.get("steps") or 0) for row in local),
                "mean_memory_samples":mean(local,"memory_samples"),
                "mean_last_projection_norm":mean(local,"projection_norm_last")})
    write_csv(out/"gem_constraint_summary.csv", list(gem_rows[0]) if gem_rows else ["job_id","method"], gem_rows)
    gradient_groups=defaultdict(list)
    for row in gradient_pairs_out:
        gradient_groups[(row["job_id"],row["diagnostic"])].append(float(row["value"]))
    gradient_summary=[]
    for (job_id,diagnostic),values in sorted(gradient_groups.items()):
        gradient_summary.append({"job_id":job_id,"diagnostic":diagnostic,
            "pair_or_value_count":len(values),"mean":statistics.fmean(values),
            "minimum":min(values),"maximum":max(values),
            "negative_count":sum(value<0 for value in values),
            "negative_fraction":sum(value<0 for value in values)/len(values)})
    write_csv(out/"gradient_interference_summary.csv", list(gradient_summary[0]) if gradient_summary else ["job_id","diagnostic"], gradient_summary)

    family_rows = []
    for key, rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        fields = ("block_id", "method", "aggregator", "selector", "participation", "scenario", "relation")
        row = dict(zip(fields, key))
        row.update({"job_count": len(rows), "mean_final_accuracy": mean(rows, "accuracy"),
                    "mean_final_old_accuracy": mean(rows, "old_accuracy"),
                    "mean_final_new_accuracy": mean(rows, "new_accuracy"),
                    "mean_integrated_forgetting": mean(rows, "integrated_forgetting"),
                    "total_training_seconds": sum(float(x["seconds"]) for x in rows),
                    "total_communication_bytes": sum(int(x["communication_bytes"]) for x in rows)})
        family_rows.append(row)
    write_csv(out/"experiment_family_summary.csv", list(family_rows[0]) if family_rows else ["block_id"], family_rows)

    valid = [row for row in registry if row.get("status") == "completed"]
    plot = [row for row in valid if row.get("final_validation_old_accuracy") is not None and row.get("final_validation_new_accuracy") is not None]
    scatter = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="650"><rect width="100%" height="100%" fill="white"/>',
              '<text x="80" y="30" font-family="sans-serif" font-size="18">Validation stability–plasticity across jobs</text>']
    for tick in range(6):
        value=tick/5; x=80+value*760; y=570-value*500
        scatter.extend([f'<path d="M {x} 70 V 570" stroke="#eeeeee"/>', f'<path d="M 80 {y} H 840" stroke="#eeeeee"/>'])
        scatter.append(f'<text x="{x}" y="590" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.1f}</text>')
    palette=defaultdict(lambda:"#2563eb", {"sgd":"#2563eb","gem":"#dc2626","agem":"#16a34a","replay":"#7c3aed"})
    for row in plot:
        x=80+float(row["final_validation_new_accuracy"])*760
        y=570-float(row["final_validation_old_accuracy"])*500
        color=palette[row.get("method")]
        scatter.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" fill-opacity="0.55"><title>{html.escape(row["job_id"])}</title></circle>')
    scatter.extend(['<path d="M 80 70 V 570 H 840" fill="none" stroke="#374151"/>',
                    '<text x="460" y="630" text-anchor="middle" font-family="sans-serif" font-size="12">New-class validation accuracy</text>',
                    '<text x="18" y="320" transform="rotate(-90 18 320)" text-anchor="middle" font-family="sans-serif" font-size="12">Old-class validation accuracy</text>', '</svg>'])
    out.mkdir(parents=True, exist_ok=True)
    (out/"stability_plasticity.svg").write_text("\n".join(scatter)+"\n", encoding="utf-8")
    (out/"README.md").write_text(
        "# CIFAR-10 result analysis\n\n"
        f"Completed jobs analyzed: {len(cache)} of {len(jobs)} plan entries.\n\n"
        "The per-round and per-class tables use validation metrics. Final confusion matrices use the official test split only after training; they were not used for selection. "
        "The SVG curves are per-job. Heatmaps count local training sample incidence and rounds in which participating clients jointly held each class pair. "
        "Family means are descriptive across configured jobs, not independent replications or significance tests.\n\n"
        "Run with: `python scripts/analyze_cifar10_production.py --results-root results/cifar10`.\n",
        encoding="utf-8")
    print(f"Wrote analysis for {len(cache)} completed jobs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
