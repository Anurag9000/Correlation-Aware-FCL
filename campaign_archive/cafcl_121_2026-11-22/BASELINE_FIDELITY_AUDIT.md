# Baseline fidelity audit

**Date:** 2026-11-22 (IST)  
**Purpose:** Verify that independent A–G schedules and runner configs use the
preserved disjoint Task-IL full-participation V25 recipe as their reference.

## Evidence checked on MtechDEServer

- Frozen reference run:
  `/home/anurag24092/Projects/Correlation-Aware-FCL-six-runs-20261121/runs/V25_TASKIL_DISJOINT10_FULL_REPLAY_FEDAVG_SEED0/design.json`
- Reference launcher:
  `/home/anurag24092/Projects/Correlation-Aware-FCL-six-runs-20261121/taskil_src/run_one_six.py`
- Reference schedule implementation:
  `/home/anurag24092/Projects/Correlation-Aware-FCL-six-runs-20261121/taskil_src/src/cafcl/streaming.py`
- Ablation config builder and schedule compiler:
  `Correlation-Aware-FCL-independent-ablations-20261122/compiler/`.

Reference code SHA-256 values recorded during the audit:

| File | SHA-256 |
|---|---|
| `taskil_src/run_one_six.py` | `11287ccafdaf8a1a9865c3e69619701f134c0e75af3e3c861e8196e2e15fe832` |
| `taskil_src/src/cafcl/streaming.py` | `52945267db6df4b7552addcb33d1ac4d802945feabc9026a5779e43f501cdc15` |
| `taskil_src/src/cafcl/config.py` | `dd6d90b03cf01fe6a566e70e266fed19bb9a32d33d5f4c331715d9d9b1f83933` |

The reference launcher resolves its `cnn` selector to
`resnet18_cifar32`. Its schedule generator produces 50 two-class tasks over 10
learning syncs for five clients, in disjoint class order 0–99, followed by a
communication-only flush. The generated schedule exactly matches the expected
five-clients × ten-syncs allocation. The archived `results_1` directory holds
sync diagrams but no original config, metrics manifest, or checkpoint, so the
preserved runner and its exact run design are the available recipe evidence.

## Frozen recipe values matched

| Mechanism | Value in reference and ablation builder |
|---|---|
| Model | `resnet18_cifar32` |
| Clients / curriculum | 5 / disjoint Task-IL |
| Task width / local task count | 2 classes / 1 task per sync |
| Local training | 10 epochs per task |
| Replay | ER, 1:1 mixed replay, per-class memory 50, reservoir fraction 1/9 |
| Server | sample-weighted FedAvg |
| Local optimizer | SGD, lr 0.03, momentum 0.9, weight decay 0.0005 |
| Momentum state | persistent |
| Augmentation | CIFAR crop and flip |
| Participation | full outside C's stated outage factor |
| Task synchronization | after complete local task; no per-batch sync |
| Final stage | communication-only flush |
| Seed | 0 |

## Config comparison and intentional differences

An exhaustive field comparison between the archived V25 full-run config and a
config reconstructed using the current ablation runner found the recipe fields
above unchanged. The only differences before adding run-specific identifiers
were empty Python list versus tuple serialization and the new explicit schedule
override (`None` in the old run, a concrete schedule in the new runner).

The A–G worker also intentionally sets a unique run ID and family label,
`resume=False` so each arm starts fresh, and `evaluate_final_only=False` to
record per-sync metrics. Evaluation cadence does not alter optimizer steps or
training examples. Each factor's documented schedule is the only change to
training assignment; the selected clients, task horizon, replay arrivals, and
head registry are emitted in its run record. For C, the outage participation
pattern is the named factor. A's client-count arms and F/G/D/E's curricula
necessarily alter the number or assignment of task events as described in the
goal document.

Task-IL scoring also matches the agreed inference rule: for each test example,
the evaluator supplies every task ID whose recorded class set contains its
true class. For duplicate classes, per-head probabilities are averaged and
then renormalized over the union. This rule is implemented in the preserved
`evaluation.py` and `models.py` code and is not a new A–G factor.

## Runtime status boundary

This audit establishes configuration/schedule fidelity, not accuracy. It does
not claim that any of the 29 A–G jobs has trained. Their controller remains
sequential and waits for its GPU admission gate; see `plans/campaign_status.json`
and `SUITE_LEDGER.md` for the latest timestamped queue snapshot.
