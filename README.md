# Correlation-Aware FCL — CIFAR-100 continuous local-task protocol (v4, DeiT-Tiny/16)

**Active research dataset: CIFAR-100 (100 classes).** The old CIFAR-10 and CIFAR-100/20 **job suites, plans, launchers and documentation are retired and removed**. The CIFAR-10 dataset loader and some synthetic compatibility tests may remain as reusable components; **they are not active experiments**. Earlier stage-based CIFAR-100 jobs are also retired, because they do not implement the specified client task stream.

The current active finite-grid registry contains **53,805 independent planned CIFAR-100 job configurations** across 3 seeds (0, 1, 2); **5,346 form the matched core**, covering all 11 local methods × all 9 server update methods × all six overlap patterns × three participation levels × three seeds. **These are registered jobs, not trained results.** All configured effective training batch sizes are **512** (physical ViT microbatch 16).

## Mandatory research backbone

**Every active CIFAR-100 experiment uses one published ImageNet-1K pretrained DeiT-Tiny/16 (224×224) from `timm==1.0.31`, with a new 100-class canonical head.** The previous CNN/MLP/linear options are removed from production validation; tiny linear/CNN models are retained solely as synthetic offline test doubles. Model initialization fails if timm or pretrained weights are unavailable. Dataset preprocessing uses bicubic resize and ImageNet mean/std; the model backbone is fully trainable. The checkpoint/output protocol is versioned `cifar100_deit_tiny_v4` and every production job records a pretrained-backbone SHA-256; mismatched backbone weights in a shared results root are rejected. See [`docs/VIT_BACKBONE.md`](docs/VIT_BACKBONE.md).

## Core semantics

At *each communication interval*, all clients train locally, irrespective of whether they are selected to upload. **Default: two sequential local tasks, each comparing two class IDs; four new classes per client before the server update in the disjoint setting.** The first local task trains the client model; the second continues from its weights with the chosen local CL method (SGD, GEM, A-GEM, replay, EWC, SI, ER-ACE, DER, DER++, MAS, or LwF). Every task uses a local softmax over its own active labels; other classifier rows are frozen. The server uses one canonical classifier row per global class, *never task-specific heads or inference task IDs*.

The default stream has **10 clients**, **20% random clients hidden** at each update, and the exact novel-class allocation:

- Communication interval 0: client 0 `(0,1)` → `(20,21)`, client 1 `(2,3)` → `(22,23)`, ... client 9 `(18,19)` → `(38,39)`.
- Interval 1: client 0 `(40,41)` → `(60,61)`, ... client 9 `(58,59)` → `(78,79)`.
- Interval 2: client 0 `(80,81)`, ... client 9 `(98,99)`; fewer tasks in the final interval because no additional labels exist.
- A **separately flagged, communication-only terminal flush** connects every client with pending updates, so all 100 globally encountered classes may be integrated before final evaluation. Set `stream_terminal_flush=false` for an explicit no-flush ablation.

A disconnected client never uploads or downloads global weights. It continues local learning, retains memories, and eventually uploads the accumulated parameter changes since its *last* download. Participating clients upload before receiving the newly aggregated server model; server synchronization replaces their local model weights but preserves their private CL state.

## Controls

| Configuration knob | Default | Purpose |
|---|---:|---|
| `classes_per_local_task` | 2 | Number of distinct labels contrasted in one local softmax |
| `local_tasks_per_round` | 2 | Number of sequential local tasks before one server update |
| `stream_overlap_kind` | `none` | None, intra-client same/other interval, inter-client same/other interval, or combined |
| `stream_overlap_fraction` | 0 | Probability of reusing a *seen class* in a local episode |
| `sample_overlap_fraction` | 0 | Intentional raw-image duplication when a class appears more than once |
| `participation` | 0.8 | Active fraction at each communication update (8 of 10 by default) |
| `offline_duration_rounds` | 0 | 0 = independent per-round random selection; positive = deterministic seeded offline cohort held out for exactly D communication rounds |
| `stream_recurrence_rounds` | 0 | Explicit additional local recurrence rounds using distinct training examples |
| `stream_terminal_flush` | true | Final communication-only step: integrate all outstanding client updates |
| `method` / `aggregator` | `sgd` / `fedavg` | Local learner / server integration algorithm |

**Terminology:** The local stream consists of sequential supervised tasks; global prediction is *class-incremental*, with no task ID or task-specific head. This should not be called conventional task-ID-assisted task-incremental inference.

## Installation and validation

```bash
python -m pip install -e '.[test]'
pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_all_cifar100.sh --plan-only
```

## Execution

```bash
# Small subset of the balanced methodology cross, GPU required by default:
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh --suite balanced --seeds 0 --limit 2

# Inspect or export the entire exact registry without running training:
bash scripts/run_all_cifar100.sh --plan-only --export-plan outputs/cifar100_deit_v4.jsonl.gz

# Full sequential suite (very large GPU commitment):
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh
```

Use `--families balanced`, `--seed 0`, `--start N`, `--limit N`, `--data-root`, and `--output-dir` for smaller controlled runs. `--allow-cpu` is test/debug only. The production runner fails without CUDA and saves separate configuration, metrics, stage and resumable checkpoints, client state, and a final model for each job. Results are under `results/cifar100_deit_v4/` by default; completed jobs are skipped only when the protocol and configuration match.

**Do not reuse earlier CIFAR-100 results roots.** All old checkpoints and performance claims belong to the old protocol and must not be counted as new results.

## Documentation

- [`docs/CLASS_INDEXED_PAIR_PROTOCOL.md`](docs/CLASS_INDEXED_PAIR_PROTOCOL.md): server/client sequence, class-wise aggregation and offline learning details.
- [`docs/OVERLAP_CASES.md`](docs/OVERLAP_CASES.md): disjoint, same-client and cross-client same/different-time overlaps.
- [`docs/FACTORIAL_SUITE.md`](docs/FACTORIAL_SUITE.md): exact 53,805-job finite-grid catalogue and families.
- [`docs/STATUS.md`](docs/STATUS.md): verified tests, limitations, and remaining GPU/learning validation.
- [`configs/designs/stream_v4_deit.json`](configs/designs/stream_v4_deit.json): immutable planned job count and digest.
