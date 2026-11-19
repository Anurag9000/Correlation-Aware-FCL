# Correlation-Aware Federated Continual Learning — CIFAR-100, native 32×32 (v6)

**Primary goal: run exactly TWO initial investigation jobs**, not the ~50K-job factorial sweep. This project simulates independent continuous client learning under partial server participation. Every client sees consecutive supervised tasks, continues training even when disconnected, and eventually uploads accumulated parameter changes since its last synchronization. Every connected client receives the newly aggregated server model after the upload; the disconnected clients receive nothing. Private A-GEM/GEM/replay/EWC/SI/DER/etc. memories persist through normal sync.

## Selected architectures — identical native 32×32 image data

| Choice | `--backbone` | Production model ID | Initialization |
|---|---|---|---|
| **Default ViT** | `vit` | `vit_tiny_patch16_32` | Native PyTorch ViT-Tiny/16, seeded random |
| **Default CNN** | `cnn` | `resnet18_cifar32` | Native PyTorch ResNet-18 with CIFAR 3×3/stride-1 stem, seeded random |

Both are fully implemented directly in `src/cafcl/models.py`; no 224×224 resize, mandatory DeiT pretrained weights, standard CNN, or silent fallback. The 100-output classifier uses canonical global label IDs. A two-class task trains the active two-label softmax while other class rows remain frozen; the shared feature extractor remains trainable. ViT-Tiny/16 has only **2×2 = four image patches** at 32×32, an important accuracy limitation. See [BACKBONES.md](docs/BACKBONES.md).

The full factorial suite defaults to effective **batch size 1024**. The two-job primary probe accepts effective batch sizes 64 or 1024 through `--batch-size` and uses physical microbatch 4 by default. For ResNet-18, BatchNorm running statistics reflect physical microbatches; see methodology caveats.

## First: two-job primary investigation

Five clients each learn **four new distinct classes per server interval**, organized as **two sequential two-class tasks**. This lasts exactly five intervals: `5 clients × 4 classes × 5 intervals = 100 unique CIFAR-100 classes`. Compare:

1. **Full baseline:** all five clients upload and synchronize each interval.
2. **Rotating-offline intervention:** exactly one different client is offline during each of the five intervals, so each client is withheld once, with its continuing local model/memory preserved.

Both arms use **Replay with independently reservoir-sampled per-class private buffers, FedAvg + class-aware row fusion at the server, the same class stream, same seed 0, no label overlap**, and a final communication-only flush. Set `--replay-fraction 0.11111111` for 50 images per class from the 450-image train split; set `--batch-size 64` to use effective batch 64 in both arms. Official test accuracy is evaluated **only once at the end**, after verifying all 100 labels reached the server. The pilot produces paired overall and per-class accuracy comparisons. This is **two jobs total**, not one per backbone; select which model the two jobs use.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --plan-only

# Two independent federated simulations share the SAME physical GPU 0:
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpu 0
# Or ResNet-18, still just two jobs:
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpu 0
# Within EACH run: five distinct client models train Task 1 concurrently,
# wait for all five, then train Task 2 concurrently, then perform FedAvg.
```

The launch tool prepares two independent processes, logs, resumable checkpoints, and a final paired report. **This repository has not been connected to `mtechdeserver` from this chat: no real-GPU training is claimed here.** See [PRIMARY_PROBE.md](docs/PRIMARY_PROBE.md).

## Revised full factorial registry (do not launch until pilot review)

The **source-generated, mathematically audited CIFAR-100 ledger contains 38,370 feasible configurations per selected backbone** (not 53,805). The stricter audit rejected **15,435** registrations: 2,178 unrealized overlap geometries, 6,534 truncated offline-duration interventions, 6,534 inert, aliased, or signal-free selection interventions, and 189 unused LwF hyperparameter configurations. The **5,346-job balanced core remains complete** across all 11 local CL methods, all nine server methods, six overlap geometries, three participation rates, and three seeds. The user-requested two-job probe uses one seed and a single rotating 80% participation policy; it is **not** the balanced core.

```bash
python scripts/audit_stream_math.py
bash scripts/run_all_cifar100.sh --backbone vit --plan-only
bash scripts/run_all_cifar100.sh --backbone cnn --plan-only
# Later, after successful pilots and method-specific training checks:
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh --backbone vit --suite balanced --seed 0 --limit 2
```

Supported research backbones are switchable by the model registry, with distinct job configuration hashes and separate results directories. Add further model variants through `models.BACKBONES` after scientific review.

## Scientific notes

- Training is sequential **task-structured**, with known local labels, but the server inference uses **one class-incremental head** over all observed classes and **no task IDs**.
- Offline clients keep training and later contribute all unsynchronized local learning; FedAvg operates on *rebased accumulated deltas*, so these runs are not conventional synchronous FedAvg.
- The class-specific server merge uses optimizer-step counts, a heuristic that requires empirical validation.
- Five communication intervals plus a final flush is mathematically valid for 5 clients and 100 distinct classes; extending this to 10 clients for five full intervals without overlap is invalid.
- Only official CIFAR-100 GPU results can establish model performance. The tests are software/mathematical validations, not research accuracy results.

[Factorial families](docs/FACTORIAL_SUITE.md) · [Primary probe protocol](docs/PRIMARY_PROBE.md) · [Overlap specification](docs/OVERLAP_CASES.md) · [Exact design and feasibility audit](configs/designs/stream_v6_native32.json) · [Status and caveats](docs/STATUS.md)


## V7 server-synchronized continual learning

The two-job primary probe now uses **Replay + FedAvg** with an independently reservoir-sampled **10% per class** private buffer (45 of 450 TRAIN examples per class on CIFAR-100). The server stores every uploaded reservoir exemplar, deduplicates by original TRAIN example ID, and broadcasts its full union to connected clients. Disconnected clients do not receive updates until rejoining. GEM/A-GEM share class-indexed constraint examples and recompute gradients at the receiving model; DER shares stored logits; EWC/SI/MAS combine fresh diagonal importance information; LwF uses a global teacher. See [the replay-reservoir v9 protocol](docs/REPLAY_RESERVOIR_V9.md) and [shared CL-state specification](docs/FEDERATED_CL_STATE_V8.md). All execution artifacts use isolated v9 result directories and cannot reuse older checkpoints. **This is a raw-data-sharing research simulation, not privacy-preserving FL.**


## Required single-GPU client concurrency (v10)

All active simulator pathways (primary pilot, single configured run, balanced and full factorial) use one persistent PyTorch model and, on CUDA, one independent CUDA stream **per client**. Every communication interval executes Task 1 across all clients concurrently, waits for each client to finish, executes Task 2 concurrently, waits again, and only then performs FedAvg and CL-state synchronization. This includes clients that are offline for federation. A terminal communication-only flush launches no local tasks. The implementation records per-task concurrent execution windows and client IDs in `metrics.json`.

The **two** pilot arms are separate operating-system processes simultaneously sharing one physical GPU by default (`--gpu 0`, legacy alias `--gpus 0,0`; distinct GPU IDs are rejected). Each pilot arm has **five independently live client ViTs plus a server ViT** (twelve live ViT objects across both runs, with independent CL state and model weights). The physical microbatch defaults to 4 for this single-device campaign, while the effective optimizer batch remains **1024**. The GPU may serialize some kernels due to hardware/resource limits; this is genuine concurrent scheduling, not a guarantee of simultaneous kernel execution. No fallback to sequential per-client training is provided. GPU OOM requires explicit memory/physical-microbatch adjustment.

Use `bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpu 0 --replay-fraction 0.10 --epochs 10 --micro-batch-size 4`. The v10 protocol and `results/primary_cifar100_single_gpu_parallel_v10_vit_seed0` output root invalidate previous sequential checkpoint reuse. See [Single-GPU parallel training](docs/PARALLEL_SINGLE_GPU_V10.md).


## v11: Server-algorithm integrity

The server now keeps **canonical whole-model** treatment for FedAvg, FedAvgM (server Nesterov momentum) and FedAdam (Reddi FedOpt uncorrected moments); every floating-point feature and classifier parameter receives the same algorithm. `uniform` is equal-mean ablation. The five remaining methods—`class_balanced`, `inverse_participation`, `conflict_aware`, `conflict_projection`, `retention_aware`—are explicitly **repository-specific** methods, not claimed as mathematically faithful instances of independently published algorithms. The custom classwise classifier merger is allowed **only** for these five; default legacy `classwise_head_merge=True` cannot corrupt canonical FedAvg/FedAvgM/FedAdam.

The server records historical exposure counts per class **only after accepting uploads**, never hidden/offline clients' pending classes. The custom `class_balanced` scheme weights clients using both new-class counts and accepted prior exposure. Canonical FedAvg deliberately uses **training-example counts, not class count**, per McMahan et al. When an offline client reconnects, its delayed weight delta is rebased against the current global model: this is clearly marked `stale_delta_adapter_noncanonical`, **not** paper-equivalent synchronous FedAvg. See [full server audit](docs/SERVER_ALGORITHM_FIDELITY_V11.md). Old checkpoints and pilot results cannot be mistaken for v11 runs.
