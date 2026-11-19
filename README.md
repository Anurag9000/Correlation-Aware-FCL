# Correlation-Aware Federated Continual Learning — CIFAR-100, native 32×32 (v6)

**Primary goal: run exactly TWO initial investigation jobs**, not the ~50K-job factorial sweep. This project simulates independent continuous client learning under partial server participation. Every client sees consecutive supervised tasks, continues training even when disconnected, and eventually uploads accumulated parameter changes since its last synchronization. Every connected client receives the newly aggregated server model after the upload; the disconnected clients receive nothing. Private A-GEM/GEM/replay/EWC/SI/DER/etc. memories persist through normal sync.

## Selected architectures — identical native 32×32 image data and effective batch 1024

| Choice | `--backbone` | Production model ID | Initialization |
|---|---|---|---|
| **Default ViT** | `vit` | `vit_tiny_patch16_32` | Native PyTorch ViT-Tiny/16, seeded random |
| **Default CNN** | `cnn` | `resnet18_cifar32` | Native PyTorch ResNet-18 with CIFAR 3×3/stride-1 stem, seeded random |

Both are fully implemented directly in `src/cafcl/models.py`; no 224×224 resize, mandatory DeiT pretrained weights, standard CNN, or silent fallback. The 100-output classifier uses canonical global label IDs. A two-class task trains the active two-label softmax while other class rows remain frozen; the shared feature extractor remains trainable. ViT-Tiny/16 has only **2×2 = four image patches** at 32×32, an important accuracy limitation. See [BACKBONES.md](docs/BACKBONES.md).

Every CIFAR production job uses effective **batch size 1024** and physical microbatch 16 by default for gradient accumulation. For ResNet-18, BatchNorm running statistics reflect microbatches, not an actual physical 1024 batch; see methodology caveats.

## First: two-job primary investigation

Five clients each learn **four new distinct classes per server interval**, organized as **two sequential two-class tasks**. This lasts exactly five intervals: `5 clients × 4 classes × 5 intervals = 100 unique CIFAR-100 classes`. Compare:

1. **Full baseline:** all five clients upload and synchronize each interval.
2. **Rotating-offline intervention:** exactly one different client is offline during each of the five intervals, so each client is withheld once, with its continuing local model/memory preserved.

Both arms use **A-GEM locally, FedAvg + class-aware row fusion at the server, the same class stream, same seed 0, no label overlap, batch 1024**, and a final communication-only flush. Official test accuracy is evaluated **only once at the end**, after verifying all 100 labels reached the server. The pilot produces paired overall and per-class accuracy comparisons. This is **two jobs total**, not one per backbone; select which model the two jobs use.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --plan-only

# Two simultaneous CUDA processes on mtechdeserver, assigned one GPU each:
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,1
# Or ResNet-18, still just two jobs:
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpus 0,1
# If only one GPU is available, both can share it if memory permits:
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,0
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

The two-job primary probe now uses **Replay + FedAvg**, a 1,000-example private buffer per client, and a server-broadcast archive of all contributed examples. GEM/A-GEM share class-indexed constraint examples and recompute gradients at the receiving model; DER shares stored logits; EWC/SI/MAS combine fresh diagonal importance information; LwF uses a global teacher. See [the protocol](docs/FEDERATED_CL_STATE_V7.md). All execution artifacts use v7 output directories rather than overwriting v6 A-GEM results. **This is a raw-data-sharing research simulation, not privacy-preserving FL.**
