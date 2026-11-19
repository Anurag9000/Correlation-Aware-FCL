# CIFAR-100 v6 implementation and validation status

## Completed in source

- **Exactly two switchable native production architectures**, both CIFAR-100 at 32×32: ViT-Tiny/16 (`vit_tiny_patch16_32`), CIFAR-stem ResNet-18 (`resnet18_cifar32`). No silent fallback to DeiT-224, SmallCNN or other CNN/ViT architectures; future model variants use the single registry in `models.py`.
- **Mandatory effective batch 1024** across research jobs, configurable physical microbatch 16 for memory; GEM/A-GEM projected gradients operate on logical batches.
- **Two-job primary A-GEM + FedAvg study:** five clients, two sequential two-class tasks per client each interval, five training intervals giving exactly 100 different classes. Full-participation baseline and an independently running arm with a different one-of-five client offline in each interval. One seed, one partial participation rate (80%), no overlap, final communication-only flush, same official 100-class test once at the end, per-class paired comparison.
- No raw data or historical memory uploaded. Offline clients continue on persistent private weights and CL histories. On reconnection, pending changes are rebased to the current server model; participating clients get the newly merged global weights; disconnected clients remain untouched.
- Canonical 100-class output mapping with a locally active task softmax; noncurrent classifier rows protected; representation backbone remains trainable.
- ResNet-18 BatchNorm running counters are handled as monotone integer buffers, rather than causing aggregation to fail when disconnected clients have different counts.
- **Mathematical audit:** of 53,805 existing per-backbone registrations, 38,370 remain structurally feasible and semantically distinguishable; 15,435 are excluded with recorded failure reasons (2,178 invalid overlap geometries, 6,534 invalid offline durations, 6,534 inert/aliased selection variants, and 189 LwF configurations varying unused parameters). Both choices cover all 99 local CL/server method combinations and the complete 5,346 balanced core.
- Dedicated offline tests cover model shapes, active classifier rows, complete 100-class task schedule, exact once-only client exclusion, paired report, model switch, and math-audit pruning.

## Not established

- **No GPU server connection or real CIFAR-100 training was performed from this chat.** The script for concurrent launch on `mtechdeserver` is ready, but must be started on the machine or through an authorized remote execution connection.
- The official final accuracy and statistical validity of the FCL research hypothesis are unknown. Two runs with one seed are explicitly a preliminary probe, not an uncertainty-calibrated result.
- ViT-Tiny/16 has only four patch tokens at 32×32; this is an important architecture limitation. Both models now initialize randomly from the controlled seed; transfer-pretrained DeiT-224 has been retired, not silently reused at native 32×32.
- ResNet BatchNorm running-statistics over microbatches are not mathematically equivalent to a physical batch of 1024. This limitation is reported.
- Standard FedAvg is not identical to the code's stale-anchor accumulated-delta and class-wise classifier merge variant. The latter needs separate scientific validation.
- The generator audits declared geometric/class-budget feasibility, but actual GPU numerical stability, training accuracy, paper fidelity of each CL algorithm, and some data/selector privacy issues are still to be established.

## Reproducibility

```bash
python -m pytest -q
python scripts/validate_factorial_plans.py
python scripts/audit_stream_math.py
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,1
```

Checkpoint protocol `cifar100_native32_v6`; distinct model-specific results roots. Never count old `cifar100_deit_v4` or v5 checkpoints as native v6 results. v6 prevents accidental resumption of CNN runs predating the BatchNorm reference-gradient fix.


## V7 shared continual-learning state

The model-only communications protocol is superseded by a v7 model + method-native-state protocol. The central server deduplicates contributed replay samples and DER target logits, aggregates nonduplicate diagonal EWC/SI/MAS importance contributions, and broadcasts its latest shared state only to clients that actually synchronize. The two-job pilot now uses Replay/FedAvg, private replay capacity 1000, and a separate v7 root. The frozen v6 job-geometry manifest is preserved, but v7 results are NOT equivalent to v6 results; old checkpoints cannot be reused. Cached GEM/A-GEM gradients from incompatible models are never naively averaged; their shared class evidence supports current-model reference-gradient recomputation. These updates are a **data-sharing simulator** and are not private FL.


## Mandatory shared CL history — v8

Every CIFAR-100 job now synchronizes full server continual-learning history with the model; the first next local task uses received compatible state. Includes mixed-method artifact routing, provenance-safe GEM, SI last-task consolidation and offline rejoin tests. See [v8 protocol](FEDERATED_CL_STATE_V8.md). v7 checkpoints are incompatible.


## Current two-run CIFAR-100 Replay probe (v9)

The two-arm primary probe now uses **classwise 10% Algorithm R reservoir replay**, not the earlier flat 1,000-example client cap or A-GEM method. Each class has 450 allowed TRAIN images (50 reserved for validation), so a client's fixed quota is 45 per class. Every client learns its own assigned two-label tasks and retains a persistent private reservoir; the server collects a deduplicated union from connected clients and broadcasts its entire exemplar archive. Hidden clients do not exchange either weights or memory. With 100 classes appearing once in this pilot, the expected final server archive is 4,500 samples after the communication-only flush.

Native 32×32 ViT-Tiny/16 and CIFAR-stem ResNet-18 remain selectable. The default ViT pilot runs only two concurrent GPU processes with effective batch size 1024, seed 0 and 10 epochs per binary episode. The full 38,370-job-per-backbone geometric ledger still applies; other methods keep their existing per-method budgets. The v9 simulator protocol, job IDs and primary output root isolate results from v8 and all preceding versions.
