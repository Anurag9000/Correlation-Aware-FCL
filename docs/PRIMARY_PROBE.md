> **CURRENT v16 (authoritative):** [Mandatory physical batch 64, Replay, SGD and evaluation audit](V16_BATCH64_RESERVOIR_SGD_AUDIT.md). All active production model forwards used for training, Replay, EWC/MAS, gradient constraints and diagnostics use 64 inputs; scoring counts only non-padded examples. The dual ResNet-18/ViT-Base suite is two concurrent processes **on GPU 0**, 50/class global shared Replay, SGD momentum 0.9 persisted between tasks and across communication, nine learning/synchronization rounds plus terminal flush. The CIFAR-100 factorial ledger is frozen in [v16 manifest](../configs/designs/stream_v16_batch64.json), 32,133 jobs/backbone. Previous v6–v15 split-batch and memory-policy descriptions below are historical; do not execute their command examples as current settings. Old result files and archived manifests remain intact for scientific provenance.

> **ACTIVE v14 DUAL-SUITE PROTOCOL:** [full v14 specification](V14_SEEN_CLASS_OPTIMIZATION_AND_SYNC_EVAL.md). Historical v10–v13 five-round statements below are **not** the current execution. Current pilot: **9 training intervals** (4 classes/client first, 2/client subsequently), **1 final flush**, five concurrent clients in each of two independent GPU-0 processes, random ViT-Base/16 or CIFAR-stem ResNet-18, SGD momentum .9 weight decay .0005, batch64, microbatch8, lr .03, 10 epochs/task, exactly 50 private replay examples/class. Produces `per_sync_accuracy/sync_001.png` through `sync_010.png` for each run, with CSV/Markdown/JSON alongside. Original class-ID stream and canonical whole-model FedAvg are preserved.

# Primary investigation: two matched CIFAR-100 jobs, 5 clients, 5 rounds

**Goal:** Determine whether periodically withholding one continuously learning client changes final global 100-class accuracy, without conflating the result with never training that client's classes.

| Controlled setting | Baseline | Intervention |
|---|---|---|
| Data | Official CIFAR-100, 100 classes, native 32×32 | Identical |
| Clients | 5 | 5 |
| Local method | Replay | Replay |
| Server integration | FedAvg + canonical class-row fusion | Identical |
| Local class overlap | None | None |
| Classes per task | 2 | 2 |
| Sequential tasks per interval | 2 | 2 |
| Training intervals | 5 | 5 |
| Novel labels | 100, each exactly once | Identical |
| Selected clients per interval | 5/5 | 4/5 |
| Offline schedule | None | One *different* offline client each of five intervals, seeded |
| Communication-only terminal flush | Yes (no unsent client remains) | Yes (integrates the fifth client's pending update) |
| Official test | Once at the end | Once at the end |
| Global classifier | 100 canonical output rows | Identical |
| Model choice | `--backbone vit` by default; alternatively `--backbone cnn` | Same chosen backbone |
| Effective batch | 64 (or 1024 by default) | 64 (or 1024 by default) |
| Random seed | 0 | 0 |

## Exact feasibility and arithmetic

`5 clients × 2 tasks/client/round × 2 new classes/task × 5 rounds = 100` unique class labels. The dataset has 100 classes, so the full stream is exactly filled. No label is repeated across task occurrences or clients. Each client receives four fresh classes in every interval. There is **exactly one distinct hidden client per interval**: the five hidden identities are a seeded permutation of {0,1,2,3,4}, hence each client is hidden once and only once.

For seed 0: offline client order = **1, 4, 2, 3, 0**. In each affected round, 4 novel labels are absent from that server update. Those labels are **not permanently excluded**; the corresponding offline client retains its 4-class learning and uploads it on its next connection. The final offline client is recovered by the mandatory **sixth communication-only flush**, which performs no new local training. Both arms receive a final flush for consistent event accounting.

The main comparison is **final CIFAR-100 test accuracy across all 100 classes**, with per-class accuracy differences reported. This does not compare against a permanently excluded class arm. The comparison evaluates the same 10,000 held-out examples, **only once after the final flush**, and records which class IDs actually reached the server. Local task labels are known during training; there are no task-specific inference heads or task IDs at prediction.

## Launch concurrently on `mtechdeserver`

Clone/pull `main` on that machine, install dependencies, check CUDA, then:

```bash
python -m pip install -e '.[test]'
python scripts/validate_factorial_plans.py
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpu 0 --seed 0 \
  --batch-size 64 --micro-batch-size 4 --epochs 10 --lr 0.003 \
  --replay-fraction 0.11111111 --plan-only
# Launch the same two jobs together on one GPU by removing --plan-only.
```

`--gpu` chooses the **one physical GPU shared by both runs**; distinct GPU IDs are rejected. Within each run five independent client models train each task ordinal concurrently, with an explicit barrier before the next task and before FedAvg. Running both backbone types would produce four experiments, not the requested two. `--seed 0`, `--lr .003`, `--epochs 10`, `--batch-size` (64 or 1024), `--micro-batch-size`, `--replay-fraction`, `--data-root`, `--output-dir`, `--no-download`, and `--plan-only` are configurable. Use `--replay-fraction 0.11111111` for 50 images per class from the 450-image training split. The parent ensures CIFAR-100 is cached before spawning the two worker processes to avoid concurrent download races. Production requires CUDA by default. Independent logs, checkpoints, and status files are saved for both jobs.

Final outputs are written under `results/primary_cifar100_single_gpu_parallel_v10_<backbone>[_pct<replay>]_bs<batch>_seed<seed>/comparison/`; the paired report is computed only after **both** runs succeed and all 100 classes are verified as integrated.

**Scientific scope:** This is a feasibility/proof-of-concept ablation, not a published statistical finding. One seed is insufficient for uncertainty estimates. Participation changes communication frequency and the trajectory of stale updates, not merely class identities. The terminal flush changes the protocol but is essential to measure the final 100-class model fairly. Study per-round coverage and pre-flush sensitivity separately later.

**Caveat:** Training a 32×32 ViT-Tiny/16 from random initialization provides only four image tokens and may yield low absolute accuracy. ViT-specific optimizer learning rate and training duration have **not** been tuned. Negative outcomes may be architecture/training issues rather than falsification of the federated-continual-learning hypothesis.

## V7 shared archive
In the current v9 primary probe, each client retains an **independent 10% per-class Algorithm R reservoir**: 45 examples from each of 20 encountered CIFAR-100 classes (up to 900 private exemplars). The old flat 1,000-sample limit is not active in this fractional mode. The server merges all uploaded private reservoir samples by class and training-ID provenance, stores compact uint8 RGB payloads, and broadcasts the entire unbounded archive to online clients. Offline clients retain their own model and reservoir until reconnection. This explicitly shares raw training images; it is NOT privacy-preserving FL.


## v9 two-arm pilot: independently sampled 10% per-class reservoirs

The current two-process CIFAR-100 probe runs plain Replay + FedAvg, five clients, five 20-new-class communication intervals, 32×32 images, effective batch 1024, seed 0. Both arms now set `replay_reservoir_fraction=0.10` and train 10 epochs per binary episode by default. For CIFAR-100's 45,000-example TRAIN split, each class has 450 examples and every client's **independent per-class Algorithm R reservoir** is capped at 45 examples of that class. Each client encounters 20 different classes across five intervals, so up to 900 private exemplars. The old 1,000-example global private cap is bypassed in this fraction mode; this is an intentional per-class budget, not the previous total-budget algorithm.

Only connected clients send their current sampled private exemplars to the server. The server retains a provenance-deduplicated, unbounded union of uploaded samples, not the entire raw training stream. The server broadcasts that union with model weights to connected clients. A disconnected client trains and samples locally without receiving updated archives. It contributes its reservoir and accumulated model delta on its next connection. No classes overlap in this pilot; the expected complete server archive after final communication-only flush is exactly **100 × 45 = 4,500 examples**. Reservoir sampling of later repeated class arrivals uses persistent private per-class seen counts and ID deduplication; server synchronization does not reset either.

The two runs remain independent processes launched together on a single GPU by `scripts/run_primary_probe_cifar100.sh`. Each run internally launches five independent client models concurrently at Task 1, joins all five, then concurrently executes Task 2 and joins before server synchronization. Use `--backbone vit --seed 0 --gpu 0 --replay-fraction 0.10 --epochs 10 --lr 0.003 --micro-batch-size 4`. Old v9 or earlier sequential checkpoints cannot be reused: job IDs, result root and simulator protocol are v10. The same command with `--plan-only` inspects both configurations without running training.

**Scope:** 10% refers to the available training pool for each class (450 per class in the standard 90/10 train/validation split), not 10% of the full 500 including held-out validation. For overlap experiments, distinct clients may observe different fractions of a class's global training pool, so the fixed quota is defined relative to the complete permitted TRAIN class, not a future-aware prediction of each client's stream length. The server archive contains all uploaded retained exemplars; it can grow beyond a 10% global limit in overlapping/recurring schedules. This method explicitly shares training examples and is not private FL.


## Strict two-level concurrency: v10

The pilot executes exactly **two Python processes concurrently on GPU 0**; each process has one server model and five separate client model objects. Each client owns its own model weights, replay state, optimizer state and CUDA stream. At every local task ordinal, **all five clients submit their work through a five-worker thread pool concurrently**. The simulator waits until every Task 1 finishes before starting any Task 2 and waits for every Task 2 before selecting/uploading clients or merging server history. Offline clients train with the same scheduling; only communications are withheld. Task 2 immediately replays the locally retained Task 1 reservoir, and tasks after server synchronization use the full synchronized archive. Physical kernel overlap depends on GPU resources; logical task barriers are enforced and recorded.

All active CIFAR-100 suite entrypoints use the same simulator implementation and protocol. See `PARALLEL_SINGLE_GPU_V10.md`.


## v11 canonical FedAvg heads and provenance

The full-participation pilot now uses original **sample-weighted whole-model FedAvg** for all floating-point ViT tensors, including *every one of the 100 classifier rows and biases*. No custom classwise postprocessing occurs. The rotating-offline pilot applies the same complete-model mean to delayed updates, but reconnected clients first rebase their accumulated displacement `G+(L-A)`: this is **not** original synchronous FedAvg and is marked separately in the metrics. The server tracks accepted class-example exposure without using unsynchronized client-only class counts. `--gpu 0` remains the single-device command, with two independently running five-client simulations. Job IDs use `V11_SERVER`, and the primary result root becomes `primary_cifar100_single_gpu_server_v11_<backbone>_seed<seed>`.


## New v12 CL-method semantics

The local 11-method training implementation now uses **paper-grounded per-example EWC Fisher and MAS sensitivity**, **SI task-only gradient path importance**, online DER/DER++ logit capture before the corresponding optimizer step, and true persistent per-class random reservoirs for memory-based methods. All 5-client task barriers and strict active classifier rows are unchanged. The primary output root is `primary_cifar100_single_gpu_cl_v12_<backbone>_seed<seed>`, and job IDs use `V12_CL`. Older run checkpoints are incompatible with the v12 protocol. For precise fidelity-vs-setting-adaptation distinctions see [CL method fidelity](CL_METHOD_FIDELITY_V12.md).


## Current v13 pilot overrides the historical five-round plans above

The 2026-11-19 protocol is now **nine training communication rounds**: first round 4 classes/client (two sequential tasks) then **eight rounds 2 classes/client** (one continual task immediately after each global model+CL-history download). One client is excluded per round in the rotating-offline arm, using cyclic repetition; the tenth communication-only round synchronizes all pending state. The current v13 pilot produces `results/primary_cifar100_first4_then2_v13_vit_seed0/`. All other CL methods in all active jobs use this exact cadence. [Current authoritative protocol](FIRST4_THEN2_FULL_CL_V13.md).
