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
| Effective batch | 1024 | 1024 |
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
bash scripts/run_primary_probe_cifar100.sh --plan-only
# Two GPUs: two independent job processes, one on each GPU
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,1
# One GPU (both processes share it; benchmark capacity first)
bash scripts/run_primary_probe_cifar100.sh --backbone vit --gpus 0,0
# Alternative, still only two jobs and same schedule, with ResNet-18
bash scripts/run_primary_probe_cifar100.sh --backbone cnn --gpus 0,1
```

`--gpus` selects GPUs for the **two runs**, not the two backbone types. Running both backbone types would produce four experiments, not the requested two. `--seed 0`, `--lr .003`, `--epochs 1`, `--memory-per-class 20`, `--data-root`, `--output-dir`, `--no-download`, `--plan-only` are configurable. The parent ensures CIFAR-100 is cached before spawning the two worker processes to avoid concurrent download races. Production requires CUDA by default. Independent logs, checkpoints, and status files are saved for both jobs.

Final outputs: `results/primary_cifar100_shared_cl_v7_<vit|cnn>_seed0/comparison/summary.json`, `per_class.csv`, `per_class.json`, and `comparison.md`; all computed only after **both** runs succeed and after verifying all 100 classes have been integrated.

**Scientific scope:** This is a feasibility/proof-of-concept ablation, not a published statistical finding. One seed is insufficient for uncertainty estimates. Participation changes communication frequency and the trajectory of stale updates, not merely class identities. The terminal flush changes the protocol but is essential to measure the final 100-class model fairly. Study per-round coverage and pre-flush sensitivity separately later.

**Caveat:** Training a 32×32 ViT-Tiny/16 from random initialization provides only four image tokens and may yield low absolute accuracy. ViT-specific optimizer learning rate and training duration have **not** been tuned. Negative outcomes may be architecture/training issues rather than falsification of the federated-continual-learning hypothesis.

## V7 shared archive
Each participating client uploads up to 1,000 privately retained training exemplars, which the server merges by class and sample provenance. It stores compact uint8 RGB sample payloads and broadcasts the entire archive to online clients with the aggregated model. Offline clients continue from their last downloaded model and archive. The combined archive is **not** capped at 1,000; 1,000 is the private client budget. The protocol explicitly shares raw images, and therefore is NOT privacy-preserving FL.


## v9 two-arm pilot: independently sampled 10% per-class reservoirs

The current two-process CIFAR-100 probe runs plain Replay + FedAvg, five clients, five 20-new-class communication intervals, 32×32 images, effective batch 1024, seed 0. Both arms now set `replay_reservoir_fraction=0.10` and train 10 epochs per binary episode by default. For CIFAR-100's 45,000-example TRAIN split, each class has 450 examples and every client's **independent per-class Algorithm R reservoir** is capped at 45 examples of that class. Each client encounters 20 different classes across five intervals, so up to 900 private exemplars. The old 1,000-example global private cap is bypassed in this fraction mode; this is an intentional per-class budget, not the previous total-budget algorithm.

Only connected clients send their current sampled private exemplars to the server. The server retains a provenance-deduplicated, unbounded union of uploaded samples, not the entire raw training stream. The server broadcasts that union with model weights to connected clients. A disconnected client trains and samples locally without receiving updated archives. It contributes its reservoir and accumulated model delta on its next connection. No classes overlap in this pilot; the expected complete server archive after final communication-only flush is exactly **100 × 45 = 4,500 examples**. Reservoir sampling of later repeated class arrivals uses persistent private per-class seen counts and ID deduplication; server synchronization does not reset either.

The two runs remain independent processes launched in parallel by `scripts/run_primary_probe_cifar100.sh` (typically GPUs 0 and 1). Use `--backbone vit --seed 0 --gpus 0,1 --replay-fraction 0.10 --epochs 10 --lr 0.003`. Old v8 runs/checkpoints cannot be reused: IDs, result root and simulator protocol are v9. The same command with `--plan-only` inspects both configurations without running training.

**Scope:** 10% refers to the available training pool for each class (450 per class in the standard 90/10 train/validation split), not 10% of the full 500 including held-out validation. For overlap experiments, distinct clients may observe different fractions of a class's global training pool, so the fixed quota is defined relative to the complete permitted TRAIN class, not a future-aware prediction of each client's stream length. The server archive contains all uploaded retained exemplars; it can grow beyond a 10% global limit in overlapping/recurring schedules. This method explicitly shares training examples and is not private FL.
