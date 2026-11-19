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
