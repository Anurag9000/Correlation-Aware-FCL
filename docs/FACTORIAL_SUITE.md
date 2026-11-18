# Full CIFAR-10 / CIFAR-100 finite-axis methodology sweep

## Scope and exact counts

| Property | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Original A–N jobs preserved | 423 | 425 |
| New distinct factorial jobs | 67,515 | 69,297 |
| **Full sequential suite jobs** | **67,938** | **69,722** |
| Client methods × server methods | 11 × 9 | 11 × 9 |
| Distinct pairings represented | **99/99** | **99/99** |
| Balanced four-scenario, five-policy core with three seeds | 5,940 | 5,940 |
| 20-class/10+10 CIFAR-100 factorial jobs | — | 1,782 |
| Batch size | **512** | **512** |
| Configured global rounds (total) | 2,717,412 | 13,651,180 |

The local methods are SGD, GEM, A-GEM, replay, EWC, SI, ER-ACE-style, DER, DER++, MAS and LwF. Server aggregators are FedAvg, uniform, FedAvgM, FedAdam, class-balanced, inverse-participation, conflict-aware, conflict projection and retention-aware. Existing partial/approximate mechanisms are not renamed as reference-faithful implementations.

## Balanced factorial core

11 client methods × 9 server methods × 4 scenarios (disjoint, overlap, related, recurrence) × 5 participation arms (random 100%, random 50%, random 25%, coverage 50%, coverage 25%) × 3 independent seeds (0,1,2) = **5,940 distinct jobs per dataset**.

## Additional distinct sweep families (post-deduplication)

| Family | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Additional data scenarios | 2,673 | 2,673 |
| All 22 client selectors across 3 scenarios | 17,820 | 17,820 |
| Participation rates/policies | 10,098 | 10,098 |
| Label overlap magnitude | 2,376 | 2,376 |
| Raw image duplication | 1,188 | 1,188 |
| Dirichlet label skew | 1,188 | 1,188 |
| Learning rate × local epochs | 6,534 | 6,534 |
| Episodic memory size | 1,512 | 1,512 |
| Importance regularization | 648 | 648 |
| FedProx coefficient | 1,782 | 1,782 |
| Server learning rate | 528 | 528 |
| Recurrence timing and holder | 2,376 | 2,376 |
| Class imbalance | 891 | 891 |
| Client quantity skew | 594 | 594 |
| Model architecture | 1,188 | 1,188 |
| Client count | 1,188 | 1,188 |
| Local momentum and state persistence | 891 | 891 |
| Memory selection policy | 189 | 189 |
| Replay ratio | 648 | 648 |
| Distillation loss coefficient | 486 | 486 |
| Replay-label coefficient | 324 | 324 |
| SI damping | 108 | 108 |
| FedAdam momentum/variance coefficients | 528 | 528 |
| GEM historical constraint grouping | 108 | 108 |
| Inactive-client pending data policy | 1,188 | 1,188 |
| Participation schedules | 2,970 | 2,970 |
| Target-client inclusion/exclusion | 1,188 | 1,188 |
| Spurious input features | 297 | 297 |
| Compatible label-noise controls | 12 | 12 |
| Heterogeneous client-local learning methods | 54 | 54 |
| CIFAR-100 20-class 10+10 | — | 1,782 |

The job generator is the source of truth; it deduplicates repeated configurations *within* the new factorial grid, retaining different random seeds. The original targeted job IDs and their dependency ordering are unchanged. Full exact job dictionaries can be exported as JSONL without committing enormous static files.

## Run/inspect

~~~bash
python -m pip install -e '.[test]'
pytest -q
bash scripts/run_all_cifar10.sh --plan-only
bash scripts/run_all_cifar100.sh --plan-only
bash scripts/run_all_cifar10.sh --plan-only --export-plan outputs/cifar10.jsonl.gz
bash scripts/run_all_cifar100.sh --plan-only --export-plan outputs/cifar100.jsonl.gz
bash scripts/run_all_cifar10.sh --suite legacy
bash scripts/run_all_cifar10.sh --suite balanced
bash scripts/run_all_cifar10.sh --suite factorial --families memory --seed 0 --limit 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar10.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh
~~~

Use --start, --limit, --seeds, --families, --data-root, --output-dir, --continue-on-error and --allow-cpu (deliberate debugging only). The default is sequential, CUDA-required execution, checkpoint/resume with independent artifact directories, and a frozen plan digest.

## Scientific limitations

These **137,660 jobs are planned, not executed** and represent an extremely large training and storage commitment. The factorial cross is exhaustive only for the declared finite axes; arbitrary combinations of every conceivable continuous hyperparameter are not attempted. Some selection policies are privileged simulation/oracle controls, and feedback based on official test-set metrics cannot be presented as leakage-free training. Split original training data into train/validation before validation-driven tuning and reserve the official test set for confirmatory evaluation. Current baseline adaptations and partial A–N block coverage remain unchanged. A planned job does not guarantee numerical stability on actual GPUs.
