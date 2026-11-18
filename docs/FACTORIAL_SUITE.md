# CIFAR-100-only continuous-local-task experiment catalogue (v3)

## Source of truth

`src/cafcl/full_matrix.py` lazily generates the active experiment ledger. Every job has a unique configuration fingerprint, a deterministic ID beginning `ST_`, batch size 512, strictly active-label task training, and the same communication-v3 simulator. **Old CIFAR-10, CIFAR-100/20, and stage-based 425-job/69,722-job suite registrations are retired.**

## Exact registered runs

| Family | Distinct CIFAR-100 jobs | Variable |
|---|---:|---|
| Balanced client/server/scenario/participation | 5,346 | 11 local learners × 9 server methods × 6 overlap patterns × 3 participation rates × 3 seeds |
| Local tasks and classes per task | 8,910 | Local tasks 1/2/3 and classes per task 2/4 |
| Controlled offline duration | 13,662 | Outages of 1,2,3,4 update intervals where feasible; deliberately recurrent rounds for longer outages |
| Overlap strength | 2,970 | 25%, 50%, 100% label reuse |
| Raw-image overlap | 1,188 | 0%, 25%, 50% train-sample duplication under cross-client overlaps |
| Client selection | 10,692 | Random, fixed, rotation, coverage, last-active, class/recurrence/relatedness and related selectors |
| Local optimization | 3,267 | LR × local epochs |
| Episodic memory | 756 | Memory per class 0,10,20,50,100 |
| Regularization | 324 | Strength for EWC, SI, MAS and LwF |
| FedProx | 891 | Proximal coefficient |
| Server optimization | 264 | FedAdam / FedAvgM server learning rate |
| Replay ratio | 324 | Replay, ER-ACE, DER, DER++ |
| Distillation coefficient | 243 | DER, DER++, LwF |
| Replay beta | 162 | ER-ACE and DER++ |
| GEM history constraint grouping | 54 | Class, local stage or aggregate memory |
| Client population | 594 | 5, 10, 20 clients |
| Recurrent rounds | 1,782 | 0, 2, 4 additional repeated-history intervals |
| Semantic CIFAR class order | 594 | Natural, related, unrelated ordering |
| Class-frequency imbalance | 891 | 0, .5, 1, 2 power |
| Client sample quantity skew | 594 | 0, 1, 3 quantity-skew coefficient |
| Terminal flush | 297 | Enable/disable final all-client communication |
| **Total** | **53,805** | 3 seeds, deduplicated among declared grids |

### Complete factorial core

Every one of the 99 local-learner/server-aggregator pairings appears in the six overlap geometries, each at participation levels 100%, 80%, and 50%, under three independent seeds:

`11 × 9 × 6 × 3 × 3 = 5,346` common-condition jobs.

**No CIFAR-10 job configurations exist in this active registry.** The `--dataset` parser accepts only `cifar100`; `cafcl suite` delegates to this registry. The prior `cafcl` old-stage job planner and its entrypoints have been removed.

### Defaults and independent knobs

- Local clients: 10; data classes: 100; classes per task: 2; sequential tasks between server updates: 2; default new classes per client per update: 4 (until 100-class exhaustion).
- The default disjoint stream consumes 40 new classes/round, followed by 40, followed by 20. No fabricated repetitions fill the final partial interval.
- Participation 80% means 8 selected and 2 offline with 10 clients. `offline_duration_rounds=0` uses independently sampled random client sets. Positive durations force rotating seeded absent cohorts for controlled disconnection tests.
- At the end, an optional explicitly marked communication-only round uploads pending updates from remaining offline clients.
- No task-ID-assisted inference or per-task output heads; official evaluation uses one canonical 100-row class-indexed classifier and all arrived labels.

### Validation and export

```bash
pytest -q
python scripts/validate_factorial_plans.py
bash scripts/run_all_cifar100.sh --plan-only
bash scripts/run_all_cifar100.sh --suite balanced --seeds 0 --plan-only
bash scripts/run_all_cifar100.sh --plan-only --export-plan outputs/cifar100_stream_v3.jsonl.gz
```

### Launch only after GPU pilot

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh --suite balanced --seed 0 --limit 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh --families offline_duration --seed 0 --limit 2
CUDA_VISIBLE_DEVICES=0 bash scripts/run_all_cifar100.sh
```

**Caution:** The full job list is a very large GPU commitment, and no official CIFAR-100 sweeps or hyperparameter-selection studies have been executed in this repository change. Official-test feedback in selection policies must not be presented as leakage-free adaptive training without a train-derived validation/control channel. Many oracle selectors and original-paper fidelity gaps remain. The factorial grid is exhaustive only with respect to the enumerated axes; arbitrary cross-products of every numeric parameter are not implied.
