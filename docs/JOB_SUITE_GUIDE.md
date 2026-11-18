# Full sequential CIFAR experiment suites — research and execution guide

> Source-grounded documentation for the **actual** frozen job manifests on `main`. The manifests are the authoritative exact configurations; these pages are a readable index, not additional invented jobs.

## Dataset coverage

| Property | CIFAR-10 | CIFAR-100 |
|---|---:|---:|
| Registered jobs | **423** | **425** |
| Configured batch size | **512** | **512** |
| Blocks with dedicated jobs | 144 | 144 |
| Default clients | 10 | 20 |
| Default stages | 2 | 10 |
| Default rounds per stage | 20 | 20 |
| CIFAR-100 20-class subset jobs | 0 | 17 |

Across both datasets: **848 scheduled jobs**, covering **144 of 152 research block IDs**. **Unscheduled IDs:** `A12`, `B06`, `B10`, `H08`, `K05`, `K08`, `M08`, `N06`.

**CIFAR-100's 20-class jobs:** They use the special `cifar100_20` loader with a selected 20-class (10+10) progression; they are not runs on all 100 classes. The other CIFAR-100 jobs use the full dataset.

## Complete per-job inventory and machine-readable index

- [CIFAR-10 — all 423 jobs](JOBS_CIFAR10.md)
- [CIFAR-100 — all 425 jobs](JOBS_CIFAR100.md)
- [Full CSV of all 848 registered jobs](JOB_INDEX.csv), with original order, block, method, selected clients, per-job exact configuration changes and output folder.
- Source-of-truth raw manifests: [`cifar10_jobs.json`](../configs/plans/cifar10_jobs.json) and [`cifar100_jobs.json`](../configs/plans/cifar100_jobs.json).

## Exact execution protocol

1. The scenario delivers all data arrivals, including arrivals at inactive clients, independently of the selection decision.
2. A selector chooses eligible participants under the job's participation rate, optional custom schedule, and any explicit client overrides.
3. All selected clients download the current canonical global model, train using permitted local data/memory with the configured local method, then upload accepted updates.
4. The server aggregates using the configured algorithm and **broadcasts the updated global model to all selected active clients**. Full synchronization of inactive clients is an optional ablation.
5. The evaluator includes all globally arrived classes/distributions, irrespective of whether their holder participated. Never-learned classes are distinguished from actual forgetting.
6. Per-job artifacts are saved independently. Multi-round forks must load the explicitly declared earlier baseline checkpoint.

## Launch scripts

```bash
python -m pip install -e ".[test]"
pytest -q
bash scripts/run_all_cifar10.sh --plan-only
bash scripts/run_all_cifar100.sh --plan-only
bash scripts/run_all_cifar10.sh
bash scripts/run_all_cifar100.sh
# Or, both datasets sequentially (CIFAR-10 first):
bash scripts/run_both_sequentially.sh
```

Use `--limit 1` for a smoke test, `--job-ids JOB_A JOB_B` to run named planned jobs in canonical plan order, and `--stop-on-error` to stop after a failure. Named selections retain the original job IDs and write to the usual per-dataset results tree, so completed representative validations can be reused by a later full-suite run. `DATA_ROOT` and `RESULTS_ROOT` override input/output roots. The CLI regenerates jobs from source; the frozen plan JSON files should be checked with `PYTHONPATH=src python scripts/validate_plans.py`. A different `--seed` is a different realization, and should use a **different results root** to avoid collision.

**Training status:** The documented jobs are configured, not executed official CIFAR findings. A job may fail at runtime and still appear in this inventory. Partial and approximate methods have not become validated paper reproductions by being listed here.

## Interpretation of the job fields

Each per-dataset inventory includes an exact 75+ field base configuration and a list of **every differing field** for every job. For an exact job, use the corresponding original `job_id` and `block_id` plus the base and differences. Values are serialized as JSON; empty arrays and dictionaries are meaningful. The CSV provides those same differences in a filterable format.

| Field | What the configuration controls |
|---|---|
| `dataset` | Loader identity; CIFAR-100 subset jobs use cifar100_20 |
| `batch_size` | Configured main DataLoader batch size (512); final minibatches may be smaller |
| `num_clients` | Simulated client count |
| `rounds_per_stage` | Global aggregation rounds per stage |
| `epochs` | Local epochs for each selected client in each round |
| `participation` | Target participating-client fraction |
| `selector` | Client selection algorithm |
| `method` | Local continual-learning optimizer/objective |
| `aggregator` | Server aggregation/update algorithm |
| `scenario` | Data partition and arrival setting |
| `stages` | Number of stream stages |
| `memory_per_class` | Private memory examples per locally observed class |
| `lr` | Local learning rate |
| `model` | Backbone/classifier architecture |
| `seed` | Random realization identifier |
| `full_sync` | Whether inactive clients also receive each aggregate |
| `relation` | Semantic class ordering: natural/related/unrelated |
| `recurrence_delay` | Stage-level recurrence offset |
| `recurrence_client_policy` | Same-client or cross-client recurrence |
| `overlap` | Intended client-label overlap |
| `dirichlet_alpha` | Non-IID class-partition concentration |
| `target_client` | ID targeted by selection intervention |
| `force_target` | Include/exclude/none intervention |
| `intervention_round` | Target intervention round; -1 denotes unrestricted/default |
| `gradient_diagnostics` | Request gradient diagnostics |
| `causal_diagnostics` | Request aggregation-level causal probes |
| `causal_max_clients` | Maximum accepted cohort for subset diagnostics |
| `subset_diagnostics` | Enumerate subsets of accepted updates |
| `prepost_diagnostics` | Measure historical loss before/after operations |
| `representation_diagnostics` | Measure representation changes |
| `resume` | Allow checkpoint resumption |
| `checkpoint_every` | Interval of additional round checkpoints |
| `participation_schedule` | Constant, staged, increasing, decreasing or feedback policy |
| `pending_policy` | Handling of inactive-client pending data |
| `max_pending_samples` | Capacity when pending data is bounded |
| `replay_ratio` | Local replay proportion |
| `der_alpha` | DER logit-preservation coefficient |
| `der_beta` | DER++ replay-label coefficient |
| `regularization_strength` | Local regularization coefficient |
| `fedprox_mu` | FedProx proximal coefficient |
| `server_lr` | Server optimizer learning rate |
| `server_beta1` | First-moment/server momentum coefficient |
| `server_beta2` | Second-moment/server adaptive coefficient |
| `fedadam_epsilon` | FedAdam denominator stabilization |
| `memory_policy` | Selection rule for client episodic memory |
| `local_optimizer` | Local optimizer family |
| `client_heterogeneity` | Allow different local methods among clients |
| `fork_source_job` | Dependency source job for multi-round causal fork |
| `fork_checkpoint_round` | Source checkpoint round for causal fork |
| `job_id` | Unique name within dataset suite |
| `block_id` | Experiment block ID among A01–N06 |
| `class_subset` | Requested source class IDs for explicit subset |
| `class_order_override` | Explicit global-label order |
| `holders_override` | Explicit class-to-client-holder allocation |
| `class_holder_weights` | Allocation weights among each class's holders |
| `class_sample_fraction` | Training-data dose per class |
| `participation_overrides` | Exact selected clients in named rounds |
| `arrival_lag_by_client` | Per-client delay in data arrivals |
| `recurrence_round_delay` | Round-level recurrence lag |
| `recurrence_period` | Spacing between repeated recurrences |
| `recurrence_repeats` | Number of recurrence events |
| `sample_overlap_fraction` | Fraction of examples duplicated across clients |
| `client_sample_skew` | Client sample-count skew |
| `class_imbalance_power` | Long-tail class frequency control |
| `stage_participation` | Explicit per-stage participation rates |
| `gem_grouping` | GEM historical reference groups: class, stage or all |
| `optimizer_state_policy` | Reset/persist local optimizer state on sync |
| `momentum` | Local SGD momentum |
| `si_xi` | Synaptic Intelligence damping |
| `selection_temperature` | Stochastic selector temperature |
| `boundary_mode` | Known versus hidden stage boundaries |
| `learning_setting` | Class-IL, task-IL, domain-IL or class-aware boundary-free |
| `label_noise_rate` | Proportion of labels corrupted |
| `spurious_strength` | Injected nuisance-marker intensity |
| `spurious_ood_eval` | Evaluate spurious feature under distribution shift |
| `reset_memory_on_sync` | Deliberately destructive private-memory reset |
| `feedback_threshold` | Adaptive participation trigger threshold |
| `feedback_gain` | Adaptive participation adjustment gain |

## Method distribution

| Local method | CIFAR-10 jobs | CIFAR-100 jobs |
|---|---:|---:|
| `agem` | 34 | 34 |
| `der` | 2 | 2 |
| `derpp` | 4 | 4 |
| `er_ace` | 2 | 2 |
| `ewc` | 3 | 3 |
| `gem` | 83 | 83 |
| `lwf` | 2 | 2 |
| `mas` | 2 | 2 |
| `replay` | 8 | 8 |
| `sgd` | 282 | 284 |
| `si` | 1 | 1 |

## Data-scenario distribution

| Scenario | CIFAR-10 jobs | CIFAR-100 jobs |
|---|---:|---:|
| `asynchronous` | 18 | 18 |
| `disjoint` | 119 | 119 |
| `domain` | 5 | 5 |
| `gradual_drift` | 2 | 2 |
| `iid` | 2 | 2 |
| `mixed` | 7 | 7 |
| `non_iid` | 6 | 6 |
| `overlap` | 123 | 123 |
| `recurrence` | 56 | 58 |
| `related` | 84 | 84 |
| `shifted_recurrence` | 1 | 1 |

## Experiment families

### A: Participation and schedules

Client participation rates, matching round/update budgets, old/new coverage, history and timing.
CIFAR-10: **47 jobs**; CIFAR-100: **47 jobs**.

### B: Old/new related classes

Relatedness, gradient compatibility, class-dose effects and selecting clients with related new classes.
CIFAR-10: **48 jobs**; CIFAR-100: **48 jobs**.

### C: Within-round class overlap

Simultaneous overlapping labels, shared-class frequencies, sample duplication, joint selection.
CIFAR-10: **41 jobs**; CIFAR-100: **41 jobs**.

### D: Temporal recurrence

Earlier classes reappearing locally or across clients, including time delays and exclusion.
CIFAR-10: **39 jobs**; CIFAR-100: **41 jobs**.

### E: Interaction studies

Relatedness combined with overlap, recurrence, asynchrony and aggregation weighting.
CIFAR-10: **30 jobs**; CIFAR-100: **30 jobs**.

### F: Client selection

Random, deterministic, diversity, gradient, forgetting-aware, fairness and oracle-style policies.
CIFAR-10: **33 jobs**; CIFAR-100: **33 jobs**.

### G: Local continual learning

SGD, GEM, A-GEM, replay, EWC, ER-ACE-style, DER/DER++, SI, MAS and LwF.
CIFAR-10: **51 jobs**; CIFAR-100: **51 jobs**.

### H: Synchronization

Broadcast scope, memory/optimizer persistence, inactivity, rejoining, synchronization frequency.
CIFAR-10: **15 jobs**; CIFAR-100: **15 jobs**.

### I: Aggregation

FedAvg, uniform, server adaptive updates, class balancing, gradient conflict and retention diagnostics.
CIFAR-10: **20 jobs**; CIFAR-100: **20 jobs**.

### J: Data heterogeneity

IID/non-IID, label/quantity skew, domain shift, imbalanced classes, arrival/noise patterns.
CIFAR-10: **29 jobs**; CIFAR-100: **29 jobs**.

### K: Learning setting

Class-, task-, domain-incremental and boundary-free variants with distinct information assumptions.
CIFAR-10: **11 jobs**; CIFAR-100: **11 jobs**.

### L: Causal and mechanistic diagnostics

Leave/add-one, accepted-update subset effects, checkpoint forks, loss and representation probes.
CIFAR-10: **34 jobs**; CIFAR-100: **34 jobs**.

### M: Robustness and resource accounting

Dataset, architecture, client-count, compute, communication, replication and resource settings.
CIFAR-10: **17 jobs**; CIFAR-100: **17 jobs**.

### N: Proposed adaptive methods

Participation balancing and selection guided by relation, recurrence, GEM and feedback.
CIFAR-10: **8 jobs**; CIFAR-100: **8 jobs**.

## Complete 152-block scheduling cross-reference

Entries with 0 jobs have no separately scheduled run under that block ID. `partial_implementation` means a real but incomplete or approximate method/experiment, not research closure.

| Block | Planned study | Implementation ledger | CIFAR-10 jobs | CIFAR-100 jobs |
|---|---|---|---:|---:|
| `A01` | Participation rate | `implemented` | 19 | 19 |
| `A02` | Fixed versus resampled participants | `implemented` | 2 | 2 |
| `A03` | Equal communication rounds | `partial_implementation` | 2 | 2 |
| `A04` | Matched total client updates | `partial_implementation` | 2 | 2 |
| `A05` | Old/new class-holder participation | `partial_implementation` | 3 | 3 |
| `A06` | Stage-dependent participation schedule | `partial_implementation` | 2 | 2 |
| `A07` | Class diversity at equal participation | `partial_implementation` | 2 | 2 |
| `A08` | Client redundancy with matched class coverage | `partial_implementation` | 2 | 2 |
| `A09` | Short frequent versus long rare participation | `partial_implementation` | 2 | 2 |
| `A10` | Monotonically changing participation | `implemented` | 8 | 8 |
| `A11` | Equal lifetime frequency, different timing | `partial_implementation` | 3 | 3 |
| `A12` | Equal-plasticity forgetting comparison | `not_implemented` | 0 | 0 |
| `B01` | Related vs unrelated new classes | `partial_implementation` | 2 | 2 |
| `B02` | Select/exclude related-class client | `partial_implementation` | 21 | 21 |
| `B03` | Relationship strength | `partial_implementation` | 2 | 2 |
| `B04` | Gradient alignment classes | `partial_implementation` | 2 | 2 |
| `B05` | One-to-one related classes | `partial_implementation` | 2 | 2 |
| `B06` | Many-to-one relations | `not_implemented` | 0 | 0 |
| `B07` | Related-class training dose | `partial_implementation` | 3 | 3 |
| `B08` | Repeated related-client selection | `partial_implementation` | 2 | 2 |
| `B09` | Mixed related/unrelated participants | `partial_implementation` | 2 | 2 |
| `B10` | Feature similarity intervention | `not_implemented` | 0 | 0 |
| `B11` | Local GEM x relationship | `partial_implementation` | 10 | 10 |
| `B12` | Per-old-class spillover | `partial_implementation` | 2 | 2 |
| `C01` | Overlap magnitude | `partial_implementation` | 4 | 4 |
| `C02` | Joint versus separate participation | `partial_implementation` | 15 | 15 |
| `C03` | Shared-class dominance | `partial_implementation` | 2 | 2 |
| `C04` | Same labels with disjoint examples | `partial_implementation` | 1 | 1 |
| `C05` | Exact sample overlap | `partial_implementation` | 3 | 3 |
| `C06` | Shared class across shifted domains | `partial_implementation` | 1 | 1 |
| `C07` | Gradient interactions of overlapping clients | `partial_implementation` | 1 | 1 |
| `C08` | Redundancy versus complementary coverage | `partial_implementation` | 3 | 3 |
| `C09` | Overlap x local GEM | `partial_implementation` | 9 | 9 |
| `C10` | Longitudinal overlap selection | `partial_implementation` | 2 | 2 |
| `D01` | Same-client recurrence | `partial_implementation` | 1 | 1 |
| `D02` | Cross-client recurrence | `partial_implementation` | 1 | 1 |
| `D03` | Related but new-label recurrence | `partial_implementation` | 2 | 2 |
| `D04` | Shifted-domain recurrence | `partial_implementation` | 2 | 2 |
| `D05` | Recurrence delay | `partial_implementation` | 1 | 3 |
| `D06` | Recurrence timing | `partial_implementation` | 3 | 3 |
| `D07` | Select/exclude recurring client | `partial_implementation` | 15 | 15 |
| `D08` | Periodic recurrence | `partial_implementation` | 2 | 2 |
| `D09` | Mixed old/new recurrence | `partial_implementation` | 2 | 2 |
| `D10` | Inactive-client recurrence | `partial_implementation` | 2 | 2 |
| `D11` | Delayed recurrent-client participation | `partial_implementation` | 2 | 2 |
| `D12` | Recurrence under GEM | `partial_implementation` | 6 | 6 |
| `E01` | Relatedness x overlap | `partial_implementation` | 3 | 3 |
| `E02` | Relatedness x recurrence | `partial_implementation` | 3 | 3 |
| `E03` | Overlap x recurrence | `partial_implementation` | 2 | 2 |
| `E04` | Triple relationship interaction | `partial_implementation` | 3 | 3 |
| `E05` | Mixed helpful and harmful client updates | `partial_implementation` | 2 | 2 |
| `E06` | Asynchronous old/new class relationships | `partial_implementation` | 3 | 3 |
| `E07` | Rare-class recurrence | `partial_implementation` | 2 | 2 |
| `E08` | Persistent vs temporary correlations | `partial_implementation` | 2 | 2 |
| `E09` | Relatedness strength x participation | `partial_implementation` | 6 | 6 |
| `E10` | Relatedness x aggregation weights | `partial_implementation` | 4 | 4 |
| `F01` | Uniform random | `implemented` | 2 | 2 |
| `F02` | Fixed subset | `implemented` | 2 | 2 |
| `F03` | Round-robin | `implemented` | 2 | 2 |
| `F04` | Least recently selected | `implemented` | 2 | 2 |
| `F05` | Class coverage maximizing | `implemented` | 2 | 2 |
| `F06` | Old-class coverage | `implemented` | 2 | 2 |
| `F07` | New-class coverage | `implemented` | 2 | 2 |
| `F08` | Balanced old and new coverage | `partial_implementation` | 2 | 2 |
| `F09` | Maximum label diversity | `partial_implementation` | 2 | 2 |
| `F10` | Maximum label overlap | `partial_implementation` | 2 | 2 |
| `F11` | Maximum feature similarity | `partial_implementation` | 1 | 1 |
| `F12` | Minimum feature similarity | `partial_implementation` | 1 | 1 |
| `F13` | Gradient alignment selection | `partial_implementation` | 1 | 1 |
| `F14` | Gradient conflict selection | `partial_implementation` | 1 | 1 |
| `F15` | Forgetting-aware selection | `partial_implementation` | 1 | 1 |
| `F16` | Recurrence-aware selection | `partial_implementation` | 2 | 2 |
| `F17` | Fairness-constrained selection | `partial_implementation` | 2 | 2 |
| `F18` | Randomized correlation-aware selection | `partial_implementation` | 2 | 2 |
| `F19` | History-aware adaptive selection | `partial_implementation` | 1 | 1 |
| `F20` | Oracle exhaustive selection | `partial_implementation` | 1 | 1 |
| `G01` | Local SGD | `implemented` | 3 | 3 |
| `G02` | Local GEM | `implemented` | 3 | 3 |
| `G03` | Local A-GEM | `implemented` | 3 | 3 |
| `G04` | Experience replay | `implemented` | 3 | 3 |
| `G05` | ER-ACE | `partial_implementation` | 2 | 2 |
| `G06` | DER and DER++ | `partial_implementation` | 4 | 4 |
| `G07` | EWC | `partial_implementation` | 3 | 3 |
| `G08` | Synaptic intelligence | `partial_implementation` | 2 | 2 |
| `G09` | MAS | `partial_implementation` | 2 | 2 |
| `G10` | Local distillation | `partial_implementation` | 2 | 2 |
| `G11` | Replay vs no replay | `partial_implementation` | 2 | 2 |
| `G12` | Memory capacity | `partial_implementation` | 9 | 9 |
| `G13` | Memory selection algorithms | `partial_implementation` | 2 | 2 |
| `G14` | GEM per-group constraints | `partial_implementation` | 3 | 3 |
| `G15` | Proximal regularization plus GEM | `partial_implementation` | 6 | 6 |
| `G16` | Heterogeneous local algorithms | `partial_implementation` | 2 | 2 |
| `H01` | Active-only synchronization | `implemented` | 2 | 2 |
| `H02` | All-client synchronization | `implemented` | 2 | 2 |
| `H03` | Sync when rejoining | `partial_implementation` | 2 | 2 |
| `H04` | Optimizer state after broadcast | `partial_implementation` | 2 | 2 |
| `H05` | Private memory persistence | `partial_implementation` | 2 | 2 |
| `H06` | Synchronization frequency | `partial_implementation` | 3 | 3 |
| `H07` | Long inactive gap | `partial_implementation` | 2 | 2 |
| `H08` | Local personalization | `not_implemented` | 0 | 0 |
| `I01` | Sample-weighted FedAvg | `implemented` | 1 | 1 |
| `I02` | Uniform aggregation | `implemented` | 1 | 1 |
| `I03` | Class-balanced aggregation | `partial_implementation` | 2 | 2 |
| `I04` | Correlation-aware aggregation | `partial_implementation` | 2 | 2 |
| `I05` | Old-retention aggregation | `partial_implementation` | 2 | 2 |
| `I06` | Server momentum/adaptive optimizer | `partial_implementation` | 8 | 8 |
| `I07` | Gradient conflict correction | `partial_implementation` | 2 | 2 |
| `I08` | Participation-probability correction | `partial_implementation` | 2 | 2 |
| `J01` | IID vs non-IID | `implemented` | 2 | 2 |
| `J02` | Label distribution skew | `partial_implementation` | 2 | 2 |
| `J03` | Client sample-count skew | `partial_implementation` | 2 | 2 |
| `J04` | Feature distribution skew | `partial_implementation` | 2 | 2 |
| `J05` | Class imbalance | `partial_implementation` | 2 | 2 |
| `J06` | Dependent versus independent client streams | `partial_implementation` | 2 | 2 |
| `J07` | Unequal arrival rates | `partial_implementation` | 2 | 2 |
| `J08` | Inactive client data retention | `partial_implementation` | 6 | 6 |
| `J09` | Recurrence timing distribution | `partial_implementation` | 3 | 3 |
| `J10` | Spurious correlations | `partial_implementation` | 2 | 2 |
| `J11` | Label noise | `partial_implementation` | 2 | 2 |
| `J12` | Changing correlation structure | `partial_implementation` | 2 | 2 |
| `K01` | Class-incremental learning | `implemented` | 1 | 1 |
| `K02` | Task-incremental learning | `partial_implementation` | 2 | 2 |
| `K03` | Domain-incremental learning | `partial_implementation` | 2 | 2 |
| `K04` | Task-free learning | `partial_implementation` | 2 | 2 |
| `K05` | Fixed versus expanding classifier | `not_implemented` | 0 | 0 |
| `K06` | Known versus hidden experience boundaries | `partial_implementation` | 2 | 2 |
| `K07` | Asynchronous versus synchronized transitions | `partial_implementation` | 2 | 2 |
| `K08` | Closed-world versus open-world class arrival | `not_implemented` | 0 | 0 |
| `L01` | Leave-one-client-out aggregation | `implemented` | 6 | 6 |
| `L02` | Add-one-client aggregation | `partial_implementation` | 18 | 18 |
| `L03` | Pairwise client interactions | `partial_implementation` | 1 | 1 |
| `L04` | Exhaustive participating subset enumeration | `partial_implementation` | 1 | 1 |
| `L05` | Pre/post local historical loss | `partial_implementation` | 1 | 1 |
| `L06` | Pre/post aggregation loss | `partial_implementation` | 1 | 1 |
| `L07` | Gradient interference analysis | `partial_implementation` | 1 | 1 |
| `L08` | Representation drift | `partial_implementation` | 2 | 2 |
| `L09` | Correlation intervention | `partial_implementation` | 2 | 2 |
| `L10` | Historical contribution attribution | `partial_implementation` | 1 | 1 |
| `M01` | Dataset generalization | `partial_implementation` | 1 | 1 |
| `M02` | Model architecture | `implemented` | 3 | 3 |
| `M03` | Client population size | `partial_implementation` | 3 | 3 |
| `M04` | Local optimization intensity | `partial_implementation` | 3 | 3 |
| `M05` | Randomness robustness | `partial_implementation` | 3 | 3 |
| `M06` | Communication efficiency | `partial_implementation` | 2 | 2 |
| `M07` | Computational efficiency | `partial_implementation` | 2 | 2 |
| `M08` | Privacy/resource restrictions | `not_implemented` | 0 | 0 |
| `N01` | Correlation-aware participation | `partial_implementation` | 1 | 1 |
| `N02` | Class-participation balancing | `partial_implementation` | 2 | 2 |
| `N03` | Recurrence-aware scheduling | `partial_implementation` | 1 | 1 |
| `N04` | GEM-informed participation | `partial_implementation` | 2 | 2 |
| `N05` | Adaptive participation rate | `partial_implementation` | 2 | 2 |
| `N06` | Joint selection and aggregation | `not_implemented` | 0 | 0 |

## Scientific limits and responsible reading

- The major local and federated mechanisms include GEM, A-GEM, SGD, experience replay, simplified EWC/ER-ACE/DER/MAS/LwF, SI, FedAvg, server adaptive variants, several selection policies and stream controls. Some are **adaptations** rather than verified exact original-paper reproductions.
- Some data-relationship selectors and historical-retention diagnostics rely on simulator-privileged class metadata, gradients or test-history performance; they are **oracle diagnostics**, not privacy-preserving deployable algorithms.
- No planned entry proves a causal benefit unless its comparison arms control confounding factors. Different participation fractions, experiences, class exposures and optimizer steps are not automatically compute- or plasticity-matched.
- `boundary_free` is a class-aware hidden-boundary setting, not unsupervised open-world class discovery. Domain/task-ID evaluation follows different conditions from global class-IL.
- Exhaustive subset diagnostics on **already computed accepted updates** do not amount to retraining every eligible client subset; targeted full-state checkpoint forks are separately configured.
- The **main** batch size is 512 for every CIFAR job. Final minibatches and some local memory/diagnostic operations can be smaller.
- Global accuracy includes all observed classes even when held by inactive clients. Accuracy drops caused by an expanded test population must not be mislabeled as forgetting.
- No official full CIFAR sweep, external baseline reproduction, confidence intervals or publication-quality results have been established by these plans.

## Result files and job failure handling

Each job writes into `results/<suite>/runs/<job_id>/` with an independent `config.json`, `scenario.pt`, `status.json`, `metrics.json`, checkpoints when scheduled, `model_final.pt` on completion, and `summary.json`. `suite_status.json` records failures. Job IDs are unique within a dataset, but the two datasets share some job names; **do not mix results roots**.

**Source documentation:** [experiment catalogue](EXPERIMENT_CATALOGUE.md), [implementation status](STATUS.md), and [technical specification](TECHNICAL_SPEC.md).
