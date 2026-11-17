# CIFAR100 — all 425 scheduled experiment jobs

> Generated from the frozen source manifest on the verified `main` branch. This is a complete inventory of **scheduled configurations**, not a record of completed training.

**Suite:** `cifar100` · **jobs:** **425** · **research block IDs:** **144/152** · **configured batch size:** **512**.

**Authoritative manifest:** [`configs/plans/cifar100_jobs.json`](../configs/plans/cifar100_jobs.json). Every job has its complete resolved configuration there.

## Reading the entries

- `#` is the original **one-based execution position**, even though entries below are grouped by scientific block for browsing.
- `Non-default parameters` lists **every** value differing from the suite-wide base record below, excluding `job_id` and `block_id` (shown separately). Applying these differences reconstructs the full manifest entry exactly.
- The main method/scenario/selection/aggregation/round fields also have their own columns for easy comparison. `Fork dependency` is mandatory for downstream counterfactual branch jobs.
- Different jobs sharing a block ID are related by scientific theme, **not automatically matched causal controls**. Consult the actual parameter differences before interpreting comparisons.

## Base configuration used for exact per-job differences

| Field | Value (JSON) |
|---|---|
| `dataset` | `"cifar100"` |
| `batch_size` | `512` |
| `num_clients` | `20` |
| `rounds_per_stage` | `20` |
| `epochs` | `1` |
| `participation` | `0.5` |
| `selector` | `"random"` |
| `method` | `"sgd"` |
| `aggregator` | `"fedavg"` |
| `scenario` | `"disjoint"` |
| `stages` | `10` |
| `memory_per_class` | `20` |
| `lr` | `0.03` |
| `model` | `"smallcnn"` |
| `seed` | `0` |
| `full_sync` | `false` |
| `relation` | `"natural"` |
| `recurrence_delay` | `1` |
| `recurrence_client_policy` | `"different"` |
| `overlap` | `0.5` |
| `dirichlet_alpha` | `0.5` |
| `target_client` | `-1` |
| `force_target` | `"none"` |
| `intervention_round` | `-1` |
| `gradient_diagnostics` | `false` |
| `causal_diagnostics` | `false` |
| `causal_max_clients` | `6` |
| `subset_diagnostics` | `false` |
| `prepost_diagnostics` | `false` |
| `representation_diagnostics` | `false` |
| `resume` | `true` |
| `checkpoint_every` | `0` |
| `participation_schedule` | `"constant"` |
| `pending_policy` | `"latest"` |
| `max_pending_samples` | `0` |
| `replay_ratio` | `1` |
| `der_alpha` | `0.5` |
| `der_beta` | `0.5` |
| `regularization_strength` | `10` |
| `fedprox_mu` | `0` |
| `server_lr` | `1` |
| `server_beta1` | `0.9` |
| `server_beta2` | `0.999` |
| `fedadam_epsilon` | `1e-8` |
| `memory_policy` | `"random_per_class"` |
| `local_optimizer` | `"sgd"` |
| `client_heterogeneity` | `false` |
| `fork_source_job` | `""` |
| `fork_checkpoint_round` | `0` |
| `class_subset` | `[]` |
| `class_order_override` | `[]` |
| `holders_override` | `{}` |
| `class_holder_weights` | `{}` |
| `class_sample_fraction` | `{}` |
| `participation_overrides` | `{}` |
| `arrival_lag_by_client` | `{}` |
| `recurrence_round_delay` | `0` |
| `recurrence_period` | `0` |
| `recurrence_repeats` | `1` |
| `sample_overlap_fraction` | `0` |
| `client_sample_skew` | `0` |
| `class_imbalance_power` | `0` |
| `stage_participation` | `[]` |
| `gem_grouping` | `"class"` |
| `optimizer_state_policy` | `"reset"` |
| `momentum` | `0` |
| `si_xi` | `0.1` |
| `selection_temperature` | `1` |
| `boundary_mode` | `"known"` |
| `learning_setting` | `"class_il"` |
| `label_noise_rate` | `0` |
| `spurious_strength` | `0` |
| `spurious_ood_eval` | `false` |
| `reset_memory_on_sync` | `false` |
| `feedback_threshold` | `0.25` |
| `feedback_gain` | `0.5` |

## Jobs by research family

| Family | Purpose | Jobs | Covered blocks |
|---|---|---:|---:|
| A | Participation and schedules | 47 | 11 |
| B | Old/new related classes | 48 | 10 |
| C | Within-round class overlap | 41 | 10 |
| D | Temporal recurrence | 41 | 12 |
| E | Interaction studies | 30 | 10 |
| F | Client selection | 33 | 20 |
| G | Local continual learning | 51 | 16 |
| H | Synchronization | 15 | 7 |
| I | Aggregation | 20 | 8 |
| J | Data heterogeneity | 29 | 12 |
| K | Learning setting | 11 | 6 |
| L | Causal and mechanistic diagnostics | 34 | 10 |
| M | Robustness and resource accounting | 17 | 7 |
| N | Proposed adaptive methods | 8 | 5 |

## Complete registered job inventory

### A. Participation and schedules

Client participation rates, matching round/update budgets, old/new coverage, history and timing.

#### A01 — Participation rate

**Implementation status:** `implemented` · **Scheduled jobs:** 19. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 1 | `A01_disjoint_sgd_full` | `cifar100` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1` | — |
| 2 | `A01_disjoint_sgd_random50` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 3 | `A01_disjoint_sgd_random25` | `cifar100` | `disjoint` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25` | — |
| 4 | `A01_disjoint_sgd_coverage50` | `cifar100` | `disjoint` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"` | — |
| 5 | `A01_disjoint_sgd_coverage25` | `cifar100` | `disjoint` | `sgd` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"` | — |
| 6 | `A01_disjoint_gem_full` | `cifar100` | `disjoint` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"` | — |
| 7 | `A01_disjoint_gem_random50` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 8 | `A01_disjoint_gem_random25` | `cifar100` | `disjoint` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"` | — |
| 9 | `A01_disjoint_gem_coverage50` | `cifar100` | `disjoint` | `gem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="gem"` | — |
| 10 | `A01_disjoint_gem_coverage25` | `cifar100` | `disjoint` | `gem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"` | — |
| 11 | `A01_disjoint_agem_full` | `cifar100` | `disjoint` | `agem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="agem"` | — |
| 12 | `A01_disjoint_agem_random50` | `cifar100` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"` | — |
| 13 | `A01_disjoint_agem_random25` | `cifar100` | `disjoint` | `agem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="agem"` | — |
| 14 | `A01_disjoint_agem_coverage50` | `cifar100` | `disjoint` | `agem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="agem"` | — |
| 15 | `A01_disjoint_agem_coverage25` | `cifar100` | `disjoint` | `agem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"` | — |
| 119 | `A01_participation_100` | `cifar100` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1` | — |
| 120 | `A01_participation_75` | `cifar100` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.75` | — |
| 121 | `A01_participation_50` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 122 | `A01_participation_25` | `cifar100` | `disjoint` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25` | — |


#### A02 — Fixed versus resampled participants

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 394 | `A02_controls_fixed_resampled_random` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 395 | `A02_controls_fixed_resampled_fixed` | `cifar100` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"` | — |


#### A03 — Equal communication rounds

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 279 | `A03_closure_roundmatched_100` | `cifar100` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1` | — |
| 280 | `A03_closure_roundmatched_50` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |


#### A04 — Matched total client updates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 281 | `A04_closure_clientupdates_full` | `cifar100` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 10 × 10 × 1 | `rounds_per_stage=10`; `participation=1` | — |
| 282 | `A04_closure_clientupdates_half` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |


#### A05 — Old/new class-holder participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 142 | `A05_extended_old_new_old` | `cifar100` | `overlap` | `sgd` | 0.5 | `old` | `fedavg` | 10 × 20 × 1 | `selector="old"`; `scenario="overlap"` | — |
| 143 | `A05_extended_old_new_new` | `cifar100` | `overlap` | `sgd` | 0.5 | `new` | `fedavg` | 10 × 20 × 1 | `selector="new"`; `scenario="overlap"` | — |
| 144 | `A05_extended_old_new_balanced` | `cifar100` | `overlap` | `sgd` | 0.5 | `balanced` | `fedavg` | 10 × 20 × 1 | `selector="balanced"`; `scenario="overlap"` | — |


#### A06 — Stage-dependent participation schedule

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 283 | `A06_closure_earlyfull` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `stages=2`; `participation_schedule="stage"`; `stage_participation=[1,0.25]` | — |
| 284 | `A06_closure_latefull` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `stages=2`; `participation_schedule="stage"`; `stage_participation=[0.25,1]` | — |


#### A07 — Class diversity at equal participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 285 | `A07_closure_diversity_coverage` | `cifar100` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 286 | `A07_closure_diversity_fixed` | `cifar100` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"`; `scenario="overlap"` | — |


#### A08 — Client redundancy with matched class coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 396 | `A08_controls_redundancy_coverage` | `cifar100` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 397 | `A08_controls_redundancy_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 10 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |


#### A09 — Short frequent versus long rare participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 287 | `A09_closure_frequency_fixed` | `cifar100` | `overlap` | `sgd` | 0.25 | `fixed` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="fixed"`; `scenario="overlap"` | — |
| 288 | `A09_closure_frequency_round_robin` | `cifar100` | `overlap` | `sgd` | 0.25 | `round_robin` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="round_robin"`; `scenario="overlap"` | — |


#### A10 — Monotonically changing participation

**Implementation status:** `implemented` · **Scheduled jobs:** 8. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 131 | `A10_extended_schedule_constant_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 132 | `A10_extended_schedule_constant_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 133 | `A10_extended_schedule_increasing_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `participation_schedule="increasing"` | — |
| 134 | `A10_extended_schedule_increasing_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `participation_schedule="increasing"` | — |
| 135 | `A10_extended_schedule_decreasing_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `participation_schedule="decreasing"` | — |
| 136 | `A10_extended_schedule_decreasing_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `participation_schedule="decreasing"` | — |
| 137 | `A10_extended_schedule_alternating_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `participation_schedule="alternating"` | — |
| 138 | `A10_extended_schedule_alternating_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `participation_schedule="alternating"` | — |


#### A11 — Equal lifetime frequency, different timing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 139 | `A11_extended_timing_fixed` | `cifar100` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"` | — |
| 140 | `A11_extended_timing_round_robin` | `cifar100` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"` | — |
| 141 | `A11_extended_timing_least_recent` | `cifar100` | `disjoint` | `sgd` | 0.5 | `least_recent` | `fedavg` | 10 × 20 × 1 | `selector="least_recent"` | — |


### B. Old/new related classes

Relatedness, gradient compatibility, class-dose effects and selecting clients with related new classes.

#### B01 — Related vs unrelated new classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 117 | `B01_class_relationship_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 118 | `B01_class_relationship_unrelated` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |


#### B02 — Select/exclude related-class client

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 21. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 31 | `B02_related_sgd_full` | `cifar100` | `related` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="related"`; `relation="related"` | — |
| 32 | `B02_related_sgd_random50` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 33 | `B02_related_sgd_random25` | `cifar100` | `related` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="related"`; `relation="related"` | — |
| 34 | `B02_related_sgd_coverage50` | `cifar100` | `related` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="related"`; `relation="related"` | — |
| 35 | `B02_related_sgd_coverage25` | `cifar100` | `related` | `sgd` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="related"`; `relation="related"` | — |
| 36 | `B02_related_gem_full` | `cifar100` | `related` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 37 | `B02_related_gem_random50` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 38 | `B02_related_gem_random25` | `cifar100` | `related` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 39 | `B02_related_gem_coverage50` | `cifar100` | `related` | `gem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 40 | `B02_related_gem_coverage25` | `cifar100` | `related` | `gem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 41 | `B02_related_agem_full` | `cifar100` | `related` | `agem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 42 | `B02_related_agem_random50` | `cifar100` | `related` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 43 | `B02_related_agem_random25` | `cifar100` | `related` | `agem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 44 | `B02_related_agem_coverage50` | `cifar100` | `related` | `agem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 45 | `B02_related_agem_coverage25` | `cifar100` | `related` | `agem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 111 | `B02_relationnatural_include` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `target_client=2`; `force_target="include"` | — |
| 112 | `B02_relationnatural_exclude` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `target_client=2`; `force_target="exclude"` | — |
| 113 | `B02_relationrelated_include` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"` | — |
| 114 | `B02_relationrelated_exclude` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"` | — |
| 115 | `B02_relationunrelated_include` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `target_client=2`; `force_target="include"` | — |
| 116 | `B02_relationunrelated_exclude` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `target_client=2`; `force_target="exclude"` | — |


#### B03 — Relationship strength

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 398 | `B03_controls_coarse_relationship_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `gradient_diagnostics=true` | — |
| 401 | `B03_controls_coarse_relationship_unrelated` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `gradient_diagnostics=true` | — |


#### B04 — Gradient alignment classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 399 | `B04_controls_gradient_relationship_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `gradient_diagnostics=true` | — |
| 402 | `B04_controls_gradient_relationship_unrelated` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `gradient_diagnostics=true` | — |


#### B05 — One-to-one related classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 289 | `B05_closure_exact_old(0, 1)_new(12, 13)_include` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="include"`; `intervention_round=20`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |
| 291 | `B05_closure_exact_old(0, 1)_new(12, 13)_exclude` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="exclude"`; `intervention_round=20`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### B07 — Related-class training dose

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 293 | `B07_closure_related_dose_25` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}`; `class_sample_fraction={"12":0.25,"13":0.25}` | — |
| 294 | `B07_closure_related_dose_50` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}`; `class_sample_fraction={"12":0.5,"13":0.5}` | — |
| 295 | `B07_closure_related_dose_100` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}`; `class_sample_fraction={"12":1,"13":1}` | — |


#### B08 — Repeated related-client selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 296 | `B08_closure_related_participation_once` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="include"`; `intervention_round=20`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |
| 297 | `B08_closure_related_participation_always` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="include"`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### B09 — Mixed related/unrelated participants

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 298 | `B09_closure_mixed_related_related` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}`; `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 299 | `B09_closure_mixed_related_unrelated` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}`; `participation_overrides={"20":[0,1,2,4,5]}` | — |


#### B11 — Local GEM x relationship

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 10. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 145 | `B11_extended_relation_related_local_sgd` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 146 | `B11_extended_relation_related_local_gem` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 147 | `B11_extended_relation_related_local_agem` | `cifar100` | `related` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 148 | `B11_extended_relation_related_local_replay` | `cifar100` | `related` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="related"` | — |
| 149 | `B11_extended_relation_related_local_derpp` | `cifar100` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="related"` | — |
| 150 | `B11_extended_relation_unrelated_local_sgd` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |
| 151 | `B11_extended_relation_unrelated_local_gem` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="unrelated"` | — |
| 152 | `B11_extended_relation_unrelated_local_agem` | `cifar100` | `related` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="unrelated"` | — |
| 153 | `B11_extended_relation_unrelated_local_replay` | `cifar100` | `related` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="unrelated"` | — |
| 154 | `B11_extended_relation_unrelated_local_derpp` | `cifar100` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="unrelated"` | — |


#### B12 — Per-old-class spillover

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 290 | `B12_closure_class_spillover_include` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="include"`; `intervention_round=20`; `prepost_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |
| 292 | `B12_closure_class_spillover_exclude` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `stages=2`; `target_client=3`; `force_target="exclude"`; `intervention_round=20`; `prepost_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


### C. Within-round class overlap

Simultaneous overlapping labels, shared-class frequencies, sample duplication, joint selection.

#### C01 — Overlap magnitude

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 93 | `C01_overlap_0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=0` | — |
| 94 | `C01_overlap_33` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=0.33` | — |
| 95 | `C01_overlap_67` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=0.67` | — |
| 96 | `C01_overlap_100` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |


#### C02 — Joint versus separate participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 15. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 16 | `C02_overlap_sgd_full` | `cifar100` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="overlap"` | — |
| 17 | `C02_overlap_sgd_random50` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 18 | `C02_overlap_sgd_random25` | `cifar100` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="overlap"` | — |
| 19 | `C02_overlap_sgd_coverage50` | `cifar100` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 20 | `C02_overlap_sgd_coverage25` | `cifar100` | `overlap` | `sgd` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="overlap"` | — |
| 21 | `C02_overlap_gem_full` | `cifar100` | `overlap` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"` | — |
| 22 | `C02_overlap_gem_random50` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 23 | `C02_overlap_gem_random25` | `cifar100` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"` | — |
| 24 | `C02_overlap_gem_coverage50` | `cifar100` | `overlap` | `gem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |
| 25 | `C02_overlap_gem_coverage25` | `cifar100` | `overlap` | `gem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |
| 26 | `C02_overlap_agem_full` | `cifar100` | `overlap` | `agem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="overlap"` | — |
| 27 | `C02_overlap_agem_random50` | `cifar100` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="overlap"` | — |
| 28 | `C02_overlap_agem_random25` | `cifar100` | `overlap` | `agem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="overlap"` | — |
| 29 | `C02_overlap_agem_coverage50` | `cifar100` | `overlap` | `agem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="overlap"` | — |
| 30 | `C02_overlap_agem_coverage25` | `cifar100` | `overlap` | `agem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="overlap"` | — |


#### C03 — Shared-class dominance

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 300 | `C03_closure_balanced` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `holders_override={"0":[0,1]}`; `class_holder_weights={"0":[0.5,0.5]}` | — |
| 301 | `C03_closure_dominant` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `holders_override={"0":[0,1]}`; `class_holder_weights={"0":[0.9,0.1]}` | — |


#### C04 — Same labels with disjoint examples

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 302 | `C04_closure_same_label_disjoint_images` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |


#### C05 — Exact sample overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 303 | `C05_closure_raw_sample_overlap_0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |
| 304 | `C05_closure_raw_sample_overlap_25` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `sample_overlap_fraction=0.25` | — |
| 305 | `C05_closure_raw_sample_overlap_50` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `sample_overlap_fraction=0.5` | — |


#### C06 — Shared class across shifted domains

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 306 | `C06_closure_client_domain_shared_class` | `cifar100` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=2`; `learning_setting="domain_il"` | — |


#### C07 — Gradient interactions of overlapping clients

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 307 | `C07_closure_overlapping_client_update_cosines` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `gradient_diagnostics=true` | — |


#### C08 — Redundancy versus complementary coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 164 | `C08_extended_overlap_selector_diversity` | `cifar100` | `overlap` | `sgd` | 0.5 | `diversity` | `fedavg` | 10 × 20 × 1 | `selector="diversity"`; `scenario="overlap"` | — |
| 165 | `C08_extended_overlap_selector_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 10 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |
| 166 | `C08_extended_overlap_selector_coverage` | `cifar100` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |


#### C09 — Overlap x local GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 9. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 155 | `C09_extended_overlap_0_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=0` | — |
| 156 | `C09_extended_overlap_0_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `overlap=0` | — |
| 157 | `C09_extended_overlap_0_agem` | `cifar100` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="overlap"`; `overlap=0` | — |
| 158 | `C09_extended_overlap_50_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 159 | `C09_extended_overlap_50_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 160 | `C09_extended_overlap_50_agem` | `cifar100` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="overlap"` | — |
| 161 | `C09_extended_overlap_100_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |
| 162 | `C09_extended_overlap_100_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `overlap=1` | — |
| 163 | `C09_extended_overlap_100_agem` | `cifar100` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="overlap"`; `overlap=1` | — |


#### C10 — Longitudinal overlap selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 308 | `C10_closure_longitudinal_fixed` | `cifar100` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"`; `scenario="overlap"`; `overlap=1` | — |
| 309 | `C10_closure_longitudinal_round_robin` | `cifar100` | `overlap` | `sgd` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"`; `scenario="overlap"`; `overlap=1` | — |


### D. Temporal recurrence

Earlier classes reappearing locally or across clients, including time delays and exclusion.

#### D01 — Same-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 97 | `D01_recurrence_same` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `recurrence_client_policy="same"` | — |


#### D02 — Cross-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 98 | `D02_recurrence_different` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"` | — |


#### D03 — Related but new-label recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 404 | `D03_controls_related_new_labels_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 405 | `D03_controls_related_new_labels_recurrence` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `relation="related"` | — |


#### D04 — Shifted-domain recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 406 | `D04_controls_domain_shift_recurrence` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2` | — |
| 407 | `D04_controls_domain_shift_shifted_recurrence` | `cifar100` | `shifted_recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="shifted_recurrence"`; `stages=2` | — |


#### D05 — Recurrence delay

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 99 | `D05_recurrence_stages2_delay1` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2` | — |
| 100 | `D05_recurrence_stages3_delay1` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 3 × 20 × 1 | `scenario="recurrence"`; `stages=3` | — |
| 101 | `D05_recurrence_stages3_delay2` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 3 × 20 × 1 | `scenario="recurrence"`; `stages=3`; `recurrence_delay=2` | — |


#### D06 — Recurrence timing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 310 | `D06_closure_lag_1_rounds` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=1` | — |
| 311 | `D06_closure_lag_5_rounds` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=5` | — |
| 312 | `D06_closure_lag_15_rounds` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=15` | — |


#### D07 — Select/exclude recurring client

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 15. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 46 | `D07_recurrence_sgd_full` | `cifar100` | `recurrence` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="recurrence"` | — |
| 47 | `D07_recurrence_sgd_random50` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"` | — |
| 48 | `D07_recurrence_sgd_random25` | `cifar100` | `recurrence` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="recurrence"` | — |
| 49 | `D07_recurrence_sgd_coverage50` | `cifar100` | `recurrence` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="recurrence"` | — |
| 50 | `D07_recurrence_sgd_coverage25` | `cifar100` | `recurrence` | `sgd` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="recurrence"` | — |
| 51 | `D07_recurrence_gem_full` | `cifar100` | `recurrence` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="recurrence"` | — |
| 52 | `D07_recurrence_gem_random50` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"` | — |
| 53 | `D07_recurrence_gem_random25` | `cifar100` | `recurrence` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="recurrence"` | — |
| 54 | `D07_recurrence_gem_coverage50` | `cifar100` | `recurrence` | `gem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="recurrence"` | — |
| 55 | `D07_recurrence_gem_coverage25` | `cifar100` | `recurrence` | `gem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="recurrence"` | — |
| 56 | `D07_recurrence_agem_full` | `cifar100` | `recurrence` | `agem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="recurrence"` | — |
| 57 | `D07_recurrence_agem_random50` | `cifar100` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="recurrence"` | — |
| 58 | `D07_recurrence_agem_random25` | `cifar100` | `recurrence` | `agem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="recurrence"` | — |
| 59 | `D07_recurrence_agem_coverage50` | `cifar100` | `recurrence` | `agem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="recurrence"` | — |
| 60 | `D07_recurrence_agem_coverage25` | `cifar100` | `recurrence` | `agem` | 0.25 | `coverage` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="recurrence"` | — |


#### D08 — Periodic recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 313 | `D08_closure_periodic_5` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=3`; `recurrence_period=5`; `recurrence_repeats=3` | — |
| 314 | `D08_closure_periodic_10` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=3`; `recurrence_period=10`; `recurrence_repeats=3` | — |


#### D09 — Mixed old/new recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 315 | `D09_closure_mixed_old_new_sgd` | `cifar100` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="mixed"` | — |
| 316 | `D09_closure_mixed_old_new_gem` | `cifar100` | `mixed` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="mixed"` | — |


#### D10 — Inactive-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 317 | `D10_closure_inactive_recurrence_include` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `target_client=3`; `force_target="include"`; `intervention_round=20` | — |
| 318 | `D10_closure_inactive_recurrence_exclude` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `target_client=3`; `force_target="exclude"`; `intervention_round=20` | — |


#### D11 — Delayed recurrent-client participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 319 | `D11_closure_recurrence_participation_immediate` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=10`; `scenario="recurrence"`; `stages=2`; `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 320 | `D11_closure_recurrence_participation_delayed` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=10`; `scenario="recurrence"`; `stages=2`; `participation_overrides={"20":[0,1,2,4,5],"21":[0,1,2,4,5],"22":[0,1,2,4,5],"23":[0,1,2,4,5],"24":[0,1,2,4,5],"25":[0,1,2,3,4]}` | — |


#### D12 — Recurrence under GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 167 | `D12_extended_recurrence_same_sgd` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 168 | `D12_extended_recurrence_same_gem` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 169 | `D12_extended_recurrence_same_agem` | `cifar100` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 170 | `D12_extended_recurrence_different_sgd` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"` | — |
| 171 | `D12_extended_recurrence_different_gem` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"` | — |
| 172 | `D12_extended_recurrence_different_agem` | `cifar100` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="recurrence"` | — |


### E. Interaction studies

Relatedness combined with overlap, recurrence, asynchrony and aggregation weighting.

#### E01 — Relatedness x overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 182 | `E01_extended_overlap_related_p25` | `cifar100` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="overlap"`; `relation="related"` | — |
| 183 | `E01_extended_overlap_related_p50` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `relation="related"` | — |
| 184 | `E01_extended_overlap_related_p100` | `cifar100` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="overlap"`; `relation="related"` | — |


#### E02 — Relatedness x recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 176 | `E02_extended_recurrence_related_p25` | `cifar100` | `recurrence` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="recurrence"`; `relation="related"` | — |
| 177 | `E02_extended_recurrence_related_p50` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `relation="related"` | — |
| 178 | `E02_extended_recurrence_related_p100` | `cifar100` | `recurrence` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="recurrence"`; `relation="related"` | — |


#### E03 — Overlap x recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 321 | `E03_closure_overlap_recurrence_sgd` | `cifar100` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="mixed"` | — |
| 322 | `E03_closure_overlap_recurrence_gem` | `cifar100` | `mixed` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="mixed"` | — |


#### E04 — Triple relationship interaction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 173 | `E04_extended_mixed_related_p25` | `cifar100` | `mixed` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="mixed"`; `relation="related"` | — |
| 174 | `E04_extended_mixed_related_p50` | `cifar100` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="mixed"`; `relation="related"` | — |
| 175 | `E04_extended_mixed_related_p100` | `cifar100` | `mixed` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="mixed"`; `relation="related"` | — |


#### E05 — Mixed helpful and harmful client updates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 323 | `E05_closure_helpful_harmful_gradient_align` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `gradient_align` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="gradient_align"`; `stages=2`; `gradient_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |
| 324 | `E05_closure_helpful_harmful_gradient_conflict` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `gradient_conflict` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="gradient_conflict"`; `stages=2`; `gradient_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### E06 — Asynchronous old/new class relationships

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 179 | `E06_extended_asynchronous_related_p25` | `cifar100` | `asynchronous` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="asynchronous"`; `relation="related"` | — |
| 180 | `E06_extended_asynchronous_related_p50` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `relation="related"` | — |
| 181 | `E06_extended_asynchronous_related_p100` | `cifar100` | `asynchronous` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="asynchronous"`; `relation="related"` | — |


#### E07 — Rare-class recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 325 | `E07_closure_rare_recurrence_10` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `class_sample_fraction={"0":0.1}` | — |
| 326 | `E07_closure_rare_recurrence_50` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `class_sample_fraction={"0":0.5}` | — |


#### E08 — Persistent vs temporary correlations

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 408 | `E08_controls_client_correlation_persistent` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"` | — |
| 409 | `E08_controls_client_correlation_temporary` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":10,"1":10}` | — |


#### E09 — Relatedness strength x participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 125 | `E09_related_p25` | `cifar100` | `related` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="related"`; `relation="related"` | — |
| 126 | `E09_overlap_p25` | `cifar100` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `scenario="overlap"` | — |
| 127 | `E09_related_p50` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 128 | `E09_overlap_p50` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 129 | `E09_related_p100` | `cifar100` | `related` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="related"`; `relation="related"` | — |
| 130 | `E09_overlap_p100` | `cifar100` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `scenario="overlap"` | — |


#### E10 — Relatedness x aggregation weights

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 185 | `E10_extended_relation_aggregation_fedavg` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 186 | `E10_extended_relation_aggregation_uniform` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="related"`; `relation="related"` | — |
| 187 | `E10_extended_relation_aggregation_fedadam` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedadam` | 10 × 20 × 1 | `aggregator="fedadam"`; `scenario="related"`; `relation="related"`; `server_lr=0.001` | — |
| 188 | `E10_extended_relation_aggregation_fedavgm` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavgm` | 10 × 20 × 1 | `aggregator="fedavgm"`; `scenario="related"`; `relation="related"` | — |


### F. Client selection

Random, deterministic, diversity, gradient, forgetting-aware, fairness and oracle-style policies.

#### F01 — Uniform random

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 61 | `F01_random_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 69 | `F01_random_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |


#### F02 — Fixed subset

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 62 | `F02_fixed_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"`; `scenario="overlap"` | — |
| 70 | `F02_fixed_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"`; `method="gem"`; `scenario="overlap"` | — |


#### F03 — Round-robin

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 63 | `F03_round_robin_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"`; `scenario="overlap"` | — |
| 71 | `F03_round_robin_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"`; `method="gem"`; `scenario="overlap"` | — |


#### F04 — Least recently selected

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 64 | `F04_least_recent_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `least_recent` | `fedavg` | 10 × 20 × 1 | `selector="least_recent"`; `scenario="overlap"` | — |
| 72 | `F04_least_recent_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `least_recent` | `fedavg` | 10 × 20 × 1 | `selector="least_recent"`; `method="gem"`; `scenario="overlap"` | — |


#### F05 — Class coverage maximizing

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 68 | `F05_coverage_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 76 | `F05_coverage_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `coverage` | `fedavg` | 10 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |


#### F06 — Old-class coverage

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 65 | `F06_old_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `old` | `fedavg` | 10 × 20 × 1 | `selector="old"`; `scenario="overlap"` | — |
| 73 | `F06_old_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `old` | `fedavg` | 10 × 20 × 1 | `selector="old"`; `method="gem"`; `scenario="overlap"` | — |


#### F07 — New-class coverage

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 66 | `F07_new_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `new` | `fedavg` | 10 × 20 × 1 | `selector="new"`; `scenario="overlap"` | — |
| 74 | `F07_new_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `new` | `fedavg` | 10 × 20 × 1 | `selector="new"`; `method="gem"`; `scenario="overlap"` | — |


#### F08 — Balanced old and new coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 189 | `F08_extended_balanced_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `balanced` | `fedavg` | 10 × 20 × 1 | `selector="balanced"`; `scenario="overlap"` | — |
| 190 | `F08_extended_balanced_related` | `cifar100` | `related` | `sgd` | 0.5 | `balanced` | `fedavg` | 10 × 20 × 1 | `selector="balanced"`; `scenario="related"`; `relation="related"` | — |


#### F09 — Maximum label diversity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 191 | `F09_extended_diversity_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `diversity` | `fedavg` | 10 × 20 × 1 | `selector="diversity"`; `scenario="overlap"` | — |
| 192 | `F09_extended_diversity_related` | `cifar100` | `related` | `sgd` | 0.5 | `diversity` | `fedavg` | 10 × 20 × 1 | `selector="diversity"`; `scenario="related"`; `relation="related"` | — |


#### F10 — Maximum label overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 67 | `F10_overlap_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 10 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |
| 75 | `F10_overlap_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `overlap` | `fedavg` | 10 × 20 × 1 | `selector="overlap"`; `method="gem"`; `scenario="overlap"` | — |


#### F11 — Maximum feature similarity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 327 | `F11_closure_feature_max` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `feature_max` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="feature_max"`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### F12 — Minimum feature similarity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 328 | `F12_closure_feature_min` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `feature_min` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="feature_min"`; `stages=2`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### F13 — Gradient alignment selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 329 | `F13_closure_gradient_align` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `gradient_align` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="gradient_align"`; `stages=2`; `gradient_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### F14 — Gradient conflict selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 330 | `F14_closure_gradient_conflict` | `cifar100_20` | `disjoint` | `sgd` | 0.5 | `gradient_conflict` | `fedavg` | 2 × 20 × 1 | `dataset="cifar100_20"`; `num_clients=10`; `selector="gradient_conflict"`; `stages=2`; `gradient_diagnostics=true`; `holders_override={"0":[0],"1":[0],"12":[3],"13":[3]}` | — |


#### F15 — Forgetting-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 331 | `F15_closure_forgetting` | `cifar100` | `disjoint` | `sgd` | 0.5 | `forgetting` | `fedavg` | 10 × 20 × 1 | `selector="forgetting"` | — |


#### F16 — Recurrence-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 410 | `F16_controls_recurrence_aware_random` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"` | — |
| 411 | `F16_controls_recurrence_aware_recurrence` | `cifar100` | `recurrence` | `sgd` | 0.5 | `recurrence` | `fedavg` | 10 × 20 × 1 | `selector="recurrence"`; `scenario="recurrence"` | — |


#### F17 — Fairness-constrained selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 193 | `F17_extended_fairness_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `fairness` | `fedavg` | 10 × 20 × 1 | `selector="fairness"`; `scenario="overlap"` | — |
| 194 | `F17_extended_fairness_related` | `cifar100` | `related` | `sgd` | 0.5 | `fairness` | `fedavg` | 10 × 20 × 1 | `selector="fairness"`; `scenario="related"`; `relation="related"` | — |


#### F18 — Randomized correlation-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 195 | `F18_extended_stochastic_related_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `stochastic_related` | `fedavg` | 10 × 20 × 1 | `selector="stochastic_related"`; `scenario="overlap"` | — |
| 196 | `F18_extended_stochastic_related_related` | `cifar100` | `related` | `sgd` | 0.5 | `stochastic_related` | `fedavg` | 10 × 20 × 1 | `selector="stochastic_related"`; `scenario="related"`; `relation="related"` | — |


#### F19 — History-aware adaptive selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 332 | `F19_closure_history_adaptive` | `cifar100` | `disjoint` | `sgd` | 0.5 | `history_adaptive` | `fedavg` | 10 × 20 × 1 | `selector="history_adaptive"` | — |


#### F20 — Oracle exhaustive selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 333 | `F20_closure_exhaustive_subsets` | `cifar100` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 10 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


### G. Local continual learning

SGD, GEM, A-GEM, replay, EWC, ER-ACE-style, DER/DER++, SI, MAS and LwF.

#### G01 — Local SGD

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 77 | `G01_local_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 199 | `G01_extended_local_sgd_disjoint` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 200 | `G01_extended_local_sgd_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |


#### G02 — Local GEM

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 78 | `G02_local_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 213 | `G02_extended_local_gem_disjoint` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 214 | `G02_extended_local_gem_related` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |


#### G03 — Local A-GEM

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 79 | `G03_local_agem` | `cifar100` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"` | — |
| 215 | `G03_extended_local_agem_disjoint` | `cifar100` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"` | — |
| 216 | `G03_extended_local_agem_related` | `cifar100` | `related` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |


#### G04 — Experience replay

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 80 | `G04_local_replay` | `cifar100` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"` | — |
| 201 | `G04_extended_local_replay_disjoint` | `cifar100` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"` | — |
| 202 | `G04_extended_local_replay_related` | `cifar100` | `related` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="related"` | — |


#### G05 — ER-ACE

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 203 | `G05_extended_local_er_ace_disjoint` | `cifar100` | `disjoint` | `er_ace` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="er_ace"` | — |
| 204 | `G05_extended_local_er_ace_related` | `cifar100` | `related` | `er_ace` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="er_ace"`; `scenario="related"`; `relation="related"` | — |


#### G06 — DER and DER++

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 205 | `G06_extended_local_der_disjoint` | `cifar100` | `disjoint` | `der` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="der"` | — |
| 206 | `G06_extended_local_der_related` | `cifar100` | `related` | `der` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="der"`; `scenario="related"`; `relation="related"` | — |
| 207 | `G06_extended_local_derpp_disjoint` | `cifar100` | `disjoint` | `derpp` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="derpp"` | — |
| 208 | `G06_extended_local_derpp_related` | `cifar100` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="related"` | — |


#### G07 — EWC

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 81 | `G07_local_ewc` | `cifar100` | `disjoint` | `ewc` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="ewc"` | — |
| 217 | `G07_extended_local_ewc_disjoint` | `cifar100` | `disjoint` | `ewc` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="ewc"` | — |
| 218 | `G07_extended_local_ewc_related` | `cifar100` | `related` | `ewc` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="ewc"`; `scenario="related"`; `relation="related"` | — |


#### G08 — Synaptic intelligence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 334 | `G08_closure_synaptic_intelligence_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 335 | `G08_closure_synaptic_intelligence_si` | `cifar100` | `disjoint` | `si` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="si"` | — |


#### G09 — MAS

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 209 | `G09_extended_local_mas_disjoint` | `cifar100` | `disjoint` | `mas` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="mas"` | — |
| 210 | `G09_extended_local_mas_related` | `cifar100` | `related` | `mas` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="mas"`; `scenario="related"`; `relation="related"` | — |


#### G10 — Local distillation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 211 | `G10_extended_local_lwf_disjoint` | `cifar100` | `disjoint` | `lwf` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="lwf"` | — |
| 212 | `G10_extended_local_lwf_related` | `cifar100` | `related` | `lwf` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="lwf"`; `scenario="related"`; `relation="related"` | — |


#### G11 — Replay vs no replay

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 339 | `G11_closure_memory_comparison_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 340 | `G11_closure_memory_comparison_replay` | `cifar100` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"` | — |


#### G12 — Memory capacity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 9. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 102 | `G12_memory0_participation100` | `cifar100` | `overlap` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 103 | `G12_memory10_participation100` | `cifar100` | `overlap` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 104 | `G12_memory20_participation100` | `cifar100` | `overlap` | `gem` | 1 | `random` | `fedavg` | 10 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"` | — |
| 105 | `G12_memory0_participation50` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 106 | `G12_memory10_participation50` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 107 | `G12_memory20_participation50` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 108 | `G12_memory0_participation25` | `cifar100` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 109 | `G12_memory10_participation25` | `cifar100` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 110 | `G12_memory20_participation25` | `cifar100` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 10 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"` | — |


#### G13 — Memory selection algorithms

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 225 | `G13_extended_memory_policy_random_per_class` | `cifar100` | `recurrence` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"`; `scenario="recurrence"` | — |
| 226 | `G13_extended_memory_policy_recent_per_class` | `cifar100` | `recurrence` | `replay` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="replay"`; `scenario="recurrence"`; `memory_policy="recent_per_class"` | — |


#### G14 — GEM per-group constraints

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 336 | `G14_closure_gem_reference_class` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 337 | `G14_closure_gem_reference_stage` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `gem_grouping="stage"` | — |
| 338 | `G14_closure_gem_reference_all` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `gem_grouping="all"` | — |


#### G15 — Proximal regularization plus GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 219 | `G15_extended_prox_0.0_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 220 | `G15_extended_prox_0.0_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 221 | `G15_extended_prox_0.001_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `fedprox_mu=0.001` | — |
| 222 | `G15_extended_prox_0.001_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `fedprox_mu=0.001` | — |
| 223 | `G15_extended_prox_0.01_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `fedprox_mu=0.01` | — |
| 224 | `G15_extended_prox_0.01_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `fedprox_mu=0.01` | — |


#### G16 — Heterogeneous local algorithms

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 227 | `G16_extended_heterogeneous_False` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"` | — |
| 228 | `G16_extended_heterogeneous_True` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `client_heterogeneity=true` | — |


### H. Synchronization

Broadcast scope, memory/optimizer persistence, inactivity, rejoining, synchronization frequency.

#### H01 — Active-only synchronization

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 82 | `H01_sync_False_aggregate_fedavg` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 83 | `H01_sync_False_aggregate_uniform` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"` | — |


#### H02 — All-client synchronization

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 84 | `H02_sync_True_aggregate_fedavg` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `full_sync=true` | — |
| 85 | `H02_sync_True_aggregate_uniform` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"`; `full_sync=true` | — |


#### H03 — Sync when rejoining

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 412 | `H03_controls_rejoin_fullsync_False` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"` | — |
| 413 | `H03_controls_rejoin_fullsync_True` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `full_sync=true` | — |


#### H04 — Optimizer state after broadcast

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 341 | `H04_closure_momentum_reset` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `momentum=0.9` | — |
| 342 | `H04_closure_momentum_persist` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `optimizer_state_policy="persist"`; `momentum=0.9` | — |


#### H05 — Private memory persistence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 414 | `H05_controls_private_memory_reset_False` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 415 | `H05_controls_private_memory_reset_True` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `reset_memory_on_sync=true` | — |


#### H06 — Synchronization frequency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 235 | `H06_extended_local_epochs_1` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 236 | `H06_extended_local_epochs_2` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 2 | `epochs=2`; `scenario="overlap"` | — |
| 237 | `H06_extended_local_epochs_4` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 4 | `epochs=4`; `scenario="overlap"` | — |


#### H07 — Long inactive gap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 343 | `H07_closure_rejoin_included` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=10`; `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 344 | `H07_closure_rejoin_excluded` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=10`; `participation_overrides={"20":[0,1,2,4,5],"21":[0,1,2,4,5],"22":[0,1,2,4,5],"23":[0,1,2,4,5],"24":[0,1,2,4,5],"25":[0,1,2,4,5],"26":[0,1,2,4,5],"27":[0,1,2,4,5]}` | — |


### I. Aggregation

FedAvg, uniform, server adaptive updates, class balancing, gradient conflict and retention diagnostics.

#### I01 — Sample-weighted FedAvg

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 416 | `I01_controls_standalone_aggregation_fedavg` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |


#### I02 — Uniform aggregation

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 417 | `I02_controls_standalone_aggregation_uniform` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"` | — |


#### I03 — Class-balanced aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 345 | `I03_closure_aggregate_fedavg_I03` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 346 | `I03_closure_aggregate_class_balanced_I03` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `class_balanced` | 10 × 20 × 1 | `aggregator="class_balanced"`; `scenario="overlap"` | — |


#### I04 — Correlation-aware aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 347 | `I04_closure_aggregate_fedavg_I04` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 348 | `I04_closure_aggregate_conflict_aware_I04` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `conflict_aware` | 10 × 20 × 1 | `aggregator="conflict_aware"`; `scenario="overlap"` | — |


#### I05 — Old-retention aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 349 | `I05_closure_aggregate_fedavg_I05` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 350 | `I05_closure_aggregate_retention_aware_I05` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `retention_aware` | 10 × 20 × 1 | `aggregator="retention_aware"`; `scenario="overlap"` | — |


#### I06 — Server momentum/adaptive optimizer

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 8. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 238 | `I06_extended_aggregate_fedavg_non_iid` | `cifar100` | `non_iid` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="non_iid"` | — |
| 239 | `I06_extended_aggregate_fedavg_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 240 | `I06_extended_aggregate_uniform_non_iid` | `cifar100` | `non_iid` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="non_iid"` | — |
| 241 | `I06_extended_aggregate_uniform_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `uniform` | 10 × 20 × 1 | `aggregator="uniform"`; `scenario="related"`; `relation="related"` | — |
| 242 | `I06_extended_aggregate_fedavgm_non_iid` | `cifar100` | `non_iid` | `sgd` | 0.5 | `random` | `fedavgm` | 10 × 20 × 1 | `aggregator="fedavgm"`; `scenario="non_iid"` | — |
| 243 | `I06_extended_aggregate_fedavgm_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavgm` | 10 × 20 × 1 | `aggregator="fedavgm"`; `scenario="related"`; `relation="related"` | — |
| 244 | `I06_extended_aggregate_fedadam_non_iid` | `cifar100` | `non_iid` | `sgd` | 0.5 | `random` | `fedadam` | 10 × 20 × 1 | `aggregator="fedadam"`; `scenario="non_iid"`; `server_lr=0.001` | — |
| 245 | `I06_extended_aggregate_fedadam_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedadam` | 10 × 20 × 1 | `aggregator="fedadam"`; `scenario="related"`; `relation="related"`; `server_lr=0.001` | — |


#### I07 — Gradient conflict correction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 351 | `I07_closure_aggregate_fedavg_I07` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 352 | `I07_closure_aggregate_conflict_projection_I07` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `conflict_projection` | 10 × 20 × 1 | `aggregator="conflict_projection"`; `scenario="overlap"` | — |


#### I08 — Participation-probability correction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 353 | `I08_closure_aggregate_fedavg_I08` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 354 | `I08_closure_aggregate_inverse_participation_I08` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `inverse_participation` | 10 × 20 × 1 | `aggregator="inverse_participation"`; `scenario="overlap"` | — |


### J. Data heterogeneity

IID/non-IID, label/quantity skew, domain shift, imbalanced classes, arrival/noise patterns.

#### J01 — IID vs non-IID

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 86 | `J01_iid_sgd` | `cifar100` | `iid` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="iid"` | — |
| 87 | `J01_iid_agem` | `cifar100` | `iid` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="iid"` | — |


#### J02 — Label distribution skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 88 | `J02_non_iid_sgd` | `cifar100` | `non_iid` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="non_iid"` | — |
| 89 | `J02_non_iid_agem` | `cifar100` | `non_iid` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `scenario="non_iid"` | — |


#### J03 — Client sample-count skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 355 | `J03_closure_client_sample_skew_0.0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 356 | `J03_closure_client_sample_skew_3.0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `client_sample_skew=3` | — |


#### J04 — Feature distribution skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 357 | `J04_closure_domain_feature_skew_2` | `cifar100` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=2`; `learning_setting="domain_il"` | — |
| 358 | `J04_closure_domain_feature_skew_3` | `cifar100` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 3 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=3`; `learning_setting="domain_il"` | — |


#### J05 — Class imbalance

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 359 | `J05_closure_long_tail_0.0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |
| 360 | `J05_closure_long_tail_1.0` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `class_imbalance_power=1` | — |


#### J06 — Dependent versus independent client streams

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 361 | `J06_closure_correlated_arrivals_independent` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"` | — |
| 362 | `J06_closure_correlated_arrivals_dependent` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":2,"1":2,"2":2}` | — |


#### J07 — Unequal arrival rates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 363 | `J07_closure_client_lag_0` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":0,"1":0}` | — |
| 364 | `J07_closure_client_lag_5` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":5,"1":5}` | — |


#### J08 — Inactive client data retention

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 229 | `J08_extended_pending_latest_sgd` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"` | — |
| 230 | `J08_extended_pending_latest_gem` | `cifar100` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="asynchronous"` | — |
| 231 | `J08_extended_pending_retain_all_sgd` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `pending_policy="retain_all"` | — |
| 232 | `J08_extended_pending_retain_all_gem` | `cifar100` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="asynchronous"`; `pending_policy="retain_all"` | — |
| 233 | `J08_extended_pending_bounded_sgd` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"`; `pending_policy="bounded"`; `max_pending_samples=400` | — |
| 234 | `J08_extended_pending_bounded_gem` | `cifar100` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="asynchronous"`; `pending_policy="bounded"`; `max_pending_samples=400` | — |


#### J09 — Recurrence timing distribution

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 365 | `J09_closure_recurrence_period_0` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=3` | — |
| 366 | `J09_closure_recurrence_period_5` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=3`; `recurrence_period=5`; `recurrence_repeats=3` | — |
| 367 | `J09_closure_recurrence_period_10` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `stages=2`; `recurrence_round_delay=3`; `recurrence_period=10`; `recurrence_repeats=3` | — |


#### J10 — Spurious correlations

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 418 | `J10_controls_spurious_patch_0.0` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 419 | `J10_controls_spurious_patch_1.0` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `spurious_strength=1`; `spurious_ood_eval=true` | — |


#### J11 — Label noise

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 420 | `J11_controls_client_label_noise_0.0` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 421 | `J11_controls_client_label_noise_0.2` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `label_noise_rate=0.2` | — |


#### J12 — Changing correlation structure

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 368 | `J12_closure_temporal_distribution_domain` | `cifar100` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=2`; `learning_setting="domain_il"` | — |
| 369 | `J12_closure_temporal_distribution_gradual_drift` | `cifar100` | `gradual_drift` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="gradual_drift"`; `stages=2`; `learning_setting="domain_il"` | — |


### K. Learning setting

Class-, task-, domain-incremental and boundary-free variants with distinct information assumptions.

#### K01 — Class-incremental learning

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 422 | `K01_controls_class_incremental_reference` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |


#### K02 — Task-incremental learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 370 | `K02_closure_task_il_sgd` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `stages=2`; `learning_setting="task_il"` | — |
| 371 | `K02_closure_task_il_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `stages=2`; `learning_setting="task_il"` | — |


#### K03 — Domain-incremental learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 372 | `K03_closure_domain_il_domain` | `cifar100` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=2`; `learning_setting="domain_il"` | — |
| 373 | `K03_closure_domain_il_gradual_drift` | `cifar100` | `gradual_drift` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="gradual_drift"`; `stages=2`; `learning_setting="domain_il"` | — |


#### K04 — Task-free learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 374 | `K04_closure_boundary_free_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |
| 375 | `K04_closure_boundary_free_agem` | `cifar100` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="agem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |


#### K06 — Known versus hidden experience boundaries

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 376 | `K06_closure_experience_boundary_known` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"` | — |
| 377 | `K06_closure_experience_boundary_hidden` | `cifar100` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |


#### K07 — Asynchronous versus synchronized transitions

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 378 | `K07_closure_transitions_disjoint` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 379 | `K07_closure_transitions_asynchronous` | `cifar100` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="asynchronous"` | — |


### L. Causal and mechanistic diagnostics

Leave/add-one, accepted-update subset effects, checkpoint forks, loss and representation probes.

#### L01 — Leave-one-client-out aggregation

**Implementation status:** `implemented` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 246 | `L01_extended_leaveone_overlap_sgd` | `cifar100` | `overlap` | `sgd` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="overlap"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 247 | `L01_extended_leaveone_overlap_gem` | `cifar100` | `overlap` | `gem` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="overlap"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 248 | `L01_extended_leaveone_related_sgd` | `cifar100` | `related` | `sgd` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="related"`; `relation="related"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 249 | `L01_extended_leaveone_related_gem` | `cifar100` | `related` | `gem` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="related"`; `relation="related"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 250 | `L01_extended_leaveone_recurrence_sgd` | `cifar100` | `recurrence` | `sgd` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="recurrence"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 251 | `L01_extended_leaveone_recurrence_gem` | `cifar100` | `recurrence` | `gem` | 0.6 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="recurrence"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |


#### L02 — Add-one-client aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 18. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 261 | `L02_forkbaseline_related_sgd` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `checkpoint_every=1` | — |
| 262 | `L02_forkinclude_related_sgd` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_sgd` @ round 20 |
| 263 | `L02_forkexclude_related_sgd` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_sgd` @ round 20 |
| 264 | `L02_forkbaseline_related_gem` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `checkpoint_every=1` | — |
| 265 | `L02_forkinclude_related_gem` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_gem` @ round 20 |
| 266 | `L02_forkexclude_related_gem` | `cifar100` | `related` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_gem` @ round 20 |
| 267 | `L02_forkbaseline_overlap_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `checkpoint_every=1` | — |
| 268 | `L02_forkinclude_overlap_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_sgd` @ round 20 |
| 269 | `L02_forkexclude_overlap_sgd` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_sgd` @ round 20 |
| 270 | `L02_forkbaseline_overlap_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `checkpoint_every=1` | — |
| 271 | `L02_forkinclude_overlap_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_gem` @ round 20 |
| 272 | `L02_forkexclude_overlap_gem` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_gem` @ round 20 |
| 273 | `L02_forkbaseline_recurrence_sgd` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `checkpoint_every=1` | — |
| 274 | `L02_forkinclude_recurrence_sgd` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_sgd` @ round 20 |
| 275 | `L02_forkexclude_recurrence_sgd` | `cifar100` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="recurrence"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_sgd` @ round 20 |
| 276 | `L02_forkbaseline_recurrence_gem` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `checkpoint_every=1` | — |
| 277 | `L02_forkinclude_recurrence_gem` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_gem` @ round 20 |
| 278 | `L02_forkexclude_recurrence_gem` | `cifar100` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_gem` @ round 20 |


#### L03 — Pairwise client interactions

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 380 | `L03_closure_counterfactual_L03` | `cifar100` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 10 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


#### L04 — Exhaustive participating subset enumeration

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 381 | `L04_closure_counterfactual_L04` | `cifar100` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 10 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


#### L05 — Pre/post local historical loss

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 382 | `L05_closure_historical_losses_L05` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `prepost_diagnostics=true` | — |


#### L06 — Pre/post aggregation loss

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 383 | `L06_closure_historical_losses_L06` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"`; `prepost_diagnostics=true` | — |


#### L07 — Gradient interference analysis

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 384 | `L07_closure_gradient_interference_L07` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `gradient_diagnostics=true`; `prepost_diagnostics=true` | — |


#### L08 — Representation drift

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 386 | `L08_closure_representation_drift_disjoint` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `representation_diagnostics=true` | — |
| 387 | `L08_closure_representation_drift_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `representation_diagnostics=true` | — |


#### L09 — Correlation intervention

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 400 | `L09_controls_permuted_class_arrival_related` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 403 | `L09_controls_permuted_class_arrival_unrelated` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |


#### L10 — Historical contribution attribution

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 385 | `L10_closure_gradient_interference_L10` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"`; `gradient_diagnostics=true`; `prepost_diagnostics=true` | — |


### M. Robustness and resource accounting

Dataset, architecture, client-count, compute, communication, replication and resource settings.

#### M01 — Dataset generalization

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 423 | `M01_controls_dataset_generalization_cifar100` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |


#### M02 — Model architecture

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 90 | `M02_backbone_linear` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `model="linear"` | — |
| 91 | `M02_backbone_mlp` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `model="mlp"` | — |
| 92 | `M02_backbone_smallcnn` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |


#### M03 — Client population size

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 255 | `M03_extended_client_population_5` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=5`; `scenario="overlap"` | — |
| 256 | `M03_extended_client_population_10` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `num_clients=10`; `scenario="overlap"` | — |
| 257 | `M03_extended_client_population_20` | `cifar100` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="overlap"` | — |


#### M04 — Local optimization intensity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 258 | `M04_extended_local_steps_1` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | — | — |
| 259 | `M04_extended_local_steps_2` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 2 | `epochs=2` | — |
| 260 | `M04_extended_local_steps_4` | `cifar100` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 4 | `epochs=4` | — |


#### M05 — Randomness robustness

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 252 | `M05_extended_independent_rep_0` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 253 | `M05_extended_independent_rep_1` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `seed=1`; `relation="related"` | — |
| 254 | `M05_extended_independent_rep_2` | `cifar100` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `scenario="related"`; `seed=2`; `relation="related"` | — |


#### M06 — Communication efficiency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 388 | `M06_closure_resource_fixed` | `cifar100` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"` | — |
| 389 | `M06_closure_resource_round_robin` | `cifar100` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"` | — |


#### M07 — Computational efficiency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 390 | `M07_closure_resource_fixed` | `cifar100` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 10 × 20 × 1 | `selector="fixed"` | — |
| 391 | `M07_closure_resource_round_robin` | `cifar100` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 10 × 20 × 1 | `selector="round_robin"` | — |


### N. Proposed adaptive methods

Participation balancing and selection guided by relation, recurrence, GEM and feedback.

#### N01 — Correlation-aware participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 123 | `N01_related` | `cifar100` | `related` | `sgd` | 0.5 | `related` | `fedavg` | 10 × 20 × 1 | `selector="related"`; `scenario="related"`; `relation="related"` | — |


#### N02 — Class-participation balancing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 197 | `N02_extended_historical_coverage_overlap` | `cifar100` | `overlap` | `sgd` | 0.5 | `historical_coverage` | `fedavg` | 10 × 20 × 1 | `selector="historical_coverage"`; `scenario="overlap"` | — |
| 198 | `N02_extended_historical_coverage_related` | `cifar100` | `related` | `sgd` | 0.5 | `historical_coverage` | `fedavg` | 10 × 20 × 1 | `selector="historical_coverage"`; `scenario="related"`; `relation="related"` | — |


#### N03 — Recurrence-aware scheduling

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 124 | `N03_recurrence` | `cifar100` | `recurrence` | `sgd` | 0.5 | `recurrence` | `fedavg` | 10 × 20 × 1 | `selector="recurrence"`; `scenario="recurrence"` | — |


#### N04 — GEM-informed participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 392 | `N04_closure_gem_informed_gem` | `cifar100` | `disjoint` | `gem` | 0.5 | `gem_informed` | `fedavg` | 10 × 20 × 1 | `selector="gem_informed"`; `method="gem"` | — |
| 393 | `N04_closure_gem_informed_agem` | `cifar100` | `disjoint` | `agem` | 0.5 | `gem_informed` | `fedavg` | 10 × 20 × 1 | `selector="gem_informed"`; `method="agem"` | — |


#### N05 — Adaptive participation rate

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 424 | `N05_controls_feedback_participation_constant` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 425 | `N05_controls_feedback_participation_feedback` | `cifar100` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 10 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `participation_schedule="feedback"` | — |


## Results, output and dependencies

The intended result path is `results/cifar100/runs/<job_id>/`. The runtime saves `config.json`, `scenario.pt`, `status.json`, `metrics.json`, full-state/periodic checkpoints when applicable, stage models, `model_final.pt` and `summary.json` on completion. None of those outcomes are implied by this documentation.

The global evaluation population includes **all classes arriving at any client**, including inactive clients. Overall accuracy, forgetting on historically learned classes, old/new accuracy, never-integrated accuracy and domain/task-specific metrics are not interchangeable.

Runs that have `fork_source_job` depend on the baseline job's recorded full-state checkpoint. Job order must remain intact for causal forks; a missing source checkpoint is a failure, not a successful experiment.

For study definitions, the field dictionary, exact run commands, privacy and scientific limitations, and the full 152-block coverage cross-reference, see [JOB_SUITE_GUIDE.md](JOB_SUITE_GUIDE.md).
