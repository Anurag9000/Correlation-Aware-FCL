# CIFAR10 — all 423 scheduled experiment jobs

> Generated from the frozen source manifest on the verified `main` branch. This is a complete inventory of **scheduled configurations**, not a record of completed training.

**Suite:** `cifar10` · **jobs:** **423** · **research block IDs:** **144/152** · **configured batch size:** **512**.

**Authoritative manifest:** [`configs/plans/cifar10_jobs.json`](../configs/plans/cifar10_jobs.json). Every job has its complete resolved configuration there.

## Reading the entries

- `#` is the original **one-based execution position**, even though entries below are grouped by scientific block for browsing.
- `Non-default parameters` lists **every** value differing from the suite-wide base record below, excluding `job_id` and `block_id` (shown separately). Applying these differences reconstructs the full manifest entry exactly.
- The main method/scenario/selection/aggregation/round fields also have their own columns for easy comparison. `Fork dependency` is mandatory for downstream counterfactual branch jobs.
- Different jobs sharing a block ID are related by scientific theme, **not automatically matched causal controls**. Consult the actual parameter differences before interpreting comparisons.

## Base configuration used for exact per-job differences

| Field | Value (JSON) |
|---|---|
| `dataset` | `"cifar10"` |
| `batch_size` | `512` |
| `num_clients` | `10` |
| `rounds_per_stage` | `20` |
| `epochs` | `1` |
| `participation` | `0.5` |
| `selector` | `"random"` |
| `method` | `"sgd"` |
| `aggregator` | `"fedavg"` |
| `scenario` | `"disjoint"` |
| `stages` | `2` |
| `memory_per_class` | `20` |
| `lr` | `0.1` |
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
| D | Temporal recurrence | 39 | 12 |
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
| 1 | `A01_disjoint_sgd_full` | `cifar10` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1` | — |
| 2 | `A01_disjoint_sgd_random50` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 3 | `A01_disjoint_sgd_random25` | `cifar10` | `disjoint` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25` | — |
| 4 | `A01_disjoint_sgd_coverage50` | `cifar10` | `disjoint` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"` | — |
| 5 | `A01_disjoint_sgd_coverage25` | `cifar10` | `disjoint` | `sgd` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"` | — |
| 6 | `A01_disjoint_gem_full` | `cifar10` | `disjoint` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"` | — |
| 7 | `A01_disjoint_gem_random50` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 8 | `A01_disjoint_gem_random25` | `cifar10` | `disjoint` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"` | — |
| 9 | `A01_disjoint_gem_coverage50` | `cifar10` | `disjoint` | `gem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="gem"` | — |
| 10 | `A01_disjoint_gem_coverage25` | `cifar10` | `disjoint` | `gem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"` | — |
| 11 | `A01_disjoint_agem_full` | `cifar10` | `disjoint` | `agem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="agem"` | — |
| 12 | `A01_disjoint_agem_random50` | `cifar10` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"` | — |
| 13 | `A01_disjoint_agem_random25` | `cifar10` | `disjoint` | `agem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="agem"` | — |
| 14 | `A01_disjoint_agem_coverage50` | `cifar10` | `disjoint` | `agem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="agem"` | — |
| 15 | `A01_disjoint_agem_coverage25` | `cifar10` | `disjoint` | `agem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"` | — |
| 117 | `A01_participation_100` | `cifar10` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1` | — |
| 118 | `A01_participation_75` | `cifar10` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.75` | — |
| 119 | `A01_participation_50` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 120 | `A01_participation_25` | `cifar10` | `disjoint` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25` | — |


#### A02 — Fixed versus resampled participants

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 392 | `A02_controls_fixed_resampled_random` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 393 | `A02_controls_fixed_resampled_fixed` | `cifar10` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"` | — |


#### A03 — Equal communication rounds

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 277 | `A03_closure_roundmatched_100` | `cifar10` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1` | — |
| 278 | `A03_closure_roundmatched_50` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |


#### A04 — Matched total client updates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 279 | `A04_closure_clientupdates_full` | `cifar10` | `disjoint` | `sgd` | 1 | `random` | `fedavg` | 2 × 10 × 1 | `rounds_per_stage=10`; `participation=1` | — |
| 280 | `A04_closure_clientupdates_half` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |


#### A05 — Old/new class-holder participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 140 | `A05_extended_old_new_old` | `cifar10` | `overlap` | `sgd` | 0.5 | `old` | `fedavg` | 2 × 20 × 1 | `selector="old"`; `scenario="overlap"` | — |
| 141 | `A05_extended_old_new_new` | `cifar10` | `overlap` | `sgd` | 0.5 | `new` | `fedavg` | 2 × 20 × 1 | `selector="new"`; `scenario="overlap"` | — |
| 142 | `A05_extended_old_new_balanced` | `cifar10` | `overlap` | `sgd` | 0.5 | `balanced` | `fedavg` | 2 × 20 × 1 | `selector="balanced"`; `scenario="overlap"` | — |


#### A06 — Stage-dependent participation schedule

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 281 | `A06_closure_earlyfull` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_schedule="stage"`; `stage_participation=[1,0.25]` | — |
| 282 | `A06_closure_latefull` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_schedule="stage"`; `stage_participation=[0.25,1]` | — |


#### A07 — Class diversity at equal participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 283 | `A07_closure_diversity_coverage` | `cifar10` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 284 | `A07_closure_diversity_fixed` | `cifar10` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"`; `scenario="overlap"` | — |


#### A08 — Client redundancy with matched class coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 394 | `A08_controls_redundancy_coverage` | `cifar10` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 395 | `A08_controls_redundancy_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 2 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |


#### A09 — Short frequent versus long rare participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 285 | `A09_closure_frequency_fixed` | `cifar10` | `overlap` | `sgd` | 0.25 | `fixed` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="fixed"`; `scenario="overlap"` | — |
| 286 | `A09_closure_frequency_round_robin` | `cifar10` | `overlap` | `sgd` | 0.25 | `round_robin` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="round_robin"`; `scenario="overlap"` | — |


#### A10 — Monotonically changing participation

**Implementation status:** `implemented` · **Scheduled jobs:** 8. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 129 | `A10_extended_schedule_constant_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 130 | `A10_extended_schedule_constant_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 131 | `A10_extended_schedule_increasing_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_schedule="increasing"` | — |
| 132 | `A10_extended_schedule_increasing_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `participation_schedule="increasing"` | — |
| 133 | `A10_extended_schedule_decreasing_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_schedule="decreasing"` | — |
| 134 | `A10_extended_schedule_decreasing_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `participation_schedule="decreasing"` | — |
| 135 | `A10_extended_schedule_alternating_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_schedule="alternating"` | — |
| 136 | `A10_extended_schedule_alternating_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `participation_schedule="alternating"` | — |


#### A11 — Equal lifetime frequency, different timing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 137 | `A11_extended_timing_fixed` | `cifar10` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"` | — |
| 138 | `A11_extended_timing_round_robin` | `cifar10` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"` | — |
| 139 | `A11_extended_timing_least_recent` | `cifar10` | `disjoint` | `sgd` | 0.5 | `least_recent` | `fedavg` | 2 × 20 × 1 | `selector="least_recent"` | — |


### B. Old/new related classes

Relatedness, gradient compatibility, class-dose effects and selecting clients with related new classes.

#### B01 — Related vs unrelated new classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 115 | `B01_class_relationship_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 116 | `B01_class_relationship_unrelated` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |


#### B02 — Select/exclude related-class client

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 21. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 31 | `B02_related_sgd_full` | `cifar10` | `related` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="related"`; `relation="related"` | — |
| 32 | `B02_related_sgd_random50` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 33 | `B02_related_sgd_random25` | `cifar10` | `related` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="related"`; `relation="related"` | — |
| 34 | `B02_related_sgd_coverage50` | `cifar10` | `related` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="related"`; `relation="related"` | — |
| 35 | `B02_related_sgd_coverage25` | `cifar10` | `related` | `sgd` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="related"`; `relation="related"` | — |
| 36 | `B02_related_gem_full` | `cifar10` | `related` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 37 | `B02_related_gem_random50` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 38 | `B02_related_gem_random25` | `cifar10` | `related` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 39 | `B02_related_gem_coverage50` | `cifar10` | `related` | `gem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 40 | `B02_related_gem_coverage25` | `cifar10` | `related` | `gem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 41 | `B02_related_agem_full` | `cifar10` | `related` | `agem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 42 | `B02_related_agem_random50` | `cifar10` | `related` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 43 | `B02_related_agem_random25` | `cifar10` | `related` | `agem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 44 | `B02_related_agem_coverage50` | `cifar10` | `related` | `agem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 45 | `B02_related_agem_coverage25` | `cifar10` | `related` | `agem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 109 | `B02_relationnatural_include` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `target_client=2`; `force_target="include"` | — |
| 110 | `B02_relationnatural_exclude` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `target_client=2`; `force_target="exclude"` | — |
| 111 | `B02_relationrelated_include` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"` | — |
| 112 | `B02_relationrelated_exclude` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"` | — |
| 113 | `B02_relationunrelated_include` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `target_client=2`; `force_target="include"` | — |
| 114 | `B02_relationunrelated_exclude` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `target_client=2`; `force_target="exclude"` | — |


#### B03 — Relationship strength

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 396 | `B03_controls_coarse_relationship_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `gradient_diagnostics=true` | — |
| 399 | `B03_controls_coarse_relationship_unrelated` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `gradient_diagnostics=true` | — |


#### B04 — Gradient alignment classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 397 | `B04_controls_gradient_relationship_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `gradient_diagnostics=true` | — |
| 400 | `B04_controls_gradient_relationship_unrelated` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"`; `gradient_diagnostics=true` | — |


#### B05 — One-to-one related classes

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 287 | `B05_closure_exact_old(1, 3)_new(5, 9)_include` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="include"`; `intervention_round=20`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |
| 289 | `B05_closure_exact_old(1, 3)_new(5, 9)_exclude` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="exclude"`; `intervention_round=20`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |


#### B07 — Related-class training dose

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 291 | `B07_closure_related_dose_25` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}`; `class_sample_fraction={"5":0.25,"9":0.25}` | — |
| 292 | `B07_closure_related_dose_50` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}`; `class_sample_fraction={"5":0.5,"9":0.5}` | — |
| 293 | `B07_closure_related_dose_100` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}`; `class_sample_fraction={"5":1,"9":1}` | — |


#### B08 — Repeated related-client selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 294 | `B08_closure_related_participation_once` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="include"`; `intervention_round=20`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |
| 295 | `B08_closure_related_participation_always` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="include"`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |


#### B09 — Mixed related/unrelated participants

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 296 | `B09_closure_mixed_related_related` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}`; `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 297 | `B09_closure_mixed_related_unrelated` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}`; `participation_overrides={"20":[0,1,2,4,5]}` | — |


#### B11 — Local GEM x relationship

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 10. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 143 | `B11_extended_relation_related_local_sgd` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 144 | `B11_extended_relation_related_local_gem` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |
| 145 | `B11_extended_relation_related_local_agem` | `cifar10` | `related` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |
| 146 | `B11_extended_relation_related_local_replay` | `cifar10` | `related` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="related"` | — |
| 147 | `B11_extended_relation_related_local_derpp` | `cifar10` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="related"` | — |
| 148 | `B11_extended_relation_unrelated_local_sgd` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |
| 149 | `B11_extended_relation_unrelated_local_gem` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="unrelated"` | — |
| 150 | `B11_extended_relation_unrelated_local_agem` | `cifar10` | `related` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="unrelated"` | — |
| 151 | `B11_extended_relation_unrelated_local_replay` | `cifar10` | `related` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="unrelated"` | — |
| 152 | `B11_extended_relation_unrelated_local_derpp` | `cifar10` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="unrelated"` | — |


#### B12 — Per-old-class spillover

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 288 | `B12_closure_class_spillover_include` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="include"`; `intervention_round=20`; `prepost_diagnostics=true`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |
| 290 | `B12_closure_class_spillover_exclude` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `target_client=3`; `force_target="exclude"`; `intervention_round=20`; `prepost_diagnostics=true`; `holders_override={"1":[0],"3":[0],"5":[3],"9":[3]}` | — |


### C. Within-round class overlap

Simultaneous overlapping labels, shared-class frequencies, sample duplication, joint selection.

#### C01 — Overlap magnitude

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 93 | `C01_overlap_0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=0` | — |
| 94 | `C01_overlap_33` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=0.33` | — |
| 95 | `C01_overlap_67` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=0.67` | — |
| 96 | `C01_overlap_100` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |


#### C02 — Joint versus separate participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 15. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 16 | `C02_overlap_sgd_full` | `cifar10` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="overlap"` | — |
| 17 | `C02_overlap_sgd_random50` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 18 | `C02_overlap_sgd_random25` | `cifar10` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="overlap"` | — |
| 19 | `C02_overlap_sgd_coverage50` | `cifar10` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 20 | `C02_overlap_sgd_coverage25` | `cifar10` | `overlap` | `sgd` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="overlap"` | — |
| 21 | `C02_overlap_gem_full` | `cifar10` | `overlap` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"` | — |
| 22 | `C02_overlap_gem_random50` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 23 | `C02_overlap_gem_random25` | `cifar10` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"` | — |
| 24 | `C02_overlap_gem_coverage50` | `cifar10` | `overlap` | `gem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |
| 25 | `C02_overlap_gem_coverage25` | `cifar10` | `overlap` | `gem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |
| 26 | `C02_overlap_agem_full` | `cifar10` | `overlap` | `agem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="overlap"` | — |
| 27 | `C02_overlap_agem_random50` | `cifar10` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="overlap"` | — |
| 28 | `C02_overlap_agem_random25` | `cifar10` | `overlap` | `agem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="overlap"` | — |
| 29 | `C02_overlap_agem_coverage50` | `cifar10` | `overlap` | `agem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="overlap"` | — |
| 30 | `C02_overlap_agem_coverage25` | `cifar10` | `overlap` | `agem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="overlap"` | — |


#### C03 — Shared-class dominance

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 298 | `C03_closure_balanced` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `holders_override={"0":[0,1]}`; `class_holder_weights={"0":[0.5,0.5]}` | — |
| 299 | `C03_closure_dominant` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `holders_override={"0":[0,1]}`; `class_holder_weights={"0":[0.9,0.1]}` | — |


#### C04 — Same labels with disjoint examples

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 300 | `C04_closure_same_label_disjoint_images` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |


#### C05 — Exact sample overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 301 | `C05_closure_raw_sample_overlap_0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |
| 302 | `C05_closure_raw_sample_overlap_25` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `sample_overlap_fraction=0.25` | — |
| 303 | `C05_closure_raw_sample_overlap_50` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `sample_overlap_fraction=0.5` | — |


#### C06 — Shared class across shifted domains

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 304 | `C06_closure_client_domain_shared_class` | `cifar10` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `learning_setting="domain_il"` | — |


#### C07 — Gradient interactions of overlapping clients

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 305 | `C07_closure_overlapping_client_update_cosines` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1`; `gradient_diagnostics=true` | — |


#### C08 — Redundancy versus complementary coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 162 | `C08_extended_overlap_selector_diversity` | `cifar10` | `overlap` | `sgd` | 0.5 | `diversity` | `fedavg` | 2 × 20 × 1 | `selector="diversity"`; `scenario="overlap"` | — |
| 163 | `C08_extended_overlap_selector_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 2 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |
| 164 | `C08_extended_overlap_selector_coverage` | `cifar10` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |


#### C09 — Overlap x local GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 9. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 153 | `C09_extended_overlap_0_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=0` | — |
| 154 | `C09_extended_overlap_0_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `overlap=0` | — |
| 155 | `C09_extended_overlap_0_agem` | `cifar10` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="overlap"`; `overlap=0` | — |
| 156 | `C09_extended_overlap_50_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 157 | `C09_extended_overlap_50_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 158 | `C09_extended_overlap_50_agem` | `cifar10` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="overlap"` | — |
| 159 | `C09_extended_overlap_100_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `overlap=1` | — |
| 160 | `C09_extended_overlap_100_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `overlap=1` | — |
| 161 | `C09_extended_overlap_100_agem` | `cifar10` | `overlap` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="overlap"`; `overlap=1` | — |


#### C10 — Longitudinal overlap selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 306 | `C10_closure_longitudinal_fixed` | `cifar10` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"`; `scenario="overlap"`; `overlap=1` | — |
| 307 | `C10_closure_longitudinal_round_robin` | `cifar10` | `overlap` | `sgd` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"`; `scenario="overlap"`; `overlap=1` | — |


### D. Temporal recurrence

Earlier classes reappearing locally or across clients, including time delays and exclusion.

#### D01 — Same-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 97 | `D01_recurrence_same` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_client_policy="same"` | — |


#### D02 — Cross-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 98 | `D02_recurrence_different` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |


#### D03 — Related but new-label recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 402 | `D03_controls_related_new_labels_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 403 | `D03_controls_related_new_labels_recurrence` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `relation="related"` | — |


#### D04 — Shifted-domain recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 404 | `D04_controls_domain_shift_recurrence` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |
| 405 | `D04_controls_domain_shift_shifted_recurrence` | `cifar10` | `shifted_recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="shifted_recurrence"` | — |


#### D05 — Recurrence delay

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 99 | `D05_recurrence_stages2_delay1` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |


#### D06 — Recurrence timing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 308 | `D06_closure_lag_1_rounds` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=1` | — |
| 309 | `D06_closure_lag_5_rounds` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=5` | — |
| 310 | `D06_closure_lag_15_rounds` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=15` | — |


#### D07 — Select/exclude recurring client

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 15. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 46 | `D07_recurrence_sgd_full` | `cifar10` | `recurrence` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="recurrence"` | — |
| 47 | `D07_recurrence_sgd_random50` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |
| 48 | `D07_recurrence_sgd_random25` | `cifar10` | `recurrence` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="recurrence"` | — |
| 49 | `D07_recurrence_sgd_coverage50` | `cifar10` | `recurrence` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="recurrence"` | — |
| 50 | `D07_recurrence_sgd_coverage25` | `cifar10` | `recurrence` | `sgd` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `scenario="recurrence"` | — |
| 51 | `D07_recurrence_gem_full` | `cifar10` | `recurrence` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="recurrence"` | — |
| 52 | `D07_recurrence_gem_random50` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"` | — |
| 53 | `D07_recurrence_gem_random25` | `cifar10` | `recurrence` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="recurrence"` | — |
| 54 | `D07_recurrence_gem_coverage50` | `cifar10` | `recurrence` | `gem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="recurrence"` | — |
| 55 | `D07_recurrence_gem_coverage25` | `cifar10` | `recurrence` | `gem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="gem"`; `scenario="recurrence"` | — |
| 56 | `D07_recurrence_agem_full` | `cifar10` | `recurrence` | `agem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="agem"`; `scenario="recurrence"` | — |
| 57 | `D07_recurrence_agem_random50` | `cifar10` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="recurrence"` | — |
| 58 | `D07_recurrence_agem_random25` | `cifar10` | `recurrence` | `agem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="agem"`; `scenario="recurrence"` | — |
| 59 | `D07_recurrence_agem_coverage50` | `cifar10` | `recurrence` | `agem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="agem"`; `scenario="recurrence"` | — |
| 60 | `D07_recurrence_agem_coverage25` | `cifar10` | `recurrence` | `agem` | 0.25 | `coverage` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `selector="coverage"`; `method="agem"`; `scenario="recurrence"` | — |


#### D08 — Periodic recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 311 | `D08_closure_periodic_5` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=3`; `recurrence_period=5`; `recurrence_repeats=3` | — |
| 312 | `D08_closure_periodic_10` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=3`; `recurrence_period=10`; `recurrence_repeats=3` | — |


#### D09 — Mixed old/new recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 313 | `D09_closure_mixed_old_new_sgd` | `cifar10` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="mixed"` | — |
| 314 | `D09_closure_mixed_old_new_gem` | `cifar10` | `mixed` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="mixed"` | — |


#### D10 — Inactive-client recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 315 | `D10_closure_inactive_recurrence_include` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `target_client=3`; `force_target="include"`; `intervention_round=20` | — |
| 316 | `D10_closure_inactive_recurrence_exclude` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `target_client=3`; `force_target="exclude"`; `intervention_round=20` | — |


#### D11 — Delayed recurrent-client participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 317 | `D11_closure_recurrence_participation_immediate` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 318 | `D11_closure_recurrence_participation_delayed` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `participation_overrides={"20":[0,1,2,4,5],"21":[0,1,2,4,5],"22":[0,1,2,4,5],"23":[0,1,2,4,5],"24":[0,1,2,4,5],"25":[0,1,2,3,4]}` | — |


#### D12 — Recurrence under GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 165 | `D12_extended_recurrence_same_sgd` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 166 | `D12_extended_recurrence_same_gem` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 167 | `D12_extended_recurrence_same_agem` | `cifar10` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="recurrence"`; `recurrence_client_policy="same"` | — |
| 168 | `D12_extended_recurrence_different_sgd` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |
| 169 | `D12_extended_recurrence_different_gem` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"` | — |
| 170 | `D12_extended_recurrence_different_agem` | `cifar10` | `recurrence` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="recurrence"` | — |


### E. Interaction studies

Relatedness combined with overlap, recurrence, asynchrony and aggregation weighting.

#### E01 — Relatedness x overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 180 | `E01_extended_overlap_related_p25` | `cifar10` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="overlap"`; `relation="related"` | — |
| 181 | `E01_extended_overlap_related_p50` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `relation="related"` | — |
| 182 | `E01_extended_overlap_related_p100` | `cifar10` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="overlap"`; `relation="related"` | — |


#### E02 — Relatedness x recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 174 | `E02_extended_recurrence_related_p25` | `cifar10` | `recurrence` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="recurrence"`; `relation="related"` | — |
| 175 | `E02_extended_recurrence_related_p50` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `relation="related"` | — |
| 176 | `E02_extended_recurrence_related_p100` | `cifar10` | `recurrence` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="recurrence"`; `relation="related"` | — |


#### E03 — Overlap x recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 319 | `E03_closure_overlap_recurrence_sgd` | `cifar10` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="mixed"` | — |
| 320 | `E03_closure_overlap_recurrence_gem` | `cifar10` | `mixed` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="mixed"` | — |


#### E04 — Triple relationship interaction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 171 | `E04_extended_mixed_related_p25` | `cifar10` | `mixed` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="mixed"`; `relation="related"` | — |
| 172 | `E04_extended_mixed_related_p50` | `cifar10` | `mixed` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="mixed"`; `relation="related"` | — |
| 173 | `E04_extended_mixed_related_p100` | `cifar10` | `mixed` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="mixed"`; `relation="related"` | — |


#### E05 — Mixed helpful and harmful client updates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 321 | `E05_closure_helpful_harmful_gradient_align` | `cifar10` | `disjoint` | `sgd` | 0.5 | `gradient_align` | `fedavg` | 2 × 20 × 1 | `selector="gradient_align"`; `gradient_diagnostics=true` | — |
| 322 | `E05_closure_helpful_harmful_gradient_conflict` | `cifar10` | `disjoint` | `sgd` | 0.5 | `gradient_conflict` | `fedavg` | 2 × 20 × 1 | `selector="gradient_conflict"`; `gradient_diagnostics=true` | — |


#### E06 — Asynchronous old/new class relationships

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 177 | `E06_extended_asynchronous_related_p25` | `cifar10` | `asynchronous` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="asynchronous"`; `relation="related"` | — |
| 178 | `E06_extended_asynchronous_related_p50` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `relation="related"` | — |
| 179 | `E06_extended_asynchronous_related_p100` | `cifar10` | `asynchronous` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="asynchronous"`; `relation="related"` | — |


#### E07 — Rare-class recurrence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 323 | `E07_closure_rare_recurrence_10` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `class_sample_fraction={"0":0.1}` | — |
| 324 | `E07_closure_rare_recurrence_50` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `class_sample_fraction={"0":0.5}` | — |


#### E08 — Persistent vs temporary correlations

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 406 | `E08_controls_client_correlation_persistent` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"` | — |
| 407 | `E08_controls_client_correlation_temporary` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":10,"1":10}` | — |


#### E09 — Relatedness strength x participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 123 | `E09_related_p25` | `cifar10` | `related` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="related"`; `relation="related"` | — |
| 124 | `E09_overlap_p25` | `cifar10` | `overlap` | `sgd` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `scenario="overlap"` | — |
| 125 | `E09_related_p50` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 126 | `E09_overlap_p50` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 127 | `E09_related_p100` | `cifar10` | `related` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="related"`; `relation="related"` | — |
| 128 | `E09_overlap_p100` | `cifar10` | `overlap` | `sgd` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `scenario="overlap"` | — |


#### E10 — Relatedness x aggregation weights

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 183 | `E10_extended_relation_aggregation_fedavg` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 184 | `E10_extended_relation_aggregation_uniform` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="related"`; `relation="related"` | — |
| 185 | `E10_extended_relation_aggregation_fedadam` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedadam` | 2 × 20 × 1 | `aggregator="fedadam"`; `scenario="related"`; `relation="related"`; `server_lr=0.001` | — |
| 186 | `E10_extended_relation_aggregation_fedavgm` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavgm` | 2 × 20 × 1 | `aggregator="fedavgm"`; `scenario="related"`; `relation="related"` | — |


### F. Client selection

Random, deterministic, diversity, gradient, forgetting-aware, fairness and oracle-style policies.

#### F01 — Uniform random

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 61 | `F01_random_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 69 | `F01_random_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |


#### F02 — Fixed subset

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 62 | `F02_fixed_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"`; `scenario="overlap"` | — |
| 70 | `F02_fixed_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"`; `method="gem"`; `scenario="overlap"` | — |


#### F03 — Round-robin

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 63 | `F03_round_robin_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"`; `scenario="overlap"` | — |
| 71 | `F03_round_robin_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"`; `method="gem"`; `scenario="overlap"` | — |


#### F04 — Least recently selected

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 64 | `F04_least_recent_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `least_recent` | `fedavg` | 2 × 20 × 1 | `selector="least_recent"`; `scenario="overlap"` | — |
| 72 | `F04_least_recent_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `least_recent` | `fedavg` | 2 × 20 × 1 | `selector="least_recent"`; `method="gem"`; `scenario="overlap"` | — |


#### F05 — Class coverage maximizing

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 68 | `F05_coverage_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `scenario="overlap"` | — |
| 76 | `F05_coverage_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `coverage` | `fedavg` | 2 × 20 × 1 | `selector="coverage"`; `method="gem"`; `scenario="overlap"` | — |


#### F06 — Old-class coverage

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 65 | `F06_old_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `old` | `fedavg` | 2 × 20 × 1 | `selector="old"`; `scenario="overlap"` | — |
| 73 | `F06_old_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `old` | `fedavg` | 2 × 20 × 1 | `selector="old"`; `method="gem"`; `scenario="overlap"` | — |


#### F07 — New-class coverage

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 66 | `F07_new_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `new` | `fedavg` | 2 × 20 × 1 | `selector="new"`; `scenario="overlap"` | — |
| 74 | `F07_new_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `new` | `fedavg` | 2 × 20 × 1 | `selector="new"`; `method="gem"`; `scenario="overlap"` | — |


#### F08 — Balanced old and new coverage

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 187 | `F08_extended_balanced_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `balanced` | `fedavg` | 2 × 20 × 1 | `selector="balanced"`; `scenario="overlap"` | — |
| 188 | `F08_extended_balanced_related` | `cifar10` | `related` | `sgd` | 0.5 | `balanced` | `fedavg` | 2 × 20 × 1 | `selector="balanced"`; `scenario="related"`; `relation="related"` | — |


#### F09 — Maximum label diversity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 189 | `F09_extended_diversity_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `diversity` | `fedavg` | 2 × 20 × 1 | `selector="diversity"`; `scenario="overlap"` | — |
| 190 | `F09_extended_diversity_related` | `cifar10` | `related` | `sgd` | 0.5 | `diversity` | `fedavg` | 2 × 20 × 1 | `selector="diversity"`; `scenario="related"`; `relation="related"` | — |


#### F10 — Maximum label overlap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 67 | `F10_overlap_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `overlap` | `fedavg` | 2 × 20 × 1 | `selector="overlap"`; `scenario="overlap"` | — |
| 75 | `F10_overlap_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `overlap` | `fedavg` | 2 × 20 × 1 | `selector="overlap"`; `method="gem"`; `scenario="overlap"` | — |


#### F11 — Maximum feature similarity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 325 | `F11_closure_feature_max` | `cifar10` | `disjoint` | `sgd` | 0.5 | `feature_max` | `fedavg` | 2 × 20 × 1 | `selector="feature_max"` | — |


#### F12 — Minimum feature similarity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 326 | `F12_closure_feature_min` | `cifar10` | `disjoint` | `sgd` | 0.5 | `feature_min` | `fedavg` | 2 × 20 × 1 | `selector="feature_min"` | — |


#### F13 — Gradient alignment selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 327 | `F13_closure_gradient_align` | `cifar10` | `disjoint` | `sgd` | 0.5 | `gradient_align` | `fedavg` | 2 × 20 × 1 | `selector="gradient_align"`; `gradient_diagnostics=true` | — |


#### F14 — Gradient conflict selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 328 | `F14_closure_gradient_conflict` | `cifar10` | `disjoint` | `sgd` | 0.5 | `gradient_conflict` | `fedavg` | 2 × 20 × 1 | `selector="gradient_conflict"`; `gradient_diagnostics=true` | — |


#### F15 — Forgetting-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 329 | `F15_closure_forgetting` | `cifar10` | `disjoint` | `sgd` | 0.5 | `forgetting` | `fedavg` | 2 × 20 × 1 | `selector="forgetting"` | — |


#### F16 — Recurrence-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 408 | `F16_controls_recurrence_aware_random` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"` | — |
| 409 | `F16_controls_recurrence_aware_recurrence` | `cifar10` | `recurrence` | `sgd` | 0.5 | `recurrence` | `fedavg` | 2 × 20 × 1 | `selector="recurrence"`; `scenario="recurrence"` | — |


#### F17 — Fairness-constrained selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 191 | `F17_extended_fairness_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `fairness` | `fedavg` | 2 × 20 × 1 | `selector="fairness"`; `scenario="overlap"` | — |
| 192 | `F17_extended_fairness_related` | `cifar10` | `related` | `sgd` | 0.5 | `fairness` | `fedavg` | 2 × 20 × 1 | `selector="fairness"`; `scenario="related"`; `relation="related"` | — |


#### F18 — Randomized correlation-aware selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 193 | `F18_extended_stochastic_related_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `stochastic_related` | `fedavg` | 2 × 20 × 1 | `selector="stochastic_related"`; `scenario="overlap"` | — |
| 194 | `F18_extended_stochastic_related_related` | `cifar10` | `related` | `sgd` | 0.5 | `stochastic_related` | `fedavg` | 2 × 20 × 1 | `selector="stochastic_related"`; `scenario="related"`; `relation="related"` | — |


#### F19 — History-aware adaptive selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 330 | `F19_closure_history_adaptive` | `cifar10` | `disjoint` | `sgd` | 0.5 | `history_adaptive` | `fedavg` | 2 × 20 × 1 | `selector="history_adaptive"` | — |


#### F20 — Oracle exhaustive selection

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 331 | `F20_closure_exhaustive_subsets` | `cifar10` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 2 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


### G. Local continual learning

SGD, GEM, A-GEM, replay, EWC, ER-ACE-style, DER/DER++, SI, MAS and LwF.

#### G01 — Local SGD

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 77 | `G01_local_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 197 | `G01_extended_local_sgd_disjoint` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 198 | `G01_extended_local_sgd_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |


#### G02 — Local GEM

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 78 | `G02_local_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 211 | `G02_extended_local_gem_disjoint` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 212 | `G02_extended_local_gem_related` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"` | — |


#### G03 — Local A-GEM

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 79 | `G03_local_agem` | `cifar10` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"` | — |
| 213 | `G03_extended_local_agem_disjoint` | `cifar10` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"` | — |
| 214 | `G03_extended_local_agem_related` | `cifar10` | `related` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="related"`; `relation="related"` | — |


#### G04 — Experience replay

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 80 | `G04_local_replay` | `cifar10` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"` | — |
| 199 | `G04_extended_local_replay_disjoint` | `cifar10` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"` | — |
| 200 | `G04_extended_local_replay_related` | `cifar10` | `related` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"`; `scenario="related"`; `relation="related"` | — |


#### G05 — ER-ACE

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 201 | `G05_extended_local_er_ace_disjoint` | `cifar10` | `disjoint` | `er_ace` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="er_ace"` | — |
| 202 | `G05_extended_local_er_ace_related` | `cifar10` | `related` | `er_ace` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="er_ace"`; `scenario="related"`; `relation="related"` | — |


#### G06 — DER and DER++

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 4. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 203 | `G06_extended_local_der_disjoint` | `cifar10` | `disjoint` | `der` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="der"` | — |
| 204 | `G06_extended_local_der_related` | `cifar10` | `related` | `der` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="der"`; `scenario="related"`; `relation="related"` | — |
| 205 | `G06_extended_local_derpp_disjoint` | `cifar10` | `disjoint` | `derpp` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="derpp"` | — |
| 206 | `G06_extended_local_derpp_related` | `cifar10` | `related` | `derpp` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="derpp"`; `scenario="related"`; `relation="related"` | — |


#### G07 — EWC

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 81 | `G07_local_ewc` | `cifar10` | `disjoint` | `ewc` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="ewc"` | — |
| 215 | `G07_extended_local_ewc_disjoint` | `cifar10` | `disjoint` | `ewc` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="ewc"` | — |
| 216 | `G07_extended_local_ewc_related` | `cifar10` | `related` | `ewc` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="ewc"`; `scenario="related"`; `relation="related"` | — |


#### G08 — Synaptic intelligence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 332 | `G08_closure_synaptic_intelligence_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 333 | `G08_closure_synaptic_intelligence_si` | `cifar10` | `disjoint` | `si` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="si"` | — |


#### G09 — MAS

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 207 | `G09_extended_local_mas_disjoint` | `cifar10` | `disjoint` | `mas` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="mas"` | — |
| 208 | `G09_extended_local_mas_related` | `cifar10` | `related` | `mas` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="mas"`; `scenario="related"`; `relation="related"` | — |


#### G10 — Local distillation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 209 | `G10_extended_local_lwf_disjoint` | `cifar10` | `disjoint` | `lwf` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="lwf"` | — |
| 210 | `G10_extended_local_lwf_related` | `cifar10` | `related` | `lwf` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="lwf"`; `scenario="related"`; `relation="related"` | — |


#### G11 — Replay vs no replay

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 337 | `G11_closure_memory_comparison_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 338 | `G11_closure_memory_comparison_replay` | `cifar10` | `disjoint` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"` | — |


#### G12 — Memory capacity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 9. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 100 | `G12_memory0_participation100` | `cifar10` | `overlap` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 101 | `G12_memory10_participation100` | `cifar10` | `overlap` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 102 | `G12_memory20_participation100` | `cifar10` | `overlap` | `gem` | 1 | `random` | `fedavg` | 2 × 20 × 1 | `participation=1`; `method="gem"`; `scenario="overlap"` | — |
| 103 | `G12_memory0_participation50` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 104 | `G12_memory10_participation50` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 105 | `G12_memory20_participation50` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 106 | `G12_memory0_participation25` | `cifar10` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"`; `memory_per_class=0` | — |
| 107 | `G12_memory10_participation25` | `cifar10` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"`; `memory_per_class=10` | — |
| 108 | `G12_memory20_participation25` | `cifar10` | `overlap` | `gem` | 0.25 | `random` | `fedavg` | 2 × 20 × 1 | `participation=0.25`; `method="gem"`; `scenario="overlap"` | — |


#### G13 — Memory selection algorithms

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 223 | `G13_extended_memory_policy_random_per_class` | `cifar10` | `recurrence` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"`; `scenario="recurrence"` | — |
| 224 | `G13_extended_memory_policy_recent_per_class` | `cifar10` | `recurrence` | `replay` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="replay"`; `scenario="recurrence"`; `memory_policy="recent_per_class"` | — |


#### G14 — GEM per-group constraints

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 334 | `G14_closure_gem_reference_class` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 335 | `G14_closure_gem_reference_stage` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `gem_grouping="stage"` | — |
| 336 | `G14_closure_gem_reference_all` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `gem_grouping="all"` | — |


#### G15 — Proximal regularization plus GEM

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 217 | `G15_extended_prox_0.0_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 218 | `G15_extended_prox_0.0_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 219 | `G15_extended_prox_0.001_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `fedprox_mu=0.001` | — |
| 220 | `G15_extended_prox_0.001_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `fedprox_mu=0.001` | — |
| 221 | `G15_extended_prox_0.01_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `fedprox_mu=0.01` | — |
| 222 | `G15_extended_prox_0.01_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `fedprox_mu=0.01` | — |


#### G16 — Heterogeneous local algorithms

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 225 | `G16_extended_heterogeneous_False` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"` | — |
| 226 | `G16_extended_heterogeneous_True` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `client_heterogeneity=true` | — |


### H. Synchronization

Broadcast scope, memory/optimizer persistence, inactivity, rejoining, synchronization frequency.

#### H01 — Active-only synchronization

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 82 | `H01_sync_False_aggregate_fedavg` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 83 | `H01_sync_False_aggregate_uniform` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"` | — |


#### H02 — All-client synchronization

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 84 | `H02_sync_True_aggregate_fedavg` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `full_sync=true` | — |
| 85 | `H02_sync_True_aggregate_uniform` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"`; `full_sync=true` | — |


#### H03 — Sync when rejoining

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 410 | `H03_controls_rejoin_fullsync_False` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"` | — |
| 411 | `H03_controls_rejoin_fullsync_True` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `full_sync=true` | — |


#### H04 — Optimizer state after broadcast

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 339 | `H04_closure_momentum_reset` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `momentum=0.9` | — |
| 340 | `H04_closure_momentum_persist` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `optimizer_state_policy="persist"`; `momentum=0.9` | — |


#### H05 — Private memory persistence

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 412 | `H05_controls_private_memory_reset_False` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 413 | `H05_controls_private_memory_reset_True` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `reset_memory_on_sync=true` | — |


#### H06 — Synchronization frequency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 233 | `H06_extended_local_epochs_1` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 234 | `H06_extended_local_epochs_2` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 2 | `epochs=2`; `scenario="overlap"` | — |
| 235 | `H06_extended_local_epochs_4` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 4 | `epochs=4`; `scenario="overlap"` | — |


#### H07 — Long inactive gap

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 341 | `H07_closure_rejoin_included` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_overrides={"20":[0,1,2,3,4]}` | — |
| 342 | `H07_closure_rejoin_excluded` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `participation_overrides={"20":[0,1,2,4,5],"21":[0,1,2,4,5],"22":[0,1,2,4,5],"23":[0,1,2,4,5],"24":[0,1,2,4,5],"25":[0,1,2,4,5],"26":[0,1,2,4,5],"27":[0,1,2,4,5]}` | — |


### I. Aggregation

FedAvg, uniform, server adaptive updates, class balancing, gradient conflict and retention diagnostics.

#### I01 — Sample-weighted FedAvg

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 414 | `I01_controls_standalone_aggregation_fedavg` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |


#### I02 — Uniform aggregation

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 415 | `I02_controls_standalone_aggregation_uniform` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="overlap"` | — |


#### I03 — Class-balanced aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 343 | `I03_closure_aggregate_fedavg_I03` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 344 | `I03_closure_aggregate_class_balanced_I03` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `class_balanced` | 2 × 20 × 1 | `aggregator="class_balanced"`; `scenario="overlap"` | — |


#### I04 — Correlation-aware aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 345 | `I04_closure_aggregate_fedavg_I04` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 346 | `I04_closure_aggregate_conflict_aware_I04` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `conflict_aware` | 2 × 20 × 1 | `aggregator="conflict_aware"`; `scenario="overlap"` | — |


#### I05 — Old-retention aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 347 | `I05_closure_aggregate_fedavg_I05` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 348 | `I05_closure_aggregate_retention_aware_I05` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `retention_aware` | 2 × 20 × 1 | `aggregator="retention_aware"`; `scenario="overlap"` | — |


#### I06 — Server momentum/adaptive optimizer

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 8. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 236 | `I06_extended_aggregate_fedavg_non_iid` | `cifar10` | `non_iid` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="non_iid"` | — |
| 237 | `I06_extended_aggregate_fedavg_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 238 | `I06_extended_aggregate_uniform_non_iid` | `cifar10` | `non_iid` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="non_iid"` | — |
| 239 | `I06_extended_aggregate_uniform_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `uniform` | 2 × 20 × 1 | `aggregator="uniform"`; `scenario="related"`; `relation="related"` | — |
| 240 | `I06_extended_aggregate_fedavgm_non_iid` | `cifar10` | `non_iid` | `sgd` | 0.5 | `random` | `fedavgm` | 2 × 20 × 1 | `aggregator="fedavgm"`; `scenario="non_iid"` | — |
| 241 | `I06_extended_aggregate_fedavgm_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavgm` | 2 × 20 × 1 | `aggregator="fedavgm"`; `scenario="related"`; `relation="related"` | — |
| 242 | `I06_extended_aggregate_fedadam_non_iid` | `cifar10` | `non_iid` | `sgd` | 0.5 | `random` | `fedadam` | 2 × 20 × 1 | `aggregator="fedadam"`; `scenario="non_iid"`; `server_lr=0.001` | — |
| 243 | `I06_extended_aggregate_fedadam_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedadam` | 2 × 20 × 1 | `aggregator="fedadam"`; `scenario="related"`; `relation="related"`; `server_lr=0.001` | — |


#### I07 — Gradient conflict correction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 349 | `I07_closure_aggregate_fedavg_I07` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 350 | `I07_closure_aggregate_conflict_projection_I07` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `conflict_projection` | 2 × 20 × 1 | `aggregator="conflict_projection"`; `scenario="overlap"` | — |


#### I08 — Participation-probability correction

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 351 | `I08_closure_aggregate_fedavg_I08` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 352 | `I08_closure_aggregate_inverse_participation_I08` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `inverse_participation` | 2 × 20 × 1 | `aggregator="inverse_participation"`; `scenario="overlap"` | — |


### J. Data heterogeneity

IID/non-IID, label/quantity skew, domain shift, imbalanced classes, arrival/noise patterns.

#### J01 — IID vs non-IID

**Implementation status:** `implemented` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 86 | `J01_iid_sgd` | `cifar10` | `iid` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="iid"` | — |
| 87 | `J01_iid_agem` | `cifar10` | `iid` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="iid"` | — |


#### J02 — Label distribution skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 88 | `J02_non_iid_sgd` | `cifar10` | `non_iid` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="non_iid"` | — |
| 89 | `J02_non_iid_agem` | `cifar10` | `non_iid` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `scenario="non_iid"` | — |


#### J03 — Client sample-count skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 353 | `J03_closure_client_sample_skew_0.0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 354 | `J03_closure_client_sample_skew_3.0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `client_sample_skew=3` | — |


#### J04 — Feature distribution skew

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 355 | `J04_closure_domain_feature_skew_2` | `cifar10` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `learning_setting="domain_il"` | — |
| 356 | `J04_closure_domain_feature_skew_3` | `cifar10` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 3 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `stages=3`; `learning_setting="domain_il"` | — |


#### J05 — Class imbalance

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 357 | `J05_closure_long_tail_0.0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 358 | `J05_closure_long_tail_1.0` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `class_imbalance_power=1` | — |


#### J06 — Dependent versus independent client streams

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 359 | `J06_closure_correlated_arrivals_independent` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"` | — |
| 360 | `J06_closure_correlated_arrivals_dependent` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":2,"1":2,"2":2}` | — |


#### J07 — Unequal arrival rates

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 361 | `J07_closure_client_lag_0` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":0,"1":0}` | — |
| 362 | `J07_closure_client_lag_5` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `arrival_lag_by_client={"0":5,"1":5}` | — |


#### J08 — Inactive client data retention

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 227 | `J08_extended_pending_latest_sgd` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"` | — |
| 228 | `J08_extended_pending_latest_gem` | `cifar10` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="asynchronous"` | — |
| 229 | `J08_extended_pending_retain_all_sgd` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `pending_policy="retain_all"` | — |
| 230 | `J08_extended_pending_retain_all_gem` | `cifar10` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="asynchronous"`; `pending_policy="retain_all"` | — |
| 231 | `J08_extended_pending_bounded_sgd` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"`; `pending_policy="bounded"`; `max_pending_samples=400` | — |
| 232 | `J08_extended_pending_bounded_gem` | `cifar10` | `asynchronous` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="asynchronous"`; `pending_policy="bounded"`; `max_pending_samples=400` | — |


#### J09 — Recurrence timing distribution

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 363 | `J09_closure_recurrence_period_0` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=3` | — |
| 364 | `J09_closure_recurrence_period_5` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=3`; `recurrence_period=5`; `recurrence_repeats=3` | — |
| 365 | `J09_closure_recurrence_period_10` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `recurrence_round_delay=3`; `recurrence_period=10`; `recurrence_repeats=3` | — |


#### J10 — Spurious correlations

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 416 | `J10_controls_spurious_patch_0.0` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 417 | `J10_controls_spurious_patch_1.0` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `spurious_strength=1`; `spurious_ood_eval=true` | — |


#### J11 — Label noise

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 418 | `J11_controls_client_label_noise_0.0` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 419 | `J11_controls_client_label_noise_0.2` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `label_noise_rate=0.2` | — |


#### J12 — Changing correlation structure

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 366 | `J12_closure_temporal_distribution_domain` | `cifar10` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `learning_setting="domain_il"` | — |
| 367 | `J12_closure_temporal_distribution_gradual_drift` | `cifar10` | `gradual_drift` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="gradual_drift"`; `learning_setting="domain_il"` | — |


### K. Learning setting

Class-, task-, domain-incremental and boundary-free variants with distinct information assumptions.

#### K01 — Class-incremental learning

**Implementation status:** `implemented` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 420 | `K01_controls_class_incremental_reference` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |


#### K02 — Task-incremental learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 368 | `K02_closure_task_il_sgd` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `learning_setting="task_il"` | — |
| 369 | `K02_closure_task_il_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `learning_setting="task_il"` | — |


#### K03 — Domain-incremental learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 370 | `K03_closure_domain_il_domain` | `cifar10` | `domain` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="domain"`; `learning_setting="domain_il"` | — |
| 371 | `K03_closure_domain_il_gradual_drift` | `cifar10` | `gradual_drift` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="gradual_drift"`; `learning_setting="domain_il"` | — |


#### K04 — Task-free learning

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 372 | `K04_closure_boundary_free_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |
| 373 | `K04_closure_boundary_free_agem` | `cifar10` | `disjoint` | `agem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="agem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |


#### K06 — Known versus hidden experience boundaries

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 374 | `K06_closure_experience_boundary_known` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"` | — |
| 375 | `K06_closure_experience_boundary_hidden` | `cifar10` | `disjoint` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `boundary_mode="hidden"`; `learning_setting="boundary_free"` | — |


#### K07 — Asynchronous versus synchronized transitions

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 376 | `K07_closure_transitions_disjoint` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 377 | `K07_closure_transitions_asynchronous` | `cifar10` | `asynchronous` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="asynchronous"` | — |


### L. Causal and mechanistic diagnostics

Leave/add-one, accepted-update subset effects, checkpoint forks, loss and representation probes.

#### L01 — Leave-one-client-out aggregation

**Implementation status:** `implemented` · **Scheduled jobs:** 6. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 244 | `L01_extended_leaveone_overlap_sgd` | `cifar10` | `overlap` | `sgd` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="overlap"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 245 | `L01_extended_leaveone_overlap_gem` | `cifar10` | `overlap` | `gem` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="overlap"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 246 | `L01_extended_leaveone_related_sgd` | `cifar10` | `related` | `sgd` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="related"`; `relation="related"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 247 | `L01_extended_leaveone_related_gem` | `cifar10` | `related` | `gem` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="related"`; `relation="related"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 248 | `L01_extended_leaveone_recurrence_sgd` | `cifar10` | `recurrence` | `sgd` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `scenario="recurrence"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |
| 249 | `L01_extended_leaveone_recurrence_gem` | `cifar10` | `recurrence` | `gem` | 0.6 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `participation=0.6`; `method="gem"`; `scenario="recurrence"`; `causal_diagnostics=true`; `causal_max_clients=5` | — |


#### L02 — Add-one-client aggregation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 18. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 259 | `L02_forkbaseline_related_sgd` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `checkpoint_every=1` | — |
| 260 | `L02_forkinclude_related_sgd` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_sgd` @ round 20 |
| 261 | `L02_forkexclude_related_sgd` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_sgd` @ round 20 |
| 262 | `L02_forkbaseline_related_gem` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `checkpoint_every=1` | — |
| 263 | `L02_forkinclude_related_gem` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_gem` @ round 20 |
| 264 | `L02_forkexclude_related_gem` | `cifar10` | `related` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="related"`; `relation="related"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_related_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_related_gem` @ round 20 |
| 265 | `L02_forkbaseline_overlap_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `checkpoint_every=1` | — |
| 266 | `L02_forkinclude_overlap_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_sgd` @ round 20 |
| 267 | `L02_forkexclude_overlap_sgd` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_sgd` @ round 20 |
| 268 | `L02_forkbaseline_overlap_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `checkpoint_every=1` | — |
| 269 | `L02_forkinclude_overlap_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_gem` @ round 20 |
| 270 | `L02_forkexclude_overlap_gem` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_overlap_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_overlap_gem` @ round 20 |
| 271 | `L02_forkbaseline_recurrence_sgd` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `checkpoint_every=1` | — |
| 272 | `L02_forkinclude_recurrence_sgd` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_sgd` @ round 20 |
| 273 | `L02_forkexclude_recurrence_sgd` | `cifar10` | `recurrence` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="recurrence"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_sgd"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_sgd` @ round 20 |
| 274 | `L02_forkbaseline_recurrence_gem` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `checkpoint_every=1` | — |
| 275 | `L02_forkinclude_recurrence_gem` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `target_client=2`; `force_target="include"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_gem` @ round 20 |
| 276 | `L02_forkexclude_recurrence_gem` | `cifar10` | `recurrence` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="recurrence"`; `target_client=2`; `force_target="exclude"`; `intervention_round=20`; `checkpoint_every=1`; `fork_source_job="L02_forkbaseline_recurrence_gem"`; `fork_checkpoint_round=20` | `L02_forkbaseline_recurrence_gem` @ round 20 |


#### L03 — Pairwise client interactions

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 378 | `L03_closure_counterfactual_L03` | `cifar10` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 2 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


#### L04 — Exhaustive participating subset enumeration

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 379 | `L04_closure_counterfactual_L04` | `cifar10` | `disjoint` | `sgd` | 0.75 | `random` | `fedavg` | 2 × 2 × 1 | `num_clients=4`; `rounds_per_stage=2`; `participation=0.75`; `model="linear"`; `causal_max_clients=4`; `subset_diagnostics=true` | — |


#### L05 — Pre/post local historical loss

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 380 | `L05_closure_historical_losses_L05` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `prepost_diagnostics=true` | — |


#### L06 — Pre/post aggregation loss

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 381 | `L06_closure_historical_losses_L06` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"`; `prepost_diagnostics=true` | — |


#### L07 — Gradient interference analysis

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 382 | `L07_closure_gradient_interference_L07` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `gradient_diagnostics=true`; `prepost_diagnostics=true` | — |


#### L08 — Representation drift

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 384 | `L08_closure_representation_drift_disjoint` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `representation_diagnostics=true` | — |
| 385 | `L08_closure_representation_drift_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `representation_diagnostics=true` | — |


#### L09 — Correlation intervention

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 398 | `L09_controls_permuted_class_arrival_related` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 401 | `L09_controls_permuted_class_arrival_unrelated` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="unrelated"` | — |


#### L10 — Historical contribution attribution

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 383 | `L10_closure_gradient_interference_L10` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"`; `gradient_diagnostics=true`; `prepost_diagnostics=true` | — |


### M. Robustness and resource accounting

Dataset, architecture, client-count, compute, communication, replication and resource settings.

#### M01 — Dataset generalization

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 421 | `M01_controls_dataset_generalization_cifar10` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |


#### M02 — Model architecture

**Implementation status:** `implemented` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 90 | `M02_backbone_linear` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `model="linear"` | — |
| 91 | `M02_backbone_mlp` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `model="mlp"` | — |
| 92 | `M02_backbone_smallcnn` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |


#### M03 — Client population size

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 253 | `M03_extended_client_population_5` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=5`; `scenario="overlap"` | — |
| 254 | `M03_extended_client_population_10` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="overlap"` | — |
| 255 | `M03_extended_client_population_20` | `cifar10` | `overlap` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `num_clients=20`; `scenario="overlap"` | — |


#### M04 — Local optimization intensity

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 256 | `M04_extended_local_steps_1` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | — | — |
| 257 | `M04_extended_local_steps_2` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 2 | `epochs=2` | — |
| 258 | `M04_extended_local_steps_4` | `cifar10` | `disjoint` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 4 | `epochs=4` | — |


#### M05 — Randomness robustness

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 3. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 250 | `M05_extended_independent_rep_0` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `relation="related"` | — |
| 251 | `M05_extended_independent_rep_1` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `seed=1`; `relation="related"` | — |
| 252 | `M05_extended_independent_rep_2` | `cifar10` | `related` | `sgd` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `scenario="related"`; `seed=2`; `relation="related"` | — |


#### M06 — Communication efficiency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 386 | `M06_closure_resource_fixed` | `cifar10` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"` | — |
| 387 | `M06_closure_resource_round_robin` | `cifar10` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"` | — |


#### M07 — Computational efficiency

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 388 | `M07_closure_resource_fixed` | `cifar10` | `disjoint` | `sgd` | 0.5 | `fixed` | `fedavg` | 2 × 20 × 1 | `selector="fixed"` | — |
| 389 | `M07_closure_resource_round_robin` | `cifar10` | `disjoint` | `sgd` | 0.5 | `round_robin` | `fedavg` | 2 × 20 × 1 | `selector="round_robin"` | — |


### N. Proposed adaptive methods

Participation balancing and selection guided by relation, recurrence, GEM and feedback.

#### N01 — Correlation-aware participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 121 | `N01_related` | `cifar10` | `related` | `sgd` | 0.5 | `related` | `fedavg` | 2 × 20 × 1 | `selector="related"`; `scenario="related"`; `relation="related"` | — |


#### N02 — Class-participation balancing

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 195 | `N02_extended_historical_coverage_overlap` | `cifar10` | `overlap` | `sgd` | 0.5 | `historical_coverage` | `fedavg` | 2 × 20 × 1 | `selector="historical_coverage"`; `scenario="overlap"` | — |
| 196 | `N02_extended_historical_coverage_related` | `cifar10` | `related` | `sgd` | 0.5 | `historical_coverage` | `fedavg` | 2 × 20 × 1 | `selector="historical_coverage"`; `scenario="related"`; `relation="related"` | — |


#### N03 — Recurrence-aware scheduling

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 1. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 122 | `N03_recurrence` | `cifar10` | `recurrence` | `sgd` | 0.5 | `recurrence` | `fedavg` | 2 × 20 × 1 | `selector="recurrence"`; `scenario="recurrence"` | — |


#### N04 — GEM-informed participation

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 390 | `N04_closure_gem_informed_gem` | `cifar10` | `disjoint` | `gem` | 0.5 | `gem_informed` | `fedavg` | 2 × 20 × 1 | `selector="gem_informed"`; `method="gem"` | — |
| 391 | `N04_closure_gem_informed_agem` | `cifar10` | `disjoint` | `agem` | 0.5 | `gem_informed` | `fedavg` | 2 × 20 × 1 | `selector="gem_informed"`; `method="agem"` | — |


#### N05 — Adaptive participation rate

**Implementation status:** `partial_implementation` · **Scheduled jobs:** 2. Study its intervention using the exact actual configurations below; partial status signifies incomplete or proxy methodology.

| # | Exact job ID | Loader | Scenario | Local method | Participation | Selector | Aggregation | Stages × rounds/stage × epochs | Non-default parameters | Fork dependency |
|---:|---|---|---|---|---:|---|---|---|---|---|
| 422 | `N05_controls_feedback_participation_constant` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"` | — |
| 423 | `N05_controls_feedback_participation_feedback` | `cifar10` | `overlap` | `gem` | 0.5 | `random` | `fedavg` | 2 × 20 × 1 | `method="gem"`; `scenario="overlap"`; `participation_schedule="feedback"` | — |


## Results, output and dependencies

The intended result path is `results/cifar10/runs/<job_id>/`. The runtime saves `config.json`, `scenario.pt`, `status.json`, `metrics.json`, full-state/periodic checkpoints when applicable, stage models, `model_final.pt` and `summary.json` on completion. None of those outcomes are implied by this documentation.

The global evaluation population includes **all classes arriving at any client**, including inactive clients. Overall accuracy, forgetting on historically learned classes, old/new accuracy, never-integrated accuracy and domain/task-specific metrics are not interchangeable.

Runs that have `fork_source_job` depend on the baseline job's recorded full-state checkpoint. Job order must remain intact for causal forks; a missing source checkpoint is a failure, not a successful experiment.

For study definitions, the field dictionary, exact run commands, privacy and scientific limitations, and the full 152-block coverage cross-reference, see [JOB_SUITE_GUIDE.md](JOB_SUITE_GUIDE.md).
