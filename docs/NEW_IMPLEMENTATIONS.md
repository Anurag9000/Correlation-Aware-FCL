# Expanded FCL mechanism contracts and implementation limits

## Full-population invariants

1. At round `r`, data-arrival events execute for all clients **before selection**, so inactive clients influence the observed global evaluation population.
2. A selected client always downloads the committed canonical global model before its local optimization. After aggregation, it receives the **new** aggregated weights and confirms tensor equality. Inactive clients may synchronize immediately (`full_sync`) or on rejoining.
3. Class/sample identity and task labels for **CIFAR-10, full CIFAR-100, and the 20-class CIFAR-100 subset** are consistent across the stratified training/validation partitions and official test split. The 20-class subset remaps original CIFAR-100 fine labels to contiguous 0–19 and preserves official coarse-superclass relations for research.
4. Per-round global validation includes all arrived labels and (for domain scenarios) the full validation distribution of each observed client-domain-label slice. It never restricts the evaluation population to selected clients. The official CIFAR test split is evaluated only after final training and is not used for selection or tuning.
5. Main `forgetting` is nonnegative past-peak degradation only for classes with prior above-chance learned performance and global incorporation. `all_observed_peak_drop` separately records peak drops regardless of global learning. Both are distinct from all-observed accuracy and never-integrated accuracy.

## Stream controls

`class_subset`, `class_order_override`, `holders_override`, `class_holder_weights`, `class_sample_fraction`, `sample_overlap_fraction`, `client_sample_skew`, `class_imbalance_power`, `arrival_lag_by_client`, `recurrence_round_delay`, `recurrence_repeats`, `recurrence_period` and `participation_overrides` provide independently configurable interventions. Every unique training sample must be allocated exactly once unless deliberate `sample_overlap_fraction` creates cross-client sample duplication; intentionally unused samples from class-dose/imbalance controls are separately recorded. Recurrence splits a reserved training-only pool into disjoint return events; it fails rather than fabricates extra examples if the reserve cannot meet repeat counts.

Data heterogeneity is not a single scalar: the simulator separates class holders, overlap of class IDs, raw-sample duplication, quantity skew, feature-domain shifts, temporal recurrence, class imbalance, and participation correlations.

## Local algorithms and synchronization

GEM solves the convex nonnegative dual QP with a device-resident accelerated projected-gradient solver and primal/KKT residual checks; A-GEM uses a single reference gradient. A SciPy solver is used only as an independent numerical test reference. GEM groups memory constraints by observed *class*, local *experience stage*, or the combined historical memory. Class-aware boundary-free mode hides the stage when choosing GEM's historical memories but does not infer arbitrary new tasks or discover unknown labels.

Synaptic Intelligence tracks locally applied optimizer-step path contributions and consolidates importance at a locally observed stage transition. The global broadcast is not counted as local training movement. `optimizer_state_policy` supports reset/persist for local SGD momentum across global broadcasts; `reset_memory_on_sync` is a deliberately destructive replay-memory ablation rather than the default.

## Aggregation and selection

Sample-weighted FedAvg remains the baseline. Alternative aggregators are explicit controlled methods: equal-client, class-balanced contribution weights, inverse observed selection frequency (a heuristic, not unbiased propensity correction), conflict-penalized weights, old-gradient-retention weights using privileged training samples, and pairwise projection of conflicting model deltas. FedAdam and FedAvgM retain server optimizer state. These are experimental mechanisms, not claims of original-method or production privacy fidelity.

New selectors use train-only historical feature prototypes, train-only class gradients, held-out validation-history forgetting (privileged oracle), observed local GEM constraint counts, or recency/coverage heuristics. Client class metadata and validation-history selectors are privileged simulation controls and must not be described as privacy-preserving deployable policies.

## Evaluation and causal diagnostics

Per-round history records validation `accuracy`, `macro_accuracy`, per-class accuracy, old/new accuracy, `never_integrated_accuracy`, `forgetting`, `integrated_forgetting`, `all_observed_peak_drop`, per-class validation cross entropy, communication payload bytes, runtime and CUDA memory. Final official test metrics are saved separately. Optional probes include client old-memory loss before/after local training, validation loss before/after aggregation, per-client update cosine, old-class gradient/update alignment, and old-feature representation drift.

`aggregation_subset_report` evaluates all nonempty subsets of already accepted same-base-version local updates (plus the empty baseline), for small cohorts only, and computes pairwise finite-difference interactions. This is different from the multi-round causal fork simulator and must not be interpreted as full joint retraining of all possible client subsets.

## 512-batch sequential suite

Both shell launchers invoke the dynamic source planner and save each job in `results/<dataset>/runs/<job-id>/`. They independently preserve config, scenario, step metrics, checkpoints, final global model and status. `scripts/run_both_sequentially.sh` runs CIFAR-10 and CIFAR-100 **one after another**. To verify exact committed job manifests without downloading CIFAR: `python scripts/validate_plans.py`.

The separate CIFAR-10 pilot tools generate a 15%-training-data, batch-512 plan with distinct `PILOT_` IDs, run jobs sequentially, wait between jobs while another CUDA compute process is present, and verify the resulting round metrics, normalized aggregation weights, CUDA device records, checkpoints, data partitions, and reloadable final models. The GPU launcher first runs a CUDA interruption/resume regression and writes an environment receipt with Git, Python, PyTorch, CUDA, GPU, dependencies, and dataset hashes. Pilot results are not production results or tuning decisions without a separate validation ledger.

The CIFAR-10 tuning tools define a paired, validation-only full-participation SGD sweep over learning rate and momentum at batch 512, with a stratified 50% training subset and eight rounds per experience. Candidate ranking uses the old/new validation-accuracy harmonic mean, followed by all-observed macro accuracy, class coverage, forgetting and training cost. Official-test metrics are excluded. The selected setting is provisional until full-data, full-schedule representative validation passes.

The current plans were **generated, not trained**. Their count and 144 covered IDs do not imply 144 complete published research validations; see `STATUS.md` and the per-block catalogue.
