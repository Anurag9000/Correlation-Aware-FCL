# CA-FCL technical and research protocol

## Temporal contract

Maintain independently: communication round `r`, stream data-arrival time, and the local experience index of every client. Stage boundaries in the simulator schedule are **not** necessarily admissible to a task-free learner. Scenario sampling, data arrival, client selection, and evaluation population building must not share a mutable view of future test labels.

## Round state machine

1. Compile an immutable scenario and disjoint sample allocations. For every round, deliver all due data to clients regardless of participation.
2. Record `globally_observed` labels; the evaluator immediately includes all corresponding official held-out test images. Record `globally_integrated` separately.
3. Select a budget-exact set of clients using the configured selection rule. Privileged selectors use simulated holdings and must be labelled `oracle/diagnostic` rather than privacy-preserving.
4. Synchronize the selected clients from the committed global model (rejoining clients never train stale weights under the synchronous protocol).
5. Train each selected client independently with its accessible current samples and permitted private memory. Upload a version-tagged local model and sample-count metadata.
6. Aggregate accepted updates. FedAvg uses normalized participating-client sample counts. FedAvgM/FedAdam apply stateful server optimization to the averaged client delta.
7. Validate model tensors, then *synchronize all selected clients* with the new committed global model. `full_sync` additionally updates inactive clients; client private memory is not erased.
8. Evaluate over all globally observed official held-out test classes, including those on inactive clients. Append per-round/per-class metrics and diagnostics; write durable checkpoints.

## Formal quantities

Let `C` be clients, `P_r ⊆ C` selected clients, `D_{k,e}` local experiences, and `W_r` the canonical server model. Local training yields `W^k_{r+1}`. FedAvg outputs `W_{r+1} = Σ_{k∈P_r} n_{k,r} W^k_{r+1} / Σ_{k∈P_r} n_{k,r}`. The independent global observation set `Y_obs(r)` is the union of classes arriving at **all** clients by that round. The evaluation bank contains every official held-out test sample whose label lies in `Y_obs(r)`, irrespective of train participation. A class may be observed but never globally integrated. Such classes contribute to global accuracy, not automatically to a forgetting event.

The implementation records class-wise `accuracy_c(r)`, previous-peak drop, old/new macro accuracy, global micro/macro accuracy, never-integrated classes and coverage. Compare identical held-out class slices to make a forgetting claim; never subtract global accuracies across changing evaluation populations and interpret the result as pure forgetting.

## Local methods

- **SGD**: cross-entropy on accessible local samples; optimizer reconstructed per selected round.
- **Replay**: mixes current accessible training samples with permitted private-memory indices. Record repeated presentations separately from unique examples.
- **GEM**: computes local reference gradients per remembered historical class; solves the convex nonnegative dual QP and verifies all primal dot-product constraints before SGD update. CPU SciPy solver is a resource cost.
- **A-GEM**: projects current gradient onto a single reference-gradient halfspace when the dot product is negative.
- **EWC/MAS**: local parameter importance and anchors survive global broadcasts; stage consolidation uses a bounded local data sample. These are documented adaptations rather than exact original-method reproductions.
- **ER-ACE-like, DER, DER++, LwF**: local stream masking / stored logits / historical-teacher outputs with the limitations in README. Historical logit memories remain private; dynamic expanding heads are not implemented.
- **FedProx**: adds `mu/2 * ||W_local - W_downloaded||²` to the local objective. It is not a server averaging rule.

## Correlation mechanisms

1. **Old–new semantic relation**: select related/unrelated class orders using CIFAR-100 coarse hierarchy or explicitly documented CIFAR-10 heuristic pairs. This is a *proxy*; actual feature and gradient similarity can differ.
2. **Same-round cross-client overlap**: assign identical labels to multiple clients with disjoint training samples by default; control the number of holders and overlap intensity.
3. **Cross-client temporal recurrence**: reserve disjoint training samples of existing classes and deliver them at later rounds to the same or another client. Relative recurrence delays and whether the recurring client participates are separate interventions.
4. **Combined scenarios**: mix relationships, client heterogeneity and asynchronous class arrival without conflating label overlap with sample duplication.
5. **Diagnostics**: use train-only feature means, train-only class gradients, label Jaccard, temporal co-occurrence and client contribution analysis. These are offline analyses; the test set may not influence online selection.

## Counterfactual semantics

`aggregation_leave_one_out` evaluates alternatives from the *same accepted local-update pool*, comparing all-client aggregation with a reweighted aggregate excluding one client. This is a valid retrospective aggregation intervention **not** a multi-round train-or-not-train counterfactual. It excludes nonexistent client gradient history and does not model future local memory divergence. A separate `fork` runner implements actual multi-round causal state forks from a saved checkpoint for fixed-budget one-round client include/exclude interventions, maintaining separate future client memories and model states. Use matched checkpoint, data arrival, compute budgets, client substitutions, and preserved RNG streams for any future trajectory-level experiment.

## Experiment execution contracts

`RunConfig` is immutable and validated. Each independent `job_id` maps to a unique resolved config; collisions fail. The suite runner executes configurations in order, saving independent metrics/checkpoints and recording failures. `--plan-only` produces an actual JSON plan before execution. CIFAR jobs require batch size 512. Changing seed requires a separate results root. A full-state latest checkpoint stores the global model, client states, observed/integrated labels, per-class peaks, history, selection RNG state, server adaptive-optimizer state, and Python/NumPy/PyTorch RNG states. Resumption only accepts a matching configuration.

## Invalid scientific inferences to avoid

- A reduction in old-class forgetting with worse new-class learning does not automatically represent a better continual learner.
- Excluding an as-yet-unlearned class can lower full global accuracy without forgetting it.
- High semantic similarity does not guarantee gradient compatibility.
- Local GEM constraints do not guarantee global retention after aggregation, particularly for classes never held by selected clients.
- A client-selection policy using omniscient simulator labels or global test outcomes is not a deployable privacy-preserving algorithm.
- 152 *research questions* do not imply 152 completed implementations or 152 independent job runs.

## Essential acceptance tests

Verify: exact FedAvg weighting, exact active-client post-sync, data arrivals to inactive clients, complete global class test coverage, monotonic class observation sets, GEM/A-GEM projection correctness, per-class historical peak metrics, class overlap, disjoint-sample recurrence, selector budget, adaptive server optimizer finite state, independent job artifacts, checkpoint resumption, and diagnostic counterfactual reproducibility. Test the complete system offline with synthetic data before downloading CIFAR datasets.

## Multi-round causal forks (implemented)

A baseline run with `checkpoint_every=1` stores round-level full-state checkpoints. The dependent L02 causal suite reads a designated checkpoint, clones the entire global model, all clients' memories, optimizer and algorithm states, data-arrival history, global retention history, and RNG streams. Two continuation jobs differ only in a prescribed one-round include/exclude-client selection intervention and preserve the participating-client count. Subsequent local experiences and global updates evolve independently. Fork validation rejects data, optimizer, architecture, or scenario changes that would confound this experiment. These controlled branches estimate the effect of that one-round intervention on future trajectories; they do not directly establish the causal mechanism of class similarity, which requires separate data-relation interventions.

Example CLI for a baseline with generated config and first-stage checkpoint:

```bash
python -m cafcl.cli fork --config path/to/original_config.json \
  --checkpoint path/to/checkpoint_round_0002.pt \
  --client 2 --arm include --round 2 --job-id intervention_include \
  --output-dir results/counterfactuals
```

The regular `all-supported` suite runs L02 fork baselines immediately before their dependent inclusion/exclusion arms, so the two CIFAR launch scripts execute these as part of their sequential job plans.
