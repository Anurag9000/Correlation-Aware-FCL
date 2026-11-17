# Implementation status — 2026-11-17

**Executable prototype; not complete research closure.**

- 152 planned experiment blocks in source catalogue.
- 73 blocks have infrastructure or partial method implementation, of which 51 are explicitly marked partial/approximate.
- 63 distinct block IDs have executable jobs in the current generated plans.
- 276 CIFAR-10 and 278 CIFAR-100 configurations, all with batch size 512 and separate job IDs/results/model artifacts.
- 35 offline tests passed (including synthetic model training, GEM projection, FedAdam/FedAvgM, selection, counterfactual diagnostics, and exact resume after an injected interruption).
- No official CIFAR full-suite training was executed in this authoring environment. These job counts are not reported experimental findings.

Implemented core: event-based data arrivals for active/inactive clients, CIFAR loaders, non-IID/overlap/semantic proxy/recurrence scenarios, selected-client pre/post synchronization, FedAvg/uniform/FedAdam/FedAvgM aggregation, local SGD/GEM/A-GEM/replay/EWC/ER-ACE-style/DER/DER++/MAS/LwF objectives, optional local FedProx, historical-memory persistence, complete globally observed class test evaluation, class retention/coverage, several simulation-privileged selectors, adaptive participation schedule, full-state checkpoint/resume, aggregation-only leave-one-out counterfactuals, **multi-round full-state include/exclude client forks**, and train-only gradient/feature relationship diagnostics.

**Major remaining work:** precise implementations for the 89 blocks without scheduled jobs; fully controlled feature/gradient-sign interventions, true task-free hidden-boundary learning and open-world expansion; SCAFFOLD and official baselines; broader higher-order causal interventions beyond the implemented checkpoint forks; sample-level client-domain test evaluation; formal privacy/secure aggregation; stronger resource monitoring; full protocol-specific ablations; rigorous external reproduction, independent trials, confidence intervals, and actual CIFAR performance measurements. See `docs/EXPERIMENT_CATALOGUE.md` and `docs/TECHNICAL_SPEC.md`.

The use of `implemented` means a software mechanism exists, **not** that its paper results were reproduced or that it has no numerical limitations. Simulation-privileged client-selection and gradient analyses are diagnostic/oracle and cannot be called private deployable algorithms.
