# Project status

## Current canonical primary FedAvg result

The primary CIFAR-100 Replay/FedAvg comparison is fixed to the ResNet-18, crop/flip, batch-64, 10-epoch, seed-0 protocol in [PRIMARY_PROBE.md](PRIMARY_PROBE.md). Client SGD momentum persists across tasks and communication. Server aggregation uses canonical sample-weighted whole-model FedAvg without classifier-row fusion.

The completed runs achieved **38.58%** with full participation and **30.58%** with rotating 4-of-5 participation; both integrated all 100 classes. The paired gain over the preceding matched V19 run was 17.92 percentage points for full participation and 15.86 points for rotating offline. This is one seed and does not establish statistical significance. The rotating arm uses the repository's explicitly marked stale-update adapter.

## Scope

The repository retains distinct continual-learning learners, factorial research configurations, and method-specific source-fidelity paths. They do not replace or alter the primary Replay/FedAvg recipe. Earlier protocol documents and run artifacts are historical records, not current primary launch instructions.

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
```
