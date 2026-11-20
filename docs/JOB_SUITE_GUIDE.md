# Experiment entry points

## Canonical primary Replay/FedAvg comparison

The only primary FedAvg recipe is the fixed CIFAR-100 ResNet-18 experiment in [PRIMARY_PROBE.md](PRIMARY_PROBE.md): seed 0, train-only crop/flip, Replay with 50 examples/class, batch 64, 10 epochs/task, persistent local momentum, nine learning syncs plus one communication-only flush, and full versus rotating 4-of-5 participation.

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The primary runner does not expose alternate optimizer, replay, augmentation, backbone, or seed variants. The completed one-seed outcomes were 38.58% full participation and 30.58% rotating offline. See the protocol for caveats and per-sync artifacts.

## Other research suites

The repository also contains independently configured local continual-learning methods, factorial plans, and server aggregation studies. They remain separate research experiments and do not replace or alter the primary FedAvg recipe. Method-specific source aggregation for the native GLFC, LGA, FEAT, and FedCLASS pilots is documented in [the original-source fidelity audit](ORIGINAL_SOURCE_FIDELITY_AUDIT.md).

Earlier v6–v20 protocol documents, manifests, and result files are archival records. They must not be used as current launch instructions for the primary comparison.
