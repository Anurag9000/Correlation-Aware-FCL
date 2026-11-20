# Experiment entry points

## Current primary Replay/FedAvg comparison

The active primary recipe is the CIFAR-100 ResNet-18 configuration in [PRIMARY_PROBE.md](PRIMARY_PROBE.md): adjacent overlapping two-class client tasks from the first sync, 20 learning syncs, 50 replay examples/class/client, batch 64, 10 epochs/task, persistent local momentum, and full participation versus a seeded random one-client outage. All five client models train concurrently in each of two concurrent arm processes; the hidden arm withholds one randomly selected model upload per learning sync while all clients keep learning and receive the aggregate replay archive.

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The run artifacts use a V23-specific root. The earlier 38.58% / 30.58% scores belong to a nine-sync disjoint-class sampled-replay protocol and are not results for the current recipe.

## Other research suites

Native GLFC, LGA, FEAT, and FedCLASS runners construct their paired schedules from the primary adjacent-chain plan while retaining method-specific losses, memories, and model implementations. Their source adaptations and limitations are documented in [the original-source fidelity audit](ORIGINAL_SOURCE_FIDELITY_AUDIT.md). Other factorial and historical stream configurations remain separate experiments.

Earlier v6–v22 protocol documents, manifests, and result files are archival records unless explicitly identified as current above.
