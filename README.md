# Correlation-Aware Federated Continual Learning

This research repository studies continual learning across clients that may continue training while disconnected, then upload accumulated updates. It includes several local continual-learning methods and separate server aggregation research baselines.

## The canonical primary FedAvg recipe

The primary CIFAR-100 comparison has one fixed recipe for both participation arms:

- CIFAR-stem ResNet-18 at native 32×32, randomly initialized
- Standard CIFAR random crop with four-pixel zero padding and horizontal flip on every training read, including Replay reads
- Replay with a 50-example-per-class Algorithm R reservoir
- Sample-weighted whole-model FedAvg, including every classifier row; no classwise head fusion
- SGD with learning rate 0.03, momentum 0.9, weight decay 0.0005; batch size 64; 10 epochs per task; persistent client momentum
- Five clients, nine learning syncs, then one communication-only flush
- Full participation versus rotating 4-of-5 participation, seed 0

The completed paired run reached **38.58%** with full participation and **30.58%** with rotating offline participation. Both models integrated all 100 classes. This is a one-seed result, not a statistical estimate. The rotating arm uses the repository's marked stale-update adapter because offline clients continue training from an older model; it is not synchronous textbook FedAvg. See [the fixed protocol and result](docs/PRIMARY_PROBE.md).

Run this exact comparison on one GPU:

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The primary runner fixes its model, training views, optimizer policy, replay budget, task stream, and seed so candidate variants cannot be mistaken for this result. Other local learners and server methods remain separate research experiments. Their method-specific aggregation paths are separate suite implementations and do not define another primary Replay/FedAvg recipe. See the [method source fidelity audit](docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md) for their scope and current limits.

## Code and documentation

- `src/cafcl/primary_probe.py`: fixed paired Replay/FedAvg experiment
- `src/cafcl/simulator.py`: simulation and the shared canonical FedAvg equation
- `src/cafcl/continual.py`: client-local learning and train-view augmentation
- `docs/PRIMARY_PROBE.md`: primary protocol, empirical result, and limitations
- `docs/STATUS.md`: current project status
- `docs/SERVER_ALGORITHM_FIDELITY_V11.md`: server-method fidelity audit
- `docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md`: separate upstream method-source provenance

Replay shares raw training examples with connected clients. This simulator is not a privacy-preserving federated-learning system.
