# Correlation-Aware Federated Continual Learning

This research repository studies continual learning across clients that may train while disconnected, then upload accumulated updates. It includes local continual-learning methods, server aggregation studies, and source-fidelity adapters.

## Current Task-IL CIFAR-100 paired experiment

The primary ResNet-18 Replay/FedAvg experiment compares full participation with a randomly selected one-client outage. New configs use task-incremental heads: each delivered class set is a task ID, shared labels have independent parameters in each head, and overlap predictions average and renormalize selected-head probabilities. See the Task-IL protocol in docs/TASK_INCREMENTAL_PROTOCOL.md. Its stream configuration is:

- CIFAR-100 at native 32×32 resolution, with 450 training examples per class.
- Five independent clients and one server model in each experiment process.
- One two-class learning task per client at every learning sync, starting immediately at sync 1.
- Adjacent class pairs overlap across clients and carry into the next sync: `(0,1),(1,2),(2,3),(3,4),(4,5)`, then `(5,6),(6,7),...`.
- Twenty learning syncs cover all 100 classes. The final pair `(98,99)` is assigned to the last two client slots so all five clients train at sync 20; class 98 consequently has three owners.
- Each client retains up to 50 examples per class. The expected deduplicated server archive after the terminal flush contains 10,000 examples: 50 for class 0, 150 for class 98, and 100 for every other class.
- In the hidden arm, one client is selected randomly for no model upload at each learning sync. All five clients continue training; an omitted client keeps its last received model and Replay archive until it is selected again.
- Each incoming 64-image batch is split into two updates, each containing 32 current images and 32 replay images. Every replay image is sampled by choosing uniformly among classes in the client’s currently available archive, then uniformly among that class’s stored images.
- Five independent client models train concurrently per process on separate CUDA streams. The two arms launch as separate concurrent processes on the same GPU, for ten client models training concurrently and two separate server models.
- Each local task uses batch 64, 10 epochs, crop/flip train augmentation, and persistent client SGD momentum. A communication-only terminal flush follows the 20 learning syncs.

Run the current two-arm plan or launch it on GPU 0:

```bash
bash scripts/run_primary_probe_cifar100.sh --plan-only
bash scripts/run_primary_probe_cifar100.sh --gpu 0
```

The completed 46.78% full / 42.13% random-offline results and the earlier 38.58% / 30.58% result predate the task-head migration. They are historical class-IL measurements and do not validate Task-IL. A one-seed Task-IL pair is now running on Mtech. See [the current protocol](docs/PRIMARY_PROBE.md) and [project status](docs/STATUS.md).

## Code and documentation

- `src/cafcl/primary_probe.py`: fixed paired Replay/FedAvg experiment.
- `src/cafcl/streaming.py`: adjacent-chain and research stream schedulers.
- `src/cafcl/parallel_clients.py`: concurrent client training and task barriers.
- `src/cafcl/continual.py`: local learning, replay, and train-view augmentation.
- `src/cafcl/simulator.py`: simulation and server synchronization.
- `docs/PRIMARY_PROBE.md`: active primary protocol and historical-result boundary.
- `docs/STATUS.md`: current project status.
- `docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md`: separate upstream method-source provenance.

Replay shares raw training examples with connected clients. This simulator is not a privacy-preserving federated-learning system.

## Joint spatiotemporal FCL extension

The opt-in `cafcl.joint` v2 protocol separates atomic local continual tasks from server communication intervals, supports both Task-IL and Class-IL, and provides constructive joint curricula and experiment catalogs. See [current implementation and validation status](docs/joint/IMPLEMENTATION_STATUS.md). Run the plan-only compiler before any training:

```bash
PYTHONPATH=src python -m cafcl.joint.runner plan --suite foundational --output runs/joint_foundational
```

The historical primary protocol is unchanged. Do not launch the generalized production campaign until the joint-specific CPU validation and short real-data diagnostics have been checked.
