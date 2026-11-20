# FedCLASS CE + FedAvg — fully connected versus round-robin offline pilot

This is the **FedCLASS baseline port** into the v16 `Correlation-Aware-FCL`
class-incremental protocol. The original **25 upstream tracked files** in
`FedCLASS/` are unchanged; all new files are under `FedCLASS/pilot/` and
`src/cafcl/` (and `tests/`). The original source snapshot is pinned to
[skylorh/FedCLASS](https://github.com/skylorh/FedCLASS) commit
`e03e30653165b76ddf980393794f395f838ee967`.

## CLI

From the project root, install the project and test dependencies:

```bash
python -m pip install -e '.[test]'
bash FedCLASS/pilot/run.sh --plan-only
bash FedCLASS/pilot/run.sh
```

**Default**: two concurrent CIFAR-100 jobs, fully connected versus rotating
offline, using the FedCLASS CE + FedAvg algorithm with the established
50-example-per-class Algorithm R experience replay. Both processes use
**physical GPU 0**, and within each process all five clients train
concurrently at each ordinal task barrier.

**Vanilla FedCLASS CE + FedAvg without replay** (also two concurrent arms):

```bash
bash FedCLASS/pilot/run.sh --memory-mode none --plan-only
bash FedCLASS/pilot/run.sh --memory-mode none
```

For multiple independent repetitions use `--seed N`. Optional arguments:
`--epochs 10 --lr 0.03 --replay-ratio 1.0 --data-root data --output-dir
results/fedclass_pilot --no-download`. The replay ratio is permitted only
for the default reservoir mode. `--gpu` must be `0`.

**Do not launch both modes at the same time on an undersized GPU.**
The full and rotating arms within *each* command are automatically
concurrent on GPU 0. Failure of either aborts the pair, and production
runs never silently shrink batches or fall back to CPU.

Outputs are separated by memory mode, seed, learning rate, epoch count
and replay ratio:

```text
results/fedclass_pilot/
  reservoir/seed0_lr0.03_epochs10_rr1/
    design.json
    job_0.log
    job_1.log
    runs/FEDCLASS_CE_ER50_FULL_SEED0/
    runs/FEDCLASS_CE_ER50_ROTATING_OFFLINE_SEED0/
    comparison/{summary.json,per_class.json,per_class.csv,comparison.md}
  none/seed0_lr0.03_epochs10/
    ...
```

Each individual run saves the deterministic stream, full model/client/optimizer
state checkpoints, cumulative server CL state, per-round metrics and
per-synchronization class-IL reports. The comparison checks that **all 100
classes** have been integrated by the final flush.

## Exact stream and communication intervention

- CIFAR-100 native 32×32; official test partition, separate 10% stratified
  validation carved **only from original TRAIN**, leaving 450
  TRAIN examples/class. No test data is used to train, select clients or
  construct memory.
- Five persistent clients; seeded, nonoverlapping class stream. At U1 each
  independently completes two successive local two-class episodes
  (4 distinct classes/client, 20 globally). From U2 through U9 every
  client receives exactly one new two-class episode (10 globally/interval).
  U10 is an **all-client communication-only flush**, no new training.
- `FULL`: all five learn, upload and receive the consolidated state at
  every U1--U9; all five synchronize again at U10.
- `ROTATING_OFFLINE`: every client still trains its assigned episode;
  **exactly one of five** cannot upload or download at each U1--U9.
  A seeded permutation cycles excluded clients across all five.
  Unsent model displacements, accumulated class evidence and local memory
  are retained until the next authorized upload. U10 completes the
  outstanding integration.
- Both runs have identical source seed, independent but equal initial
  model tensors, data allocations, local epochs, optimizer settings and
  absolute class IDs. The **only deliberate arm difference** is communication
  visibility. Offline-client training must not consult the updated global
  model or other clients' newly uploaded classes.
- **Task-ID-free class-incremental inference**: one shared 100-column
  classifier; predictions compete across all globally arrived labels,
  *not* the oracle task's two labels. Report the all-arrived accuracy
  including classes still hidden, and server-integrated-only accuracy.
- Physical training and diagnostic/evaluation forwards of **64 samples**
  per step, padding short final batches with TRAIN examples and correcting
  evaluation denominators for padded examples.
- Local SGD with momentum 0.9 and weight decay 0.0005, optimizer
  state persisted across tasks. All classes known from local learning
  or a successful server download compete in the cross-entropy loss.
  Unknown future classes do not.
- Whole-model, sample-weighted FedAvg for freshly synchronized clients.
  The prior v16 server explicitly applies the noncanonical
  `G_current + (L_client - A_last_download)` stale-delta rebase for a
  returning offline client, with version/provenance metadata. This is an
  **experimental adaptation**, not an algorithm from original FedAvg.
- All accepted client learning states are acknowledged exactly once;
  online clients receive the accepted server-global CL history.
  Disconnected clients receive **nothing** until their next sync.

## Source-method fidelity

| Component | FedCLASS upstream | This pilot |
|---|---|---|
| Local baseline | `models/ce.py`: SGD + cross entropy | Same CE objective over explicitly *locally known* sparse class IDs |
| Backbone | `convs/resnet_client.py`: ResNet-8 | Imports the **exact original ResNet-8 implementation** |
| Classifier | `convs/linears.py`: `SimpleLinear` | Imports the **exact original** initialization and module |
| Aggregation | `utils/fed_algorithm.py`: sample-weighted FedAvg | Same weighted model mean for fresh updates; documented stale-delta adapter for returnees |
| Original stream | globally contiguous new class blocks, random client sampling | Our immutable five-client first-four-then-two stream and seeded rotating blackout |
| Classifier growth | `CENet.update_fc` on globally synchronized tasks | Fixed globally indexed 100-row classifier from initialization, with all not-yet-known output columns masked **during training only** |
| SGD optimizer | recreated for every original local training call | Persistent momentum in v16, per our experiment |
| Replay | original CE supports optional memory selection in `BaseLearner` | Primary 50/class independent Algorithm R, server union; optional `none` removes it |
| Inference | argmax global class logits | no per-sample task ID, global all-arrived class-IL evaluation |

The primary default **is FedCLASS CE + ER50/FedAvg**, *not*
pure vanilla FedAvg. To compare strict bare FedAvg without old-class exemplars,
run `--memory-mode none`. Likewise, class-stream changes, fixed-size global
head, different optimizer state persistence and noncanonical stale deltas
mean these are **protocol-harmonized adaptations**, not bitwise reproduction
of any published FedCLASS experimental table.

The source files are never modified, re-sampled or re-labeled without a
recorded, deterministic manifest. All variants are explicitly marked.

## Verification and integrity

```bash
pytest -q tests/test_fedclass_offline_pilot.py
pytest -q
python scripts/validate_factorial_plans.py
```

The tests assert: native-source model/classifier, 100-class logits,
50-class-indexed examples vs zero memory, strict 100-class schedule,
identical intervention-pair configurations, deterministic offline
rotation, 2 processes simultaneously on GPU 0, all five clients training
while one is hidden, and offline completion plus full final integration
in a small synthetic simulation.

Note that GitHub Actions CPU testing validates code and protocol invariants,
**not** actual two-arm CUDA memory capacity or final CIFAR-100 accuracy.
Full production measurements must be run on a CUDA-equipped machine.
