# FedCLASS authors-source CE/FedAvg pilot (fidelity hold)

The selected method is the **original FedCLASS repository's CE/FedAvg
baseline**, not the paper's proposed `OURS` self-distillation. The original
ResNet-8 + SimpleLinear, `CE._train`, original first-k exemplar selection and
original sample-size FedAvg source are now invoked in the pilot.

`src/cafcl/upstream_fedclass_ce.py` adapts the original source to the
**five-client / first-four-then-two / 9+U10** curriculum, full batch64,
and cumulative delayed offline updates. It currently retains the canonical
fixed 100-row output head and a separate experimental server protocol;
`FlTrainer.fl_train` is not executed. It is **not yet certified as a
complete original-author trainer reproduction**.

## Memory policy: do not mislabel it

`--memory-mode authors_exemplars` (default) executes the author's
`BaseLearner._construct_exemplar_unified` and chooses the **first 50 eligible
TRAIN examples/class** in dataset order, NOT Algorithm R and NOT 50/50
mixed sampling. `--memory-mode none` turns off replay while using
`CE._train`. The source recreates its SGD optimizer and AMP scaler each
task instead of persisting client momentum across tasks. Learning rate,
weight decay and number of epochs are user protocol settings.

`--replay-ratio` only accepts the backward-compatible value 1; there is no
source FedCLASS parameter controlling this ratio.

## Safety and provenance

All original source files are byte-identical to pinned upstream commit
`e03e30653165b76ddf980393794f395f838ee967`.
New source-runner experiment identity is `SOURCE_V2`; results reside under
`results/fedclass_pilot/authors_exemplars/seed0_lr0.03_epochs10_source_v2`
or the corresponding `none` path. Previous ER50/Algorithm-R V1 artifacts
are distinct and are not silently reused.

The source-fidelity gate **blocks the full GPU launch by default** until
all remaining original-method mechanisms have been reviewed and verified.
`--permit-adapted-legacy` is only for explicitly incomplete ablations;
it does NOT confer paper fidelity.

```bash
bash FedCLASS/pilot/run.sh --plan-only
bash FedCLASS/pilot/run.sh --memory-mode none --plan-only
```

Details: [full author-fidelity audit](../../docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md).
