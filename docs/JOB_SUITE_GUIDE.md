# Experiment ledger and recommended execution order

**Run the two-job preliminary pilot first.** Do not launch the 38,370-job factorial suite to answer the initial research-direction question.

- Primary probe: `bash scripts/run_primary_probe_cifar100.sh --plan-only` (always exactly two jobs: baseline 5/5 visible, versus rotating 4/5 visible; seed 0; A-GEM + FedAvg; no overlap; 100 unique classes over five intervals; final flush).
- Native default backbone: `--backbone vit` for 32×32 ViT-Tiny/16. Switch to `--backbone cnn` for 32×32 ResNet-18 with CIFAR stem. Both retain batch1024. Different backbones constitute separate studies, not additional arms hidden within the two-job pilot.
- After preliminary review: `bash scripts/run_all_cifar100.sh --backbone vit --plan-only` for 38,370 feasibility-screened jobs with all 99 local/server method pairings. The separate CNN plan has the same number of configurations and a different SHA256.
- The exact frozen manifests and overlap/outage/selection feasibility audit are in `configs/designs/stream_v6_native32.json`.

**Old CIFAR-10 plans and 20-class jobs remain retired.** CIFAR-100 streams end when the 100 distinct labels have been allocated; the default 10-client disjoint curriculum naturally lasts 3 communication intervals (40+40+20), not five full 40-class intervals. The pilot uses five clients to produce exactly five full intervals.

See [`PRIMARY_PROBE.md`](PRIMARY_PROBE.md), [`BACKBONES.md`](BACKBONES.md), [`FACTORIAL_SUITE.md`](FACTORIAL_SUITE.md), and [`STATUS.md`](STATUS.md).


## v10 execution invariant

Every active CIFAR-100 job now schedules independent client model instances concurrently **at each task ordinal**, with a full barrier between tasks, before any server update. The two-arm primary pilot launches both whole federated simulations concurrently on the same GPU (`--gpu 0`). There is no client-sequential training path in the active simulator. Results use v10-specific paths; see `PARALLEL_SINGLE_GPU_V10.md`.
