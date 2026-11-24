# Class-incremental Replay/FedAvg accuracy experiment log

This Kakarot-side file tracks the server-only experiment cycle. Implementation,
plan validation, training, artifact inspection, and result analysis happen on
MtechDEServer. Run only one experiment at a time; wait for completion, inspect
all per-sync evidence, record the diagnosis and next controlled variant here,
then proceed. Do not use this log as evidence that a server job is currently
running; verify live state on Mtech.

## Objective and invariants

- Optimize the **fully seen, all-client class-incremental arm**. The hidden arm
  is secondary; a larger full-versus-hidden accuracy gap is acceptable and
  preferred when the full arm improves.
- Keep client learning as ordinary Experience Replay using stored images and
  server aggregation as whole-model FedAvg. Do not replace either method.
- The target base stream is five clients, one synchronization after each
  client's full local task, CIFAR-100 class-incremental learning.
- Change one experimental factor at a time when testing a mechanism. Give each
  variant a unique job ID and output root; record the exact code revision,
  configuration delta, seed, status, metrics, and artifact paths.
- Each completed run must be reviewed with per-sync accuracy and a client ×
  sync table containing the classes introduced for that client and the
  corresponding server-model per-class accuracy. Distinguish this from
  client-local model accuracy.

## Base reference

| Run | Full arm | Hidden arm | Protocol | Evidence |
|---|---:|---:|---|---|
| V25 seed 0 | 46.78% test / 46.08% validation | 42.13% / 42.12% | 20-sync adjacent-chain class-IL; image Replay; FedAvg; terminal communication flush | Server `docs/STATUS.md` and `docs/PRIMARY_PROBE.md`; per-sync results were documented under `results/primary_v25_fresh_seed0/` |

The recorded sync-10 scores were 42.47% test / 43.80% validation for full
participation and 38.78% / 40.59% for random-offline. These are historical
recorded results; the raw V25 result directory was absent from its documented
location during the 2026-11-22 inspection, so recoverable per-sync files must
be re-inventoried before claiming they are available.

## Server state at start of this investigation

- Server: MtechDEServer1, checkout `/home/anurag24092/Projects/Correlation-Aware-FCL`.
- The server checkout contains the V25 runner and numerous untracked
  experiment/queue scripts. Preserve these; inspect queued-run state before
  launching anything.
- At 2026-11-22 12:20 IST, before launch:
  Three GPU processes from other campaigns were present (PIDs 20842, 20844,
  32142); together they used about 7.7 GiB of the V100's 32 GiB while GPU
  utilization reported 100%. This class-IL run was then launched with those
  pre-existing workloads still active. A separate task-IL campaign controller
  and several old queue watchers were also present; none were modified.
- Monitoring cadence: inspect freely before launch and during the first 10
  minutes after launch. If the run is healthy for those 10 minutes, check
  status at intervals of at least 30 minutes. Check immediately when an error
  or anomaly is observed.

## Static code audit used for V26-E20

- The V25 `make_jobs()` full arm is the base: ResNet-18 with the CIFAR stem,
  random initialization, CIFAR normalization and crop/flip augmentation;
  five clients train behind a task barrier before each server sync.
- Local SGD is learning rate 0.03, momentum 0.9, weight decay 0.0005, with
  client momentum state persisted across local experiences. The inspected
  local training path has no learning-rate scheduler, so V26-E20 changes only
  epochs while keeping the 0.03 rate fixed.
- A model download replaces `shared_weights` and the sync anchor but does not
  clear `optimizer_state`; the next local SGD instance reloads that state.
  Thus persistent momentum really does survive each full-arm FedAvg broadcast.
- Replay uses 50 Algorithm-R exemplars per class/client and class-uniform
  sampling. With a 1:1 replay ratio, each incoming 64-image batch is split
  into two 32-current + 32-replay optimizer updates once history exists; the
  first local experience has no old archive to sample. The full aggregate
  image archive is sent to selected clients at sync, while each update samples
  a minibatch from that archive rather than replaying the entire archive in
  one optimizer step.
- Server aggregation is sample-weighted whole-model FedAvg with classwise
  head fusion disabled. The model's floating BatchNorm buffers are part of
  that whole-state averaging; this remains a candidate mechanism to study
  after the epoch comparison, not a change in V26-E20.
- There is one class-indexed 100-row head, not task-specific heads. Local
  Replay loss competes across all classes the client has observed; future rows
  are frozen. Global evaluation masks only classes that have not arrived and
  has no task-ID oracle. This confirms the run's class-IL protocol.
- CIFAR-100's 500 official training images/class are split into 450 training
  and 50 validation images/class. The stream partitions the 450 training
  indices across repeated class occurrences, so overlapping client ownership
  does not imply duplicated raw training images. Each client reservoir holds
  up to 50 images/class; the server merges accepted reservoirs and broadcasts
  the complete archive to every selected client.
- The chain has 200 client-class slots: class 0 occurs once (450 source
  examples for its owner), classes 1-99 occur twice except class 98, which
  occurs three times. Thus ordinary shared classes contribute 225 distinct
  current examples per owner and class 98 contributes 150 per owner. Use the
  per-client table to distinguish accuracy changes caused by task arrival
  from those caused by unequal source examples or cumulative replay.
- At the final full-participation sync, the deduplicated server archive is
  expected to contain 10,000 images (50 for class 0, 100 for most classes,
  and 150 for class 98); every client receives that complete archive. Its
  local Replay sampler therefore sees the union archive; repeated sample IDs
  from its own private buffer are deduplicated. The inspected server merge has
  no additional global exemplar cap and stores payloads for the accepted IDs.
- `per_sync_accuracy/sync_###.md` and `.png` are cumulative client-by-sync
  tables. Their per-class accuracy values come from the current global server
  model on the held-out test split; they are not private client-model scores.
  Aggregate validation accuracy is also written per sync and is the selection
  signal; test diagnostics remain reporting-only.
- Before launch, `py_compile` passed for the dedicated runner, its `--plan-only`
  output confirmed the 10-to-20 epoch-only delta and fixed protocol, the
  primary launcher passed `bash -n`, and `git diff --check` passed. The full
  pytest suite was not run; these are preflight checks, not a repository-wide
  test pass.

## Variant ledger

| ID | Hypothesis / single change | Arm | Server job ID / output root | Status | Full test / validation | Sync table reviewed | Interpretation / next decision |
|---|---|---|---|---|---|---|---|
| V25 reference | Existing image Replay + whole-model FedAvg base | Full | `PRIMARY_FULL_REPLAY_RESERVOIR11p1111_V25_CHAIN20_resnet18_cifar32`; `results/primary_v25_fresh_seed0/` | Completed (historical record) | 46.78% / 46.08% | Raw artifacts need re-inventory | Reference only; do not merge with Task-IL scores |
| V26-E20 | Increase local epochs from 10 to 20; all other V25 full-arm settings fixed | Full only | `CLASSIL_FULL_REPLAY_FEDAVG_CHAIN20_RESNET18_E20_SEED0`; `results/class_il_full_epochs20_v26_seed0/` | Completed 2026-11-22 13:03:52 IST; 20 learning syncs + terminal communication-only flush, all five clients trained every sync, all 100 classes integrated | 51.59% test / 52.84% validation; V25 reference 46.78% / 46.08% | Reviewed; see `analysis/client_by_sync.md`, `analysis/sync_summary.md`, `analysis/class_trajectory.csv`, `analysis/diagnosis.md` | +4.81 test and +6.76 validation points over V25. Highest raw sync validation was 53.45% at sync 13 over 66 seen classes; per-sync aggregate validation has a changing class denominator. Sync 20 added classes 96–99 at 3.00% average test accuracy; classes 0–95 averaged 53.61%. Their mean true-logit margin was −3.19 even though their mean classifier bias was +0.052 vs −0.002 for classes 0–95, pointing toward weak feature/class discrimination rather than simply low new-row bias (diagnostic association, not causal proof). At sync 20, 52/96 previously seen classes fell from sync 19 even as their mean improved +1.60 points; this is selective class forgetting masked by aggregate gains. Final peak-drop diagnostics: 9.75% all-observed and 12.38% integrated-class. Classes 90–95 averaged 12.83% test at sync 19 and 50.67% after sync 20 with another full-client Replay/FedAvg round. Paired across 96 classes with a later sync, accuracy rose from 3.93% at first arrival to 49.91% one sync later (+45.98 points; 94/96 improved). This supports testing post-arrival replay consolidation, though it is observational because other classes arrive at the same sync. This motivates a possible terminal replay-only consolidation variant after V27, if its tail remains weak. The communication-only flush made no accuracy change. |
| V27-E30 | Increase local epochs from 20 to 30; keep LR 0.03 and remaining V26 settings fixed | Full only | `CLASSIL_FULL_REPLAY_FEDAVG_CHAIN20_RESNET18_E30_SEED0`; `results/class_il_full_epochs30_v27_seed0/` | Completed 2026-11-22 14:09:35 IST in 48m36s; 20 task syncs plus the communication-only terminal flush (21 reports); all five clients trained each task sync and all 100 classes integrated. No OOM or logged exception. All 13 captured source fingerprints still match. | **54.13% test / 55.58% validation**, +2.54 / +2.74 pp vs V26-E20. Peak allocated CUDA memory 2.77 GB. | Fully reviewed; 21 JSON/CSV/MD/PNG sync reports and analysis saved on Kakarot and server: `analysis/client_by_sync.md`, `analysis/sync_summary.md`, `analysis/class_trajectory.csv`, `analysis/class_age_summary.md`, `analysis/comparison_v26_v27.md`. Transferred 95 non-checkpoint files (5.4 MB); excluded 24 model/checkpoint files (about 30.7 GB). | V27 validation exceeded V26 at every sync. Final classes 0–95 mean rose 2.71 pp; classes 96–99 fell 1.50 pp (3.00%→1.50% mean). Across the 100 classes, 61 improved, 6 tied, 33 declined. For 96 classes with a subsequent sync, first-arrival mean was 3.18%, rising to 55.05% after one more Replay/FedAvg sync (+51.88 pp; 94 improved, 1 declined). This strongly motivates a controlled terminal replay-only training sync. The existing final flush is communication-only. |
| V28-TermReplay | Add one full-client training sync over each client’s complete synchronized archive after the 20 class tasks; retain V27’s 30 epochs for normal tasks, use one archive pass in the added phase | Full only | TBD; output root TBD | Hypothesis selected; not implemented/launched yet | Pending | Pending | Test whether an extra simple Replay/FedAvg pass rescues classes 96–99. Must represent the archive-only event explicitly; current scheduler performs no training with an empty stream event. Use final validation to select and report test only. |

## Reporting checklist for each completed variant

1. Verify `status.json`, `summary.json`, and the complete per-sync report set;
   keep logs and checkpoints separate from scalar results.
2. Record final seen-class test and validation accuracy, integrated class
   count, and per-sync trajectories.
3. Build/review the table with syncs as columns and clients as rows. Include
   each client's classes at that sync, their server-model class accuracies,
   and pending-upload labels where applicable.
4. Compare old-class accuracy before/after each sync to identify forgetting;
   distinguish acquisition, server integration delay, and local client
   forgetting. Do not infer client-local accuracy from server-model reports.
5. Write a concrete diagnosis and one next controlled hypothesis before
   launching the next job. Preserve failed runs and their traceback.
6. Sync the run's design, provenance, status, logs, metrics and complete
   per-sync JSON/CSV/Markdown/PNG reports from Mtech to its unique Kakarot
   results directory. Exclude model/checkpoint files (`.pt`, `.pth`, `.ckpt`,
   and equivalent formats); never overwrite an existing local result folder.

## Next-variant decision rule

- V26-E20 completed with 52.84% final validation vs V25 at 46.08%; highest raw per-sync validation was 53.45% at sync 13 (66 observed classes; cross-sync denominators differ), and sync 20 ended without a terminal-flush collapse. Its client-by-sync and per-class trajectories show weak one-sync acquisition for classes 96–99 plus selective class-level forgetting. V27-E30 completed with a 2.74 pp final validation gain over V26 but worse accuracy on the final four classes. V28-TermReplay is the next controlled hypothesis, motivated by paired class-age gains after one later sync. Add one explicit full-participation archive-only Replay event and aggregate with whole-model FedAvg; do not treat the current communication-only flush as training. Compare variants on final validation; report test only. A static scheduler check found the configured terminal flush is communication-only: `train_round_task_barriers` returns no work for an empty stream event. A replay-only terminal training experiment therefore needs an explicit replay event/schedule path; do not describe the current flush as training.
- If validation is flat or worse, use the table to distinguish weak new-class
  acquisition from old-class forgetting. A fixed-rate/high-epoch interaction
  is a leading hypothesis because V25 uses constant SGD LR 0.03 and has no
  local scheduler; test a scheduler at the same 20 epochs before changing
  Replay or FedAvg.
- Official test accuracy is reported for the completed run but does not
  choose between variants.
