# 121-cell campaign ledger

`CAMPAIGN_LEDGER_121.jsonl` contains one record for each unique cell across the seven documented suites. Retries and reference runs are stored under each cell's `attempts` array and do not increase the unique-cell count. `CAMPAIGN_LEDGER_121_SUMMARY.json` gives the suite counts and timestamp of the most recently reconciled live snapshot.

The `state` field uses the campaign vocabulary: `not_started`, `plan_validated`, `queued`, `running`, `completed`, `failed`, `interrupted`, or `needs_audit`. States reflect the last evidence captured from the respective execution host; recheck live status before making a later claim. A completed attempt can still have a cell state of `needs_audit` until configuration, schedules, updates, metrics, and artifacts pass the completion audit.

Hosts and roots follow the documented assignments. Mtech is `192.168.3.30`; the V100 records use the actual output root under `.../results/v25_offline_epoch_grid_20261122_v2/`. The V100 source tree and output root are separate paths. The separate four-run FedQCL five-epoch reference block and `Trash/results_1` / `Trash/results_2` are excluded from the 121 cells.

The 15:47 IST ledger snapshot had 38 `needs_audit`, 15 `failed`, 8 `running`, and 60 `queued` cells. At that point the task-size Task-IL width-5/offline-2 OOM was preserved, with a second broker wrapper still queued.

The 30-minute server-poll interval applies only after all currently fitting jobs have been launched, VRAM is nearly full, no further job fits, and admitted jobs have shown about five continuous minutes of healthy learning. Audits, ledger work, and other useful work can continue between those polls. Failed and interrupted runs remain the first retry priority. The interval is not active while admission or recovery is still underway.

At 16:31 IST the ledger had 44 `needs_audit`, 14 `failed`, 8 `running`, and 55 `queued` cells. Attempt 02 later ended by SIGTERM after one sync; its original status was preserved and reconciled as interrupted.

At the 17:08 IST snapshot the ledger had 44 `needs_audit`, 14 `failed`, 8 `running`, and 55 `queued` cells. The task-size width-5/offline-2 cell has a preserved CUDA OOM attempt, a queued duplicate dispatch that stopped before admission, an interrupted SIGTERM attempt 02, and plan-validated attempt 03 submitted in persistent Mtech tmux with an 8192 MiB estimate plus 1024 MiB margin. Worker admission is unverified, so the cell remains unresolved.

At 17:35 IST, one V100 experiment worker (PID 429871) was verified active. No Mtech campaign worker or broker was found in the process inventory; an unidentified GPU process used 5,978 MiB. Mtech had 26,519 MiB free VRAM. V100 free VRAM was clipped from the captured pane. The 11 A-G jobs still named active by a stale manifest are marked `needs_audit` pending process/output reconciliation.


### 2026-11-22 18:03 IST audit update

Audited and marked complete the three Task-IL, width-4, five-client task-size cells with offline counts 1, 2, and 3. Verified status, five learning syncs, communication-only terminal flush, seed-0 config, replay ratio 1.0, 25 four-class heads covering all 100 classes, six per-sync CSV reports, finite metrics, and no traceback/CUDA/CUBLAS/error matches in worker logs. SHA-256 for config, summary, and metrics is recorded per cell in `cell_manifests/task_size_4_vs_5/`. Source-tree and dataset provenance still need to be reconciled into the central archive.


### 2026-11-22 18:23 IST A-G failed-cell retry dispatch

All 14 previously failed A-G cells now have preserved attempt_01 evidence and an isolated attempt_02. All 14 retry plans passed runtime schedule validation. Seven workers were admitted and seven brokers were waiting for VRAM at the snapshot; Mtech V100 had 32,422 MiB used and 79 MiB free. Each admission requests a 4,096 MiB estimate plus a 1,024 MiB margin, based on previously audited A-G peaks up to about 3,187 MiB allocated. Per-cell broker/worker PIDs, plan paths, output paths, and runner/patch hashes are recorded in `cell_manifests/independent_ablation_A_G/`. The 30-minute health interval is not active yet; jobs need five continuous minutes of verified learning after the last admitted worker and no further fitting work.


### 2026-11-22 18:33 IST VRAM admission correction

F width-5 attempt_02 failed with CUDA OOM: its 20 MiB allocation found only 5.12 MiB free while six other workers were active. Preserve this failure. Six other attempt_02 workers remain active and their broker logs show advancing rounds. The seven brokers that had not admitted a worker were stopped only after confirming they had no worker child and no admission receipt; their waiting logs remain preserved. Those seven cells plus F have isolated attempt_03 plans queued with a 6,144 MiB estimate and 1,024 MiB margin, reflecting the observed 5,840 MiB peak in an active run. The 30-minute monitoring cadence remains inactive while these corrected retries are being admitted and health is re-established.


### 2026-11-22 18:45 IST width-5 Task-IL audit and retry reconciliation

Audited width-5/offline-1 as complete (4 learning syncs plus terminal communication flush; 5 reports; 100-class coverage; finite metrics; clean worker log). Audited offline-2 attempt 03 as complete with the same schedule and report coverage; preserved its earlier OOM and SIGTERM attempts. Reconciled offline-3 attempt 01 as interrupted: exit -15 after round 2/5, stale running status PID 338 absent, partial outputs retained. Attempt 02 is recorded as the priority retry pending VRAM admission; no server broker has been submitted yet because the conservative peak-plus-margin estimate exceeds current free VRAM. Source-tree Git identity and dataset identity receipts remain archival gaps for successful cells.


### 2026-11-22 18:46 IST V100 reconciliation

Rechecked V100 through the existing B550 tmux route. PID 429871 and CAFCL controllers/workers are absent; the raw status file still says running. GPU has 21,178 MiB free but the root filesystem is full (0 bytes available), so V100 jobs cannot safely write results or be relaunched. Unrelated GPU processes were left untouched. The cell is `needs_audit`, not running or completed, pending terminal artifact reconciliation and a durable write path.


### 2026-11-22 18:52 IST A-G retry completion detected

A_clients_3_class_il_seed0 attempt 02 passed terminal audit: 18 rounds/reports, finite metrics, per-sync rows ending at 100, broker exit 0, peak 4,218 MiB, final accuracy 0.409, and no traceback/OOM/error markers. Its attempt_01 failure remains preserved. Mtech had 5,013 MiB free VRAM at the earlier sample; five other A-G attempt_02 workers remained running.


### 2026-11-22 19:01 IST stale-worker reconciliation

Rechecked Mtech worker PIDs and terminal logs. Five A-G retry workers are live; four root `running` status files (C hidden-5, D cross-step-2, G overlap-2, G overlap-4) have no live worker and logs record SIGTERM. Their attempt_01 artifacts are preserved and attempt_02 is priority-pending with measured peak plus 1,024 MiB admission margins (5,884–7,002 MiB). Task-size width-4 Class-IL/offline-2 also has a stale PID 9003 and a parallel-attempt log ending SIGTERM at 5,854 MiB; its isolated retry requires 6,878 MiB free. Last Mtech sample: 5,013 MiB GPU VRAM free, 193 GB disk free, 87 GB available RAM. These retries are not broker-submitted because none of their conservative VRAM thresholds fit at that sample. V100 remains disk-blocked with 0 bytes free on root; unrelated Boltz processes occupy 11,310 MiB VRAM.


### 2026-11-22 19:11 IST interrupted A-G retries resubmitted

Prepared isolated attempt_02 specs for C hidden-5, D cross-step-2, G overlap-2, and G overlap-4. All four passed runtime plan validation (100 classes, zero sample overlap, terminal flush). Their shared VRAM brokers are submitted and waiting, ordered by estimated peak to allow the lowest threshold to admit first. Required free VRAM including 1,024 MiB margin is 5,884–7,002 MiB; Mtech had 5,013 MiB free at submission. No worker has been admitted yet. The stale attempt_01 logs/statuses and partial artifacts remain preserved.


### 2026-11-22 19:18 IST additional A-G retry completions

A_clients_3_task_il_seed0 attempt 02 completed and passed audit (17 learning syncs plus terminal flush, 18 reports, final accuracy 0.9054, broker exit 0, 4,220 MiB peak). E_within_step_2_seed0 attempt 02 completed and passed audit (13 learning syncs plus terminal flush, 14 reports, final accuracy 0.909, broker exit 0, 6,184 MiB peak). Both have finite metrics, expected seen-class report progression, and no traceback/CUDA/CUBLAS/error markers. Checkpoint-excluded result transfer and source/dataset provenance receipts remain outstanding.
