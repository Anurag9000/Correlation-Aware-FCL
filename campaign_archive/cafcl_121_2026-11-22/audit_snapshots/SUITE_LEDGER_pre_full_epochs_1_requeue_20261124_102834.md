# Correlation-Aware-FCL ablation suite log

This is the running ledger for newly created ablation suites. Add one dated entry per suite, in creation order. Keep designed, plan-validated, queued, running, completed, and failed states distinct. Record host and absolute paths so later results can be matched to the exact code and run matrix.

## Suite 1 — Disjoint task-size 4 vs 5 classes/task

**Created:** 2026-11-22  
**Host:** MtechDEServer (`192.168.3.30`)  
**State:** Plan-validated; controller is waiting for GPU capacity; no run in this suite has started.  
**Last status observation:** 2026-11-22 03:00 IST; 981 MiB GPU memory free, controller admission threshold 24,576 MiB.  
**Controller PID:** 22724  
**One-seed policy:** seed 0 for all 12 runs.

### Purpose and matrix

Compare task size while preserving the V25 Replay/FedAvg recipe and strict disjoint class ownership. This covers the six disjoint participation cases—Task-IL and Class-IL, each with 1, 2, or 3 randomly offline clients—at two task sizes:

| IL setting | Randomly offline clients | Classes/task | Runs |
|---|---:|---:|---:|
| Task-IL | 1, 2, 3 | 4 and 5 | 6 |
| Class-IL | 1, 2, 3 | 4 and 5 | 6 |
| **Total** |  |  | **12** |

There are five clients and 100 CIFAR-100 classes. Each client receives 20 distinct classes over its curriculum. The scheduler assigns one new task to each client per learning sync and assigns every class exactly once.

### Schedule and output heads

- **4 classes/task:** 5 tasks/client; 5 learning syncs; 25 total task instances. A final communication-only flush follows sync 5.
- **5 classes/task:** 4 tasks/client; 4 learning syncs; 20 total task instances. A final communication-only flush follows sync 4.
- A task contains 1,800 fresh training examples at width 4 and 2,250 at width 5, based on 450 training examples per CIFAR-100 class.
- Task-IL uses one task-specific head per task instance: 25 width-4 heads or 20 width-5 heads. Task-head updates are merged with the existing head-specific FedAvg path.
- Class-IL retains one shared 100-class softmax head and ordinary whole-model FedAvg.

### Training and communication recipe

- CIFAR-100, CIFAR-stem ResNet-18, 5 clients, seed 0.
- Ordinary Experience Replay (`method=replay`) with sample-weighted FedAvg.
- 10 local epochs per task; SGD learning rate 0.03, momentum 0.9, weight decay 0.0005; client optimizer state persists.
- Training crop/flip augmentation is enabled.
- Replay uses a 50-example-per-class private reservoir and sampled 1:1 current/replay updates.
- Random selection accepts 4/5, 3/5, or 2/5 client uploads per learning sync for 1, 2, or 3 offline clients, respectively. Offline clients still train and retain local updates; the terminal flush communicates pending updates without additional training.
- Per-sync seen-class evaluation and final post-flush evaluation are enabled.

### Implementation and validation

The isolated suite copy at `/home/anurag24092/Projects/Correlation-Aware-FCL-tasksize45-disjoint-ablation-20261122` contains separate Task-IL and Class-IL source variants, the campaign launcher, plan files, and results. The scheduler in each isolated variant was generalized so the disjoint one-task-per-client-per-sync branch accepts the configured task width and rejects curricula that cannot give every client the same integer number of complete tasks. The main checkout was not modified for this ablation.

All 12 schedules passed plan validation: complete class coverage without reuse, equal task counts per client, and one task per client at each learning sync. The controller runs four blocks of three jobs (offline counts 1/2/3 within each IL/task-size pair) and waits for at least 24 GiB free VRAM and existing CAFCL/FedQCL/optimizer-reset workloads to clear. It does not terminate existing jobs.

Important paths:

- Suite record and launcher: `/home/anurag24092/Projects/Correlation-Aware-FCL-tasksize45-disjoint-ablation-20261122/TASK_SIZE_4_5_ABLATION.md` and `scripts/tasksize45_campaign.py`
- Queue state: `/home/anurag24092/Projects/Correlation-Aware-FCL-tasksize45-disjoint-ablation-20261122/queue/campaign_status.json`
- Configs and plan summaries: same suite's `queue/`
- Run outputs: same suite's `results/<task_il|class_il>/task_<4|5>/offline_<1|2|3>/`

## Entry template for future suites

### Suite N — <short name>

**Created:** YYYY-MM-DD  
**Host:** <host and address>  
**State:** <planned / plan-validated / queued / running / completed / failed>  
**Seed(s):** <values>

#### Purpose and matrix

<Factors, levels, run count, and comparison being isolated.>

#### Protocol and implementation

<Dataset, clients, task curriculum, model heads, training, replay, aggregation, sync/flush rules, and code changes.>

#### Validation and execution status

<Plan validation, launched/completed/failed counts, resource gate, and relevant paths.>

## Suite 2 — Original disjoint intra-task sync frequency

**Created:** 2026-11-22  
**Host:** MtechDEServer (`192.168.3.30`)  
**Last queue-state evidence (2026-11-22 06:18 IST):** Six plans validated; queue file said waiting for capacity at position 1 of 6; Task-IL / 2-sync was next. This is not a current worker-state claim.  
**Controller PID:** 29969  
**One-seed policy:** seed 0 for all runs.

Detailed standalone protocol and implementation record: `INTRATASK_SYNC_FREQUENCY_ABLATION.md`.

### Purpose and matrix

Keep the original disjoint two-class task configuration and compare more frequent server synchronization within each local task. The two model arms are Task-IL disjoint and Class-IL disjoint. All five clients participate at every learning sync. Sync frequency is 2, 5, or 10 times per task, yielding six runs total. The existing one-sync-per-task configuration is the reference and is not duplicated.

Each task still receives 10 local epochs, divided evenly between syncs:

| Syncs/task | Epochs/sync | Learning syncs over CIFAR-100 | Total with terminal flush |
|---:|---:|---:|---:|
| 2 | 5 | 20 | 21 |
| 5 | 2 | 50 | 51 |
| 10 | 1 | 100 | 101 |

Each client retains its original 10 two-class tasks. The same task identity and training indices persist across its sync segments; continuation segments do not create new local experiences or add the task to replay memory again. Each segment aggregates and broadcasts the model. The terminal flush is communication-only.

### Fixed recipe and heads

V25 Replay/FedAvg settings remain: CIFAR-100, five clients, ResNet-18 CIFAR stem, Replay, sample-weighted FedAvg, seed 0, SGD at 0.03 with momentum 0.9 and weight decay 0.0005, persistent client momentum, crop/flip augmentation, 50 examples per class in the private reservoir, and sampled 1:1 current/replay updates. Task-IL has 50 two-class task heads across the curriculum; Class-IL has one shared 100-class head.

### Implementation, validation, and paths

The isolated implementation is at `/home/anurag24092/Projects/Correlation-Aware-FCL-intratask-sync-ablation-20261122`; it does not modify the waiting task-size suite. It adds a per-task sync count, preserves the logical task boundary and replay insertion across segments, divides local epochs exactly, and gives each segment a server aggregate/broadcast barrier.

All six plans passed preflight: 50 task instances across 10 task rounds, exactly 2/5/10 sync segments per task, 5/2/1 epochs per segment, and 20/50/100 learning syncs plus a flush. These are plan checks, not training results.

- Mtech design record: `INTRATASK_SYNC_ABLATION.md`
- Launcher: `scripts/intratask_sync_campaign.py`
- Queue status and configs: `queue/`
- Outputs: `results/<task_il|class_il>/syncs_per_task_<2|5|10>/`

The controller processes one run at a time and requires 24 GiB free VRAM. An earlier 03:00 IST snapshot showed 91 MiB free; the newer 06:14:52 IST queue-file sample showed 29,175 MiB free but still listed the FedQCL and task-size jobs as external work. No existing job was stopped.

**2026-11-22 06:18 IST server verification:** Direct SSH to `192.168.3.30` confirmed the isolated suite root, sync runner, all six run plans, and both IL schedule adapters. Plan validation preserves five clients, 50 two-class task instances, ten epochs/task, stable task identity and dataset indices across segments, 2/5/10 sync segments with 5/2/1 epochs per segment, and 20/50/100 learning syncs plus a communication-only flush. The `campaign_status.json` read at that time said `waiting_for_capacity`, position 1 of 6, with the 2-sync Task-IL run next; its own timestamp was 06:14:52 IST and listed 29,175 MiB free against a 24,576 MiB threshold plus existing FedQCL and task-size jobs. This is a queued, plan-validated suite, not a launched or trained result. No duplicate was started and no server job was stopped.

**2026-11-22 06:58 IST plan re-verification:** Direct SSH re-read the isolated campaign source and all six server plan summaries. The six unique cells remain exactly Task-IL/Class-IL × 2/5/10 syncs per task; every plan records five clients uploading on every learning sync, 50 two-class task instances, 10 tasks/client, 10 epochs/task, 5/2/1 epochs per sync, and 20/50/100 learning syncs plus a communication-only flush (21/51/101 total events). Task identity, data indices, and replay-memory task boundary are preserved across sync segments in the implementation. Every plan is explicitly `plan_only`; no training accuracy is implied. The A–G controller's 06:50 snapshot listed this suite's campaign command among external processes, but this verification did not reread the sync suite's queue state. No duplicate suite or worker was launched, and no server process was stopped or modified.

**2026-11-22 07:02:44 IST live queue check:** Sync-frequency controller PID 29969 is alive and reports `waiting_for_capacity`, position 1/6, with `V25_TASK_IL_DISJOINT_SYNC2_PER_TASK_REPLAY_FEDAVG_SEED0` next. The queue file was updated at 06:59:52 IST: 18,531 MiB free vs. 24,576 MiB required. It lists FedQCL and the task-size campaign as external jobs. No sync-frequency training worker has started according to the queue state; the campaign is configured and waiting. No jobs were stopped or altered.

## Goal campaign — Independent CIFAR-100 Task-IL ablations (A–G)

**Created:** 2026-11-22  
**Host:** MtechDEServer (`192.168.3.30`)  
**Documentation:** Kakarot `/home/anurag-basistha/CORRELATION_AWARE_FCL_ABLATION_GOAL_2026-11-22.md`  
**State at 2026-11-22 07:47 IST:** All 29 plans passed preflight. Three A runs are terminal and fully archived on Kakarot. A_clients_2_class_il_seed0 finished at 40.72% test / 39.98% validation after 25 learning syncs; its 26th sync was communication-only. A_clients_2_task_il_seed0 (PID 16232) was active at round 3/26, with 77.1667% test / 78.8333% validation over 12 classes. Its 29 current non-checkpoint files (1,463,940 bytes) are staged and hash-verified.  
**One-seed policy:** seed 0 per run.

**Last A–G queue observation (2026-11-22 07:47:45 IST):** Controller PID 34759 was alive; its queue file was updated at 07:45:41 and said `waiting_for_capacity`, index 1, next run `A_clients_10_class_il_seed0`, with 25,929 MiB free against 24,576 MiB and other campaigns still listed. This stale cursor predates the manual sequence; the controller's worker guard recognizes completed output directories, preventing overwrite if the first entry is retried. PID 16232 was the only active A–G worker at this snapshot. No existing process was stopped.

The source-of-truth document defines 29 new runs across seven independent
factors: client count (6); temporal recurrence delay (5); hidden-client outage
duration (6); cross-step task recurrence (3); within-step shared tasks (3);
strict disjoint task width (3); and width-5 task overlap (3). The base is the
original CIFAR-100 disjoint Task-IL run. Class-IL appears only in the client
count ablation; the earlier six-run intra-task sync suite remains its own
Task-IL/Class-IL full-participation comparison.

The design explicitly handles the two impossible exact partitions: 3 clients
with 100 classes in width-2 tasks use a balanced 17/17/16 allocation, and
width-3 tasks across five clients use six width-3 syncs plus a final width-2
task per client. See the goal document for schedule semantics, tail reporting,
frozen baseline, per-run artifacts, GPU admission, and completion criteria.

Implementation root: `/home/anurag24092/Projects/Correlation-Aware-FCL-independent-ablations-20261122`. The controller uses `plans/server_compiled_v4/`; after safe promotion, its B inputs now match v6. The pre-promotion B folder is retained at `plans/archive/server_compiled_v4_B_before_v6_promotion/`. Runner source derives from the preserved `Correlation-Aware-FCL-six-runs-20261121` Task-IL/Class-IL snapshots, with only explicit schedule adapters added. The plan-only runner loaded the real CIFAR-100 bank and compiled all 29 scenarios with valid class coverage, no raw-image duplication, and unique Task-IL task IDs. This preflight did not train any model. Kakarot copies of the goal, plans, scripts, and isolated source trees are under `/home/anurag-basistha/CORRELATION_AWARE_FCL_ABLATIONS_2026-11-22/`.

**2026-11-22 plan refinement and promotion:** B's initial delay schedules changed downstream novel-class assignment when the target recurrence consumed a task slot. A matched 18-sync design was generated as `plans/server_compiled_v6/`: all five levels preserve the same task multiset and per-class occurrence counts, exchanging the target's reserved return task with one explicitly recorded displaced task. All 29 runner/data preflights passed again on v6. At 04:51:07 IST, a live snapshot showed controller PID 34759 waiting before its first A run; the original v4 B folder was archived and the validated v6 B inputs were promoted into the controller's existing v4 root. The active plan then byte-matched v6 across all 89 files. No run was restarted or terminated.

**2026-11-22 04:59:35 IST queue update:** Controller PID 34759 is still waiting at index 1 for `A_clients_10_class_il_seed0`; no A–G worker has started. V100 memory is 31,926 MiB used / 575 MiB free at 100% utilization, below the 24,576 MiB launch gate. The queue and all pre-existing processes were left unchanged.

**2026-11-22 05:05:19 IST queue update:** Controller PID 34759 remains at index 1 for `A_clients_10_class_il_seed0`. V100 memory is 25,742 MiB used / 6,759 MiB free at 100% utilization, still below the launch gate. No A–G worker has started; no existing job was touched.

**2026-11-22 05:11:13 IST queue update:** The live process table confirmed controller PID 34759 is still alive at index 1 and no A–G worker status files exist. `nvidia-smi` reported 22,156 MiB used / 10,345 MiB free at 100% utilization, below the launch gate. The queue file listed 10,351 MiB free at its 05:10:39 update; this small difference reflects sampling time. No existing job was touched.

**Baseline fidelity audit:** On Mtech's campaign environment, the frozen V25 disjoint Task-IL full-run design was compared with the ablation recipe and the preserved schedule generator. It resolves to `resnet18_cifar32`, matches 50 width-2 tasks over 10 learning syncs, and uses the documented replay, optimizer, augmentation, and FedAvg values. Only the campaign's explicit schedule, fresh-run/resume flag, per-sync reporting cadence, IDs, and list/tuple serialization differ. SHA-256 and exact field details are in `BASELINE_FIDELITY_AUDIT.md` under Kakarot's archive.

**Worker provenance update:** The isolated worker was extended to create per-run config, schedule, plan-validation, source-hash, and environment files before training, in addition to its design/status/log and simulator outputs. Mtech Python 3.12 parsed the updated script and the hash helper successfully hashed the Task-IL runtime, compiler, and worker. No training or CUDA context was started by this check.

**2026-11-22 05:17:52 IST queue update:** Controller PID 34759 is still alive and waiting at index 1 for `A_clients_10_class_il_seed0`; no A–G worker status files exist. `nvidia-smi` reported 22,234 MiB used / 10,267 MiB free at 99% utilization, below the launch gate. Existing campaign workloads were left alone.

**Result transfer/storage plan:** Kakarot had 549 GiB free at the last disk check. One preserved 10-learning-sync run uses 16 GiB, and A–G schedules contain 460 learning syncs plus 29 flushes, projecting roughly 600 GiB of checkpoint data. Added `tools/sync_completed_runs_from_mtech.py`: it pulls terminal run folders and all non-checkpoint evidence, computes local file hashes, and records server checkpoint paths/sizes; no destination deletions occur. Checkpoints remain on Mtech to avoid exceeding local free space. The exact boundary is in `RESULT_TRANSFER_POLICY.md`.

**2026-11-22 05:24 IST transfer check:** The Kakarot sync script successfully queried the Mtech results root, found zero terminal A–G run directories, and created `results/transfer_manifest.json`. No training output was available to copy yet. The live controller/process state remains last verified at 05:17 IST.

**2026-11-22 05:31:42 IST queue update:** Controller PID 34759 is alive and waiting at index 1 for `A_clients_10_class_il_seed0`; no A–G worker status files exist. `nvidia-smi` reported 16,052 MiB used / 16,449 MiB free at 100% utilization, below the 24,576 MiB gate. Existing other campaigns remain detected and untouched.

**2026-11-22 05:37:16 IST queue update:** PID 34759 remains alive at index 1 for `A_clients_10_class_il_seed0`; no A–G worker status files exist. The current `nvidia-smi` sample is 13,660 MiB used / 18,841 MiB free at 100% utilization, still below the 24,576 MiB gate. The controller's last-written capacity value was 16,449 MiB from its earlier 05:35:39 check; all existing workloads remain untouched.

**2026-11-22 05:43:23 IST queue update:** PID 34759 is still alive at index 1; no A–G worker status files exist. `nvidia-smi` reported 13,660 MiB used / 18,841 MiB free at 99% utilization. The first queued run remains `A_clients_10_class_il_seed0`; existing campaign controllers still prevent admission. No process was touched.

**2026-11-22 05:54:57 IST first A–G run launched:** `A_clients_10_class_il_seed0` (10 clients, Class-IL, full participation) was started as PID 4289 with 18,841 MiB free and an 8,192 MiB prelaunch floor. The equivalent frozen V25 full run had peaked at 2,760,641,536 CUDA-allocated bytes. PID 4289 was alive at 06:02:17 IST with three of six simulator rounds in its log. An initial status inspection used the wrong result directory because the run ID includes the `A_` prefix; at 06:03 IST, the open log was moved to `results/A/A_clients_10_class_il_seed0/worker.log` without interrupting the worker. The correct worker status/summary still needs the next scheduled check; no completion or accuracy result is claimed. To prevent collision with the still-waiting controller PID 34759, the updated worker returns successfully for an already-completed run or waits in 300-second intervals on an existing live worker PID. The launch record is archived as `plans/manual_worker_launch_A_clients10_classil.json`.

**2026-11-22 06:07:31 IST:** Correct-path `worker_status.json` confirms PID 4289 is running; `worker.log` has rounds 1–4 of 6. V100 was at 24,052 MiB used / 8,449 MiB free, 100% utilization. Parent controller PID 34759 remained waiting at index 1 with its 24,576 MiB gate. Config, schedule, plan-validation, design, environment, and source-checksum files were copied to Kakarot; all six remote/local hashes match. Final reports and metrics await terminal state.

**2026-11-22 06:21:24 IST first A run completed and synced:** `A_clients_10_class_il_seed0` has terminal `completed` status; worker PID 4289 is no longer running. The seen-class report at learning sync 5 covers all 100 classes: test accuracy 21.72%, validation accuracy 21.46%. The final communication-only flush leaves both metrics unchanged. `final_metrics.accuracy` and the worker's legacy `final_accuracy` equal 0.2146, matching validation, so the explicit `test_accuracy` field is used for the test result. Kakarot copied 39 non-checkpoint evidence files (2,419,306 bytes) and recorded eight server-only checkpoint paths/sizes; manifest: `results/transfer_manifest.json`. At the same timestamp PID 34759 was alive and the V100 reported 3,368 MiB used / 29,133 MiB free at 97% utilization. The snapshot did not include the queue position; inspect the controller's actual queue file/log before deciding whether it has advanced. No other job was stopped.

**2026-11-22 06:31:22 IST second A run launched:** `A_clients_10_task_il_seed0` (10 clients, Task-IL, full participation, seed 0) was admitted as a single manual worker, PID 8139, after verifying 29,129 MiB free VRAM and the validated unique plan. Its `worker_status.json` reported `running` at 06:31:22. Config, schedule, plan validation, design, environment, and source checksum were staged on Kakarot; all six SHA-256 values match Mtech. The queue controller remained alive but was waiting on detected external campaign controllers. Two preparatory launch attempts exited before training; their logs are retained in `plans/failed_launch_attempt_A_clients_10_task_il_*.log`, and only the exact attempt-created scaffolding was removed before successful launch. No existing job was stopped.

**2026-11-22 06:44:55 IST A task-IL progress:** PID 8139 was alive after 13m43s. `worker.log` reached rounds 1 and 2 of 6, with `acc` 0.5090 then 0.5460 (validation under this run config). V100 reported 13,558 MiB used / 18,943 MiB free at 99% utilization; `nvidia-smi` attributed 10,186 MiB to this worker and 3,366 MiB to another Python process. The controller remained waiting at index 1; its state file last updated at 06:40:40 IST showed 19,561 MiB free, below the gate. This is intermediate, per-sync evidence; final accuracy and run completion remain pending.

**2026-11-22 06:50:10 IST A task-IL progress and partial sync:** PID 8139 was alive after 18m58s at round 3/6. Test accuracy across 20/40/60 seen classes is 50.55% / 54.05% / 70.10%; validation accuracy is 50.90% / 54.60% / 69.57%. All ten clients were selected at each learning sync. V100 reported 13,558 MiB used / 18,943 MiB free at 98% utilization; the worker process used 10,186 MiB and a separate Python process used 3,366 MiB. The queue remains waiting at index 1 because free VRAM is below 24,576 MiB and external campaign commands are present. After this status check, 25 non-checkpoint files totaling 1,608,728 bytes were checksum-synchronized to Kakarot, including all three per-sync reports. This remains an intermediate one-seed result, not final accuracy.

**2026-11-22 06:55:24 IST A task-IL runtime check:** PID 8139 remains alive at round 3/6. No new round marker appeared in the five minutes since 06:50, while the worker used 234% CPU and 10,598 MiB VRAM; GPU utilization was 100%, so the run remains in active computation rather than a confirmed stall. V100 had 18,531 MiB free; a separate Python process used 3,366 MiB. The controller queue status, last updated at 06:50:40, remains at index 1 below its memory threshold. No process was stopped.

**2026-11-22 07:08:07 IST A task-IL progress and evidence sync:** PID 8139 was alive at round 4/6. Sync 4 evaluated 80 observed classes and reported 80.4625% test accuracy and 81.475% validation accuracy; earlier test values were 50.55%, 54.05%, and 70.10%, with validation 50.90%, 54.60%, and 69.57%. All ten clients participated in each learning sync. GPU utilization was 100% with 18,531 MiB free; the worker process used 238% CPU. I checksum-synchronized the active run's non-checkpoint files to Kakarot without deletion: 29 files, 2,046,424 bytes, including the sync-4 report. `results/transfer_manifest.json` records their local SHA-256 values and marks the run provisional pending its terminal refresh. The A–G controller PID 34759 remains alive but its queue file is waiting at index 1, still naming completed A_clients_10_class_il_seed0 next; its 07:05:40 snapshot reports 18,531 MiB free vs. 24,576 MiB required. No server process was stopped or restarted.

**2026-11-22 07:14:51 IST G-plan provenance fix:** Recompiling all 29 plans in the Mtech campaign environment exposed a metadata mismatch in G: the compiler actually used seed 500 for both class and client ordering but recorded 502/503/504 as `client_order_seed`. Corrected the compiler to record `schedule_rng_seed=500` and regenerated the G configs/validation summaries in active server plan roots v4 and v6, the unversioned G plan copy, and the Kakarot archive. All three G schedule files are byte-identical to the prior schedules; plan validation still confirms adjacent overlaps of exactly 2, 3, and 4. The old v4 G config, validation, and schedule files are preserved in `plans/archive/G_seed_metadata_before_fix/`. Only G plan metadata and compiler provenance changed; no training code, task assignments, or run artifacts changed. The 29-plan compiler pass completed; this is a plan check, not training evidence.

**2026-11-22 07:17:46 IST A Task-IL terminal audit and next launch:** PID 8139 reported `completed`. The final learning sync evaluated all 100 classes at 83.31% test / 83.46% validation; sync 6 was the terminal communication-only flush at model version 5 and repeated those metrics. The non-checkpoint terminal transfer found 39 files per 10-client run; all staged local hashes match the transfer manifest, and checkpoint paths/sizes are recorded separately on Mtech. With 29,129 MiB free and no active A–G worker, the one-worker helper launched A_clients_2_class_il_seed0 (PID 13218, seed 0) at 07:19:17 IST, following lexical controller order after the two 10-client cells. No existing job was stopped. Its initial launch receipt is saved under `plans/manual_worker_launch_A_clients_2_class_il_seed0.json`.

**2026-11-22 07:23:40 IST A 2-client Class-IL progress and staged sync:** PID 13218 remained `running` after 4m22s at round 7/26. The current per-sync report covers 28 observed classes and gives 44.7143% test / 43.0714% validation; the simulator log accuracy marker matches validation. All seven per-sync reports and current metadata/scenario/log files were checksum-synchronized to Kakarot (41 files, 1,624,723 bytes), with zero local hash mismatches. GPU was 99% utilized with 25,563 MiB free; this worker used 189% CPU. No additional run was launched at this check.

**2026-11-22 07:28:36 IST G runtime plan-only verification:** Using the corrected Mtech compiler/runtime environment, all three G plans (`width5_overlap_2/3/4`) returned `runner_plan_validated`, with their schedule overrides loaded, all 100 classes covered, no raw training-image reuse, and an empty terminal flush stage. They retained 6/8/12 learning syncs and per-task adjacent overlap 2/3/4. The machine-readable report and correction receipt are copied to Kakarot under `plans/`; these checks do not train models or claim accuracy.

**2026-11-22 07:30:45 IST A 2-client Class-IL progress and staged sync:** PID 13218 remained `running` at round 16/26. Sync 16 covers 64 observed classes, with 40.9063% test and 40.6875% validation accuracy. The worker log reached round 16 at 183% CPU; V100 utilization was 99% with 25,525 MiB free. I checksum-synchronized all current non-checkpoint files to Kakarot (81 files, 4,361,555 bytes); manifest hash/size checks passed. The run remains provisional until terminal refresh.

**2026-11-22 07:36:05 IST A 2-client Class-IL progress and staged sync:** PID 13218 remained `running` at round 22/26. Sync 22 covers 88 observed classes, with 42.2159% test and 41.9773% validation accuracy. The worker used 177% CPU; V100 utilization was 100% with 25,517 MiB free. I checksum-synchronized all current non-checkpoint files to Kakarot (105 files, 6,415,784 bytes); manifest hash/size checks passed. The run remains provisional until terminal refresh.

**2026-11-22 07:41:29 IST A 2-client Class-IL terminal audit and next launch:** PID 13218 completed. Its final learning sync evaluated all 100 classes at 40.72% test / 39.98% validation; sync 26 was communication-only and preserved those values. The terminal transfer recorded 119 non-checkpoint files (7,673,816 bytes) and 28 server-only checkpoint paths; all local hashes match. With 28,821 MiB free and no active A–G worker, the helper launched A_clients_2_task_il_seed0 as PID 16232 at 07:42:15 IST (seed 0). Its launch receipt is saved on Kakarot; no existing job was stopped.

**2026-11-22 07:44:06 IST A 2-client Task-IL initial artifact sync:** The launch helper had returned `launched` for PID 16232 at 07:42:15. A non-checkpoint sync copied its configuration, schedule, environment/source provenance, scenario, metrics, log, and first two reports: 21 files totaling 1,059,000 bytes. The copied worker status file, timestamped 07:42:17, said `running`; the log and report 2 were written by 07:44:06. Sync 2 reports 57.625% test / 54.50% validation over eight observed classes. This is staged early-run evidence, not final accuracy.

**2026-11-22 07:47:45 IST A 2-client Task-IL progress and staged sync:** PID 16232 was running at round 3/26, elapsed 5m31s, at 143% CPU. Sync 3 evaluated 12 classes at 77.1667% test / 78.8333% validation. V100 had 25,929 MiB free at 100% utilization. The refreshed non-checkpoint copy contains 29 files totaling 1,463,940 bytes; local hashes were verified. The run remains provisional.

**2026-11-22 07:30 IST transfer-manifest audit:** The terminal refresh for A_clients_10_task_il_seed0 had updated its final file hashes but left the earlier partial-copy entry in `provisional_runs`. I updated `tools/sync_completed_runs_from_mtech.py` so terminal runs drop their provisional record, removed the stale entry, and re-audited both completed runs plus the active A_clients_2_class_il_seed0 snapshot. All 39/39, 39/39, and 41/41 files respectively match their recorded SHA-256 and size. The updated script hash is reflected in `code/SOURCE_MANIFEST.json`; no server training or run artifacts were modified.

## Suite 3 — Local client epochs per sync × client visibility

**Created/launched:** 2026-11-22  
**Host:** MtechDEServer (192.168.3.30)  
**State at last observation:** Wave 1 active with four processes; two wave-1 jobs exited (one early training/evaluation failure and one completed 50 rounds but failed saving its result). Wave 2 has not started and is gated on all six wave-1 processes exiting. No final ablation comparison is available.  
**Last status observation:** 2026-11-22 03:51:48 IST  
**Controller:** PID 14328; /home/anurag24092/fedqcl_12runs_controller_20261122_012754.log  
**Campaign root:** /home/anurag24092/Projects/FedQCL_Task/campaigns/cifar100_12_local_epoch_runs_20261122_012754  
**Launcher:** /home/anurag24092/Projects/FedQCL_Task/launch_cifar100_12_runs.sh  
**Python:** /home/anurag24092/Projects/Correlation-Aware-FCL/.venv/bin/python, Python 3.12.15, PyTorch 2.5.1+cu121, CUDA 12.1; preflight saw a Tesla V100-PCIE-32GB.

### Purpose and matrix

Ablate local optimization length while retaining the CIFAR-100 FedQCL task-incremental protocol and changing only local epochs per sync and client visibility. Each of the four visibility modes is paired with 1, 2, and 10 local epochs, for 12 runs total:

| Visibility condition | 1 local epoch/sync | 2 local epochs/sync | 10 local epochs/sync |
|---|---|---|---|
| Full: 0 clients hidden | full_epochs_1 | full_epochs_2 | full_epochs_10 |
| 1 client randomly hidden/sync | 1_off_epochs_1 | 1_off_epochs_2 | 1_off_epochs_10 |
| 2 clients randomly hidden/sync | 2_off_epochs_1 | 2_off_epochs_2 | 2_off_epochs_10 |
| 3 clients randomly hidden/sync | 3_off_epochs_1 | 3_off_epochs_2 | 3_off_epochs_10 |

“Hidden” excludes the randomly selected client(s) from both client-to-server upload and server-to-client broadcast for that sync. The full condition has all five clients visible. Hiding is re-sampled each sync.

### Fixed protocol

- CIFAR-100, five clients, five tasks, 20 classes/task, non-IID alpha 10.0.
- ResNet-18 and FedAvg; 50 global rounds (10 rounds per task), with aggregation and global-model broadcast at every round.
- Seed 42, local batch size 13, learning rate 0.005, memory size 500, memory batch size 13, V=200, delta=1.
- The varied parameter is local_epochs (-ls): 1, 2, or 10 local epochs for each participating client at each sync. Other run-shaping settings are held constant.
- This is a single seed per cell; current logs are execution evidence, not a completed comparison or statistical result.

### Execution waves

To limit concurrency to six processes and balance expected client-round work, the controller runs two waves:

1. full and 3_off, each at 1, 2, and 10 epochs (six jobs).
2. Only after wave 1 exits, 1_off and 2_off, each at 1, 2, and 10 epochs (six jobs).

At the last observation, wave 1 had four active runs with round markers in their logs: full_epochs_1 (48), full_epochs_2 (35), full_epochs_10 (14), and 3_off_epochs_10 (30). 3_off_epochs_1 exited during evaluation with CUBLAS_STATUS_ALLOC_FAILED. 3_off_epochs_2 reached all 50 round markers and logged final accuracy metrics, but then failed at save_results because the initial launcher passed an absolute dataset path while this code concatenates the dataset string into a relative result filename. Its .h5 result was not written. The controller writes each process status only after its ordered wait, so status.tsv was not yet present while it remained blocked on the first job.

full_epochs_1 and full_epochs_2 logs contain CUDA out-of-memory exceptions while they remain active. These runs may have missed client updates; treat their results as potentially incomplete until logs and artifacts are audited. full_epochs_10 and 3_off_epochs_10 had no traceback/OOM signatures in the logs at the last scan.

### Resource contention and output-path correction

At 03:51:48 IST, the V100 reported 32,494 MiB used of 32,768 MiB, 7 MiB free, and 100% utilization. Four unrelated Correlation-Aware-FCL eight-run campaign processes accounted for 20,460 MiB in the process snapshot; no jobs were stopped. The near-full shared GPU coincided with the failed allocation and OOM traces.

The launcher file has since been corrected to pass a relative ../dataset/... path and create a per-run dataset symlink, preserving the application’s relative result-path behavior. The already-running controller retains the original absolute dataset argument in its in-memory shell state, so nested results/home/anurag24092/Projects/FedQCL_Task/dataset/ directories were created in all 12 run workdirs to give still-running and queued current-campaign processes a destination for their .h5 outputs. This does not recover the already exited 3_off_epochs_2 artifact. No run was restarted.

### Source and artifacts

The Mtech checkout was dirty at launch: HEAD 49d709b0227e63198cf0207d94d7016b46f6c476, with tracked changes in README.md, system/flcore/clients/clientavg_tl.py, system/flcore/servers/serveravg.py, and system/main.py. Runs used that working tree, not a clean checkout. The experiment outputs/logs and per-run isolated working directories are under the campaign root above. The prelaunch attempt at campaigns/cifar100_12_local_epoch_runs_20261122_011705 is a separate failed Python-2 startup attempt and is not part of this 12-run suite.

**2026-11-22 07:53:39 IST A–G progress check:** Direct SSH reconfirmed host
`MtechDEServer1`. A_clients_2_task_il_seed0 worker PID 16232 was alive at
11m25s elapsed and 146% CPU; its log had advanced to learning sync 6/26, with
the sync-6 accuracy line at 0.8675. Its current per-sync outputs, metrics,
metadata, schedule, scenario, log, and status were copied to Kakarot. A new
`tools/sync_active_runs_from_mtech.py` staged the run without deletion and
verified all 37 copied file sizes and SHA-256 digests against a post-transfer
server inventory (1,914,319 bytes total). Seven checkpoint files
(3,831,164,986 bytes total) remain on Mtech and
are listed in the transfer manifest.

The same snapshot showed V100 6,986 MiB used / 25,515 MiB free at 99%
utilization, 110 GiB host memory available, and 619 GiB free on the project
filesystem. A–G controller PID 34759 remained alive; its queue file was waiting
for capacity, cursor 1/29, with A_clients_10_class_il_seed0 next and several
external campaign commands listed. The cursor may be stale because that
10-client Class-IL run is already terminal; the runner has a completed-output
guard, so a repeated dispatch should return `already_completed` without
training or overwriting. No second A–G worker was started and no process was
stopped.

The 3-client A plans assign the 50 disjoint two-class tasks as 17/17/16 over
17 learning syncs. At sync 17, two clients train their final tasks and the
third has no new task. This is not an omitted client from the full-participation
communication cohort: with `participation=1.0`, the runtime selects all three
clients; the idle client's anchored upload is empty and excluded from FedAvg,
while it still receives the committed model. This tail is documented as a
finite-CIFAR-100 constraint, not as a 3-client partial-participation arm.

**2026-11-22 07:59:02 IST A–G progress check:** The same A2 Task-IL worker
PID 16232 remained in `running` state after 16m48s, and its log advanced from
sync 6 to sync 8 of 26. The sync-8 report covers 32 observed/integrated classes
and records 87.59375% test accuracy and 86.25% validation accuracy under the
Task-IL evaluation protocol. The V100 reported 7,026 MiB used / 25,475 MiB
free at 100% utilization. The controller remained at cursor 1/29, waiting for
capacity with external campaigns active; no new A–G worker was admitted. The
active transfer refresh copied 45 non-checkpoint files (2,396,604 bytes) and
verified hashes; nine server-only checkpoint files total 4,945,235,926 bytes.
These metrics are an intermediate sync result, not a terminal-run result.

**2026-11-22 08:04:26 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running (22m12s elapsed, 149% CPU) and advanced to sync 9/26.
Its report covers 36 observed classes and records 87.9722% test accuracy and
87.7778% validation accuracy under Task-IL evaluation. The queue still had
cursor 1/29 and listed the same external campaigns; no additional A–G worker
was admitted. V100 use was 7,026 MiB / 32 GiB, with 25,475 MiB free and 100%
utilization; system memory availability was 110 GiB and project filesystem
free space was 617 GiB. The Kakarot active-run snapshot now contains 49
non-checkpoint files (2,646,984 bytes), all size/hash-verified; ten checkpoint
files (5,505,413,716 bytes) remain on Mtech. Continue waiting for at least
sync 10 before evaluating this run's learning behavior.

**2026-11-22 08:09:39 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 27m24s and its log reached sync 11/26. The
requested sync-10 gate is now met: all 40 seen classes are integrated, test
accuracy is 88.775%, validation accuracy is 88.65%, all 20 seen task heads are
present, and the report lists no zero-prediction classes or pending classes.
Sync 11 covers 44 classes and records 89.1136% test / 86.9545% validation
accuracy, with 22 task entries and no zero-prediction classes. These reports
show no current sign of the earlier near-zero accuracy pathology in this
Task-IL arm; they are still intermediate and use task-aware Task-IL scoring.
The A–G controller remained waiting at position 1/29 with the same external
campaigns listed. V100 had 7,026 MiB used / 25,475 MiB free at 99%
utilization; available host memory was 110 GiB and project free space 616 GiB.
Kakarot's staged snapshot now has 57 non-checkpoint files (3,175,808 bytes)
verified by size and SHA-256; 12 checkpoints (6,632,054,000 bytes) remain on
Mtech. No jobs were started or stopped in this check.

**2026-11-22 08:15:55 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 33m41s and advanced to sync 12/26. The report
covers 48 classes, with 89.125% test and 89.0833% validation accuracy and no
zero-prediction classes. The queue remained at position 1/29, waiting on the
same external campaigns; its recorded update time was 08:05:41 IST. V100 use
was 7,026 MiB / 32 GiB with 25,475 MiB free at 99% utilization. Available host
memory was 110 GiB and project free space 616 GiB. Kakarot now has 61
non-checkpoint files (3,455,550 bytes) for this active run, all verified by
size and SHA-256; 13 checkpoint files totaling 7,198,516,494 bytes remain on
Mtech. No other A–G worker was admitted and no process was stopped.

**2026-11-22 08:22:06 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 39m52s and advanced to sync 13/26. Its report
covers 52 classes and records 88.8462% test / 89.4231% validation accuracy;
there are no zero-prediction classes. The controller remained waiting at
position 1/29 with the same external jobs listed. The V100 reported 7,026 MiB
used / 25,475 MiB free at 100% utilization; host memory available was 110 GiB
and project filesystem free space was 615 GiB. Kakarot's active snapshot now
contains 65 non-checkpoint files (3,743,921 bytes), all size/hash-verified;
14 checkpoints (7,767,074,758 bytes) remain on Mtech. The sync-13 report is
intermediate. No jobs were started or stopped.

**2026-11-22 08:28:03 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 45m49s and advanced to sync 15/26. Its report
covers 60 classes and records 90.5% test / 89.8667% validation accuracy with
no zero-prediction classes. The queue stayed at position 1/29, waiting on the
same external campaigns; V100 usage was 7,026 MiB with 25,475 MiB free at 99%
utilization. Host memory availability was 110 GiB and project free space was
614 GiB. Kakarot now has 73 non-checkpoint files (4,350,728 bytes), each
size/hash-verified; 16 checkpoints (8,910,481,146 bytes) remain on Mtech. No
other A–G worker was admitted and no process was stopped.

**2026-11-22 09:05:01 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 is still `running`; it has not emitted a new sync after sync 20/26
since the 08:52:14 check. Its elapsed time advanced to 1h22m47s and CPU time
to 2h08m17s (about 11m45s more CPU time over the preceding 6m21s), so the
process is actively computing despite the unchanged completed-sync counter.
The worker log has no new error lines. V100 usage was 17,338 MiB with 15,163
MiB free at 100% utilization, while other FedQCL jobs remained in the queue's
external-job list. The A–G controller continued to wait at position 1/29.
Kakarot's refresh verified the same 93 non-checkpoint files (6,027,846 bytes);
the same 21 checkpoints (11,805,679,132 bytes) remain on Mtech. Treat this as
slow sync progress under GPU contention, not as a confirmed stall or failure;
recheck at the next five-minute window. No process was stopped.

**2026-11-22 09:12:33 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained in `Rsl` state with elapsed time 1h30m18s and CPU time
2h17m31s. The last completed sync is still 20/26, unchanged since 08:52:14.
The latest `checkpoint_latest.pt` mtime is 08:49:16 IST; the active worker's
`/proc/16232/io` reports 22,490,631,827 bytes written, and it holds 3,304 MiB
of GPU memory. `nvidia-smi pmon` showed other CUDA workers using the remaining
GPU, while aggregate utilization was 100% (17,614 MiB used / 14,887 MiB free).
The worker log has no new error, and it continues to accumulate CPU time, but
no sync/checkpoint has advanced for roughly 20 minutes. Kakarot re-verified
the unchanged 93 non-checkpoint files (6,027,846 bytes); 21 checkpoints
(11,805,679,132 bytes) remain server-side. This is now a prolonged sync delay
under shared GPU/disk load. Keep it running and compare the same counters at
the next five-minute check; do not restart or terminate it.

**2026-11-22 09:19:12 IST A–G progress check:** The same worker PID 16232
remains running at sync 20/26 (elapsed 1h36m58s, CPU time 2h24m11s). Relative
to 09:12:33, CPU time advanced by 6m40s, but `/proc/16232/io` counters did not
change, the log did not add a sync, and the newest checkpoint still has mtime
08:49:16 IST. A one-sample `nvidia-smi pmon` showed PID 16232 allocated
3,304 MiB and at 9% SM / 3% memory utilization while other Python jobs shared
the GPU (aggregate 17,800 MiB used / 14,701 MiB free, 100% utilization).
This is an extended unreported local phase with some CPU/GPU activity; it is
not yet proven deadlocked. The queue remains waiting at position 1/29. Kakarot
reverified the same 93 non-checkpoint files (6,027,846 bytes), plus the 21
server-only checkpoints (11,805,679,132 bytes). At the next allowed check,
compare the sync log, checkpoint mtime, I/O counters, and per-process GPU use;
leave the running process untouched.

**2026-11-22 09:31:47 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at elapsed 1h49m33s / CPU time 2h42m27s; the latest
completed sync remains 21/26. The A–G controller is still waiting at position
1/29, with FedQCL, task-size, and sync-frequency campaign controllers/jobs in
its external-job list. The V100 reports 18,150 MiB used / 14,351 MiB free at
100% utilization; host memory available was 85 GiB and project free space
610 GiB. No new output appeared, so the Kakarot snapshot remains at 97
non-checkpoint files (6,391,891 bytes) with verified hashes and 22 Mtech-only
checkpoints (12,391,007,062 bytes). The worker is active between logged sync
21 and its next sync; the log does not expose its exact local subphase. No
jobs were stopped.

**2026-11-22 09:37:51 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at elapsed 1h55m37s / CPU time 2h53m35s, with sync
21/26 still the latest completed event. The A–G controller remained waiting
at position 1/29 behind the same campaign families. V100 use was 18,440 MiB /
32 GiB with 14,061 MiB free at 100% utilization; host memory available was
85 GiB and project free space 610 GiB. No new report appeared; the active
Kakarot snapshot remains 97 non-checkpoint files (6,391,891 bytes), all
hash-verified, and 22 server-only checkpoints (12,391,007,062 bytes). No
additional A–G worker started and no process was stopped.

**2026-11-22 09:49:42 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 2h07m27s elapsed / 3h10m05s CPU, still between
syncs 21 and 22. Since 09:43:47, CPU time advanced about 5m55s, but the sync
log, `/proc` I/O counters, and checkpoint mtime remained unchanged. V100 use
was 18,684 MiB used / 13,817 MiB free at 100% utilization; host memory
available was 85 GiB and project free space 610 GiB. The A–G controller
remained waiting at 1/29 behind external campaigns. Kakarot's active snapshot
is unchanged at 97 non-checkpoint files (6,391,891 bytes), SHA-256 verified,
with 22 Mtech-only checkpoints totaling 12,391,007,062 bytes. No job was
started or stopped in this check.

**2026-11-22 10:01:20 IST A–G progress check:** The snapshot caught the worker
between log writes: `checkpoint_latest.pt` had advanced to 10:01:19 while the
initial `tail` still ended at sync 21. The immediate non-checkpoint transfer
then found sync 22/26, confirming forward progress. A_clients_2_task_il_seed0
PID 16232 remained running at elapsed 2h19m06s / CPU time 3h21m44s. Sync 22
covers 88 classes with 90.8068% test / 90.75% validation accuracy and no
zero-prediction classes. The A–G controller remains waiting at 1/29 behind
other campaigns. V100 reported 16,538 MiB used / 15,963 MiB free at 100%
utilization; host available memory was 89 GiB and project free space 610 GiB.
The Kakarot snapshot now has 101 non-checkpoint files (6,763,451 bytes), all
hash-verified; 23 checkpoints totaling 12,978,431,120 bytes remain on Mtech.
No process was stopped.

**2026-11-22 10:07:30 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at elapsed 2h25m16s / CPU time 3h33m04s, and sync
22/26 remained the latest completed report. The A–G controller stayed waiting
at position 1/29 with the same external campaigns. V100 reported 16,738 MiB
used / 15,763 MiB free at 100% utilization; host memory availability was
90 GiB and project free space 610 GiB. No new non-checkpoint outputs appeared;
the Kakarot snapshot remains 101 files (6,763,451 bytes), all hash-verified,
and 23 server-only checkpoints (12,978,431,120 bytes). No additional A–G
worker started and no process was stopped.

**2026-11-22 09:55:29 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 2h13m15s elapsed / 3h15m53s CPU; the log remains
at sync 21/26 and `/proc` I/O plus `checkpoint_latest.pt` mtime are unchanged
since the previous sample. The V100 reported 16,378 MiB used / 16,123 MiB
free at 100% utilization; host memory availability was 90 GiB and project
free space 610 GiB. The A–G queue is still waiting at position 1/29. Kakarot's
97-file active snapshot remains SHA-256 verified; 22 checkpoints remain on
Mtech. CPU time advances while the client update remains unfinished. The
last such phase resolved after roughly 33 minutes; continue non-invasive
five-minute checks. No process was stopped.

**2026-11-22 09:43:47 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 2h01m32s elapsed / 3h04m10s CPU, with the log
still at sync 21/26. Relative to 09:37:51, CPU time advanced 10m35s while no
new sync, I/O counter, or checkpoint timestamp appeared. V100 use was
18,594 MiB used / 13,907 MiB free at 100% utilization; host memory available
was 85 GiB and project free space 610 GiB. The A–G queue still waited at
position 1/29 behind other campaigns. Kakarot's unchanged snapshot has 97
non-checkpoint files (6,391,891 bytes), hash-verified, and 22 server-only
checkpoints (12,391,007,062 bytes). The worker remains CPU-active between
syncs; this is a slow local interval, not yet a terminal failure. No process
was stopped.

**2026-11-22 09:25:46 IST A–G progress check:** The apparent long sync-20
phase completed without intervention. A_clients_2_task_il_seed0 PID 16232
advanced to sync 21/26; the checkpoint mtime advanced to 09:24:55 IST and
`/proc/16232/io` write counters rose from 22.49 GB to 23.66 GB since the prior
snapshot. The sync-21 report covers 84 classes and records 90.9643% test /
91.0238% validation accuracy, with no zero-prediction classes. The worker is
still running and the queue still waits at position 1/29; several FedQCL CUDA
workers share the V100 (17,900 MiB used / 14,601 MiB free, 100% utilization).
The 97 non-checkpoint files (6,391,891 bytes) were transferred to Kakarot and
all size/hash checks passed; 22 checkpoints totaling 12,391,007,062 bytes
remain on Mtech. The previous interval was a very slow task-training phase
under shared-GPU load, not a confirmed stall. No process was stopped.

**2026-11-22 08:34:02 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 51m48s and advanced to sync 16/26. The report
covers 64 classes and records 91.0625% test / 90.125% validation accuracy,
with no zero-prediction classes. The controller remained waiting at position
1/29 with the same external campaigns. V100 used 7,026 MiB with 25,475 MiB
free at 99% utilization; host available memory was 110 GiB and project free
space was 613 GiB. The verified Kakarot snapshot contains 77 non-checkpoint
files (4,668,796 bytes); 17 server-only checkpoints total 9,485,328,500 bytes.
No additional worker was admitted and no job was stopped.

**2026-11-22 08:40:02 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 57m48s and advanced to sync 17/26. The report
covers 68 classes and records 89.8971% test / 89.6176% validation accuracy,
with no zero-prediction classes. The A–G queue is still waiting at position
1/29 with external campaigns present. V100 used 7,026 MiB with 25,475 MiB
free at 99% utilization; host memory available was 110 GiB and project free
space 613 GiB. Kakarot's active snapshot has 81 non-checkpoint files
(4,996,097 bytes), all size/hash-verified; 18 checkpoints totaling
10,062,271,982 bytes remain on Mtech. No additional worker started and no
process was stopped.

**2026-11-22 08:46:07 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 1h03m53s and advanced to sync 18/26. The report
covers 72 classes and records 90.875% test / 89.7778% validation accuracy,
with no zero-prediction classes. The controller remained waiting at position
1/29 with the same external campaign commands. V100 reported 7,026 MiB used / 25,475 MiB free at 98% utilization; host available memory was 110 GiB and
project free space 612 GiB. Kakarot's staged run now contains 85
non-checkpoint files (5,329,700 bytes), all size/hash-verified; 19
checkpoints totaling 10,641,311,592 bytes remain on Mtech. No second A–G
worker started and no process was stopped.

**2026-11-22 08:52:14 IST A–G progress check:** A_clients_2_task_il_seed0
PID 16232 remained running at 1h09m59s and advanced to sync 20/26. Its report
covers 80 classes and records 91.1125% test / 91.75% validation accuracy,
with no zero-prediction classes. The A–G queue stayed at position 1/29,
waiting on external campaigns. Since the prior snapshot, several FedQCL
one-epoch and two-epoch jobs appeared in the controller's external-job list.
V100 memory usage rose to 15,976 MiB used / 16,575 MiB free at 100%
utilization; the A2 worker remained alive and progressing. Host memory
available was 86 GiB and project free space 611 GiB. The active Kakarot
snapshot contains 93 non-checkpoint files (6,027,846 bytes), all size/hash-
verified; 21 checkpoints totaling 11,805,679,132 bytes remain on Mtech. No
other A–G worker was admitted and no process was stopped.

**2026-11-22 10:20:21 IST live queue refresh:** Direct SSH confirmed controller PID 29969 is still alive and the six-run campaign remains `waiting_for_capacity`, at position 1/6. No intra-task sync-frequency training worker has started. The V100 snapshot was 16,794 MiB used / 15,707 MiB free at 100% utilization; host RAM available was 90 GiB and project filesystem free space was 610 GiB. The controller requires 24,576 MiB free VRAM, so it correctly remains queued. No jobs were stopped or altered. This is a current queue snapshot, not a training result.

**2026-11-22 10:26:30 IST combined campaign check:** The sync-frequency controller (PID 29969) remains alive and `waiting_for_capacity`, position 1/6; Task-IL / 2-sync is next. It needs 24,576 MiB free, while the V100 had 18,751 MiB free at 100% utilization. The A–G controller (PID 34759) is also waiting at cursor 1/29; its worker A_clients_2_task_il_seed0 (PID 16232) is still running (elapsed 2:44:15; CPU 4:00:26), with substantial cumulative reads/writes. Its newest saved evaluation is still sync 22 (88 classes; 90.8068% task-aware test accuracy), so there is no newer accuracy report in this snapshot. The active-run transfer helper refreshed and SHA-verified available non-checkpoint artifacts. No jobs were stopped or changed.

**2026-11-22 10:32:42 IST live check:** The independent intra-task sync campaign controller PID 29969 is alive and remains `waiting_for_capacity`, position 1/6, with Task-IL / 2-sync next. Free VRAM is 18,659 MiB, below its 24,576 MiB gate; the A–G controller is also waiting for the same capacity. The active A_clients_2_task_il_seed0 worker PID 16232 remains alive at elapsed 2h50m28s / CPU 4h07m05s. Its checkpoint mtime advanced to 10:32:10 IST and sync 23/26 is now saved: 92 observed classes, 91.25% task-aware test accuracy and 91.5652% validation accuracy. The non-checkpoint snapshot was refreshed and hash-verified (105 files, 7,143,174 bytes); 24 checkpoints remain server-side. No jobs were stopped or changed.

**2026-11-22 10:39:46 IST check:** Both campaign controllers remain alive but gated: the intra-task sync suite is waiting at position 1/6, and A–G waits at cursor 1/29. The V100 reports 21,231 MiB free, still 3,345 MiB below the sync suite’s 24,576 MiB threshold. A_clients_2_task_il_seed0 PID 16232 remains alive (2h57m32s elapsed; CPU 4h20m13s), but the latest sync is still 23/26 and checkpoint mtime is unchanged since 10:32:10. Its CPU time advanced by about 13 minutes since the prior check, while `/proc` I/O counters were unchanged; no terminal state or new report is evident. A fresh transfer refresh verified the same 105 non-checkpoint files (7,143,174 bytes) and 24 server-side checkpoints. This is a prolonged unreported phase with ongoing CPU activity; keep it running and compare at the next allowed check. No jobs were stopped or changed.

**2026-11-22 10:45:53 IST check:** Sync-frequency controller PID 29969 remains `waiting_for_capacity` at 1/6; A–G controller PID 34759 also waits at 1/29. Free VRAM increased to 21,175 MiB but remains below the 24,576 MiB gate. A_clients_2_task_il_seed0 PID 16232 is still alive at 3h03m39s elapsed / 4h29m27s CPU. Its latest report remains sync 23/26 and checkpoint mtime remains 10:32:10 IST; `/proc` I/O counters also match the previous sample. No terminal state or new saved evaluation is present. Keep the worker untouched; inspect its current log and per-process GPU activity at the next check if the delay persists.

**2026-11-22 10:52:06 IST check:** Both suite controllers remain alive and waiting for capacity. The intra-task sync suite remains at position 1/6 with Task-IL / 2-sync next; V100 free VRAM is 21,127 MiB, 3,449 MiB below its 24,576 MiB gate. A_clients_2_task_il_seed0 PID 16232 remains alive (elapsed 3h09m52s / CPU 4h35m40s), but sync 23/26 and checkpoint mtime 10:32:10 are unchanged. `/proc` I/O counters have not changed since the 10:32 snapshot. The single `nvidia-smi pmon` sample showed PID 16232 present on the GPU but no reported per-process SM/memory percentages; two other compute PIDs were active. The worker log mtime is 10:22:12, with no newer log write. CPU time continues to accumulate, so this is a prolonged silent phase rather than evidence that the process exited. Do not restart or terminate it; inspect the worker status/log and per-process memory at the next allowed check.

**2026-11-22 10:58:11 IST live check:** Sync 24/26 is now saved for A_clients_2_task_il_seed0: 96 observed classes, 91.9583% task-aware test accuracy and 91.8958% validation accuracy. PID 16232 remains alive and its log/checkpoint advanced; this leaves one learning sync and the terminal communication-only flush. The refreshed, hash-verified Kakarot snapshot has 109 non-checkpoint files (7,533,061 bytes); 25 checkpoints remain server-side. The A–G controller and the independent intra-task sync controller remain waiting for capacity. V100 free VRAM was 21,127 MiB against the sync suite's 24,576 MiB admission threshold. No jobs were stopped or changed.

**2026-11-22 11:04:45 IST check:** The intra-task sync controller PID 29969 remains `waiting_for_capacity`, position 1/6; the A–G controller PID 34759 remains waiting at 1/29. V100 free VRAM is unchanged at 21,127 MiB, below the sync campaign's 24,576 MiB gate. A_clients_2_task_il_seed0 PID 16232 remains alive and reported `running` (3h22m31s elapsed, 4h57m19s CPU, 146% CPU, about 6.2 GiB RSS). The latest saved report remains sync 24/26 and checkpoint mtime 10:54:30; `/proc` I/O and worker log are unchanged from 10:58. Its process is still accumulating CPU time, so the quiet interval is not evidence of termination. Leave it running and compare again after the required interval.

**2026-11-22 11:11:06 IST check:** The six-run intra-task sync controller remains alive and `waiting_for_capacity` at 1/6; A–G also remains queued at 1/29. Their queue-file samples report free-VRAM readings of 20,973 and 24,059 MiB respectively; the direct `nvidia-smi` sample at 11:11:06 showed 24,059 MiB free, 517 MiB below the 24,576 MiB admission threshold. A_clients_2_task_il_seed0 PID 16232 remains `running` at 3h28m52s elapsed / 5h03m40s CPU, about 145% CPU; its report remains sync 24/26 and latest checkpoint remains 10:54:30. I/O counters and worker log have not advanced since 10:58, but the process remains alive and CPU-active. No jobs were stopped or altered.

**2026-11-22 11:16:58 IST live check:** A_clients_2_task_il_seed0 advanced to sync 25/26, covering all 100 classes at 91.96% task-aware test accuracy and 91.90% validation accuracy. Only the communication-only terminal flush remains. PID 16232 is alive; its checkpoint advanced to 11:14:16. The refreshed Kakarot snapshot contains 113 non-checkpoint files (7,930,788 bytes), hash-verified; 26 checkpoints remain server-side. Both the A–G controller and the six-run intra-task sync controller are still waiting; live V100 free VRAM is 24,007 MiB, 569 MiB below their 24,576 MiB gate. No jobs were stopped or changed.

**2026-11-22 11:22:56 IST A–G terminal update:** A_clients_2_task_il_seed0 completed. Sync 26/26 was the terminal communication-only flush and retained sync 25 accuracy: 91.96% task-aware test / 91.90% validation over all 100 classes. Terminal sync copied 119 non-checkpoint files (8,464,668 bytes) and 28 server-only checkpoints. A local audit verified every copied file against its manifest size and SHA-256 (zero errors). All four A runs are now terminal; 25 of the 29 A–G runs remain. Both controllers' sampled queue files still said waiting, while the direct V100 reading showed 27,263 MiB free; the queue-file timestamps need refresh before claiming either controller has advanced. No job was stopped or modified.

## Post-cutoff suite inventory addendum — suites created after 2026-11-21 20:00 IST

This addendum records three campaigns that were present on the servers but
missing from this Kakarot ledger at the time of audit. The cutoff is 20:00 IST
on 2026-11-21. Counts below are unique intended cells; retries and malformed
launch attempts are recorded as execution history, not additional ablation
conditions.

### Suite 4 — V25 incremental curriculum, full vs. random-offline

**Created:** 2026-11-21 20:26 IST  
**Host:** MtechDEServer (`192.168.3.30`)  
**State:** All 6 unique runs completed.  
**Seed:** 0 for each run.  
**Server root:** `/home/anurag24092/Projects/Correlation-Aware-FCL-six-runs-20261121`

This suite compared full participation with one randomly offline client per
learning sync for the V25 Replay/FedAvg setting. Its six cells were Task-IL
disjoint, Task-IL joint adjacent-chain overlap, and Class-IL disjoint, each at
full and random-offline participation. It used five clients, 10 local epochs
per task, persistent client momentum, crop/flip augmentation, sampled 1:1
current/replay training, and the original communication-after-task rule. The
disjoint curricula have 10 learning syncs plus one communication-only flush;
the joint-overlap curricula have 20 learning syncs plus a flush. Per-sync
reports and run summaries are under `runs/<run-id>/` in the server root.

### Suite 5 — V25 participation levels 2 and 3 offline

**Created:** 2026-11-22 01:13 IST; corrected retry root created 01:27 IST  
**Host:** MtechDEServer (`192.168.3.30`)  
**State:** The corrected retry completed all 8 intended cells.  
**Seed:** 0 for each run.  
**Initial attempt:** `/home/anurag24092/Projects/Correlation-Aware-FCL-eight-runs-20261122`  
**Completed retry:** `/home/anurag24092/Projects/Correlation-Aware-FCL-eight-runs-20261122-retry`

The intended matrix is Task-IL/Class-IL × disjoint/joint adjacent-chain
curriculum × 2/3 randomly offline clients (8 unique cells). The first
campaign root is malformed: its manifest lists duplicate job IDs and only
joint arms, so those outputs are not the authoritative full matrix. The
corrected retry has unique output paths and a finished manifest with all 8
cells completed. Count this as one eight-cell suite with a failed initial
launch attempt and a completed retry, not as two independent eight-run
ablations. The retry reuses launch code in the initial root.

### Suite 6 — V100 local epochs × participation × IL × curriculum

**Created:** 2026-11-22 02:14 IST  
**Host:** V100 server (`192.168.2.252`)  
**Latest verified snapshot (2026-11-22 12:13 IST):** Controller PID 376487
was alive but `waiting_for_gpu`; all 48 V2 rows remained queued and zero
training runs had launched. The V100 reported 25,564 MiB used and 6,936 MiB
free at 100% utilization. Host RAM was 93 GiB total, 20 GiB used, and 70 GiB
available.  
**Seed:** 0.  
**Implementation root:** `/home/anurag24092/Projects/Correlation-Aware-FCL-mtech-fullcopy-20261122/v25_offline_epoch_grid_20261122_v2`  
**Launcher:** `scripts/run_v25_offline_epoch_grid.py`

This is one 48-cell grid: 4 participation levels (0/1/2/3 randomly offline)
× Task-IL/Class-IL × disjoint/joint curriculum × 1/2/5 local epochs per task.
The first root, `results/v25_offline_epoch_grid_20261122/`, has a failed
planning manifest (`plan_failed`); its malformed attempt is not an additional
suite. The corrected root, `results/v25_offline_epoch_grid_20261122_v2/`,
contains the 48 intended unique cells. Its original controller is configured
for 12 blocks of four jobs and a 28,000 MiB admission gate. At the latest
verified snapshot that gate left every row queued and no training run started.
The first and corrected roots are both recorded here as execution history for
the same matrix. This V100 suite is distinct from the FedQCL epoch suite and
the Mtech A–G campaign. No other post-cutoff V100 ablation suite was found in
the audited V100 campaign roots; the separate Mtech suites are listed above.

**2026-11-22 12:23:52 IST launch update:** The 48-cell design and outputs
remain unchanged. To avoid leaving the campaign idle behind the old VRAM
threshold, the V100 launcher was updated to use one worker at a time with no
free-VRAM admission gate, and a serial-resume controller was started against
the corrected V2 manifest. The earlier controller (PID 376487 at the 12:13
snapshot) was not stopped; worker-level `flock` serializes its eventual child
processes with the new controller, and completed worker outputs are treated
idempotently. The Python controller is PID 414350 (launched by shell PID
414348), and the first worker is PID 414351,
`V25_TASKIL_DISJOINT10_ALLSYNC_EPOCHS1_REPLAY_FEDAVG_SEED0`; its
`worker_status.json` says `running`. This is launch evidence only; no accuracy
result is available yet. The override records separate progress at
`results/v25_offline_epoch_grid_20261122_v2/sequential_override/manifest.json`
and leaves the original V2 manifest intact. Kakarot's mirrored launchers are
under `code/v100/`: `run_v25_offline_epoch_grid.py` SHA-256
`11502aa841abdb0785868a30d7bfcdb5e631cecc0f035410d95b0d987fb118f3` and
`run_v25_offline_epoch_grid_sequential_override.py` SHA-256
`f129484d9668ab3a4973419010b3375bb9852d55e188648c9fe3f547bca9975b`;
the corresponding remote files had matching hashes immediately after copy.
The previous server launcher was preserved before replacement at
`scripts/archive/run_v25_offline_epoch_grid.pre_serial_20261122_1225.py`,
SHA-256 `caa589ada726bf1a88fb6273c0a89a44367598247fab1f43f6fc93ad34c42a2e`.

### Suite 3 status addendum — FedQCL failures and retries

**Last evidence:** Kakarot top-level suite log, 2026-11-22 11:41 IST.  
In the 12-cell FedQCL campaign, the original controller had 8 runs marked
completed with `.h5` outputs, 2 failed, and 2 still active. The `full_epochs_1`
and `full_epochs_2` logs contain CUDA OOM exceptions despite successful exit
and result files, so their artifacts require quality review before treating
them as complete measurements. The failed `3_off_epochs_1` and
`3_off_epochs_2` cells were relaunched in isolated retry directories as
`3_off_epochs_1_retry1` (PID 32140) and `3_off_epochs_2_retry1` (PID 32142).
At 11:41 both retries were active, with no retry `.h5` outputs yet. The
original `1_off_epochs_10` and `2_off_epochs_10` cells were also active at
that observation. These are historical states; refresh before reporting
current activity.


## FedQCL_Task CIFAR-100 ablations — scope and provenance addendum (2026-11-22)

This append-only entry records the two related FedQCL_Task experiment blocks on MtechDEServer and distinguishes their purposes. These are FedQCL_Task runs, not the separate Correlation-Aware-FCL/V25 suites in this ledger.

### Reference block: four visibility runs at five local epochs per sync

The initial CIFAR-100 experiment consists of four runs: all five clients visible at every sync (`all`), or 1, 2, or 3 clients randomly hidden at each sync. All four use the original five-local-epochs-per-sync setting. This is a visibility comparison, not the later 1/2/10 local-epoch factorial ablation. The selected hidden clients are excluded in both directions for that sync: they neither upload local updates to the server nor receive the server aggregate; the selection is resampled at each sync.

Server logs: `/home/anurag24092/fedqcl_cifar100_all.log`, `fedqcl_cifar100_one_hidden.log`, `fedqcl_cifar100_two_hidden.log`, and `fedqcl_cifar100_three_hidden.log`. Kakarot's `~/Trash/results_2/` contains `full/`, `1 off/`, `2 off/`, and `3 off/`; each folder has 150 averaged-test-accuracy records in `per_sync_accuracy.csv` and 150 corresponding PNG tables. These are global averaged accuracy records; the source logs do not provide the client-by-class cells available in the distinct `results_1` artifact. Do not interpret 150 evaluation records as 150 server rounds: this original run configuration has 50 server rounds.

### Suite 3: local epochs per sync × client visibility (12 intended cells)

The later campaign holds the FedQCL_Task CIFAR-100 protocol fixed and crosses local epochs per sync (1, 2, 10) with the same four visibility levels (0, 1, 2, 3 hidden clients):

| Clients hidden each sync | 1 local epoch | 2 local epochs | 10 local epochs |
|---|---|---|---|
| 0 (full visibility) | `full_epochs_1` | `full_epochs_2` | `full_epochs_10` |
| 1 | `1_off_epochs_1` | `1_off_epochs_2` | `1_off_epochs_10` |
| 2 | `2_off_epochs_1` | `2_off_epochs_2` | `2_off_epochs_10` |
| 3 | `3_off_epochs_1` | `3_off_epochs_2` | `3_off_epochs_10` |

The five-client setting means 5, 4, 3, or 2 clients respectively can communicate and train in a sync. Hidden-client selection is random per sync. Visibility changes the number of client updates and broadcasts as well as participation; it is not a client dropout only on the upload path.

### Fixed experiment recipe and learning/communication sequence

- Dataset: CIFAR-100, partitioned into five sequential tasks of 20 classes each; five clients; non-IID class allocation uses Dirichlet alpha 10.0. The experiment seed is 42.
- Model/server: ResNet-18 and FedAvg. At each server round, participating clients perform local training, upload their model updates, the server aggregates received client models, and the resulting global model is broadcast to participating clients. No client excluded for that sync receives the broadcast.
- Client continual-learning state: the FedQCL_Task client implementation (`system/flcore/clients/clientavg_tl.py`) keeps task memory and Q values, writes task memory at task boundaries, and performs the configured Q updates over earlier task memories. These mechanisms are fixed across the visibility and epoch cells; this is not a replay-versus-no-replay or FedAvg-versus-another-aggregator ablation.
- Other fixed values recorded for the campaign: learning rate 0.005, local batch size 13, memory size 500, memory batch size 13, V=200, delta=1. There are 50 server rounds total, 10 per task. Consequently, a client present in every sync receives 10, 20, or 100 local epochs per task under the 1-, 2-, or 10-epoch cell. A hidden client has fewer participating syncs and therefore a random, lower realized training/broadcast count.
- The sole intended factors are local epochs per sync and number of clients hidden per sync. There is one seed per cell, so the matrix alone does not estimate across-seed variance.

### Execution plan, failures, retries, and result interpretation

The controller was configured for two waves of six jobs to cap simultaneous processes: wave 1 runs full visibility and 3-hidden at each of the three epoch counts; wave 2 runs 1-hidden and 2-hidden at each epoch count after wave 1 exits. The campaign root is `/home/anurag24092/Projects/FedQCL_Task/campaigns/cifar100_12_local_epoch_runs_20261122_012754`; the launcher is `/home/anurag24092/Projects/FedQCL_Task/launch_cifar100_12_runs.sh`. Runs used the Mtech checkout with HEAD `49d709b0227e63198cf0207d94d7016b46f6c476` and tracked local changes in README, client, server, and main files; this is the actual dirty working tree provenance, not a clean committed snapshot. The launch environment was the Correlation-Aware-FCL virtualenv (Python 3.12.15, PyTorch 2.5.1+cu121, CUDA 12.1), and preflight identified a Tesla V100-PCIE-32GB.

The first attempt at `campaigns/cifar100_12_local_epoch_runs_20261122_011705` used `/usr/bin/python` (Python 2) and failed before training; it is not part of the 12-cell experiment. In the corrected campaign, `3_off_epochs_1` failed during evaluation with a CUBLAS allocation error. `3_off_epochs_2` reached round 50 and logged final metrics but failed to save its result because the initial absolute dataset path interacted incorrectly with the code's relative result-path construction. The launcher was later corrected to use a relative dataset path and per-run dataset symlink; the already-running controller retained its original argument, so per-run output directories were prepared to accommodate that controller. The path fix does not recover the already exited missing artifact.

At the 2026-11-22 12:15:18 IST reconciliation, eight original processes had completed with `.h5` outputs, two original jobs (`1_off_epochs_10`, `2_off_epochs_10`) were still active, and the two original 3-hidden jobs above had failed. `full_epochs_1` and `full_epochs_2` had `.h5` files but their logs also contained CUDA OOM exceptions, so successful process exit/result writing is not sufficient evidence that every client update completed; treat those cells as potentially compromised. Two replacement executions were launched in an isolated retry directory for the failed 3-hidden cells. `3_off_epochs_1_retry1` had completed and saved an `.h5` file; `3_off_epochs_2_retry1` was still active at 39/50 round markers at that same observation. Retries replace failed executions and are not additional factorial cells. These counts and statuses are historical, not a current server-health claim.

The original four five-epoch runs and the 12-cell 1/2/10-epoch suite must be kept separate in summaries and cell counts. `~/Trash/results_2/` is for the four original visibility runs; it is not the result directory for the later 12-cell campaign. The campaign's per-run logs/status/results remain under its server-side campaign root above.

**Live-state limitation at append time:** a fresh SSH status query could not be obtained because the B550 Tailscale SSH hop requested reauthentication and timed out during banner exchange. Therefore the 12:15:18 IST server snapshot above is the latest verifiable state available in this update; refresh the server before reporting current completion or process health.

## Kakarot Trash run artifacts: `results_1` and `results_2`

The user-referenced `runs_1`/`runs_2` names correspond to the actual local
folder names `/home/anurag-basistha/Trash/results_1/` and
`/home/anurag-basistha/Trash/results_2/`; no folders literally named
`runs_1` or `runs_2` were found. This inventory documents the saved artifacts,
not a fresh server-state check.

### `Trash/results_1` — eight per-sync Replay/FedAvg diagram arms

The directory contains 128 PNGs (7,516,236 bytes), arranged as
joint/disjoint × Task-IL/Class-IL × FULL/OFFLINE. Each disjoint arm has 11
images (`SYNC_001`–`SYNC_011`); each joint arm has 21 (`SYNC_001`–`SYNC_021`).
The disjoint Task-IL FULL arm is the baseline diagram set cited by the A–G
goal. The matching `/home/anurag-basistha/Trash/results_1.zip` has 128 entries.
A sampled disjoint Task-IL FULL image identifies seed 0 and Replay/FedAvg V25
and shows global test/validation values, observed/integrated classes,
never-predicted classes, upload clients, task/client layout, and per-client,
per-class accuracy cells. The directory contains no machine-readable config,
checkpoint, metrics JSON, or run manifest; it therefore cannot establish every
training setting or run provenance on its own. The experiment recipe remains
reconstructed from the preserved runner and run notes, as described in
`GOAL.md` and `BASELINE_FIDELITY_AUDIT.md`.

### `Trash/results_2` — four FedQCL_Task visibility reference runs

The four conditions are full visibility and 1/2/3 randomly hidden clients per
sync; these are the original five-local-epochs-per-sync reference runs, not
the later 12-cell 1/2/10-epoch suite. Project records attribute them to the
FedQCL_Task CIFAR-100 protocol: five clients, five sequential 20-class tasks,
Dirichlet alpha 10, ResNet-18/FedAvg, seed 42, and 50 server rounds (10 per
task), with learning rate 0.005, batch 13, memory size/batch 500/13, V=200,
delta=1, and five local epochs per sync. The server logs named in the
FedQCL_Task addendum are the provenance references; the Trash folders contain
the tables, CSVs, and READMEs, not the original configs or logs.

Each condition folder (`full/`, `1 off/`, `2 off/`, `3 off/`) has 150 PNG
tables and 150 rows in `per_sync_accuracy.csv`; each row is global averaged
test accuracy only. There are 600 PNGs, four CSVs, five README files, 609
files total (10,843,216 bytes); `results_2.zip` also has 609 entries. The
tables/CSVs are numbered `SYNC_001`–`SYNC_150`, while the documented runner
configuration has 50 server rounds. The local artifacts do not explain the
150-to-50 counter mapping. Keep the observed 150-record index separate from
the 50-round protocol until the source log/emission code resolves it; do not
claim 150 global server rounds from the filenames.

| Condition | First record | Last record | Highest record value | Highest record index |
|---|---:|---:|---:|---:|
| Full | 5.40% | 59.95% | 60.55% | 145 |
| 1 hidden | 5.05% | 59.80% | 60.05% | 140 |
| 2 hidden | 4.45% | 58.05% | 58.05% | 150 |
| 3 hidden | 5.35% | 52.80% | 53.15% | 100 |

These are extrema/final values from the saved CSV records, not a per-class
analysis or proof that the highest-index record corresponds to the last
server round. The original 5-epoch reference block is excluded from the
seven-suite/121-cell ablation count; the separate 12-cell epoch-by-visibility
Suite 3 remains counted once.


### Suite 3 live status refresh — 2026-11-22 13:35 IST

Direct SSH to `MtechDEServer1` succeeded after Tailscale reauthentication. At 13:34:48 IST, controller PID 14328 was alive with two child training processes: PID 20842 (`1_off_epochs_10`) and PID 20844 (`2_off_epochs_10`). Both remained present at the 13:35:19 IST sample. Their logs were recently modified (13:19 and 13:27 respectively) and showed local training progress; the 2-off log showed end-of-task Q updates for task 4. The V100 was a Tesla V100-PCIE-32GB at 100% utilization, 11,764 MiB used and 20,737 MiB free. This establishes active processes and recent training output, not successful eventual completion.

## 2026-11-24 dual-server execution reconciliation

At 10:03 IST, the authoritative `CAMPAIGN_LEDGER_121.jsonl` contained 121
unique cells, all in terminal `completed` state. Counts by suite are task size
12, intra-task sync frequency 6, independent A–G 29, FedQCL local epochs ×
visibility 12, V25 incremental curriculum 6, V25 participation levels 8, and
the V25 local-epochs × participation × IL × curriculum grid 48. The four
five-epoch FedQCL reference runs remain separate from this 121-cell campaign,
as recorded above; they are pre-existing results rather than queued cells.

The 48-cell V25 grid ran after the other 73 cells: the latest recorded
non-grid terminal completion is Nov 23 at 19:00 IST, and the grid's recorded
terminal attempts run from Nov 24 at 03:21 IST onward. Attempt host assignments
reconcile to 31 cells completed on Mtech and 17 on V100. The 31 Mtech IDs
match the V100 manifest's delegated-to-Mtech IDs; the 17 V100 IDs match the
central ledger's V100 attempt assignments. The Mtech-delegated cell
`V25_TASKIL_DISJOINT10_OFFLINE1_EPOCHS2_REPLAY_FEDAVG_SEED0` is included in
V100's 17 verified completions. Both controller manifests have now been
reconciled to completed, their prior waiting manifests were backed up, and the
two idle campaign supervisors were stopped after confirming no training
workers remained. The reconciled manifests were published to their existing
Kakarot mirrors.

The central ledger's audit labels were refreshed from recorded terminal
attempt evidence, including the 13 V100 epoch-1 terminal audits. Its final
classification counts are 25 `completed_verified`, 27
`completed_verified_with_metadata_reconciliation`, and 69 `completed` with
cell-level audit details; verification depth varies by suite. The current ledger SHA-256 is
`8f7b9066c56cc508055216ff04a71fde8e7426441d76d8698510abcac21f6e4b`.

At 10:03 IST, Mtech had 32,497 MiB free VRAM and no training process. V100 had
6,614 MiB free; its 6,184 MiB worker estimate plus 1,024 MiB margin exceeds
that, and the GPU was occupied by three unrelated Python processes. No ledger
cell remained queued or eligible for launch. The unrelated V100 processes
were left untouched.

Mtech checkpoint cleanup deleted 163 owned model/checkpoint `.pt` files
(1,064,278,218 bytes); its post-cleanup scan found zero model-named `.pt`
files. The receipt is under
`/home/anurag24092/checkpoint_cleanup_receipts/20261124_mtech/`. The earlier
V100 cleanup removed 2,236 owned model checkpoints; one 1.32 GB checkpoint
file remains open by the existing `rclone` VFS cache process at
`/dev/shm/cafcl-vfs-cache5/vfs/kakarot/v100/.../checkpoint_latest.pt`. It was
left in place while the cache process holds it open; the remote result archive
was not modified through the mount.

The original campaign `status.tsv` now records eight cells completed, two 3-hidden cells failed, and the first four wave-2 cells completed; the final two wave-2 cells are still active and have not yet been recorded as terminal rows because the controller waits on them. Both replacement attempts, `3_off_epochs_1_retry1` and `3_off_epochs_2_retry1`, are marked completed in the retry `status.tsv`; each has a saved `.h5` result. The original campaign has eight `.h5` outputs; including the two retry outputs there are ten `.h5` artifacts so far. Thus the intended 12-cell grid is not yet fully terminal: two original 10-epoch wave-2 cells remain active, while the two failed original 3-hidden cells have completed replacement attempts.

Interpretation caveats remain: the original `3_off_epochs_1` and `3_off_epochs_2` attempts failed; use their retry outputs for those cells. The original `full_epochs_1` and `full_epochs_2` result files were saved despite CUDA OOM traces and require quality review. No process was stopped or restarted during this status check.


## Committee rationale and observation plan — 2026-11-23

This addendum states why each registered comparison is useful for the BTP
presentation/report and what evidence should be examined. These are planned
interpretations, not empirical findings. A single seed, execution failures,
OOMs, missing artifacts, or incomplete terminal audits limit claims; report
the observed evidence and these limitations rather than claiming a result the
runs do not support. Do not change a suite's configured factors to satisfy
this interpretation plan.

### Suite 1 — Disjoint task width (4 vs. 5 classes/task; 12 cells)

**Observe:** For each Task-IL/Class-IL and 1/2/3-offline condition, compare
per-sync test and validation performance at matched seen-class coverage;
final post-flush metrics; learning sync/task count; class coverage and
duplicate ownership; task-head counts for Task-IL; and completion/failure
evidence. Interpret offline levels as the configured participation stress,
not as a change to task width.

**Committee rationale:** This tests whether the granularity of each incoming
experience changes learning and retention when the total class inventory and
Replay/FedAvg recipe are held fixed, and whether any width effect depends on
label-space evaluation (Task-IL vs. Class-IL) or missed client communication.
It motivates the chosen task size and documents how task granularity interacts
with intermittent participation. It does not establish a universal optimal
width from one seed.

### Suite 2 — Intra-task synchronization frequency (2/5/10 syncs per task; 6 cells)

**Observe:** Per-sync learning curves and final metrics by IL mode and
frequency; number of communication rounds; epochs performed per segment;
task identity/data-index continuity across segments; replay insertion only at
the task boundary; all-client participation; and the terminal communication-
only flush. Audit wall-clock and communication counts if reliable records
exist; do not infer communication efficiency from accuracy alone.

**Committee rationale:** This isolates how often the server aggregates during
a fixed ten-epoch local task, exposing the tradeoff between longer local
optimization and more frequent coordination. It supports a discussion of
communication cadence in federated continual learning while preserving the
underlying task and data experience. It is a one-seed comparison and cannot
alone establish a best cadence.

### Goal campaign — Independent A–G ablations (29 cells)

The shared observation set is per-sync test/validation accuracy at comparable
class coverage, final post-flush metrics, schedule/coverage validation,
client participation and state transitions, and logs/checkpoints sufficient
to verify completion. Report Task-IL with its task-aware evaluation protocol;
do not compare its raw accuracy directly with Class-IL without explaining the
different prediction spaces.

- **A — Client population (2/3/10 clients; Task-IL and Class-IL):** Observe
  sync count, classes and tasks per client, the 17/17/16 tail for three
  clients, final accuracy, and whether the no-new-task client is represented
  correctly (no fabricated zero-weight upload). **Why for committee:** shows
  how federated population size and the resulting schedule length affect
  global learning, and whether the pattern differs between task-aware and
  shared-label evaluation. The finite 100-class partition imposes documented
  schedule tails.
- **B — Temporal class recurrence (delay 1/2/5/10/15):** Observe recovery
  after a previously seen class returns, performance on returning versus
  newly encountered classes when reports permit, matched 18-sync horizon,
  task multiset/per-class exposure counts, fresh task IDs/data, and displaced
  task records. **Why:** tests whether spacing repeated exposure changes
  forgetting and recovery while controlling the overall task inventory and
  exposure counts. It concerns label recurrence, not replay of raw images.
- **C — Client outage duration (1/2/3/5/10/15 syncs):** Observe the seeded
  hidden-client schedule, missed upload/downlink intervals, stale local model
  state, reconnection synchronization, per-sync progress, and common
  16-sync horizon/tail for long outages. **Why:** evaluates robustness to
  temporary client unavailability and the consequences of reconnecting with
  stale state. This is an asynchronous stale-client condition, not ordinary
  synchronous dropout.
- **D — Cross-client task recurrence (2/3/4 clients):** Observe which clients
  receive prior-step class sets, task identity, raw-sample disjointness,
  schedule horizons (16/24/46), and performance when known labels move across
  clients. **Why:** probes whether knowledge tied to previously seen labels
  transfers when those labels recur on different clients, separating label
  recurrence from raw-example leakage.
- **E — Within-step task sharing (2/3/4 clients):** Observe simultaneous
  client overlap, partitioned raw examples, class-set/task assignment,
  novelty tail behavior, and performance as the same task is shared by more
  clients. **Why:** tests whether concurrent exposure to the same task across
  clients changes aggregation and learning, while preventing duplicate raw
  training images from confounding the result.
- **F — Task width (3/5/10 classes):** Observe seen-class performance,
  number/size of tasks and syncs, exact once-only coverage, and explicitly
  report the width-3 final two-class remainder. **Why:** measures the effect
  of experience granularity on learning under the same Task-IL setup; the
  remainder is a finite-dataset constraint and must be visible in the
  interpretation.
- **G — Adjacent class overlap (2/3/4 shared labels at width 5):** Observe
  exact overlap for every adjacent pair, schedule seed 500 consistency,
  raw-image disjointness, non-adjacent tail intersections, and performance
  around task transitions. **Why:** tests whether a controlled amount of
  shared label knowledge between consecutive tasks changes transition
  difficulty and retention. Claims should distinguish the exact adjacent
  constraint from other tail intersections.

**Committee framing for A–G:** Together these are a structured sensitivity
analysis around the disjoint two-class Task-IL baseline: population,
recurrence timing, availability, cross-client transfer, simultaneous sharing,
task granularity, and adjacent overlap. They help explain which aspects of
the federated task schedule may influence continual learning. They do not
establish novelty or causal generality beyond the tested settings and single
seed.

### Suite 3 — FedQCL local epochs × client visibility (12 cells)

**Observe:** Across 0/1/2/3 randomly hidden clients and 1/2/10 local epochs
per participating sync, inspect server-round metrics, realized participating
clients, task-boundary memory/Q updates, final result files, and errors.
Because ten syncs per task make the epoch settings imply 10/20/100 local
epochs per task for a continuously participating client, state this coupled
training-budget consequence. Audit the OOM-affected full/1 and full/2 outputs;
use replacement attempts for the two failed 3-hidden cells and keep originals
as failed evidence. Keep the separate five-epoch reference block distinct.

**Committee rationale:** This studies the joint practical effects of local
optimization length and intermittent visibility in the FedQCL_Task method:
more local work may change optimization and resource cost, while hidden
clients miss both upload and broadcast and retain stale state. It informs
choices about client work and participation frequency. The matrix is
single-seed, and OOM-contaminated results cannot support clean comparisons
until audited or replaced by valid attempts.

### Suite 4 — V25 incremental curriculum participation (6 cells)

**Observe:** Compare full participation with one randomly offline client for
Task-IL disjoint, Task-IL joint adjacent-chain, and Class-IL disjoint.
Inspect per-sync curves at matched coverage, client selection, delayed
updates, final flush, final metrics, and schedule completion.

**Committee rationale:** This is the basic participation stress test for the
V25 Replay/FedAvg curriculum, asking whether one intermittently unavailable
client changes learning across distinct curriculum/evaluation settings. It
provides a controlled baseline for the higher-offline conditions in Suite 5.

### Suite 5 — V25 participation levels (8 cells)

**Observe:** For Task-IL/Class-IL × disjoint/joint curricula, compare two vs.
three randomly offline clients. Check actual online-client counts, per-sync
class coverage and curves, retained local state and resynchronization,
terminal flush, and verified final metrics. Interpret alongside Suite 4's
full and one-offline conditions, while keeping each suite's evidence and
provenance identifiable.

**Committee rationale:** Extends the participation stress test to stronger
communication loss and asks whether its impact depends on curriculum
structure or evaluation mode. Together with Suite 4, it can describe a
participation-level trend for these configurations; do not claim a smooth
dose-response unless the verified measurements support it.

### Suite 6 — V100 local epochs × participation × IL × curriculum (48 cells)

**Observe:** For each 1/2/5 epoch, 0/1/2/3 offline, Task-IL/Class-IL, and
disjoint/joint combination, verify config and schedule, per-cell terminal
state, final-only metrics, class coverage, actual participation, and resource
or failure evidence. Compare only matched cells differing in the intended
factor; preserve one output per unique cell and attach retries to that cell.
The suite is reserved to run last under the user's execution ordering.

**Committee rationale:** This factorial grid checks whether local training
length interacts with participation level, prediction protocol, and curriculum
structure in the V25 recipe. It can identify configuration-specific patterns
and guide which combinations deserve deeper study. Because evaluation is
final-only and each cell has one seed, it cannot by itself explain learning
dynamics over time or quantify across-seed uncertainty.

### General reporting rule

For every suite, separate the question/design, plan validation, execution
state, verified measurements, and interpretation. Report missing or failed
cells, retries, OOMs, stale snapshots, metric scope (test vs. validation and
Task-IL vs. Class-IL), and the one-seed limitation. Committee justification
should explain why the comparison is relevant to the project question; it
must not present an expected result as an observed result or claim novelty
without a separate literature review.
