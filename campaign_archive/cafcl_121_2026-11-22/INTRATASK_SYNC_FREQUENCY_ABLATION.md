# Intra-task server-sync frequency ablation

**Suite ID:** Suite 2 in `SUITE_LEDGER.md`  
**Created:** 2026-11-22  
**Dataset:** CIFAR-100  
**Seed:** 0, one seed per run  
**Implementation host:** MtechDEServer (`192.168.3.30`)  
**Documentation host:** Kakarot, `/home/anurag-basistha`  
**Implementation root:** `/home/anurag24092/Projects/Correlation-Aware-FCL-intratask-sync-ablation-20261122`

## Question and scope

Measure the effect of making server synchronization happen multiple times
while a client is completing one task. This is a separate suite from A–G and
from the task-size and local-epoch suites. The only model protocols are the
two original disjoint, full-participation configurations: Task-IL and
Class-IL. Do not combine this factor with offline clients, overlapping
classes, a different client count, a different task width, or a different
learning algorithm.

The six new cells are:

| Model protocol | Server syncs during each task | Runs |
|---|---:|---:|
| Disjoint Task-IL | 2, 5, 10 | 3 |
| Disjoint Class-IL | 2, 5, 10 | 3 |
| **Total** |  | **6** |

The original one-sync-per-task results are the reference and are not rerun as
additional suite cells.

## Frozen experiment contract

The control conditions use the original five-client, CIFAR-100 disjoint
curriculum: each client receives ten unique two-class tasks, one task per
client per task index, and all 100 labels are covered exactly once. All five
clients participate at every server sync. A client completes the task's
specified local epoch budget across its sync segments before advancing to its
next task.

All other recorded baseline fields stay fixed: ResNet-18 with the CIFAR stem;
Experience Replay; 10 local epochs per task; sample-weighted FedAvg; SGD with
learning rate 0.03, momentum 0.9, weight decay 0.0005, and persistent client
optimizer state; crop/flip augmentation; private 50-example-per-class
reservoir; sampled 1:1 current/replay mixing; and seed 0. Task-IL retains its
task-specific heads and task-aware evaluation contract. Class-IL uses one
shared 100-way head and ordinary shared-head evaluation.

## What changes at each level

The ten local epochs for each task are partitioned evenly among that task's
sync segments:

| Segments per task | Local epochs per segment | Learning syncs over 10 tasks/client | Communication-only flush | Total sync events |
|---:|---:|---:|---:|---:|
| 2 | 5 | 20 | 1 | 21 |
| 5 | 2 | 50 | 1 | 51 |
| 10 | 1 | 100 | 1 | 101 |

For each segment, clients train the same in-progress task for its allocated
epochs; then all five upload, the server applies the existing FedAvg path,
and the resulting global state is broadcast before the next segment. A
segment boundary does not create a new task, task ID, or replay-memory
insertion. The replay archive is updated only at the original task boundary.
This preserves ten task experiences per client while changing only the
communication frequency during each experience.

The final flush is recorded separately. It communicates pending state only;
it performs no forward/backward pass, optimizer update, or additional epoch.

## Implementation and output layout

The isolated Mtech implementation is recorded at:

- `INTRATASK_SYNC_ABLATION.md`
- `scripts/intratask_sync_campaign.py`
- `queue/` for the six configs and campaign/controller state
- `results/<task_il|class_il>/syncs_per_task_<2|5|10>/` for run outputs

The implementation adds the per-task segment count to the isolated runner
and preserves task identity, local epoch budget, and replay insertion across
segments. The campaign controller runs one job at a time and has an admission
threshold of 24,576 MiB free VRAM. It is queued independently of the A–G
controller. Existing server jobs are not to be terminated or altered to
create capacity. Each run must keep a unique output directory and save its
resolved config, task/sync schedule, source and environment provenance,
worker status/log, per-sync evaluation, and final report.

## Validation gates

Before accepting a plan or result, verify each of the six cells against these
invariants:

1. Exactly five clients; full participation at every learning sync.
2. Exactly ten two-class disjoint task instances per client; all 100 CIFAR-100
   classes occur once in the novel stream.
3. Exactly 2, 5, or 10 training segments per task, with 5, 2, or 1 local
   epochs respectively; the total remains ten epochs per task.
4. Learning-sync counts are 20, 50, or 100; the separately recorded flush
   brings total communication events to 21, 51, or 101.
5. Segment boundaries retain the same current task and do not insert that
   task into replay more than once.
6. Every learning event uses the same FedAvg and broadcast implementation;
   the flush has no training updates.
7. Task-IL and Class-IL retain their own head and evaluation contracts.
8. IDs, configs, schedules, and output paths are unique; all use seed 0.

Report evaluation using the existing per-learning-sync seen-class protocol
and final post-flush report. Preserve per-class accuracy and task/head
exposure so early and late sync frequency can be compared. A plan/preflight
is not an accuracy result; the six cells remain unvalidated empirically until
each run has a terminal state and its result files have been audited.

## Evidence and current verification boundary

The six plans passed preflight with the expected task counts, segment counts,
epoch allocations, total learning syncs, continuation flags, and unchanged
task contents. I inspected the compiler, both Task-IL and Class-IL schedule
adapters, and all six server plan-validation JSONs over SSH. Each plan says
`plan_only`; this verifies implementation and schedule arithmetic, not
training or accuracy.

At 2026-11-22 06:18 IST, direct SSH to `192.168.3.30` verified the isolated
server root and read `queue/campaign_status.json`. The controller reported
`waiting_for_capacity`, position 1 of 6, with the Task-IL / 2-sync run next.
At 06:58 IST, direct SSH re-verified the isolated runner and all six plan
summaries. They record the expected `learning_syncs` (20/50/100),
`total_syncs_including_flush` (21/51/101), five uploading clients per learning
sync, 50 task instances, 10 epochs/task, and preserved task identity, contents,
and replay boundary. Each plan remains explicitly `plan_only`.

At 07:02 IST, direct SSH also read this suite's live queue file and process
table: controller PID 29969 is alive, status `waiting_for_capacity`, position
1/6, with Task-IL / 2-sync next. Its last queue update was 06:59:52 IST and
recorded 18,531 MiB free against 24,576 MiB required; FedQCL and the task-size
campaign were listed as external jobs. Thus none of the six workers is claimed
as started or completed. The suite is queued and ready; do not duplicate its
six runs. No server process was stopped or modified during this documentation
pass.

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
