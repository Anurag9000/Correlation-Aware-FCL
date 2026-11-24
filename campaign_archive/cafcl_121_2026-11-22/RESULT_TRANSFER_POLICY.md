# Mtech-to-Kakarot result transfer

The Kakarot archive has the complete A–G goal/specification, per-run plan
configs and schedules, preflight reports, compiler/runner sources, source
hashes, and timestamped queue snapshots. Completed or failed run evidence is
retrieved using `tools/sync_completed_runs_from_mtech.py`.

For each terminal run state, the sync script copies the whole run directory
except the large `model_stage_*.pt`, `model_final.pt`, and
`checkpoint_latest.pt` model files. It includes each run's design/config,
schedule, plan validation, environment and source checksum, worker log/status,
scenario, final summary, metrics, and per-sync JSON/CSV/Markdown/PNG reports.
It hashes each copied local file and records the server path and byte size of
the omitted checkpoint files in `results/transfer_manifest.json`. The source
files remain intact on Mtech; the script uses no deletion or mirror-delete
options.

This transfer boundary is based on measured storage: one preserved 10-learning-
sync run occupies about 16 GiB on Mtech, mostly checkpoint files. The A–G
plans total 460 learning syncs plus 29 communication-only flushes. A rough
checkpoint-only projection is around 600 GiB, while Kakarot had 549 GiB free
at the last disk check. Copying every model checkpoint would therefore risk
filling the local filesystem. The excluded checkpoints remain available at
their recorded Mtech paths; the per-sync reports and all experiment metadata
are copied locally.

The terminal sync script transfers only directories whose `worker_status.json`
is terminal (`completed`, `finished`, or `failed`). A terminal refresh replaces
the run's final manifest record and removes its earlier provisional snapshot.
While a worker is active, use
`tools/sync_active_runs_from_mtech.py`. It finds active A–G workers, copies
their current non-checkpoint files to the same run directory on Kakarot with
rsync checksum comparison, then verifies each copied file's size and SHA-256
against a fresh server inventory. The transfer manifest records the worker
PID, observation time, bounded log tail, staged paths and hashes, and server
checkpoint paths/sizes. This active snapshot is provisional; a later terminal
sync supersedes it. No mirror-delete option is used.

A_clients_10_task_il_seed0 is now terminal: all six sync reports,
metrics, metadata, and logs are present locally, with 39 non-checkpoint files
hash-verified; its checkpoint paths and sizes remain recorded on Mtech. The
completed A_clients_2_class_il_seed0 also received a terminal refresh, with
119 non-checkpoint files and 28 server-only checkpoint paths recorded. The
active A_clients_2_task_il_seed0 is staged during progress checks and will
receive a full non-checkpoint refresh at terminal state. Both transfers are
safe to repeat without deleting destination files.
