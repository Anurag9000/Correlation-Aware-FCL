# CAFCL 121-cell campaign archive

The complete local campaign archive is stored in [`campaign_archive/cafcl_121_2026-11-22`](../campaign_archive/cafcl_121_2026-11-22/). Its `ARCHIVE_MANIFEST.json` lists each included file with its size and SHA-256.

The archive contains the authoritative 121-cell ledger and summary, run plans, source and runner files, audit snapshots, transferred terminal outputs, V100 result artifacts already held on Kakarot, and the fresh Mtech terminal-output transfer. The Mtech transfer manifest records 15 terminal runs and 4 provisional snapshots; all 1,193 file hashes recorded for terminal runs were checked locally with no mismatch.

The ledger snapshot has 121 terminal execution states. At the latest audit refresh, 41 cells were `completed_verified`, 24 were `completed_verified_with_metadata_reconciliation`, and 56 remained `needs_audit`. Execution completion and audit completion are separate; the archive preserves this distinction.

Model checkpoints and weight files (`.pt`, except `scenario.pt` inputs), temporary files, lock/PID files, the transport test directory, the vendored rclone executable, and the 900 MiB transport write-through probe are excluded. The source campaign archive on this machine is retained unchanged. No remote files were deleted as part of synchronization.
