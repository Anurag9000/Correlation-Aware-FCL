# CAFCL campaign archive

The local campaign archive is stored in [`campaign_archive/cafcl_121_2026-11-22`](../campaign_archive/cafcl_121_2026-11-22/). `ARCHIVE_MANIFEST.json` lists each included file with its size and SHA-256.

The archive includes the authoritative 121-cell ledger and summary, plans, source and runner files, audit snapshots, locally held V100 outputs, and Mtech result directories. The Mtech transfer receipt records the ledger roots and attempt paths found on Mtech, absent paths, worker status files, source hashes, and transfer exclusions.

The latest ledger snapshot has 121 terminal execution states: 41 `completed_verified`, 24 `completed_verified_with_metadata_reconciliation`, and 56 `needs_audit`. Training completion and result-audit completion remain separate ledger states.

The expanded Mtech transfer found 131 existing locations among 173 ledger-linked paths. It copied 8,436 result, source, and run-evidence files totaling 693,475,438 bytes after excluding 41,602 Python bytecode cache files. Every retained file matched its Mtech SHA-256 after transfer. The source status files include 149 `completed`, 31 `running`, 4 `failed`, and 2 `interrupted` snapshots; the `running` values are preserved as historical records. A contemporaneous Mtech check found no CAFCL campaign process and 4 MiB VRAM in use.

Forty-two ledger-linked paths were absent from Mtech. Many belong to V100-assigned attempts. The local archive retains the V100 artifacts that were already available on Kakarot; direct V100 SSH still rejects authentication, so these missing paths are listed in the Mtech transfer receipt rather than represented as transferred.

Model/checkpoint `.pt` files were excluded except `scenario.pt` run inputs, along with `.tmp`, `.lock`, `.pid`, Python bytecode caches, the vendored rclone executable, transport-test files, and the 900 MiB write-through probe. Source files on Mtech and V100 were not deleted by synchronization.
