# Joint spatiotemporal FCL implementation status

This is an incremental extension, not a replacement for historical V25/V27/V28.
The atomic compiler provides exact replication groups, common/private partial
overlap, explicit arbitrary sequences, stable task and head identities,
fixed unique-data partitioning, interval grouping and controlled upload cohorts.
The 48-condition foundational registry is a design registry, not evidence of
completed training. All plan records are built from the TRAIN split targets.
No GPU runs have been performed by this change.

Current limitations: the planner has no automatic raw-sample reuse/domain
allocator, arbitrary topology constraint solver or automatic A–R family
expander. Explicit events can encode many such curriculum geometries, but
their requested treatment invariants require additional family-specific
validators. Class-IL model restoration, causal runtime provenance, lifecycle
evaluation, runtime integration and full campaign orchestration require
separate implementation and validation. Do not claim these are implemented
from the planner alone.

Run planner tests with:
    PYTHONPATH=src python -m unittest discover -s tests -p test_joint_curriculum.py

The core plan is deliberately independent of torch and GPU resources.
