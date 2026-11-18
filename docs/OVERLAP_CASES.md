# CIFAR-100 overlap design space — same/different clients, same/different intervals

Overlap is in **class identity** unless raw sample duplication is intentionally enabled. Each task still contains >=2 *distinct* global labels. Repeated labels refer to the same classifier output row, not a task-specific head. Two clients may hold different training images of the same category without sharing any physical image.

| `stream_overlap_kind` | Example | Scientific contrast |
|---|---|---|
| `none` | Client A `(0,1)` then `(20,21)`; Client B `(2,3)` then `(22,23)` | One appearance of every label across all client tasks; no overlap |
| `intra_same_round` | Client A `(0,1)` then `(0,21)` *before the same aggregation* | Same label recurs in the same client within one communication interval |
| `intra_cross_round` | Client A trains `(0,1)` in interval 0, then `(0,41)` in interval 1 | Same label recurs in the same client after a server-communication boundary |
| `inter_same_round` | Client A `(0,1)` and Client B `(0,3)` in interval 0 | Same label is learned at different clients before the same server update |
| `inter_cross_round` | Client B learns `(2,3)` in interval 0; Client A receives `(2,41)` in interval 1 | Shared label appears at different clients across different updates |
| `combined` | Deterministic mixture of the four above geometries | Multiple overlap sources in one experiment |

`stream_overlap_fraction` is the probability of replacing one of a task's *novel* classes with an already-seen class from the specified relationship, so values `.25`, `.5`, `1.0` are genuinely different data-stream conditions. A replacement **does not consume** the next unused class; the schedule extends to later intervals until all 100 class IDs have been introduced. When a final task would have too few novel labels, previously encountered labels may fill it while keeping all classes distinct within the task; such reuse is recorded.

For two clients seeing one label, the scheduler partitions that class's **train indices disjointly among its appearances** by default (`sample_overlap_fraction=0`). To test deliberate raw-example duplication, set `sample_overlap_fraction>0`; duplicate training index IDs are explicitly recorded and measured. The untouched official test data are never allocated to clients.

Because overlap reduces the rate of introducing fresh labels, it can increase the number of actual training communication intervals. The stream duration comes from the generated schedule; **it is not blindly forced to 20 or 200 nominal rounds**. Extra recurrent rounds, if requested, are explicitly declared and separately tracked as repeated old-class data, not fresh new classes.

Every pattern is crossed with all eleven supported local learners and nine supported server update methods in the active registry. The suite varies participation and offline duration independently, but is still a **finite declared factorial grid**, not every possible joint combination of every numeric hyperparameter.
