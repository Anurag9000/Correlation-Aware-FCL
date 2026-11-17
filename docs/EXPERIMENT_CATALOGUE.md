# Complete 152-block experiment catalogue and current executable coverage

Blocks are research questions; jobs are concrete batch-512 configurations; neither is evidence of a completed training result. `partial implementation` means genuine runnable methodology with important missing variants/controls. Multi-round causal forks are now scheduled as dependent jobs that replay identical full client/server state from a common checkpoint.

| ID | Scientific experiment | Status | CIFAR-10 jobs | CIFAR-100 jobs |
|---|---|---|---:|---:|
| A01 | Participation rate | implemented | 19 | 19 |
| A02 | Fixed versus resampled participants | implemented | 0 | 0 |
| A03 | Equal communication rounds | not implemented | 0 | 0 |
| A04 | Matched total client updates | not implemented | 0 | 0 |
| A05 | Old/new class-holder participation | partial implementation | 3 | 3 |
| A06 | Stage-dependent participation schedule | not implemented | 0 | 0 |
| A07 | Class diversity at equal participation | not implemented | 0 | 0 |
| A08 | Client redundancy with matched class coverage | partial implementation | 0 | 0 |
| A09 | Short frequent versus long rare participation | not implemented | 0 | 0 |
| A10 | Monotonically changing participation | implemented | 8 | 8 |
| A11 | Equal lifetime frequency, different timing | partial implementation | 3 | 3 |
| A12 | Equal-plasticity forgetting comparison | not implemented | 0 | 0 |
| B01 | Related vs unrelated new classes | partial implementation | 2 | 2 |
| B02 | Select/exclude related-class client | partial implementation | 21 | 21 |
| B03 | Relationship strength | not implemented | 0 | 0 |
| B04 | Gradient alignment classes | not implemented | 0 | 0 |
| B05 | One-to-one related classes | not implemented | 0 | 0 |
| B06 | Many-to-one relations | not implemented | 0 | 0 |
| B07 | Related-class training dose | not implemented | 0 | 0 |
| B08 | Repeated related-client selection | not implemented | 0 | 0 |
| B09 | Mixed related/unrelated participants | not implemented | 0 | 0 |
| B10 | Feature similarity intervention | not implemented | 0 | 0 |
| B11 | Local GEM x relationship | partial implementation | 10 | 10 |
| B12 | Per-old-class spillover | not implemented | 0 | 0 |
| C01 | Overlap magnitude | partial implementation | 4 | 4 |
| C02 | Joint versus separate participation | partial implementation | 15 | 15 |
| C03 | Shared-class dominance | not implemented | 0 | 0 |
| C04 | Same labels with disjoint examples | not implemented | 0 | 0 |
| C05 | Exact sample overlap | not implemented | 0 | 0 |
| C06 | Shared class across shifted domains | not implemented | 0 | 0 |
| C07 | Gradient interactions of overlapping clients | not implemented | 0 | 0 |
| C08 | Redundancy versus complementary coverage | partial implementation | 3 | 3 |
| C09 | Overlap x local GEM | partial implementation | 9 | 9 |
| C10 | Longitudinal overlap selection | not implemented | 0 | 0 |
| D01 | Same-client recurrence | partial implementation | 1 | 1 |
| D02 | Cross-client recurrence | partial implementation | 1 | 1 |
| D03 | Related but new-label recurrence | not implemented | 0 | 0 |
| D04 | Shifted-domain recurrence | not implemented | 0 | 0 |
| D05 | Recurrence delay | partial implementation | 1 | 3 |
| D06 | Recurrence timing | not implemented | 0 | 0 |
| D07 | Select/exclude recurring client | partial implementation | 15 | 15 |
| D08 | Periodic recurrence | not implemented | 0 | 0 |
| D09 | Mixed old/new recurrence | not implemented | 0 | 0 |
| D10 | Inactive-client recurrence | not implemented | 0 | 0 |
| D11 | Delayed recurrent-client participation | not implemented | 0 | 0 |
| D12 | Recurrence under GEM | partial implementation | 6 | 6 |
| E01 | Relatedness x overlap | partial implementation | 3 | 3 |
| E02 | Relatedness x recurrence | partial implementation | 3 | 3 |
| E03 | Overlap x recurrence | not implemented | 0 | 0 |
| E04 | Triple relationship interaction | partial implementation | 3 | 3 |
| E05 | Mixed helpful and harmful client updates | not implemented | 0 | 0 |
| E06 | Asynchronous old/new class relationships | partial implementation | 3 | 3 |
| E07 | Rare-class recurrence | not implemented | 0 | 0 |
| E08 | Persistent vs temporary correlations | not implemented | 0 | 0 |
| E09 | Relatedness strength x participation | partial implementation | 6 | 6 |
| E10 | Relatedness x aggregation weights | partial implementation | 4 | 4 |
| F01 | Uniform random | implemented | 2 | 2 |
| F02 | Fixed subset | implemented | 2 | 2 |
| F03 | Round-robin | implemented | 2 | 2 |
| F04 | Least recently selected | implemented | 2 | 2 |
| F05 | Class coverage maximizing | implemented | 2 | 2 |
| F06 | Old-class coverage | implemented | 2 | 2 |
| F07 | New-class coverage | implemented | 2 | 2 |
| F08 | Balanced old and new coverage | partial implementation | 2 | 2 |
| F09 | Maximum label diversity | partial implementation | 2 | 2 |
| F10 | Maximum label overlap | partial implementation | 2 | 2 |
| F11 | Maximum feature similarity | not implemented | 0 | 0 |
| F12 | Minimum feature similarity | not implemented | 0 | 0 |
| F13 | Gradient alignment selection | not implemented | 0 | 0 |
| F14 | Gradient conflict selection | not implemented | 0 | 0 |
| F15 | Forgetting-aware selection | not implemented | 0 | 0 |
| F16 | Recurrence-aware selection | partial implementation | 0 | 0 |
| F17 | Fairness-constrained selection | partial implementation | 2 | 2 |
| F18 | Randomized correlation-aware selection | partial implementation | 2 | 2 |
| F19 | History-aware adaptive selection | not implemented | 0 | 0 |
| F20 | Oracle exhaustive selection | not implemented | 0 | 0 |
| G01 | Local SGD | implemented | 3 | 3 |
| G02 | Local GEM | implemented | 3 | 3 |
| G03 | Local A-GEM | implemented | 3 | 3 |
| G04 | Experience replay | implemented | 3 | 3 |
| G05 | ER-ACE | partial implementation | 2 | 2 |
| G06 | DER and DER++ | partial implementation | 4 | 4 |
| G07 | EWC | partial implementation | 3 | 3 |
| G08 | Synaptic intelligence | not implemented | 0 | 0 |
| G09 | MAS | partial implementation | 2 | 2 |
| G10 | Local distillation | partial implementation | 2 | 2 |
| G11 | Replay vs no replay | partial implementation | 0 | 0 |
| G12 | Memory capacity | partial implementation | 9 | 9 |
| G13 | Memory selection algorithms | partial implementation | 2 | 2 |
| G14 | GEM per-group constraints | not implemented | 0 | 0 |
| G15 | Proximal regularization plus GEM | partial implementation | 6 | 6 |
| G16 | Heterogeneous local algorithms | partial implementation | 2 | 2 |
| H01 | Active-only synchronization | implemented | 2 | 2 |
| H02 | All-client synchronization | implemented | 2 | 2 |
| H03 | Sync when rejoining | partial implementation | 0 | 0 |
| H04 | Optimizer state after broadcast | not implemented | 0 | 0 |
| H05 | Private memory persistence | partial implementation | 0 | 0 |
| H06 | Synchronization frequency | partial implementation | 3 | 3 |
| H07 | Long inactive gap | not implemented | 0 | 0 |
| H08 | Local personalization | not implemented | 0 | 0 |
| I01 | Sample-weighted FedAvg | implemented | 0 | 0 |
| I02 | Uniform aggregation | implemented | 0 | 0 |
| I03 | Class-balanced aggregation | not implemented | 0 | 0 |
| I04 | Correlation-aware aggregation | not implemented | 0 | 0 |
| I05 | Old-retention aggregation | not implemented | 0 | 0 |
| I06 | Server momentum/adaptive optimizer | partial implementation | 8 | 8 |
| I07 | Gradient conflict correction | not implemented | 0 | 0 |
| I08 | Participation-probability correction | not implemented | 0 | 0 |
| J01 | IID vs non-IID | implemented | 2 | 2 |
| J02 | Label distribution skew | partial implementation | 2 | 2 |
| J03 | Client sample-count skew | not implemented | 0 | 0 |
| J04 | Feature distribution skew | not implemented | 0 | 0 |
| J05 | Class imbalance | not implemented | 0 | 0 |
| J06 | Dependent versus independent client streams | not implemented | 0 | 0 |
| J07 | Unequal arrival rates | not implemented | 0 | 0 |
| J08 | Inactive client data retention | partial implementation | 6 | 6 |
| J09 | Recurrence timing distribution | not implemented | 0 | 0 |
| J10 | Spurious correlations | not implemented | 0 | 0 |
| J11 | Label noise | not implemented | 0 | 0 |
| J12 | Changing correlation structure | not implemented | 0 | 0 |
| K01 | Class-incremental learning | implemented | 0 | 0 |
| K02 | Task-incremental learning | not implemented | 0 | 0 |
| K03 | Domain-incremental learning | not implemented | 0 | 0 |
| K04 | Task-free learning | not implemented | 0 | 0 |
| K05 | Fixed versus expanding classifier | not implemented | 0 | 0 |
| K06 | Known versus hidden experience boundaries | not implemented | 0 | 0 |
| K07 | Asynchronous versus synchronized transitions | not implemented | 0 | 0 |
| K08 | Closed-world versus open-world class arrival | not implemented | 0 | 0 |
| L01 | Leave-one-client-out aggregation | implemented | 6 | 6 |
| L02 | Add-one-client aggregation | partial implementation | 18 | 18 |
| L03 | Pairwise client interactions | not implemented | 0 | 0 |
| L04 | Exhaustive participating subset enumeration | not implemented | 0 | 0 |
| L05 | Pre/post local historical loss | not implemented | 0 | 0 |
| L06 | Pre/post aggregation loss | not implemented | 0 | 0 |
| L07 | Gradient interference analysis | not implemented | 0 | 0 |
| L08 | Representation drift | not implemented | 0 | 0 |
| L09 | Correlation intervention | not implemented | 0 | 0 |
| L10 | Historical contribution attribution | not implemented | 0 | 0 |
| M01 | Dataset generalization | partial implementation | 0 | 0 |
| M02 | Model architecture | implemented | 3 | 3 |
| M03 | Client population size | partial implementation | 3 | 3 |
| M04 | Local optimization intensity | partial implementation | 3 | 3 |
| M05 | Randomness robustness | partial implementation | 3 | 3 |
| M06 | Communication efficiency | not implemented | 0 | 0 |
| M07 | Computational efficiency | not implemented | 0 | 0 |
| M08 | Privacy/resource restrictions | not implemented | 0 | 0 |
| N01 | Correlation-aware participation | partial implementation | 1 | 1 |
| N02 | Class-participation balancing | partial implementation | 2 | 2 |
| N03 | Recurrence-aware scheduling | partial implementation | 1 | 1 |
| N04 | GEM-informed participation | not implemented | 0 | 0 |
| N05 | Adaptive participation rate | not implemented | 0 | 0 |
| N06 | Joint selection and aggregation | not implemented | 0 | 0 |

**Totals:** 152 planned blocks; 73 with implementation coverage (51 partial); 63 blocks with independent jobs; 276 CIFAR-10 jobs and 278 CIFAR-100 jobs.

**CAUSAL JOB DEPENDENCIES:** L02 creates a baseline with saved round checkpoints, then runs fixed-budget include and exclude counterfactual branches. `fork_source_job` and `fork_checkpoint_round` are recorded in the job configuration. These are full future-trajectories starting from identical server/client/memory/RNG state, changing one designated client-selection round. This does not implement all possible causal analyses.

Missing blocks are explicitly `not implemented`. No official CIFAR runs, publication-level validation, multi-seed confidence intervals, or external-method reproductions were executed here.
