# 152-block research experiment registry and executable coverage

Status is about implementation coverage, **not completed CIFAR training**. A partial implementation may be an oracle, a proxy, or a limited comparison. No official CIFAR suite was executed as part of this commit.

| ID | Research question | Status | CIFAR-10 jobs | CIFAR-100 jobs |
|---|---|---|---:|---:|
| A01 | Participation rate | implemented | 19 | 19 |
| A02 | Fixed versus resampled participants | implemented | 2 | 2 |
| A03 | Equal communication rounds | partial_implementation | 2 | 2 |
| A04 | Matched total client updates | partial_implementation | 2 | 2 |
| A05 | Old/new class-holder participation | partial_implementation | 3 | 3 |
| A06 | Stage-dependent participation schedule | partial_implementation | 2 | 2 |
| A07 | Class diversity at equal participation | partial_implementation | 2 | 2 |
| A08 | Client redundancy with matched class coverage | partial_implementation | 2 | 2 |
| A09 | Short frequent versus long rare participation | partial_implementation | 2 | 2 |
| A10 | Monotonically changing participation | implemented | 8 | 8 |
| A11 | Equal lifetime frequency, different timing | partial_implementation | 3 | 3 |
| A12 | Equal-plasticity forgetting comparison | not_implemented | 0 | 0 |
| B01 | Related vs unrelated new classes | partial_implementation | 2 | 2 |
| B02 | Select/exclude related-class client | partial_implementation | 21 | 21 |
| B03 | Relationship strength | partial_implementation | 2 | 2 |
| B04 | Gradient alignment classes | partial_implementation | 2 | 2 |
| B05 | One-to-one related classes | partial_implementation | 2 | 2 |
| B06 | Many-to-one relations | not_implemented | 0 | 0 |
| B07 | Related-class training dose | partial_implementation | 3 | 3 |
| B08 | Repeated related-client selection | partial_implementation | 2 | 2 |
| B09 | Mixed related/unrelated participants | partial_implementation | 2 | 2 |
| B10 | Feature similarity intervention | not_implemented | 0 | 0 |
| B11 | Local GEM x relationship | partial_implementation | 10 | 10 |
| B12 | Per-old-class spillover | partial_implementation | 2 | 2 |
| C01 | Overlap magnitude | partial_implementation | 4 | 4 |
| C02 | Joint versus separate participation | partial_implementation | 15 | 15 |
| C03 | Shared-class dominance | partial_implementation | 2 | 2 |
| C04 | Same labels with disjoint examples | partial_implementation | 1 | 1 |
| C05 | Exact sample overlap | partial_implementation | 3 | 3 |
| C06 | Shared class across shifted domains | partial_implementation | 1 | 1 |
| C07 | Gradient interactions of overlapping clients | partial_implementation | 1 | 1 |
| C08 | Redundancy versus complementary coverage | partial_implementation | 3 | 3 |
| C09 | Overlap x local GEM | partial_implementation | 9 | 9 |
| C10 | Longitudinal overlap selection | partial_implementation | 2 | 2 |
| D01 | Same-client recurrence | partial_implementation | 1 | 1 |
| D02 | Cross-client recurrence | partial_implementation | 1 | 1 |
| D03 | Related but new-label recurrence | partial_implementation | 2 | 2 |
| D04 | Shifted-domain recurrence | partial_implementation | 2 | 2 |
| D05 | Recurrence delay | partial_implementation | 1 | 3 |
| D06 | Recurrence timing | partial_implementation | 3 | 3 |
| D07 | Select/exclude recurring client | partial_implementation | 15 | 15 |
| D08 | Periodic recurrence | partial_implementation | 2 | 2 |
| D09 | Mixed old/new recurrence | partial_implementation | 2 | 2 |
| D10 | Inactive-client recurrence | partial_implementation | 2 | 2 |
| D11 | Delayed recurrent-client participation | partial_implementation | 2 | 2 |
| D12 | Recurrence under GEM | partial_implementation | 6 | 6 |
| E01 | Relatedness x overlap | partial_implementation | 3 | 3 |
| E02 | Relatedness x recurrence | partial_implementation | 3 | 3 |
| E03 | Overlap x recurrence | partial_implementation | 2 | 2 |
| E04 | Triple relationship interaction | partial_implementation | 3 | 3 |
| E05 | Mixed helpful and harmful client updates | partial_implementation | 2 | 2 |
| E06 | Asynchronous old/new class relationships | partial_implementation | 3 | 3 |
| E07 | Rare-class recurrence | partial_implementation | 2 | 2 |
| E08 | Persistent vs temporary correlations | partial_implementation | 2 | 2 |
| E09 | Relatedness strength x participation | partial_implementation | 6 | 6 |
| E10 | Relatedness x aggregation weights | partial_implementation | 4 | 4 |
| F01 | Uniform random | implemented | 2 | 2 |
| F02 | Fixed subset | implemented | 2 | 2 |
| F03 | Round-robin | implemented | 2 | 2 |
| F04 | Least recently selected | implemented | 2 | 2 |
| F05 | Class coverage maximizing | implemented | 2 | 2 |
| F06 | Old-class coverage | implemented | 2 | 2 |
| F07 | New-class coverage | implemented | 2 | 2 |
| F08 | Balanced old and new coverage | partial_implementation | 2 | 2 |
| F09 | Maximum label diversity | partial_implementation | 2 | 2 |
| F10 | Maximum label overlap | partial_implementation | 2 | 2 |
| F11 | Maximum feature similarity | partial_implementation | 1 | 1 |
| F12 | Minimum feature similarity | partial_implementation | 1 | 1 |
| F13 | Gradient alignment selection | partial_implementation | 1 | 1 |
| F14 | Gradient conflict selection | partial_implementation | 1 | 1 |
| F15 | Forgetting-aware selection | partial_implementation | 1 | 1 |
| F16 | Recurrence-aware selection | partial_implementation | 2 | 2 |
| F17 | Fairness-constrained selection | partial_implementation | 2 | 2 |
| F18 | Randomized correlation-aware selection | partial_implementation | 2 | 2 |
| F19 | History-aware adaptive selection | partial_implementation | 1 | 1 |
| F20 | Oracle exhaustive selection | partial_implementation | 1 | 1 |
| G01 | Local SGD | implemented | 3 | 3 |
| G02 | Local GEM | implemented | 3 | 3 |
| G03 | Local A-GEM | implemented | 3 | 3 |
| G04 | Experience replay | implemented | 3 | 3 |
| G05 | ER-ACE | partial_implementation | 2 | 2 |
| G06 | DER and DER++ | partial_implementation | 4 | 4 |
| G07 | EWC | partial_implementation | 3 | 3 |
| G08 | Synaptic intelligence | partial_implementation | 2 | 2 |
| G09 | MAS | partial_implementation | 2 | 2 |
| G10 | Local distillation | partial_implementation | 2 | 2 |
| G11 | Replay vs no replay | partial_implementation | 2 | 2 |
| G12 | Memory capacity | partial_implementation | 9 | 9 |
| G13 | Memory selection algorithms | partial_implementation | 2 | 2 |
| G14 | GEM per-group constraints | partial_implementation | 3 | 3 |
| G15 | Proximal regularization plus GEM | partial_implementation | 6 | 6 |
| G16 | Heterogeneous local algorithms | partial_implementation | 2 | 2 |
| H01 | Active-only synchronization | implemented | 2 | 2 |
| H02 | All-client synchronization | implemented | 2 | 2 |
| H03 | Sync when rejoining | partial_implementation | 2 | 2 |
| H04 | Optimizer state after broadcast | partial_implementation | 2 | 2 |
| H05 | Private memory persistence | partial_implementation | 2 | 2 |
| H06 | Synchronization frequency | partial_implementation | 3 | 3 |
| H07 | Long inactive gap | partial_implementation | 2 | 2 |
| H08 | Local personalization | not_implemented | 0 | 0 |
| I01 | Sample-weighted FedAvg | implemented | 1 | 1 |
| I02 | Uniform aggregation | implemented | 1 | 1 |
| I03 | Class-balanced aggregation | partial_implementation | 2 | 2 |
| I04 | Correlation-aware aggregation | partial_implementation | 2 | 2 |
| I05 | Old-retention aggregation | partial_implementation | 2 | 2 |
| I06 | Server momentum/adaptive optimizer | partial_implementation | 8 | 8 |
| I07 | Gradient conflict correction | partial_implementation | 2 | 2 |
| I08 | Participation-probability correction | partial_implementation | 2 | 2 |
| J01 | IID vs non-IID | implemented | 2 | 2 |
| J02 | Label distribution skew | partial_implementation | 2 | 2 |
| J03 | Client sample-count skew | partial_implementation | 2 | 2 |
| J04 | Feature distribution skew | partial_implementation | 2 | 2 |
| J05 | Class imbalance | partial_implementation | 2 | 2 |
| J06 | Dependent versus independent client streams | partial_implementation | 2 | 2 |
| J07 | Unequal arrival rates | partial_implementation | 2 | 2 |
| J08 | Inactive client data retention | partial_implementation | 6 | 6 |
| J09 | Recurrence timing distribution | partial_implementation | 3 | 3 |
| J10 | Spurious correlations | partial_implementation | 2 | 2 |
| J11 | Label noise | partial_implementation | 2 | 2 |
| J12 | Changing correlation structure | partial_implementation | 2 | 2 |
| K01 | Class-incremental learning | implemented | 1 | 1 |
| K02 | Task-incremental learning | partial_implementation | 2 | 2 |
| K03 | Domain-incremental learning | partial_implementation | 2 | 2 |
| K04 | Task-free learning | partial_implementation | 2 | 2 |
| K05 | Fixed versus expanding classifier | not_implemented | 0 | 0 |
| K06 | Known versus hidden experience boundaries | partial_implementation | 2 | 2 |
| K07 | Asynchronous versus synchronized transitions | partial_implementation | 2 | 2 |
| K08 | Closed-world versus open-world class arrival | not_implemented | 0 | 0 |
| L01 | Leave-one-client-out aggregation | implemented | 6 | 6 |
| L02 | Add-one-client aggregation | partial_implementation | 18 | 18 |
| L03 | Pairwise client interactions | partial_implementation | 1 | 1 |
| L04 | Exhaustive participating subset enumeration | partial_implementation | 1 | 1 |
| L05 | Pre/post local historical loss | partial_implementation | 1 | 1 |
| L06 | Pre/post aggregation loss | partial_implementation | 1 | 1 |
| L07 | Gradient interference analysis | partial_implementation | 1 | 1 |
| L08 | Representation drift | partial_implementation | 2 | 2 |
| L09 | Correlation intervention | partial_implementation | 2 | 2 |
| L10 | Historical contribution attribution | partial_implementation | 1 | 1 |
| M01 | Dataset generalization | partial_implementation | 1 | 1 |
| M02 | Model architecture | implemented | 3 | 3 |
| M03 | Client population size | partial_implementation | 3 | 3 |
| M04 | Local optimization intensity | partial_implementation | 3 | 3 |
| M05 | Randomness robustness | partial_implementation | 3 | 3 |
| M06 | Communication efficiency | partial_implementation | 2 | 2 |
| M07 | Computational efficiency | partial_implementation | 2 | 2 |
| M08 | Privacy/resource restrictions | not_implemented | 0 | 0 |
| N01 | Correlation-aware participation | partial_implementation | 1 | 1 |
| N02 | Class-participation balancing | partial_implementation | 2 | 2 |
| N03 | Recurrence-aware scheduling | partial_implementation | 1 | 1 |
| N04 | GEM-informed participation | partial_implementation | 2 | 2 |
| N05 | Adaptive participation rate | partial_implementation | 2 | 2 |
| N06 | Joint selection and aggregation | not_implemented | 0 | 0 |

**Totals:** 152 blocks; 144 with implementation coverage, 122 explicitly partial, 8 without implementation coverage; 423 CIFAR-10 jobs and 425 CIFAR-100 jobs.

**Restrictions:** Task-free controls remain class-aware and boundary-free; selection policies using simulator class/gradient information are privileged oracles; the 20-class CIFAR-100 10+10 suite does not reproduce every natural 100-class scenario; original-paper algorithm fidelity, secure aggregation/DP, and publication-quality results are unverified.
