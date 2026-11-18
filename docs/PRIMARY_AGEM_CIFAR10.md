# Primary CIFAR-10 A-GEM × FedAvg pilot — four 25% client exclusions

The focused pilot contains **five runs**: one baseline with four participating clients, plus **four arms excluding a different one of those clients** throughout. Every arm omits exactly **25% of clients**. CIFAR-10 has ten labels, so four disjoint class holders cannot own equal 2.5-label sets: seeded class ownership is **3/3/2/2**, and the omitted class percentages are respectively **30%, 30%, 20%, 20%**. No class is excluded in more than one arm.

| Job | Active clients | Omitted labels | Matched final-server comparison |
|---|---|---|---|
| `P00_full_participation` | All four | None | Reference |
| `P01_omit_client_0` | 1,2,3 | Owner 0's 3 labels | Both final models on the same other 7 test classes |
| `P02_omit_client_1` | 0,2,3 | Owner 1's 3 labels | Both final models on the same other 7 classes |
| `P03_omit_client_2` | 0,1,3 | Owner 2's 2 labels | Both final models on the same other 8 classes |
| `P04_omit_client_3` | 0,1,2 | Owner 3's 2 labels | Both final models on the same other 8 classes |

All run configurations keep batch size 512, local A-GEM and FedAvg with the class-wise output-row fusion adaptation. Each class belongs to exactly one client, and the allocation is seeded once and shared across arms. Two global class-arrival stages and twenty communication rounds per stage are used by default. Clients excluded from communication **still train locally and keep their A-GEM memories**; they upload nothing, and the server never sends them post-round weights. The global evaluation still includes every arrived class. The **paired final comparison** excludes withheld true-label test images from both baseline and corresponding partial result.

**Important limitation:** Four class-disjoint CIFAR-10 clients each receiving two pairs of *distinct* classes in one timestep would require at least sixteen distinct labels. This five-job pilot therefore **cannot** implement that exact strict local curriculum for every client. Some arrivals are singletons and remain compatibility fallbacks, while two simultaneous classes use binary active-class training. See [class-indexed protocol](CLASS_INDEXED_PAIR_PROTOCOL.md). This study tests the **25% communication exclusion** hypothesis, not a fully conforming four-classes-per-client curriculum.

Omitting one client reduces data exposure and server updates as well as excluding its class identities. Differences on other classes are useful preliminary signals but not isolated causal effects of the omitted labels. Repeat seeds and add update-count/data-budget controls before publication. Do not tune using the official test set.

```bash
git pull --ff-only origin main
pip install -e '.[test]'
pytest -q tests/test_primary_agem.py
bash scripts/run_primary_agem_cifar10.sh --plan-only --seed 0
bash scripts/run_primary_agem_cifar10.sh --quick --seed 0 --allow-cpu
CUDA_VISIBLE_DEVICES=0 bash scripts/run_primary_agem_cifar10.sh --seed 0
```

Each job stores its own model and metrics. The final results are in `results/primary_agem_cifar10/production/reports/`, including `paired_comparison.md`, `paired_summary.csv`, and `seen_class_effects.csv`. The exact omitted labels for each seed are in the generated `ledger.json`. No full CIFAR results have been executed or verified in this commit.
