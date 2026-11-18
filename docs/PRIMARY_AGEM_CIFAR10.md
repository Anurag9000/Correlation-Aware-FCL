# Primary investigation: six-run CIFAR-10 A-GEM × FedAvg class-holder ablation

## Research question

Before any 67,938-job factorial campaign, test **whether permanent nonparticipation of a particular pair of class-holding clients' labels changes the final server model's accuracy on the *other* eight classes**. This is deliberately a small, descriptive feasibility study—not the full research programme or a formally identified class-identity causal effect.

## Mathematical constraint: five disjoint omissions cannot each be 25%

CIFAR-10 has ten labels; 25% is 2.5 labels, not an integer. Five non-overlapping omission sets of 25% require 12.5 unique labels, which is impossible. To honor five omission arms and *no class omitted twice*, the primary design uses **two omitted classes per arm = 20%**. It does **not** falsely report 25% class/client dropout. If exact 25% client dropout is essential, use a separately approved *four-client/four-arm* design; it cannot give five distinct one-client omissions without reusing a client.

## Why five clients, not ten?

Each of five clients owns **two distinct classes**, one in the first experience (CIFAR-10 labels 0–4) and one in the second (labels 5–9). This provides genuine within-client sequential A-GEM learning: the client's first-stage examples become historical memory when it starts training on its second-stage class. Using ten clients with one class each would largely remove that A-GEM mechanism.

An experiment-seeded random permutation assigns one old class and one new class to each client. This assignment is generated **once**, recorded, and shared by the baseline and all five omission arms. No label is shared by clients, and no image is duplicated across clients.

## Exact six jobs

| Job ID | Selected clients every round | Withheld classes | Inclusion set for final comparison |
|---|---|---|---|
| `P00_full_participation` | All 5 clients | None | All 10 (contextual baseline); re-evaluate on each arm's eight included classes |
| `P01_omit_client_0` | Clients 1,2,3,4 | Both labels assigned to client 0 | Other 8 labels |
| `P02_omit_client_1` | Clients 0,2,3,4 | Both labels assigned to client 1 | Other 8 labels |
| `P03_omit_client_2` | Clients 0,1,3,4 | Both labels assigned to client 2 | Other 8 labels |
| `P04_omit_client_3` | Clients 0,1,2,4 | Both labels assigned to client 3 | Other 8 labels |
| `P05_omit_client_4` | Clients 0,1,2,3 | Both labels assigned to client 4 | Other 8 labels |

The **identity** of the excluded client/class pair is randomized by `--seed` at ledger construction, but remains excluded at every communication round for that arm. This is *permanent fixed exclusion*, not independently randomized 20% dropout on each round. Rotating client dropout would not make its classes permanently absent and would answer a different question.

## Shared controlled conditions

- Official CIFAR-10, disjoint class/client data and no deliberately repeated training images.
- Two experiences: original labels 0–4 first, 5–9 second; 20 communication rounds per experience by default.
- Same PyTorch initialization seed, same data allocations, model architecture, learning rate, batch size **512**, local epoch count (1), A-GEM memory size, FedAvg aggregation and evaluation protocol in all six arms.
- Full baseline: all 5 class-holding clients contribute at every round.
- Partial arms: exactly 4 of 5 clients contribute at *every* round; the fifth is never selected. The excluded client still receives its data arrival events as required by the global FCL test-coverage invariant.
- Normal A-GEM historical memory survives synchronization. The server receives and aggregates selected client weights; selected clients receive the new global weights immediately after aggregation.
- Training uses the normal **10-logit** classifier for every job. No hidden post-hoc masking of logits creates artificially easy results.

## Primary measurement: identical seen-class test population

For arm `Pk`, define its held-out label set \(H_k\) (two unique classes), and its included set \(S_k=\{0,\dots,9\}\setminus H_k\) (eight classes).

Let \(f_{\mathrm{full}}\) and \(f_k\) denote the saved final server models. Evaluate both using **exactly** the official CIFAR-10 test images whose true labels are in \(S_k\):

\[
A_{\mathrm{full}\mid S_k}=\frac{1}{|T_{S_k}|}\sum_{(x,y)\in T_{S_k}}\mathbf 1[f_{\mathrm{full}}(x)=y],\quad
A_{k\mid S_k}=\frac{1}{|T_{S_k}|}\sum_{(x,y)\in T_{S_k}}\mathbf 1[f_k(x)=y].
\]

The primary effect in percentage points is \(\Delta_k=100(A_{k\mid S_k}-A_{\mathrm{full}\mid S_k})\). **Negative** means omitting that client/class pair worsens accuracy on *other* labels; **positive** means it improves them. The report includes per-class deltas on each of the eight included labels. Withheld classes never appear as rows in these comparisons.

The baseline ten-class test accuracy is reported *only for context* and must **not** be directly compared with an eight-class partial accuracy; it uses a different test-label population. The simulator's regular global `metrics.json` still evaluates **all classes observed anywhere**, including inactive class holders, as required by the existing global evaluation contract. Those global metrics are not used to construct the primary eight-class comparison.

## Files and run commands

```bash
pip install -e '.[test]'
pytest -q tests/test_primary_agem.py

# Preview the six arms and the randomized, non-repeated omitted class pairs.
bash scripts/run_primary_agem_cifar10.sh --plan-only --seed 0

# Fast functional test: only 10% train images per class, 2 rounds/stage, linear network.
# These are not publishable production results; outputs are isolated under /quick.
bash scripts/run_primary_agem_cifar10.sh --quick --seed 0

# Six complete runs, one after another; requires CUDA by default.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_primary_agem_cifar10.sh --seed 0

# Recompute the side-by-side report from completed model checkpoints.
CUDA_VISIBLE_DEVICES=0 bash scripts/run_primary_agem_cifar10.sh --seed 0 --compare-only
```

Use `DATA_ROOT` to change the CIFAR download location, `RESULTS_ROOT` for the output directory. `--allow-cpu` is provided only for debugging. `--no-download` forces an offline CIFAR dataset. `--limit 1` runs only the baseline as a smoke test and appropriately does **not** create a misleading five-arm comparison. The same command can be safely rerun; completed jobs with matching configurations are skipped. For new hyperparameters or seeds, use a separate result root to avoid mixing designs.

### Output layout

```text
results/primary_agem_cifar10/
  production/
    ledger.json                    # deterministic class-holder assignment and all six configs
    runs/
      P00_full_participation/      # config.json, metrics.json, model_final.pt, checkpoints, status
      P01_omit_client_0/
      P02_omit_client_1/
      P03_omit_client_2/
      P04_omit_client_3/
      P05_omit_client_4/
    reports/
      paired_comparison.md         # primary five-row comparison
      paired_summary.csv           # baseline eight-class vs partial eight-class and deltas
      paired_summary.json
      seen_class_effects.csv       # 40 class-level comparisons; no omitted labels
  quick/                            # separate diagnostic outputs
```

## Interpretation and limits

This experiment is reasonable as a **first diagnostic screen** for indirect class-holder effects. It has six jobs, a common initialization/data mapping, explicitly recorded exclusions, matched test populations, and a method that actually has within-client sequential experiences.

It **cannot by itself conclude** that a specific class's information causally improves or harms other classes, because omitting its holder also reduces the number of active contributors, total training examples and aggregation mixture (5 vs 4 participants). To strengthen a subsequent study, add a matched four-client-aggregation budget control that rotates exclusions but still exposes all ten labels over training, and repeat randomized class assignments/seeds for uncertainty estimates. Also, A-GEM's local gradient protection is not a guarantee of global retention after FedAvg.

The **official test split must not be used for choosing hyperparameters or tuning the model**. Preselect hyperparameters or tune them on a held-out subset of original training data before interpreting this confirmatory comparison. The initial implementation is explicitly a preliminary diagnostic and performs no tuning.

**This six-job ledger is independent of and does not modify the existing 67,938-job CIFAR-10 factorial ledger.**
