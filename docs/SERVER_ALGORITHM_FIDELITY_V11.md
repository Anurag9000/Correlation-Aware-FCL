# v11: Server aggregation and broadcast fidelity audit

Scope: **all nine server aggregation names** available in the Correlation-Aware-FCL repository. This document concerns server aggregation, class-history accounting, and model/CL-state broadcasts. Client losses, local continual-learning methods, sampling schedules and GPU concurrency are unchanged.

## Algorithm inventory and scientific status

| Server strategy | Classification | Server equation/meaning | Class-history treatment |
|---|---|---|---|
| \`fedavg\` | **McMahan et al. 2017 canonical, when all anchors are fresh** | Complete-model \`sum(n_k W_k)/sum(n_k)\`, including classifier rows | Dataset/sample count only; server-seen class count does not appear in the canonical formula |
| \`uniform\` | Exact equal-client model mean (ablation) | Complete-model \`sum(W_k)/|S|\` | No class-history reweighting |
| \`fedavgm\` | Hsu et al. 2019 Nesterov server momentum variant | \`delta=weighted_mean(W_k)-G\`; \`velocity=beta*velocity+delta\`; \`G'=G+eta*(delta+beta*velocity)\` for all floats | No class-specific weighting beyond original sample-weighted base delta |
| \`fedadam\` | Reddi et al. 2021 FedOpt Adam server rule | \`delta=weighted_mean(W_k)-G\`; \`m=beta1*m+(1-beta1)*delta\`; \`v=beta2*v+(1-beta2)*delta^2\`; \`G'=G+eta*m/(sqrt(v)+tau)\`, \`v0=tau^2\`, **without bias correction** | No class-specific weighting beyond original sample-weighted base delta |
| \`class_balanced\` | **Repository-specific research algorithm** | \`score_k=sum_c n_{kc}/(H_c+sum_j n_{jc})\`, normalized across uploaders | \`H_c\` is only the server's prior accepted **TRAIN example** exposure count, not clients' unknown/unsent history |
| \`inverse_participation\` | Repository-specific research algorithm | Scores proportional to new uploaded examples divided by `1 +` prior accepted selection count | No direct class-count term; biases toward less frequently selected clients |
| \`conflict_aware\` | Repository-specific cosine-conflict reweighting | Downweight model deltas that oppose other uploaders | No explicit class-count term |
| \`conflict_projection\` | Repository-specific PCGrad-style client-delta projection | Project conflicting flattened model deltas, then weighted-average the corrected updates | No explicit class-count term |
| \`retention_aware\` | Repository-specific history-gradient weighting | Rescale client update weights based on alignment with historical surrogate gradient | Uses privileged historical TRAIN-gradient evidence and is not privacy-preserving |

### Why canonical FedAvg does NOT directly weigh by class count

McMahan et al. define \`n_k=|P_k|\`, the **number of client training examples**. A client seeing 8 classes instead of 4 does not automatically have double the canonical FedAvg weight; it has greater weight only insofar as it has more relevant training samples. Class-count/recency/exposure weighting is an *additional research-specific aggregation mechanism*, not the original FedAvg formula.

The v11 server maintains \`class_sample_exposure[c]\` by summing only \`class_counts[c]\` from uploads actually **accepted** at communications. Thus after a server sync the server can distinguish: (i) globally integrated classes, (ii) how often classes were represented in accepted training examples, (iii) incoming per-class counts, and (iv) the local client's older model anchor version. A hidden client continues learning and accumulating an unsent delta/class counts locally; neither its class-history table nor private data enters the server's exposure table before reconnection.

The \`class_balanced\` custom rule uses that accepted exposure table. All the other custom methods retain their separately documented weighting rules. **No method silently changes FedAvg to history-weighted FedAvg.**

### Classifier head synchronization

Previous versions averaged the backbone and then overwrote every classifier row with an exposure-step-specific \`classwise_merge\`, including in \`fedavg\`, \`uniform\`, \`fedavgm\` and \`fedadam\`. That was not the original full-model algorithm. In v11 **the entire model—including all 100 classifier rows and classifier bias—is aggregated with the same canonical server formula**. \`classwise_head_merge\` is applied only to the five explicitly custom schemes; its historical configuration default remains serialized identically for compatibility with the frozen factorial *geometry* audit, but it is not executed in canonical methods. The primary FedAvg pilot additionally sets it explicitly to \`False\`.

Non-floating BatchNorm \`num_batches_tracked\` has no defined arithmetic mean in the original method. The repository keeps the maximum counter. All floating BatchNorm running statistics are combined under the chosen server float rule. FedBN-style local-BN preservation is **not** implemented.

### Offline reconnection: an unavoidable distinction

Original synchronous FedAvg requires every selected client to start local training from **the same current global model**. This project's offline client must keep training while disconnected. Its own stale parameter trajectory therefore cannot be both retained *and* treated as an ordinary fresh synchronous FedAvg model. When it reconnects:

\`\`\`text
A_k = last global model received by client k
L_k = client's privately trained model since A_k
G_t = latest server model
rebased_k = G_t + (L_k - A_k)
\`\`\`

The server runs the chosen complete-model aggregation on this rebased candidate. This retains the offline client's unsent learning while being an **explicit noncanonical stale-delta adapter**, not the exact original FedAvg, FedAvgM or FedAdam on that round. No extra stale-age damping, re-training from current global weights, or custom class calibration is silently introduced. Every round records \`stale_rebased_client_ids\`, \`upload_model_origins\`, \`aggregation_family\`, \`server_class_sample_exposure\`, \`server_integrated_class_count\`, and \`classwise_head_merge_applied\`.

### Server communication boundary and state

After accepting uploads, the server (1) aggregates accepted complete model representations; (2) optionally applies a published server optimizer or repository-specific custom transformation; (3) merges new client CL artifacts into the central archive; (4) increments accepted-only class exposure; (5) broadcasts the **identical committed global model and complete CL-state snapshot** to all selected/online clients. Hidden clients are neither overwritten nor acknowledged. The terminal flush uploads pending changes without local training. Checkpoints include server moments, class-iteration counts, accepted-class exposure, merged CL archives and client-local states.

### Exact numerical validation

\`tests/test_server_fidelity_v11.py\` checks: all nine methods finite and dimension-consistent; sample-weighted full-model FedAvg and classifier bias; uniform equal mean; FedAvgM two-step Nesterov oracle; FedAdam two-step uncorrected-moment oracle; historical-class-weighted custom rule; fresh/stale upload provenance; canonical skip of custom head merging; and server-to-client full-weight broadcasts.

**Source papers:**

* McMahan et al., 2017, *Communication-Efficient Learning of Deep Networks from Decentralized Data*: https://proceedings.mlr.press/v54/mcmahan17a.html
* Hsu, Qi & Brown, 2019, *Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification*: https://arxiv.org/abs/1909.06335 ; Flower reproduction Nesterov note: https://flower.ai/docs/baselines/fedavgm.html
* Reddi et al., 2021, *Adaptive Federated Optimization* (FedAdam/FedOpt): https://arxiv.org/abs/2003.00295

### Scope limitations

Source-code and CPU numerical tests **do not demonstrate** high CIFAR-100 accuracy, exact floating-point bit parity against every external library, or actual CUDA operation. Class-aware/conflict/retention server algorithms are original experiments rather than canonical paper reproductions. Stale online reconnection necessarily remains outside original synchronous FedAvg; paper-faithful asynchronous optimization would require selecting a separately specified asynchronous algorithm and implementing its own staleness dynamics and baselines.


### Privacy boundary of server gradients and class statistics

Server-side historical-gradient diagnostics and retention-aware weighting now restrict training-data references to `acknowledged_train_indices`—the union of indices belonging to clients **whose model upload was accepted**. Hidden clients' unsent sample IDs are not included until an accepted upload. Representation-drift measurements reuse the *same acknowledged historical cohort* before and after aggregation, avoiding a changing-denominator artifact. This is still a privileged common-dataset simulation: merely knowing acknowledged IDs is not the same as having uploaded every raw image. Production FL would require transmitting the authorized reference samples or sufficient gradient summaries explicitly.
