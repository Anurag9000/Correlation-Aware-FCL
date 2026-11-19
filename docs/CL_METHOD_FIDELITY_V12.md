# v12: Whole-repository continual-learning method fidelity ledger

**Scope:** the actual local CL methods registered under \`RunConfig.method\`, their paper-derived mathematical objectives and states, their training-time memory/importance mechanics, client-private to server-synchronized CL histories, and objective-preserving projection. This audit deliberately **does not modify** the original CIFAR-100 federated task stream, its 2-class tasks, class-incremental global evaluation, five client/task barriers, full versus rotating-offline participation, single shared GPU, common 100-class output layer, frozen inactive classifier rows, shared image archive, or canonical v11 server aggregation.

**Counting:** there are **11 primary training modes** (including SGD baseline) and **10 distinct nontrivial CL methods**: Replay, GEM, A-GEM, EWC, SI, MAS, LwF, ER-ACE, DER, DER++. Additional method ablations (GEM memory grouping, reservoir policy/capacity, importance budgets, optimizer state, FedProx, replay ratio, strength coefficients) are factorial *variants*, not 20 separate original papers.

## Fidelity classifications

* **Mathematical core verified:** the named paper's signature update/projection/penalty is mathematically recognizable and independently tested.
* **Setting adaptation:** required by the user's pre-existing 100-class shared head, strict active-class softmax, inactive classifier-row freezing, or cross-client synchronization. These changes make a *full historical experiment reproduction* impossible without changing the setting, which the user forbids.
* **Approximation:** a stochastic/bounded estimator consistent with the source definition, but finite-sample rather than exact expectation.
* **Method variant:** a registered option differs in a specific named way (for instance per-class constraints rather than per-task GEM constraints). This must not be called the canonical algorithm without qualification.
* **Pending scientific verification:** one or more all-GPU or cross-seed accuracy and long-stream checks were not run on the user’s machine.

## Shared local training contract (all 11 methods)

Every client maintains an independently trained ViT-Tiny/16 or CIFAR ResNet-18. Task 1 trains in parallel across five clients, barrier; Task 2 in parallel across the same five, barrier; then server merge/broadcast to selected clients. Locally, each task has two currently active class IDs (or the explicitly configured count); the backbone and current-class linear rows train, and inactive *classifier rows and biases* are protected by gradient masking and exact post-step restoration, including momentum buffers. The shared 100-dimensional classifier is **not a collection of independent paper task heads**, and the current-task softmax denominator is restricted to active task classes by design. No task oracle is added at global inference.

This local setting creates unavoidable deviations from papers that use full-class new-data cross-entropy, separate old/new heads, or differently constrained output layers. Those differences are **not hidden** and are not "repaired" by changing the stream.

## 1. SGD / Fine-tuning baseline (not a CL paper)

Core: ordinary \`torch.optim.SGD\` (momentum optionally configured), current-task supervised cross-entropy and the frozen-inactive-classifier-row invariant. Optional \`optimizer_state_policy\` chooses reset or persistence. Gradient accumulation preserves effective logical batch \`cfg.batch_size\` even when physical \`micro_batch_size\` is much smaller. This is the **naïve sequential fine-tuning baseline under the strict stream**, not joint offline retraining. No episodic memory/importance/teacher is created.

## 2. Experience Replay (ER)

Sources: Chaudhry et al., *On Tiny Episodic Memories in Continual Learning* (2019), https://arxiv.org/abs/1902.10486; https://github.com/facebookresearch/agem.

Signature objective: current supervised loss plus labeled historical-example supervised loss. In **this strict local setting**, new data uses active-pair CE; replayed old samples use 100-class CE with fixed old classifier rows and trainable backbone/current rows. That is a **setting-adapted ER objective**, *not the paper’s identical loss applied jointly to new and replay data*. Do not claim full original experiment parity.

Memory: the two-arm pilot retains \`ceil(0.10*450)=45\` private samples/class by **persistent independent per-class Algorithm R**, with sample-ID deduplication across reobservations. Other nonfractional memory methods now also use exact per-class Algorithm R for \`random_per_class\`; \`recent_per_class\` is explicitly a distinct FIFO-like variant. A synchronized client uses its private sample reservoir **plus** the complete server-accepted exemplar union. The server archive is intentionally unbounded, unlike a standalone small-buffer ER benchmark; this is an explicit federated-CL adaptation and means retained historical sample count can grow with communication.

Replay is not introduced into a client before it actually receives that server archive. Hidden clients keep only their own earlier memory and model until they reconnect.

## 3. Gradient Episodic Memory (GEM)

Source: Lopez-Paz & Ranzato, *Gradient Episodic Memory for Continual Learning* (NeurIPS 2017), https://arxiv.org/abs/1706.08840; https://github.com/facebookresearch/GradientEpisodicMemory.

At each current-task optimizer step, compute current gradient \`g\` and reference gradients \`g_j\` on previous-memory sets. If any \`g·g_j < 0\`, solve
\`min_z 0.5 ||z-g||²\` subject to \`z·g_j >= 0\` for **each** previous reference group. The solver implements a constrained nonnegative dual quadratic problem with double-precision CPU solve and verifies primal feasibility (no silent unprojected fallback). The projection is applied to the same model parameters as the current gradient. BatchNorm running statistics are restored after reference-gradient probes.

**Grouping options matter:** \`gem_grouping='stage'\` groups reference examples by their *source client and original local experience*, including remote examples; it now evaluates the *past experience’s active-class loss*, not the new task’s labels. \`gem_grouping='class'\` (historical default) implements **one constraint per old class**, a legitimate but distinct research variant rather than the paper’s per-past-task GEM. \`'all'\` collapses everything into one constraint, also a custom variant. If a reservoir has only one available label in an experience, it falls back to full-output CE for that group rather than inventing missing labels.

Inactive classifier-row gradient masking remains enforced under all three grouping strategies. Since the original paper did not freeze old-class classifier rows in this exact way, comparisons must state the class-head adaptation.

## 4. Averaged GEM (A-GEM)

Source: Chaudhry et al., *Efficient Lifelong Learning with A-GEM* (ICLR 2019), https://openreview.net/forum?id=Hkf2_sC5FX; https://github.com/facebookresearch/agem.

Signature projection: draw a reference minibatch from the combined past episodic memory. If \`g·g_ref >= 0\`, update with \`g\`; otherwise use
\`g' = g - (g·g_ref / ||g_ref||²) g_ref\`.
\`agem_project\` handles zero-reference vectors and all finite dimensions. Current-task active-class gradients and old-class reference-gradient masking retain the user-defined head freeze. In this setting, current-task CE is restricted to active labels but aggregated historical reference CE spans the full output, a **loss-geometry adaptation** that differs from an unconstrained common-loss original reference. The projection inequality itself is mathematically verified.

An online client recomputes reference gradients against **its current** model from the synchronized exemplar evidence. The server does **not** average task-specific gradients computed against different client weights.

## 5. Elastic Weight Consolidation (EWC)

Source: Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks* (PNAS 2017), https://www.pnas.org/doi/10.1073/pnas.1611835114.

Signature diagonal quadratic penalty:
\`(lambda/2) * sum_i F_i (theta_i - theta_i*)²\`.
The previous version incorrectly computed **square of the mean gradient of a single training minibatch**, \`(mean g)^2\`, which is not mean squared per-example gradient \`mean(g²)\`.

**v12 correction:** for each completed local experience, sample up to \`importance_samples_per_experience\` distinct *TRAIN* examples (default **32**). For each example, sample a predicted label from the **current model's predictive distribution** over that episode's active output classes, compute its individual NLL gradient, square it, and average the squares. This is a bounded Monte Carlo **model-predictive diagonal Fisher** estimate. Estimation uses \`model.eval()\`, preserving dropout and BatchNorm state, and never reads validation/test data. It is not the exact expectation over all possible data/labels; approximation uncertainty is explicit.

Accumulate all past diagonal Fisher precisions and precision-weighted past anchors. Replacing a sum of diagonal quadratics by one precision-summed quadratic with a precision-weighted center is mathematically equivalent up to a parameter-independent constant. Server accepts only NEW pending increments, not rebroadcast echoes, and clients rehydrate the merged precision and anchors. This is a federated approximate-EWC extension; there is no claim that the original paper used federated precision fusion. No artificial decay \`gamma\` is silently introduced—online EWC with exponential discounting is **not** registered as a separate method.

## 6. Synaptic Intelligence (SI)

Source: Zenke, Poole & Ganguli, *Continual Learning Through Synaptic Intelligence* (ICML 2017), https://proceedings.mlr.press/v70/zenke17a.html; https://github.com/ganguli-lab/pathint.

Signature path importance:
\`omega_i = sum_steps -g_i^task * (theta_i^(step+1)-theta_i^step)\`,
\`Omega_i = max(omega_i/(Delta_theta_i² + xi),0)\`, then cumulative quadratic prior penalty over old experience anchors.

**v12 correction:** \`g_i^task\` is now derived from the **unregularized current task loss**, not the already-regularized total gradient (which could include SI/FedProx). Actual optimizer displacement continues to include effects of the chosen optimizer and any regularizer, consistent with the path integral over realized local updates. Frozen classifier rows have zero net displacement. Accumulators are finalized **before** task/communication boundaries, not one experience late, and newly consolidated increments are uploaded only once.

The server merges the per-parameter precision-weighted statistics as a federated extension; this is not in the original nonfederated paper.

## 7. Memory Aware Synapses (MAS)

Source: Aljundi et al., *Memory Aware Synapses: Learning what (not) to forget* (ECCV 2018), https://arxiv.org/abs/1711.09601.

Signature label-independent importance:
\`Omega_i = mean_x | d(||f_theta(x)||²) / d theta_i |\`.
The previous version took the absolute gradient **after averaging a minibatch function**: \`|mean_x g_i(x)|\`. This can incorrectly cancel positive and negative sensitivities, underestimating importance.

**v12 correction:** compute the squared L2 output norm for **each individual TRAIN example**, differentiate it, take the absolute gradient per sample, then average. Preserve model buffers and use the same bounded sample budget as EWC. For strict active-class tasks, the norm is taken on that task's active logits (the task's own output subspace); unlike the original unconstrained single-head formulation, future/unseen random class logits are excluded by the required stream setting. Additive precision and precision-weighted anchors remain available locally and through server sync.

## 8. Learning without Forgetting (LwF)

Source: Li & Hoiem, *Learning without Forgetting* (ECCV 2016), https://arxiv.org/abs/1606.09282; https://github.com/lizhitwo/LearningWithoutForgetting.

Maintain a **frozen previous model teacher** and train the current model using current-data supervised CE plus old-class temperature-scaled distillation:
\`lambda_KD T² KL(p_teacher_old(T) || p_student_old(T))\`. The code uses temperature \`T=2\` and the existing \`der_alpha\` parameter for distillation strength, because that is how the frozen factorial study indexes the ablation; it is **not a DER replay buffer**.

The teacher snapshot is moved at local experience transitions; online clients receive the latest committed global teacher at synchronization. Historical class IDs are selected explicitly, rather than distilling over all 100 random/future logits.

**Adaptations:** LwF originally used old/new task outputs rather than this repository’s single 100-class shared linear output; the server-wide teacher snapshot is an additional federated feature. The local task softmax and inactive rows still obey the repository's strict episode setting. This is a recognizable LwF distillation mechanism, not literal reproduction of its architecture/training experiments.

## 9. ER with Asymmetric Cross-Entropy (ER-ACE)

Source: Caccia et al., *New Insights on Reducing Abrupt Representation Change in Online Continual Learning* (ICLR 2022), https://arxiv.org/abs/2104.05025.

Signature asymmetric objective: incoming-example CE masked to the current classes, plus ordinary full-class labeled replay CE. In the **strict local-pair** configuration, the incoming loss is the required 2-class active-row CE, and replay loss uses all shared-head logits. This is *close to the core asymmetric mechanism*, but the original ER-ACE procedure does not universally freeze all historical classifier rows or use exactly this pairwise task protocol. Historical replay bytes and per-class reservoir are synchronized under the user's federated protocol. The \`der_beta\` field controls the replay CE coefficient in existing ablations.

## 10. Dark Experience Replay (DER)

Source: Buzzega et al., *Dark Experience for General Continual Learning: A Strong, Simple Baseline* (NeurIPS 2020), https://arxiv.org/abs/2004.07211; https://github.com/aimagelab/mammoth.

Signature replay term: \`alpha * E_{(x,z_old)~buffer} ||f_theta(x)-z_old||²\` added to incoming-data CE, where stored \`z_old\` are **raw logits**, not softmax probabilities.

**v12 correction:** online per-class reservoir sampling occurs as new training minibatches are processed; logits are captured from the live network **before** the corresponding optimizer step. When a new exemplar enters the private reservoir, its raw trajectory logits are stored and **never rewritten** by a later task-consolidation model. This fixes the older v11 stage-end logit snapshot, which could not represent the trajectory. Logits for evicted private examples are removed unless already retained by the server's merged history. Offline clients keep their private reservoir/logits until reconnection. Received examples without historical logits (possible in mixed-method federations) are excluded from DER's logit-matching references rather than assigned fictitious targets.

**Adaptation:** The private reservoir is a per-class fixed quota, not the paper's globally uniform reservoir; the cross-client server retains an unbounded deduplicated union. Both are explicit study design constraints, not paper features.

## 11. Dark Experience Replay++ (DER++)

Same Buzzega et al. source. Objective:
\`L_incoming + alpha * MSE(f_theta(x_mem), z_old) + beta * CE(f_theta(x_mem), y_mem)\`.
It uses the same **v12 online pre-step logit capture** and reservoir mechanics as DER, with the additional supervised memory CE term. Its two replay losses draw **independent memory minibatches**, as in the original DER++ implementation, rather than reusing a single correlated replay selection. The code stores both historical raw logits and labels. Unlike DER alone, DER++ directly optimizes supervised old-class replay accuracy through the shared backbone (while inactive local classifier rows stay frozen).

## Auxiliary FedProx regularizer (not a twelfth CL method)

Source: Li et al., *Federated Optimization in Heterogeneous Networks* (MLSys 2020), https://proceedings.mlsys.org/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html.

When \`fedprox_mu>0\`, the local objective adds \`(mu/2)*||theta-theta_last_downloaded||²\`, anchored to the **last global model actually received**. A disconnected client must not be anchored to an unseen future global snapshot. FedProx does not replace the selected server aggregation method; it is an orthogonal local training option.

## Federated state exchange and causal restrictions

All 11 methods retain independent local model/experience state, including periods of disconnection. The server accepts only participating clients' pending contributions. Experience replay exemplars move as real 32x32 RGB bytes, not merely a dataset index; gradient-constraint evidence is merged as exemplar tables (reference gradients recomputed by the receiver); EWC/SI/MAS importance is merged by natural-parameter precision and numerator; DER historical logits are associated with stable example IDs; LwF transfers the versioned global teacher. The server broadcasts committed weights plus the full compatible archive only to connected clients. Offline clients are not altered.

The server history and accepted class-exposure counts are **not** allowed to use hidden clients' unacknowledged samples or classes. Server aggregation itself remains the v11 numerical algorithm, unchanged here.

## Tests, reproducibility, and remaining differences

* \`tests/test_cl_fidelity_v12.py\`: paper-oracle numerical tests for per-example EWC Fisher and MAS importance; BN state invariance; GEM/A-GEM constraints; persistent nonfractional per-class reservoir; **online DER/DER++ PRE-step logits**; all 11 CL methods under the strict active-class episode; inactive classifier-row immutability.
* Existing \`test_federated_cl_v8.py\` covers each method’s first post-sync experience, disconnected client rejoin, and server state fusion.
* Existing \`test_reference_gradients.py\` checks BatchNorm invariance under GEM/A-GEM reference-gradient probes.
* Existing \`test_reservoir_v9.py\` covers the 10%-per-class primary pilot and unbiased reservoir inclusion.
* Experiment identity is bumped to **v12** for the primary pilot and full suite. New CL estimator sample cap \`importance_samples_per_experience=32\` is explicit and participates in run configuration, but its default is omitted from legacy frozen factorial geometry fingerprints so their **set of planned configurations** remains comparable (results/checkpoints do not).

**What is NOT claimed:** exact paper parity for different original settings, unlimited-budget Fisher/MAS expectations, old-version checkpoint reproducibility, all-GPU/VRAM success, final accuracy gains, calibration of a globally evaluated 100-way classifier, or original authors' exact architectural hyperparameters and datasets. Federated history fusion itself is an additional, separately defined experimental protocol.

**Research validity rule:** All original method names should be accompanied by "adapted to fixed shared-head, frozen-inactive-row federated 2-class episode stream" in results and comparisons. In particular, **GEM-class**, **GEM-all**, **per-class DER reservoir**, and **globally synchronized LwF teacher** are variants and should not be silently labeled identical to original author experiments.
