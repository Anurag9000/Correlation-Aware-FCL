# Task-incremental protocol

## Task IDs and heads

The active setting is task-incremental learning. A task ID is the sorted set of global class IDs assigned together to one client in one local task, such as (0, 1) or (1, 2). A class may recur in a later task. Each distinct task ID owns a separate classifier head. The feature extractor is shared, while a class repeated in two tasks has separate output parameters in each head. Model construction derives the complete head registry from the compiled scenario before clients and server are initialized.

A client trains current examples with the head for their current task ID. The compiled schedule records each training image's source task ID, and replay routes that image back to its source head. If the same raw image is intentionally reused by multiple tasks, its loss is averaged across those source heads. Merely sharing a class label does not route an image to another task's head. Replay targets are mapped from global class IDs into each head's local column order.

## Prediction and overlap

Task IDs are supplied at prediction. For an input whose known task-ID set selects more than one head, each selected head produces its own softmax over its local classes. The implementation maps those probabilities back to global class IDs, averages probabilities for labels represented in multiple heads, takes the union of labels, and renormalizes the union probabilities. A single selected head is the non-overlap case. Metrics are task-ID-assisted Task-IL and are not comparable to class-incremental scores.

The CIFAR evaluator has no independent per-image task annotation. At each sync, it uses the known stream schedule to supply every task ID from a client task that has seen the image's ground-truth class by that sync. If multiple clients/timesteps saw the class under different task heads, all those heads are selected; their probabilities are mapped to global class IDs, averaged for duplicate labels, and renormalized over the union. This is the explicitly chosen class-conditioned Task-IL oracle protocol and must be reported as such.

When the final evaluation batch is shorter than the physical batch size, its image and label rows are padded together for forward computation. Task-ID sets are built for those padded rows too, and every padded prediction is discarded before metrics are computed.

The all-observed score uses every arrived task head, including a head whose client update is still offline. The separate server-integrated-only diagnostic filters both the evaluated labels and the task-head set to heads represented by training examples acknowledged by the server. This prevents an untrained pending head from competing for an already integrated class in the accepted-only diagnostic.

## Federated synchronization

The shared feature extractor follows the configured server aggregation rule. For canonical FedAvg, each task head is averaged only over uploads that report training exposure for that head. Head exposure is accumulated with a client's pending delta and cleared only when that client receives a server model, so an offline client does not lose provenance for unsent head updates. Non-contributing uploads cannot dilute an untouched head with stale parameters. FedAdam, FedAvgM, and custom server rules retain their own aggregation and optimizer update for all model tensors.

The task-head layout and per-example training-task provenance are fixed from the scenario before round one. They are included in model metadata/checkpoint compatibility and reconstructed deterministically from the saved scenario. A run created with the old single class-indexed layer or the earlier all-known-head replay rule is not compatible with this source-task replay implementation and must not be resumed under it.

## Scope and results

Task-IL is the default for new configurations and the primary runners. Historical class-IL results and dated class-indexed protocol documents remain records of the experiments that produced them; they are not Task-IL results. Neither the task-head migration nor the source-task replay-routing correction has been empirically validated yet. The running V26 pair used the earlier all-known-head replay rule. Task-IL configs reject `native_glfc`, `native_lga`, and `native_feat`: their class-indexed source trainers are bypassed by the task-head path, so silently running the generic task objective would misreport method fidelity. DER/DER++ now distill per-sample logits using the source task head; configs reject raw-sample overlap for these methods because a single stored logit target cannot disambiguate multiple source heads. These adaptations still need empirical validation.
