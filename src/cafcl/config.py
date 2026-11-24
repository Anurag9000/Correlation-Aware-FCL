"""Strict experiment configurations, independently serializable per job."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json


class ConfigValidationError(ValueError, AssertionError):
    """Explicit invalid configuration error (legacy AssertionError compatibility)."""


def _require(condition, message):
    if not condition:
        raise ConfigValidationError(message)

@dataclass(frozen=True)
class RunConfig:
    dataset: str = "cifar100"
    batch_size: int = 64
    physical_batch_policy: str = 'fixed_pad'
    num_clients: int = 10
    rounds_per_stage: int = 20
    epochs: int = 1
    participation: float = 0.5
    selector: str = "random"
    method: str = "sgd"
    aggregator: str = "fedavg"
    scenario: str = "disjoint"
    stages: int = 2
    memory_per_class: int = 20
    replay_reservoir_fraction: float = 0.0  # positive => fixed per-class Algorithm R reservoir
    share_continual_state: bool = True
    server_memory_per_class: int = 0  # 0 preserves the historical append-only archive
    stale_update_policy: str = 'damped_delta'
    terminal_stale_guard: bool = True
    aggregation_weight_policy: str = 'fresh_examples'
    task_head_weight_policy: str = 'historical_exposure'
    lr: float = 0.03
    model: str = "vit_base_patch16_32"
    seed: int = 0
    full_sync: bool = False
    relation: str = "natural"
    recurrence_delay: int = 1
    recurrence_client_policy: str = "different"
    overlap: float = 0.5
    dirichlet_alpha: float = 0.5
    target_client: int = -1
    force_target: str = "none"
    intervention_round: int = -1
    gradient_diagnostics: bool = False
    causal_diagnostics: bool = False
    causal_max_clients: int = 6
    subset_diagnostics: bool = False
    prepost_diagnostics: bool = False
    representation_diagnostics: bool = False
    resume: bool = True
    checkpoint_every: int = 0
    participation_schedule: str = "constant"
    pending_policy: str = "latest"
    max_pending_samples: int = 0
    replay_ratio: float = 1.0
    replay_class_sampling_policy: str = 'distinct_first'
    replay_update_policy: str = "sampled_mixed"
    replay_bn_policy: str = "batch_update"
    replay_buffer_order_policy: str = "archive_order"
    replay_padding_policy: str = "repeat_chunk"
    der_alpha: float = 0.5
    der_beta: float = 0.5
    regularization_strength: float = 10.0
    fedprox_mu: float = 0.0
    server_lr: float = 1.0
    server_beta1: float = 0.9
    server_beta2: float = 0.999
    fedadam_epsilon: float = 1e-8
    memory_policy: str = "random_per_class"
    local_optimizer: str = "sgd"
    client_heterogeneity: bool = False
    fork_source_job: str = ""
    fork_checkpoint_round: int = 0
    job_id: str = ""
    block_id: str = ""
    # Controlled class allocation and original 10+10 CIFAR-100 experiment.
    class_subset: tuple[int, ...] = ()
    class_order_override: tuple[int, ...] = ()
    holders_override: dict[str, tuple[int, ...]] = field(default_factory=dict)
    class_holder_weights: dict[str, tuple[float, ...]] = field(default_factory=dict)
    class_sample_fraction: dict[str, float] = field(default_factory=dict)
    participation_overrides: dict[str, tuple[int, ...]] = field(default_factory=dict)
    arrival_lag_by_client: dict[str, int] = field(default_factory=dict)
    # Explicit data and time interventions.
    recurrence_round_delay: int = 0
    recurrence_period: int = 0
    recurrence_repeats: int = 1
    sample_overlap_fraction: float = 0.0
    client_sample_skew: float = 0.0
    class_imbalance_power: float = 0.0
    stage_participation: tuple[float, ...] = ()
    # Local continual-learning and optimizer ablations.
    gem_grouping: str = "class"
    optimizer_state_policy: str = "persist"
    momentum: float = 0.9
    weight_decay: float = 5e-4
    si_xi: float = 0.1
    importance_samples_per_experience: int = 32
    # Additional supervised selection and aggregation variants are privileged
    # simulator controls, not deployable privacy-preserving policies.
    selection_temperature: float = 1.0
    boundary_mode: str = "known"  # hidden = class-aware, task-boundary-free
    learning_setting: str = "task_il"
    label_noise_rate: float = 0.0
    spurious_strength: float = 0.0
    spurious_ood_eval: bool = False
    reset_memory_on_sync: bool = False
    feedback_threshold: float = 0.25
    feedback_gain: float = 0.5
    # Strict two-label task IDs; singleton tasks cannot define discriminative heads.
    class_pair_protocol: str = 'strict'
    classwise_head_merge: bool = True
    # Zero keeps all active clients concurrent. A lower explicit value runs
    # independent client tasks in bounded waves to fit shared GPU memory.
    client_training_concurrency: int = 0

    # Optional train-only CIFAR-100 crop/flip transform used by the primary run.
    training_augmentation: str = 'none'

    # New communication-interval task stream; orthogonal design knobs.
    stream_protocol: bool = False
    classes_per_local_task: int = 2
    local_tasks_per_round: int = 2
    stream_client_chain: bool = False
    stream_overlap_kind: str = 'none'
    stream_overlap_fraction: float = 0.0
    stream_recurrence_rounds: int = 0
    offline_duration_rounds: int = 0
    stream_terminal_flush: bool = True
    stream_offline_policy: str = "random"  # random or rotate_once (pilot only)
    stream_sync_every_batch: bool = False
    stream_partial_batch_policy: str = "include_padding"
    evaluate_final_only: bool = False
    # Historical runs explicitly audited test each sync; opt out for strict final-only test.
    audit_test_every_sync: bool = True
    # Opt-in joint protocol. Empty values preserve legacy signatures.
    joint_plan_path: str = ''
    joint_archive_upload_policy: str = 'selected'
    joint_archive_download_policy: str = 'selected'
    joint_plan_hash: str = ''
    joint_local_evaluation: bool = False
    joint_allow_partial_coverage: bool = False
    joint_control_kind: str = 'federated'
    joint_domain_mode: str = 'none'
    joint_epoch_segments: int = 1
    joint_mechanistic_probes: bool = False
    joint_checkpoint_policy: str = 'all'

    def validate(self):
        _require(self.dataset in {"cifar10", "cifar100", "cifar100_20", "synthetic"}, self.dataset)
        if self.batch_size<2:raise ValueError('Batch size must be at least two')
        if self.physical_batch_policy not in {'fixed_pad','variable'}:raise ValueError('Unknown physical batch policy')
        if not self.joint_plan_path:
            _require(self.batch_size == 64 or self.dataset == 'synthetic', 'Legacy production protocols use physical batch 64')
            if self.physical_batch_policy!='fixed_pad':raise ValueError('Variable physical batches require the joint protocol')
        if self.physical_batch_policy=='variable' and (self.method not in {'sgd','replay'} or (self.method=='replay' and self.replay_update_policy not in {'sampled_mixed','sampled_mixed_accumulated'})):
            raise ValueError('Variable batches require SGD or sampled Replay')
        if self.replay_class_sampling_policy not in {'distinct_first','class_uniform_with_replacement'}:raise ValueError('Unknown replay class-sampling policy')
        _require(self.num_clients >= (1 if self.joint_plan_path and self.joint_control_kind=='centralized_sequential' else 2), "Invalid configuration: self.num_clients >= (1 if self.joint_plan_path and self.joint_control_kind=='centralized_sequential' else 2)")
        _require(0 <= self.client_training_concurrency <= self.num_clients, "Invalid configuration: 0 <= self.client_training_concurrency <= self.num_clients")
        _require(self.training_augmentation in {'none','cifar_crop_flip'}, "Invalid configuration: self.training_augmentation in {'none','cifar_crop_flip'}")
        if self.training_augmentation != 'none':
            _require(self.dataset == 'cifar100' and self.stream_protocol and (self.scenario == 'disjoint' or (self.joint_plan_path and self.scenario=='overlap')), "Invalid configuration: self.dataset == 'cifar100' and self.stream_protocol and (self.scenario == 'disjoint' or (self.joint_plan_path and self.scenario=='overlap'))")
        _require(self.classes_per_local_task >= 2 and self.local_tasks_per_round >= 1, "Invalid configuration: self.classes_per_local_task >= 2 and self.local_tasks_per_round >= 1")
        if self.joint_plan_path:
            _require(self.joint_control_kind in {'federated','local_only','centralized_sequential','simultaneous_access'}, "Invalid configuration: self.joint_control_kind in {'federated','local_only','centralized_sequential','simultaneous_access'}")
            _require(self.joint_domain_mode in {'none','client_fixed','client_periodic','client_corruption'}, "Invalid configuration: self.joint_domain_mode in {'none','client_fixed','client_periodic','client_corruption'}")
            _require(self.joint_checkpoint_policy in {'all','latest'}, "Invalid configuration: self.joint_checkpoint_policy in {'all','latest'}")
            _require(self.joint_epoch_segments>=1, "Invalid configuration: self.joint_epoch_segments>=1")
            if self.joint_epoch_segments>1:
                _require(self.joint_control_kind=='federated' and self.sample_overlap_fraction==0, "Invalid configuration: self.joint_control_kind=='federated' and self.sample_overlap_fraction==0")
                _require(self.local_tasks_per_round==1 and self.participation==1., "Invalid configuration: self.local_tasks_per_round==1 and self.participation==1.")
            if self.joint_control_kind=='centralized_sequential':_require(self.num_clients==1, "Invalid configuration: self.num_clients==1")
            if self.joint_control_kind=='simultaneous_access':
                _require(self.sample_overlap_fraction==0 and self.replay_update_policy=='sampled_mixed', "Invalid configuration: self.sample_overlap_fraction==0 and self.replay_update_policy=='sampled_mixed'")
            _require(self.joint_plan_hash and len(self.joint_plan_hash)==64, "Invalid configuration: self.joint_plan_hash and len(self.joint_plan_hash)==64")
            _require(self.stream_protocol and not self.stream_client_chain, "Invalid configuration: self.stream_protocol and not self.stream_client_chain")
            _require(not self.stream_sync_every_batch and self.stream_overlap_kind=='none', "Invalid configuration: not self.stream_sync_every_batch and self.stream_overlap_kind=='none'")
            _require(self.method in {'replay','sgd'} and self.aggregator=='fedavg', "Invalid configuration: self.method in {'replay','sgd'} and self.aggregator=='fedavg'")
            _require(self.selector=='random' and self.boundary_mode=='known', "Invalid configuration: self.selector=='random' and self.boundary_mode=='known'")
            _require(self.pending_policy=='latest' and not self.client_heterogeneity, "Invalid configuration: self.pending_policy=='latest' and not self.client_heterogeneity")
            _require(not self.class_order_override, "Invalid configuration: not self.class_order_override")
            _require(not self.class_sample_fraction and not self.client_sample_skew and not self.class_imbalance_power, "Invalid configuration: not self.class_sample_fraction and not self.client_sample_skew and not self.class_imbalance_power")
            _require(not self.label_noise_rate and not self.spurious_strength, "Invalid configuration: not self.label_noise_rate and not self.spurious_strength")
            _require(self.model in {'resnet18_cifar32','test_linear','test_mlp','test_smallcnn'}, "Invalid configuration: self.model in {'resnet18_cifar32','test_linear','test_mlp','test_smallcnn'}")
            _require(not self.gradient_diagnostics and not self.causal_diagnostics and not self.subset_diagnostics and not self.representation_diagnostics, "Invalid configuration: not self.gradient_diagnostics and not self.causal_diagnostics and not self.subset_diagnostics and not self.representation_diagnostics")
        else:
            _require(not self.joint_plan_hash and not self.joint_local_evaluation and not self.joint_allow_partial_coverage and self.joint_control_kind=='federated' and self.joint_domain_mode=='none' and self.joint_epoch_segments==1 and not self.joint_mechanistic_probes and self.joint_checkpoint_policy=='all', "Invalid configuration: not self.joint_plan_hash and not self.joint_local_evaluation and not self.joint_allow_partial_coverage and self.joint_control_kind=='federated' and self.joint_domain_mode=='none' and self.joint_epoch_segments==1 and not self.joint_mechanistic_probes and self.joint_checkpoint_policy=='all'")
        if self.stream_client_chain:
            _require(self.stream_protocol and self.class_pair_protocol == 'strict', "Invalid configuration: self.stream_protocol and self.class_pair_protocol == 'strict'")
            _require(self.num_clients == 5 and self.classes_per_local_task == 2, "Invalid configuration: self.num_clients == 5 and self.classes_per_local_task == 2")
            _require(self.local_tasks_per_round == 1 and self.stream_recurrence_rounds == 0, "Invalid configuration: self.local_tasks_per_round == 1 and self.stream_recurrence_rounds == 0")
            _require(self.stream_overlap_kind == 'none' and self.stream_overlap_fraction == 0, "Invalid configuration: self.stream_overlap_kind == 'none' and self.stream_overlap_fraction == 0")
            _require(self.sample_overlap_fraction == 0, "Invalid configuration: self.sample_overlap_fraction == 0")
            _require(self.stream_terminal_flush, 'Client-chain streams require a final archive/model flush')
        _require(self.stream_overlap_kind in {'none','intra_same_round','intra_cross_round','inter_same_round','inter_cross_round','combined'}, "Invalid configuration: self.stream_overlap_kind in {'none','intra_same_round','intra_cross_round','inter_same_round','inter_cross_round','combined'}")
        _require(0 <= self.stream_overlap_fraction <= 1, "Invalid configuration: 0 <= self.stream_overlap_fraction <= 1")
        _require(self.stream_recurrence_rounds >= 0 and self.offline_duration_rounds >= 0, "Invalid configuration: self.stream_recurrence_rounds >= 0 and self.offline_duration_rounds >= 0")
        _require(self.stream_offline_policy in {'random','rotate_once'}, "Invalid configuration: self.stream_offline_policy in {'random','rotate_once'}")
        _require(self.stream_partial_batch_policy in {'include_padding','mask_padding_loss'}, "Invalid configuration: self.stream_partial_batch_policy in {'include_padding','mask_padding_loss'}")
        if self.stream_partial_batch_policy == 'mask_padding_loss':
            if self.replay_ratio != 1.0:
                raise ValueError('Masked partial-batch replay currently supports only ratio 1.0')
            _require(self.stream_sync_every_batch and self.stream_client_chain and self.stream_protocol, "Invalid configuration: self.stream_sync_every_batch and self.stream_client_chain and self.stream_protocol")
            _require(self.method == 'replay' and self.replay_update_policy in {'sampled_mixed','sampled_mixed_accumulated'}, "Invalid configuration: self.method == 'replay' and self.replay_update_policy in {'sampled_mixed','sampled_mixed_accumulated'}")
        if not self.audit_test_every_sync and not self.evaluate_final_only:
            raise ValueError('Disabling per-sync test audit requires evaluate_final_only')
        if self.evaluate_final_only:
            _require(self.selector in {'random','fixed','round_robin'}, 'Test split cannot guide online client selection')
            _require(not self.prepost_diagnostics and not self.gradient_diagnostics, "Invalid configuration: not self.prepost_diagnostics and not self.gradient_diagnostics")
        if self.stream_protocol:
            _require(self.dataset in {'cifar100','synthetic'}, "Invalid configuration: self.dataset in {'cifar100','synthetic'}")
            _require(self.learning_setting == 'task_il' or (self.joint_plan_path and self.learning_setting=='class_il'), 'Legacy streams use task-specific heads')
            _require(self.class_pair_protocol == 'strict', "Invalid configuration: self.class_pair_protocol == 'strict'")
            _require(not self.full_sync and not self.reset_memory_on_sync, 'Offline clients must not receive server model weights or lose memory')
            # Canonical methods update all classifier rows with the same
            # aggregation rule. Classwise fusion is only for custom methods.
            _require(not self.class_subset, 'CIFAR-100 subset is insufficient for default 10-client curriculum')
            _require(not self.holders_override and not self.arrival_lag_by_client, "Invalid configuration: not self.holders_override and not self.arrival_lag_by_client")
            _require(not self.stage_participation and not self.participation_overrides, "Invalid configuration: not self.stage_participation and not self.participation_overrides")
            _require(not self.fork_source_job, 'Legacy fork checkpoints cannot be used with new streams')
            if self.offline_duration_rounds:_require(self.selector == 'random', "Invalid configuration: self.selector == 'random'")
            if self.stream_offline_policy == 'rotate_once':
                _require(self.offline_duration_rounds == 0 and self.selector == 'random', "Invalid configuration: self.offline_duration_rounds == 0 and self.selector == 'random'")
                _require((self.stream_overlap_kind == 'none' and not self.stream_client_chain and self.stream_recurrence_rounds == 0), "Invalid configuration: (self.stream_overlap_kind == 'none' and not self.stream_client_chain and self.stream_recurrence_rounds == 0)")
                _require(self.participation == 1 - 1/self.num_clients, "Invalid configuration: self.participation == 1 - 1/self.num_clients")
                _require(self.stream_terminal_flush, "Invalid configuration: self.stream_terminal_flush")
        _require(self.class_pair_protocol in {'compatible','strict'}, "Invalid configuration: self.class_pair_protocol in {'compatible','strict'}")
        _require(self.rounds_per_stage >= 1 and self.stages >= 1, "Invalid configuration: self.rounds_per_stage >= 1 and self.stages >= 1")
        _require(self.epochs >= 1 and self.lr > 0, "Invalid configuration: self.epochs >= 1 and self.lr > 0")
        _require(0 < self.participation <= 1, "Invalid configuration: 0 < self.participation <= 1")
        _require(self.selector in {"feature_max", "feature_min", "forgetting", "gem_informed", "history_adaptive", "gradient_align", "gradient_conflict", "random", "fixed", "round_robin", "coverage", "old", "new", "least_recent", "related", "recurrence", "overlap", "balanced", "diversity", "fairness", "stochastic_related", "historical_coverage"}, "Invalid configuration: self.selector in {\"feature_max\", \"feature_min\", \"forgetting\", \"gem_informed\", \"history_adaptive\", \"gradient_align\", \"gradient_conflict\", \"random\", \"fixed\", \"round_robin\", \"coverage\", \"old\", \"new\", \"least_recent\", \"related\", \"recurrence\", \"overlap\", \"balanced\", \"diversity\", \"fairness\", \"stochastic_related\", \"historical_coverage\"}")
        _require(self.method in {"si", "sgd", "replay", "agem", "gem", "ewc", "er_ace", "der", "derpp", "mas", "lwf", "native_glfc", "native_lga", "native_feat"}, "Invalid configuration: self.method in {\"si\", \"sgd\", \"replay\", \"agem\", \"gem\", \"ewc\", \"er_ace\", \"der\", \"derpp\", \"mas\", \"lwf\", \"native_glfc\", \"native_lga\", \"native_feat\"}")
        _require(self.aggregator in {"class_balanced", "inverse_participation", "conflict_aware", "retention_aware", "conflict_projection", "fedavg", "uniform", "fedadam", "fedavgm"}, "Invalid configuration: self.aggregator in {\"class_balanced\", \"inverse_participation\", \"conflict_aware\", \"retention_aware\", \"conflict_projection\", \"fedavg\", \"uniform\", \"fedadam\", \"fedavgm\"}")
        _require(self.scenario in {"stationary", "gradual_drift", "shifted_recurrence", "task_incremental", "disjoint", "iid", "overlap", "related", "recurrence", "asynchronous", "mixed", "domain", "non_iid"}, "Invalid configuration: self.scenario in {\"stationary\", \"gradual_drift\", \"shifted_recurrence\", \"task_incremental\", \"disjoint\", \"iid\", \"overlap\", \"related\", \"recurrence\", \"asynchronous\", \"mixed\", \"domain\", \"non_iid\"}")
        if self.dataset == 'synthetic':
            _require(self.model in {'test_linear','test_mlp','test_smallcnn','vit_base_patch16_32','resnet18_cifar32'}, "Invalid configuration: self.model in {'test_linear','test_mlp','test_smallcnn','vit_base_patch16_32','resnet18_cifar32'}")
        else:
            from .models import MODEL_IDS
            _require(self.model in MODEL_IDS, 'Research requires an explicitly registered native 32x32 research backbone')
        _require(self.memory_per_class >= 0, "Invalid configuration: self.memory_per_class >= 0")
        _require(0 <= self.replay_reservoir_fraction <= 1, 'Reservoir fraction must be within [0,1]')
        if self.replay_reservoir_fraction:
            _require(self.method in {'replay','native_glfc','native_lga','native_feat'}, 'Per-class reservoir fraction applies to replay-capable methods only')
        if self.dataset=='cifar100':
            _require(self.stream_protocol and self.class_pair_protocol=='strict', ( 'Every active CIFAR-100 job must use the strict class-pair stream'))
            if not self.joint_plan_path:_require(self.share_continual_state, 'Legacy CIFAR-100 requires synchronized CL state')
            _require(not self.reset_memory_on_sync, 'Synchronized history must remain available')
            _require(not (self.full_sync and self.participation<1), 'Hidden clients cannot receive server broadcasts')
        if self.stream_protocol and not self.joint_plan_path:_require(self.share_continual_state, 'Legacy streams synchronize method-specific CL state')
        if self.joint_archive_upload_policy not in {'selected','all'} or self.joint_archive_download_policy not in {'selected','all','none'}:
            raise ValueError('Unknown archive exchange policy')
        if (self.joint_archive_upload_policy,self.joint_archive_download_policy)!=('selected','selected'):
            if not self.joint_plan_path or not self.share_continual_state or self.method!='replay' or self.joint_control_kind!='federated':
                raise ValueError('Independent archive exchange requires federated joint shared Replay')
        if self.server_memory_per_class<0:raise ValueError('Negative server replay capacity')
        if self.server_memory_per_class and not self.share_continual_state:raise ValueError('Server replay cap requires a shared archive')
        if self.stale_update_policy not in {'damped_delta','undamped_delta','direct_model'}:raise ValueError('Unknown stale update policy')
        if self.task_head_weight_policy not in {'historical_exposure','measured_exposure','equal_contributors'}:raise ValueError('Unknown task-head weight policy')
        if self.task_head_weight_policy!='historical_exposure' and (not self.joint_plan_path or self.learning_setting!='task_il' or self.aggregator!='fedavg'):
            raise ValueError('Head weighting controls require joint Task-IL FedAvg')
        if self.aggregation_weight_policy not in {'fresh_examples','optimizer_steps'}:raise ValueError('Unknown aggregation weight policy')
        if self.aggregation_weight_policy!='fresh_examples' and (not self.joint_plan_path or self.aggregator!='fedavg'):raise ValueError('Optimizer-step weighting is a joint FedAvg control')
        _require(self.causal_max_clients >= 1 and self.checkpoint_every >= 0, "Invalid configuration: self.causal_max_clients >= 1 and self.checkpoint_every >= 0")
        _require(self.participation_schedule in {"feedback", "stage", "constant", "increasing", "decreasing", "alternating"}, "Invalid configuration: self.participation_schedule in {\"feedback\", \"stage\", \"constant\", \"increasing\", \"decreasing\", \"alternating\"}")
        _require(self.pending_policy in {"latest", "retain_all", "bounded"}, "Invalid configuration: self.pending_policy in {\"latest\", \"retain_all\", \"bounded\"}")
        _require(self.max_pending_samples >= 0, "Invalid configuration: self.max_pending_samples >= 0")
        _require(self.replay_ratio >= 0 and self.der_alpha >= 0 and self.der_beta >= 0, "Invalid configuration: self.replay_ratio >= 0 and self.der_alpha >= 0 and self.der_beta >= 0")
        _require(self.replay_update_policy in {"exhaustive", "sampled_mixed", "sampled_mixed_accumulated", "sampled_separate"}, "Invalid configuration: self.replay_update_policy in {\"exhaustive\", \"sampled_mixed\", \"sampled_mixed_accumulated\", \"sampled_separate\"}")
        _require(self.replay_bn_policy in {"batch_update", "freeze_replay_buffers"}, "Invalid configuration: self.replay_bn_policy in {\"batch_update\", \"freeze_replay_buffers\"}")
        _require(self.replay_buffer_order_policy in {"archive_order", "shuffled_each_pass"}, "Invalid configuration: self.replay_buffer_order_policy in {\"archive_order\", \"shuffled_each_pass\"}")
        _require(self.replay_padding_policy in {"repeat_chunk", "archive_sample"}, "Invalid configuration: self.replay_padding_policy in {\"repeat_chunk\", \"archive_sample\"}")
        _require(0 <= self.label_noise_rate <= 1 and self.spurious_strength >= 0, "Invalid configuration: 0 <= self.label_noise_rate <= 1 and self.spurious_strength >= 0")
        _require(0 <= self.feedback_threshold <= 1 and self.feedback_gain >= 0, "Invalid configuration: 0 <= self.feedback_threshold <= 1 and self.feedback_gain >= 0")
        if self.label_noise_rate:
            # No privileged clean-label memory, class-balanced server weights
            # or simulator label-aware client selection in the noise baseline.
            _require(self.method=='sgd' and self.selector=='random', "Invalid configuration: self.method=='sgd' and self.selector=='random'")
            _require(self.aggregator=='fedavg', "Invalid configuration: self.aggregator=='fedavg'")
        _require(self.importance_samples_per_experience >= 1, "Invalid configuration: self.importance_samples_per_experience >= 1")
        _require(self.regularization_strength >= 0 and self.fedprox_mu >= 0, "Invalid configuration: self.regularization_strength >= 0 and self.fedprox_mu >= 0")
        _require(self.server_lr > 0 and 0 <= self.server_beta1 < 1 and 0 <= self.server_beta2 < 1, "Invalid configuration: self.server_lr > 0 and 0 <= self.server_beta1 < 1 and 0 <= self.server_beta2 < 1")
        _require(self.fedadam_epsilon > 0, "Invalid configuration: self.fedadam_epsilon > 0")
        _require(self.memory_policy in {"random_per_class", "recent_per_class"}, "Invalid configuration: self.memory_policy in {\"random_per_class\", \"recent_per_class\"}")
        _require(self.local_optimizer in {"sgd"}, "Invalid configuration: self.local_optimizer in {\"sgd\"}")
        _require(self.optimizer_state_policy in {"reset", "persist", "reset_on_sync", "reset_on_task"}, "Invalid configuration: self.optimizer_state_policy in {\"reset\", \"persist\", \"reset_on_sync\", \"reset_on_task\"}")
        if self.optimizer_state_policy in {'reset_on_sync','reset_on_task'} and not self.joint_plan_path:
            raise ValueError('New optimizer-reset controls require joint protocol')
        _require(self.boundary_mode in {"known", "hidden"}, "Invalid configuration: self.boundary_mode in {\"known\", \"hidden\"}")
        _require(self.learning_setting == 'task_il' or (self.joint_plan_path and self.learning_setting=='class_il'), 'Class-IL requires the explicit joint protocol')
        if self.learning_setting=='task_il':
            _require(self.class_pair_protocol=='strict', "Invalid configuration: self.class_pair_protocol=='strict'")
            _require(self.boundary_mode=='known', "Invalid configuration: self.boundary_mode=='known'")
            if self.method in {'native_glfc','native_lga','native_feat'}:
                raise ValueError(
                    f'{self.method} has no task-head adaptation; refusing to report the generic Task-IL fallback as that method')
        if self.learning_setting=='domain_il':
            _require(self.scenario in {"domain","gradual_drift"}, "Invalid configuration: self.scenario in {\"domain\",\"gradual_drift\"}")
        if self.learning_setting=='boundary_free':
            _require(self.boundary_mode=='hidden', "Invalid configuration: self.boundary_mode=='hidden'")
        if self.boundary_mode=='hidden':
            # Stage-dependent consolidation would leak hidden task boundaries.
            _require(self.method in {"sgd","replay","gem","agem","er_ace","der","derpp","native_glfc","native_lga","native_feat"}, "Invalid configuration: self.method in {\"sgd\",\"replay\",\"gem\",\"agem\",\"er_ace\",\"der\",\"derpp\",\"native_glfc\",\"native_lga\",\"native_feat\"}")
            _require(self.gem_grouping!='stage', "Invalid configuration: self.gem_grouping!='stage'")
        _require(self.gem_grouping in {"class", "stage", "all"}, "Invalid configuration: self.gem_grouping in {\"class\", \"stage\", \"all\"}")
        _require(0 <= self.momentum < 1 and self.weight_decay >= 0, "Invalid configuration: 0 <= self.momentum < 1 and self.weight_decay >= 0")
        _require(self.si_xi > 0 and self.selection_temperature > 0, "Invalid configuration: self.si_xi > 0 and self.selection_temperature > 0")
        _require(self.relation in {"natural", "related", "unrelated"}, "Invalid configuration: self.relation in {\"natural\", \"related\", \"unrelated\"}")
        _require(self.recurrence_client_policy in {"same", "different"}, "Invalid configuration: self.recurrence_client_policy in {\"same\", \"different\"}")
        _require(self.recurrence_round_delay >= 0 and self.recurrence_period >= 0 and self.recurrence_repeats >= 1, "Invalid configuration: self.recurrence_round_delay >= 0 and self.recurrence_period >= 0 and self.recurrence_repeats >= 1")
        _require(0 <= self.sample_overlap_fraction <= 1, "Invalid configuration: 0 <= self.sample_overlap_fraction <= 1")
        _require(self.client_sample_skew >= 0 and self.class_imbalance_power >= 0, "Invalid configuration: self.client_sample_skew >= 0 and self.class_imbalance_power >= 0")
        _require(not self.stage_participation or (len(self.stage_participation) == self.stages and all(0 < rate <= 1 for rate in self.stage_participation)), "Invalid configuration: not self.stage_participation or (len(self.stage_participation) == self.stages and all(0 < rate <= 1 for rate in self.stage_participation))")
        _require(len(set(self.class_subset)) == len(self.class_subset), "Invalid configuration: len(set(self.class_subset)) == len(self.class_subset)")
        _require(len(set(self.class_order_override)) == len(self.class_order_override), "Invalid configuration: len(set(self.class_order_override)) == len(self.class_order_override)")
        for cls,clients in self.holders_override.items():
            _require(str(int(cls)) == str(cls) and clients, "Invalid configuration: str(int(cls)) == str(cls) and clients")
            _require(len(set(clients)) == len(clients), "Invalid configuration: len(set(clients)) == len(clients)")
            _require(all(0 <= c < self.num_clients for c in clients), "Invalid configuration: all(0 <= c < self.num_clients for c in clients)")
        for cls,fraction in self.class_sample_fraction.items():
            _require(str(int(cls)) == str(cls) and 0 < fraction <= 1, "Invalid configuration: str(int(cls)) == str(cls) and 0 < fraction <= 1")
        for cls,weights in self.class_holder_weights.items():
            _require(str(int(cls)) == str(cls) and weights, "Invalid configuration: str(int(cls)) == str(cls) and weights")
            _require(all(w >= 0 for w in weights) and sum(weights)>0, "Invalid configuration: all(w >= 0 for w in weights) and sum(weights)>0")
            if cls in self.holders_override:
                _require(len(weights)==len(self.holders_override[cls]), "Invalid configuration: len(weights)==len(self.holders_override[cls])")
        for rnd,chosen in self.participation_overrides.items():
            _require(str(int(rnd)) == str(rnd) and int(rnd)>=0, "Invalid configuration: str(int(rnd)) == str(rnd) and int(rnd)>=0")
            _require(len(set(chosen))==len(chosen), "Invalid configuration: len(set(chosen))==len(chosen)")
            _require(all(0 <= cid < self.num_clients for cid in chosen), "Invalid configuration: all(0 <= cid < self.num_clients for cid in chosen)")
        for cid,lag in self.arrival_lag_by_client.items():
            _require(str(int(cid))==str(cid) and 0<=int(cid)<self.num_clients, "Invalid configuration: str(int(cid))==str(cid) and 0<=int(cid)<self.num_clients")
            _require(isinstance(lag,int) and lag>=0, "Invalid configuration: isinstance(lag,int) and lag>=0")
        _require(self.force_target in {"none", "include", "exclude"}, "Invalid configuration: self.force_target in {\"none\", \"include\", \"exclude\"}")
        _require(self.intervention_round >= -1, "Invalid configuration: self.intervention_round >= -1")
        _require(0 <= self.overlap <= 1, "Invalid configuration: 0 <= self.overlap <= 1")
        _require(self.dirichlet_alpha > 0, "Invalid configuration: self.dirichlet_alpha > 0")
        if self.fork_source_job:
            _require(self.job_id and self.job_id != self.fork_source_job, "Invalid configuration: self.job_id and self.job_id != self.fork_source_job")
            _require(self.fork_checkpoint_round >= 1, "Invalid configuration: self.fork_checkpoint_round >= 1")
            _require(self.force_target != 'none', "Invalid configuration: self.force_target != 'none'")
            _require(self.intervention_round >= self.fork_checkpoint_round, "Invalid configuration: self.intervention_round >= self.fork_checkpoint_round")
            _require(self.intervention_round < self.stages*self.rounds_per_stage, "Invalid configuration: self.intervention_round < self.stages*self.rounds_per_stage")
        if self.force_target != "none":
            _require(0 <= self.target_client < self.num_clients, "Invalid configuration: 0 <= self.target_client < self.num_clients")
            _require(round(self.num_clients * self.participation) < self.num_clients or self.force_target == "include", "Invalid configuration: round(self.num_clients * self.participation) < self.num_clients or self.force_target == \"include\"")
        if self.dataset == "cifar100_20":
            _require(self.stages <= 20 and not self.class_subset, "Invalid configuration: self.stages <= 20 and not self.class_subset")
        if self.class_subset:
            _require(self.dataset == "cifar100" and len(self.class_subset) >= self.stages, "Invalid configuration: self.dataset == \"cifar100\" and len(self.class_subset) >= self.stages")
            _require(all(0 <= x < 100 for x in self.class_subset), "Invalid configuration: all(0 <= x < 100 for x in self.class_subset)")
        if self.dataset == "cifar10":
            _require(self.stages <= 10, "Invalid configuration: self.stages <= 10")
        if self.dataset == "cifar100":
            _require(self.stages <= (len(self.class_subset) or 100), "Invalid configuration: self.stages <= (len(self.class_subset) or 100)")
        return self

    def to_dict(self):
        data=asdict(self)
        if data['audit_test_every_sync'] is True:data.pop('audit_test_every_sync')
        if data['joint_archive_upload_policy']=='selected':data.pop('joint_archive_upload_policy')
        if data['joint_archive_download_policy']=='selected':data.pop('joint_archive_download_policy')
        if data['task_head_weight_policy']=='historical_exposure':data.pop('task_head_weight_policy')
        if data['physical_batch_policy']=='fixed_pad':data.pop('physical_batch_policy')
        if data['replay_class_sampling_policy']=='distinct_first':data.pop('replay_class_sampling_policy')
        for key,default in (('server_memory_per_class',0),('stale_update_policy','damped_delta'),('terminal_stale_guard',True),('aggregation_weight_policy','fresh_examples')):
            if data[key]==default:data.pop(key)
        # Preserve old plan-file serialization for unchanged job configurations.
        # Every nondefault protocol variant is still explicitly serialized.
        if data['class_pair_protocol']=='compatible':data.pop('class_pair_protocol')
        if data['classwise_head_merge'] is True:data.pop('classwise_head_merge')
        # Preserve the frozen v6 geometry ledger. Execution semantics and roots are v7.
        if data['importance_samples_per_experience']==32:data.pop('importance_samples_per_experience')
        if data['replay_reservoir_fraction']==0.0:data.pop('replay_reservoir_fraction')
        if data['share_continual_state'] is True:data.pop('share_continual_state')
        if data['client_training_concurrency']==0:data.pop('client_training_concurrency')
        if data['training_augmentation']=='none':data.pop('training_augmentation')
        if not data['joint_plan_path']:
            for key in ('joint_plan_path','joint_plan_hash','joint_local_evaluation','joint_allow_partial_coverage','joint_control_kind','joint_domain_mode','joint_epoch_segments','joint_mechanistic_probes','joint_checkpoint_policy'):
                data.pop(key)
        return data

    def signature(self):
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
