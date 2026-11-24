"""Strict experiment configurations, independently serializable per job."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json

@dataclass(frozen=True)
class RunConfig:
    dataset: str = "cifar100"
    batch_size: int = 64
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
    # Opt-in joint protocol. Empty values preserve legacy signatures.
    joint_plan_path: str = ''
    joint_plan_hash: str = ''
    joint_local_evaluation: bool = False
    joint_allow_partial_coverage: bool = False
    joint_control_kind: str = 'federated'

    def validate(self):
        assert self.dataset in {"cifar10", "cifar100", "cifar100_20", "synthetic"}, self.dataset
        assert self.batch_size == 64 or self.dataset == "synthetic", "All production optimization must use physical batch 64"
        assert self.num_clients >= (1 if self.joint_plan_path and self.joint_control_kind=='centralized_sequential' else 2)
        assert 0 <= self.client_training_concurrency <= self.num_clients
        assert self.training_augmentation in {'none','cifar_crop_flip'}
        if self.training_augmentation != 'none':
            assert self.dataset == 'cifar100' and self.stream_protocol and (self.scenario == 'disjoint' or (self.joint_plan_path and self.scenario=='overlap'))
        assert self.classes_per_local_task >= 2 and self.local_tasks_per_round >= 1
        if self.joint_plan_path:
            assert self.joint_control_kind in {'federated','local_only','centralized_sequential','simultaneous_access'}
            if self.joint_control_kind=='centralized_sequential':assert self.num_clients==1
            if self.joint_control_kind=='simultaneous_access':
                assert self.sample_overlap_fraction==0 and self.replay_update_policy=='sampled_mixed'
            assert self.joint_plan_hash and len(self.joint_plan_hash)==64
            assert self.stream_protocol and not self.stream_client_chain
            assert not self.stream_sync_every_batch and self.stream_overlap_kind=='none'
            assert self.method in {'replay','sgd'} and self.aggregator=='fedavg'
            assert self.selector=='random' and self.boundary_mode=='known'
            assert self.pending_policy=='latest' and not self.client_heterogeneity
            assert not self.class_order_override
            assert not self.class_sample_fraction and not self.client_sample_skew and not self.class_imbalance_power
            assert not self.label_noise_rate and not self.spurious_strength
            assert self.model in {'resnet18_cifar32','test_linear','test_mlp','test_smallcnn'}
            assert not self.gradient_diagnostics and not self.causal_diagnostics and not self.subset_diagnostics and not self.representation_diagnostics
        else:
            assert not self.joint_plan_hash and not self.joint_local_evaluation and not self.joint_allow_partial_coverage and self.joint_control_kind=='federated'
        if self.stream_client_chain:
            assert self.stream_protocol and self.class_pair_protocol == 'strict'
            assert self.num_clients == 5 and self.classes_per_local_task == 2
            assert self.local_tasks_per_round == 1 and self.stream_recurrence_rounds == 0
            assert self.stream_overlap_kind == 'none' and self.stream_overlap_fraction == 0
            assert self.sample_overlap_fraction == 0
            assert self.stream_terminal_flush, 'Client-chain streams require a final archive/model flush'
        assert self.stream_overlap_kind in {'none','intra_same_round','intra_cross_round','inter_same_round','inter_cross_round','combined'}
        assert 0 <= self.stream_overlap_fraction <= 1
        assert self.stream_recurrence_rounds >= 0 and self.offline_duration_rounds >= 0
        assert self.stream_offline_policy in {'random','rotate_once'}
        assert self.stream_partial_batch_policy in {'include_padding','mask_padding_loss'}
        if self.stream_partial_batch_policy == 'mask_padding_loss':
            assert self.stream_sync_every_batch and self.stream_client_chain and self.stream_protocol
            assert self.method == 'replay' and self.replay_update_policy in {'sampled_mixed','sampled_mixed_accumulated'}
        if self.evaluate_final_only:
            assert self.selector in {'random','fixed','round_robin'}, 'Test split cannot guide online client selection'
            assert not self.prepost_diagnostics and not self.gradient_diagnostics
        if self.stream_protocol:
            assert self.dataset in {'cifar100','synthetic'}
            assert self.learning_setting == 'task_il' or (self.joint_plan_path and self.learning_setting=='class_il'), 'Legacy streams use task-specific heads'
            assert self.class_pair_protocol == 'strict'
            assert not self.full_sync and not self.reset_memory_on_sync, 'Offline clients must not receive server model weights or lose memory'
            # Canonical methods update all classifier rows with the same
            # aggregation rule. Classwise fusion is only for custom methods.
            assert not self.class_subset, 'CIFAR-100 subset is insufficient for default 10-client curriculum'
            assert not self.holders_override and not self.arrival_lag_by_client
            assert not self.stage_participation and not self.participation_overrides
            assert not self.fork_source_job, 'Legacy fork checkpoints cannot be used with new streams'
            if self.offline_duration_rounds:assert self.selector == 'random'
            if self.stream_offline_policy == 'rotate_once':
                assert self.offline_duration_rounds == 0 and self.selector == 'random'
                assert (self.stream_overlap_kind == 'none' and not self.stream_client_chain
                        and self.stream_recurrence_rounds == 0)
                assert self.participation == 1 - 1/self.num_clients
                assert self.stream_terminal_flush
        assert self.class_pair_protocol in {'compatible','strict'}
        assert self.rounds_per_stage >= 1 and self.stages >= 1
        assert self.epochs >= 1 and self.lr > 0
        assert 0 < self.participation <= 1
        assert self.selector in {"feature_max", "feature_min", "forgetting", "gem_informed", "history_adaptive", "gradient_align", "gradient_conflict", "random", "fixed", "round_robin", "coverage", "old", "new", "least_recent", "related", "recurrence", "overlap", "balanced", "diversity", "fairness", "stochastic_related", "historical_coverage"}
        assert self.method in {"si", "sgd", "replay", "agem", "gem", "ewc", "er_ace", "der", "derpp", "mas", "lwf", "native_glfc", "native_lga", "native_feat"}
        assert self.aggregator in {"class_balanced", "inverse_participation", "conflict_aware", "retention_aware", "conflict_projection", "fedavg", "uniform", "fedadam", "fedavgm"}
        assert self.scenario in {"stationary", "gradual_drift", "shifted_recurrence", "task_incremental", "disjoint", "iid", "overlap", "related", "recurrence", "asynchronous", "mixed", "domain", "non_iid"}
        if self.dataset == 'synthetic':
            assert self.model in {'test_linear','test_mlp','test_smallcnn','vit_base_patch16_32','resnet18_cifar32'}
        else:
            from .models import MODEL_IDS
            assert self.model in MODEL_IDS, 'Research requires an explicitly registered native 32x32 research backbone'
        assert self.memory_per_class >= 0
        assert 0 <= self.replay_reservoir_fraction <= 1, 'Reservoir fraction must be within [0,1]'
        if self.replay_reservoir_fraction:
            assert self.method in {'replay','native_glfc','native_lga','native_feat'}, 'Per-class reservoir fraction applies to replay-capable methods only'
        if self.dataset=='cifar100':
            assert self.stream_protocol and self.class_pair_protocol=='strict', (
                'Every active CIFAR-100 job must use the strict class-pair stream')
            assert self.share_continual_state, 'All CIFAR-100 jobs require synchronized CL state'
            assert not self.reset_memory_on_sync, 'Synchronized history must remain available'
            assert not (self.full_sync and self.participation<1), 'Hidden clients cannot receive server broadcasts'
        if self.stream_protocol:assert self.share_continual_state, 'Streaming jobs must synchronize method-specific CL state'
        assert self.causal_max_clients >= 1 and self.checkpoint_every >= 0
        assert self.participation_schedule in {"feedback", "stage", "constant", "increasing", "decreasing", "alternating"}
        assert self.pending_policy in {"latest", "retain_all", "bounded"}
        assert self.max_pending_samples >= 0
        assert self.replay_ratio >= 0 and self.der_alpha >= 0 and self.der_beta >= 0
        assert self.replay_update_policy in {"exhaustive", "sampled_mixed", "sampled_mixed_accumulated", "sampled_separate"}
        assert self.replay_bn_policy in {"batch_update", "freeze_replay_buffers"}
        assert self.replay_buffer_order_policy in {"archive_order", "shuffled_each_pass"}
        assert self.replay_padding_policy in {"repeat_chunk", "archive_sample"}
        assert 0 <= self.label_noise_rate <= 1 and self.spurious_strength >= 0
        assert 0 <= self.feedback_threshold <= 1 and self.feedback_gain >= 0
        if self.label_noise_rate:
            # No privileged clean-label memory, class-balanced server weights
            # or simulator label-aware client selection in the noise baseline.
            assert self.method=='sgd' and self.selector=='random'
            assert self.aggregator=='fedavg' 
        assert self.importance_samples_per_experience >= 1
        assert self.regularization_strength >= 0 and self.fedprox_mu >= 0
        assert self.server_lr > 0 and 0 <= self.server_beta1 < 1 and 0 <= self.server_beta2 < 1
        assert self.fedadam_epsilon > 0
        assert self.memory_policy in {"random_per_class", "recent_per_class"}
        assert self.local_optimizer in {"sgd"}
        assert self.optimizer_state_policy in {"reset", "persist"}
        assert self.boundary_mode in {"known", "hidden"}
        assert self.learning_setting == 'task_il' or (self.joint_plan_path and self.learning_setting=='class_il'), 'Class-IL requires the explicit joint protocol'
        if self.learning_setting=='task_il':
            assert self.class_pair_protocol=='strict'
            assert self.boundary_mode=='known'
            if self.method in {'native_glfc','native_lga','native_feat'}:
                raise ValueError(
                    f'{self.method} has no task-head adaptation; refusing to report the generic Task-IL fallback as that method')
            if self.method in {'der','derpp'} and self.sample_overlap_fraction>0:
                raise ValueError('Task-IL DER/DER++ cannot disambiguate one sample reused by multiple source task heads')
        if self.learning_setting=='domain_il':
            assert self.scenario in {"domain","gradual_drift"}
        if self.learning_setting=='boundary_free':
            assert self.boundary_mode=='hidden' 
        if self.boundary_mode=='hidden':
            # Stage-dependent consolidation would leak hidden task boundaries.
            assert self.method in {"sgd","replay","gem","agem","er_ace","der","derpp","native_glfc","native_lga","native_feat"}
            assert self.gem_grouping!='stage' 
        assert self.gem_grouping in {"class", "stage", "all"}
        assert 0 <= self.momentum < 1 and self.weight_decay >= 0
        assert self.si_xi > 0 and self.selection_temperature > 0
        assert self.relation in {"natural", "related", "unrelated"}
        assert self.recurrence_client_policy in {"same", "different"}
        assert self.recurrence_round_delay >= 0 and self.recurrence_period >= 0 and self.recurrence_repeats >= 1
        assert 0 <= self.sample_overlap_fraction <= 1
        assert self.client_sample_skew >= 0 and self.class_imbalance_power >= 0
        assert not self.stage_participation or (len(self.stage_participation) == self.stages and
                all(0 < rate <= 1 for rate in self.stage_participation))
        assert len(set(self.class_subset)) == len(self.class_subset)
        assert len(set(self.class_order_override)) == len(self.class_order_override)
        for cls,clients in self.holders_override.items():
            assert str(int(cls)) == str(cls) and clients
            assert len(set(clients)) == len(clients)
            assert all(0 <= c < self.num_clients for c in clients)
        for cls,fraction in self.class_sample_fraction.items():
            assert str(int(cls)) == str(cls) and 0 < fraction <= 1
        for cls,weights in self.class_holder_weights.items():
            assert str(int(cls)) == str(cls) and weights
            assert all(w >= 0 for w in weights) and sum(weights)>0
            if cls in self.holders_override:
                assert len(weights)==len(self.holders_override[cls])
        for rnd,chosen in self.participation_overrides.items():
            assert str(int(rnd)) == str(rnd) and int(rnd)>=0
            assert len(set(chosen))==len(chosen)
            assert all(0 <= cid < self.num_clients for cid in chosen)
        for cid,lag in self.arrival_lag_by_client.items():
            assert str(int(cid))==str(cid) and 0<=int(cid)<self.num_clients
            assert isinstance(lag,int) and lag>=0
        assert self.force_target in {"none", "include", "exclude"}
        assert self.intervention_round >= -1
        assert 0 <= self.overlap <= 1
        assert self.dirichlet_alpha > 0
        if self.fork_source_job:
            assert self.job_id and self.job_id != self.fork_source_job
            assert self.fork_checkpoint_round >= 1
            assert self.force_target != 'none'
            assert self.intervention_round >= self.fork_checkpoint_round
            assert self.intervention_round < self.stages*self.rounds_per_stage
        if self.force_target != "none":
            assert 0 <= self.target_client < self.num_clients
            assert round(self.num_clients * self.participation) < self.num_clients or self.force_target == "include"
        if self.dataset == "cifar100_20":
            assert self.stages <= 20 and not self.class_subset
        if self.class_subset:
            assert self.dataset == "cifar100" and len(self.class_subset) >= self.stages
            assert all(0 <= x < 100 for x in self.class_subset)
        if self.dataset == "cifar10":
            assert self.stages <= 10
        if self.dataset == "cifar100":
            assert self.stages <= (len(self.class_subset) or 100)
        return self

    def to_dict(self):
        data=asdict(self)
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
            for key in ('joint_plan_path','joint_plan_hash','joint_local_evaluation','joint_allow_partial_coverage','joint_control_kind'):
                data.pop(key)
        return data

    def signature(self):
        raw = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
