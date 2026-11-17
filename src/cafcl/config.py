"""Strict experiment configurations, independently serializable per job."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import hashlib
import json

@dataclass(frozen=True)
class RunConfig:
    dataset: str = "cifar10"
    batch_size: int = 512
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
    lr: float = 0.03
    model: str = "smallcnn"
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
    optimizer_state_policy: str = "reset"
    momentum: float = 0.0
    si_xi: float = 0.1
    # Additional supervised selection and aggregation variants are privileged
    # simulator controls, not deployable privacy-preserving policies.
    selection_temperature: float = 1.0
    boundary_mode: str = "known"  # hidden = class-aware, task-boundary-free
    learning_setting: str = "class_il"
    label_noise_rate: float = 0.0
    spurious_strength: float = 0.0
    spurious_ood_eval: bool = False
    reset_memory_on_sync: bool = False
    feedback_threshold: float = 0.25
    feedback_gain: float = 0.5

    def validate(self):
        assert self.dataset in {"cifar10", "cifar100", "cifar100_20", "synthetic"}, self.dataset
        assert self.batch_size == 512 or self.dataset == "synthetic", "CIFAR run batch size must remain 512"
        assert self.num_clients >= 2
        assert self.rounds_per_stage >= 1 and self.stages >= 1
        assert self.epochs >= 1 and self.lr > 0
        assert 0 < self.participation <= 1
        assert self.selector in {"feature_max", "feature_min", "forgetting", "gem_informed", "history_adaptive", "gradient_align", "gradient_conflict", "random", "fixed", "round_robin", "coverage", "old", "new", "least_recent", "related", "recurrence", "overlap", "balanced", "diversity", "fairness", "stochastic_related", "historical_coverage"}
        assert self.method in {"si", "sgd", "replay", "agem", "gem", "ewc", "er_ace", "der", "derpp", "mas", "lwf"}
        assert self.aggregator in {"class_balanced", "inverse_participation", "conflict_aware", "retention_aware", "conflict_projection", "fedavg", "uniform", "fedadam", "fedavgm"}
        assert self.scenario in {"stationary", "gradual_drift", "shifted_recurrence", "task_incremental", "disjoint", "iid", "overlap", "related", "recurrence", "asynchronous", "mixed", "domain", "non_iid"}
        assert self.model in {"smallcnn", "linear", "mlp"}
        assert self.memory_per_class >= 0
        assert self.causal_max_clients >= 1 and self.checkpoint_every >= 0
        assert self.participation_schedule in {"feedback", "stage", "constant", "increasing", "decreasing", "alternating"}
        assert self.pending_policy in {"latest", "retain_all", "bounded"}
        assert self.max_pending_samples >= 0
        assert self.replay_ratio >= 0 and self.der_alpha >= 0 and self.der_beta >= 0
        assert 0 <= self.label_noise_rate <= 1 and self.spurious_strength >= 0
        assert 0 <= self.feedback_threshold <= 1 and self.feedback_gain >= 0
        if self.label_noise_rate:
            # No privileged clean-label memory, class-balanced server weights
            # or simulator label-aware client selection in the noise baseline.
            assert self.method=='sgd' and self.selector=='random'
            assert self.aggregator=='fedavg'
        assert self.regularization_strength >= 0 and self.fedprox_mu >= 0
        assert self.server_lr > 0 and 0 <= self.server_beta1 < 1 and 0 <= self.server_beta2 < 1
        assert self.fedadam_epsilon > 0
        assert self.memory_policy in {"random_per_class", "recent_per_class"}
        assert self.local_optimizer in {"sgd"}
        assert self.optimizer_state_policy in {"reset", "persist"}
        assert self.boundary_mode in {"known", "hidden"}
        assert self.learning_setting in {"class_il","task_il","domain_il","boundary_free"}
        if self.learning_setting=='task_il':
            assert self.scenario in {"disjoint", "related", "task_incremental"}
            assert self.pending_policy=='latest' and self.boundary_mode=='known'
        if self.learning_setting=='domain_il':
            assert self.scenario in {"domain","gradual_drift"}
        if self.learning_setting=='boundary_free':
            assert self.boundary_mode=='hidden'
        if self.boundary_mode=='hidden':
            # Stage-dependent consolidation would leak hidden task boundaries.
            assert self.method in {"sgd","replay","gem","agem","er_ace","der","derpp"}
            assert self.gem_grouping!='stage'
        assert self.gem_grouping in {"class", "stage", "all"}
        assert 0 <= self.momentum < 1 and self.si_xi > 0 and self.selection_temperature > 0
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
        return asdict(self)

    def signature(self):
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
