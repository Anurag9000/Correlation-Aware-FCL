"""Strict experiment configurations, independently serializable per job."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json

@dataclass(frozen=True)
class RunConfig:
    dataset: str = "cifar10"
    batch_size: int = 512
    num_clients: int = 10
    rounds_per_stage: int = 2
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
    gradient_diagnostics: bool = False
    causal_diagnostics: bool = False
    causal_max_clients: int = 6
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
    job_id: str = ""
    block_id: str = ""

    def validate(self):
        assert self.dataset in {"cifar10", "cifar100", "synthetic"}, self.dataset
        assert self.batch_size == 512 or self.dataset == "synthetic", "CIFAR run batch size must remain 512"
        assert self.num_clients >= 2
        assert self.rounds_per_stage >= 1 and self.stages >= 1
        assert self.epochs >= 1 and self.lr > 0
        assert 0 < self.participation <= 1
        assert self.selector in {"random", "fixed", "round_robin", "coverage", "old", "new", "least_recent", "related", "recurrence", "overlap", "balanced", "diversity", "fairness", "stochastic_related", "historical_coverage"}
        assert self.method in {"sgd", "replay", "agem", "gem", "ewc", "er_ace", "der", "derpp", "mas", "lwf"}
        assert self.aggregator in {"fedavg", "uniform", "fedadam", "fedavgm"}
        assert self.scenario in {"disjoint", "iid", "overlap", "related", "recurrence", "asynchronous", "mixed", "domain", "non_iid"}
        assert self.model in {"smallcnn", "linear", "mlp"}
        assert self.memory_per_class >= 0
        assert self.causal_max_clients >= 1 and self.checkpoint_every >= 0
        assert self.participation_schedule in {"constant", "increasing", "decreasing", "alternating"}
        assert self.pending_policy in {"latest", "retain_all", "bounded"}
        assert self.max_pending_samples >= 0
        assert self.replay_ratio >= 0 and self.der_alpha >= 0 and self.der_beta >= 0
        assert self.regularization_strength >= 0 and self.fedprox_mu >= 0
        assert self.server_lr > 0 and 0 <= self.server_beta1 < 1 and 0 <= self.server_beta2 < 1
        assert self.fedadam_epsilon > 0
        assert self.memory_policy in {"random_per_class", "recent_per_class"}
        assert self.local_optimizer in {"sgd"}
        assert self.relation in {"natural", "related", "unrelated"}
        assert self.recurrence_client_policy in {"same", "different"}
        assert self.force_target in {"none", "include", "exclude"}
        assert 0 <= self.overlap <= 1
        assert self.dirichlet_alpha > 0
        if self.force_target != "none":
            assert 0 <= self.target_client < self.num_clients
            assert round(self.num_clients * self.participation) < self.num_clients or self.force_target == "include"
        if self.dataset == "cifar10":
            assert self.stages <= 10
        if self.dataset == "cifar100":
            assert self.stages <= 100
        return self

    def to_dict(self):
        return asdict(self)

    def signature(self):
        raw = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
