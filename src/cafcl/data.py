"""Dataset loading and immutable, disjoint data allocation for CIFAR FCL."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_MEAN = (.4914, .4822, .4465)
CIFAR10_STD = (.2470, .2435, .2616)
CIFAR100_MEAN = (.5071, .4867, .4408)
CIFAR100_STD = (.2675, .2565, .2761)

class TinySynthetic(Dataset):
    """Offline smoke-test dataset with 10 labels and 3x32x32 inputs."""
    def __init__(self, train=True, seed=0):
        rng = np.random.default_rng(seed + (0 if train else 10000))
        count = 12 if train else 6
        self.targets = np.repeat(np.arange(10), count).astype(int).tolist()
        prototypes = np.random.default_rng(777).normal(size=(10, 3, 32, 32))
        self.data = torch.tensor(np.stack([prototypes[y] + .3*rng.normal(size=(3,32,32)) for y in self.targets]), dtype=torch.float32)
    def __len__(self): return len(self.targets)
    def __getitem__(self, i): return self.data[i], int(self.targets[i])

@dataclass
class DataBank:
    train: Dataset
    test: Dataset
    num_classes: int
    coarse_by_fine: dict[int,int]


def get_data_bank(name: str, data_root: str, *, download=True, seed=0) -> DataBank:
    if name == "synthetic":
        return DataBank(TinySynthetic(True, seed), TinySynthetic(False, seed), 10, {x:x//2 for x in range(10)})
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if name == "cifar10" else (CIFAR100_MEAN, CIFAR100_STD)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean,std)])
    train = cls(root=data_root, train=True, transform=transform, download=download)
    test = cls(root=data_root, train=False, transform=transform, download=download)
    coarse = {}
    if name == "cifar100":
        path = Path(data_root) / 'cifar-100-python' / 'train'
        if path.exists():
            with path.open('rb') as fh:
                raw = pickle.load(fh, encoding='bytes')
            for fine, group in zip(raw[b'fine_labels'], raw[b'coarse_labels']):
                if fine in coarse and coarse[fine] != group:
                    raise ValueError('Inconsistent CIFAR-100 superclass membership')
                coarse[int(fine)] = int(group)
        if len(coarse) != 100:
            raise ValueError('CIFAR-100 superclass mapping unavailable; refusing to guess semantic relationships')
    else:
        # Research-defined heuristic pairs (NOT an official CIFAR-10 taxonomy):
        # airplane/ship, automobile/truck, bird/frog, cat/dog, deer/horse.
        # Membership must not be interpreted as measured feature similarity.
        coarse = {0:0,8:0,1:1,9:1,2:2,6:2,3:3,5:3,4:4,7:4}
    return DataBank(train, test, len(train.classes), coarse)


def class_order(bank: DataBank, relation: str, stages: int | None = None) -> list[int]:
    classes = list(range(bank.num_classes))
    if relation == 'natural':
        return classes
    if relation == 'related' and bank.coarse_by_fine:
        # Spread each superclass across multiple stages: first member of each
        # group, then second member, etc. With CIFAR-10 2x5 this places
        # a semantic-pair member in each consecutive stage.
        groups = defaultdict(list)
        for c in classes: groups[bank.coarse_by_fine[c]].append(c)
        if stages is not None and stages > 1:
            # Distribute each superclass across stages while balancing stage
            # sizes. Adjacent experiences can thus have related but new labels.
            bins = [[] for _ in range(stages)]
            for group_id, (_, labels) in enumerate(sorted(groups.items())):
                for member_index, c in enumerate(labels):
                    bins[(group_id + member_index) % stages].append(int(c))
            order = [c for bin_ in bins for c in bin_]
            if len(order) != len(classes) or len(set(order)) != len(classes):
                raise AssertionError('Class ordering lost or duplicated labels')
            return order
        return [c for j in range(max(map(len,groups.values())))
                for grp in sorted(groups) for c in groups[grp][j:j+1]]
    if relation == 'unrelated' and bank.coarse_by_fine:
        # Keep semantically related labels within stages, so across-stage
        # transitions primarily introduce new superclass groups.
        return sorted(classes, key=lambda c:(bank.coarse_by_fine[c],c))
    # No official semantic relationship is asserted for CIFAR-10.
    return classes


@dataclass(frozen=True)
class Arrival:
    round_id: int
    client_id: int
    stage: int
    indices: tuple[int,...]
    labels: tuple[int,...]
    recurrence: bool = False

@dataclass
class Scenario:
    arrivals: list[Arrival]
    stages: list[list[int]]
    label_holders: dict[int,set[int]]
    num_clients: int
    num_classes: int

    def arrivals_at(self, round_id):
        return [a for a in self.arrivals if a.round_id == round_id]


def _allocate_holders(cls, k, mode, rng, alpha, overlap):
    if mode == 'iid': return list(range(k))
    if mode == 'disjoint': return [cls % k]
    if mode == 'overlap':
        n=1+int(round(overlap*(min(k,4)-1)))
        return sorted({(cls+i)%k for i in range(n)})
    if mode == 'mixed':
        return sorted(set([cls % k, (cls+1)%k, (cls+3)%k, (cls+5)%k]))
    if mode == 'non_iid':
        weights = rng.dirichlet(np.full(k,alpha))
        top = np.argsort(weights)[-min(3,k):].tolist()
        return sorted(top)
    return sorted(set([cls % k, (cls+1)%k]))


def compile_scenario(bank:DataBank, cfg) -> Scenario:
    """Allocate training samples exactly once; recurrence uses held-out train samples."""
    rng = np.random.default_rng(cfg.seed + 1901)
    order = class_order(bank, cfg.relation, cfg.stages)
    assert cfg.stages <= len(order)
    stages = [list(map(int,a)) for a in np.array_split(order,cfg.stages)]
    stage_of = {int(c):s for s,classes in enumerate(stages) for c in classes}
    idx_by_class = defaultdict(list)
    for idx,label in enumerate(bank.train.targets): idx_by_class[int(label)].append(idx)
    arrivals = []
    holders = defaultdict(set)
    for cls in order:
        indices = np.array(idx_by_class[cls], dtype=int)
        rng.shuffle(indices)
        count = max(1,int(len(indices)*.2)) if cfg.scenario in {'recurrence','mixed'} else 0
        reserve = indices[:count]
        initial = indices[count:]
        recipients = _allocate_holders(cls, cfg.num_clients, cfg.scenario, rng, cfg.dirichlet_alpha, cfg.overlap)
        if cfg.scenario == 'non_iid':
            # Realized Dirichlet label skew, not just choosing label holders.
            weights = rng.dirichlet(np.full(cfg.num_clients,cfg.dirichlet_alpha))
            counts = rng.multinomial(len(initial),weights)
            recipients = list(range(cfg.num_clients))
            offsets = np.cumsum(counts)[:-1]
            chunks = np.split(initial,offsets)
        else:
            chunks = np.array_split(initial,len(recipients))
        for cid,chunk in zip(recipients,chunks):
            if len(chunk):
                stage = stage_of[cls]
                r = stage*cfg.rounds_per_stage
                if cfg.scenario == 'asynchronous':
                    r = min(cfg.stages*cfg.rounds_per_stage-1,r+(cid%max(1,cfg.rounds_per_stage)))
                arrivals.append(Arrival(r,int(cid),int(stage),tuple(map(int,chunk)),(int(cls),)))
                holders[int(cls)].add(int(cid))
        if len(reserve):
            # Recurrence at a different client, using *disjoint* original training examples.
            to_cid = recipients[0] if cfg.recurrence_client_policy == "same" else (recipients[0] + 3) % cfg.num_clients
            recurrent_stage = min(cfg.stages-1,stage_of[cls]+max(1,cfg.recurrence_delay))
            recurrence_r = min(cfg.stages*cfg.rounds_per_stage-1,recurrent_stage*cfg.rounds_per_stage)
            # A last-stage class cannot recur later; do not leak it into the past.
            if recurrent_stage > stage_of[cls]:
                arrivals.append(Arrival(recurrence_r,to_cid,recurrent_stage,tuple(map(int,reserve)),(int(cls),),True))
                holders[int(cls)].add(to_cid)
            else:
                # Return reserved examples to the original arrival event.
                arrivals.append(Arrival(stage_of[cls]*cfg.rounds_per_stage,recipients[0],stage_of[cls],tuple(map(int,reserve)),(int(cls),)))
    # Uniqueness: no training image can be allocated to two events unless explicitly enabled.
    used = [i for a in arrivals for i in a.indices]
    if len(used) != len(set(used)):
        raise AssertionError('Unintentional training sample duplication across clients/events')
    if len(used) != len(bank.train):
        raise AssertionError(f'Lost samples: allocated {len(used)} vs dataset {len(bank.train)}')
    return Scenario(sorted(arrivals,key=lambda a:(a.round_id,a.client_id)), stages,dict(holders),cfg.num_clients,bank.num_classes)
