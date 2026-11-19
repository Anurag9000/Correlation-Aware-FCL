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

class RestrictedLabels(Dataset):
    """Training/test subset with a stable contiguous global class mapping.

    `original_labels` supplies the old CIFAR IDs in new global-label order.
    Both partitions use the identical mapping; no test sample is used in
    constructing a training partition or learned class relationship.
    """
    def __init__(self, underlying, original_labels):
        self.original_labels = tuple(int(c) for c in original_labels)
        mapping = {int(c): i for i, c in enumerate(self.original_labels)}
        self.indices = [i for i, c in enumerate(underlying.targets) if int(c) in mapping]
        self.targets = [mapping[int(underlying.targets[i])] for i in self.indices]
        self.underlying = underlying
        self.classes = list(range(len(mapping)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, _ = self.underlying[self.indices[index]]
        return x, self.targets[index]


@dataclass
class DataBank:
    train: Dataset
    test: Dataset
    num_classes: int
    coarse_by_fine: dict[int,int]
    original_labels: tuple[int, ...] = ()


def get_data_bank(name: str, data_root: str, *, download=True, seed=0, class_subset=()) -> DataBank:
    if name == "synthetic":
        return DataBank(TinySynthetic(True, seed), TinySynthetic(False, seed), 10, {x:x//2 for x in range(10)})
    cls = datasets.CIFAR10 if name == "cifar10" else datasets.CIFAR100
    mean, std = (CIFAR10_MEAN, CIFAR10_STD) if name == "cifar10" else (CIFAR100_MEAN, CIFAR100_STD)
    # Native CIFAR-100 resolution for both ViT-Tiny/16 and CIFAR-stem ResNet-18.
    # Both use identical preprocessing; synthetic fixtures remain 32x32.
    from .models import INPUT_SIZE,NORMALIZATION_MEAN,NORMALIZATION_STD
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(NORMALIZATION_MEAN,NORMALIZATION_STD)])
    train = cls(root=data_root, train=True, transform=transform, download=download)
    test = cls(root=data_root, train=False, transform=transform, download=download)
    coarse = {}
    if name in {"cifar100", "cifar100_20"}:
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
    original_labels=tuple(range(len(train.classes)))
    if name == "cifar100_20":
        # Ten superclasses, exactly two fine classes per superclass. The
        # deliberate ordering places old class 0 with new 12, and old class 1
        # with new 13, exactly as in the proposed whiteboard intervention.
        groups=defaultdict(list)
        for fine, superclass in sorted(coarse.items()):
            groups[superclass].append(fine)
        groups=[groups[g] for g in sorted(groups)[:10]]
        first=[g[0] for g in groups]
        second=[g[1] for g in groups]
        original_labels=tuple(first+second[-2:]+second[:-2])
    elif class_subset:
        if name != "cifar100":
            raise ValueError("Explicit class subset currently requires CIFAR-100")
        original_labels=tuple(map(int,class_subset))
        if len(set(original_labels))!=len(original_labels) or any(c not in coarse for c in original_labels):
            raise ValueError("Invalid CIFAR-100 subset")
    if name == "cifar100_20" or class_subset:
        train=RestrictedLabels(train,original_labels)
        test=RestrictedLabels(test,original_labels)
        coarse={i:coarse[original] for i,original in enumerate(original_labels)}
    return DataBank(train, test, len(train.classes), coarse, original_labels)


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
    duplicate_sample_ids: tuple[int, ...] = ()
    unused_sample_ids: tuple[int, ...] = ()

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
    order = list(cfg.class_order_override) if cfg.class_order_override else class_order(bank, cfg.relation, cfg.stages)
    if len(order) != bank.num_classes or set(order) != set(range(bank.num_classes)):
        raise ValueError("Class order must be a permutation of all active global labels")
    assert cfg.stages <= len(order)
    if cfg.scenario in {'domain','gradual_drift','stationary'}:
        stages=[list(map(int,order)) for _ in range(cfg.stages)]
    else:
        stages = [list(map(int,a)) for a in np.array_split(order,cfg.stages)]
    stage_of = {int(c):0 for c in order} if cfg.scenario in {'domain','gradual_drift','stationary'} else {
        int(c):s for s,classes in enumerate(stages) for c in classes}
    idx_by_class = defaultdict(list)
    for idx,label in enumerate(bank.train.targets): idx_by_class[int(label)].append(idx)
    arrivals = []
    holders = defaultdict(set)
    unused=[]
    for cls in order:
        indices = np.array(idx_by_class[cls], dtype=int)
        rng.shuffle(indices)
        if str(cls) in cfg.class_sample_fraction:
            fraction=cfg.class_sample_fraction[str(cls)]
            keep=max(1,int(round(len(indices)*fraction)))
            unused.extend(map(int,indices[keep:]))
            indices=indices[:keep]
        if cfg.class_imbalance_power:
            # A deterministic long-tail sample budget; discarded training
            # samples are explicitly recorded, never exposed or evaluated as
            # train data, and the held-out test bank remains unmodified.
            rank=stage_of[cls] + 1
            keep=max(1,int(round(len(indices)*rank**(-cfg.class_imbalance_power))))
            unused.extend(map(int,indices[keep:]))
            indices=indices[:keep]
        count = max(1,int(len(indices)*.2)) if cfg.scenario in {'recurrence','mixed','shifted_recurrence'} else 0
        reserve = indices[:count]
        initial = indices[count:]
        recipients = list(cfg.holders_override[str(cls)]) if str(cls) in cfg.holders_override else _allocate_holders(
            cls, cfg.num_clients, cfg.scenario, rng, cfg.dirichlet_alpha, cfg.overlap)
        if cfg.scenario in {'domain','gradual_drift','stationary'}:
            # Every class occurs at every stage, but each raw training sample
            # is allocated exactly once. The per-stage transform determines
            # the changing domain; stationary applies the identity transform.
            for stage,stage_chunk in enumerate(np.array_split(initial,cfg.stages)):
                for cid,chunk in zip(recipients,np.array_split(stage_chunk,len(recipients))):
                    if len(chunk):
                        r=min(cfg.stages*cfg.rounds_per_stage-1,
                              stage*cfg.rounds_per_stage+cfg.arrival_lag_by_client.get(str(cid),0))
                        arrivals.append(Arrival(r,int(cid),stage,
                                                tuple(map(int,chunk)),(int(cls),)))
                        holders[int(cls)].add(int(cid))
            if len(reserve):
                # No reserved recurrence should be silently discarded in the
                # stationary/domain cases; ordinarily reserve is empty.
                unused.extend(map(int,reserve))
            continue
        if cfg.scenario == 'non_iid':
            # Realized Dirichlet label skew, not just choosing label holders.
            weights = rng.dirichlet(np.full(cfg.num_clients,cfg.dirichlet_alpha))
            counts = rng.multinomial(len(initial),weights)
            recipients = list(range(cfg.num_clients))
            offsets = np.cumsum(counts)[:-1]
            chunks = np.split(initial,offsets)
        else:
            if str(cls) in cfg.class_holder_weights:
                weight=np.array(cfg.class_holder_weights[str(cls)],dtype=float)
                if len(weight)!=len(recipients):
                    raise ValueError('Explicit holder weights mismatch selected recipients')
                weight/=weight.sum()
                counts=rng.multinomial(len(initial),weight)
                chunks=np.split(initial,np.cumsum(counts)[:-1])
            elif cfg.client_sample_skew and len(recipients)>1:
                proportions=rng.dirichlet(np.full(len(recipients),1./cfg.client_sample_skew))
                counts=rng.multinomial(len(initial),proportions)
                chunks=np.split(initial,np.cumsum(counts)[:-1])
            else:
                chunks = np.array_split(initial,len(recipients))
        if cfg.sample_overlap_fraction and len(recipients)>1:
            # Deliberate cross-client raw-sample duplication; not confused
            # with same-class overlap using distinct training examples.
            chunks=[np.array(c,dtype=int) for c in chunks]
            extra=chunks[0][:int(round(len(chunks[0])*cfg.sample_overlap_fraction))]
            chunks[1]=np.concatenate([chunks[1],extra])
        for cid,chunk in zip(recipients,chunks):
            if len(chunk):
                stage = stage_of[cls]
                r = stage*cfg.rounds_per_stage
                if cfg.scenario == 'asynchronous':
                    r = min(cfg.stages*cfg.rounds_per_stage-1,r+(cid%max(1,cfg.rounds_per_stage)))
                if str(cid) in cfg.arrival_lag_by_client:
                    r=min(cfg.stages*cfg.rounds_per_stage-1,
                          r+cfg.arrival_lag_by_client[str(cid)])
                arrivals.append(Arrival(r,int(cid),int(stage),tuple(map(int,chunk)),(int(cls),)))
                holders[int(cls)].add(int(cid))
        if len(reserve):
            to_cid = recipients[0] if cfg.recurrence_client_policy == "same" else (recipients[0] + 3) % cfg.num_clients
            origin_round=stage_of[cls]*cfg.rounds_per_stage
            recurrence_r=(origin_round+cfg.recurrence_round_delay if cfg.recurrence_round_delay>0
                          else (stage_of[cls]+max(1,cfg.recurrence_delay))*cfg.rounds_per_stage)
            final_round=cfg.stages*cfg.rounds_per_stage-1
            times=[]
            for j in range(cfg.recurrence_repeats):
                when=recurrence_r+j*cfg.recurrence_period
                if origin_round<when<=final_round and when not in times:
                    times.append(when)
            if times:
                if len(reserve)<len(times):
                    raise ValueError('Too few distinct reserved samples for the requested recurring events')
                # Disjoint recurrence samples for each repeat, no implicit
                # reuse that would accidentally create episodic replay.
                reserved_chunks=np.array_split(reserve,len(times))
                for when,chunk in zip(times,reserved_chunks):
                    if len(chunk):
                        arrivals.append(Arrival(when,to_cid,when//cfg.rounds_per_stage,
                                                tuple(map(int,chunk)),(int(cls),),True))
                        holders[int(cls)].add(to_cid)
            else:
                arrivals.append(Arrival(origin_round,recipients[0],stage_of[cls],
                                        tuple(map(int,reserve)),(int(cls),)))
    # A declared sample-duplication intervention is the ONLY case in which
    # a training image may belong to multiple client events. Every other
    # training image must appear exactly once or be accounted for as unused.
    from collections import Counter
    counts=Counter(i for a in arrivals for i in a.indices)
    duplicates=tuple(sorted(i for i,n in counts.items() if n>1))
    if duplicates and cfg.sample_overlap_fraction==0:
        raise AssertionError('Unintentional training sample duplication across clients/events')
    if set(counts)&set(unused):
        raise AssertionError('An unused sample was allocated for training')
    if len(counts)+len(unused)!=len(bank.train):
        raise AssertionError(f'Unaccounted training samples: {len(counts)} allocated '
                             f'+{len(unused)} unused !={len(bank.train)}')
    return Scenario(sorted(arrivals,key=lambda a:(a.round_id,a.client_id)), stages,
                    dict(holders),cfg.num_clients,bank.num_classes,duplicates,tuple(sorted(unused)))
