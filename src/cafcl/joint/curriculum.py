"""Deterministic atomic curriculum compiler, independent of communication time.

Class sets are semantic tasks; occurrence IDs are assigned before communication
grouping. No training indices from validation or test are accepted by the
allocation API. Its caller supplies the actual training split targets.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import random
import math
from typing import Mapping, Sequence


def digest(value):
    # JSON object keys are strings on disk. Normalize before hashing so
    # integer class-ID maps survive a JSON round trip unchanged.
    normalized=json.loads(json.dumps(value,allow_nan=False))
    return sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def stable_seed(*parts):
    return int(digest(parts)[:16], 16)


@dataclass(frozen=True)
class AtomicTask:
    occurrence_id: str
    semantic_task_id: str
    head_id: str
    client_id: int
    atomic_step: int
    local_experience_index: int
    classes: tuple[int, ...]
    train_indices: tuple[int, ...]


@dataclass(frozen=True)
class Interval:
    interval_id: int
    atomic_steps: tuple[int, ...]
    occurrences: tuple[str, ...]
    upload_clients: tuple[int, ...]
    terminal_flush: bool = False


@dataclass(frozen=True)
class JointPlan:
    num_classes: int
    num_clients: int
    seed: int
    head_policy: str
    tasks: tuple[AtomicTask, ...]
    intervals: tuple[Interval, ...]
    overlap_audit: dict
    allocation_audit: dict

    def to_dict(self):
        out = asdict(self)
        out["atomic_hash"] = digest(out["tasks"])
        out["plan_hash"] = digest(out)
        return out


def foundational_stream(spatial, local_overlap, *, num_classes=100,
                        num_clients=5, steps=100, class_order=None):
    """Three two-class spatial templates, with matched atomic horizons.

    O0/O1 denotes adjacent local class overlap, not total historical recurrence.
    A full cycle is required for the default balanced comparison.
    """
    if num_clients != 5 or num_classes < 10 or num_classes % 2:
        raise ValueError("Foundational templates require five clients and even >=10 classes")
    if spatial not in ("S1", "S2", "S3") or local_overlap not in (0, 1):
        raise ValueError("Unknown foundational treatment")
    if steps < 1:
        raise ValueError("Positive horizon required")
    order = list(range(num_classes)) if class_order is None else list(class_order)
    if sorted(order) != list(range(num_classes)):
        raise ValueError("class_order must be a permutation")
    out = []
    for t in range(steps):
        a = (2-local_overlap)*t
        for i in range(num_clients):
            offset = (0 if spatial == "S1" else
                      (0 if i < 2 else num_classes//2) if spatial == "S2"
                      else i)
            out.append((t, i, (order[(a+offset)%num_classes],
                                order[(a+offset+1)%num_classes])))
    return out


def exact_groups(groups, *, width, steps, num_classes=100):
    """Disjoint semantic tasks across groups; exact sharing within each group."""
    groups = tuple(int(x) for x in groups)
    if not groups or min(groups) < 1 or max(groups) < 2:
        raise ValueError("Joint replication requires a group of at least two")
    if width < 2 or width*len(groups) > num_classes or steps < 1:
        raise ValueError("Infeasible exact group template")
    out = []
    for t in range(steps):
        cid = 0
        for g, size in enumerate(groups):
            a = (t*width*len(groups)+g*width) % num_classes
            classes = tuple((a+j)%num_classes for j in range(width))
            for _ in range(size):
                out.append((t, cid, classes))
                cid += 1
    return out


def shared_anchor_stream(*, num_clients, width, shared, steps, num_classes=100,
                         fixed_anchor=False):
    """Every client shares a common subset; private subsets are disjoint."""
    if not 1 <= shared < width or num_clients < 2 or steps < 1:
        raise ValueError("Partial common-anchor template required")
    span = shared + num_clients*(width-shared)
    if span > num_classes:
        raise ValueError("Insufficient labels for private subsets")
    out = []
    for t in range(steps):
        if fixed_anchor:
            anchor = tuple(range(shared))
            private = tuple(shared+(t*num_clients*(width-shared)+j)%
                            (num_classes-shared)
                            for j in range(num_clients*(width-shared)))
        else:
            labels = tuple((t*span+j)%num_classes for j in range(span))
            anchor, private = labels[:shared], labels[shared:]
        for i in range(num_clients):
            start = i*(width-shared)
            out.append((t, i, anchor+private[start:start+width-shared]))
    return out


def explicit_sequences(sequences: Mapping[int, Sequence[Sequence[int]]]):
    """Support heterogeneous depths, widths, recurrence and arbitrary order."""
    return [(t, int(i), tuple(classes))
            for i, tasks in sorted(sequences.items())
            for t, classes in enumerate(tasks)]


def compile_plan(events, train_targets, *, num_clients, num_classes,
                 local_depth=1, seed=0, head_policy="occurrence_specific",
                 offline_clients=0, outage_duration=1, terminal_flush=True,
                 minimum_samples=1, require_all_classes=True,
                 interval_boundaries=None, upload_cohorts=None,
                 allocation_order='timeline',raw_sample_overlap=0.0,allow_cross_semantic_reuse=False,allocation_weights=None):
    """Compile exact explicit events (atomic_step, client, class tuple).

    Fixed unique-data allocation: each observed class's entire TRAIN pool is
    split over its occurrences; by default each sample is assigned once.
    Optional same-semantic reuse preserves draw counts but reduces unique support.
    No automatic tail events, padding or class insertion.
    Explicit interval boundaries are exclusive atomic-step endpoints.
    """
    if num_clients < 2 or num_classes < 2 or local_depth < 1:
        raise ValueError("Invalid client/class/depth count")
    if not 0 <= offline_clients < num_clients or outage_duration < 1:
        raise ValueError("Invalid outage specification")
    if not 0 <= raw_sample_overlap <= 1:
        raise ValueError('Raw sample overlap must lie in [0,1]')
    if raw_sample_overlap and head_policy!='semantic_shared' and not allow_cross_semantic_reuse:
        raise ValueError('Raw reuse requires shared semantic heads')
    if allocation_order not in ('timeline','client_semantic'):
        raise ValueError('Unsupported allocation order')
    if minimum_samples < 1:
        raise ValueError("Minimum sample support must be positive")
    if head_policy not in ("occurrence_specific", "semantic_shared", "client_semantic"):
        raise ValueError("Unsupported head policy")
    ordered = sorted((int(t), int(i), tuple(sorted(map(int, classes))))
                     for t, i, classes in events)
    if not ordered:
        raise ValueError("Empty atomic stream")
    coordinates = set()
    owners = defaultdict(set)
    counts = Counter()
    for t, i, classes in ordered:
        if t < 0 or not 0 <= i < num_clients or (t, i) in coordinates:
            raise ValueError("Invalid or duplicate atomic coordinate")
        if len(classes) < 2 or len(set(classes)) != len(classes):
            raise ValueError("Task must have at least two distinct classes")
        if min(classes) < 0 or max(classes) >= num_classes:
            raise ValueError("Invalid class label")
        coordinates.add((t, i))
        counts.update(classes)
        for c in classes:
            owners[c].add(i)
    if {i for _, i, _ in ordered} != set(range(num_clients)):
        raise ValueError("Every declared client must have a task")
    if require_all_classes and set(counts) != set(range(num_classes)):
        raise ValueError("Incomplete class coverage")
    if not any(len(v) > 1 for v in owners.values()):
        raise ValueError("Primary joint curriculum must have cross-client overlap")
    if raw_sample_overlap and not allow_cross_semantic_reuse:
        # A raw image may recur only under one semantic class set, so its
        # source-head provenance cannot activate a future different task.
        semantic_by_class=defaultdict(set)
        for _,_,classes in ordered:
            for label in classes:semantic_by_class[label].add(classes)
        if any(len(v)>1 for v in semantic_by_class.values()):
            raise ValueError('Ambiguous raw reuse across different semantic tasks is unsupported')
    pool = defaultdict(list)
    for index, label in enumerate(train_targets):
        c = int(label)
        if not 0 <= c < num_classes:
            raise ValueError("Invalid training label")
        pool[c].append(index)
    for c, n in counts.items():
        if len(pool[c]) < n*minimum_samples:
            raise ValueError(f"Insufficient TRAIN samples for class {c}: {len(pool[c])}/{n}")
        random.Random(stable_seed(seed, "allocation", c)).shuffle(pool[c])
    weights=dict(allocation_weights or {})
    valid_keys={f'{t}:{i}:{label}' for t,i,classes in ordered for label in classes}
    if not set(weights)<=valid_keys or any(not math.isfinite(float(v)) or float(v)<=0 for v in weights.values()):
        raise ValueError('Allocation weights must be finite positive values for existing class occurrences')
    if weights and raw_sample_overlap:
        raise ValueError('Combined weighted allocation and raw reuse is not yet a supported control')
    seen_occ = Counter()
    # Order by client/semantic task for sequence-order ablations. Each
    # client's semantic occurrences receive identical sample subsets when
    # only their task order is permuted (repetitions retain their local rank).
    allocation_events=(ordered if allocation_order=='timeline' else
                       sorted(ordered,key=lambda x:(x[1],x[2],x[0])))
    allocated={}
    weighted_counts={}
    if weights:
        by_label=defaultdict(list)
        for t,i,classes in allocation_events:
            for label in classes:by_label[label].append(f'{t}:{i}:{label}')
        for label,keys in by_label.items():
            values=[float(weights.get(k,1.0)) for k in keys]
            budget=len(pool[label])-minimum_samples*len(keys)
            # Largest remainder allocation, with deterministic tie breaking.
            ideal=[budget*v/sum(values) for v in values]
            sizes=[math.floor(v) for v in ideal]
            for j in sorted(range(len(keys)),key=lambda j:(-(ideal[j]-sizes[j]),j))[:budget-sum(sizes)]:
                sizes[j]+=1
            weighted_counts.update({k:n+minimum_samples for k,n in zip(keys,sizes)})
    shared_counts={c:int((len(pool[c])//n)*raw_sample_overlap) if n>1 else 0
                   for c,n in counts.items()}
    private_cursor=Counter(shared_counts)
    allocated_class_sizes=[]
    for t,i,classes in allocation_events:
        ids=[]
        for label in classes:
            j=seen_occ[label];seen_occ[label]+=1;n=len(pool[label])
            size=(j+1)*n//counts[label]-j*n//counts[label]
            shared=shared_counts[label]
            if weights:
                size=weighted_counts[f'{t}:{i}:{label}']
                start=private_cursor[label];end=start+size
                ids.extend(pool[label][start:end]);private_cursor[label]=end
            elif shared:
                start=private_cursor[label];end=start+size-shared
                ids.extend(pool[label][:shared]);ids.extend(pool[label][start:end])
                private_cursor[label]=end
            else:
                ids.extend(pool[label][j*n//counts[label]:(j+1)*n//counts[label]])
            allocated_class_sizes.append(size)
        allocated[t,i]=tuple(ids)
    local_exp = Counter()
    tasks = []
    for t, i, classes in ordered:
        local_exp[i] += 1
        semantic = "classes_"+"_".join(f"{c:03d}" for c in classes)
        occ = f"client_{i:03d}__atomic_{t:06d}__{semantic}"
        task_key='task_'+'_'.join(f'{c:03d}' for c in classes)
        head = (f'{task_key}__client_{i:03d}__atomic_{t:06d}' if head_policy == 'occurrence_specific' else
                task_key if head_policy == 'semantic_shared' else
                f'{task_key}__client_{i:03d}')
        tasks.append(AtomicTask(occ, semantic, head, i, t, local_exp[i],
                                classes, allocated[t,i]))
    horizon = ordered[-1][0]+1
    active_steps = sorted({t for t, _, _ in ordered})
    if active_steps != list(range(horizon)):
        raise ValueError("Atomic timeline has empty global steps")
    ends = (list(range(local_depth, horizon, local_depth))+[horizon]
            if interval_boundaries is None else list(map(int, interval_boundaries)))
    if not ends or ends[-1] != horizon or ends != sorted(set(ends)) or ends[0] <= 0:
        raise ValueError("Invalid exclusive communication boundaries")
    if upload_cohorts is not None and len(upload_cohorts) != len(ends):
        raise ValueError("One explicit upload cohort per interval required")
    intervals = []
    start = 0
    omitted = set()
    for r, end in enumerate(ends):
        if upload_cohorts is not None:
            cohort = tuple(sorted(map(int, upload_cohorts[r])))
            if len(cohort) != len(set(cohort)) or not cohort or not set(cohort) <= set(range(num_clients)):
                raise ValueError("Invalid upload cohort")
        else:
            if r % outage_duration == 0:
                omitted = set(random.Random(stable_seed(seed, "outage", r//outage_duration))
                              .sample(range(num_clients), offline_clients))
            cohort = tuple(i for i in range(num_clients) if i not in omitted)
        occurrences = tuple(x.occurrence_id for x in tasks if start <= x.atomic_step < end)
        intervals.append(Interval(r, tuple(range(start, end)), occurrences, cohort))
        start = end
    if terminal_flush:
        intervals.append(Interval(len(intervals), (), (), tuple(range(num_clients)), True))
    by_step = defaultdict(list)
    by_client = defaultdict(list)
    for x in tasks:
        by_step[x.atomic_step].append(x)
        by_client[x.client_id].append(x)
    spatial = Counter()
    local = Counter()
    for group in by_step.values():
        for k, a in enumerate(group):
            for b in group[k+1:]:
                overlap = len(set(a.classes)&set(b.classes))
                spatial[str(overlap)] += 1
    for seq in by_client.values():
        for a, b in zip(seq, seq[1:]):
            local[str(len(set(a.classes)&set(b.classes)))] += 1
    task_owners = defaultdict(set)
    for x in tasks:
        task_owners[x.semantic_task_id].add(x.client_id)
    all_indices = [idx for x in tasks for idx in x.train_indices]
    if any(len(x.train_indices)!=len(set(x.train_indices)) for x in tasks):
        raise AssertionError('Duplicate raw sample within one atomic task')
    if not raw_sample_overlap and len(set(all_indices)) != len(all_indices):
        raise AssertionError("Unexpected duplicate allocation")
    audit = {
        "class_occurrences": dict(sorted(counts.items())),
        "class_owners": {c: sorted(v) for c, v in sorted(owners.items())},
        "semantic_task_owners": {k: sorted(v) for k, v in sorted(task_owners.items())},
        "spatial_intersection_histogram": dict(spatial),
        "adjacent_local_intersection_histogram": dict(local),
        "atomic_steps": horizon,
        "task_occurrences": len(tasks),
        "semantic_tasks": len(task_owners),
        "classifier_heads": len({x.head_id for x in tasks}),
    }
    allocation = {"unique_assigned": len(set(all_indices)), "assignments": len(all_indices),
                  "unused_training_indices": sorted(set(range(len(train_targets)))-set(all_indices)),
                  "minimum_samples_per_occurrence": min(len(x.train_indices) for x in tasks),
                  "minimum_samples_per_class_occurrence": min(allocated_class_sizes),
                  "raw_sample_overlap": float(raw_sample_overlap),
                  "allocation_order": allocation_order,
                  "training_target_hash": digest(list(map(int, train_targets)))}
    if weights:
        allocation['allocation_weights']={k:float(v) for k,v in sorted(weights.items())}
        allocation['allocation_minimum_samples']=minimum_samples
        allocation['realized_class_occurrence_sizes']=weighted_counts
        allocation['allocation_contract']='fixed_unique_data_weighted_occurrence_partition'
    if allow_cross_semantic_reuse:
        allocation['allow_cross_semantic_reuse']=True
    if raw_sample_overlap:
        allocation['duplicate_sample_ids']=sorted(i for i,n in Counter(all_indices).items() if n>1)
        allocation['shared_samples_per_class_occurrence']=dict(sorted(shared_counts.items()))
        allocation['allocation_contract']='fixed_fresh_draws_reduced_unique_support'
    return JointPlan(num_classes, num_clients, seed, head_policy, tuple(tasks),
                     tuple(intervals), audit, allocation)


def foundational_registry(depths=(1, 2, 3, 5), *, seed=0):
    """48 nominal conditions; execution status is not implied by planning."""
    for spatial in ("S1", "S2", "S3"):
        for overlap in (0, 1):
            for depth in depths:
                for setting in ("task_il", "class_il"):
                    yield {"job_id": f"J-H-{spatial}-O{overlap}-L{depth}-{setting}",
                           "spatial": spatial, "local_overlap": overlap,
                           "local_depth": depth, "learning_setting": setting,
                           "seed": seed, "epochs": 5, "num_clients": 5,
                           "head_policy": "semantic_shared",
                           "protocol": "joint_spatiotemporal_fcl_v1"}
