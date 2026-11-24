"""Frozen matched designs, independent of PyTorch and legacy stream runtime."""
from collections import Counter, defaultdict
import hashlib
import json
import random
import math


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8],"big")


def split_counts(n, weights):
    if n < 0 or not weights or min(weights)<0 or sum(weights)<=0:
        raise ValueError("Invalid allocation")
    ideal=[n*w/sum(weights) for w in weights]
    result=[math.floor(x) for x in ideal]
    for i in sorted(range(len(weights)),key=lambda i:(-(ideal[i]-result[i]),i))[:n-sum(result)]:
        result[i]+=1
    return result


def compile_matched(job, targets, num_classes=100):
    """Allocate every unique TRAIN index once; never inspect held-out data.

    A unit is a semantic binary task; MC/CM mix assigned semantic units in
    one local stage batch, MO visits its two units sequentially.
    """
    family=job["family"]; k=job["clients"]; seed=job["seed"]
    if family not in ("MC","CM","MO") or num_classes%10:
        raise ValueError("Unsupported matched design")
    if family=="MO" and k!=5:
        raise ValueError("MO uses five clients")
    pool=defaultdict(list)
    for i,y in enumerate(targets):
        pool[int(y)].append(i)
    if set(pool)!=set(range(num_classes)):
        raise ValueError("Expected complete contiguous class set")
    for c,ids in pool.items():
        random.Random(stable_seed(seed,"class",c)).shuffle(ids)
    stages=[]
    for t in range(num_classes//10):
        classes=list(range(10*t,10*t+10))
        pairs=[tuple(classes[2*i:2*i+2]) for i in range(5)]
        units=[[] for _ in range(k)]
        if family=="MO":
            other=pairs if job["curriculum"]!="matched_partial_class_overlap" else [
                (classes[(2*i+1)%10],classes[(2*i+2)%10]) for i in range(5)]
            other=[tuple(sorted(x)) for x in other]
            for i,pair in enumerate(pairs):
                units[i].append({"head":pair,"slot":0,"indices":[]})
            for i,pair in enumerate(other):
                owner=(i+1)%k if job["curriculum"]=="matched_complete_task_overlap" else i
                units[owner].append({"head":pair,"slot":1,"indices":[]})
            for c in classes:
                holders=[u for us in units for u in us if c in u["head"]]
                if len(holders)!=2:
                    raise ValueError("Each MO class must occur twice")
                cut=len(pool[c])//2
                for u,ids in zip(holders,[pool[c][:cut],pool[c][cut:]]):
                    u["indices"].extend(ids)
        else:
            for p,pair in enumerate(pairs):
                for c in pair:
                    if job["partition"]=="iid":
                        weights=[1.]*k
                    elif job["partition"]=="controlled_label_skew":
                        # A fixed 80% dominant mixture plus 20% uniform
                        # support; class ownership rotates across stages.
                        owner=(c+t)%k
                        weights=[.2/k]*k
                        weights[owner]+=.8
                    else:
                        raise ValueError("Unknown partition")
                    counts=split_counts(len(pool[c]),weights)
                    pos=0
                    for i,n in enumerate(counts):
                        u=next((u for u in units[i] if u["head"]==pair),None)
                        if u is None:
                            u={"head":pair,"slot":p,"indices":[]}
                            units[i].append(u)
                        u["indices"].extend(pool[c][pos:pos+n])
                        pos+=n
            units=[[u for u in us if u["indices"]] for us in units]
        if any(not us or not all(u["indices"] for u in us) for us in units):
            raise ValueError("Empty client or local unit")
        m=round(k*job["participation"])
        if not 1<=m<=k:
            raise ValueError("Invalid cohort")
        cohort=sorted(random.Random(stable_seed(seed,"matched-cohort",t)).sample(range(k),m))
        stages.append({"stage":t,"classes":classes,"units":units,"cohort":cohort})
    indices=[i for s in stages for us in s["units"] for u in us for i in u["indices"]]
    if sorted(indices)!=list(range(len(targets))):
        raise ValueError("Training index coverage/uniqueness violated")
    for s in stages:
        for us in s["units"]:
            for u in us:
                if not set(int(targets[i]) for i in u["indices"])<=set(u["head"]):
                    raise ValueError("Task-label mismatch")
    return stages


def batch_stream(records, replay, labels, seed, epochs, epoch_start=0):
    """Stable physical batch 64; exact epoch-indexed schedules across R.

    Current rows are padded cyclically after deterministic shuffle, never
    dropped. Replay is class-uniform, sampled with replacement. Archive must
    be frozen throughout the parent stage for compute-matched comparisons.
    """
    if not records:
        raise ValueError("Empty batch stream")
    width=32 if replay else 64
    by_class=defaultdict(list)
    for record in replay:
        by_class[int(labels[record[0]])].append(record)
    classes=sorted(by_class)
    for epoch in range(epoch_start,epoch_start+epochs):
        rng=random.Random(stable_seed(seed,epoch))
        order=list(records);rng.shuffle(order)
        for offset in range(0,len(order),width):
            real=order[offset:offset+width]
            padded=real+[order[i%len(order)] for i in range(width-len(real))]
            history=[rng.choice(by_class[rng.choice(classes)]) for _ in range(32)] if replay else []
            yield padded+history,len(real),len(padded),len(history)


def plan_audit(job,stages,targets):
    out=[]
    for s in stages:
        out.append({"stage":s["stage"],"classes":s["classes"],"cohort":s["cohort"],
                    "client_sizes":[sum(len(u["indices"]) for u in us) for us in s["units"]],
                    "client_class_counts":[dict(Counter(int(targets[i]) for u in us for i in u["indices"])) for us in s["units"]],
                    "task_instances":[len(us) for us in s["units"]]})
    return {"job_id":job["job_id"],"stages":out,"unique_train_samples":len(targets),
            "allocation_sha256":hashlib.sha256(json.dumps(stages,sort_keys=True).encode()).hexdigest()}
