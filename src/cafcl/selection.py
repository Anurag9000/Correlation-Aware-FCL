"""Budget-exact selectors. Data-aware selectors are oracle simulation controls.
They access simulated client class metadata and MUST NOT be described as
privacy-preserving deployable policies.
"""
from __future__ import annotations
import math
import numpy as np

def participation_budget(k, fraction):
    return max(1, min(k, int(math.floor(k*fraction+0.5))))


def effective_participation(cfg, round_id):
    total = max(1, cfg.stages * cfg.rounds_per_stage - 1)
    progress = round_id / total
    if cfg.participation_schedule == 'increasing':
        return max(1.0 / cfg.num_clients, cfg.participation * (0.5 + 0.5 * progress))
    if cfg.participation_schedule == 'decreasing':
        return max(1.0 / cfg.num_clients, cfg.participation * (1.0 - 0.5 * progress))
    if cfg.participation_schedule == 'alternating':
        return cfg.participation if round_id % 2 == 0 else max(1.0 / cfg.num_clients, cfg.participation / 2)
    return cfg.participation


def select_clients(states, cfg, round_id, rng, *, old_classes=frozenset(),new_classes=frozenset(),coarse_by_fine=None):
    ids=sorted(states)
    m=participation_budget(len(ids),effective_participation(cfg,round_id))
    scores={i:0.0 for i in ids}
    policy=cfg.selector
    if policy=='random':
        chosen=rng.choice(ids,size=m,replace=False).tolist()
    elif policy=='fixed':
        chosen=ids[:m]
    elif policy=='round_robin':
        start=(round_id*m)%len(ids)
        chosen=[ids[(start+j)%len(ids)] for j in range(m)]
    elif policy in {'least_recent','fairness'}:
        chosen=sorted(ids,key=lambda i:(states[i].last_selected,i))[:m]
    else:
        # The following selectors are PRIVILEGED/ORACLE: class holdings are
        # read from the simulator, not inferred from privacy-preserving uploads.
        new=set(new_classes);old=set(old_classes)
        holding={i:set(states[i].observed_classes) for i in ids}
        if policy=='old':
            scores={i:len(holding[i]&old) for i in ids}
        elif policy=='new':
            scores={i:len(holding[i]&new) for i in ids}
        elif policy=='historical_coverage':
            scores={i:sum(1./(1.+max(0,round_id-states[i].last_selected)) for c in (holding[i]&old)) for i in ids}
        elif policy=='balanced':
            chosen=[];old_covered=set();new_covered=set()
            for _ in range(m):
                remaining=[i for i in ids if i not in chosen]
                i=max(remaining,key=lambda i:(len((holding[i]&old)-old_covered)
                                             +len((holding[i]&new)-new_covered),-i))
                chosen.append(i)
                old_covered.update(holding[i]&old)
                new_covered.update(holding[i]&new)
        elif policy=='diversity':
            chosen=[];covered=set()
            for _ in range(m):
                remaining=[i for i in ids if i not in chosen]
                i=max(remaining,key=lambda i:(len(holding[i]-covered),-i))
                chosen.append(i);covered.update(holding[i])
        elif policy=='related' or policy=='stochastic_related':
            coarse=coarse_by_fine or {}
            old_groups={coarse[c] for c in old if c in coarse}
            scores={i:sum(1 for c in holding[i]&new if coarse.get(c,-1) in old_groups) for i in ids}
        elif policy=='recurrence':
            scores={i:sum(1./(1.+round_id-r) for r in states[i].recurrence_rounds.values()
                          if r<=round_id) for i in ids}
        elif policy=='overlap':
            scores={i:sum(len(holding[i]&holding[j]) for j in ids if j!=i) for i in ids}
        elif policy=='coverage':
            chosen=[];covered=set()
            for _ in range(m):
                remaining=[i for i in ids if i not in chosen]
                # Maximize new distinct labels, tie-break by client ID.
                i=max(remaining,key=lambda i:(len(holding[i]-covered),-i))
                chosen.append(i)
                covered.update(holding[i])
        else: raise ValueError(policy)
        if policy=='stochastic_related':
            weights=np.array([max(0.,scores[i])+1. for i in ids],dtype=np.float64)
            weights/=weights.sum()
            chosen=rng.choice(ids,size=m,replace=False,p=weights).tolist()
        elif policy not in {'coverage','balanced','diversity'}:
            chosen=sorted(ids,key=lambda i:(-scores[i],i))[:m]
    if cfg.force_target!='none':
        target=cfg.target_client
        if cfg.force_target=='include' and target not in chosen:
            chosen[-1]=target
        if cfg.force_target=='exclude' and target in chosen:
            replacement=next((i for i in ids if i not in chosen and i!=target),None)
            if replacement is None: raise ValueError('No available fixed-budget replacement for excluded client')
            chosen[chosen.index(target)]=replacement
    assert len(chosen)==m and len(set(chosen))==m
    return sorted(chosen)
