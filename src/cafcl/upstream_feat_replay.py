"""FEAT authors' exact FLDataSelector retained-example selection.

Extracts the original, unmodified author's FLDataSelector CLASS AST from
FEAT/methods/FEAT.py, invoking ALL four methods: mask_local_data,
compute_local_leverage_scores, aggregate_leverage_scores and sample_data.

Experimental adaptation: classes are initially disjoint across clients and
offline clients cannot send features. Therefore selecting source masks/scores
is performed per class at each client's local end-of-experience, never with
hidden peer data. This preserves exact author QR/SVD probabilities and
with-replacement tickets but does NOT reproduce the original cross-client
joint query-budget allocation. Faithfulness remains blocked.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from threading import RLock

import torch
from torch.utils.data import DataLoader
from .federated_cl import combined_memory

ROOT=Path(__file__).resolve().parents[2]
SOURCE_METHODS=('mask_local_data','compute_local_leverage_scores',
                'aggregate_leverage_scores','sample_data')
_TORCH_CPU_SELECTOR_LOCK=RLock()


@lru_cache(maxsize=1)
def original_feat_selector():
    source=ROOT/'FEAT/methods/FEAT.py'
    module=ast.parse(source.read_text(),filename=str(source))
    classes=[n for n in module.body if isinstance(n,ast.ClassDef) and
             n.name=='FLDataSelector']
    if len(classes)!=1:raise RuntimeError('Authors FLDataSelector is missing')
    members={n.name for n in classes[0].body if isinstance(n,ast.FunctionDef)}
    if set(SOURCE_METHODS)|{'__init__','generate_random_orthogonal_matrix'}>members:
        raise RuntimeError('Authors source FLDataSelector has incomplete methods')
    isolated=ast.Module(body=classes,type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace={'torch':torch}
    exec(compile(isolated,str(source),'exec'),namespace)
    return namespace['FLDataSelector']


def _source_att_features(model,bank,indices,device,batch_size):
    """Authors' FEAT._extract_client_features obtains network()['att'].

    The FEAT upstream backbone returns ['features'] and ['att']; the latter
    is selected here, just as the author's selector. Only local TRAIN samples
    are indexed. Last partial forward is padded to exactly physical batch64
    and its duplicated outputs discarded.
    """
    if not indices:
        raise ValueError('Feature selection requires nonempty local data')
    was_training=model.training
    model.eval()
    result=[]
    try:
        with torch.no_grad():
            for offset in range(0,len(indices),batch_size):
                current=indices[offset:offset+batch_size]
                raw=torch.stack([bank.train[i][0] for i in current],dim=0)
                valid=len(current)
                if valid<batch_size:
                    suffix=torch.arange(batch_size-valid)%valid
                    raw=torch.cat((raw,raw.index_select(0,suffix)),dim=0)
                raw=raw.to(device)
                if hasattr(model,'features') and hasattr(model.features,'original'):
                    source=model.features.original(raw)
                    if not isinstance(source,dict) or 'att' not in source:
                        raise RuntimeError('Source FEAT backbone did not expose att')
                    feats=source['att']
                elif bank.num_classes==4 and hasattr(model,'features'):
                    # Tiny synthetic unit-test fixture only, never production.
                    feats=model.features(raw)
                else:
                    raise RuntimeError('Original FEAT attention-feature source missing')
                result.append(feats[:valid].detach().cpu())
    finally:
        model.train(was_training)
    return torch.cat(result,dim=0)


def source_feat_select_local(model,client,bank,cfg,device,indices,active_classes):
    """Original FLDataSelector per-class source operation under sparse streams.

    Source multinomial uses *replacement=True*, meaning the same sample can
    occupy several original retention tickets. Keep duplicate tickets in
    client.feat_selection_tickets; the server-visible archive and shared
    memory index map are deduplicated by source image index. This explicit
    communication difference is not silently treated as original FEAT FL.
    """
    new=tuple(sorted(set(int(c) for c in active_classes)))
    if not new or not indices:return {}
    classes={int(bank.train.targets[i]) for i in indices}
    if classes!=set(new):
        raise RuntimeError(f'FEAT retention may access only new class pair; {classes} != {new}')
    source_class=original_feat_selector()
    retained={}
    details={}
    for cls in new:
        candidates=sorted(int(i) for i in indices if int(bank.train.targets[i])==cls)
        if not candidates:
            raise RuntimeError('FEAT source selection received empty class')
        feats=_source_att_features(model,bank,candidates,device,cfg.batch_size)
        capacity=min(int(cfg.memory_per_class),len(candidates))
        if capacity<=0:
            continue
        # The original torch.rand/QR/multinomial uses ambient CPU RNG.
        # fork_rng + lock prevents concurrent client calls from corrupting
        # each other's global RNG or making streams nondeterministic.
        with _TORCH_CPU_SELECTOR_LOCK:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(cfg.seed+179_007+
                        101*client.client_id+10009*client.local_experience+cls))
                selector=source_class(num_clients=1,local_features=[feats],
                                      query_budget=capacity)
                masked=selector.mask_local_data()
                scores=selector.compute_local_leverage_scores(masked)
                p=selector.aggregate_leverage_scores(scores)
                chosen=selector.sample_data(p)
        selection=[candidates[int(j)] for j in chosen[0]]
        if len(selection)!=capacity:
            raise RuntimeError('Authors FLDataSelector returned incorrect ticket count')
        retained[cls]=selection
        details[cls]={'available':len(candidates),'tickets':len(selection),
                      'unique':len(set(selection))}
    # Retain previous experiences, including duplicates as selected
    # originally. The communication-level sample archive deduplicates bytes.
    for cls,tickets in retained.items():
        client.private_memory[cls]=list(tickets)
        client.feat_selection_tickets[cls]=list(tickets)
    client.memory=combined_memory(client.private_memory,client.received_memory)
    client.feat_selection_receipts.append(dict(
        experience=int(client.local_experience),client=int(client.client_id),
        classes=list(new),source='FEAT/methods/FEAT.py:FLDataSelector',
        selection=details))
    return retained
