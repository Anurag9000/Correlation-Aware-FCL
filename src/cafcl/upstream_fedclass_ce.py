"""FedCLASS exemplar selection plus source CE and exhaustive adapted replay.

The authors' CE._train body is compiled verbatim for no-replay. Replay uses
an adapted trainer so every available exemplar contributes to each update.
The copied authors' BaseLearner._construct_exemplar_unified memory-selection
body is compiled verbatim from models/base.py.

Protocol adapters change only:
- label coordinates (global sparse class IDs to visible compact head);
- data source (CIFAR-100 per-client incremental task + received archives);
- physical input batches (64, tail padded to 64);
- exhaustive replay gradient accumulation for each incoming batch update;
- memory budget (50/class user setting) and timing (per local task);
- five independent persistent clients, offline communication and delayed
  synchronization, NOT the published original trainer's client schedule.

The original CE body RESETS SGD and AMP scaler at each task. The replay
adaptation uses SGD without the source AMP loop. No new-model self-distillation
is part of the selected CE baseline. Original CIFAR100 protocol reproduction
is not yet verified; this is NOT a full-paper-fidelity certification.
"""
from __future__ import annotations

import ast
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import logging

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torch.cuda.amp import autocast,GradScaler

ROOT=Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def original_class_method(file,class_name,method):
    path=ROOT/file
    module=ast.parse(path.read_text(),filename=str(path))
    classes=[c for c in module.body if isinstance(c,ast.ClassDef)
             and c.name==class_name]
    if len(classes)!=1:raise RuntimeError(f'{class_name} missing in {file}')
    methods=[n for n in classes[0].body
             if isinstance(n,ast.FunctionDef) and n.name==method]
    if len(methods)!=1:raise RuntimeError(f'{method} missing in {file}')
    isolated=ast.Module(body=methods,type_ignores=[])
    ast.fix_missing_locations(isolated)
    # The executed function is exactly the original author AST; no custom
    # loss/memory source code is substituted.
    env={'torch':torch,'np':np,'optim':optim,'F':F,'logging':logging,
         'GradScaler':GradScaler,'autocast':autocast,
         'tensor2numpy':(lambda x:x.cpu().data.numpy()
                         if x.is_cuda else x.data.numpy())}
    exec(compile(isolated,str(path),'exec'),env)
    return env[method]


class _KnownHead(nn.Module):
    """Shape-stable model with the same dynamic visible-class CE semantics."""
    def __init__(self,model,known):
        super().__init__()
        self.core=model
        self.register_buffer('output_ids',
            torch.as_tensor(known,dtype=torch.long),persistent=False)

    def forward(self,x):
        return self.core(x).index_select(1,self.output_ids)


class _IndicesWithLabels(Dataset):
    def __init__(self,images,identifiers,known,targets):
        self.images=images
        self.identifiers=list(identifiers)
        self.labels={int(c):i for i,c in enumerate(known)}
        self.targets=targets

    def __len__(self):
        return len(self.identifiers)

    def __getitem__(self,position):
        idx=int(self.identifiers[position])
        x,y=self.images[idx]
        label=int(y)
        if label not in self.labels:
            raise RuntimeError(f'FedCLASS original CE received unseen label {label}')
        if label!=int(self.targets[idx]):
            raise RuntimeError(f'FedCLASS source replay label mismatch at {idx}')
        return idx,x,self.labels[label]


class _OriginalMemoryProvider:
    """The authors' selection operates on sample ids, not image tensor values."""
    def __init__(self,train,permitted):
        self.train=train
        self.permitted=set(map(int,permitted))

    def get_dataset(self,classes,source,mode,ret_data,client_id):
        if source!='train' or mode!='test' or not ret_data:
            raise RuntimeError('Invalid original exemplar selection request')
        target={int(v) for v in classes}
        ids=[int(i) for i in sorted(self.permitted)
             if int(self.train.targets[i]) in target]
        return np.asarray(ids,dtype=np.int64),np.asarray(
            [int(self.train.targets[i]) for i in ids],dtype=np.int64),None


def update_original_exemplars(client,bank,indices,k):
    """Execute unchanged author _construct_exemplar_unified for 50/class.

    Returned raw-memory array cells contain global TRAIN sample IDs rather than
    redundant pixel arrays. The original implementation merely slices and
    concatenates those arrays, so index-level selection is identical to source
    array selection and is used to retrieve the same immutable RGB bytes.
    """
    from .federated_cl import combined_memory
    if k<=0:return
    current=sorted({int(bank.train.targets[i]) for i in indices})
    old_ids=[int(i) for cls,vals in client.private_memory.items()
             for i in vals]
    if len(set(old_ids))!=len(old_ids):
        raise RuntimeError('Source memory has repeated image ids')
    existing_targets=[int(bank.train.targets[i]) for i in old_ids]
    provider=_OriginalMemoryProvider(bank.train,indices)
    obj=SimpleNamespace(
        data_manager=provider,client_id=client.client_id,
        cur_classes=np.asarray(current,dtype=np.int64),
        data_memory=np.asarray(old_ids,dtype=np.int64),
        targets_memory=np.asarray(existing_targets,dtype=np.int64),
    )
    original_class_method('FedCLASS/models/base.py','BaseLearner',
                          '_construct_exemplar_unified')(obj,k)
    chosen=defaultdict(list)
    for idx,cls in zip(obj.data_memory,obj.targets_memory):
        idx=int(idx);cls=int(cls)
        if int(bank.train.targets[idx])!=cls:
            raise RuntimeError('Original CE exemplar class mismatch')
        if idx not in chosen[cls]:chosen[cls].append(idx)
    if any(len(vals)>k for vals in chosen.values()):
        raise RuntimeError('Original CE exemplar cap exceeded')
    client.private_memory={int(c):list(vals) for c,vals in chosen.items()}
    client.memory=combined_memory(client.private_memory,client.received_memory)


def train_original_ce(model,client,bank,cfg,device,round_id,
                      task_classes=None,*,experience_end=False,
                      active_pair_classes=None):
    """Execute authors' original CE._train directly in each local task."""
    if client.shared_weights is None:
        raise RuntimeError('FedCLASS local model not initialized')
    client.last_learning_method=cfg.method
    model.load_state_dict(client.shared_weights)
    model.to(device)
    new_indices=list(dict.fromkeys(map(int,client.current_indices)))
    if not new_indices:return None
    known=tuple(sorted(map(int,client.observed_classes)))
    if len(known)<2:
        raise RuntimeError('FedCLASS CE requires two visible classes')
    from .federated_cl import ClientArchiveDataset
    from .continual import _FixedBatchSampler
    train_data=(ClientArchiveDataset(bank.train,client.received_example_payloads)
                if cfg.share_continual_state and client.received_example_payloads
                else bank.train)
    history=list(dict.fromkeys(int(i) for group in client.memory.values() for i in group))
    allowed_ids=list(dict.fromkeys(new_indices+history))
    for index in history:
        if index not in client.seen_indices and index not in client.received_example_payloads:
            raise RuntimeError('FedCLASS replay image was not legitimately downloaded')
    dataset=_IndicesWithLabels(train_data,allowed_ids,known,bank.train.targets)
    batches=_FixedBatchSampler(len(dataset),cfg.batch_size,
                               cfg.seed+round_id*100003+client.client_id*127,
                               shuffle=True)
    loader=DataLoader(dataset,batch_sampler=batches,num_workers=0,
                      pin_memory=(device.type=='cuda'))
    wrapped=_KnownHead(model,known)
    obj=SimpleNamespace(
        network=wrapped,device=device,train_loader=loader,
        epochs=cfg.epochs,epoches=cfg.epochs,lr=cfg.lr,
        momentum=cfg.momentum,weight_decay=cfg.weight_decay,
        client_id=client.client_id,cur_task=client.local_experience,
        cur_round=round_id,optimizer=None)
    if cfg.method=='replay' and history:
        # Every incoming physical minibatch defines one update. Accumulate its
        # gradient with the mean gradient of the complete local+server archive,
        # reading at most cfg.batch_size replay examples at once.
        optimizer=optim.SGD(filter(lambda p:p.requires_grad,wrapped.parameters()),
            momentum=cfg.momentum,lr=cfg.lr,weight_decay=cfg.weight_decay)
        current=_IndicesWithLabels(train_data,new_indices,known,bank.train.targets)
        current_sampler=_FixedBatchSampler(len(current),cfg.batch_size,
            cfg.seed+round_id*100003+client.client_id*127,shuffle=True)
        current_loader=DataLoader(current,batch_sampler=current_sampler,
                                  num_workers=0,pin_memory=(device.type=='cuda'))
        steps=int(cfg.epochs)*len(current_loader)
        history_dataset=_IndicesWithLabels(train_data,history,known,bank.train.targets)
        history_loader=DataLoader(history_dataset,batch_size=cfg.batch_size,
                                  shuffle=False,num_workers=0,
                                  pin_memory=(device.type=='cuda'))
        current_scale=1./(1.+cfg.replay_ratio)
        history_scale=cfg.replay_ratio/(1.+cfg.replay_ratio)
        for _epoch in range(cfg.epochs):
            for _ids,inputs,targets in current_loader:
                optimizer.zero_grad(set_to_none=True)
                inputs,targets=inputs.to(device),targets.to(device)
                (F.cross_entropy(wrapped(inputs),targets)*current_scale).backward()
                for _hids,hinputs,htargets in history_loader:
                    hinputs,htargets=hinputs.to(device),htargets.to(device)
                    weight=history_scale*len(htargets)/len(history_dataset)
                    (F.cross_entropy(wrapped(hinputs),htargets)*weight).backward()
                optimizer.step()
    else:
        # No persistent optimizer is loaded: original CE._train recreates SGD
        # and GradScaler every invocation.
        original_class_method('FedCLASS/models/ce.py','CE','_train')(obj)
        steps=int(cfg.epochs)*len(loader)
    if steps<=0:
        raise RuntimeError('FedCLASS original CE performed zero optimizer steps')
    if experience_end and cfg.method=='replay':
        update_original_exemplars(client,bank,new_indices,cfg.memory_per_class)
    client.optimizer_state=None # source resets every local task
    client.train_steps+=steps
    client.trained_classes.update(map(int,known))
    client.activated_labels.update(map(int,known))
    counts=dict(Counter(int(bank.train.targets[i]) for i in new_indices))
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if not all(torch.isfinite(v.float()).all() for v in state.values()):
        raise RuntimeError('Original FedCLASS CE produced non-finite parameters')
    return dict(
        weights=state,
        examples=len(allowed_ids),  # source FedAvg weights actual augmented train size
        class_counts=counts,
        class_update_steps={int(c):steps for c in known},
        active_pair_classes=list(active_pair_classes) if active_pair_classes else None,
        steps=steps,
        loss=None, # author method writes detailed per-epoch loss to the log
        constraint_violations=0,qp_calls=0,projection_norm_last=0.,
        memory_samples=sum(map(len,client.memory.values())),
        method_used=cfg.method,fedprox_mu=0.,
        local_old_loss_before=None,local_old_loss_after=None,
        optimizer_state_policy='source_reset_each_task',
        sgd_momentum=cfg.momentum,sgd_weight_decay=cfg.weight_decay,
        classification_competitor_classes=list(known),
        source_method='FedCLASS/models/ce.py:CE._train',
        source_replay='FedCLASS/models/base.py:BaseLearner._construct_exemplar_unified',
        source_aggregation='FedCLASS/utils/fed_algorithm.py:FedAvg.update',
    )
