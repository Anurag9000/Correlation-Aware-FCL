"""Source-executed FEAT local update/fine-tune on sparse client classes.

FEAT/methods/FEAT.py defines the two SGD training routines. Their AST bodies
are compiled unchanged and EXECUTED, not imitated by the generic CL trainer.

The upstream FEAT code itself contains an incompatible prototype orientation:
gsa_distill_loss expects [D,C] but _local_update passes model.fc.weight [C,D].
This adapter uses the author's ORIGINAL gsa_distill_loss body on weight.T.
That exact coordinate correction is identified as an unavoidable repair of
unexecutable upstream source, NOT bitwise identity.

The author ResNet is unchanged. The fixed ETF head is reconstructed because
the publisher omitted utils/inc_net.py. Its published local optimization,
SGD momentum=.9 and weight_decay are source-controlled (user sets epochs and
lr); source ECC estimate is called by _local_finetune. The FLDataSelector
retention is source executed after training. Entire FEAT still NOT certified.
"""
from __future__ import annotations

import ast
import copy
import logging
from collections import Counter
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader

from .upstream_losses import original_feat_gsa_source
from .upstream_feat_geometry import original_feat_geometry

ROOT=Path(__file__).resolve().parents[2]


class _CPUTransferOnly(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            node.func.attr='to'
            node.args=[ast.Constant(value='cpu')]
            node.keywords=[]
        return node


@lru_cache(maxsize=None)
def original_feat_train_methods(*,cpu_test=False):
    path=ROOT/'FEAT/methods/FEAT.py'
    tree=ast.parse(path.read_text(),filename=str(path))
    classes=[n for n in tree.body
             if isinstance(n,ast.ClassDef) and n.name=='FEAT']
    if len(classes)!=1:raise RuntimeError('Original FEAT class unavailable')
    target={'_local_update','_local_finetune'}
    method_nodes=[n for n in classes[0].body
                  if isinstance(n,ast.FunctionDef) and n.name in target]
    if set(n.name for n in method_nodes)!=target:
        raise RuntimeError('Original FEAT train methods unavailable')
    cls=ast.ClassDef(name='SourceFEAT',bases=[],keywords=[],
                     body=method_nodes,decorator_list=[])
    mod=ast.Module(body=[cls],type_ignores=[])
    if cpu_test:mod=_CPUTransferOnly().visit(mod)
    ast.fix_missing_locations(mod)
    # Source gsa_distill_loss expects a matrix with prototype COLUMNS,
    # although author _local_update passes the standard Linear WEIGHT
    # matrix with prototypes in ROWS. Transpose only at this boundary.
    source_gsa=original_feat_gsa_source()
    def gsa_orientation_bridge(features,labels,weights):
        if weights.ndim!=2 or weights.shape[0]!=int(weights.size(0)):
            raise RuntimeError('Invalid FEAT author classifier weights')
        if weights.shape[0]<=max(labels.tolist()):
            raise RuntimeError('FEAT prototype labels out of range')
        if weights.shape[1]!=features.shape[1]:
            raise RuntimeError('FEAT GSA source shape mismatch')
        return source_gsa(features,labels,weights.t().contiguous())
    geometry=original_feat_geometry()
    namespace={'torch':torch,'F':F,
               'gsa_distill_loss':gsa_orientation_bridge,
               'estimate_ecc_priors':geometry['estimate_ecc_priors']}
    exec(compile(mod,str(path),'exec'),namespace)
    return namespace['SourceFEAT']


class _FEATCompatView(nn.Module):
    """Original FEAT returns logits/features dict and exposes .fc.weight.

    Source training and original ECC consume a compact C_seen-class view;
    the public model retains exactly 100 global-class output parameters.
    """
    def __init__(self,core,known):
        super().__init__()
        self.core=core
        self.ids=tuple(int(c) for c in known)
        self.register_buffer('known_idx',
            torch.tensor(self.ids,dtype=torch.long),persistent=False)
        # The author only READS the FC prototype tensor. Its compact
        # view deliberately does not add a second parameter to optimizer.
        self.fc=SimpleNamespace(weight=core.classifier.weight.index_select(
            0,self.known_idx.to(core.classifier.weight.device)))
    def forward(self,x):
        features=self.core.features(x)
        logits=self.core.classifier(features).index_select(
            1,self.known_idx.to(features.device))
        return {'features':features,'logits':logits}


class _FEATTrainSubset(Dataset):
    def __init__(self,dataset,ids,classes,targets):
        self.dataset=dataset
        self.ids=tuple(map(int,ids))  # preserve author duplicate sample tickets
        self.class_map={int(c):i for i,c in enumerate(classes)}
        self.targets=targets
        if not self.ids:raise RuntimeError('Empty FEAT source TRAIN subset')
        for idx in self.ids:
            if int(targets[idx]) not in self.class_map:
                raise RuntimeError('FEAT replay contains unknown class')
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        idx=self.ids[i]
        image,label=self.dataset[idx]
        if int(label)!=int(self.targets[idx]):
            raise RuntimeError('FEAT source received incorrect sample payload')
        return idx,image,self.class_map[int(label)]


class _FEATTrainCollate:
    def __init__(self,cfg,client_id,experience,round_id):
        self.method=cfg.dataset
        self.seed=(cfg.seed+client_id*1_000_003+
                   round_id*100003+experience*1009)
        self.calls=0
    def __call__(self,batch):
        original_indices=torch.tensor([x[0] for x in batch])
        images=torch.stack([x[1] for x in batch])
        labels=torch.tensor([x[2] for x in batch],dtype=torch.long)
        if len(images)!=64:
            raise RuntimeError('FEAT authors source runner requires physical 64')
        if self.method=='cifar100':
            from .upstream_feat_augmentation import original_feat_augment_batch
            images=original_feat_augment_batch(images,self.seed+self.calls*197)
        self.calls+=1
        return original_indices,images,labels


def train_original_feat(model,client,bank,cfg,device,round_id,task_classes,
                        *,experience_end=False,active_pair_classes=None):
    if cfg.model=='native_feat_etf' and cfg.method!='native_feat':
        raise RuntimeError('ETF classifier belongs only to the FEAT method')
    if client.shared_weights is None:
        raise RuntimeError('FEAT client has not received its initial server model')
    client.last_learning_method='native_feat'
    model.load_state_dict(client.shared_weights)
    model.to(device)
    current=tuple(sorted(set(map(int,active_pair_classes or ()))))
    if len(current)!=2:
        raise RuntimeError('FEAT source requires exactly two current classes')
    observed=set(map(int,client.observed_classes))
    if not set(current).issubset(observed):
        raise RuntimeError('FEAT source encountered future labels')
    old=tuple(sorted(observed-set(current)))
    order=old+current  # original stage contiguous class range
    history=[]
    for cls,ids in client.memory.items():
        source_tickets=client.feat_selection_tickets.get(int(cls))
        history.extend(map(int,source_tickets if source_tickets is not None else ids))
    allowed=set(map(int,client.seen_indices)) | set(map(int,client.received_example_payloads))
    if any(int(i) not in allowed for i in history):
        raise RuntimeError('FEAT tried to train on unreceived remote data')
    fresh=list(dict.fromkeys(map(int,client.current_indices)))
    if not fresh:return None
    ids=fresh+history
    from .federated_cl import ClientArchiveDataset
    from .continual import _FixedBatchSampler
    train=(ClientArchiveDataset(bank.train,client.received_example_payloads)
           if cfg.share_continual_state and client.received_example_payloads
           else bank.train)
    dataset=_FEATTrainSubset(train,ids,order,bank.train.targets)
    # Original FEAT combines current TRAIN with n=1 copies of old retained
    # data. Here all accepted class-specific history is additionally included
    # per the user's cross-client CL synchronization intervention.
    sampler=_FixedBatchSampler(len(dataset),cfg.batch_size,
               cfg.seed+round_id*100003+client.client_id*127,shuffle=True)
    loader=DataLoader(dataset,batch_sampler=sampler,num_workers=0,
       collate_fn=_FEATTrainCollate(cfg,client.client_id,
                                   client.local_experience,round_id))
    current_view=_FEATCompatView(model,order).to(device)
    original=original_feat_train_methods(cpu_test=(device.type=='cpu'))()
    original.args={'local_ep':int(cfg.epochs),
                   'weight_decay':float(cfg.weight_decay),
                   'kd':.1,'ema_decay':.8}
    original.logger=logging.getLogger('cafcl.feat.source_train')
    original._known_classes=len(old)
    original._total_classes=len(order)
    original.source_prototype_orientation_fix='GSA [C,D] to [D,C]'
    if old:
        prior=copy.deepcopy(model)
        if client.native_teacher is not None:
            prior.load_state_dict(client.native_teacher)
        prior.to(device).eval()
        teacher_view=_FEATCompatView(prior,old).to(device).eval()
        _,loss,priors,ph,pt=original._local_finetune(
            teacher_view,current_view,loader,client.local_experience,
            client.client_id,float(cfg.lr))
        if experience_end and priors is not None and ph is not None and pt is not None:
            if int(priors['nT'])>0:
                client.pending_feat_ecc.append(dict(
                    priors=dict(priors),PH=ph.detach().cpu().clone(),
                    PT=pt.detach().cpu().clone(),
                    r_H=max(len(current)-1,1),r_T=max(len(old)-1,1),
                    client=int(client.client_id),
                    local_experience=int(client.local_experience),
                    new_classes=list(current),old_classes=list(old),
                    source='original FEAT._local_finetune/estimate_ecc_priors',
                ))
    else:
        _,loss=original._local_update(current_view,loader,float(cfg.lr))
    if experience_end:
        from .upstream_feat_replay import source_feat_select_local
        source_feat_select_local(model,client,bank,cfg,device,fresh,current)
    client.optimizer_state=None  # FEAT original resets each invocation
    steps=int(cfg.epochs)*len(loader)
    client.train_steps+=steps
    client.trained_classes.update(int(bank.train.targets[i]) for i in fresh)
    client.activated_labels.update(order)
    snapshot={name:value.detach().cpu().clone()
              for name,value in model.state_dict().items()}
    if any(not torch.isfinite(t.float()).all() for t in snapshot.values()):
        raise RuntimeError('FEAT source training produced nonfinite state')
    if experience_end:
        client.native_teacher=copy.deepcopy(snapshot)
        client.native_teacher_classes=set(order)
    counts=dict(Counter(int(bank.train.targets[i]) for i in fresh))
    return dict(
        weights=snapshot,examples=len(ids),
        class_counts=counts,class_update_steps={int(c):steps for c in order},
        active_pair_classes=list(current),steps=steps,
        loss=(float(loss.detach().cpu()) if isinstance(loss,torch.Tensor)
              else float(loss)),constraint_violations=0,qp_calls=0,
        projection_norm_last=0.,
        memory_samples=sum(map(len,client.memory.values())),
        method_used=cfg.method,fedprox_mu=0.,
        local_old_loss_before=None,local_old_loss_after=None,
        optimizer_state_policy='original_FEAT_fresh_SGD',
        sgd_momentum=.9,sgd_weight_decay=cfg.weight_decay,
        classification_competitor_classes=list(order),
        source_method=('FEAT/methods/FEAT.py:FEAT._local_finetune'
                       if old else 'FEAT/methods/FEAT.py:FEAT._local_update'),
        source_ema_decay=.8,
        source_gsa_shape_repair='transpose [C,D] into source [D,C]',
    )
