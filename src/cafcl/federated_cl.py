"""Explicit data-sharing, federated continual-learning state protocol.

The server keeps a provenance-deduplicated union of actual TRAIN-image
exemplars, versioned DER targets, and additive diagonal-importance statistics.
GEM/A-GEM constraints are class-indexed reference-example tables: model-relative
gradient vectors MUST be recomputed at the receiving client's current weights,
not averaged across incompatible local parameter frames.

This is deliberately NOT private FL: accepted client images are shared with the
server and other synchronized clients. Raw sample IDs in the simulator reference
a common training-index namespace; image bytes are also explicitly transmitted.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import torch

REGULARIZERS=('ewc','si','mas')
MEMORY_METHODS=frozenset({'replay','gem','agem','er_ace','der','derpp'})

def _cpu_dict(values):
    return {str(k):v.detach().cpu().clone() for k,v in values.items()}

def _clone_memory(values):
    return {int(c):list(dict.fromkeys(map(int,indices))) for c,indices in values.items() if indices}

def combined_memory(private,shared):
    output=_clone_memory(shared)
    for cls,indices in private.items():
        group=output.setdefault(int(cls),[])
        seen=set(group)
        for idx in indices:
            idx=int(idx)
            if idx not in seen:
                group.append(idx);seen.add(idx)
    return output

def add_importance(client,kind,importance,anchor):
    """Store NEW local natural-parameter increments, not rebroadcast echoes."""
    if kind not in REGULARIZERS:raise ValueError(f'Unknown method: {kind}')
    incoming=_cpu_dict(importance);location=_cpu_dict(anchor)
    pending=client.pending_importance.setdefault(kind,{'precision':{},'weighted_anchor':{}})
    for name,value in incoming.items():
        if value.shape!=location[name].shape:raise ValueError('Importance shape mismatch')
        value=value.clamp(min=0);weighted=value*location[name]
        pending['precision'][name]=pending['precision'].get(name,torch.zeros_like(value))+value
        pending['weighted_anchor'][name]=pending['weighted_anchor'].get(name,torch.zeros_like(weighted))+weighted

def weighted_importance(old_importance,old_anchor,new_importance,new_anchor):
    if old_importance is None or old_anchor is None:
        return _cpu_dict(new_importance),_cpu_dict(new_anchor)
    merged={};anchors={}
    for name,importance in new_importance.items():
        a=old_importance[name].detach().cpu()
        b=importance.detach().cpu().clamp(min=0)
        ref=old_anchor[name].detach().cpu()
        new=new_anchor[name].detach().cpu()
        merged[name]=a+b
        anchors[name]=torch.where(merged[name]>0,
            (a*ref+b*new)/merged[name].clamp_min(1e-30),new)
    return merged,anchors

@dataclass
class FederatedCLState:
    """Central history, persisted with the round and final checkpoints."""
    version:int=0
    memory:dict[int,list[int]]=field(default_factory=dict)
    memory_logits:dict[int,torch.Tensor]=field(default_factory=dict)
    exemplar_payloads:dict[int,tuple[torch.Tensor,int]]=field(default_factory=dict)
    logits_source:dict[int,tuple[int,int]]=field(default_factory=dict)
    importance:dict[str,dict]=field(default_factory=dict)
    teacher:dict[str,torch.Tensor]|None=None
    teacher_version:int=-1
    uploaded_clients:set[int]=field(default_factory=set)
    integrated_classes:set[int]=field(default_factory=set)
    memory_provenance:dict[int,set[int]]=field(default_factory=dict)

    def merge_clients(self,uploaded_clients,bank,method,teacher_model,version):
        """Merge ONLY selected/uploaded private contributions once per source."""
        for client in uploaded_clients:
            self.integrated_classes.update(int(c) for c in client.pending_upload_classes)
        if method in MEMORY_METHODS:
            pool=_clone_memory(self.memory)
            ids={int(i) for group in pool.values() for i in group}
            for client in uploaded_clients:
                for cls,examples in client.private_memory.items():
                    cls=int(cls);group=pool.setdefault(cls,[])
                    for idx in examples:
                        idx=int(idx)
                        if idx<0 or idx>=len(bank.train):
                            raise ValueError(f'Exemplar index outside TRAIN: {idx}')
                        if int(bank.train.targets[idx])!=cls:
                            raise ValueError(f'Exemplar label mismatch: {idx} != {cls}')
                        self.memory_provenance.setdefault(idx,set()).add(int(client.client_id))
                        if idx not in ids:
                            group.append(idx);ids.add(idx)
                            # Actually move compact RGB payload bytes, not just IDs.
                            from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
                            x,label=bank.train[idx]
                            mean=torch.tensor(NORMALIZATION_MEAN).view(3,1,1)
                            std=torch.tensor(NORMALIZATION_STD).view(3,1,1)
                            pixels=((x.detach().cpu()*std+mean).clamp(0,1)*255).round().to(torch.uint8)
                            self.exemplar_payloads[idx]=(pixels,int(label))
                if method in {'der','derpp'}:
                    for idx,logits in client.memory_logits.items():
                        idx=int(idx)
                        if idx not in ids:continue
                        if idx not in self.memory_logits:
                            if logits.ndim!=1 or logits.shape[0]!=bank.num_classes:
                                raise ValueError('DER logit target/class count mismatch')
                            self.memory_logits[idx]=logits.detach().cpu().clone()
                            self.logits_source[idx]=(int(version),int(client.client_id))
            self.memory=pool
        if method in REGULARIZERS:
            table=self.importance.setdefault(method,{'precision':{},'weighted_anchor':{}})
            for client in uploaded_clients:
                delta=client.pending_importance.get(method)
                if not delta:continue
                for key in ('precision','weighted_anchor'):
                    for name,x in delta[key].items():
                        if not torch.isfinite(x).all():raise ValueError(f'Nonfinite {method} state')
                        old=table[key].get(name)
                        table[key][name]=x.detach().cpu().clone() if old is None else old+x.detach().cpu()
        if method=='lwf':
            # Use the newly committed global model as a common teacher;
            # stale client teacher weights cannot be meaningfully averaged.
            self.teacher=_cpu_dict(teacher_model)
            self.teacher_version=int(version)
        self.uploaded_clients.update(int(c.client_id) for c in uploaded_clients)
        self.version=int(version)

    def broadcast_to(self,client,method):
        """Receive the whole server state only at actual model synchronization."""
        for cls in self.integrated_classes:
            client.first_seen_local_experience.setdefault(int(cls),client.local_experience)
            client.trained_classes.add(int(cls))
            client.observed_classes.add(int(cls))
        if method in MEMORY_METHODS:
            client.received_memory=_clone_memory(self.memory)
            client.memory=combined_memory(client.private_memory,client.received_memory)
            client.received_example_payloads={int(i):(x.clone(),int(y))
                for i,(x,y) in self.exemplar_payloads.items()}
            if method in {'gem','agem'}:
                client.received_gradient_supports={
                    int(c):tuple(map(int,ids)) for c,ids in self.memory.items()}
            if method in {'der','derpp'}:
                client.memory_logits={int(i):v.detach().cpu().clone()
                                      for i,v in self.memory_logits.items()}
            for cls in client.received_memory:
                client.first_seen_local_experience.setdefault(int(cls),client.local_experience)
                client.trained_classes.add(int(cls))
                client.observed_classes.add(int(cls))
        elif method in REGULARIZERS:
            stats=self.importance.get(method,{})
            precision=stats.get('precision',{})
            numerator=stats.get('weighted_anchor',{})
            if precision:
                anchor={name:torch.where(p>0,numerator[name]/p.clamp_min(1e-30),
                    client.shared_weights[name].detach().cpu()) for name,p in precision.items()}
                importance={name:p.detach().cpu().clone() for name,p in precision.items()}
                if method=='ewc':client.ewc_fisher,client.ewc_anchor=importance,anchor
                elif method=='si':client.si_omega,client.si_anchor=importance,anchor
                else:client.mas_omega,client.mas_anchor=importance,anchor
        elif method=='lwf' and self.teacher:
            client.lwf_teacher=_cpu_dict(self.teacher)
            client.lwf_teacher_from_server=True
        client.federated_cl_version=self.version

    def statistics(self):
        return {'version':self.version,
            'shared_replay_samples':sum(map(len,self.memory.values())),
            'shared_replay_classes':len(self.memory),
            'shared_der_logits':len(self.memory_logits),
            'shared_image_payloads':len(self.exemplar_payloads),
            'shared_image_bytes':sum(x.numel()*x.element_size()
                                    for x,_ in self.exemplar_payloads.values()),
            'gradient_constraint_supports':sum(map(len,self.memory.values())),
            'importance_methods':sorted(self.importance),
            'teacher_version':self.teacher_version,
            'contributing_clients':sorted(self.uploaded_clients),
            'integrated_classes':len(self.integrated_classes)}
