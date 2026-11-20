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
import copy
import torch

REGULARIZERS=('ewc','si','mas')
MEMORY_METHODS=frozenset({'replay','gem','agem','er_ace','der','derpp','native_glfc','native_lga','native_feat'})

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
    # Older or heterogeneous contributors may have a sparse importance table.
    # Missing coordinates mean zero precision, never a missing-parameter crash.
    for name in sorted(set(old_importance)|set(new_importance)):
        template=(new_importance.get(name) if name in new_importance else old_importance[name]).detach().cpu()
        a=old_importance[name].detach().cpu().clamp(min=0) if name in old_importance else torch.zeros_like(template)
        b=new_importance[name].detach().cpu().clamp(min=0) if name in new_importance else torch.zeros_like(template)
        ref=old_anchor[name].detach().cpu() if name in old_anchor else new_anchor[name].detach().cpu()
        new=new_anchor[name].detach().cpu() if name in new_anchor else ref
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
    # Cumulative TRAIN example counts of server-accepted client uploads only.
    class_sample_exposure:dict[int,int]=field(default_factory=dict)
    # The server knows only samples that came from accepted client uploads.
    acknowledged_train_indices:set[int]=field(default_factory=set)
    memory_provenance:dict[int,set[int]]=field(default_factory=dict)
    # Stable provenance of each exemplar, including which local task supplied it.
    memory_origins:dict[int,tuple[int,int]]=field(default_factory=dict)
    feat_ecc:dict|None=None
    feat_ecc_history:list[dict]=field(default_factory=list)
    source_method_artifacts:dict[int,dict]=field(default_factory=dict)
    native_lga_radius:float|None=None
    native_prototype_gradients:dict[tuple[int,int,int],dict]=field(default_factory=dict)
    native_latest_accepted_prototypes:list[tuple[int,int,int]]=field(default_factory=list)
    # Proxy source reconstruction, teacher history and pending accepted
    # gradient backlog. Never includes a hidden client's unsent receipts.
    native_proxy_state:dict|None=None
    native_proxy_pending_receipts:list[dict]=field(default_factory=list)
    native_proxy_history:list[dict]=field(default_factory=list)

    def merge_clients(self,uploaded_clients,bank,method,teacher_model,version):
        """Merge ONLY selected/uploaded private contributions once per source."""
        for client in uploaded_clients:
            self.integrated_classes.update(int(c) for c in client.pending_upload_classes)
            self.acknowledged_train_indices.update(int(i) for i in client.seen_indices)
        # Each accepted client's source-method artifact ledger becomes visible
        # only at an actual upload. In particular, offline GLFC/LGA entropy
        # and FEAT selection tickets must never leak before reconnection.
        for client in uploaded_clients:
            if (client.last_learning_method or method) in {
                    'native_glfc','native_lga','native_feat'}:
                self.source_method_artifacts[int(client.client_id)] = dict(
                    source_method=str(client.last_learning_method or method),
                    native_memory_state=copy.deepcopy(client.native_memory_state),
                    feat_selection_tickets=copy.deepcopy(client.feat_selection_tickets),
                    feat_selection_receipts=copy.deepcopy(client.feat_selection_receipts),
                    local_experience=int(client.local_experience),
                    received_at_server_version=int(version),
                )
        self.native_latest_accepted_prototypes=[]
        for client in uploaded_clients:
            if client.last_learning_method not in {'native_glfc','native_lga'}:
                continue
            for receipt in client.pending_native_prototypes:
                key=(int(receipt['client']),int(receipt['local_experience']),
                     int(receipt['cls']))
                if int(receipt['client'])!=int(client.client_id):
                    raise RuntimeError('Forged GLFC/LGA prototype sender')
                if key in self.native_prototype_gradients:
                    raise RuntimeError('Duplicate original prototype receipt')
                if any(not torch.isfinite(t).all()
                       for t in receipt['gradients']):
                    raise RuntimeError('Nonfinite original proto gradient')
                self.native_prototype_gradients[key]=copy.deepcopy(receipt)
                self.native_latest_accepted_prototypes.append(key)
        method_by_client={int(c.client_id):(c.last_learning_method or method)
                          for c in uploaded_clients}
        if any(m in MEMORY_METHODS for m in method_by_client.values()):
            pool=_clone_memory(self.memory)
            ids={int(i) for group in pool.values() for i in group}
            for client in uploaded_clients:
                if method_by_client[client.client_id] not in MEMORY_METHODS:continue
                for cls,examples in client.private_memory.items():
                    cls=int(cls);group=pool.setdefault(cls,[])
                    for idx in examples:
                        idx=int(idx)
                        if idx<0 or idx>=len(bank.train):
                            raise ValueError(f'Exemplar index outside TRAIN: {idx}')
                        if int(bank.train.targets[idx])!=cls:
                            raise ValueError(f'Exemplar label mismatch: {idx} != {cls}')
                        self.memory_provenance.setdefault(idx,set()).add(int(client.client_id))
                        self.memory_origins.setdefault(idx,(int(client.client_id),int(
                            client.sample_first_local_experience.get(idx,
                                client.sample_first_stage.get(idx,client.local_experience)))))
                        if idx not in ids:
                            group.append(idx);ids.add(idx)
                            # Actually move compact RGB payload bytes, not just IDs.
                            from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
                            x,label=bank.train[idx]
                            mean=torch.tensor(NORMALIZATION_MEAN).view(3,1,1)
                            std=torch.tensor(NORMALIZATION_STD).view(3,1,1)
                            pixels=((x.detach().cpu()*std+mean).clamp(0,1)*255).round().to(torch.uint8)
                            self.exemplar_payloads[idx]=(pixels,int(label))
                if method_by_client[client.client_id] in {'der','derpp'}:
                    for idx,logits in client.memory_logits.items():
                        idx=int(idx)
                        if idx not in ids:continue
                        if idx not in self.memory_logits:
                            if logits.ndim!=1 or logits.shape[0]!=bank.num_classes:
                                raise ValueError('DER logit target/class count mismatch')
                            self.memory_logits[idx]=logits.detach().cpu().clone()
                            self.logits_source[idx]=(int(version),int(client.client_id))
            self.memory=pool
        for kind in REGULARIZERS:
            if not any(m==kind for m in method_by_client.values()):continue
            table=self.importance.setdefault(kind,{'precision':{},'weighted_anchor':{}})
            for client in uploaded_clients:
                if method_by_client[client.client_id]!=kind:continue
                delta=client.pending_importance.get(kind)
                if not delta:continue
                for key in ('precision','weighted_anchor'):
                    for name,x in delta[key].items():
                        if not torch.isfinite(x).all():raise ValueError(f'Nonfinite {kind} state')
                        old=table[key].get(name)
                        table[key][name]=x.detach().cpu().clone() if old is None else old+x.detach().cpu()
        # FEAT method-specific ECC is uploaded only by accepted clients;
        # all hidden clients' pending priors remain private until reconnection.
        if method=='native_feat':
            from .upstream_feat_geometry import consolidate_original_ecc,clone_feat_ecc
            new_receipts=[receipt for client in uploaded_clients
                          for receipt in client.pending_feat_ecc]
            if new_receipts:
                result=consolidate_original_ecc(new_receipts)
                if result is not None:
                    self.feat_ecc=clone_feat_ecc(result)
                    self.feat_ecc_history.append(dict(version=int(version),
                        accepted_clients=sorted(int(c.client_id) for c in uploaded_clients),
                        source_receipts=len(new_receipts),
                        old_energy=dict(result['energy'])))
        # Publish the current committed global model as LwF teacher on
        # EVERY synchronization, not only when a LwF client participated.
        # The latest model is the cumulative teacher for all prior classes.
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
        # Full server archive and provenance are always delivered, irrespective
        # of the active method. A method only *uses* the compatible entries.
        client.received_memory=_clone_memory(self.memory)
        client.memory=combined_memory(client.private_memory,client.received_memory)
        client.received_sample_origins=dict(self.memory_origins)
        # Every selected client receives the entire acknowledged class table,
        # not any classes still held privately by disconnected clients.
        client.received_server_class_sample_exposure=dict(self.class_sample_exposure)
        client.received_gradient_supports={int(c):tuple(map(int,ids))
            for c,ids in self.memory.items()}
        client.received_memory_logits={int(i):v.detach().cpu().clone()
            for i,v in self.memory_logits.items()}
        client.received_importance={kind:{key:{str(n):x.detach().cpu().clone()
            for n,x in stat.items()} for key,stat in table.items()}
            for kind,table in self.importance.items()}
        client.received_lwf_teacher=(_cpu_dict(self.teacher) if self.teacher else None)
        client.received_lwf_teacher_version=self.teacher_version
        client.received_native_artifacts=copy.deepcopy(self.source_method_artifacts)
        client.received_lga_radius=self.native_lga_radius
        client.received_native_prototypes=copy.deepcopy(self.native_prototype_gradients)
        if method in {'native_glfc','native_lga'}:
            client.received_native_proxy=copy.deepcopy(self.native_proxy_state)
        if method=='native_feat':
            from .upstream_feat_geometry import clone_feat_ecc
            client.received_feat_ecc=clone_feat_ecc(self.feat_ecc)
        # Transmit actual bytes to *every* selected client. The active local
        # algorithm decides whether to use replay, constraints, or regularizers.
        client.received_example_payloads={int(i):(x.clone(),int(y))
            for i,(x,y) in self.exemplar_payloads.items()}
        # ALL selected clients now install the entire available CL history.
        # Applying it remains method-specific in local_train: replay, GEM,
        # Fisher, SI, MAS and LwF continue to use their own original updates.
        # Passive snapshots alone are insufficient when methods change or
        # mixed-method clients arrive later; keep full active copies as well.
        private_only={int(i):v.detach().cpu().clone()
            for i,v in client.memory_logits.items() if int(i) not in self.memory_logits}
        client.memory_logits={**private_only,**{int(i):v.detach().cpu().clone()
                              for i,v in self.memory_logits.items()}}
        for cls in self.integrated_classes:
            client.first_seen_stage.setdefault(int(cls),int(self.version))
            client.first_seen_local_experience.setdefault(int(cls),client.local_experience)
        for kind in REGULARIZERS:
            stats=self.importance.get(kind,{})
            precision=stats.get('precision',{})
            numerator=stats.get('weighted_anchor',{})
            if not precision:continue
            anchors={name:torch.where(p>0,numerator[name]/p.clamp_min(1e-30),
                client.shared_weights[name].detach().cpu()) for name,p in precision.items()}
            importance={name:p.detach().cpu().clone() for name,p in precision.items()}
            if kind=='ewc':client.ewc_fisher,client.ewc_anchor=importance,anchors
            elif kind=='si':client.si_omega,client.si_anchor=importance,anchors
            else:client.mas_omega,client.mas_anchor=importance,anchors
        if self.teacher is not None:
            client.lwf_teacher=_cpu_dict(self.teacher)
            client.lwf_teacher_from_server=True
        if method in {'native_glfc','native_lga','native_feat'}:
            # Native old-model anchor is created ONLY during a real download.
            client.native_teacher=_cpu_dict(self.teacher)
            client.native_teacher_classes=set(self.integrated_classes)
        client.federated_cl_version=self.version

    def assert_complete_download(self,client):
        """Validate ALL CL artifacts after every selected-client server sync.

        Check the acknowledged global CL-state ledger, not only the currently
        enabled method. Offline clients must never be called here.
        """
        if client.federated_cl_version != self.version:
            raise RuntimeError('Client CL-state version differs from committed server')
        if set(client.received_memory)!=set(self.memory):
            raise RuntimeError('Client missing globally integrated replay classes')
        for cls,indices in self.memory.items():
            if set(indices)!=set(client.received_memory[cls]):
                raise RuntimeError(f'Client missing server replay evidence for class {cls}')
        if set(client.received_example_payloads)!=set(self.exemplar_payloads):
            raise RuntimeError('Client missing raw exemplar payloads')
        for idx,(raw,label) in self.exemplar_payloads.items():
            x,y=client.received_example_payloads[idx]
            if int(y)!=int(label) or not torch.equal(x,raw):
                raise RuntimeError(f'Client received corrupt exemplar bytes for sample {idx}')
        if set(client.received_memory_logits)!=set(self.memory_logits):
            raise RuntimeError('Client missing historical DER trajectory logits')
        for idx,logits in self.memory_logits.items():
            if not torch.equal(client.received_memory_logits[idx],logits):
                raise RuntimeError(f'Client received wrong DER target for sample {idx}')
        if set(client.received_gradient_supports)!=set(self.memory):
            raise RuntimeError('Client missing GEM/A-GEM historical gradient supports')
        for cls,ids in self.memory.items():
            if tuple(client.received_gradient_supports[cls])!=tuple(ids):
                raise RuntimeError(f'Client GEM/A-GEM historical support mismatch: {cls}')
        if set(client.received_importance)!=set(self.importance):
            raise RuntimeError('Client missing some EWC/SI/MAS importance methods')
        for kind,table in self.importance.items():
            for key in ('precision','weighted_anchor'):
                remote=client.received_importance[kind][key]
                if set(remote)!=set(table[key]):
                    raise RuntimeError(f'Client missing {kind} {key} coordinates')
                for name,x in table[key].items():
                    if not torch.equal(remote[name],x):
                        raise RuntimeError(f'Client {kind} importance disagrees with server')
        if client.received_lwf_teacher_version!=self.teacher_version:
            raise RuntimeError('Client has stale global LwF teacher')
        if self.teacher is not None:
            if client.lwf_teacher is None or set(client.lwf_teacher)!=set(self.teacher):
                raise RuntimeError('Client missing complete global teacher')
            for name,param in self.teacher.items():
                if not torch.equal(client.lwf_teacher[name],param):
                    raise RuntimeError(f'Client LwF teacher mismatch: {name}')
        if client.last_learning_method=='native_feat':
            if (client.received_feat_ecc is None)!=(self.feat_ecc is None):
                raise RuntimeError('Client missing accepted FEAT ECC state')
            if self.feat_ecc is not None:
                for k in ('PH','PT'):
                    if not torch.equal(client.received_feat_ecc[k],self.feat_ecc[k]):
                        raise RuntimeError('Corrupted FEAT original ECC projector')
                if client.received_feat_ecc['energy']!=self.feat_ecc['energy']:
                    raise RuntimeError('Corrupted FEAT original ECC energy')
        if set(client.received_native_prototypes)!=set(self.native_prototype_gradients):
            raise RuntimeError('Client missed acknowledged gradient prototypes')
        for key,receipt in self.native_prototype_gradients.items():
            local=client.received_native_prototypes[key]
            if int(local['cls'])!=int(receipt['cls']) or len(
                    local['gradients'])!=len(receipt['gradients']):
                raise RuntimeError('Corrupt original prototype receipt')
            if any(not torch.equal(x,y) for x,y in zip(
                    local['gradients'],receipt['gradients'])):
                raise RuntimeError('Corrupt GLFC/LGA source gradient tensors')
        if client.last_learning_method in {'native_glfc','native_lga'}:
            local=client.received_native_proxy
            source=self.native_proxy_state
            if (local is None)!=(source is None):
                raise RuntimeError('Client did not receive original source proxy teacher state')
            if source is not None:
                for name in ('best_model_1','best_model_2'):
                    a=local[name]; b=source[name]
                    if (a is None)!=(b is None):
                        raise RuntimeError('Original proxy teacher slot missing')
                    if a is not None:
                        if set(a)!=set(b) or any(
                          not torch.equal(a[k],b[k]) for k in b):
                            raise RuntimeError('Corrupted original proxy teacher')
                if (local['best_perf']!=source['best_perf'] or
                    local['server_round']!=source['server_round'] or
                    local['reconstructed_images']!=source['reconstructed_images']):
                    raise RuntimeError('Incorrect source proxy reconstruction metadata')
                if local['new_set_label']!=source['new_set_label']:
                    raise RuntimeError('Incorrect source proxy class labels')
        if client.received_lga_radius!=self.native_lga_radius:
            raise RuntimeError('Client missing original LGA radius')
        if client.received_native_artifacts!=self.source_method_artifacts:
            raise RuntimeError('Client missing accepted original-method artifacts')
        if client.received_server_class_sample_exposure!=self.class_sample_exposure:
            raise RuntimeError('Client historical class exposure table is incomplete')
        if not set(self.integrated_classes).issubset(client.observed_classes):
            raise RuntimeError('Client did not inherit all globally integrated class labels')

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
            'feat_ecc_accepted_receipts':sum(x['source_receipts'] for x in self.feat_ecc_history),
            'feat_ecc_history_entries':len(self.feat_ecc_history),
            'source_method_artifact_clients':len(self.source_method_artifacts),
            'native_lga_radius':self.native_lga_radius,
            'original_proto_gradient_receipts':len(self.native_prototype_gradients),
            'original_proxy_accepted_backlog':len(self.native_proxy_pending_receipts),
            'original_proxy_source_steps':sum(
                int(e.get('ran_source_dataloader',False)) for e in self.native_proxy_history),
            'original_proxy_teacher_available':bool(
                self.native_proxy_state and
                self.native_proxy_state.get('best_model_2') is not None),
            'contributing_clients':sorted(self.uploaded_clients),
            'integrated_classes':len(self.integrated_classes)}


class ClientArchiveDataset:
    """Replay remote examples from the bytes received in the synchronization.

    The underlying dataset provides *new local* examples, while the server
    archive supplies samples contributed by other online clients.
    """
    def __init__(self,underlying,payloads):
        self.underlying=underlying
        self.payloads=payloads
        self.targets=underlying.targets
    def __len__(self):return len(self.underlying)
    def __getitem__(self,index):
        example=self.payloads.get(int(index))
        if example is None:return self.underlying[index]
        pixels,label=example
        from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
        mean=torch.tensor(NORMALIZATION_MEAN,dtype=torch.float32).view(3,1,1)
        std=torch.tensor(NORMALIZATION_STD,dtype=torch.float32).view(3,1,1)
        return ((pixels.float()/255.)-mean)/std,int(label)
