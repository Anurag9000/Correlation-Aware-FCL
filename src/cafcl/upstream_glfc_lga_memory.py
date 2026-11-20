"""Execute the published GLFC/LGA entropy trigger and exemplar herding.

Implements the author's update_new_set, entropy_signal, _reduce_exemplar_sets
and _construct_exemplar_set function bodies unchanged via original AST.
The authors' original protocol assumes 6- or >2-class stages, a global
contiguous class order, full-class forward batches and one global sample pool.
Only protocol adapters are supplied here:
  * two currently arriving classes; prior-local-pair tracking;
  * class-index compact logits; private TRAIN images ONLY;
  * fixed 64-input feature forwards and analytical feature mean;
  * memory_budget = 50 * count of learned classes (user's 50/class);
  * uploaded memory still follows the experimental sync protocol.

Do not advertise this as complete GLFC/LGA: prototype-mask gradient creation
and the proxy server's gradient reconstruction/monitoring are NOT yet active.
"""
from __future__ import annotations

import ast
import hashlib
import logging
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2]
SOURCE_FILES={
    "native_glfc":("GLFC/src/GLFC.py","GLFC_model"),
    "native_lga":("LGA/LGA.py","LGA"),
}


class _CPUDeviceOnly(ast.NodeTransformer):
    """In tests ONLY, translate source .cuda(...) into .to('cpu')."""
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=="cuda":
            node.func.attr="to"
            node.args=[ast.Constant(value="cpu")]
            node.keywords=[]
        return node


@lru_cache(maxsize=None)
def original_memory_methods(method,*,cpu_test=False):
    if method not in SOURCE_FILES:
        raise ValueError(f"Unknown original entropy/herding source {method}")
    rel,cls=SOURCE_FILES[method]
    path=ROOT/rel
    tree=ast.parse(path.read_text(),filename=str(path))
    candidates=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls]
    if len(candidates)!=1:
        raise RuntimeError("Authors original learner class unavailable")
    names={"update_new_set","entropy_signal","_construct_exemplar_set",
           "_reduce_exemplar_sets"}
    subset=[n for n in candidates[0].body
            if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in subset}!=names:
        raise RuntimeError(f"Authors {method} entropy and herding not complete")
    entropy=[n for n in tree.body
             if isinstance(n,ast.FunctionDef) and n.name=="entropy"]
    if len(entropy)!=1:
        raise RuntimeError("Authors entropy helper missing")
    bridge=ast.Module(body=entropy+[ast.ClassDef(
        name="SourceMemory",bases=[],keywords=[],body=subset,
        decorator_list=[])],type_ignores=[])
    if cpu_test:
        bridge=_CPUDeviceOnly().visit(bridge)
    ast.fix_missing_locations(bridge)
    namespace={"torch":torch,"nn":nn,"np":np,"F":F,
               "model_to_device":lambda m,p,d:m.to(
                   "cpu" if cpu_test else f"cuda:{int(d)}")}
    exec(compile(bridge,str(path),"exec"),namespace)
    return namespace["SourceMemory"]


class _CompactLogits(nn.Module):
    """Present the author's contiguous seen-class logits, without future IDs."""
    def __init__(self,model,known):
        super().__init__()
        self.core=model
        self.register_buffer("known",torch.tensor(known,dtype=torch.long),
                             persistent=False)
    def forward(self,x):
        output=self.core(x)
        return output.index_select(1,self.known)


class _CurrentTrain(Dataset):
    def __init__(self,train,indices,classes):
        self.train=train
        self.indices=list(dict.fromkeys(int(i) for i in indices))
        self.map={int(c):i for i,c in enumerate(classes)}
        if not self.indices:
            raise ValueError("No local TRAIN examples for original entropy")
        if any(int(train.targets[i]) not in self.map for i in self.indices):
            raise RuntimeError("Original entropy encountered an unseen class")
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        idx=self.indices[i]
        x,label=self.train[idx]
        return idx,x,self.map[int(label)]


class _PrivateSourceImages:
    """Supply only local examples already arrived, never future/client data."""
    def __init__(self,train,allowed):
        self.train=train
        self.allowed=tuple(sorted(set(map(int,allowed))))
        self.cache={}
    def get_image_class(self,cls):
        cls=int(cls)
        ids=[i for i in self.allowed if int(self.train.targets[i])==cls]
        if not ids:
            raise RuntimeError(f"Original herding cannot read unseen class {cls}")
        from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
        mean=torch.tensor(NORMALIZATION_MEAN,dtype=torch.float32).view(3,1,1)
        std=torch.tensor(NORMALIZATION_STD,dtype=torch.float32).view(3,1,1)
        arrays=[]
        for idx in ids:
            pixels,label=self.train[idx]
            if int(label)!=cls:
                raise RuntimeError("Bad original TRAIN image label")
            if tuple(pixels.shape)!=(3,32,32):
                raise ValueError("Authors CIFAR memory needs 32x32 RGB images")
            raw=((pixels.detach().cpu().float()*std+mean).clamp(0,1)*255).round()
            arrays.append(raw.byte().permute(1,2,0).contiguous().numpy())
        images=np.stack(arrays)
        self.cache[cls]=(ids,images)
        return images
    def original_index(self,cls,pixels):
        ids,images=self.cache[int(cls)]
        # Original selection returns RGB arrays, not their dataset indices.
        # Compare bytes to map the author's exact exemplar back into the
        # index-addressed private archive without re-running its herding.
        matches=np.flatnonzero(np.all(images==pixels,axis=(1,2,3)))
        if len(matches)<1:
            raise RuntimeError("Original herding selected an image not in private TRAIN")
        return ids[int(matches[0])]


def original_before_local_task(method,model,client,bank,cfg,device,active_pair):
    """Execute *original* update_new_set before each actual client local task."""
    if method not in SOURCE_FILES:raise ValueError(method)
    new=tuple(sorted(map(int,active_pair or ())))
    if len(new)!=2:
        raise RuntimeError("Source memory adaptation requires exactly two local classes")
    known=tuple(sorted(map(int,client.observed_classes)))
    if not set(new).issubset(known):
        raise RuntimeError("Unseen class in original memory trigger")
    if not set(client.current_indices).issubset(client.seen_indices):
        raise RuntimeError("Current original entropy examples not locally arrived")
    from .continual import _FixedBatchSampler
    data=_CurrentTrain(bank.train,client.current_indices,known)
    batch=_FixedBatchSampler(len(data),cfg.batch_size,
                             cfg.seed+client.client_id*1009+
                             client.local_experience*31,shuffle=False)
    loader=DataLoader(data,batch_sampler=batch,num_workers=0)
    learned=client.native_memory_state or {}
    last_classes=tuple(learned.get("last_classes",()))
    learned_classes=list(learned.get("learned_classes",()))
    source=original_memory_methods(method,cpu_test=(device.type=="cpu"))
    obj=source()
    obj.device=0 if device.index is None else device.index
    obj.model=_CompactLogits(model,known).to(device)
    obj.last_entropy=float(learned.get("last_entropy",0.))
    obj.last_class=list(last_classes) if last_classes else None
    obj.current_class=list(new)
    obj.train_loader=loader
    obj.learned_numclass=len(learned_classes)
    obj.learned_classes=list(learned_classes)
    obj.memory_size=cfg.memory_per_class*max(1,len(learned_classes)+len(last_classes))
    obj.exemplar_set=[]
    provider=_PrivateSourceImages(bank.train,client.seen_indices)
    obj.train_dataset=provider
    for cls in learned_classes:
        saved=list(client.private_memory.get(int(cls),()))
        # Reconstruct retained source numpy arrays in original class order.
        provider.get_image_class(cls)
        known_ids,images=provider.cache[int(cls)]
        by_id={n:images[pos] for pos,n in enumerate(known_ids)}
        if any(i not in by_id for i in saved):
            raise RuntimeError("Original memory had an image beyond local TRAIN")
        obj.exemplar_set.append([by_id[i] for i in saved])
    obj.signal=False

    def compute_class_mean(self,images,transform):
        # GLFC source runs all 450 pixels in one GPU forward; LGA slices in
        # 50s. The requested physical 64 batch adapts only chunking, not
        # normalized feature-mean/herding arithmetic. Return [N,512].
        from .models import NORMALIZATION_MEAN,NORMALIZATION_STD
        mean=torch.tensor(NORMALIZATION_MEAN,device=device).view(1,3,1,1)
        std=torch.tensor(NORMALIZATION_STD,device=device).view(1,3,1,1)
        self.model.eval()
        pieces=[]
        with torch.no_grad():
            for first in range(0,len(images),cfg.batch_size):
                chunk=torch.from_numpy(np.array(images[first:first+cfg.batch_size],
                                                 copy=True)).to(device)
                norm=(chunk.permute(0,3,1,2).float()/255.-mean)/std
                valid=norm.shape[0]
                if valid<cfg.batch_size:
                    padding=torch.arange(cfg.batch_size-valid,device=device)%valid
                    norm=torch.cat([norm,norm.index_select(0,padding)],dim=0)
                z=model.features(norm)[:valid]
                if z.ndim!=2 or z.shape[1]!=512:
                    raise RuntimeError("GLFC/LGA source herding requires 512-d features")
                pieces.append(F.normalize(z,dim=1).detach().cpu().numpy())
        vectors=np.concatenate(pieces,axis=0)
        return np.mean(vectors,axis=0),vectors

    obj.compute_class_mean=MethodType(compute_class_mean,obj)
    obj._get_train_and_test_dataloader=lambda classes,mix: loader
    # Important: the ORIGINAL update_new_set method handles entropy trigger,
    # source old-class retention, reduction, herding and the two data passes.
    source.update_new_set(obj)
    if len(obj.learned_classes)!=len(obj.exemplar_set):
        raise RuntimeError("Original exemplar/class ordering diverged")
    selected={}
    for cls,images in zip(obj.learned_classes,obj.exemplar_set):
        provider.get_image_class(cls)
        selected[int(cls)]=[
            provider.original_index(cls,image) for image in images]
    client.private_memory=selected
    from .federated_cl import combined_memory
    client.memory=combined_memory(client.private_memory,client.received_memory)
    client.native_memory_state={
        "last_classes":list(new),"learned_classes":list(map(int,obj.learned_classes)),
        "last_entropy":float(obj.last_entropy),
        "last_signal":bool(obj.signal),
        "last_trigger_experience":(int(client.local_experience)
                                   if obj.signal else
                                   int(learned.get("last_trigger_experience",-1))),
        "retained_per_class":{int(k):len(v) for k,v in selected.items()},
        "source":"original_GLFC_LGA_update_new_set",
    }
    return dict(client.native_memory_state)
