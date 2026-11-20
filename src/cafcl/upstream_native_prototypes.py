"""Execute original GLFC/LGA prototype_mask and proto_grad_sharing source.

The original methods receive only a strict coordinate adapter for sparse
per-client globally indexed classes. The source 50-iteration prototype
optimization, BCE, fixed original LeNet encoder, CE gradients and original
label recovery math remain unchanged. PyTorch neural model *forwards* are
physically padded to the user's batch64 while slicing back to original
logical sample counts, preserving original loss/reconstruction math.

Encoder output is only a PERMUTATION of original 100-class LeNet logits:
the source uses compact class labels, while the gradient of the untouched
encoder weights still identifies the global class. The CrossEntropyLoss
denominator remains the original 100 classes, not compacted probabilities.
A source trigger failure produces zero gradients; we never force it true.

The upstream proxy gradient-INVERSION and model_back paths are not
implemented by this module, so pilot fidelity remains fail-closed.
"""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]
SOURCES={'native_glfc':'GLFC/src/myNetwork.py',
         'native_lga':'LGA/myNetwork.py'}


@lru_cache(maxsize=None)
def authors_encoder_sources(method):
    if method not in SOURCES:raise ValueError(method)
    path=ROOT/SOURCES[method]
    tree=ast.parse(path.read_text(),filename=str(path))
    nodes=[n for n in tree.body if
        (isinstance(n,ast.ClassDef) and n.name=='LeNet') or
        (isinstance(n,ast.FunctionDef) and n.name=='weights_init')]
    if len(nodes)!=2:raise RuntimeError('Original GLFC/LGA LeNet source incomplete')
    bridge=ast.Module(body=nodes,type_ignores=[])
    ast.fix_missing_locations(bridge)
    env={'nn':nn}
    exec(compile(bridge,str(path),'exec'),env)
    return env['LeNet'],env['weights_init']


def original_source_le_net(method,seed,device):
    LeNet,weights_init=authors_encoder_sources(method)
    # Authors' source fixed encoder is independently initialized once;
    # every client and proxy must share the identical frozen-initialization
    # source parameters for their gradient tensors to be comparable.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed)+614_003)
        if method=='native_lga':
            model=LeNet(5,2,num_classes=100)
        else:
            model=LeNet(num_classes=100)
        model.apply(weights_init)
    return model.to(device)


class _Physical64SourceClassifier(nn.Module):
    """Model forward64, source objective remains its original logical B."""
    def __init__(self,source_model):
        super().__init__()
        self.core=source_model
    def forward(self,x):
        valid=len(x)
        if valid<1 or valid>64:raise RuntimeError('Invalid source logical batch')
        if valid<64:
            repeat=torch.arange(64-valid,device=x.device)%valid
            x=torch.cat((x,x.index_select(0,repeat)),dim=0)
        return self.core(x)[:valid]


class _Physical64PermutationLeNet(nn.Module):
    """Original LeNet CE100 under a class-coordinate output permutation."""
    def __init__(self,encoder,global_order):
        super().__init__()
        self.core=encoder
        observed=tuple(map(int,global_order))
        if len(set(observed))!=len(observed) or any(c<0 or c>=100 for c in observed):
            raise ValueError('Invalid original LeNet class permutation')
        rest=tuple(c for c in range(100) if c not in observed)
        self.register_buffer('permutation',torch.tensor(observed+rest,dtype=torch.long),
                             persistent=False)
    def forward(self,x):
        n=len(x)
        if not 1<=n<=64:raise RuntimeError('Invalid LeNet logical batch size')
        if n<64:
            fill=torch.arange(64-n,device=x.device)%n
            x=torch.cat((x,x.index_select(0,fill)),dim=0)
        logits=self.core(x)[:n]
        return logits.index_select(1,self.permutation.to(logits.device))


class _GlobalToCompactSourceImages:
    def __init__(self,source_images,order):
        self.provider=source_images
        self.order=tuple(map(int,order))
    def get_image_class(self,local_class):
        local_class=int(local_class)
        if not 0<=local_class<len(self.order):
            raise ValueError('Source exemplar class not mapped')
        return self.provider.get_image_class(self.order[local_class])


def source_prototype_gradients(method,source,cfg,client,model,device):
    """Invoke real authors' proto_grad_sharing with explicit coordinate map."""
    if method not in SOURCES:raise ValueError(method)
    if not bool(source.signal):
        return []
    classes=tuple(map(int,source.source_global_order))
    current_global=tuple(sorted(
        map(int,client.native_memory_state.get('last_classes',()))))
    if len(current_global)!=2:
        raise RuntimeError('Original prototype sharing requires current class pair')
    if not set(current_global).issubset(classes):
        raise RuntimeError('Source prototype pair includes unseen global label')
    # The original source prototype_mask expects compact classifier labels
    # and a shared 100-way auxiliary encoder; this is a pure permutation
    # of that encoder's 100 logits to preserve global gradient labels.
    source.current_class=[classes.index(c) for c in current_global]
    source.train_dataset=_GlobalToCompactSourceImages(source.train_dataset,classes)
    source.model=_Physical64SourceClassifier(source.model).to(device).eval()
    source.encode_model=_Physical64PermutationLeNet(
        original_source_le_net(method,cfg.seed,device),classes).to(device)
    source.args.encode='lenet'  # published LGA lenet branch
    try:
        original_gradients=source.proto_grad_sharing()
    finally:
        source.current_class=list(current_global)
    if original_gradients is None:
        raise RuntimeError('True authors source trigger produced no gradient')
    if len(original_gradients)!=len(current_global):
        raise RuntimeError('Source gradients and current classes disagree')
    result=[]
    for cls,grad_list in zip(current_global,original_gradients):
        grads=[g.detach().cpu().clone() for g in grad_list]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise RuntimeError('Source prototype encoder gradients nonfinite')
        # The author's original ProxyServer.gradient2label extracts the
        # class from the FINAL LEARNABLE WEIGHT gradient.
        predicted=int(grads[-2].sum(dim=-1).argmin().item())
        if predicted!=cls:
            raise RuntimeError(
                f'Original gradient label recovery mismatch: {predicted}!={cls}')
        result.append(dict(
            client=int(client.client_id),
            local_experience=int(client.local_experience),
            cls=int(cls),
            gradients=grads,
            source=('GLFC/src/GLFC.py' if method=='native_glfc' else
                    'LGA/LGA.py')+':proto_grad_sharing/prototype_mask',
            method=method,
        ))
    return result
