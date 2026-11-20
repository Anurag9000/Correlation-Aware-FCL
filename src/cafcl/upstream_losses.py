"""GLFC/LGA published loss methods executed from their immutable source AST.

The original _compute_loss, efficient_old_class_weight and get_one_hot bodies
are compiled and executed without rewriting. A strict label-index adapter
converts sparse per-client global IDs to the contiguous label coordinates
the authors' implementations expect. The adapter changes only the label
coordinate system / available classes, not the original source loss logic.

Other GLFC/LGA author mechanisms (entropy-triggered exemplar selection,
prototype gradients, reconstruction, geometry, model monitor) still need
actual integration and are NOT certified here.

The original .cuda() calls execute unchanged on CUDA. On CPU, where the
original author's GPU-only code cannot execute, only those .cuda calls are
translated into .to('cpu') during isolated AST compilation for synthetic
software tests. The mathematical loss implementation is unchanged; no CPU
mode is used by production pilot launchers.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.autograd import Variable

ROOT=Path(__file__).resolve().parents[2]
MODULES={
    'native_glfc':('GLFC/src/GLFC.py','GLFC_model'),
    'native_lga':('LGA/LGA.py','LGA'),
}


class _CpuDeviceOnly(ast.NodeTransformer):
    """Translate *only* original Tensor.cuda() calls for CPU-only test runs."""
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            node.func.attr='to'
            node.args=[ast.Constant(value='cpu')]
            node.keywords=[]
        return node


@lru_cache(maxsize=None)
def original_loss_core(method,*,cpu_test=False):
    if method not in MODULES:
        raise ValueError(f'No original source loss for {method}')
    rel,name=MODULES[method]
    path=ROOT/rel
    module=ast.parse(path.read_text(),filename=str(path))
    named=[n for n in module.body if isinstance(n,ast.ClassDef) and n.name==name]
    if len(named)!=1:raise RuntimeError(f'Original {name} class missing in {path}')
    original_functions=[
        n for n in named[0].body
        if isinstance(n,ast.FunctionDef) and
        n.name in {'_compute_loss','efficient_old_class_weight'}
    ]
    if set(n.name for n in original_functions)!={'_compute_loss','efficient_old_class_weight'}:
        raise RuntimeError(f'Original source methods incomplete in {path}')
    get_one_hot=[n for n in module.body
                 if isinstance(n,ast.FunctionDef) and n.name=='get_one_hot']
    if len(get_one_hot)!=1:raise RuntimeError('Original one-hot helper missing')
    source_class=ast.ClassDef(name='SourceLoss',bases=[],keywords=[],
                              body=original_functions,decorator_list=[])
    isolated=ast.Module(body=get_one_hot+[source_class],type_ignores=[])
    if cpu_test:
        isolated=_CpuDeviceOnly().visit(isolated)
    ast.fix_missing_locations(isolated)
    env={'torch':torch,'F':F,'Variable':Variable}
    exec(compile(isolated,str(path),'exec'),env)
    return env['SourceLoss']


def original_loss(method,global_logits,global_labels,known_classes,
                  old_classes=(),old_teacher=None,inputs=None):
    """Run exact authors' BCE/teacher formula on compact visible class space.

    Guarantees that each old class precedes new classes in the compact label
    ordering, as both upstream sources expect old teachers in prefix columns.
    The intended global class ID is never changed outside this call.
    """
    if method not in MODULES:raise ValueError(method)
    if global_logits.ndim!=2 or global_labels.ndim!=1:
        raise ValueError('Expected batch logits [B,C] and labels [B]')
    if len(global_logits)!=len(global_labels):
        raise ValueError('Mismatched labels and batch')
    known=set(int(x) for x in known_classes)
    prior=tuple(sorted(known & set(int(x) for x in old_classes)))
    novel=tuple(sorted(known-set(prior)))
    order=prior+novel
    if not order or any(y not in known for y in global_labels.tolist()):
        raise ValueError('Unknown class label in author loss bridge')
    if global_logits.shape[1]<=max(order):
        raise ValueError('Classifier does not contain every visible class')
    index=torch.tensor(order,device=global_labels.device,dtype=torch.long)
    mapping=torch.full((global_logits.shape[1],),-1,
                       device=global_labels.device,dtype=torch.long)
    mapping[index]=torch.arange(len(order),device=global_labels.device)
    remapped=mapping[global_labels.long()]
    if (remapped<0).any():raise ValueError('Unseen label entered author loss')

    Source=original_loss_core(method,cpu_test=not global_logits.is_cuda)
    obj=Source.__new__(Source)
    obj.numclass=len(order)
    obj.learned_classes=list(range(len(prior)))
    obj.task_size=2
    obj.task_id_old=len(prior)//obj.task_size if prior else -1
    obj.device=global_logits.device.index
    obj.args=SimpleNamespace(lam=1.)
    # Model forward is already computed exactly once by the training loop;
    # avoid a second BatchNorm update while feeding the original BCE method.
    compact=global_logits.index_select(1,index)
    obj.model=lambda unused:compact
    if prior and old_teacher is not None:
        if inputs is None:raise ValueError('Old teacher must receive inputs')
        old_idx=torch.tensor(prior,device=inputs.device)
        obj.old_model=lambda x:old_teacher(x).index_select(1,old_idx)
    else:
        obj.old_model=None
    return obj._compute_loss(None,inputs,remapped)


@lru_cache(maxsize=1)
def original_feat_gsa_source():
    """Compile authors' exact FEAT.py:gsa_distill_loss function body."""
    path=ROOT/'FEAT/methods/FEAT.py'
    mod=ast.parse(path.read_text(),filename=str(path))
    definitions=[n for n in mod.body if isinstance(n,ast.FunctionDef)
                 and n.name=='gsa_distill_loss']
    if len(definitions)!=1:
        raise RuntimeError('Published FEAT source missing gsa_distill_loss')
    isolated=ast.Module(body=definitions,type_ignores=[])
    ast.fix_missing_locations(isolated)
    env={'torch':torch,'F':F}
    # The source function has Torch type annotations evaluated on definition.
    exec(compile(isolated,str(path),'exec'),env)
    return env['gsa_distill_loss']


def original_feat_gsa(features,labels,weight,known_classes,tau=.07,eps=1e-8):
    """Run original relation-KL body with compact class-id compatibility.

    The imported FEAT ETF head is unavailable from the official source.
    This compatibility call transposes our standard Linear classifier [C,d]
    into the published relation loss's documented [d,C] prototype orientation.
    This does NOT reproduce the original ETF classifier or whole FEAT method.
    """
    known=tuple(sorted(set(int(c) for c in known_classes)))
    if not known:
        raise ValueError('FEAT cannot compute GSA without known classes')
    ids=torch.as_tensor(known,device=labels.device,dtype=torch.long)
    mapped=torch.full((weight.shape[0],),-1,dtype=torch.long,device=labels.device)
    mapped[ids]=torch.arange(len(ids),device=labels.device)
    y=mapped[labels.long()]
    if (y<0).any():
        raise ValueError('FEAT GSA labels must all be locally known')
    compact_w=weight.index_select(0,ids).t().contiguous()
    return original_feat_gsa_source()(features,y,compact_w,tau=tau,eps=eps)
