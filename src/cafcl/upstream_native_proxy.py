"""Unmodified author GLFC/LGA proxy-server methods, extracted from source.

The frozen source's gradient2label, reconstruction, dataloader, monitor,
f_monitor and model_back functions are compiled directly via AST. This file
supplies only their dependency namespace and a seed-consistent compatibility
encoder. It does not silently replace gradient inversion with synthetic
data, a normalizing flow, cached images or a handcrafted detector.

This is a source mechanism component, NOT evidence that the entire federated
runner has been rewired to invoke the complete proxy-server lifecycle.
Original GLFC/LGA methods assume enough reconstructed 20/class images
to form at least one 64-image drop_last monitoring batch. That assumption
does not always hold in the user's two-class streams. A later protocol
adapter must explicitly handle the insufficient-pool case, never feed
unaccepted client gradients, and preserve the original monitoring decision.
"""
from __future__ import annotations

import ast
import copy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
 'native_glfc':('GLFC/src/ProxyServer.py','GLFC/src/proxy_data.py'),
 'native_lga':('LGA/ProxyServer.py','LGA/proxy_data.py'),
}
AUTHOR_METHODS={
 'gradient2label','reconstruction','dataloader',
 'monitor','f_monitor','model_back'
}
# The author implementation has a separate constructor, but the
# caller's original model and device are attached via a compatibility
# adapter rather than importing ambiguous root-level "myNetwork" modules.


class _CPUDeviceOnly(ast.NodeTransformer):
    def visit_Call(self,node):
        self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr=='cuda':
            node.func.attr='to'
            node.args=[ast.Constant(value='cpu')]
            node.keywords=[]
        return node


@lru_cache(maxsize=None)
def original_proxy_class(method,*,cpu_test=False):
    if method not in SOURCES:raise ValueError(method)
    path=ROOT/SOURCES[method][0]
    mod=ast.parse(path.read_text(),filename=str(path))
    original=[n for n in mod.body if isinstance(n,ast.ClassDef)
              and n.name=='proxyServer']
    if len(original)!=1:raise RuntimeError('Original proxyServer not found')
    wanted=[n for n in original[0].body if isinstance(n,ast.FunctionDef)
            and n.name in AUTHOR_METHODS]
    # f_monitor is LGA only, never insert a replacement into GLFC.
    required=AUTHOR_METHODS if method=='native_lga' else AUTHOR_METHODS-{'f_monitor'}
    if {n.name for n in wanted}!=required:
        raise RuntimeError(f'Original {method} proxyServer methods missing')
    cls=ast.ClassDef(name='SourceProxy',bases=[],keywords=[],
                     body=wanted,decorator_list=[])
    module=ast.Module(body=[cls],type_ignores=[])
    if cpu_test:module=_CPUDeviceOnly().visit(module)
    ast.fix_missing_locations(module)
    def model_to_device(model,parallel,device):
        if parallel:raise RuntimeError('Source proxy expected single GPU')
        return model.to(torch.device('cpu') if cpu_test
                        else torch.device(f'cuda:{int(device)}'))
    env={'torch':torch,'nn':nn,'F':F,'np':np,'optim':optim,'copy':copy,
         'transforms':transforms,'Image':Image,'Variable':Variable,
         'DataLoader':DataLoader,'model_to_device':model_to_device}
    exec(compile(module,str(path),'exec'),env)
    return env['SourceProxy']


@lru_cache(maxsize=None)
def original_proxy_data_class(method):
    if method not in SOURCES:raise ValueError(method)
    path=ROOT/SOURCES[method][1]
    tree=ast.parse(path.read_text(),filename=str(path))
    matches=[n for n in tree.body if isinstance(n,ast.ClassDef)
             and n.name=='Proxy_Data']
    if len(matches)!=1:raise RuntimeError('Original Proxy_Data missing')
    sub=ast.Module(body=matches,type_ignores=[])
    ast.fix_missing_locations(sub)
    env={'torch':torch,'np':np,'Image':Image}
    exec(compile(sub,str(path),'exec'),env)
    return env['Proxy_Data']


def source_gradient_labels(method,receipts,device):
    """Execute the actual author's gradient2label, never trust claimed IDs."""
    if method not in SOURCES:raise ValueError(method)
    if not receipts:return []
    source=original_proxy_class(method,cpu_test=device.type=='cpu')()
    source.pool_grad=[r['gradients'] for r in receipts]
    ids=[int(i) for i in source.gradient2label()]
    if len(ids)!=len(receipts):raise RuntimeError('Original gradient label count mismatch')
    for value,receipt in zip(ids,receipts):
        if value!=int(receipt['cls']):
            raise RuntimeError(
                f'Original gradient2label contradicts uploaded class {value}!={receipt["cls"]}')
        if receipt.get('method')!=method:
            raise RuntimeError('Proxy refused foreign-method prototype receipt')
    return ids


def construct_source_proxy(method,model,seed,device,*,prior=None,radius=None,ep_g=0):
    """Bind exactly the original ProxyServer's required state, preserving resumes.

    source.encode_model is the original initial LeNet at the *same*
    deterministic source initialization as clients. No gradients are
    reconstructed before an acknowledged communication upload.
    """
    if method not in SOURCES:raise ValueError(method)
    from .upstream_native_prototypes import (
        original_source_le_net,_Physical64SourceClassifier)
    cls=original_proxy_class(method,cpu_test=device.type=='cpu')
    proxy=cls()
    proxy.Iteration=(250 if method=='native_glfc' else 250)
    proxy.num_image=20
    proxy.learning_rate=.03
    proxy.device=0 if device.index is None else device.index
    proxy.model=copy.deepcopy(model).to(device).eval()
    if method=='native_lga':
        proxy.model.radius=float(radius) if radius is not None else 0.0
        proxy.args=type('ProxyArgs',(),{'tasks_global':1,'proxy_init':0,'img_size':32})()
        proxy.ep_g=int(ep_g)
    # Each source reconstruction forward receives one logical image;
    # the physical-64 wrapper preserves the original logical output,
    # CE100 and the original LeNet's tensor gradient shapes.
    original_encoder=original_source_le_net(method,seed,device)
    proxy.encode_model=_Physical64SourceClassifier(original_encoder)
    proxy.pool_grad=None
    proxy.best_model_1=None
    proxy.best_model_2=None
    proxy.best_perf=0.
    proxy.cur_perf=0.
    proxy.new_set=[]
    proxy.new_set_label=[]
    if prior is not None:
        proxy.best_perf=float(prior.get('best_perf',0.))
        proxy.new_set=copy.deepcopy(prior.get('new_set',[]))
        proxy.new_set_label=copy.deepcopy(prior.get('new_set_label',[]))
        if prior.get('best_model_1') is not None:
            first=copy.deepcopy(model)
            first.load_state_dict(prior['best_model_1'])
            proxy.best_model_1=first.eval().to(device)
        if prior.get('best_model_2') is not None:
            second=copy.deepcopy(model)
            second.load_state_dict(prior['best_model_2'])
            proxy.best_model_2=second.eval().to(device)
    return proxy


def proxy_snapshot(proxy):
    """Serializable source model_back + reconstructed visual evidence."""
    before,after=proxy.model_back()
    state=lambda m:({k:v.detach().cpu().clone()
                     for k,v in m.state_dict().items()} if m is not None else None)
    return dict(
        best_model_1=state(before),best_model_2=state(after),
        best_perf=float(proxy.best_perf),
        new_set=copy.deepcopy(proxy.new_set),
        new_set_label=copy.deepcopy(proxy.new_set_label),
        reconstructed_images=sum(map(len,proxy.new_set)),
        source='original_proxyServer.model_back',
    )
