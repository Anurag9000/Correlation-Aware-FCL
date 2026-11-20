"""Execute the ORIGINAL authors' FedAvg source for each FCIL pilot.

Only aggregation itself is sourced. The surrounding class-stream policy,
delayed-update rebasing, and model compatibility adaptations are experimental
extensions and do not constitute a complete faithful method reproduction.

GLFC and LGA: exact unmodified FedAvg function bodies from Fed_utils.py.
FEAT: exact unmodified average_weights function body from utils/dmc.py.
FedCLASS CE: exact unmodified FedAvg class from utils/fed_algorithm.py.

Source extraction uses Python AST so the function/class implementation is
executed directly from vendored original bytes; it is not a transcription,
a mathematical rewrite, or a generic shared weighted aggregator.
"""
from __future__ import annotations

import ast
import copy
import importlib.util
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[2]
SOURCES={
    'native_glfc':('GLFC/src/Fed_utils.py','FedAvg','uniform'),
    'native_lga':('LGA/Fed_utils.py','FedAvg','uniform'),
    'native_feat':('FEAT/utils/dmc.py','average_weights','uniform'),
    'fedclass_resnet8':('FedCLASS/utils/fed_algorithm.py','FedAvg','sample_weighted'),
}


@lru_cache(maxsize=None)
def upstream_fedavg(source_method:str):
    if source_method not in SOURCES:
        raise ValueError(f'Unknown original source: {source_method}')
    relative,component,_=SOURCES[source_method]
    path=ROOT/relative
    if not path.is_file():
        raise FileNotFoundError(f'Original upstream source missing: {path}')
    if source_method=='fedclass_resnet8':
        spec=importlib.util.spec_from_file_location(
            '_cafcl_authors_original_fedclass_fedavg',path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.FedAvg
    # The upstream Fed_utils/dmc.py files import unrelated legacy
    # dependencies (and could shadow one another's 'myNetwork' module).
    # Compile only the exact original FedAvg function syntax tree,
    # byte-for-byte sourced from the author's file, without importing
    # unneeded libraries or replacing the author implementation.
    parsed=ast.parse(path.read_text(),filename=str(path))
    matching=[node for node in parsed.body
              if isinstance(node,ast.FunctionDef) and node.name==component]
    if len(matching)!=1:
        raise RuntimeError(f'Expected exactly one upstream {component} in {path}')
    sub=ast.Module(body=[matching[0]],type_ignores=[])
    ast.fix_missing_locations(sub)
    namespace={'copy':copy,'torch':torch}
    exec(compile(sub,str(path),'exec'),namespace)
    return namespace[component]


class _StateModel:
    """Only the PyTorch model interface required by FedCLASS.FedAvg.update."""
    def __init__(self,weights):
        self.weights=weights
    def cpu(self):
        return self
    def state_dict(self):
        return self.weights


def original_aggregate(source_method,uploads,base):
    """Directly execute the authors' aggregation algorithm on accepted uploads.

    The stale-client adapter must independently produce valid candidate
    models *before* this call. No optimizer or CL policy is changed here.
    """
    if not uploads:
        raise RuntimeError('Original aggregator requires accepted uploads')
    if source_method not in SOURCES:
        raise ValueError(source_method)
    keys=set(base)
    if any(set(up['weights'])!=keys for up in uploads):
        raise RuntimeError('Upstream FedAvg model-key mismatch')
    for upload in uploads:
        for name,target in base.items():
            if upload['weights'][name].shape!=target.shape:
                raise RuntimeError(f'Upstream FedAvg shape mismatch: {name}')
    if source_method=='fedclass_resnet8':
        # Run actual FedCLASS/utils/fed_algorithm.py:FedAvg.update.
        lengths=[int(u['examples']) for u in uploads]
        if any(n<=0 for n in lengths):
            raise ValueError('FedCLASS requires positive train-size weights')
        klass=upstream_fedavg(source_method)
        algorithm=klass(lengths,OrderedDict(base))
        result=algorithm.update(
            [_StateModel(OrderedDict(u['weights'])) for u in uploads],
            list(range(len(uploads))),_StateModel(OrderedDict(base)))
    else:
        # GLFC/LGA: FedAvg(w_local), FEAT: average_weights(local_weights)
        fn=upstream_fedavg(source_method)
        result=fn([copy.deepcopy(u['weights']) for u in uploads])
    if set(result)!=keys:
        raise RuntimeError('Original aggregator lost or added parameter keys')
    result={k:result[k].to(dtype=base[k].dtype,device='cpu')
            for k in base}
    for name,tensor in result.items():
        if not torch.isfinite(tensor.float()).all():
            raise RuntimeError(f'Nonfinite original FedAvg result: {name}')
    return result
