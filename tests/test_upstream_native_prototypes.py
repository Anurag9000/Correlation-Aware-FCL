"""Exact author prototype source AST and 100-way global-label semantics."""
import torch
import pytest
from torch import nn
from types import SimpleNamespace

from cafcl.upstream_glfc_lga_memory import original_memory_methods
from cafcl.upstream_native_prototypes import (
    authors_encoder_sources,original_source_le_net,
    _Physical64PermutationLeNet,_Physical64SourceClassifier,
    source_prototype_gradients)


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_original_source_mask_and_encoder_exist_without_reimplementation(method):
    Source=original_memory_methods(method,cpu_test=True)
    assert Source.proto_grad_sharing.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')
    assert Source.prototype_mask.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')
    klass,init=authors_encoder_sources(method)
    assert klass.forward.__code__.co_filename.endswith(
        'GLFC/src/myNetwork.py' if method=='native_glfc' else 'LGA/myNetwork.py')
    assert init.__code__.co_filename.endswith(
        'GLFC/src/myNetwork.py' if method=='native_glfc' else 'LGA/myNetwork.py')


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_original_100way_encoder_permutation_recovers_global_class_gradient(method):
    enc=original_source_le_net(method,101,torch.device('cpu'))
    wrap=_Physical64PermutationLeNet(enc,global_order=(18,47,77,9))
    values=[]
    hook=enc.body.register_forward_pre_hook(
        lambda m,args:values.append(len(args[0])))
    try:
        xx=torch.randn((1,3,32,32))
        result=wrap(xx)
        original=enc(xx)
        torch.testing.assert_close(result[0,:4],original[0,[18,47,77,9]])
        assert values==[64,1]
        g=torch.autograd.grad(
            torch.nn.functional.cross_entropy(result,torch.tensor([2])),
            wrap.parameters())
        assert int(g[-2].sum(-1).argmin())==77
    finally:
        hook.remove()


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_original_source_does_not_generate_untriggered_prototypes(method):
    obj=original_memory_methods(method,cpu_test=True)()
    obj.signal=False
    obj.proto_grad_sharing=obj.proto_grad_sharing.__get__(obj,type(obj))
    client=SimpleNamespace(
        client_id=0,native_memory_state={'last_classes':[0,1]},
        local_experience=0)
    result=source_prototype_gradients(method,obj,
        SimpleNamespace(seed=1),client,nn.Linear(2,2),
        torch.device('cpu'))
    assert result==[]


def test_original_aux_encoder_init_is_client_seed_deterministic():
    a=original_source_le_net('native_glfc',42,torch.device('cpu'))
    b=original_source_le_net('native_glfc',42,torch.device('cpu'))
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))
