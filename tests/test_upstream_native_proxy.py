"""Direct author-source GLFC/LGA proxy entrypoints and receipt validation."""
import torch
import pytest
from torch import nn
from cafcl.upstream_native_proxy import (
    original_proxy_class,original_proxy_data_class,
    source_gradient_labels,construct_source_proxy,proxy_snapshot)


@pytest.mark.parametrize('method,path',[
    ('native_glfc','GLFC/src/ProxyServer.py'),
    ('native_lga','LGA/ProxyServer.py'),
])
def test_proxy_source_provenance_and_all_original_functions(method,path):
    klass=original_proxy_class(method,cpu_test=True)
    for method_name in ('dataloader','gradient2label','monitor',
                        'reconstruction','model_back'):
        assert getattr(klass,method_name).__code__.co_filename.endswith(path)
    if method=='native_lga':
        assert klass.f_monitor.__code__.co_filename.endswith(path)
    assert original_proxy_data_class(method).__getitem__.__code__.co_filename.endswith(
        'GLFC/src/proxy_data.py' if method=='native_glfc' else 'LGA/proxy_data.py')


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_original_gradient2label_requires_valid_and_accepted_class_claim(method):
    matrix=torch.zeros(100,6)
    matrix[24]=-1.
    receipt=dict(method=method,cls=24,gradients=[
        torch.zeros(2),matrix,torch.zeros(100)])
    assert source_gradient_labels(method,[receipt],torch.device('cpu'))==[24]
    with pytest.raises(RuntimeError,match='contradicts'):
        source_gradient_labels(method,[dict(receipt,cls=1)],torch.device('cpu'))
    with pytest.raises(RuntimeError,match='foreign'):
        source_gradient_labels(method,[dict(receipt,method='native_feat')],
                               torch.device('cpu'))


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_proxy_model_back_is_actual_author_selection(method):
    original=original_proxy_class(method,cpu_test=True)()
    original.best_model_1=nn.Linear(3,2)
    original.best_model_2=nn.Linear(3,2)
    original.best_perf=.42
    original.new_set=[]
    original.new_set_label=[]
    a,b=original.model_back()
    assert a is original.best_model_1 and b is original.best_model_2
    snap=proxy_snapshot(original)
    assert snap['best_perf']==.42
    assert snap['reconstructed_images']==0
    assert snap['best_model_2']['weight'].shape==(2,3)


def test_original_proxy_monitor_executes_on_source_64_image_batch():
    class Fixed(nn.Module):
        def forward(self,x):
            result=torch.zeros((x.size(0),100),device=x.device)
            result[:,17]=10.
            return result
    class Dataset:
        def __len__(self):return 64
        def __getitem__(self,i):
            return torch.zeros((3,32,32)),torch.tensor(17)
    from torch.utils.data import DataLoader
    proxy=original_proxy_class('native_glfc',cpu_test=True)()
    proxy.device=0
    proxy.model=Fixed()
    proxy.monitor_loader=DataLoader(Dataset(),batch_size=64)
    assert float(proxy.monitor())==100.


def test_proxy_source_LGA_requires_exact_original_lenet_seed():
    import copy
    from cafcl.upstream_native_prototypes import original_source_le_net
    # A model with no classifier adaptation is used only to verify
    # deterministic auxiliary LeNet source construction, not proxy training.
    a=original_source_le_net('native_lga',52,torch.device('cpu'))
    b=original_source_le_net('native_lga',52,torch.device('cpu'))
    assert all(torch.equal(x,y) for x,y in zip(a.parameters(),b.parameters()))


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_source_proxy_dataset_keeps_original_accessors_and_fixes_numpy_comparison(method):
    import numpy as np
    from cafcl.upstream_native_proxy import (
        proxy_dataset_adapter_class,original_proxy_test_transform)
    source=proxy_dataset_adapter_class(method)(original_proxy_test_transform(method))
    data=[[np.zeros((32,32,3),dtype=np.uint8)
           for _ in range(20)] for _ in range(4)]
    source.getTestData(data,[0,1,2,3])
    assert len(source)==80
    image,label=source[42]
    assert label==2 and tuple(image.shape)==(3,32,32)
    assert source.__class__.__mro__[1].__getitem__.__code__.co_filename.endswith(
        'GLFC/src/proxy_data.py' if method=='native_glfc' else 'LGA/proxy_data.py')
