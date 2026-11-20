"""Original author FedAvg implementations, not generic approximations."""
import copy
from collections import OrderedDict
import torch
import pytest

from cafcl.original_aggregation import SOURCES,upstream_fedavg,original_aggregate
from cafcl.simulator import aggregate


def _uploads():
    return [
        {'client_id':0,'weights':{'w':torch.tensor([3.,9.]),
                                  'bn.num_batches_tracked':torch.tensor(3,dtype=torch.int64)},
         'examples':3},
        {'client_id':1,'weights':{'w':torch.tensor([9.,3.]),
                                  'bn.num_batches_tracked':torch.tensor(9,dtype=torch.int64)},
         'examples':1},
    ]


@pytest.mark.parametrize('source',sorted(SOURCES))
def test_each_pilot_calls_exact_original_aggregation(source,monkeypatch):
    """Instrument the executed author symbol and its outputs."""
    import cafcl.original_aggregation as original
    real=original.upstream_fedavg(source)
    calls=[]
    if source=='fedclass_resnet8':
        original_method=real.update
        def traced(self,*args,**kwargs):
            calls.append('original-fedclass.update')
            return original_method(self,*args,**kwargs)
        monkeypatch.setattr(real,'update',traced)
    else:
        def traced(*args,**kwargs):
            calls.append('original-source-function')
            return real(*args,**kwargs)
        monkeypatch.setattr(original,'upstream_fedavg',lambda x:traced)
    initial={'w':torch.zeros(2),'bn.num_batches_tracked':torch.tensor(0,dtype=torch.int64)}
    got=aggregate(_uploads(),initial,'fedavg',source_method=source)
    assert len(calls)==1,f'Upstream {source} did not actually execute'
    expected=(torch.tensor([4.5,7.5]) if source=='fedclass_resnet8'
              else torch.tensor([6.,6.]))
    torch.testing.assert_close(got['w'],expected)
    assert got['bn.num_batches_tracked'].item() == (4 if source=='fedclass_resnet8' else 6)


@pytest.mark.parametrize('source',sorted(SOURCES))
def test_frozen_authors_upstream_function_symbol(source):
    value=upstream_fedavg(source)
    assert callable(value)
    assert value.__name__==('FedAvg' if source!='native_feat' else 'average_weights')


def test_original_source_rejects_invalid_model_shapes():
    with pytest.raises(RuntimeError,match='shape mismatch'):
        original_aggregate('native_glfc',
            [{'weights':{'w':torch.zeros(3)},'examples':1}],
            {'w':torch.zeros(2)})


def test_generic_fedavg_stays_separate_for_unrelated_baselines():
    u=_uploads()
    got=aggregate(u,{'w':torch.zeros(2),'bn.num_batches_tracked':torch.tensor(0)},'fedavg')
    torch.testing.assert_close(got['w'],torch.tensor([4.5,7.5]))


def test_uniform_paper_fedavg_is_distinct_from_sample_weighted():
    u=_uploads()
    b={'w':torch.zeros(2),'bn.num_batches_tracked':torch.tensor(0,dtype=torch.int64)}
    x=original_aggregate('native_glfc',u,b)
    y=original_aggregate('fedclass_resnet8',u,b)
    assert not torch.equal(x['w'],y['w'])
