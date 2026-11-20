"""Original FEAT source FLDataSelector and private offline retention."""
from types import SimpleNamespace
from collections import Counter
import torch
from torch import nn

from cafcl.upstream_feat_replay import original_feat_selector,source_feat_select_local


def test_original_author_selector_methods_really_are_source_functions():
    cls=original_feat_selector()
    for name in ('mask_local_data','compute_local_leverage_scores',
                 'aggregate_leverage_scores','sample_data'):
        assert cls.__dict__[name].__code__.co_filename.endswith('FEAT/methods/FEAT.py')


def test_original_feat_selector_full_qr_svd_sampling():
    torch.manual_seed(81)
    Source=original_feat_selector()
    feats=torch.randn(12,5)
    fn=Source(1,[feats],4)
    masked=fn.mask_local_data()
    assert len(masked)==1 and masked[0].shape==(12,5)
    lev=fn.compute_local_leverage_scores(masked)
    p=fn.aggregate_leverage_scores(lev)
    assert p.shape==(12,)
    torch.testing.assert_close(p.sum(),torch.tensor(1.))
    ids=fn.sample_data(p)
    assert len(ids[0])==4 and all(0<=int(i)<12 for i in ids[0])


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*4*4,7))
    def forward(self,x):return self.features(x)


class FakeBank:
    num_classes=4
    class Train:
        targets=[0]*6+[1]*6
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            return torch.ones(3,4,4)*float(i+1)/20,self.targets[i]
    train=Train()


def test_private_feat_retention_uses_exact_source_and_preserves_tickets():
    from cafcl.continual import LocalState
    torch.manual_seed(3)
    state=LocalState(0)
    cfg=SimpleNamespace(batch_size=64,memory_per_class=3,seed=4,dataset='synthetic')
    chosen=source_feat_select_local(
        FakeModel(),state,FakeBank(),cfg,torch.device('cpu'),
        list(range(12)),(0,1))
    assert set(chosen)=={0,1}
    assert {c:len(ids) for c,ids in chosen.items()}=={0:3,1:3}
    for c,ids in chosen.items():
        assert all(FakeBank.train.targets[i]==c for i in ids)
        assert state.feat_selection_tickets[c]==ids
        assert set(state.memory[c])==set(ids)
    assert state.feat_selection_receipts[0]['source'].endswith('FLDataSelector')


def test_feat_selector_fails_closed_on_cross_client_future_samples():
    from cafcl.continual import LocalState
    cfg=SimpleNamespace(batch_size=64,memory_per_class=3,seed=4)
    state=LocalState(1)
    import pytest
    with pytest.raises(RuntimeError,match='only new class pair'):
        source_feat_select_local(FakeModel(),state,FakeBank(),cfg,
                                 torch.device('cpu'),list(range(12)),(1,2))
