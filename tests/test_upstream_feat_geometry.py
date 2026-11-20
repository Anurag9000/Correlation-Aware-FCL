"""Verify actual FEAT original projector/prior source and offline sync."""
import torch
from torch import nn

from cafcl.upstream_feat_geometry import (
    original_feat_geometry,compute_client_feat_ecc,
    consolidate_original_ecc,clone_feat_ecc)


def test_original_ecc_three_functions_have_author_source_filename():
    functions=original_feat_geometry()
    assert set(functions)=={
        '_build_projector_from_W',
        'estimate_ecc_priors',
        'aggregate_ecc_priors_from_clients'}
    for method in functions.values():
        assert method.__code__.co_filename.endswith('FEAT/methods/FEAT.py')


def test_source_ecc_aggregate_and_projector_finite():
    source=original_feat_geometry()
    w=torch.randn(8,3)
    P=source['_build_projector_from_W'](w)
    assert P.shape==(8,8)
    torch.testing.assert_close(P,P.t())
    torch.testing.assert_close(P@P,P,atol=1e-4,rtol=1e-4)
    agg=source['aggregate_ecc_priors_from_clients']([
        {'bar_eH_T':1.,'bar_eT_T':2.,'nT':3},
        {'bar_eH_T':5.,'bar_eT_T':6.,'nT':1}],eps=1e-12)
    assert abs(agg['bar_eH_G']-2)<1e-9
    assert abs(agg['bar_eT_G']-3)<1e-9
    assert agg['nT_sum']==4


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*4*4,8))
        self.classifier=nn.Linear(8,100)
    def forward(self,x):return self.classifier(self.features(x))


class Bank:
    class Train:
        targets=[0]*4+[1]*4+[2]*4+[3]*4
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            return torch.full((3,4,4),float(i)/16),self.targets[i]
    train=Train()


def test_actual_feat_ecc_source_is_called_with_sparse_ids():
    from cafcl.continual import LocalState
    model=Tiny().eval()
    client=LocalState(0)
    client.observed_classes={0,1,2,3}
    client.current_indices=list(range(8,16))
    client.seen_indices=set(range(16))
    client.memory={0:[0,1],1:[4,5]}
    client.local_experience=2
    v=compute_client_feat_ecc(model,client,Bank(),torch.device('cpu'),
                              active_pair=(2,3),batch_size=64)
    assert v is not None
    assert v['priors']['nT']>0
    assert v['PH'].shape==(8,8) and v['PT'].shape==(8,8)
    assert v['new_classes']==[2,3]
    assert v['old_classes']==[0,1]
    aggregate=consolidate_original_ecc([v])
    assert aggregate is not None
    assert aggregate['accepted_clients']==[0]


def test_hidden_client_ecc_not_transferred_prior_to_upload():
    from cafcl.continual import LocalState
    from cafcl.federated_cl import FederatedCLState
    a,b=LocalState(0),LocalState(1)
    source=original_feat_geometry()
    P=source['_build_projector_from_W'](torch.randn(3,2))
    sample=dict(priors={'bar_eH_T':.2,'bar_eT_T':.4,'nT':4},
                PH=P,PT=P,r_H=1,r_T=1,client=1,local_experience=2)
    b.pending_feat_ecc=[sample]
    b.last_learning_method='native_feat'
    class FakeBank:
        train=Bank.train
    state=FederatedCLState()
    state.merge_clients([a],FakeBank(),'native_feat',{'w':torch.tensor([1.])},1)
    assert state.feat_ecc is None
    assert b.pending_feat_ecc==[sample]
    a.receive({'w':torch.tensor([1.])},1)
    a.last_learning_method='native_feat'
    state.broadcast_to(a,'native_feat')
    assert a.received_feat_ecc is None
    state.merge_clients([b],FakeBank(),'native_feat',{'w':torch.tensor([2.])},2)
    assert state.feat_ecc is not None
    assert state.feat_ecc['accepted_clients']==[1]
    b.receive({'w':torch.tensor([2.])},2)
    state.broadcast_to(b,'native_feat')
    assert b.received_feat_ecc is not None
    assert b.pending_feat_ecc==[]


def test_client_receipt_clones_projector_tensors():
    P=torch.eye(4)
    x={'PH':P,'PT':P,'energy':{'nT_sum':2}}
    y=clone_feat_ecc(x)
    y['PH'][0,0]=4
    assert x['PH'][0,0]==1
