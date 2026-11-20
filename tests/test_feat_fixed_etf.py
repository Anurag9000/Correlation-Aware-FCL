"""Simplex-ETF structural tests for the FEAT paper-derived head."""
import torch
import torch.nn.functional as F

from cafcl.models import FixedSimplexETFHead,create_model


def test_fixed_simplex_etf_geometry_100_classes():
    torch.manual_seed(17)
    head=FixedSimplexETFHead(512,100)
    w=head.weight.detach().double()
    gram=w@w.T
    expected=torch.full((100,100),-1/99,dtype=torch.double)
    expected.fill_diagonal_(1)
    torch.testing.assert_close(gram,expected,atol=1e-5,rtol=1e-5)
    assert head.weight.requires_grad is False
    assert head.weight.grad is None
    assert not any(p.requires_grad for p in head.parameters())


def test_fixed_etf_head_receives_feature_gradients_but_not_prototype_grads():
    head=FixedSimplexETFHead(12,4)
    features=torch.randn(64,12,requires_grad=True)
    targets=torch.tensor([0,1,2,3]*16)
    loss=F.cross_entropy(head(features),targets)
    loss.backward()
    assert features.grad is not None and torch.count_nonzero(features.grad)>0
    assert head.weight.grad is None


def test_feat_v3_uses_original_backbone_and_reconstructed_fixed_prototypes():
    model=create_model('native_feat_etf',100)
    assert isinstance(model.classifier,FixedSimplexETFHead)
    assert model.features.original.__class__.__name__=='ResNet'
    assert 'paper_reconstructed_fixed_ETF' in model.backbone_description['implementation']
    assert model.classifier.weight.requires_grad is False
    with torch.no_grad():
        assert model(torch.randn(2,3,32,32)).shape==(2,100)


def test_native_feat_pilot_now_selects_etf_not_trainable_linear():
    from cafcl.native_primary import make_native_jobs
    full,offline=make_native_jobs('feat')
    assert full.model==offline.model=='native_feat_etf'
    assert 'SOURCE_CORE_V5' in full.job_id and 'SOURCE_CORE_V5' in offline.job_id
    assert full.stream_client_chain and offline.stream_client_chain
    assert (full.client_training_concurrency or full.num_clients)==5
    assert (offline.client_training_concurrency or offline.num_clients)==5
    assert offline.stream_offline_policy=='random'
