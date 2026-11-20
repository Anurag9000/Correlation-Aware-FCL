"""Run exact FEAT source GSA relation-KL code from pinned upstream."""
import torch
import pytest

from cafcl.upstream_losses import original_feat_gsa_source,original_feat_gsa
from cafcl.native_methods import feat_gsa


def test_original_feat_gsa_source_is_provenance_preserving():
    fn=original_feat_gsa_source()
    assert fn.__name__=='gsa_distill_loss'
    assert fn.__code__.co_filename.endswith('FEAT/methods/FEAT.py')


def test_original_feat_relation_loss_actual_execution_and_backprop(monkeypatch):
    import cafcl.upstream_losses as upstream
    original=upstream.original_feat_gsa_source()
    calls=[]
    def traced(*args,**kwargs):
        calls.append((args,kwargs))
        return original(*args,**kwargs)
    monkeypatch.setattr(upstream,'original_feat_gsa_source',lambda:traced)
    features=torch.randn(64,12,requires_grad=True)
    classifier=torch.randn(100,12,requires_grad=True)
    labels=torch.tensor([4,5]*32)
    got=feat_gsa(features,labels,classifier,{4,5})
    assert len(calls)==1
    expected=original(features,torch.tensor([0,1]*32),classifier[[4,5]].t())
    torch.testing.assert_close(got,expected)
    assert torch.isfinite(got)
    got.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    # The authors explicitly .detach() logPP (prototype relation targets).
    # Consequently GSA is a student feature regularizer and MUST NOT update
    # any classifier/prototype weights through this loss.
    assert classifier.grad is None


def test_feat_no_future_class_access():
    f=torch.randn(64,12,requires_grad=True)
    w=torch.randn(100,12,requires_grad=True)
    y=torch.tensor([4,5]*32)
    loss=original_feat_gsa(f,y,w,{4,5})
    loss.backward()
    assert w.grad is None  # original FEAT logPP target is detached
