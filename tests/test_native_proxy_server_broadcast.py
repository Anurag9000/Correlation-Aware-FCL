"""Exact original GLFC/LGA proxy teacher broadcast is accepted-only."""
import copy
import torch
from torch import nn
import pytest
from cafcl.continual import LocalState
from cafcl.federated_cl import FederatedCLState


def test_source_proxy_state_is_downloaded_only_by_synchronized_client():
    server=FederatedCLState()
    a,b=LocalState(0),LocalState(1)
    a.last_learning_method=b.last_learning_method='native_glfc'
    a.native_teacher_classes={0,1}
    b.native_teacher_classes={2,3}
    original={'weight':torch.tensor([[1.,2.]])}
    server.native_proxy_state=dict(
        best_model_1=None,best_model_2=original,
        best_perf=.4,server_round=2,reconstructed_images=80,
        new_set_label=[0,1,2,3])
    # A selected client gets the method-specific server teacher.
    a.receive({'weight':torch.tensor([[0.,0.]])},1)
    server.broadcast_to(a,'native_glfc')
    assert a.received_native_proxy['best_model_2'] is not original
    server.assert_complete_download(a)
    # Hidden B has neither downloaded nor overwritten local history.
    assert b.received_native_proxy is None
    assert b.native_teacher_classes=={2,3}
    a.received_native_proxy['best_model_2']['weight']+=10
    with pytest.raises(RuntimeError,match='Corrupted original proxy'):
        server.assert_complete_download(a)


def test_proxy_transmitted_history_preserves_server_round_and_best_perf():
    state=FederatedCLState()
    a=LocalState(0)
    a.last_learning_method='native_lga'
    state.native_proxy_pending_receipts=[]
    state.native_proxy_history=[dict(
        round=2,newly_accepted=0,buffered=2,
        ran_source_dataloader=False)]
    assert state.statistics()['original_proxy_accepted_backlog']==0
    assert state.statistics()['original_proxy_source_steps']==0
