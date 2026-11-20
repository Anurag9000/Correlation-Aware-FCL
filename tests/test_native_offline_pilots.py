from __future__ import annotations
import torch
from torch import nn
from cafcl.native_primary import make_native_jobs,SUITES
from cafcl.primary_probe import audited_plan
from cafcl.native_methods import native_loss,weighted_bce,feat_gsa

def test_all_three_paired_pilot_schedules():
    for name,(method,model) in SUITES.items():
        full,hidden=make_native_jobs(name,seed=0)
        assert (full.method,full.model)==(method,model)
        assert (hidden.method,hidden.model)==(method,model)
        assert full.participation==1 and hidden.participation==.8
        assert hidden.stream_offline_policy=='rotate_once'
        assert full.classwise_head_merge is False
        assert full.batch_size==hidden.batch_size==64
        assert full.memory_per_class==hidden.memory_per_class==50
        assert full.optimizer_state_policy=='persist'
        assert full.momentum==hidden.momentum==.9
        assert full.weight_decay==hidden.weight_decay==.0005
        plan=audited_plan((full,hidden))
        assert plan['training_intervals']==9
        assert plan['total_jobs']==2
        assert plan['expected_server_exemplars_after_flush']==5000
        assert len(plan['offline_schedule'])==9
        assert set(x['offline_client'] for x in plan['offline_schedule'])==set(range(5))
        assert all(len(x['online_clients'])==4 for x in plan['offline_schedule'])
        assert set(map(int,range(100)))==set(range(100))

class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(3*8*8,16),nn.Tanh())
        self.classifier=nn.Linear(16,100)
    def forward(self,x):return self.classifier(self.features(x))

def test_native_objectives_have_finite_gradients():
    torch.manual_seed(11)
    x=torch.randn(64,3,8,8)
    y=torch.tensor([0,1,2,3]*16)
    for method in ('native_glfc','native_lga','native_feat'):
        m=SmallNet()
        prior=SmallNet()
        prior.load_state_dict(m.state_dict())
        loss,logits=native_loss(method,m,x,y,[0,1,2,3],teacher=prior,
                                 old_classes=(0,1))
        assert logits.shape==(64,100)
        assert torch.isfinite(loss)
        loss.backward()
        assert sum(int(p.grad is not None) for p in m.parameters())>0
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)

def test_native_old_teacher_mask_and_future_class_independence():
    x=torch.randn(64,4)
    y=torch.tensor([0,1,2,3]*16)
    z=torch.randn(64,100,requires_grad=True)
    t=torch.randn(64,100)
    loss=weighted_bce(z,y,[0,1,2,3],[0,1],t)
    loss.backward()
    assert torch.all(z.grad[:,4:]==0)
    assert torch.count_nonzero(z.grad[:,:4])>0

def test_native_original_models_forward_without_extra_dependencies():
    from cafcl.models import create_model,MODEL_IDS
    for arch in ('native_glfc_cbam','native_lga_cbam','native_feat_resnet'):
        assert arch in MODEL_IDS
        model=create_model(arch,100)
        model.eval()
        with torch.no_grad():
            out=model(torch.randn(2,3,32,32))
        assert out.shape==(2,100)
