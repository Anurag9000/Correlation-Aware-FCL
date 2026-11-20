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
        assert hidden.stream_offline_policy=='random'
        assert full.classwise_head_merge is False
        assert full.batch_size==hidden.batch_size==64
        assert full.memory_per_class==hidden.memory_per_class==50
        assert full.optimizer_state_policy=='persist'
        assert full.momentum==hidden.momentum==.9
        assert full.weight_decay==hidden.weight_decay==.0005
        plan=audited_plan((full,hidden))
        assert plan['training_intervals']==20
        assert plan['total_jobs']==2
        assert plan['expected_server_exemplars_after_flush']==10000
        assert len(plan['offline_schedule'])==20
        assert all(len(x['online_clients'])==4 for x in plan['offline_schedule'])
        assert plan['client_training_concurrency']==5
        assert plan['concurrently_training_client_models_across_both_arms']==10
        assert set(map(int,range(100)))==set(range(100))


def test_bounded_client_execution_is_included_in_native_pair_plan():
    jobs=make_native_jobs('glfc',client_training_concurrency=2)
    assert jobs[0].client_training_concurrency==jobs[1].client_training_concurrency==2
    plan=audited_plan(jobs)
    assert plan['client_execution_mode']=='bounded_concurrent_task_barriers'
    assert plan['client_training_concurrency']==2

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


def test_native_client_continues_through_disconnection_and_replay_uses_64():
    from cafcl.config import RunConfig
    from cafcl.continual import LocalState,local_train
    class Bank:
        class Train:
            targets=[0]*3+[1]*3+[2]*3+[3]*3
            def __len__(self):return len(self.targets)
            def __getitem__(self,i):
                return torch.full((3,8,8),float(self.targets[i])*.08),self.targets[i]
        train=Train()
        num_classes=100
    bank=Bank()
    for method in ('native_glfc','native_lga','native_feat'):
        model=SmallNet()
        cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=2,
            method=method,epochs=1,lr=.01,batch_size=64,
            memory_per_class=2,replay_reservoir_fraction=.5,
            share_continual_state=True)
        client=LocalState(0)
        client.receive(model.state_dict(),0)
        client.local_experience=0
        client.first_seen_local_experience.update({0:0,1:0})
        client.arrival(list(range(6)),(0,1),0)
        shapes=[]
        handle=model.features.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
        try:
            first=local_train(model,client,bank,cfg,torch.device('cpu'),0,
                              experience_end=True,active_pair_classes=(0,1))
            client.shared_weights=first['weights']
            assert first['steps']==1
            assert client.native_teacher is not None
            if method=='native_feat':
                assert set(client.private_memory)=={0,1}
            else:
                # Original GLFC/LGA update_new_set doesn't construct
                # exemplars on the FIRST task (last_class is None).
                assert client.private_memory=={}
                assert client.native_memory_state['learned_classes']==[]
            assert client.model_version==0
            client.local_experience=1
            client.first_seen_local_experience.update({2:1,3:1})
            client.arrival(list(range(6,12)),(2,3),1)
            second=local_train(model,client,bank,cfg,torch.device('cpu'),1,
                               experience_end=True,active_pair_classes=(2,3))
            if method=='native_feat':
                # The author's source trainer makes one physical batch64
                # from current TRAIN plus retained source tickets. It does
                # NOT use generic mixed replay that created two updates.
                assert second['steps']==1
                assert second['source_method'].endswith('FEAT._local_finetune')
                assert set(client.memory)=={0,1,2,3}
            else:
                # If source entropy increase <= 1.2, source legitimately
                # does NOT construct old exemplars. No fake replay.
                assert second['steps']==1
                assert client.native_memory_state['last_classes']==[2,3]
                assert client.native_memory_state['last_signal'] is False
                assert client.memory=={}
            assert set(client.native_teacher_classes)=={0,1,2,3}
            assert client.model_version==0  # No communication: still offline.
            assert shapes and set(shapes)=={64}
        finally:
            handle.remove()

def test_native_method_server_sync_is_acknowledged_only_for_online_clients():
    from cafcl.continual import LocalState
    from cafcl.federated_cl import FederatedCLState
    class Bank:
        class Train:
            targets=[0,0,1,1,2,2,3,3]
            def __len__(self):return 8
            def __getitem__(self,i):return torch.zeros(3,32,32),self.targets[i]
        train=Train()
        num_classes=4
    weights={'w':torch.tensor([1.,2.])}
    for method in ('native_glfc','native_lga','native_feat'):
        online=LocalState(0)
        offline=LocalState(1)
        online.receive(weights,0);offline.receive(weights,0)
        online.last_learning_method=method
        online.private_memory={0:[0,1],1:[2,3]}
        online.pending_upload_classes={0,1}
        online.seen_indices={0,1,2,3}
        offline.native_teacher={'w':torch.tensor([9.,8.])}
        offline.native_teacher_classes={2,3}
        server=FederatedCLState()
        server.merge_clients([online],Bank(),method,weights,1)
        online.receive(weights,1)
        server.broadcast_to(online,method)
        server.assert_complete_download(online)
        assert set(online.native_teacher_classes)=={0,1}
        assert set(online.received_memory)=={0,1}
        assert offline.model_version==0
        assert offline.federated_cl_version==0
        assert offline.native_teacher_classes=={2,3}
        assert not offline.received_memory


def test_all_three_pilot_entrypoints_write_audited_plans(tmp_path):
    import json
    from cafcl import native_primary
    for name in SUITES:
        native_primary.main(['--suite',name,'--seed','0',
                             '--output-dir',str(tmp_path),'--plan-only'])
        design=json.loads((tmp_path/name/'seed0'/('source_v5' if name=='feat' else 'source_v4')/'design.json').read_text())
        assert design['training_intervals']==20
        assert design['expected_server_exemplars_after_flush'] is None
        assert design['expected_server_exemplars_per_class'] is None
        assert design['terminal_communication_only_flush'] is True
        assert design['full_and_offline_same_gpu']==0
        assert design['concurrent_processes']==2
        assert design['total_jobs']==2

def test_pilot_dispatch_spawns_both_arms_concurrently_on_gpu_zero(monkeypatch,tmp_path):
    from cafcl import native_primary as n
    import cafcl.data as data
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(data,'get_data_bank',lambda *args,**kwargs:None)
    def fake_build_report(root,jobs):
        result=root/'comparison'/'comparison.md'
        result.parent.mkdir(parents=True,exist_ok=True)
        result.write_text('Two matched runs,')
        return {'mock_only':True}
    monkeypatch.setattr(n,'build_report',fake_build_report)
    launched=[]
    class FakeProcess:
        pid=1234
        def wait(self):return 0
        def poll(self):return 0
        def terminate(self):raise AssertionError('Successful worker must not terminate')
    def fake_popen(cmd,**kwargs):
        launched.append((cmd,kwargs))
        return FakeProcess()
    monkeypatch.setattr(n.subprocess,'Popen',fake_popen)
    n.main(['--suite','glfc','--output-dir',str(tmp_path),'--no-download','--permit-adapted-legacy'])
    assert len(launched)==2
    assert [cmd[-2:] for cmd,_ in launched]==[['--worker','0'],['--worker','1']]
    assert all(kwargs['env']['CUDA_VISIBLE_DEVICES']=='0' for _,kwargs in launched)
    assert all(kwargs['stderr']==n.subprocess.STDOUT for _,kwargs in launched)
    assert 'GLFC source-method adapter/FedAvg' in (
        tmp_path/'glfc'/'seed0'/'source_v4'/'comparison'/'comparison.md').read_text()
