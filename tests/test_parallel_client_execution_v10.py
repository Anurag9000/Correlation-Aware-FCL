"""One independent model per client; all clients run Task 1 before Task 2.

These tests prove true concurrent worker scheduling and task-order barriers.
Hardware-level CUDA kernel overlap depends on the GPU and is not asserted.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import perf_counter, sleep
from types import SimpleNamespace
import torch
import pytest

from cafcl import parallel_clients
from cafcl.continual import LocalState
from cafcl.config import RunConfig
from cafcl.models import create_model


def _fixture(n=5):
    cfg=RunConfig(dataset='synthetic',model='test_linear',num_clients=n,
        stream_protocol=True,local_tasks_per_round=2,classes_per_local_task=2,
        method='replay',batch_size=1024,share_continual_state=True)
    init=create_model('test_linear',100,allow_test_model=True).state_dict()
    clients={cid:LocalState(cid) for cid in range(n)}
    for client in clients.values():
        client.receive(init,version=0)
    bank=SimpleNamespace(num_classes=100)
    device=torch.device('cpu')
    runtimes=parallel_clients.create_client_runtimes(
        cfg,bank,clients,device)
    arrivals={}
    for cid in clients:
        arrivals[cid]=[
            {'indices':[cid], 'class_ids':(cid*2,cid*2+1),
             'global_stage':0,'is_recurrence':False},
            {'indices':[cid+50], 'class_ids':(cid*2+20,cid*2+21),
             'global_stage':0,'is_recurrence':False},
        ]
    return cfg,bank,clients,device,runtimes,arrivals


def test_all_five_clients_have_separate_live_model_objects():
    cfg,bank,clients,device,runtimes,arrivals=_fixture()
    assert len(runtimes)==5
    assert len({id(x.model) for x in runtimes.values()})==5
    assert all(clients[cid].model_version==0 for cid in clients)


def test_all_five_task_one_workers_overlap_then_all_task_two_workers(monkeypatch):
    cfg,bank,clients,device,runtimes,arrivals=_fixture()
    lock=Lock()
    evidence=[]
    def fake_train(model,client,bank,cfg,device,round_id,**kw):
        with lock:
            evidence.append(('start',client.client_id,
                             client.local_experience,perf_counter(),id(model)))
        sleep(.07+.007*client.client_id)
        with lock:
            evidence.append(('end',client.client_id,
                             client.local_experience,perf_counter(),id(model)))
        classes=kw['active_pair_classes']
        return {'weights':model.state_dict(),'examples':1,'steps':1,
                'class_update_steps':dict.fromkeys(classes,1),
                'class_counts':dict.fromkeys(classes,1)}
    monkeypatch.setattr(parallel_clients,'local_train',fake_train)
    with ThreadPoolExecutor(max_workers=cfg.num_clients) as pool:
        details,trained,barriers=parallel_clients.train_round_task_barriers(
            runtimes=runtimes,executor=pool,clients=clients,bank=bank,cfg=cfg,
            device=device,round_id=0,current_stage=0,
            scenario=SimpleNamespace(stages=[[0,1]]),new_events=arrivals)
    assert len(details)==10
    assert sorted(set(trained))==list(range(5))
    assert [x['task_ordinal'] for x in barriers]==[1,2]
    assert all(x['client_ids']==list(range(5)) for x in barriers)
    assert all(x['completed_count']==5 for x in barriers)
    assert all(x['overlap_window_seconds']>.01 for x in barriers)
    task1=[x for x in evidence if x[2]==0]
    task2=[x for x in evidence if x[2]==1]
    assert len(task1)==10 and len(task2)==10
    assert max(x[3] for x in task1 if x[0]=='end')<=min(
        x[3] for x in task2 if x[0]=='start')
    assert len({x[4] for x in evidence})==5
    assert all(c.pending_upload_steps==2 for c in clients.values())
    assert all(c.model_version==0 for c in clients.values())
    assert all(len(c.local_training_events)==2 for c in clients.values())


def test_partial_final_round_only_schedules_actual_tasks(monkeypatch):
    cfg,bank,clients,device,runtimes,arrivals=_fixture()
    for cid in (3,4):
        arrivals[cid]=arrivals[cid][:1]
    def fake_train(model,client,bank,cfg,device,round_id,**kw):
        classes=kw['active_pair_classes']
        return {'weights':model.state_dict(),'examples':1,'steps':1,
                'class_update_steps':dict.fromkeys(classes,1),
                'class_counts':dict.fromkeys(classes,1)}
    monkeypatch.setattr(parallel_clients,'local_train',fake_train)
    with ThreadPoolExecutor(max_workers=cfg.num_clients) as pool:
        details,trained,barriers=parallel_clients.train_round_task_barriers(
            runtimes=runtimes,executor=pool,clients=clients,bank=bank,cfg=cfg,
            device=device,round_id=0,current_stage=0,
            scenario=SimpleNamespace(stages=[]),new_events=arrivals)
    assert [v['client_ids'] for v in barriers]==[[0,1,2,3,4],[0,1,2]]
    assert len(details)==8
    assert clients[3].pending_upload_steps==1
    assert clients[4].pending_upload_steps==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable in CI')
def test_real_cuda_clients_have_separate_streams():
    cfg,bank,clients,device,runtimes,arrivals=_fixture()
    cuda_runtimes=parallel_clients.create_client_runtimes(
        cfg,bank,clients,torch.device('cuda'))
    assert len({id(v.stream) for v in cuda_runtimes.values()})==5
    assert all(v.stream is not None for v in cuda_runtimes.values())
