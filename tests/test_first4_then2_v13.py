"""v13 invariant: first communication has 4 classes/client, then one class
pair/client/communication; global full CL evidence downloaded to every online
client and used on the first post-sync continual experience.
"""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from cafcl.config import RunConfig
from cafcl.full_matrix import base, METHODS, AGGS
from cafcl.streaming import make_task_schedule, select_offline_cohort
from cafcl.continual import LocalState
from cafcl.federated_cl import FederatedCLState,add_importance
from cafcl.simulator import run_experiment


def test_5_clients_100_class_initial_four_then_eight_two_class_rounds():
    s=make_task_schedule(100,replace(base(0),num_clients=5))
    assert len(s)==50
    r=Counter(x[0] for x in s)
    assert r=={0:10,1:5,2:5,3:5,4:5,5:5,6:5,7:5,8:5}
    assignments={(rnd,cid,task):tuple(labels) for rnd,cid,task,labels,recur in s}
    for cid in range(5):
        assert assignments[0,cid,0]==(4*cid,4*cid+1)
        assert assignments[0,cid,1]==(4*cid+2,4*cid+3)
        for rd in range(1,9):
            first=20+(rd-1)*10+cid*2
            assert assignments[rd,cid,0]==(first,first+1)
    assert Counter(c for *_,labels,recur in s for c in labels)==Counter(range(100))


def test_9_round_offline_rotation_covers_every_client_and_flushes():
    cfg=replace(base(0),num_clients=5,participation=.8,
                stream_offline_policy='rotate_once')
    absent=[]
    for rd in range(9):
        chosen=set(select_offline_cohort(cfg,rd,10,list(range(4))))
        assert len(chosen)==4
        missing=next(iter(set(range(5))-chosen))
        absent.append(missing)
    assert set(absent[:5])==set(range(5))
    assert absent[5:]==absent[:4]
    assert set(select_offline_cohort(cfg,9,10,list(range(4))))==set(range(5))


class TinyBank:
    num_classes=4
    class Train:
        targets=[0,0,1,1,2,2,3,3]
        def __len__(self):return 8
        def __getitem__(self,i):
            v=self.targets[i]
            return torch.ones(3,32,32)*float(v)/8,v
    train=Train()


def client(cid):
    c=LocalState(cid)
    state={'w':torch.tensor([0.,1.])}
    c.receive(state,0)
    c.last_learning_method='sgd'
    return c


def test_all_methods_receive_all_cumulative_state_even_non_matching_method():
    bank=TinyBank()
    clients={i:client(i) for i in range(5)}
    sender=clients[0]
    sender.last_learning_method='derpp'
    sender.private_memory={0:[0,1],1:[2,3]}
    sender.memory_logits={i:torch.arange(4.) for i in (0,1,2,3)}
    sender.pending_upload_classes={0,1}
    sender.pending_upload_examples=4
    sender.seen_indices={0,1,2,3}
    for kind,cid in [('ewc',1),('si',2),('mas',3)]:
        c=clients[cid]
        c.last_learning_method=kind
        c.pending_upload_classes={2,3}
        c.seen_indices={4,5,6,7}
        add_importance(c,kind,{'w':torch.tensor([float(cid),2.])},
                       {'w':torch.tensor([3.,4.])})
    clients[4].last_learning_method='lwf'
    clients[4].pending_upload_classes={2,3}
    server=FederatedCLState()
    first_model={'w':torch.tensor([1.,2.])}
    server.merge_clients(list(clients.values()),bank,'sgd',first_model,1)
    assert len(server.memory_logits)==4
    assert len(server.exemplar_payloads)==4
    assert set(server.importance)=={'ewc','si','mas'}
    assert server.teacher_version==1
    for c in clients.values():
        c.receive(first_model,1)
        server.broadcast_to(c,c.last_learning_method)
        assert c.model_version==1 and c.federated_cl_version==1
        assert set(c.memory)=={0,1}
        assert set(c.received_gradient_supports)=={0,1}
        assert len(c.received_example_payloads)==4
        assert len(c.memory_logits)==4
        assert set(c.received_importance)=={'ewc','si','mas'}
        assert c.lwf_teacher_from_server
        assert c.received_lwf_teacher_version==1
        assert set(c.received_server_class_sample_exposure)==set()
        assert torch.equal(c.lwf_teacher['w'],first_model['w'])
        assert set(c.ewc_fisher)=={'w'}
        assert set(c.si_omega)=={'w'}
        assert set(c.mas_omega)=={'w'}
        for kind in ('ewc','si','mas'):
            torch.testing.assert_close(
                getattr(c,kind+'_fisher' if kind=='ewc' else kind+'_omega')['w'],
                server.importance[kind]['precision']['w'])

    # Server continues accumulating later accepted increments; every online
    # client gets the full updated table. The client left offline does not.
    second=client(10)
    second.last_learning_method='replay'
    second.private_memory={2:[4],3:[6]}
    second.pending_upload_classes={2,3}
    second.seen_indices={4,6}
    m2={'w':torch.tensor([3.,4.])}
    server.merge_clients([second],bank,'replay',m2,2)
    for c in list(clients.values())[:4]:
        c.receive(m2,2)
        server.broadcast_to(c,c.last_learning_method)
        assert set(c.memory)=={0,1,2,3}
        assert server.teacher_version==2
        assert c.received_lwf_teacher_version==2
        assert torch.equal(c.lwf_teacher['w'],m2['w'])
    hidden=clients[4]
    assert hidden.federated_cl_version==1
    assert hidden.model_version==1
    assert set(hidden.memory)=={0,1}
    # A reconnection obtains all historical evidence from both syncs.
    hidden.receive(m2,2)
    server.broadcast_to(hidden,'lwf')
    assert set(hidden.memory)=={0,1,2,3}
    assert hidden.received_lwf_teacher_version==2


@pytest.mark.parametrize('method',(
    'sgd','replay','gem','agem','ewc','si','mas','lwf',
    'er_ace','der','derpp'))
def test_one_pair_after_initial_full_sync_uses_prior_global_state(tmp_path,method):
    cfg=RunConfig(dataset='synthetic',model='test_linear',stream_protocol=True,
        class_pair_protocol='strict',num_clients=2,classes_per_local_task=2,
        local_tasks_per_round=2,stages=1,rounds_per_stage=1,
        participation=1.,epochs=1,batch_size=1024,micro_batch_size=4,
        method=method,aggregator='fedavg',
        memory_per_class=2,importance_samples_per_experience=4,
        job_id='global_next_'+method,seed=17)
    assert run_experiment(cfg,str(tmp_path),verbose=False)['status']=='completed'
    rows=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert len(rows)==3  # 8 classes first; 2 next; final flush
    assert [len(x['task_barriers']) for x in rows]==[2,1,0]
    assert len(rows[0]['local_details'])==4
    assert len(rows[1]['local_details'])==1
    assert rows[0]['server_integrated_class_count']==8
    assert rows[1]['server_integrated_class_count']==10
    assert rows[0]['task_barriers'][0]['client_ids']==[0,1]
    assert rows[0]['task_barriers'][1]['client_ids']==[0,1]
    assert rows[1]['task_barriers'][0]['client_ids']==[0]
    assert rows[1]['local_details'][0]['active_pair_classes']==[8,9]
    ck=torch.load(tmp_path/cfg.job_id/'model_final.pt',weights_only=False)
    server=ck['server_cl_state']
    assert server.integrated_classes==set(range(10))
    for cid,st in ck['clients'].items():
        assert st['federated_cl_version']==server.version
        assert st['model_version']==ck['model_version']
        assert st['received_server_class_sample_exposure']==server.class_sample_exposure
        assert st['received_lwf_teacher_version']==server.teacher_version
        assert set(st['observed_classes'])==set(range(10))
    if method in ('gem','agem','replay','er_ace','der','derpp'):
        assert len(server.memory)>0
        assert rows[1]['local_details'][0]['memory_samples']>=8
    if method in ('ewc','si','mas'):
        assert method in server.importance
    if method=='lwf':
        assert ck['clients'][0]['lwf_teacher'] is not None


def test_first4_then2_is_fixed_across_every_method_server_pair():
    for method in METHODS:
        for aggregator in AGGS:
            config=replace(base(0),num_clients=5,method=method,aggregator=aggregator)
            s=make_task_schedule(100,config)
            groups=Counter((r,cid) for r,cid,*_ in s)
            assert len(groups)==45
            assert all(groups[r,cid]==(2 if r==0 else 1)
                       for r in range(9) for cid in range(5))


def test_synced_state_integrity_rejects_corrupted_exemplars_and_teacher():
    a=client(0)
    a.last_learning_method='replay'
    a.private_memory={0:[0]}
    a.pending_upload_classes={0}
    a.seen_indices={0}
    weights={'w':torch.tensor([2.,3.])}
    server=FederatedCLState()
    server.merge_clients([a],TinyBank(),'replay',weights,1)
    b=client(1)
    b.receive(weights,1)
    server.broadcast_to(b,'sgd')
    server.assert_complete_download(b)
    px,y=b.received_example_payloads[0]
    b.received_example_payloads[0]=(torch.full_like(px,255),y)
    with pytest.raises(RuntimeError,match='corrupt exemplar'):
        server.assert_complete_download(b)
    b.received_example_payloads[0]=(px,y)
    b.lwf_teacher['w'][0]+=1.
    with pytest.raises(RuntimeError,match='LwF teacher mismatch'):
        server.assert_complete_download(b)
