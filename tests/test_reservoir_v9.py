"""Per-class 10% Algorithm R reservoir and federated server-archive tests."""
from collections import Counter
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from cafcl.continual import LocalState
from cafcl.config import RunConfig
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.federated_cl import FederatedCLState

def apply(client,indices,labels,rng):
    client.update_memory(indices,labels,0,rng,total_limit=1,
        reservoir_fraction=.10,class_totals=dict(Counter(map(int,labels))))

def test_batch_invariance_and_exact_class_quota():
    labels=np.repeat(np.arange(4),450).tolist()
    a,b=LocalState(0),LocalState(1)
    r1=np.random.default_rng(98765);r2=np.random.default_rng(98765)
    apply(a,list(range(1800)),labels,r1)
    for i in range(0,1800,17):
        apply(b,list(range(i,min(1800,i+17))),labels,r2)
    assert a.private_memory==b.private_memory
    assert all(len(a.private_memory[c])==45 for c in range(4))
    assert sum(map(len,a.private_memory.values()))==180
    assert a.private_reservoir_seen_counts=={c:450 for c in range(4)}
    a.received_memory={9:[9000]}
    apply(a,list(range(25)),labels,r1)
    assert len(a.private_memory[0])==45
    assert a.memory[9]==[9000]

def test_sample_not_biased_toward_first_or_last():
    labels=[0]*100
    used=np.zeros(100,dtype=int)
    for seed in range(300):
        c=LocalState(0)
        apply(c,list(range(100)),labels,np.random.default_rng(seed))
        assert len(c.private_memory[0])==10
        for i in c.private_memory[0]:used[i]+=1
    assert used[:20].mean()>15 and used[-20:].mean()>15
    assert abs(used[:20].mean()-used[-20:].mean())<9

def test_server_only_receives_online_coreservoir_then_rejoin():
    labels=np.repeat(np.arange(3),10).tolist()
    class Train:
        targets=labels
        def __len__(self):return len(labels)
        def __getitem__(self,i):return torch.ones(3,32,32)*.2,int(labels[i])
    bank=SimpleNamespace(train=Train(),num_classes=3)
    state={'a':torch.ones(2)}
    clients=[LocalState(i) for i in range(3)]
    for cid in range(3):
        c=clients[cid]
        c.receive(state,0)
        apply(c,list(range(cid*10,(cid+1)*10)),labels,np.random.default_rng(cid))
        c.pending_upload_classes={cid}
        assert len(c.private_memory[cid])==1
    server=FederatedCLState()
    server.merge_clients(clients[:2],bank,'replay',state,1)
    assert set(server.memory)=={0,1}
    assert len(server.exemplar_payloads)==2
    server.broadcast_to(clients[0],'replay')
    assert clients[2].received_memory=={}
    server.merge_clients([clients[2]],bank,'replay',state,2)
    for c in clients:server.broadcast_to(c,'replay')
    assert set(server.memory)=={0,1,2}
    assert all(set(c.memory)=={0,1,2} for c in clients)
    assert all(len(c.received_example_payloads)==3 for c in clients)

@pytest.mark.parametrize('backbone',('vit','cnn'))
def test_paired_probe_both_use_reservoir_and_share_state(backbone):
    full,partial=make_jobs(backbone=backbone)
    for c in (full,partial):
        assert c.method=='replay' and c.share_continual_state
        assert c.replay_reservoir_fraction==pytest.approx(.10)
        assert c.batch_size==1024 and 'RESERVOIR10_V9' in c.job_id
    plan=audited_plan((full,partial))
    assert plan['total_jobs']==2
    assert plan['private_reservoir_per_class']==45
    assert plan['expected_server_exemplars_after_flush']==4500

def test_reservoir_guardrails():
    with pytest.raises(AssertionError):
        RunConfig(method='agem',replay_reservoir_fraction=.10).validate()
    with pytest.raises(AssertionError):
        RunConfig(method='replay',replay_reservoir_fraction=1.10).validate()

def test_rate_change_separates_job_ids_and_budget():
    full,partial=make_jobs(replay_fraction=.20)
    plan=audited_plan((full,partial))
    assert 'RESERVOIR20_V9' in full.job_id
    assert plan['private_reservoir_per_class']==90
    assert plan['expected_server_exemplars_after_flush']==9000
