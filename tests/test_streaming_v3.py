"""Source-grounded CIFAR-100 streaming FCL invariants, without downloading data."""
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from cafcl.config import RunConfig
from cafcl.streaming import make_task_schedule,compile_stream_scenario,OVERLAP_KINDS,select_offline_cohort
from cafcl.full_matrix import expand,manifest,base,METHODS,AGGS
from cafcl.data import get_data_bank
from cafcl.simulator import run_experiment
from cafcl.class_indexed import canonical_active_loss


def cfg(**changes):
    return replace(base(0),**changes)


def grouped(schedule):
    return {(r,c,t):labels for r,c,t,labels,_ in schedule}


def test_exact_user_10_client_first4_then2_schedule():
    events=make_task_schedule(100,cfg())
    x=grouped(events)
    assert len(events)==50
    assert {r for r,_,_,_,_ in events}=={0,1,2,3}
    assert x[0,0,0]==(0,1) and x[0,0,1]==(2,3)
    assert x[0,1,0]==(4,5) and x[0,1,1]==(6,7)
    assert x[0,4,0]==(16,17) and x[0,4,1]==(18,19)
    assert x[1,0,0]==(40,41)
    assert x[1,1,0]==(42,43)
    assert x[3,9,0]==(98,99)
    assert all(t==0 for r,_,t,_,_ in events if r>0)
    assert Counter(cls for _,_,_,labels,_ in events for cls in labels)==Counter(range(100))


@pytest.mark.parametrize('kind',OVERLAP_KINDS)
def test_every_overlap_geometry_is_valid_distinct_and_covers_all_classes(kind):
    # With one task/round later, 100 labels and 10 clients cannot finish
    # a 100%-inter-same-round overlap with the final singleton budget.
    # Five clients supplies a geometrically feasible counterpart.
    n_clients=5 if kind=='inter_same_round' else 10
    c=cfg(num_clients=n_clients,stream_overlap_kind=kind,
          stream_overlap_fraction=(0 if kind=='none' else 1.))
    a=make_task_schedule(100,c);x=grouped(a)
    assert all(len(set(labels))>=2 for _,_,_,labels,_ in a)
    assert {cls for _,_,_,labels,_ in a for cls in labels}==set(range(100))
    occurrences=defaultdict(list)
    for r,cid,t,labels,_ in a:
        for cls in labels: occurrences[cls].append((r,cid,t))
    duplicate=[v for v in occurrences.values() if len(v)>1]
    if kind=='none':
        assert not duplicate
    else:
        assert duplicate
    if kind=='intra_same_round':
        assert any(len({cid for _,cid,_ in v})==1 and len({r for r,_,_ in v})==1
                   and len({t for _,_,t in v})>1 for v in duplicate)
    if kind=='inter_same_round':
        assert any(len({cid for _,cid,_ in v})>1 and len({r for r,_,_ in v})==1
                   for v in duplicate)
    if kind=='intra_cross_round':
        assert any(len({cid for _,cid,_ in v})==1 and len({r for r,_,_ in v})>1
                   for v in duplicate)
    if kind=='inter_cross_round':
        assert any(len({cid for _,cid,_ in v})>1 and len({r for r,_,_ in v})>1
                   for v in duplicate)


@pytest.mark.parametrize('clients',(5,10,20))
def test_fixed_four_then_two_only(clients):
    events=make_task_schedule(100,cfg(num_clients=clients))
    assert sum(len(labels) for _,_,_,labels,_ in events)==100
    assert len({c for _,_,_,labels,_ in events for c in labels})==100
    per_round=Counter((r,cid) for r,cid,_,_,_ in events)
    assert all(n<= (2 if r==0 else 1) for (r,cid),n in per_round.items())
    assert all(t==0 for r,cid,t,labels,_ in events if r>0)


@pytest.mark.parametrize('classes,tasks',((2,1),(2,3),(4,1),(4,2),(4,3)))
def test_old_task_length_options_are_disallowed(classes,tasks):
    with pytest.raises(ValueError,match='Only two initial'):
        make_task_schedule(100,cfg(classes_per_local_task=classes,local_tasks_per_round=tasks))


class FakeDataset:
    def __init__(self):self.targets=[c for c in range(100) for _ in range(100)]
    def __len__(self):return len(self.targets)

class FakeBank:
    def __init__(self):
        self.train=FakeDataset();self.test=FakeDataset();self.num_classes=100
        self.coarse_by_fine={c:c//5 for c in range(100)}


def test_training_indices_partitioned_with_label_overlap_no_raw_duplication():
    bank=FakeBank()
    for kind in OVERLAP_KINDS:
        num_clients=5 if kind=='inter_same_round' else 10
        sc=compile_stream_scenario(bank,cfg(num_clients=num_clients,
                                stream_overlap_kind=kind,
                                stream_overlap_fraction=0 if kind=='none' else 1.))
        seen=[idx for a in sc.arrivals for idx in a.indices]
        assert len(seen)==len(set(seen))==10000
        assert not sc.duplicate_sample_ids
        assert not sc.unused_sample_ids
        assert sc.stages[-1]==[]  # deliberate terminal communication-only flush
        assert set(sc.label_holders)==set(range(100))


def test_intentional_raw_overlap_recorded_and_causal_control():
    bank=FakeBank()
    sc=compile_stream_scenario(bank,cfg(num_clients=5,
              stream_overlap_kind='inter_same_round',
              stream_overlap_fraction=1.,sample_overlap_fraction=.5))
    x=[i for a in sc.arrivals for i in a.indices]
    assert len(x)>len(set(x))
    assert sc.duplicate_sample_ids


def test_two_round_offline_client_returns_and_has_accumulated_all_classes():
    c=cfg(participation=.8,offline_duration_rounds=2)
    selected=[set(select_offline_cohort(c,r,4,list(range(8)))) for r in range(3)]
    hidden=set(range(10))-selected[0]
    assert len(hidden)==2
    assert selected[0]==selected[1]
    assert hidden<=selected[2]   # recovered at round 2
    a=make_task_schedule(100,c)
    for cid in hidden:
        uploaded={cls for r,client,t,labels,_ in a if client==cid and r<=2 for cls in labels}
        assert len(uploaded)==8  # 4+2+2 classes accumulated before reconnection


def test_active_multi_class_softmax_only_uses_active_rows():
    logits=torch.tensor([[1.,2.,3.,4.,5.]],requires_grad=True)
    loss=canonical_active_loss(logits,torch.tensor([2]),(2,3,4))
    loss.backward()
    assert logits.grad[0,0]==0 and logits.grad[0,1]==0
    assert all(logits.grad[0,j].abs()>0 for j in (2,3,4))


def test_synthetic_entire_reconnect_and_final_flush(tmp_path):
    c=RunConfig(dataset='synthetic',model='test_linear',stream_protocol=True,class_pair_protocol='strict',
         num_clients=2,classes_per_local_task=2,local_tasks_per_round=2,
         stages=1,rounds_per_stage=1,participation=.5,offline_duration_rounds=1,
         method='agem',aggregator='fedavg',batch_size=1024,
         memory_per_class=2,job_id='stream_smoke',seed=6)
    run_experiment(c,str(tmp_path),verbose=False)
    h=json.loads((tmp_path/'stream_smoke'/'metrics.json').read_text())
    assert len(h)==3
    assert set(h[0]['locally_trained_clients'])=={0,1}
    assert len(h[0]['selected_clients'])==1
    hidden=list(set((0,1))-set(h[0]['selected_clients']))[0]
    assert h[0]['client_unsent_steps'][str(hidden)]>0
    assert hidden in h[1]['selected_clients']
    assert len(h[1]['uploaded_accumulated_classes'][str(hidden)])>=4
    assert h[-1]['terminal_communication_only_flush'] is True
    assert h[-1]['client_unsent_steps']=={'0':0,'1':0}
    assert set(h[-1]['globally_integrated_classes'])==set(range(10))
    ck=torch.load(tmp_path/'stream_smoke'/'model_final.pt',weights_only=False)
    assert all(ck['clients'][cid]['memory'] for cid in (0,1))


def test_full_suite_retired_cifar10_and_20_class_and_cover_all_methods():
    with pytest.raises(ValueError):list(expand('cifar10',seeds=(0,),families=['balanced']))
    configs=list(expand('cifar100',seeds=(0,),families=['balanced']))
    assert configs and len(configs)<=11*9*6*3
    assert all(c.dataset=='cifar100' and c.stream_protocol and
               c.class_pair_protocol=='strict' for _,c in configs)
    assert len({(c.method,c.aggregator) for _,c in configs})==99
    assert {c.stream_overlap_kind for _,c in configs}==set(OVERLAP_KINDS)
    assert all(c.batch_size==64 for _,c in configs)

@pytest.mark.parametrize('method',('sgd','gem','agem','replay','ewc','si',
                                    'er_ace','der','derpp','mas','lwf'))
def test_every_client_learning_method_streams_offline_and_reconnects(tmp_path,method):
    c=RunConfig(dataset='synthetic',model='test_linear',stream_protocol=True,class_pair_protocol='strict',
         num_clients=2,classes_per_local_task=2,local_tasks_per_round=2,
         stages=1,rounds_per_stage=1,participation=.5,offline_duration_rounds=1,
         method=method,aggregator='fedavg',batch_size=1024,
         memory_per_class=2,job_id='stream_'+method,seed=5)
    result=run_experiment(c,str(tmp_path),verbose=False)
    assert result['status']=='completed'
    h=json.loads((tmp_path/c.job_id/'metrics.json').read_text())
    assert h[-1]['terminal_communication_only_flush']
    assert set(h[-1]['globally_integrated_classes'])==set(range(10))
    assert all(row['locally_trained_count']>=1 for row in h[:-1])
    assert all(v==0 for v in h[-1]['client_unsent_steps'].values())
    ck=torch.load(tmp_path/c.job_id/'model_final.pt',weights_only=False)
    assert all(len(ck['clients'][cid]['local_training_events'])>=2 for cid in (0,1))
    if method in ('gem','agem','replay','er_ace','der','derpp'):
        assert all(ck['clients'][cid]['memory'] for cid in (0,1))
