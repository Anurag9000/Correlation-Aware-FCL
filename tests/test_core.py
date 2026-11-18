import json
from dataclasses import replace
import numpy as np
import pytest
import torch
from cafcl.catalogue import CATALOGUE,IMPLEMENTED
from cafcl.config import RunConfig
from cafcl.data import (get_data_bank,compile_scenario,class_order,
                        stratified_train_validation_split,data_partition_manifest)
from cafcl.models import create_model,copy_cpu_state
from cafcl.continual import gem_project,agem_project,LocalState
from cafcl.selection import select_clients
from cafcl.simulator import aggregate,run_experiment
from cafcl.planner import core_suite,all_supported_suite


def test_catalogue_complete():
    assert len(CATALOGUE)==152
    assert set(IMPLEMENTED)<=set(CATALOGUE)

@pytest.mark.parametrize('dataset,count',[('cifar10',423),('cifar100',425)])
def test_plans(dataset,count):
    jobs=all_supported_suite(dataset)
    assert len(jobs)==count
    assert len({x.job_id for x in jobs})==count
    assert all(x.batch_size==512 for x in jobs)
    assert all(x.require_cuda for x in jobs)
    assert len(core_suite(dataset))==60
    for j in jobs:j.validate()


def test_synthetic_data_complete_unique_allocations():
    bank=get_data_bank('synthetic','unused')
    cfg=RunConfig(dataset='synthetic',num_clients=4,stages=2,rounds_per_stage=1,scenario='recurrence')
    scenario=compile_scenario(bank,cfg)
    all_ids=[i for a in scenario.arrivals for i in a.indices]
    assert sorted(all_ids)==list(range(len(bank.train)))
    assert any(a.recurrence for a in scenario.arrivals)


def test_stratified_train_validation_split_is_disjoint_and_deterministic():
    source=get_data_bank('synthetic','unused').train.underlying
    train,validation=stratified_train_validation_split(source,.1,seed=71)
    train2,validation2=stratified_train_validation_split(source,.1,seed=71)
    assert set(train.indices).isdisjoint(validation.indices)
    assert set(train.indices)|set(validation.indices)==set(range(len(source)))
    assert train.indices==train2.indices and validation.indices==validation2.indices
    assert all(validation.targets.count(c)==1 for c in range(10))
    assert all(train.targets.count(c)==11 for c in range(10))
    bank=get_data_bank('synthetic','unused',train_fraction=.2,seed=9)
    manifest=data_partition_manifest(bank,'synthetic',.1,.2)
    assert manifest['train_validation_source_indices_disjoint']
    assert manifest['train_samples']==20
    assert manifest['validation_samples']==10
    assert not manifest['official_test_is_original_cifar_test_split']


def test_semantic_order_puts_cifar10_pairs_across_stages():
    bank=get_data_bank('synthetic','unused')
    # The synthetic set has 2 classes per surrogate group.
    classes=class_order(bank,'related')
    assert set(classes[:5])=={0,2,4,6,8}
    assert set(classes[5:])=={1,3,5,7,9}


def test_gem_projection_two_halfspaces():
    g=torch.tensor([-1.,-2.,3.])
    refs=[torch.tensor([1.,0.,0.]),torch.tensor([0.,1.,0.])]
    z,n,v=gem_project(g,refs)
    assert z.tolist()==pytest.approx([0.,0.,3.],abs=1e-4)
    assert v==2 and n>0
    assert all(torch.dot(z,ref).item()>=-1e-5 for ref in refs)


def test_gem_projection_matches_independent_qp_reference():
    from scipy.optimize import minimize
    gradient=torch.tensor([-2.,-1.,.5],dtype=torch.float64)
    refs=[torch.tensor([1.,0.,0.],dtype=torch.float64),
          torch.tensor([.5,1.,0.],dtype=torch.float64),
          torch.tensor([-.25,1.,0.],dtype=torch.float64)]
    projected,_,_=gem_project(gradient,refs)
    G=torch.stack(refs).numpy();g=gradient.numpy()
    Q=G@G.T;b=G@g
    objective=lambda lam:.5*lam@Q@lam+b@lam
    jacobian=lambda lam:Q@lam+b
    reference=minimize(objective,np.zeros(len(refs)),jac=jacobian,
                       method='L-BFGS-B',bounds=[(0,None)]*len(refs),
                       options={'maxiter':5000,'ftol':1e-14,'gtol':1e-12})
    expected=g+G.T@reference.x
    assert projected.tolist()==pytest.approx(expected,abs=2e-5)


@pytest.mark.parametrize('seed',range(8))
def test_gem_projection_randomized_constraints_match_scipy(seed):
    from scipy.optimize import minimize
    rng=np.random.default_rng(seed)
    count=2+seed%6
    matrix=rng.normal(size=(count,32))
    if seed%4==0 and count>2:
        matrix[-1]=matrix[0]+1e-6*matrix[-1]
    gradient=rng.normal(size=32)
    refs=[torch.as_tensor(row,dtype=torch.float64) for row in matrix]
    projected,_,_=gem_project(torch.as_tensor(gradient,dtype=torch.float64),refs)
    gram=matrix@matrix.T;linear=matrix@gradient
    objective=lambda value:.5*value@gram@value+linear@value
    jacobian=lambda value:gram@value+linear
    reference=minimize(objective,np.zeros(count),jac=jacobian,method='L-BFGS-B',
                       bounds=[(0,None)]*count,
                       options={'maxiter':10000,'ftol':1e-14,'gtol':1e-12})
    expected=gradient+matrix.T@reference.x
    assert projected.tolist()==pytest.approx(expected,abs=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gem_projection_keeps_qp_on_cuda():
    gradient=torch.tensor([-1.,-2.,3.],device='cuda')
    refs=[torch.tensor([1.,0.,0.],device='cuda'),
          torch.tensor([0.,1.,0.],device='cuda')]
    projected,_,_=gem_project(gradient,refs)
    assert projected.device.type=='cuda'
    assert all(torch.dot(projected,ref).item()>=-1e-5 for ref in refs)


def test_gem_nonconflicting_equals_sgd():
    g=torch.tensor([1.,2.])
    z,n,v=gem_project(g,[torch.tensor([1.,0.])])
    assert torch.equal(z,g)
    assert n==0 and v==0


def test_agem_projection():
    z,n,v=agem_project(torch.tensor([-1.,1.]),torch.tensor([1.,0.]))
    assert z.tolist()==pytest.approx([0.,1.],abs=1e-6)
    assert n>0 and v==1


def test_fedavg_weights_and_uniform():
    base={'w':torch.tensor([0.,0.])}
    a=[{'weights':{'w':torch.tensor([2.,4.])},'examples':1},
       {'weights':{'w':torch.tensor([6.,8.])},'examples':3}]
    assert aggregate(a,base,'fedavg')['w'].tolist()==pytest.approx([5.,7.])
    assert aggregate(a,base,'uniform')['w'].tolist()==pytest.approx([4.,6.])


def test_every_aggregator_reports_valid_normalized_client_weights():
    base={'w':torch.tensor([0.,0.])}
    updates=[
        {'client_id':0,'weights':{'w':torch.tensor([1.,0.])},'examples':2,
         'class_counts':{0:2}},
        {'client_id':1,'weights':{'w':torch.tensor([-1.,1.])},'examples':1,
         'class_counts':{1:1}},
    ]
    clients={0:LocalState(0,selection_count=2),1:LocalState(1,selection_count=1)}
    methods=('fedavg','uniform','fedavgm','fedadam','class_balanced',
             'inverse_participation','conflict_aware','conflict_projection','retention_aware')
    for method in methods:
        state,weights=aggregate(updates,base,method,clients=clients,
                                old_gradient=torch.tensor([1.,0.]),return_weights=True)
        assert set(weights)=={0,1}
        assert all(np.isfinite(value) and value>=0 for value in weights.values())
        assert sum(weights.values())==pytest.approx(1.)
        assert all(torch.isfinite(value).all() for value in state.values())


def test_post_sync_and_all_observed_inactive(tmp_path):
    cfg=RunConfig(dataset='synthetic',num_clients=3,stages=2,rounds_per_stage=1,
                  participation=1/3,model='linear',batch_size=512,job_id='test_sync')
    result=run_experiment(cfg,str(tmp_path/'runs'),str(tmp_path/'data'),verbose=False)
    assert result['status']=='completed'
    rd=tmp_path/'runs'/'test_sync'
    history=json.loads((rd/'metrics.json').read_text())
    assert sorted(history[-1]['observed_classes'])==list(range(10))
    assert history[-1]['evaluation_split']=='validation'
    assert history[-1]['evaluation_samples']==10
    assert all(row['evaluation_split']=='validation' for row in history)
    assert history[-1]['selected_count']==1
    summary=json.loads((rd/'summary.json').read_text())
    assert summary['model_execution_device']=='cpu'
    assert summary['final_test_metrics']['evaluation_split']=='test'
    assert summary['final_test_metrics']['evaluation_samples']==60
    ckpt=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    selected=history[-1]['selected_clients']
    for cid in selected:
        client=ckpt['clients'][cid]
        assert client['model_version']==ckpt['model_version']
        for key,value in ckpt['model_state'].items():
            assert torch.equal(client['shared_weights'][key],value)
    assert (rd/'summary.json').exists()


def test_require_cuda_fails_before_creating_run_output(tmp_path,monkeypatch):
    from cafcl import simulator
    monkeypatch.setattr(simulator.torch.cuda,'is_available',lambda:False)
    cfg=RunConfig(dataset='synthetic',require_cuda=True,job_id='cuda_required')
    output=tmp_path/'runs'
    with pytest.raises(RuntimeError,match='CUDA is required'):
        run_experiment(cfg,str(output),verbose=False)
    assert not output.exists()


def test_client_history_gem_only_previous_stages():
    c=LocalState(0)
    c.arrival([1,2],[0],0)
    assert c.first_seen_stage[0]==0
    c.arrival([3],[0],1)
    assert c.first_seen_stage[0]==0
    assert c.current_stage==1


def test_overlapping_client_holders_changes_with_control():
    bank=get_data_bank('synthetic','unused')
    base=RunConfig(dataset='synthetic',num_clients=5,stages=2,rounds_per_stage=1,
                   scenario='overlap',overlap=0.)
    x=compile_scenario(bank,base)
    y=compile_scenario(bank,replace(base,overlap=1.))
    assert all(len(v)==1 for v in x.label_holders.values())
    assert all(len(v)==4 for v in y.label_holders.values())


def test_dirichlet_allocator_is_not_uniform():
    bank=get_data_bank('synthetic','unused')
    cfg=RunConfig(dataset='synthetic',num_clients=5,stages=2,rounds_per_stage=1,
                  scenario='non_iid',dirichlet_alpha=.01)
    x=compile_scenario(bank,cfg)
    counts=[len(a.indices) for a in x.arrivals if a.labels==(0,)]
    assert len(counts)<=5
    assert max(counts)>=5


def test_recurrence_policy_same_vs_different_client():
    bank=get_data_bank('synthetic','unused')
    base=RunConfig(dataset='synthetic',num_clients=5,stages=2,rounds_per_stage=1,
                   scenario='recurrence')
    same=compile_scenario(bank,replace(base,recurrence_client_policy='same'))
    diff=compile_scenario(bank,replace(base,recurrence_client_policy='different'))
    a=next(x for x in same.arrivals if x.recurrence and x.labels==(0,))
    b=next(x for x in diff.arrivals if x.recurrence and x.labels==(0,))
    assert a.client_id!=b.client_id


def test_gem_full_simulation(tmp_path):
    # First stage creates local memory; second stage can activate GEM constraints.
    cfg=RunConfig(dataset='synthetic',num_clients=2,stages=2,rounds_per_stage=1,
                  participation=1.,model='linear',batch_size=512,method='gem',
                  job_id='gem_integration')
    run_experiment(cfg,str(tmp_path),str(tmp_path/'data'),verbose=False)
    h=json.loads((tmp_path/'gem_integration'/'metrics.json').read_text())
    assert len(h)==2
    assert all(row['evaluation_split']=='validation' for row in h)
    assert all(c['memory_samples']>=0 for row in h for c in row['local_details'])
