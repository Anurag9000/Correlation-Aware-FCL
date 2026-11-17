"""Regression tests for extended research mechanics and archival contracts."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
import numpy as np
from cafcl.config import RunConfig
from cafcl.continual import LocalState
from cafcl.data import get_data_bank,compile_scenario,class_order
from cafcl.selection import select_clients,effective_participation,participation_budget
from cafcl.simulator import aggregate,run_experiment,server_optimizer_step
from cafcl.causal import pair_interaction,aggregation_leave_one_out
from cafcl.models import create_model
from cafcl.planner import all_supported_suite


@pytest.mark.parametrize('method',['er_ace','der','derpp','mas','lwf','replay','ewc'])
def test_extended_local_methods_execute(tmp_path,method):
    cfg=RunConfig(dataset='synthetic',num_clients=2,stages=2,rounds_per_stage=1,
                  participation=1.,model='linear',batch_size=512,method=method,
                  job_id=f'local_{method}')
    run_experiment(cfg,str(tmp_path),verbose=False)
    h=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert len(h)==2
    assert h[-1]['globally_integrated_classes']
    assert (tmp_path/cfg.job_id/'model_final.pt').is_file()


def test_fedadam_and_fedavgm_zero_parameter_shapes():
    base={'w':torch.tensor([0.,2.])}
    mean={'w':torch.tensor([2.,1.])}
    cfg=RunConfig(dataset='synthetic')
    for method in ('fedadam','fedavgm'):
        out,state=server_optimizer_step(base,mean,method,cfg,{})
        assert out['w'].shape==base['w'].shape
        assert torch.isfinite(out['w']).all()
        assert state['step']==1
        second,next_state=server_optimizer_step(out,mean,method,cfg,state)
        assert torch.isfinite(second['w']).all() and next_state['step']==2


def test_selection_policies_are_budget_exact():
    c={i:LocalState(i) for i in range(6)}
    for i in c:
        c[i].observed_classes={i%3,(i+1)%3}
    for policy in ('balanced','diversity','fairness','stochastic_related','historical_coverage'):
        cfg=RunConfig(dataset='synthetic',num_clients=6,participation=.5,selector=policy)
        selected=select_clients(c,cfg,0,np.random.default_rng(16),old_classes={0,1},new_classes={2},coarse_by_fine={0:0,1:0,2:1})
        assert len(selected)==3 and len(set(selected))==3


def test_participation_schedules():
    cfg=RunConfig(dataset='synthetic',num_clients=20,stages=2,rounds_per_stage=5,
                  participation=.8,participation_schedule='increasing')
    assert effective_participation(cfg,0)<effective_participation(cfg,9)
    assert participation_budget(20,effective_participation(cfg,0))>0


def test_related_synthetic_stage_order():
    bank=get_data_bank('synthetic','unused')
    cfg=RunConfig(dataset='synthetic',stages=2,scenario='related',relation='related')
    s=compile_scenario(bank,cfg)
    byclass={c:i for i,group in enumerate(s.stages) for c in group}
    assert all(byclass[2*i]!=byclass[2*i+1] for i in range(5))


def test_pair_interaction_counterfactual_math():
    values={frozenset():0.,frozenset({1}):1.,
            frozenset({2}):2.,frozenset({1,2}):4.}
    assert pair_interaction(values,[],1,2)==pytest.approx(1.)


def test_causal_aggregation_diagnostics(tmp_path):
    cfg=RunConfig(dataset='synthetic',num_clients=3,stages=2,rounds_per_stage=1,
                  participation=1.,model='linear',causal_diagnostics=True,
                  causal_max_clients=3,job_id='causal_test')
    run_experiment(cfg,str(tmp_path),verbose=False)
    metrics=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert metrics[0]['causal_diagnostics']['status']=='evaluated'
    assert len(metrics[0]['causal_diagnostics']['effects'])==3


def test_checkpoint_can_restore_final_round(tmp_path):
    cfg=RunConfig(dataset='synthetic',num_clients=2,stages=2,rounds_per_stage=1,
                  participation=1.,model='linear',job_id='resume_smoke',checkpoint_every=1)
    rd=tmp_path/'resume_smoke'
    run_experiment(cfg,str(tmp_path),verbose=False)
    final=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    status=rd/'status.json'
    status.write_text(json.dumps({'status':'interrupted'}))
    (rd/'model_final.pt').unlink()
    result=run_experiment(cfg,str(tmp_path),verbose=False)
    assert result['status']=='completed'
    restored=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(final['model_state'][k],restored['model_state'][k])
               for k in final['model_state'])


def test_pending_policy_retains_all_training_data(tmp_path):
    cfg=RunConfig(dataset='synthetic',num_clients=2,stages=2,rounds_per_stage=1,
                  participation=1.,model='linear',job_id='retain_all',pending_policy='retain_all')
    run_experiment(cfg,str(tmp_path),verbose=False)
    ckpt=torch.load(tmp_path/'retain_all'/'model_final.pt',map_location='cpu',weights_only=False)
    assert any(len(c['current_indices'])>len(set(c['current_indices'])) or len(c['current_indices'])>20
               for c in ckpt['clients'].values())


def test_class_relationship_diagnostics():
    from cafcl.diagnostics import (label_jaccard,vector_cosine,
        feature_relationships,diagnostic_gradient_relationships)
    assert label_jaccard({0,1},{0,8})==pytest.approx(1/3)
    assert vector_cosine(torch.tensor([1.,0.]),torch.tensor([0.,1.]))==pytest.approx(0.)
    bank=get_data_bank('synthetic','unused')
    model=create_model('linear',bank.num_classes)
    feature=feature_relationships(model,bank,[0,1],torch.device('cpu'),4)
    gradient=diagnostic_gradient_relationships(model,bank,[0,1],torch.device('cpu'),4)
    assert '0:1' in feature and '0:1' in gradient
    assert -1<=gradient['0:1']<=1


def test_resume_after_interrupted_round_is_identical(tmp_path,monkeypatch):
    import cafcl.simulator as sim
    from dataclasses import replace
    base=RunConfig(dataset='synthetic',num_clients=3,stages=2,rounds_per_stage=2,
                   participation=2/3,model='linear',checkpoint_every=1,
                   job_id='interrupted_midrun')
    original_evaluate=sim.evaluate
    calls={'n':0}
    def fail_once(*args,**kwargs):
        calls['n']+=1
        if calls['n']==2:raise RuntimeError('injected interruption')
        return original_evaluate(*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(sim,'evaluate',fail_once)
        with pytest.raises(RuntimeError,match='injected interruption'):
            sim.run_experiment(base,str(tmp_path),verbose=False)
    assert (tmp_path/'interrupted_midrun'/'checkpoint_latest.pt').is_file()
    resumed=sim.run_experiment(base,str(tmp_path),verbose=False)
    pristine=replace(base,job_id='pristine_control')
    sim.run_experiment(pristine,str(tmp_path),verbose=False)
    a=torch.load(tmp_path/'interrupted_midrun'/'model_final.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'pristine_control'/'model_final.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model_state'][k],b['model_state'][k]) for k in a['model_state'])
    h1=json.loads((tmp_path/'interrupted_midrun'/'metrics.json').read_text())
    h2=json.loads((tmp_path/'pristine_control'/'metrics.json').read_text())
    assert [x['accuracy'] for x in h1]==[x['accuracy'] for x in h2]
    assert resumed['status']=='completed'
