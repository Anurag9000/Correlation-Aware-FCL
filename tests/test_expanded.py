"""Regression tests for the November 2026 research-closure expansion."""
from dataclasses import replace
import json
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from cafcl.config import RunConfig
from cafcl.data import DataBank,TinySynthetic,RestrictedLabels,compile_scenario
from cafcl.domains import GlobalDomainTest,ClientDomainDataset
from cafcl.selection import select_clients
from cafcl.continual import LocalState
from cafcl.simulator import aggregate,run_experiment
from cafcl.evaluation import evaluate
from cafcl.causal import aggregation_subset_report


def conf(**kwargs):
    return replace(RunConfig(dataset='synthetic',num_clients=5,stages=2,
                             rounds_per_stage=2,model='linear'),**kwargs)


def test_manual_participation_exact_budget():
    states={i:LocalState(i) for i in range(5)}
    cfg=conf(participation=.4,participation_overrides={'1':(2,4)})
    assert select_clients(states,cfg,1,np.random.default_rng(5))==[2,4]
    with pytest.raises(ValueError):
        select_clients(states,replace(cfg,participation_overrides={'1':(2,)}),1,
                       np.random.default_rng(5))


def test_stage_participation():
    from cafcl.selection import effective_participation
    cfg=conf(participation_schedule='stage',stage_participation=(1.,.2))
    assert effective_participation(cfg,0)==1.
    assert effective_participation(cfg,3)==.2


def test_overlapping_raw_samples_only_when_explicit():
    bank=DataBank(TinySynthetic(),TinySynthetic(False),10,{i:i//2 for i in range(10)})
    base=conf(scenario='overlap',overlap=1)
    no=compile_scenario(bank,base)
    yes=compile_scenario(bank,replace(base,sample_overlap_fraction=.5))
    assert not no.duplicate_sample_ids
    assert yes.duplicate_sample_ids
    assert len({i for a in yes.arrivals for i in a.indices})==len(bank.train)


def test_controlled_exact_holders_and_class_dose():
    bank=DataBank(TinySynthetic(),TinySynthetic(False),10,{i:i//2 for i in range(10)})
    cfg=conf(holders_override={'0':(0,3),'1':(0,)},
             class_holder_weights={'0':(.9,.1)},
             class_sample_fraction={'2':.5})
    scenario=compile_scenario(bank,cfg)
    assert scenario.label_holders[0]=={0,3}
    assert scenario.label_holders[1]=={0}
    assert len(scenario.unused_sample_ids)==6
    assert len({i for a in scenario.arrivals for i in a.indices})==len(bank.train)-6


def test_recurrence_round_delay_and_period():
    bank=DataBank(TinySynthetic(),TinySynthetic(False),10,{i:i//2 for i in range(10)})
    cfg=conf(scenario='recurrence',rounds_per_stage=5,
             recurrence_round_delay=2,recurrence_period=2,recurrence_repeats=2)
    sc=compile_scenario(bank,cfg)
    times={a.round_id for a in sc.arrivals if a.recurrence and 0 in a.labels}
    assert times=={2,4}
    assert len([i for a in sc.arrivals for i in a.indices])==len(bank.train)


def test_domain_arrivals_and_heldout_slices():
    bank=DataBank(TinySynthetic(),TinySynthetic(False),10,{i:i//2 for i in range(10)})
    cfg=conf(scenario='domain',rounds_per_stage=1)
    sc=compile_scenario(bank,cfg)
    assert sc.stages==[list(range(10)),list(range(10))]
    assert any(a.stage==1 for a in sc.arrivals)
    assert len(set(i for a in sc.arrivals for i in a.indices))==len(bank.train)
    test=GlobalDomainTest(bank.test,[(0,0,[0,1]),(1,1,[0,1])])
    assert len(test)==24
    assert test.targets.count(0)==12
    x,_=test[12]
    assert not torch.equal(x,bank.test[0][0])


def test_task_oracle_masks_other_task_logit():
    class Tiny(Dataset):
        targets=[0,2]
        def __len__(self):return 2
        def __getitem__(self,i):return torch.tensor([float(i)]),self.targets[i]
    class Fixed(torch.nn.Module):
        def forward(self,x):
            return torch.tensor([[3.,2.,4.,0.],[5.,0.,3.,2.]],device=x.device)[:len(x)]
    bank=DataBank(Tiny(),Tiny(),4,{})
    ordinary,_=evaluate(Fixed(),bank,{0,2},batch_size=512,device='cpu')
    masked,_=evaluate(Fixed(),bank,{0,2},batch_size=512,device='cpu',
                      task_class_groups=[[0,1],[2,3]])
    assert masked['accuracy']>ordinary['accuracy']
    assert masked['task_id_at_inference'] is True


def test_class_balanced_aggregation():
    base={'weight':torch.tensor([0.])}
    ups=[{'client_id':0,'weights':{'weight':torch.tensor([2.])},'examples':100,
          'class_counts':{0:100}},
         {'client_id':1,'weights':{'weight':torch.tensor([8.])},'examples':10,
          'class_counts':{1:10}}]
    std=aggregate(ups,base,'fedavg')['weight'].item()
    balanced=aggregate(ups,base,'class_balanced')['weight'].item()
    assert std==pytest.approx((200+80)/110)
    assert balanced==pytest.approx(5.)


def test_conflict_projection_and_retention_aggregate():
    base={'w':torch.tensor([0.,0.])}
    ups=[{'client_id':0,'weights':{'w':torch.tensor([2.,0.])},'examples':1},
         {'client_id':1,'weights':{'w':torch.tensor([-1.,2.])},'examples':1}]
    corrected=aggregate(ups,base,'conflict_projection')['w']
    assert torch.isfinite(corrected).all()
    retained=aggregate(ups,base,'retention_aware',old_gradient=torch.tensor([1.,0.]))['w']
    assert torch.isfinite(retained).all()


def test_subset_counterfactual_evaluates_all_combinations():
    bank=DataBank(TinySynthetic(),TinySynthetic(False),10,{i:i//2 for i in range(10)})
    from cafcl.models import create_model,copy_cpu_state
    model=create_model('linear',10)
    base=copy_cpu_state(model)
    ups=[]
    for cid in range(3):
        weights={k:v.detach().clone() for k,v in base.items()}
        for k,v in weights.items():
            if v.is_floating_point():v.add_(0.01*cid)
        ups.append({'client_id':cid,'weights':weights,'examples':4})
    report=aggregation_subset_report(model,bank,{0,1},ups,base,'fedavg',512,'cpu',limit=3)
    assert report['status']=='evaluated'
    assert report['evaluated_subsets']==8
    assert len(report['pair_interactions'])==3


def test_synaptic_intelligence_and_hidden_boundaries(tmp_path):
    cfg=conf(num_clients=2,participation=1.,rounds_per_stage=1,method='si',job_id='si_test')
    run_experiment(cfg,str(tmp_path),verbose=False)
    ck=torch.load(tmp_path/'si_test'/'model_final.pt',weights_only=False)
    assert ck['clients'][0]['si_omega'] is not None
    assert all(torch.isfinite(t).all() for t in ck['clients'][0]['si_omega'].values())
    hidden=replace(cfg,job_id='boundary_free',method='gem',boundary_mode='hidden',
                   learning_setting='boundary_free',gem_grouping='class')
    run_experiment(hidden,str(tmp_path),verbose=False)
    with pytest.raises(AssertionError):
        replace(hidden,method='si').validate()


def test_post_sync_domain_and_task_il(tmp_path):
    for scenario,setting in [('domain','domain_il'),('disjoint','task_il')]:
        cfg=conf(num_clients=2,participation=.5,rounds_per_stage=1,
                 scenario=scenario,learning_setting=setting,
                 job_id='check_'+setting)
        run_experiment(cfg,str(tmp_path),verbose=False)
        hist=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
        assert hist[-1]['global_test_includes_inactive']
        if scenario=='domain':
            assert hist[-1]['domain_evaluation_slices']>0
            assert hist[-1]['test_samples']>len(TinySynthetic(False))
        else:assert hist[-1]['task_id_at_inference']


def test_sgd_does_not_allocate_raw_replay_memory(tmp_path):
    cfg=conf(num_clients=2,participation=1.,rounds_per_stage=1,job_id='sgd_memory')
    run_experiment(cfg,str(tmp_path),verbose=False)
    state=torch.load(tmp_path/cfg.job_id/'model_final.pt',weights_only=False)
    assert all(not c['memory'] for c in state['clients'].values())


def test_spurious_patch_and_label_noise_are_deterministic():
    from cafcl.domains import ClientAugmentedDataset,SpuriousTest
    base=TinySynthetic()
    noise=ClientAugmentedDataset(base,0,{},label_noise_rate=1.,num_classes=10,seed=5)
    assert noise[3][1]==noise[3][1]
    assert noise[3][1]!=base[3][1]
    matched=SpuriousTest(base,1.)
    reverse=SpuriousTest(base,1.,reverse=True)
    assert not torch.equal(matched[0][0],reverse[0][0])
    assert torch.equal(matched[0][0][...,4:,4:],reverse[0][0][...,4:,4:])


def test_active_client_memory_reset_ablation(tmp_path):
    for reset in (False,True):
        cfg=conf(num_clients=2,participation=1.,method='gem',rounds_per_stage=1,
                 reset_memory_on_sync=reset,job_id=f'gem_memory_reset_{reset}')
        run_experiment(cfg,str(tmp_path),verbose=False)
        state=torch.load(tmp_path/cfg.job_id/'model_final.pt',weights_only=False)
        sample_count=sum(len(v) for client in state['clients'].values()
                         for v in client['memory'].values())
        assert (sample_count==0) if reset else (sample_count>0)


def test_adaptive_participation_from_local_constraint_feedback():
    from cafcl.selection import effective_participation
    states={i:LocalState(i,last_constraint_violations=(2 if i<2 else 0)) for i in range(4)}
    cfg=conf(num_clients=4,participation=.5,participation_schedule='feedback',
             feedback_threshold=.25,feedback_gain=.5)
    assert effective_participation(cfg,1,states)==.75
    assert effective_participation(cfg,1,{i:LocalState(i) for i in range(4)})==.5


def test_sample_lag_and_multiple_domains_pipeline(tmp_path):
    cfg=conf(num_clients=2,participation=1.,rounds_per_stage=2,
             scenario='shifted_recurrence',recurrence_round_delay=2,
             job_id='shifted_recurrence_smoke')
    run_experiment(cfg,str(tmp_path),verbose=False)
    rows=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert any(r['domain_evaluation_slices']>0 for r in rows)
    assert rows[-1]['test_samples']>=len(TinySynthetic(False))


def test_spurious_and_noise_end_to_end(tmp_path):
    for tag,kw in [('spurious',{'spurious_strength':1.,'spurious_ood_eval':True}),
                   ('noise',{'label_noise_rate':.2})]:
        cfg=conf(num_clients=2,participation=1.,rounds_per_stage=1,
                 job_id='extra_'+tag,**kw)
        run_experiment(cfg,str(tmp_path),verbose=False)
        final=json.loads((tmp_path/cfg.job_id/'summary.json').read_text())['final_metrics']
        if tag=='spurious':
            assert final['spurious_reversed_accuracy'] is not None
        assert 0<=final['accuracy']<=1


def test_research_plans_preserve_20class_cifar100_experiment():
    from cafcl.planner import all_supported_suite
    planned=all_supported_suite('cifar100')
    specific=[x for x in planned if x.dataset=='cifar100_20']
    assert specific
    assert all(x.stages==2 and x.rounds_per_stage==20 and x.batch_size==512 for x in specific)
    assert any(x.holders_override.get('12')==(3,) and x.holders_override.get('13')==(3,)
               for x in specific)
