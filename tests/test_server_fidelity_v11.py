"""v11 server-algorithm fidelity: numerical oracles, heads, offline state.

Canonical: FedAvg (McMahan 2017), FedAvgM Nesterov (Hsu 2019),
FedAdam FedOpt uncorrected moments (Reddi 2021).
Uniform and the five named class/gradient/participation methods are separate
ablation or custom research strategies, not claimed to be source-paper twins.
"""
from dataclasses import replace
import json
import pytest
import torch

from cafcl.config import RunConfig
from cafcl.continual import LocalState
from cafcl.simulator import (aggregate,anchored_upload,server_optimizer_step,
                             run_experiment)


def _server_cfg(**kwargs):
    """Valid multi-class Task-IL fixtures for server-route integration checks."""
    holders={str(cls):(0,1) for cls in range(10)}
    return RunConfig(dataset='synthetic',model='test_linear',num_clients=2,
        scenario='stationary',holders_override=holders,**kwargs)


def _values():
    base={
        'features.weight':torch.tensor([0.,0.]),
        'classifier.weight':torch.zeros((2,2)),
        'classifier.bias':torch.zeros(2),
        'bn.num_batches_tracked':torch.tensor(0,dtype=torch.long),
    }
    def update(value,count,cid,classes):
        weights={k:(torch.full_like(v,float(value)) if v.is_floating_point()
                    else torch.tensor(10+cid,dtype=torch.long))
                 for k,v in base.items()}
        return {'client_id':cid,'weights':weights,'examples':count,
                'class_counts':classes,'class_update_steps':dict.fromkeys(classes,1)}
    return base,[update(2.,1,0,{0:1}),update(6.,3,1,{1:3})]


def test_canonical_sample_weighted_fedavg_includes_every_output_weight_and_bias():
    base,updates=_values()
    result=aggregate(updates,base,'fedavg')
    for key,value in result.items():
        if value.is_floating_point():
            assert torch.allclose(value,torch.full_like(value,5.))
    # Non-floating BN batch counter has no meaningful floating-point mean;
    # this repo's counter policy takes the max separately.
    assert result['bn.num_batches_tracked'].item()==11


def test_uniform_is_equal_model_mean_not_sample_weighted():
    base,updates=_values()
    result=aggregate(updates,base,'uniform')
    for key,val in result.items():
        if val.is_floating_point():
            assert torch.allclose(val,torch.full_like(val,4.))


def test_fedavgm_nesterov_exact_two_step_oracle_and_classifier():
    base,updates=_values()
    cfg=replace(RunConfig(dataset='synthetic'),server_beta1=.9,server_lr=1.)
    mean=aggregate(updates,base,'fedavg')
    one,state=server_optimizer_step(base,mean,'fedavgm',cfg,{})
    for key,v in one.items():
        if v.is_floating_point():assert torch.allclose(v,torch.full_like(v,9.5))
    # Delta 5; velocity 5; nesterov applied 5 + .9*5 = 9.5.
    second_mean={k:(v+2 if v.is_floating_point() else v)
                 for k,v in one.items()}
    two,new_state=server_optimizer_step(one,second_mean,'fedavgm',cfg,state)
    # Velocity = 0.9*5 + 2 = 6.5; next step 2 + .9*6.5 = 7.85.
    assert torch.allclose(two['classifier.bias'],torch.full((2,),17.35),atol=1e-5)
    assert state['step']==1 and new_state['step']==2


def test_fedadam_original_fedopt_uncorrected_not_pytorch_adam():
    base,updates=_values()
    cfg=replace(RunConfig(dataset='synthetic'),
                server_beta1=.9,server_beta2=.999,server_lr=.2,
                fedadam_epsilon=1e-3)
    mean=aggregate(updates,base,'fedavg')  # Δ = +5 on all float params
    actual,state=server_optimizer_step(base,mean,'fedadam',cfg,{})
    moment=(1-.9)*5.
    variance=.999*(1e-3)**2+(1-.999)*25.
    expected=.2*moment/(variance**.5+1e-3)
    for name,v in actual.items():
        if v.is_floating_point():
            assert torch.allclose(v,torch.full_like(v,expected),atol=1e-6)
    assert state['step']==1
    assert state['v']['classifier.bias'][0].item()==pytest.approx(variance)
    # A bias-corrected Adam step would be much larger and is NOT the oracle.
    assert expected != pytest.approx(1.)


def test_class_balanced_respects_only_server_accepted_historical_exposure():
    base,updates=_values()
    no_history=aggregate(updates,base,'class_balanced')
    assert no_history['features.weight'][0].item()==pytest.approx(4.)
    # Client 0's old class has already contributed 99 TRAIN examples.
    # Scores: 1/(1+99)=.01 for client 0 versus 3/3=1 for client 1.
    prior={0:99}
    weighted=aggregate(updates,base,'class_balanced',class_history_counts=prior)
    expected=(2*.01+6*1.)/1.01
    assert weighted['features.weight'][0].item()==pytest.approx(expected,abs=1e-6)
    # Hidden client never uploaded class 1 previously: no count is present.


def test_all_nine_server_aggregators_are_finite_and_shape_safe():
    base,updates=_values()
    clients={i:LocalState(i) for i in range(2)}
    clients[0].selection_count=1
    clients[1].selection_count=3
    names=('fedavg','uniform','fedavgm','fedadam','class_balanced',
           'inverse_participation','conflict_aware','conflict_projection',
           'retention_aware')
    assert len(names)==9
    for name in names:
        mean=aggregate(updates,base,name,clients=clients,
                       old_gradient=torch.ones(8),
                       class_history_counts={0:3,1:2})
        cfg=RunConfig(dataset='synthetic')
        final,_=server_optimizer_step(base,mean,name,cfg,{})
        assert final.keys()==base.keys()
        for key,v in final.items():
            assert v.shape==base[key].shape
            if v.is_floating_point():
                assert torch.isfinite(v).all(),name


def test_fresh_upload_is_exact_local_model_and_stale_is_labeled_and_rebased():
    client=LocalState(1)
    anchor={'w':torch.tensor([1.,4.])}
    client.receive(anchor,0)
    client.shared_weights={'w':torch.tensor([2.,7.])}
    client.pending_upload_examples=20
    client.pending_upload_steps=2
    client.pending_upload_class_counts={0:10,1:10}
    client.pending_upload_classes={0,1}
    fresh=anchored_upload(client,anchor,current_version=0)
    assert torch.equal(fresh['weights']['w'],client.shared_weights['w'])
    assert fresh['stale_rebased'] is False
    assert fresh['aggregation_input_semantics']=='fresh_local_model'
    current={'w':torch.tensor([10.,20.])}
    delayed=anchored_upload(client,current,current_version=1)
    assert delayed['stale_rebased'] is True
    assert delayed['staleness_versions']==1
    assert delayed['stale_update_scale']==pytest.approx(.5)
    assert delayed['aggregation_input_semantics']=='stale_delta_damped_bn_safe'
    assert torch.allclose(delayed['weights']['w'],torch.tensor([10.5,21.5]))
    with pytest.raises(RuntimeError,match='ahead'):
        anchored_upload(client,current,current_version=-1)


@pytest.mark.parametrize('server_algorithm',('fedavg','uniform','fedavgm','fedadam'))
def test_canonical_server_route_never_uses_classwise_head_fusion(
    server_algorithm,monkeypatch,tmp_path
):
    # Give each client multiple labels in each round so the canonical Task-IL
    # head contract is valid while these tests exercise server behavior.
    from cafcl import simulator
    def forbidden(*args,**kwargs):
        raise AssertionError('Canonical server route overwrote classifier rows')
    monkeypatch.setattr(simulator,'classwise_merge',forbidden)
    cfg=_server_cfg(
        stages=1,rounds_per_stage=1,participation=1.,
        aggregator=server_algorithm,method='sgd',
        # Legacy default stays True for old factorial config hashes;
        # canonical methods must nevertheless ignore this custom option.
        classwise_head_merge=True,job_id='complete_'+server_algorithm)
    outcome=run_experiment(cfg,str(tmp_path),verbose=False)
    assert outcome['status']=='completed'
    metrics=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert all(not row['classwise_head_merge_applied'] for row in metrics)
    ck=torch.load(tmp_path/cfg.job_id/'model_final.pt',map_location='cpu',
                  weights_only=False)
    for client in ck['clients'].values():
        assert client['model_version']==ck['model_version']
        assert all(torch.equal(client['shared_weights'][k],v)
                   for k,v in ck['model_state'].items())


def test_custom_row_aggregation_is_a_separately_identified_experimental_arm(tmp_path):
    cfg=_server_cfg(
        stages=1,rounds_per_stage=1,participation=1.,
        aggregator='class_balanced',method='sgd',
        job_id='custom_head')
    result=run_experiment(cfg,str(tmp_path),verbose=False)
    assert result['status']=='completed'
    metrics=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert metrics[0]['classwise_head_merge_applied'] is True
    assert metrics[0]['aggregation_family']=='repository_specific_custom'


def test_offline_client_unsent_class_exposure_does_not_enter_server_history(tmp_path):
    cfg=_server_cfg(
        stages=1,rounds_per_stage=1,participation=.5,method='replay',
        aggregator='class_balanced',job_id='only_online_counts',
        participation_overrides={'0':(1,)})
    result=run_experiment(cfg,str(tmp_path),verbose=False)
    assert result['status']=='completed'
    metrics=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())[0]
    counts={int(c):v for c,v in metrics['server_class_sample_exposure'].items()}
    accepted=metrics['uploaded_accumulated_classes']
    only_online=set(int(c) for cs in accepted.values() for c in cs)
    assert set(counts)==only_online
    assert sum(counts.values())==sum(
        count for count in counts.values())
    assert set(counts)==set(metrics['globally_integrated_classes'])


def test_inverse_participation_excludes_past_nonsynchronizing_client_counts():
    base,updates=_values()
    updates[0]['examples']=2
    updates[1]['examples']=2
    clients={i:LocalState(i) for i in range(2)}
    clients[0].selection_count=0  # first ever upload
    clients[1].selection_count=3  # three previous accepted uploads
    got=aggregate(updates,base,'inverse_participation',clients=clients)
    # Scores 2/(1+0)=2, 2/(1+3)=.5 -> weights .8 and .2.
    assert got['features.weight'][0].item()==pytest.approx(2*.8+6*.2)


def test_server_acknowledges_only_online_client_dataset_ids(tmp_path):
    cfg=_server_cfg(
        stages=1,rounds_per_stage=1,participation=.5,
        participation_overrides={'0':(1,)},method='sgd',
        aggregator='fedavg',job_id='accepted_only_ids')
    assert run_experiment(cfg,str(tmp_path),verbose=False)['status']=='completed'
    ck=torch.load(tmp_path/cfg.job_id/'model_final.pt',map_location='cpu',
                  weights_only=False)
    server=ck['server_cl_state']
    selected=set(ck['clients'][1]['seen_indices'])
    hidden=set(ck['clients'][0]['seen_indices'])
    assert server.acknowledged_train_indices==selected
    # Only the connected client receives the merged server class-history table.
    assert ck['clients'][1]['received_server_class_sample_exposure']==server.class_sample_exposure
    assert not ck['clients'][0]['received_server_class_sample_exposure']
    assert not server.acknowledged_train_indices.intersection(hidden)
    h=json.loads((tmp_path/cfg.job_id/'metrics.json').read_text())
    assert h[0]['server_acknowledged_train_examples']==len(selected)


def test_retention_server_gradient_never_consults_hidden_client_indices(monkeypatch,tmp_path):
    from cafcl import simulator as sim
    original=sim.class_gradient_representations
    observed_inputs=[]
    def captured(model,bank,classes,device,*,allowed_indices=None,**kwargs):
        observed_inputs.append(set(allowed_indices or ()))
        return original(model,bank,classes,device,allowed_indices=allowed_indices,**kwargs)
    monkeypatch.setattr(sim,'class_gradient_representations',captured)
    cfg=_server_cfg(
        stages=2,rounds_per_stage=1,participation=.5,
        participation_overrides={'0':(1,),'1':(1,)},
        method='sgd',aggregator='retention_aware',job_id='retention_only_ack')
    assert run_experiment(cfg,str(tmp_path),verbose=False)['status']=='completed'
    ck=torch.load(tmp_path/cfg.job_id/'model_final.pt',map_location='cpu',weights_only=False)
    hidden=set(ck['clients'][0]['seen_indices'])
    online=set(ck['clients'][1]['seen_indices'])
    assert observed_inputs
    assert all(s.issubset(online) and s.isdisjoint(hidden) for s in observed_inputs)
