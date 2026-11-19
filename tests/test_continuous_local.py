"""Regression tests for continuous local CL independent of communication."""
from dataclasses import replace
import json
import torch
import pytest
from cafcl.config import RunConfig
from cafcl.simulator import run_experiment,anchored_upload,PROTOCOL_VERSION
from cafcl.continual import LocalState

# Every supported local method uses the same unconditional local-training loop.
ALL_METHODS=('sgd','agem','gem','replay','ewc','si','er_ace','der','derpp','mas','lwf')


def config(method='agem',**kw):
    holders={str(c):(c%5,) for c in range(10)}
    selection={'0':(1,2,3,4),'1':(1,2,3,4),
               '2':(1,2,3,4),'3':(0,1,2,3)}
    return RunConfig(dataset='synthetic',model='test_linear',num_clients=5,stages=2,
        rounds_per_stage=2,participation=.8,batch_size=1024,
        scenario='disjoint',method=method,aggregator='fedavg',
        holders_override=holders,participation_overrides=selection,
        memory_per_class=8,checkpoint_every=1,**kw)


@pytest.mark.parametrize('method',ALL_METHODS)
def test_hidden_clients_train_locally_and_rejoin_with_history(method,tmp_path):
    cfg=replace(config(method),job_id=f'continuous_{method}')
    out=tmp_path/'runs'
    assert run_experiment(cfg,str(out),verbose=False)['status']=='completed'
    rd=out/cfg.job_id
    hist=json.loads((rd/'metrics.json').read_text())
    assert len(hist)==4
    for i in range(3):
        assert 0 not in hist[i]['selected_clients']
        assert 0 in hist[i]['locally_trained_clients']
        assert '0' not in hist[i]['uploaded_accumulated_steps']
        assert hist[i]['client_unsent_steps']['0']>0
        assert hist[i]['uploaded_count']==4
    ckpt=torch.load(rd/'checkpoint_round_0003.pt',map_location='cpu',weights_only=False)
    c=ckpt['clients'][0]
    assert ckpt['protocol_version']==PROTOCOL_VERSION
    assert c['model_version']==0, 'hidden client must not download updated server model'
    assert c['pending_upload_steps']>0
    assert {0,5}<=set(c['pending_upload_classes'])
    assert [z['class'] for z in c['local_training_events']]==[0,5]
    assert c['local_experience']==1
    assert any(not torch.equal(c['shared_weights'][k],c['sync_anchor'][k])
               for k in c['shared_weights'] if c['shared_weights'][k].is_floating_point())
    if method in {'agem','gem','replay','er_ace','der','derpp'}:
        assert {0,5} <= set(c['memory']), 'continual memory survives hidden rounds'
    if method=='ewc':assert c['ewc_anchor'] is not None
    if method=='si':assert c['si_anchor'] is not None
    if method=='mas':assert c['mas_anchor'] is not None
    if method=='lwf':assert c['lwf_teacher'] is not None
    at_rejoin=hist[3]
    assert 0 in at_rejoin['selected_clients']
    assert set(at_rejoin['uploaded_accumulated_classes']['0'])=={0,5}
    assert at_rejoin['uploaded_since_version']['0']==0
    assert at_rejoin['uploaded_accumulated_steps']['0']>=3
    saved=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    c=saved['clients'][0]
    assert c['pending_upload_steps']==0
    assert c['model_version']==saved['model_version']
    assert all(torch.equal(c['shared_weights'][k],v) for k,v in saved['model_state'].items())


def test_local_sequential_class_events_within_same_round(tmp_path):
    # Ten classes are distributed among five clients: class 0 and class 5
    # arrive to the same client in the same time step. They must be trained
    # sequentially, not treated as one pooled minibatch.
    cfg=replace(config('agem'),job_id='same_round_sequence',stages=1,
        rounds_per_stage=1,participation_overrides={'0':(1,2,3,4)})
    run_experiment(cfg,str(tmp_path),verbose=False)
    saved=torch.load(tmp_path/cfg.job_id/'model_final.pt',map_location='cpu',weights_only=False)
    c=saved['clients'][0]
    assert [r['classes'] for r in c['local_training_events']]==[[0,5]]
    assert [r['local_experience'] for r in c['local_training_events']]==[0]
    assert c['model_version']==0 and c['pending_upload_steps']>0
    assert {0,5}<=set(c['memory'])


def test_rebased_deltas_include_unsent_offline_updates():
    client=LocalState(2)
    client.receive({'w':torch.tensor([2.,2.])},0)
    client.shared_weights={'w':torch.tensor([5.,3.])}
    client.pending_upload_steps=4
    client.pending_upload_examples=20
    client.pending_upload_classes={0,5}
    client.pending_upload_class_counts={0:10,5:10}
    # Server has moved since client downloaded 2,2; private change is 3,1.
    uploaded=anchored_upload(client,{'w':torch.tensor([9.,9.])})
    assert uploaded['weights']['w'].tolist()==pytest.approx([12.,10.])
    assert uploaded['anchor_version']==0
    assert uploaded['accumulated_classes']==[0,5]


def test_completed_runs_from_previous_protocol_are_not_skipped(tmp_path):
    cfg=replace(config('sgd'),job_id='old_protocol')
    rd=tmp_path/cfg.job_id
    rd.mkdir()
    (rd/'status.json').write_text('{"status":"completed"}')
    with pytest.raises(RuntimeError,match='incompatible local-training protocol'):
        run_experiment(cfg,str(tmp_path),verbose=False)


def test_resume_restores_offline_client_model_and_pending_delta(tmp_path):
    cfg=replace(config('agem'),job_id='resume_offline')
    rd=tmp_path/cfg.job_id
    run_experiment(cfg,str(tmp_path),verbose=False)
    saved=torch.load(rd/'checkpoint_round_0003.pt',map_location='cpu',weights_only=False)
    # A resume from round 2 should match the uninterrupted last-round result.
    original=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    (rd/'checkpoint_latest.pt').write_bytes((rd/'checkpoint_round_0003.pt').read_bytes())
    (rd/'status.json').write_text(json.dumps({'status':'failed','protocol_version':PROTOCOL_VERSION}))
    assert run_experiment(cfg,str(tmp_path),verbose=False)['status']=='completed'
    replayed=torch.load(rd/'model_final.pt',map_location='cpu',weights_only=False)
    for k,v in original['model_state'].items():
        assert torch.equal(v,replayed['model_state'][k]),k
    assert original['clients'][0]['train_steps']==replayed['clients'][0]['train_steps']
