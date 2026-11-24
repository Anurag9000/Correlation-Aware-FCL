"""Targeted regression checks for audit remediation policies."""
from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
import torch
from cafcl.config import RunConfig
from cafcl.data import get_data_bank
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.adapter import load_plan
from cafcl.joint.audit import server_boundary_digest
from cafcl.federated_cl import FederatedCLState
from cafcl.continual import LocalState
from cafcl.diagnostics import class_gradient_representations
from cafcl.models import create_model
from cafcl.simulator import anchored_upload
from cafcl.selection import effective_participation


def test_fixed_occurrence_support_and_reconstruction(tmp_path):
    labels=[c for c in range(4) for _ in range(20)]
    events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(2,3)),(1,1,(2,3))]
    plan=compile_plan(events,labels,num_clients=2,num_classes=4,
                      samples_per_class_occurrence=3).to_dict()
    assert all(len(x['train_indices'])==6 for x in plan['tasks'])
    assert plan['allocation_audit']['unique_assigned']==24
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    assert load_plan(p,plan['plan_hash'],labels)['plan_hash']==plan['plan_hash']


def test_payload_mutation_changes_boundary_digest():
    state=FederatedCLState()
    state.exemplar_payloads[1]=(torch.zeros(3,32,32,dtype=torch.uint8),0)
    first=server_boundary_digest({'x':torch.ones(1)},0,state)
    state.exemplar_payloads[1][0][0,0,0]=1
    assert server_boundary_digest({'x':torch.ones(1)},0,state)!=first


def test_task_gradient_global_labels():
    bank=get_data_bank('synthetic','data',download=False)
    groups={'task_008_009':(8,9)}
    model=create_model('test_linear',10,allow_test_model=True,task_class_sets=groups)
    vectors=class_gradient_representations(model,bank,[8,9],torch.device('cpu'),task_class_groups=groups)
    assert set(vectors)=={8,9}
    assert all(torch.isfinite(v).all() for v in vectors.values())


def test_stale_policies():
    state=LocalState(0)
    state.receive({'weight':torch.tensor([1.])},0)
    state.shared_weights={'weight':torch.tensor([3.])}
    state.pending_upload_steps=1;state.pending_upload_examples=1
    server={'weight':torch.tensor([5.])}
    assert anchored_upload(state,server,current_version=1)['weights']['weight'].item()==6.
    assert anchored_upload(state,server,current_version=1,stale_policy='undamped_delta')['weights']['weight'].item()==7.
    assert anchored_upload(state,server,current_version=1,stale_policy='direct_model')['weights']['weight'].item()==3.


def test_schedule_compiled_horizon():
    cfg=SimpleNamespace(participation_schedule='increasing',participation=.8,
                        num_clients=5,stages=1,rounds_per_stage=1)
    assert effective_participation(cfg,10,learning_rounds=21)==pytest.approx(.6)


def test_model_only_variable_batch_runtime(tmp_path):
    from cafcl.simulator import run_experiment
    from cafcl.verify_final import verify_final
    bank=get_data_bank('synthetic','data',download=False)
    events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(2,3)),(1,1,(2,3))]
    plan=compile_plan(events,bank.train.targets,num_clients=2,num_classes=10,
                      head_policy='semantic_shared',require_all_classes=False).to_dict()
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    cfg=RunConfig(dataset='synthetic',model='test_linear',stream_protocol=True,
        num_clients=2,epochs=1,batch_size=8,physical_batch_policy='variable',
        participation=1.,selector='random',scenario='overlap',stages=1,rounds_per_stage=1,
        local_tasks_per_round=1,method='replay',learning_setting='task_il',
        memory_per_class=2,share_continual_state=False,client_training_concurrency=1,
        joint_plan_path=str(p),joint_plan_hash=plan['plan_hash'],
        joint_allow_partial_coverage=True,joint_checkpoint_policy='latest',
        classwise_head_merge=False,job_id='private_variable')
    result=run_experiment(cfg,str(tmp_path),download=False,verbose=False)
    assert result['status']=='completed'
    root=tmp_path/'private_variable'
    summary=json.loads((root/'summary.json').read_text())
    assert summary['shared_continual_state']['shared_replay_samples']==0
    assert verify_final(root,device='cpu')['verified']
