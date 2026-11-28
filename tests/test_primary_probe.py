"""Mathematical and reporting controls for the 2-job pilot."""
import json
from dataclasses import replace
import pytest
from cafcl.primary_probe import make_jobs,audited_plan,build_report
from cafcl.models import CNN_ID
from cafcl.full_matrix import expand
from cafcl.stream_audit import check_stream

def test_probe_is_the_fixed_two_job_100_label_recipe():
    full,random_offline=make_jobs()
    assert full.model==random_offline.model==CNN_ID
    assert full.batch_size==random_offline.batch_size==64
    assert full.method==random_offline.method=='replay'
    assert full.momentum==random_offline.momentum==0.9
    assert full.weight_decay==random_offline.weight_decay==0.0005
    assert full.memory_per_class==random_offline.memory_per_class==50
    assert full.share_continual_state and random_offline.share_continual_state
    assert full.aggregator==random_offline.aggregator=='fedavg'
    assert full.seed==random_offline.seed==0
    assert full.optimizer_state_policy==random_offline.optimizer_state_policy=='persist'
    assert full.training_augmentation==random_offline.training_augmentation=='cifar_crop_flip'
    assert full.client_training_concurrency==random_offline.client_training_concurrency==5
    assert full.stream_client_chain and random_offline.stream_client_chain
    assert full.local_tasks_per_round==random_offline.local_tasks_per_round==1
    assert 'V27_SOURCE_TASK_REPLAY' in full.job_id and 'V27_SOURCE_TASK_REPLAY' in random_offline.job_id
    assert full.evaluate_final_only and random_offline.evaluate_final_only
    assert full.participation==1. and random_offline.participation==.8
    plan=audited_plan((full,random_offline))
    assert plan['training_intervals']==20
    assert plan['initial_local_tasks']==1 and plan['subsequent_local_tasks']==1
    assert plan['new_unique_classes_first_sync']==6
    assert plan['new_unique_classes_syncs_2_to_19']==5
    assert plan['new_unique_classes_final_sync']==4
    assert plan['all_classes_covered'] and 'all_classes_once' not in plan
    assert plan['total_jobs']==2
    assert len(plan['offline_schedule'])==20
    assert all(len(x['online_clients'])==4 and len(x['offline_classes'])==2
               for x in plan['offline_schedule'])
    assert plan['all_clients_train_each_sync']
    assert plan['client_training_concurrency']==5
    assert plan['models_per_run']==6 and plan['server_models_across_both_arms']==2
    assert plan['client_models_across_both_arms']==10
    assert 'five-client task barrier' in plan['server_model_updates']
    assert plan['concurrently_training_client_models_across_both_arms']==10
    assert plan['expected_server_exemplars_after_flush']==10000
    assert plan['expected_server_exemplars_per_class']['0']==50
    assert plan['expected_server_exemplars_per_class']['98']==150
    assert plan['expected_server_exemplars_per_class']['99']==100
    assert plan['class_occurrence_counts']['0']==1
    assert all(plan['class_occurrence_counts'][str(c)]==2 for c in range(1,100) if c!=98)


def test_pilot_incompatible_random_offline_plan_rejected():
    full,partial=make_jobs()
    with pytest.raises(AssertionError):replace(partial,num_clients=10).validate()
    with pytest.raises(AssertionError):replace(partial,stream_terminal_flush=False).validate()


def test_report_requires_both_complete_and_all_classes(tmp_path):
    full,partial=make_jobs()
    root=tmp_path
    assert build_report(root,(full,partial)) is None
    for job,accuracy in ((full,.3),(partial,.4)):
        path=root/'runs'/job.job_id
        path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps({'final_metrics':{
            'accuracy':accuracy,
            'standard_task_conditioned_test':{
                'macro_head_accuracy':accuracy,
                'per_class':{str(i):accuracy for i in range(100)}},
            'globally_integrated_classes':list(range(100))}}))
    report=build_report(root,(full,partial))
    assert report['delta']==pytest.approx(.1)
    assert len(json.loads((root/'comparison'/'per_class.json').read_text()))==100
    assert (root/'comparison'/'comparison.md').exists()
    # Cardinality alone is not enough: 100 out-of-range labels are invalid.
    path=root/'runs'/partial.job_id/'summary.json'
    invalid=json.loads(path.read_text())
    invalid['final_metrics']['globally_integrated_classes']=list(range(1,101))
    path.write_text(json.dumps(invalid))
    with pytest.raises(RuntimeError,match='exactly 0..99'):
        build_report(root,(full,partial))


def test_retired_task_length_arms_cannot_enter_active_experiment_plan():
    from cafcl.stream_audit import check_job
    from cafcl.full_matrix import base
    for classes,tasks in ((2,1),(2,3),(4,1),(4,2),(4,3)):
        variant=replace(base(0),classes_per_local_task=classes,
                        local_tasks_per_round=tasks)
        assert check_job(variant)=='retired-variable-task-length-protocol'
    assert all(c.classes_per_local_task==2 and c.local_tasks_per_round==2
               for _,c in expand(seeds=(0,),families=('task_length',)))
