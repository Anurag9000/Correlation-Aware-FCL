"""Mathematical and reporting controls for the 2-job pilot."""
import json
from dataclasses import replace
import pytest
from cafcl.primary_probe import make_jobs,audited_plan,build_report
from cafcl.models import CNN_ID,VIT_ID
from cafcl.full_matrix import expand
from cafcl.stream_audit import check_stream

@pytest.mark.parametrize('backbone',('vit','cnn'))
def test_probe_has_two_matched_100_label_jobs_and_exhaustive_rotation(backbone):
    full,rotating=make_jobs(backbone=backbone)
    assert full.model==rotating.model==({'vit':VIT_ID,'cnn':CNN_ID}[backbone])
    assert full.batch_size==rotating.batch_size==1024
    assert full.method==rotating.method=='agem'
    assert full.aggregator==rotating.aggregator=='fedavg'
    assert full.seed==rotating.seed==0
    assert full.evaluate_final_only and rotating.evaluate_final_only
    assert full.participation==1. and rotating.participation==.8
    plan=audited_plan((full,rotating))
    assert plan['training_intervals']==5
    assert plan['total_jobs']==2
    assert sorted(x['offline_client'] for x in plan['offline_schedule'])==list(range(5))
    assert len(set(c for x in plan['offline_schedule'] for c in x['offline_classes']))==20
    assert all(len(x['offline_classes'])==4 for x in plan['offline_schedule'])


def test_pilot_incompatible_rotation_rejected():
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
            'accuracy':accuracy,'per_class':{str(i):accuracy for i in range(100)},
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


def test_mathematical_audit_discards_impossible_overlap_arms():
    full=list(expand(seeds=(0,),families=('task_length',),include_invalid=True))
    feasible=list(expand(seeds=(0,),families=('task_length',)))
    assert len(feasible)<len(full)
    assert any(c.stream_overlap_kind=='intra_same_round' and c.local_tasks_per_round==1
               for _,c in full)
    assert not any(c.stream_overlap_kind=='intra_same_round' and c.local_tasks_per_round==1
                   for _,c in feasible)
