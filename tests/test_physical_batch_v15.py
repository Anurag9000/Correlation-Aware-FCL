"""The two-run campaign must use actual physical batches, not accumulation."""
import pytest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.models import VIT_ID,CNN_ID

@pytest.mark.parametrize('backbone,model',[('cnn',CNN_ID),('vit',VIT_ID)])
def test_dual_pilot_uses_one_physical_batch_per_update(backbone,model):
    jobs=make_jobs(backbone=backbone)
    assert len(jobs)==2
    for cfg in jobs:
        assert cfg.model==model
        assert cfg.batch_size==64
        assert cfg.micro_batch_size==64
        assert cfg.method=='replay' and cfg.aggregator=='fedavg'
        assert cfg.momentum==.9 and cfg.weight_decay==.0005
        assert '_V15_NO_ACCUMULATION_' in cfg.job_id
    plan=audited_plan(jobs)
    assert plan['batch_size']==64 and plan['micro_batch_size']==64
    assert plan['training_intervals']==9 and plan['terminal_communication_only_flush']
    assert jobs[1].stream_offline_policy=='rotate_once'

def test_physical_batch_tracks_selected_batch_size():
    jobs=make_jobs(backbone='cnn',batch_size=1024)
    assert all(c.batch_size==c.micro_batch_size==1024 for c in jobs)
