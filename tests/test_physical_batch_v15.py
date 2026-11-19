"""Production pilots use unsplit physical batches of 64."""
import pytest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.models import VIT_ID,CNN_ID
from cafcl.config import RunConfig

@pytest.mark.parametrize('backbone,model',[('cnn',CNN_ID),('vit',VIT_ID)])
def test_pilot_uses_64_and_persistent_sgd(backbone,model):
    jobs=make_jobs(backbone=backbone)
    assert len(jobs)==2
    for cfg in jobs:
        assert cfg.model==model and cfg.batch_size==64
        assert cfg.method=='replay' and cfg.aggregator=='fedavg'
        assert cfg.optimizer_state_policy=='persist'
        assert cfg.momentum==.9 and cfg.weight_decay==.0005
        assert '_V16_BATCH64_' in cfg.job_id
    plan=audited_plan(jobs)
    assert plan['batch_size']==64
    assert plan['training_intervals']==9 and plan['terminal_communication_only_flush']
    assert jobs[1].stream_offline_policy=='rotate_once'

def test_no_production_batch_1024():
    with pytest.raises(AssertionError,match='64'):
        RunConfig(batch_size=1024).validate()
