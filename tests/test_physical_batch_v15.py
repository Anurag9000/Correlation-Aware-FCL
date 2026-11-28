"""Production pilots use unsplit physical batches of 64."""
import pytest
from cafcl.primary_probe import make_jobs,audited_plan
from cafcl.models import CNN_ID
from cafcl.config import RunConfig

def test_pilot_uses_64_cropflip_and_persistent_sgd():
    jobs=make_jobs()
    assert len(jobs)==2
    for cfg in jobs:
        assert cfg.model==CNN_ID and cfg.batch_size==64
        assert cfg.method=='replay' and cfg.aggregator=='fedavg'
        assert cfg.optimizer_state_policy=='persist'
        assert cfg.training_augmentation=='cifar_crop_flip'
        assert cfg.momentum==.9 and cfg.weight_decay==.0005
        assert '_TASKIL_V27_SOURCE_TASK_REPLAY_' in cfg.job_id
    plan=audited_plan(jobs)
    assert plan['batch_size']==64
    assert plan['training_intervals']==20 and plan['terminal_communication_only_flush']
    assert jobs[1].stream_offline_policy=='random'

def test_no_production_batch_1024():
    with pytest.raises(AssertionError,match='64'):
        RunConfig(batch_size=1024).validate()
