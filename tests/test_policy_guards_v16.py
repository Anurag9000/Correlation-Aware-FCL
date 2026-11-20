"""All active production pathways use one physical batch of 64."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from cafcl.config import RunConfig
from cafcl.full_matrix import base,expand
from cafcl.primary_probe import make_jobs
from cafcl.continual import LocalState

def test_no_legacy_split_batch_parameter_in_active_code():
    root=Path(__file__).resolve().parents[1]
    for path in (root/'src'/'cafcl').glob('*.py'):
        contents=path.read_text()
        assert 'micro_batch_size' not in contents, str(path)
        assert '--micro-batch-size' not in contents, str(path)
    for name in ('cifar10','cifar100','cifar100_20'):
        cfg=RunConfig(dataset=name)
        assert cfg.batch_size==64
        with pytest.raises(AssertionError,match='64'):
            replace(cfg,batch_size=1024).validate()

def test_full_factorial_replay_uses_pilot_classwise_reservoir_policy():
    job=next(cfg for family,cfg in expand(seeds=(0,),families=('balanced',),
        model='cnn') if cfg.method=='replay' and cfg.aggregator=='fedavg')
    pilot=make_jobs()[0]
    assert job.memory_policy==pilot.memory_policy=='random_per_class'
    assert job.memory_per_class==pilot.memory_per_class==50
    assert job.replay_reservoir_fraction==0.
    assert pilot.replay_reservoir_fraction==pytest.approx(1/9)
    caps={c.memory_per_class for family,c in expand(seeds=(0,),
        families=('memory',),model='cnn') if c.method=='replay'}
    assert caps=={0,10,20,50,100}

def test_memory_capacity_ablation_really_changes_sampled_reservoir():
    targets=[0]*100+[1]*100
    for quota in (0,10,20,50,100):
        client=LocalState(0)
        client.update_memory(range(200),targets,quota,np.random.default_rng(22))
        assert all(len(client.private_memory.get(c,[]))==quota for c in (0,1))
        if quota:
            assert client.private_reservoir_seen_counts=={0:100,1:100}
        assert all(len(g)<=quota for g in client.private_memory.values())

def test_persistent_optimizer_and_cropflip_stay_fixed_for_primary():
    full,partial=make_jobs()
    for cfg in (full,partial):
        assert cfg.optimizer_state_policy=='persist'
        assert cfg.training_augmentation=='cifar_crop_flip'
        assert cfg.client_training_concurrency==1
        assert cfg.batch_size==64 and cfg.momentum==.9 and cfg.weight_decay==5e-4
        assert cfg.stream_terminal_flush and cfg.share_continual_state
    assert partial.stream_offline_policy=='rotate_once'
