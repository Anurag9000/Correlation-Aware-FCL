"""Focused regression tests for the static audit remediation."""
from types import SimpleNamespace
import pytest
from cafcl.selection import effective_participation
from cafcl.stationary_diagnostic import partitions
from cafcl.config import RunConfig

def test_stream_schedule_uses_compiled_horizon():
    cfg=SimpleNamespace(participation_schedule="increasing", participation=0.8,
                        num_clients=5, stages=1, rounds_per_stage=1)
    assert effective_participation(cfg,0,learning_rounds=21)==pytest.approx(0.4)
    assert effective_participation(cfg,10,learning_rounds=21)==pytest.approx(0.6)
    assert effective_participation(cfg,20,learning_rounds=21)==pytest.approx(0.8)
    assert effective_participation(cfg,30,learning_rounds=21)==pytest.approx(0.8)

def test_unknown_stationary_partition_fails_closed():
    with pytest.raises(ValueError,match="Unsupported stationary partition"):
        partitions(list(range(30)),2,"typo",0)

def test_central_partition_requires_one_learner():
    with pytest.raises(ValueError,match="Invalid learner count"):
        partitions(list(range(30)),2,"centralized",0)

def test_masked_replay_rejects_unsupported_ratio():
    cfg=RunConfig(dataset="synthetic",stream_partial_batch_policy="mask_padding_loss",
                  replay_ratio=2.0)
    with pytest.raises(ValueError,match="only ratio 1.0"):
        cfg.validate()
