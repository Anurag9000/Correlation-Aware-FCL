from dataclasses import replace
from cafcl.reference_fcil import ReferenceConfig,compile_reference

def test_reference_exact_partition_and_cohorts():
    labels=[i for i in range(10) for _ in range(30)]
    cfg=ReferenceConfig(clients=3,classes_per_task=2,rounds_per_task=3,participation=.66,seed=7)
    a=compile_reference(labels,10,cfg)
    assert a==compile_reference(labels,10,cfg)
    assert sorted(i for stage in a['stages'] for part in stage['parts'] for i in part)==list(range(len(labels)))
    assert all(len(c)==2 for stage in a['stages'] for c in stage['cohorts'])
    assert a['plan_hash']!=compile_reference(labels,10,replace(cfg,seed=8))['plan_hash']
