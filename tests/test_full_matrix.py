"""Exact-count regression tests for declared finite-axis CIFAR designs."""
from itertools import islice
import pytest
from cafcl.full_matrix import METHODS,AGGS,MAIN,POLICIES,expand,plan
from cafcl.planner import all_supported_suite
@pytest.mark.parametrize('dataset,n,digest',[
 ('cifar10',67938,'56086a0234db8681458b514b1db87247b905b9721dfda496b71a686f24377b7b'),
 ('cifar100',69722,'7eda27c5047cc740f2c8fa0ac61a32e8c977c0da1ba107d1c6fe2e02f9c22870')])
def test_plan(dataset,n,digest):
    r=plan(dataset)
    assert r['total_jobs']==n and r['manifest_sha256']==digest
    assert r['method_server_pairs']==99
    assert r['family_counts']['balanced']==11*9*4*5*3
def test_legacy_preservation_and_cross():
    for d,n in (('cifar10',423),('cifar100',425)):
        prefix=list(islice(expand(d),n))
        assert [c.job_id for _,c in prefix]==[c.job_id for c in all_supported_suite(d)]
    grid=list(expand('cifar10',seeds=(0,),include_legacy=False,families=('balanced',)))
    assert len(grid)==1980 and len({(c.method,c.aggregator) for _,c in grid})==99
    assert all(c.batch_size==512 for _,c in grid)
def test_cifar100_subbenchmark():
    grid=list(expand('cifar100',seeds=(0,),include_legacy=False,families=('cifar100_20',)))
    assert len(grid)==594
    assert {(c.method,c.aggregator) for _,c in grid}=={(m,a) for m in METHODS for a in AGGS}
    assert all(c.dataset=='cifar100_20' and c.stages==2 for _,c in grid)
