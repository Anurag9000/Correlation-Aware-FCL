"""Verify frozen deterministic factorial job counts and SHA-256 identities."""
from cafcl.full_matrix import plan
EXPECTED={
  'cifar10':(67938,'56086a0234db8681458b514b1db87247b905b9721dfda496b71a686f24377b7b'),
  'cifar100':(69722,'7eda27c5047cc740f2c8fa0ac61a32e8c977c0da1ba107d1c6fe2e02f9c22870'),
}
for dataset,(jobs,digest) in EXPECTED.items():
    r=plan(dataset)
    assert r['total_jobs']==jobs
    assert r['manifest_sha256']==digest
    assert r['method_server_pairs']==99
    assert r['batch_size']==512
    print(dataset,jobs,'jobs, 99 pairs, batch 512')
