"""Verify immutable CIFAR job manifest content against the source planner."""
from __future__ import annotations
import json
from pathlib import Path
from cafcl.catalogue import CATALOGUE
from cafcl.planner import all_supported_suite


def check(dataset):
    root=Path(__file__).resolve().parents[1]
    path=root/'configs'/'plans'/f'{dataset}_jobs.json'
    frozen=json.loads(path.read_text())
    expected=json.loads(json.dumps([job.to_dict() for job in all_supported_suite(dataset)]))
    assert frozen['dataset']==dataset
    assert frozen['batch_size']==512
    assert frozen['jobs']==expected, f'{dataset} plan stale or changed'
    ids=[job['job_id'] for job in expected]
    assert len(ids)==len(set(ids))
    assert all(job['batch_size']==512 for job in expected)
    assert set(job['block_id'] for job in expected)<=set(CATALOGUE)
    print(f'{dataset}: {len(ids)} jobs, '
          f'{len(set(job["block_id"] for job in expected))} block IDs verified')


if __name__=='__main__':
    check('cifar10')
    check('cifar100')
