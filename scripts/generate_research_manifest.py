"""Produce committed per-dataset job plans and the truthful block ledger."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import json
from cafcl.catalogue import CATALOGUE,PARTIAL,IMPLEMENTED,catalogue_report
from cafcl.planner import all_supported_suite

root=Path(__file__).resolve().parents[1]
base=root/'configs'/'plans'
base.mkdir(parents=True,exist_ok=True)
jobs={d:all_supported_suite(d) for d in ('cifar10','cifar100')}
for dataset,items in jobs.items():
    from cafcl.planner import write_plan
    plan={'dataset':dataset,'batch_size':512,
          'jobs':[job.to_dict() for job in items],
          'catalogue_coverage':catalogue_report(),
          'registered_block_ids':sorted({job.block_id for job in items}),
          'coverage_warning':'A registered job does NOT imply complete methods or trained/validated CIFAR results.'}
    (base/f'{dataset}_jobs.json').write_text(json.dumps(plan,separators=(',',':'))+'\n')
    print(dataset,len(items))
coverage=catalogue_report()
(root/'docs'/'experiment_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
counts={d:Counter(job.block_id for job in items) for d,items in jobs.items()}
lines=['# 152-block research experiment registry and executable coverage','',
       'Status is about implementation coverage, **not completed CIFAR training**. '
       'A partial implementation may be an oracle, a proxy, or a limited comparison. '
       'No official CIFAR suite was executed as part of this commit.','',
       '| ID | Research question | Status | CIFAR-10 jobs | CIFAR-100 jobs |',
       '|---|---|---|---:|---:|']
for item in coverage:
    bid=item['id'];name=item['name'];status=item['status']
    lines.append(f'| {bid} | {name} | {status} | {counts["cifar10"][bid]} | {counts["cifar100"][bid]} |')
lines.extend(['',f'**Totals:** {len(coverage)} blocks; '
              f'{len(IMPLEMENTED)} with implementation coverage, '
              f'{len(PARTIAL)} explicitly partial, '
              f'{len(set(CATALOGUE)-IMPLEMENTED)} without implementation coverage; '
              f'{len(jobs["cifar10"])} CIFAR-10 jobs and '
              f'{len(jobs["cifar100"])} CIFAR-100 jobs.', '',
              '**Restrictions:** Task-free controls remain class-aware and boundary-free; '
              'selection policies using simulator class/gradient information are privileged oracles; '
              'the 20-class CIFAR-100 10+10 suite does not reproduce every natural 100-class scenario; '
              'original-paper algorithm fidelity, secure aggregation/DP, and publication-quality results are unverified.', ''])
(root/'docs'/'EXPERIMENT_CATALOGUE.md').write_text('\n'.join(lines))
