"""Preliminary class-omission design: exactly six jobs and valid comparisons."""
from pathlib import Path
import csv
import json
import pytest

from cafcl.config import RunConfig
from cafcl.data import get_data_bank
from cafcl.primary_agem import make_ledger, verify_partition, compare_finished


def test_five_jobs_nonrepeated_omissions_25_percent_clients():
    ledger = make_ledger(seed=21)
    assert len(ledger['jobs']) == 5
    assert ledger['jobs'][0]['kind'] == 'baseline'
    assert ledger['jobs'][0]['config']['participation'] == 1.0
    assert all(j['config']['batch_size'] == 512 for j in ledger['jobs'])
    assert all(j['config']['method'] == 'agem' and j['config']['aggregator'] == 'fedavg'
               and j['config']['scenario'] == 'disjoint' for j in ledger['jobs'])
    assert all(j['config']['holders_override'] == ledger['jobs'][0]['config']['holders_override']
               for j in ledger['jobs'])
    omitted = [c for j in ledger['jobs'][1:] for c in j['off_classes']]
    assert sorted(omitted) == list(range(10))
    assert sorted(len(j['off_classes']) for j in ledger['jobs'][1:])==[2,2,3,3]
    assert all(j['config']['participation'] == .75 for j in ledger['jobs'][1:])
    assert all(len(set(j['off_classes']) & set(j['included_classes'])) == 0
               for j in ledger['jobs'][1:])
    assert sorted(j['off_client'] for j in ledger['jobs'][1:]) == list(range(4))
    assert '25%' in ledger['note']
    assert make_ledger(seed=21) == ledger
    assert ledger != make_ledger(seed=22)


def test_allocation_is_explicit_disjoint_and_stage_balanced():
    ledger=make_ledger(seed=0,quick=True)
    bank=get_data_bank('synthetic','unused',download=False,seed=0)
    scenario=verify_partition(bank,RunConfig(**ledger['jobs'][0]['config']))
    assert len(scenario.label_holders) == 10
    assert all(len(h)==1 for h in scenario.label_holders.values())
    assert not scenario.duplicate_sample_ids
    for cid,group in ledger['client_class_groups'].items():
        assert len(group) in (2,3) and any(x<5 for x in group) and any(x>=5 for x in group)
        assert all(scenario.label_holders[x]=={int(cid)} for x in group)


def test_validation_guards():
    with pytest.raises(ValueError):make_ledger(train_fraction=0.)
    with pytest.raises(ValueError):make_ledger(rounds_per_stage=0)
    with pytest.raises(ValueError):make_ledger(memory_per_class=0)


def test_synthetic_five_job_run_and_seen_class_comparison(tmp_path, monkeypatch):
    import cafcl.simulator as simulator
    import cafcl.primary_agem as primary
    original_get_data_bank = get_data_bank
    def bank(name, data_root, *, download=True, seed=0, **kwargs):
        return original_get_data_bank('synthetic', str(data_root), download=False, seed=seed)
    monkeypatch.setattr(simulator,'get_data_bank',bank)
    monkeypatch.setattr(primary,'get_data_bank',bank)
    ledger=make_ledger(seed=9,quick=True,rounds_per_stage=1,model='linear',train_fraction=.2)
    root=tmp_path/'pilot'
    for j in ledger['jobs']:
        cfg=RunConfig(**j['config'])
        result=simulator.run_experiment(cfg,str(root/'runs'),str(tmp_path),False,verbose=False)
        assert result['status']=='completed'
        assert simulator.run_experiment(cfg,str(root/'runs'),str(tmp_path),False,verbose=False)['status']=='already_completed'
    rows=compare_finished(ledger,root,str(tmp_path),False,allow_cpu=True)
    assert len(rows)==4
    assert all(x['paired_test_samples'] in (7*6,8*6) for x in rows)
    for row in rows:
        assert len(row['off_classes'].split(',')) in (2,3)
        assert -100 <= row['delta_seen_percentage_points'] <= 100
    with (root/'reports'/'seen_class_effects.csv').open() as file:
        details=list(csv.DictReader(file))
    assert len(details) == 30
    by_job={j['job_id']:set(j['off_classes']) for j in ledger['jobs'][1:]}
    for item in details:
        assert int(item['evaluated_class']) not in by_job[item['partial_job']]
    assert (root/'reports'/'paired_comparison.md').is_file()
    assert json.loads((root/'reports'/'paired_summary.json').read_text())['comparable_labels_only']
