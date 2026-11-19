"""Mathematical semantics not captured by label-count-only auditing."""
from dataclasses import replace
from cafcl.full_matrix import base, expand
from cafcl.stream_audit import check_job, check_stream


def test_incomplete_final_outage_cohort_is_not_exact_duration():
    # 10 clients, 4 labels/client/round: 40 + 40 + 20 = 100.
    # Four training rounds: a two-round offline cohort now fits exactly;
    # three-round outages leave a truncated final cohort.
    c=replace(base(0),participation=.8,offline_duration_rounds=2)
    assert check_job(c) is None
    assert check_job(replace(c,offline_duration_rounds=3))=='truncated-or-extended-offline-duration'
    assert check_job(replace(c,offline_duration_rounds=1)) is None


def test_selection_alias_and_unavailable_signal_arms_removed():
    c=base(0)
    for selector in ('fairness','balanced','diversity'):
        assert check_job(replace(c,selector=selector))=='selection-policy-exact-alias'
    assert check_job(replace(c,selector='fixed',participation=1.))=='selection-policy-inert-at-full-participation'
    assert check_job(replace(c,selector='recurrence'))=='recurrence-selector-has-no-recurrence-signal'
    assert check_job(replace(c,selector='overlap'))=='overlap-selector-has-no-overlap-signal'
    assert check_job(replace(c,selector='random',participation=1.)) is None


def test_degenerate_selection_and_truncated_outages_absent_from_plan():
    selected=list(expand(seeds=(0,),families=('selection',)))
    assert not any(cfg.selector in {'fairness','balanced','diversity','recurrence','overlap'}
                   for _,cfg in selected)
    assert not any(cfg.participation==1 and cfg.selector!='random'
                   for _,cfg in selected)
    selected_outages=list(expand(seeds=(0,),families=('offline_duration',)))
    assert selected_outages
    assert all(check_job(cfg) is None for _,cfg in selected_outages)


def test_lwf_unused_hyperparameter_arms_are_excluded():
    c=replace(base(0),method='lwf')
    assert check_job(c) is None
    assert check_job(replace(c,memory_per_class=50))=='lwf-memory-capacity-unused'
    assert check_job(replace(c,regularization_strength=1.0))=='lwf-regularization-strength-unused'
    assert not any(cfg.method=='lwf' and cfg.memory_per_class!=20
                   for _,cfg in expand(seeds=(0,),families=('memory',)))
    assert not any(cfg.method=='lwf' and cfg.regularization_strength!=10.0
                   for _,cfg in expand(seeds=(0,),families=('regularization',)))
