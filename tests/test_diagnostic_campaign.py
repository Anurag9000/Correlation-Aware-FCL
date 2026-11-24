from collections import Counter
from cafcl.diagnostic_campaign import load_jobs, inspect_job, signature


def test_ledger_counts_and_fail_closed(tmp_path):
    jobs=load_jobs()
    assert Counter(inspect_job(j,tmp_path) for j in jobs)=={
        "ready":11,"blocked":131,"reserved":10}
    assert len(set(signature(j) for j in jobs))==152


def test_balanced_overlap_schedules():
    for j in load_jobs():
        if j["family"]=="OF":
            assert len(j["cohort_schedule"])==20
            assert Counter(c for cohort in j["cohort_schedule"] for c in cohort)=={i:8 for i in range(5)}
