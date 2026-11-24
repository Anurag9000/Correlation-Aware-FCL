from collections import Counter
import json
import torch
from torch.utils.data import Dataset
from cafcl.data import DataBank
from cafcl.diagnostic_campaign import load_jobs, inspect_job, signature
from cafcl.diagnostic_designs import compile_matched,batch_stream
from cafcl.matched_diagnostic import run_reference,candidate


def test_ledger_counts_and_fail_closed(tmp_path):
    jobs=load_jobs()
    assert Counter(inspect_job(j,tmp_path) for j in jobs)=={
        "ready":71,"blocked":71,"reserved":10}
    assert len(set(signature(j) for j in jobs))==152


def test_balanced_overlap_schedules():
    for j in load_jobs():
        if j["family"]=="OF":
            assert len(j["cohort_schedule"])==20
            assert Counter(c for cohort in j["cohort_schedule"] for c in cohort)=={i:8 for i in range(5)}


def test_matched_plans_and_budget():
    labels=[c for c in range(100) for _ in range(450)]
    plans={}
    for j in load_jobs():
        if j.get("binding")!="matched_reference_v1":continue
        p=compile_matched(j,labels)
        assert len(p)==10
        assert all(sum(len(u["indices"]) for us in s["units"] for u in us)==4500 for s in p)
        key=(j["family"],j["clients"],j.get("partition"),j["curriculum"],j["participation"])
        assert key not in plans or p==plans[key]
        plans[key]=p
        if j["family"]=="MO":
            assert all([sum(len(u["indices"]) for u in us) for us in s["units"]]==[900]*5 for s in p)
    rows=[(i,(0,1)) for i in range(97)]
    replay=[(i,(0,1)) for i in range(16)]
    y=[i%2 for i in range(100)]
    baseline=list(batch_stream(rows,replay,y,0,20))
    for r in (5,10):
        segmented=[x for block in range(r) for x in batch_stream(rows,replay,y,0,20//r,block*(20//r))]
        assert baseline==segmented


def test_stale_candidate():
    base={"p":torch.tensor([4.]),"bn":torch.tensor([7.])}
    local={"p":torch.tensor([8.]),"bn":torch.tensor([3.])}
    anchor={"p":torch.tensor([2.]),"bn":torch.tensor([1.])}
    assert candidate(base,local,anchor,0,{"p"}) is local
    stale=candidate(base,local,anchor,2,{"p"})
    assert stale["p"].item()==6 and stale["bn"].item()==7


class Fixture(Dataset):
    def __init__(self,n):
        self.targets=[c for c in range(20) for _ in range(n)]
        self.x=torch.randn(len(self.targets),3,32,32,generator=torch.Generator().manual_seed(n))
    def __len__(self):return len(self.targets)
    def __getitem__(self,i):return self.x[i],self.targets[i]


def test_matched_smoke(tmp_path):
    torch.set_num_threads(1)
    bank=DataBank(Fixture(8),Fixture(2),20,{},(),Fixture(2))
    for il in ("class_il","task_il"):
        j=next(x for x in load_jobs() if x["family"]=="MC" and x["clients"]==2 and x["partition"]=="iid" and x["il"]==il)
        j={**j,"local_epochs":1,"rounds_per_stage":2}
        out=tmp_path/il
        rows=run_reference(j,out,bank,model_name="test_linear",subset_probes=True)
        assert len(rows)==4
        s=json.loads((out/"result_summary.json").read_text())
        assert s["learning_syncs"]==4 and s["counts"]["replay_draws"]>0
        assert 0<=s["final_accuracy"]<=1
        assert rows[-1]["test_forgetting"] is not None


def test_reconstructed_plans():
    from cafcl.diagnostic_designs import compile_reconstructed,head_labels
    labels=[c for c in range(100) for _ in range(450)]
    for j in load_jobs():
        if not j.get("reconstruction_binding"):continue
        p=compile_reconstructed(j,labels)
        assert sum(len(u["indices"]) for s in p for us in s["units"] for u in us)==45000
        assert len(p)==(20 if j["curriculum"]=="joint" else (50+j["clients"]-1)//j["clients"])
        if j["curriculum"]=="joint":
            from collections import Counter
            counts=Counter(c for s in p for us in s["units"] for u in us for c in head_labels(u["head"]))
            assert counts[0]==1 and counts[98]==3 and counts[99]==2
        if j["family"]=="CC" and j["clients"]==3:
            assert not p[-1]["units"][2]


def test_reconstructed_smoke(tmp_path):
    torch.set_num_threads(1)
    bank=DataBank(Fixture(8),Fixture(2),20,{},(),Fixture(2))
    for il in ("class_il","task_il"):
        j=next(x for x in load_jobs() if x["family"]=="JR" and x["curriculum"]=="joint" and x["participation"]==.4 and x["il"]==il)
        j={**j,"binding":"historical_reconstruction_v1","local_epochs":1}
        out=tmp_path/il
        rows=run_reference(j,out,bank,model_name="test_linear",subset_probes=True)
        assert len(rows)==5 and rows[-1]["flush"]
        summary=json.loads((out/"result_summary.json").read_text())
        assert not summary["historical_reproduction_verified"]
        assert 0<=summary["final_accuracy"]<=1
        if il=="task_il":assert 0<=summary["legacy_union_accuracy"]<=1


def test_effective_full_suite():
    from cafcl.diagnostic_campaign import effective_jobs,expected_rounds
    jobs=effective_jobs(True,True)
    assert Counter(j.get("binding","reserved") for j in jobs)=={
        "stationary_reference":11,"matched_reference_v1":60,"historical_reconstruction_v1":71,"reserved":10}
    assert all(expected_rounds(j)[0]>0 for j in jobs if j.get("binding"))
    assert all(j.get("execution_fidelity")=="reconstructed_not_exact"
               for j in jobs if j.get("binding")=="historical_reconstruction_v1")


def test_artifact_validation_rejects_incomplete(tmp_path):
    from cafcl.diagnostic_campaign import validate_artifacts
    import pytest
    j=load_jobs()[0]
    with pytest.raises(ValueError,match="Missing"):
        validate_artifacts(j,tmp_path)
