"""FedCLASS CE/FedAvg adjacent-chain offline pilot regression suite."""
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from cafcl.fedclass_primary import (
    SOURCE_COMMIT, make_fedclass_jobs, make_design, main, _output_root)
from cafcl.primary_probe import audited_plan
from cafcl.streaming import make_task_schedule, select_offline_cohort
from cafcl.models import create_model


@pytest.mark.parametrize("mode,method,count",[
    ("authors_exemplars","replay",50),
    ("none","sgd",0),
])
def test_paired_fedclass_configs_and_schedule(mode,method,count):
    full,hidden=make_fedclass_jobs(memory_mode=mode,seed=13,epochs=10,lr=.03,
                                   client_training_concurrency=2)
    assert full.model==hidden.model=="fedclass_resnet8"
    assert full.method==hidden.method==method
    assert full.aggregator==hidden.aggregator=="fedavg"
    assert not full.classwise_head_merge and not hidden.classwise_head_merge
    assert full.num_clients==hidden.num_clients==5
    assert full.participation==1. and hidden.participation==.8
    assert hidden.stream_offline_policy=="random"
    assert full.batch_size==hidden.batch_size==64
    assert full.epochs==hidden.epochs==10
    assert full.momentum==hidden.momentum==.9
    assert full.weight_decay==hidden.weight_decay==.0005
    assert full.optimizer_state_policy==hidden.optimizer_state_policy=="reset"
    assert full.memory_per_class==hidden.memory_per_class==count
    assert full.replay_reservoir_fraction==0  # Authors source does not use Algorithm R.
    assert full.share_continual_state and hidden.share_continual_state
    assert full.learning_setting=="class_il" and hidden.learning_setting=="class_il"
    assert full.class_pair_protocol=="strict" and hidden.class_pair_protocol=="strict"
    assert full.client_training_concurrency==hidden.client_training_concurrency==2
    assert set(full.to_dict())==set(hidden.to_dict())
    diff={k for k in full.to_dict() if full.to_dict()[k]!=hidden.to_dict()[k]}
    assert diff=={"job_id","participation"}

    stream=make_task_schedule(100,full)
    assert len(stream)==100
    assert set(c for *_,labels,recur in stream for c in labels)==set(range(100))
    assert all(len(labels)==2 for _,_,_,labels,_ in stream)
    for cid in range(5):
        assert [labels for rnd,c,_,labels,_ in stream if rnd==0 and c==cid]==[(cid,cid+1)]
        for rnd in range(1,20):
            expected=(98,99) if rnd==19 and cid==4 else (rnd*5+cid,rnd*5+cid+1)
            assert [labels for r,c,_,labels,_ in stream if r==rnd and c==cid]==[expected]
    plan=audited_plan((full,hidden))
    assert plan["training_intervals"]==20
    assert plan["terminal_communication_only_flush"]
    assert plan["total_jobs"]==2
    assert plan["private_reservoir_per_class"]==count
    assert plan["expected_server_exemplars_after_flush"]==200*count
    missing=[]
    for rnd in range(20):
        online=set(select_offline_cohort(hidden,rnd,10,list(range(4))))
        assert len(online)==4
        missing.append(next(iter(set(range(5))-online)))
    assert len(missing)==20
    meta=make_design((full,hidden),mode=mode,replay_ratio=1.)
    assert meta["upstream_commit"]==SOURCE_COMMIT
    assert meta["test_oracle_task_id"] is False
    assert meta["concurrent_processes"]==2
    assert meta["full_and_offline_same_gpu"]==0


def test_original_fedclass_modules_exact_feature_head_forward():
    import torch.nn.functional as F
    torch.manual_seed(51)
    model=create_model("fedclass_resnet8",100)
    assert model.features.original.__class__.__name__=="ResNet"
    assert model.features.original.out_dim==64
    assert model.classifier.__class__.__name__=="SimpleLinear"
    assert model.classifier.in_features==64
    assert model.classifier.out_features==100
    assert torch.count_nonzero(model.classifier.bias).item()==0
    assert model.backbone_description["implementation"]=="vendored_original_FedCLASS_architecture"
    model.eval()
    x=torch.randn(4,3,32,32)
    with torch.no_grad():
        feats=model.features(x)
        output=model(x)
        source_manual=F.linear(feats,model.classifier.weight,model.classifier.bias)
    assert feats.shape==(4,64)
    assert output.shape==(4,100)
    torch.testing.assert_close(output,source_manual)


def test_fedclass_cli_generates_both_plans_without_cuda(tmp_path,capsys):
    for mode in ("authors_exemplars","none"):
        main(["--memory-mode",mode,"--plan-only","--output-dir",str(tmp_path)])
        location=tmp_path/mode/"seed0_lr0.03_epochs10_source_v3"
        data=json.loads((location/"design.json").read_text())
        assert data["suite"]=="FedCLASS"
        assert data["concurrent_processes"]==2
        assert data["full_and_offline_same_gpu"]==0
        assert data["total_jobs"]==2
        assert data["training_intervals"]==20
        assert data["original_first_k_per_class"]==(50 if mode=="authors_exemplars" else 0)
    with pytest.raises(SystemExit):
        main(["--gpu","1","--plan-only","--output-dir",str(tmp_path)])
    with pytest.raises(ValueError):
        make_fedclass_jobs(memory_mode="authors_exemplars",replay_ratio=0)


def test_fedclass_launcher_starts_two_workers_before_wait_and_both_gpu0(
        monkeypatch,tmp_path):
    import cafcl.fedclass_primary as runner
    import cafcl.data as data
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(data,"get_data_bank",lambda *a,**kw:None)
    created=[]
    class FakeProcess:
        pid=1024
        def wait(self):
            assert len(created)==2, "First worker waited before second started"
            return 0
        def poll(self):return 0
        def terminate(self):raise AssertionError("Do not terminate successful workers")
    def popen(cmd,**kwargs):
        created.append((cmd,kwargs))
        return FakeProcess()
    def report(root,jobs):
        target=root/"comparison"/"comparison.md"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text("Two matched Replay/FedAvg runs,")
        return {"mock_success":True}
    monkeypatch.setattr(runner.subprocess,"Popen",popen)
    monkeypatch.setattr(runner,"build_report",report)
    runner.main(["--no-download","--output-dir",str(tmp_path),"--memory-mode","authors_exemplars","--permit-adapted-legacy"])
    assert len(created)==2
    assert [cmd[-2:] for cmd,_ in created]==[
        ["--worker","0"],["--worker","1"]]
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="0" for _,kwargs in created)
    assert all(kwargs["stderr"]==runner.subprocess.STDOUT for _,kwargs in created)
    assert "FedCLASS source CE + authors original first-50/class exemplars" in (
        tmp_path/"authors_exemplars"/"seed0_lr0.03_epochs10_source_v3"/
        "comparison"/"comparison.md").read_text()


@pytest.mark.parametrize("method,memory",[
    ("replay",2),
    ("sgd",0),
])
def test_offline_fedavg_complete_synthetic_end_to_end(tmp_path,method,memory):
    """Full and hidden training both run; only upload visibility differs."""
    from cafcl.config import RunConfig
    from cafcl.simulator import run_experiment
    root=RunConfig(
        dataset="synthetic",model="test_smallcnn",
        stream_protocol=True,class_pair_protocol="strict",
        classes_per_local_task=2,local_tasks_per_round=2,
        num_clients=2,stages=1,rounds_per_stage=1,
        scenario="disjoint",stream_overlap_kind="none",
        stream_terminal_flush=True,share_continual_state=True,
        aggregator="fedavg",classwise_head_merge=False,
        selector="random",method=method,memory_per_class=memory,
        replay_ratio=(1. if memory else 0.),
        batch_size=64,epochs=1,lr=.01,seed=3,
        participation=1.,job_id=f"FULL_{method}",resume=False)
    offline=replace(root,participation=.5,stream_offline_policy="rotate_once",
                    job_id=f"HIDDEN_{method}")
    for cfg in (root,offline):
        outcome=run_experiment(cfg,str(tmp_path),verbose=False)
        assert outcome["status"]=="completed"
        rows=json.loads((tmp_path/cfg.job_id/"metrics.json").read_text())
        assert len(rows)==3
        assert rows[0]["locally_trained_count"]==2
        assert rows[0]["selected_count"]==(2 if cfg is root else 1)
        assert rows[-1]["terminal_communication_only_flush"]
        assert rows[-1]["selected_count"]==2
        assert rows[-1]["locally_trained_count"]==0
        assert rows[-1]["server_integrated_class_count"]==10
        assert set(map(int,rows[-1]["per_class"]))==set(range(10))
        if cfg is offline:
            first_hidden=next(iter(set(range(2))-set(rows[0]["selected_clients"])))
            assert rows[0]["client_federated_cl_versions"][str(first_hidden)]==0
            assert rows[0]["client_unsent_steps"][str(first_hidden)]>0
            assert len(rows[0]["per_sync_pending_classes"])==4
            assert rows[1]["stale_rebased_client_ids"], "Returning offline model was not rebased"


def test_original_fedclass_fedavg_matches_shared_fresh_weighted_aggregation():
    """The published FedCLASS FedAvg arithmetic equals the v16 fresh model mean."""
    import importlib.util
    from collections import OrderedDict
    from pathlib import Path
    from cafcl.simulator import aggregate

    source=Path(__file__).resolve().parents[1]/"FedCLASS"/"utils"/"fed_algorithm.py"
    spec=importlib.util.spec_from_file_location("_fedclass_original_aggregator_test",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    a=torch.nn.Linear(2,2,bias=False)
    b=torch.nn.Linear(2,2,bias=False)
    with torch.no_grad():
        a.weight.fill_(2.)
        b.weight.fill_(6.)
    original=module.FedAvg(
        [3,1],OrderedDict([("weight",torch.zeros_like(a.weight))]))
    expected=original.update([a,b],[0,1])["weight"]
    actual=aggregate(
        [{"weights":{"weight":a.weight.detach().clone()},"examples":3},
         {"weights":{"weight":b.weight.detach().clone()},"examples":1}],
        {"weight":torch.zeros_like(a.weight)},"fedavg")["weight"]
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(actual,torch.full((2,2),3.))
