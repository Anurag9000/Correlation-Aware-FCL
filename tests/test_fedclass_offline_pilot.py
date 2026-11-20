"""FedCLASS CE/FedAvg first-four-then-two offline pilot regression suite."""
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
    ("reservoir","replay",50),
    ("none","sgd",0),
])
def test_paired_fedclass_configs_and_schedule(mode,method,count):
    full,hidden=make_fedclass_jobs(memory_mode=mode,seed=13,epochs=10,lr=.03)
    assert full.model==hidden.model=="fedclass_resnet8"
    assert full.method==hidden.method==method
    assert full.aggregator==hidden.aggregator=="fedavg"
    assert not full.classwise_head_merge and not hidden.classwise_head_merge
    assert full.num_clients==hidden.num_clients==5
    assert full.participation==1. and hidden.participation==.8
    assert hidden.stream_offline_policy=="rotate_once"
    assert full.batch_size==hidden.batch_size==64
    assert full.epochs==hidden.epochs==10
    assert full.momentum==hidden.momentum==.9
    assert full.weight_decay==hidden.weight_decay==.0005
    assert full.optimizer_state_policy==hidden.optimizer_state_policy=="persist"
    assert full.memory_per_class==hidden.memory_per_class==count
    assert full.replay_reservoir_fraction==(1/9 if count else 0)
    assert full.share_continual_state and hidden.share_continual_state
    assert full.learning_setting=="class_il" and hidden.learning_setting=="class_il"
    assert full.class_pair_protocol=="strict" and hidden.class_pair_protocol=="strict"
    assert set(full.to_dict())==set(hidden.to_dict())
    diff={k for k in full.to_dict() if full.to_dict()[k]!=hidden.to_dict()[k]}
    assert diff=={"job_id","participation","stream_offline_policy"}

    stream=make_task_schedule(100,full)
    assert len(stream)==50
    assert sorted(c for *_,labels,recur in stream for c in labels)==list(range(100))
    assert all(len(labels)==2 for _,_,_,labels,_ in stream)
    for cid in range(5):
        assert [labels for rnd,c,_,labels,_ in stream if rnd==0 and c==cid]==[
            (4*cid,4*cid+1),(4*cid+2,4*cid+3)]
        for rnd in range(1,9):
            assert [labels for r,c,_,labels,_ in stream if r==rnd and c==cid]==[
                (20+10*(rnd-1)+cid*2,21+10*(rnd-1)+cid*2)]
    plan=audited_plan((full,hidden))
    assert plan["training_intervals"]==9
    assert plan["terminal_communication_only_flush"]
    assert plan["total_jobs"]==2
    assert plan["private_reservoir_per_class"]==count
    assert plan["expected_server_exemplars_after_flush"]==100*count
    missing=[]
    for rnd in range(9):
        online=set(select_offline_cohort(hidden,rnd,10,list(range(4))))
        assert len(online)==4
        missing.append(next(iter(set(range(5))-online)))
    assert set(missing[:5])==set(range(5))
    assert missing[5:]==missing[:4]
    assert set(select_offline_cohort(hidden,9,10,[]))==set(range(5))
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
    for mode in ("reservoir","none"):
        main(["--memory-mode",mode,"--plan-only","--output-dir",str(tmp_path)])
        location=tmp_path/mode/("seed0_lr0.03_epochs10_rr1"
                                if mode=="reservoir" else "seed0_lr0.03_epochs10")
        data=json.loads((location/"design.json").read_text())
        assert data["suite"]=="FedCLASS"
        assert data["concurrent_processes"]==2
        assert data["full_and_offline_same_gpu"]==0
        assert data["total_jobs"]==2
        assert data["training_intervals"]==9
        assert data["per_class_reservoir_capacity"]==(50 if mode=="reservoir" else 0)
    with pytest.raises(SystemExit):
        main(["--gpu","1","--plan-only","--output-dir",str(tmp_path)])
    with pytest.raises(ValueError):
        make_fedclass_jobs(memory_mode="reservoir",replay_ratio=0)


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
    runner.main(["--no-download","--output-dir",str(tmp_path),"--memory-mode","reservoir"])
    assert len(created)==2
    assert [cmd[-2:] for cmd,_ in created]==[
        ["--worker","0"],["--worker","1"]]
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"]=="0" for _,kwargs in created)
    assert all(kwargs["stderr"]==runner.subprocess.STDOUT for _,kwargs in created)
    assert "FedCLASS CE + 50/class Algorithm R" in (
        tmp_path/"reservoir"/"seed0_lr0.03_epochs10_rr1"/
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
