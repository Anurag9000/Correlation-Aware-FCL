"""Matched-acquisition forgetting forks from identical learned checkpoints."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import torch

from .diagnostic_campaign import atomic_json,effective_jobs,sha_file
from .diagnostic_analysis import latest_valid,write_csv
from .diagnostic_designs import compile_matched
from .data import get_data_bank
from .models import create_model,copy_cpu_state
from .matched_diagnostic import (aggregate,train_block,records,evaluate,targets,macro,
    grad_probe,difference)


def fork_branch(anchor,job,bank,device,output,model_name="resnet18_cifar32"):
    """Train only the next parent stage; all clients start from identical state.

    Replay archive and evaluation cohort are fixed. Client momentum starts
    empty in every branch; no claim of continuing the historical trajectory.
    """
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stages=compile_matched(job,bank.train.targets,bank.num_classes)
    start=anchor["stage"]+1
    if start>=len(stages):raise ValueError("No subsequent stage")
    s=stages[start]
    all_heads={tuple(u["head"]) for stage in stages for us in stage["units"] for u in us}
    prior={tuple(h) for h in anchor["prior_heads"]}
    current={tuple(u["head"]) for us in s["units"] for u in us}
    setting=job["il"];observed=set(anchor["observed"])|set(s["classes"])
    model=create_model(model_name,bank.num_classes,allow_test_model=model_name.startswith("test_"),
        task_class_sets=sorted(all_heads) if setting=="task_il" else None,learning_setting=setting).to(device)
    server=anchor["server"];version=0;model.load_state_dict(server)
    names={n for n,p in model.named_parameters()}
    history_keys=targets(prior,setting);current_keys=targets(current,setting)
    clients=[{"state":server,"anchor":server,"version":0,"optimizer":None,"pending":{}}
             for _ in range(job["clients"])]
    archive=[(i,tuple(h)) for i,h in anchor["archive"]]
    before=evaluate(model,server,bank.validation,prior|current,setting,observed,device)
    old_only=evaluate(model,server,bank.validation,prior,setting,set(anchor["observed"]),device)
    base_history=macro(before,history_keys)
    rows=[{"round":0,"current_accuracy":macro(before,current_keys),"history_accuracy":base_history,
           "history_drop":0.,"history_accuracy_old_competitors":macro(old_only,history_keys),
           "optimizer_steps":0,"current_draws":0,"replay_draws":0}]
    total_steps=current_draws=replay_draws=0
    for rr in range(job["rounds_per_stage"]):
        for cid,us in enumerate(s["units"]):
            c=clients[cid]
            st,opt,counts=train_block(model,c["state"],c["optimizer"],us,archive,bank,job,start,cid,rr,observed,device)
            c.update(state=st,optimizer=opt,pending={i:(i,h) for i,h in records(us)})
            total_steps+=counts["optimizer_steps"];current_draws+=counts["current_draws"];replay_draws+=counts["replay_draws"]
        server,_=aggregate(server,clients,list(range(len(clients))),version,names)
        version+=1
        for c in clients:c.update(state=server,anchor=server,version=version,pending={})
        ev=evaluate(model,server,bank.validation,prior|current,setting,observed,device)
        grad=grad_probe(model,server,bank,s["units"],setting,observed,device)
        rows.append({"round":rr+1,"current_accuracy":macro(ev,current_keys),"history_accuracy":macro(ev,history_keys),
            "history_drop":difference(base_history,macro(ev,history_keys)),
            "current_loss":macro(ev,current_keys,"loss"),"history_loss":macro(ev,history_keys,"loss"),
            "sampled_gradient_norm_squared":grad["sampled_weighted_current_objective_gradient_norm_squared"],
            "optimizer_steps":total_steps,"current_draws":current_draws,"replay_draws":replay_draws})
    final=evaluate(model,server,bank.test,prior|current,setting,observed,device)
    # Thresholds predeclared; use earliest measured validation crossing.
    # Never extrapolate a threshold that was not achieved.
    thresholds={}
    for threshold in (.5,.6,.7,.8,.9):
        hit=next((r for r in rows if r["current_accuracy"]>=threshold),None)
        thresholds[str(threshold)]=hit
    write_csv(output/"metrics.csv",rows)
    result={"job_id":job["job_id"],"clients":job["clients"],"partition":job["partition"],"il":setting,
            "historical_checkpoint_job":anchor["job_id"],"stage":start,"thresholds":thresholds,
            "final_validation":rows[-1],"final_test_current":macro(final,current_keys),"final_test_history":macro(final,history_keys),
            "no_test_feedback":True,"optimizer_reset_at_fork":True}
    atomic_json(output/"result_summary.json",result)
    return result


def run_forks(root,data_root,device,allow_reconstruction=False,subset_probes=False,download=False):
    root=Path(root);jobs=effective_jobs(allow_reconstruction,subset_probes)
    out=root/"forks";out.mkdir(parents=True,exist_ok=True)
    anchors={}
    for il in ("class_il","task_il"):
        j=next(x for x in jobs if x["family"]=="MC" and x["clients"]==2 and x["partition"]=="iid" and x["rounds_per_stage"]==1 and x["il"]==il)
        p=latest_valid(j,root)
        checkpoint=p/"checkpoints"/"fork_reference.pt" if p else None
        if checkpoint is None or not checkpoint.is_file():
            atomic_json(out/"fork_status.json",{"state":"blocked","reason":"Missing validated MC anchor checkpoint: "+il})
            return 2
        anchors[il]=checkpoint
    bank=get_data_bank("cifar100",data_root,download=download,seed=0)
    results=[]
    for il,checkpoint in anchors.items():
        anchor=torch.load(checkpoint,map_location="cpu",weights_only=True)
        digest=sha_file(checkpoint)
        for partition in ("iid","controlled_label_skew"):
            for k in (2,3,10):
                j=next(
                    x for x in jobs if x["family"]=="MC" and x["clients"]==k and x["partition"]==partition and x["rounds_per_stage"]==10 and x["il"]==il)
                # All branches use 10 communication rounds ×2 local epochs.
                j={**j,"job_id":f"FORK-{il}-{partition}-K{k}","local_epochs":2}
                path=out/il/partition/f"K{k}"
                spec={"job":j,"anchor_sha256":digest}
                if (path/"result_summary.json").is_file() and (path/"spec.json").is_file() and json.loads((path/"spec.json").read_text())==spec:
                    result=json.loads((path/"result_summary.json").read_text())
                else:
                    if path.exists() and any(path.iterdir()):
                        path=path.parent/(path.name+"_"+digest[:12])
                    path.mkdir(parents=True,exist_ok=True)
                    atomic_json(path/"spec.json",spec)
                    result=fork_branch(anchor,j,bank,device,path)
                results.append(result)
    rows=[]
    for result in results:
        for threshold,hit in result["thresholds"].items():
            rows.append({"job_id":result["job_id"],"clients":result["clients"],"il":result["il"],
                         "partition":result["partition"],"threshold":float(threshold),"reached":hit is not None,
                         **(hit or {})})
    write_csv(out/"matched_acquisition_forgetting.csv",rows)
    atomic_json(out/"fork_status.json",{"state":"completed","branches":len(results),"results":results})
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="runs/diagnostic_campaign")
    p.add_argument("--data-root",default="data");p.add_argument("--device",default="cuda:0")
    p.add_argument("--allow-reconstruction",action="store_true");p.add_argument("--subset-probes",action="store_true")
    a=p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    raise SystemExit(run_forks(Path(a.output).resolve(),a.data_root,a.device,a.allow_reconstruction,a.subset_probes))


if __name__=="__main__":main()
