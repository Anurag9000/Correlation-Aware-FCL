"""Opt-in matched-design FCL reference. Not a V25/V27 reproduction.

Shared replay is simulated TRAIN-index sharing, not private FL. The reference
uses global observed-class CE or semantic task heads, persistent SGD,
sample-weighted full-model FedAvg and stale-delta rebasing. No test feedback.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import combinations
import copy
import csv
import json
import math
from pathlib import Path
import random

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .data import get_data_bank, data_partition_manifest
from .models import create_model, copy_cpu_state
from .diagnostic_campaign import atomic_json
from .diagnostic_designs import compile_matched, batch_stream, plan_audit, stable_seed
from .stationary_diagnostic import average


def to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:to_cpu(v) for k,v in value.items()}
    if isinstance(value,list):
        return [to_cpu(v) for v in value]
    if isinstance(value,tuple):
        return tuple(to_cpu(v) for v in value)
    return copy.deepcopy(value)


def records(units):
    return [(i,tuple(u["head"])) for u in units for i in u["indices"]]


def capped_memory(items,labels,seed,budget=50):
    by=defaultdict(dict)
    for i,h in items:
        by[int(labels[i])][int(i)]=(int(i),tuple(h))
    return [x for c in sorted(by) for x in sorted(by[c].values(),
            key=lambda x:(stable_seed(seed,"memory",x[0]),x[0]))[:budget]]


def load_records(dataset,rows,device):
    batch=[dataset[i] for i,_ in rows]
    return (torch.stack([x for x,_ in batch]).to(device),
            torch.tensor([int(y) for _,y in batch],device=device),
            [tuple(h) for _,h in rows])


def classification_loss(model,x,y,heads,setting,observed):
    if setting=="class_il":
        allowed=torch.tensor(sorted(observed),device=x.device)
        lookup=torch.full((model.num_classes,),-1,dtype=torch.long,device=x.device)
        lookup[allowed]=torch.arange(len(allowed),device=x.device)
        targets=lookup[y]
        if (targets<0).any():
            raise ValueError("Label not in global observed set")
        return F.cross_entropy(model(x).index_select(1,allowed),targets)
    features=model.features(x)
    total=features.new_zeros(())
    for h in sorted(set(heads)):
        idx=torch.tensor([i for i,v in enumerate(heads) if v==h],device=x.device)
        allowed=torch.tensor(h,device=x.device)
        truth=y[idx,None]==allowed[None,:]
        if not truth.any(1).all():
            raise ValueError("Semantic head provenance mismatch")
        total=total+F.cross_entropy(model.classifier(features[idx],h),truth.long().argmax(1),reduction="sum")
    return total/len(y)


def train_block(model,state,opt_state,units,replay,bank,job,stage,cid,r,observed,device):
    model.load_state_dict(state);model.train()
    opt=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9,weight_decay=5e-4)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    # MO: two sequential atomic tasks; MC/CM: mixed semantic units per stage.
    groups=[records([u]) for u in units] if job["family"]=="MO" else [records(units)]
    counts={"optimizer_steps":0,"unique_current_visits":0,"current_draws":0,"replay_draws":0}
    for slot,group in enumerate(groups):
        epochs=job["local_epochs"]
        seed=stable_seed(job["seed"],"batch",stage,cid,slot)
        for rows,real,current,history in batch_stream(group,replay,bank.train.targets,seed,epochs,r*epochs):
            x,y,heads=load_records(bank.train,rows,device)
            opt.zero_grad(set_to_none=True)
            loss=classification_loss(model,x,y,heads,job["il"],observed)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward();opt.step()
            counts["optimizer_steps"]+=1
            counts["unique_current_visits"]+=real
            counts["current_draws"]+=current
            counts["replay_draws"]+=history
    return copy_cpu_state(model),to_cpu(opt.state_dict()),counts


@torch.no_grad()
def evaluate(model,state,dataset,heads,setting,observed,device,limit=None):
    """Independent semantic-head metrics in TIL; per-class metrics in CIL.

    Evaluate each held-out example once through the backbone. TIL heads use
    the actual semantic task ID, not a class-membership oracle union.
    """
    if not heads:
        return {}
    model.load_state_dict(state);model.eval()
    labels={c for h in heads for c in h}
    by=defaultdict(list)
    for i,y in enumerate(dataset.targets):
        if int(y) in labels:
            by[int(y)].append(i)
    ids=[i for c in sorted(by) for i in (by[c] if limit is None else by[c][:limit])]
    accum=defaultdict(lambda:[0,0,0.])
    allowed=torch.tensor(sorted(observed),device=device)
    lookup=torch.full((model.num_classes,),-1,dtype=torch.long,device=device)
    lookup[allowed]=torch.arange(len(allowed),device=device)
    for x,y in DataLoader(Subset(dataset,ids),batch_size=64,shuffle=False):
        x,y=x.to(device),y.to(device)
        features=model.features(x)
        if setting=="class_il":
            logits=model.classifier(features).index_select(1,allowed)
            losses=F.cross_entropy(logits,lookup[y],reduction="none")
            pred=allowed[logits.argmax(1)]
            for c in y.unique().tolist():
                mask=y==c; a=accum[str(c)]
                a[0]+=int((pred[mask]==y[mask]).sum());a[1]+=int(mask.sum());a[2]+=float(losses[mask].sum())
        else:
            for h in sorted(heads):
                classes=torch.tensor(h,device=device)
                mask=(y[:,None]==classes[None,:]).any(1)
                if not mask.any():
                    continue
                logits=model.classifier(features[mask],h)
                truth=(y[mask,None]==classes[None,:]).long().argmax(1)
                a=accum[head_key(h)]
                a[0]+=int((logits.argmax(1)==truth).sum());a[1]+=int(mask.sum())
                a[2]+=float(F.cross_entropy(logits,truth,reduction="sum"))
    return {k:{"accuracy":a[0]/a[1],"loss":a[2]/a[1],"correct":a[0],"total":a[1]} for k,a in accum.items() if a[1]}


def head_key(h):
    return ":".join(map(str,h))


def targets(heads,setting):
    return {str(c) for h in heads for c in h} if setting=="class_il" else {head_key(h) for h in heads}


def macro(values,keys,metric="accuracy"):
    vals=[values[k][metric] for k in sorted(keys) if k in values]
    return sum(vals)/len(vals) if vals else None


def difference(a,b):
    return a-b if a is not None and b is not None else None


def candidate(base,local,anchor,lag,param_names,beta=1.):
    if lag<0:
        raise ValueError("Negative version lag")
    if not lag:
        return local
    scale=(1+lag)**(-beta)
    return {k:(v+(local[k]-anchor[k])*scale if k in param_names else v)
            for k,v in base.items()}


def aggregate(base,clients,cohort,version,param_names):
    uploads=[c for c in cohort if clients[c]["pending"]]
    if not uploads:
        return base,[]
    candidates=[candidate(base,clients[c]["state"],clients[c]["anchor"],
                          version-clients[c]["version"],param_names) for c in uploads]
    weights=[len(clients[c]["pending"]) for c in uploads]
    return average(candidates,weights),[{"client":c,"lag":version-clients[c]["version"],
               "unique_pending_samples":w,"weight":w/sum(weights)}
               for c,w in zip(uploads,weights)]


def grad_probe(model,base,bank,units,setting,observed,device,limit=32):
    """Sampled TRAIN current-objective gradient at identical server weights.

    Per-client sample count weights estimate global gradient; not a claim
    of exact full-dataset stationarity. Eval-mode normalization is explicit.
    """
    params=[p for p in model.parameters() if p.requires_grad]
    grads=[];weights=[]
    model.load_state_dict(base);model.eval()
    for us in units:
        rr=records(us)
        if not rr:
            continue
        sample=random.Random(stable_seed("gradient",len(rr))).sample(rr,min(limit,len(rr)))
        x,y,h=load_records(bank.train,sample,device)
        loss=classification_loss(model,x,y,h,setting,observed)
        g=torch.autograd.grad(loss,params,allow_unused=True)
        grads.append([v.detach().cpu() if v is not None else None for v in g]);weights.append(len(rr))
    total=sum(weights);norm=0.;pairs={}
    for j,p in enumerate(params):
        agg=torch.zeros_like(p,device="cpu",dtype=torch.float64)
        for g,w in zip(grads,weights):
            if g[j] is not None:
                agg.add_(g[j].double(),alpha=w/total)
        norm+=float(agg.square().sum())
    # Backbone-only pairwise alignment (head coordinates may differ).
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    for i,j in combinations(range(len(grads)),2):
        dot=ni=nj=0.
        for name,a,b in zip(names,grads[i],grads[j]):
            if name.startswith("classifier."):
                continue
            if a is not None:ni+=float(a.double().square().sum())
            if b is not None:nj+=float(b.double().square().sum())
            if a is not None and b is not None:dot+=float((a.double()*b.double()).sum())
        pairs[f"{i}:{j}"]=dot/math.sqrt(ni*nj) if ni and nj else None
    return {"sampled_global_current_gradient_norm_squared":norm,"pairwise_backbone_cosine":pairs,
            "split":"train","max_examples_per_client":limit,"bn_mode":"eval","no_feedback":True}


def run_reference(job,output,bank,device="cpu",model_name="resnet18_cifar32",subset_probes=False):
    if job.get("binding")!="matched_reference_v1":
        raise ValueError("No matched reference binding")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed=job["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    stages=compile_matched(job,bank.train.targets,bank.num_classes)
    all_heads={tuple(u["head"]) for s in stages for us in s["units"] for u in us}
    setting=job["il"]
    model=create_model(model_name,bank.num_classes,allow_test_model=model_name.startswith("test_"),
        task_class_sets=sorted(all_heads) if setting=="task_il" else None,learning_setting=setting).to(device)
    param_names={n for n,p in model.named_parameters()}
    server=copy_cpu_state(model);version=0
    clients=[{"state":server,"anchor":server,"version":0,"optimizer":None,"memory":[],"shared":[],"pending":{}}
             for _ in range(job["clients"])]
    archive=[];prior=set();observed=set();peak={};rows=[];all_counts=defaultdict(int)
    audit=plan_audit(job,stages,bank.train.targets)
    atomic_json(output/"partition.json",{**audit,"split":data_partition_manifest(bank,"cifar100" if bank.num_classes==100 else "synthetic",.1,1.)})
    atomic_json(output/"schedule.json",stages)
    diagnostics=output/"per_round_metrics";diagnostics.mkdir(exist_ok=True)
    checkpoints=output/"checkpoints";checkpoints.mkdir(exist_ok=True)
    budget=job.get("replay_budget_per_class",50)
    ctr=0
    def eval_panel(state,heads,limit=4):
        return evaluate(model,state,bank.validation,heads,setting,observed,device,limit)
    def commit(cohort,stage,current,flush=False,train_counts=None,local=None,gradient=None):
        nonlocal server,version,archive,ctr
        ctr+=1
        base=server
        history_keys=targets(prior,setting);current_keys=targets(current,setting)
        before=eval_panel(base,prior|current)
        updated,uploads=aggregate(base,clients,cohort,version,param_names)
        after=eval_panel(updated,prior|current)
        hist_before=macro(before,history_keys);hist_after=macro(after,history_keys)
        current_before=macro(before,current_keys);current_after=macro(after,current_keys)
        subsets=[]
        if subset_probes and len(clients)<=5 and not flush:
            for n in (2,3):
                for ids in combinations(range(len(clients)),n):
                    alt,_=aggregate(base,clients,ids,version,param_names)
                    ev=eval_panel(alt,prior|current)
                    subsets.append({"cohort":list(ids),"history_accuracy":macro(ev,history_keys),
                        "current_accuracy":macro(ev,current_keys),"history_loss":macro(ev,history_keys,"loss")})
        server=updated
        if uploads:version+=1
        actual=[u["client"] for u in uploads]
        for c in actual:
            archive=capped_memory(archive+list(clients[c]["pending"].values()),bank.train.targets,seed,budget)
        sync=[]
        for c in actual:
            state=clients[c]
            pre=eval_panel(state["state"],prior) if prior else {}
            sync.append({"client":c,"download_history_drop":difference(macro(pre,history_keys),hist_after)})
            state.update(state=server,anchor=server,version=version,shared=list(archive),pending={})
        row={"sync":ctr,"stage":stage,"flush":flush,"uploaders":actual,
             "history_before":hist_before,"history_after":hist_after,
             "aggregation_forgetting":difference(hist_before,hist_after),
             "current_before":current_before,"current_after":current_after,
             "history_loss_before":macro(before,history_keys,"loss"),"history_loss_after":macro(after,history_keys,"loss"),
             "current_loss_before":macro(before,current_keys,"loss"),"current_loss_after":macro(after,current_keys,"loss"),
             "shared_replay_size":len(archive),"uploads":uploads,"local":local or [],
             "download":sync,"counts":train_counts or {},"gradient":gradient,"counterfactual_subsets":subsets,
             "diagnostic_split":"validation","panel_examples_per_class":4}
        atomic_json(diagnostics/f"sync_{ctr:04d}.json",row)
        return row
    for s in stages:
        t=s["stage"];current={tuple(u["head"]) for us in s["units"] for u in us}
        observed.update(s["classes"])
        # Freeze replay at parent-stage start: compute matched arms process
        # identical current and replay draws under different sync frequency.
        frozen=[capped_memory(c["memory"]+c["shared"],bank.train.targets,seed,budget) for c in clients]
        for rr in range(job["rounds_per_stage"]):
            local=[];counts=defaultdict(int)
            gradient=grad_probe(model,server,bank,s["units"],setting,observed,device) if rr==0 else None
            for cid,us in enumerate(s["units"]):
                c=clients[cid];hist_keys=targets(prior,setting)
                own_heads={tuple(u["head"]) for u in us}
                own_keys=targets(own_heads,setting)
                pre=eval_panel(c["state"],prior|own_heads)
                new,opt,cc=train_block(model,c["state"],c["optimizer"],us,frozen[cid],bank,job,t,cid,rr,observed,device)
                post=eval_panel(new,prior|own_heads)
                local.append({"client":cid,"history_before":macro(pre,hist_keys),"history_after":macro(post,hist_keys),
                              "local_forgetting":difference(macro(pre,hist_keys),macro(post,hist_keys)),
                              "current_before":macro(pre,own_keys),"current_after":macro(post,own_keys)})
                c.update(state=new,optimizer=opt)
                c["pending"].update({i:(i,h) for i,h in records(us)})
                for key,v in cc.items():counts[key]+=v;all_counts[key]+=v
            if rr==job["rounds_per_stage"]-1:
                for cid,us in enumerate(s["units"]):
                    c=clients[cid]
                    c["memory"]=capped_memory(c["memory"]+records(us),bank.train.targets,seed,budget)
            row=commit(s["cohort"],t,current,train_counts=dict(counts),local=local,gradient=gradient)
            row["round_in_stage"]=rr+1
            rows.append(row)
        # Official test is diagnostic only. Fixed historical keys distinguish
        # forgetting from the newly expanded evaluation denominator.
        test=evaluate(model,server,bank.test,prior|current,setting,observed,device)
        val=evaluate(model,server,bank.validation,prior|current,setting,observed,device)
        keys=targets(prior,setting)
        forgetting=[max(0.,peak[k]-test[k]["accuracy"]) for k in keys if k in peak and k in test]
        for k,a in test.items():peak[k]=max(peak.get(k,0.),a["accuracy"])
        rows[-1].update(test_current=macro(test,targets(current,setting)),test_history=macro(test,keys),
                       test_global=macro(test,set(test)),test_forgetting=sum(forgetting)/len(forgetting) if forgetting else None)
        atomic_json(output/f"stage_{t:02d}_evaluation.json",{"stage":t,"test":test,"validation":val,"current":sorted(targets(current,setting)),"historical":sorted(keys)})
        torch.save({"server":server,"version":version,"stage":t},checkpoints/"latest.pt")
        prior.update(current)
    if job["participation"]<1:
        row=commit(list(range(len(clients))),len(stages),set(),flush=True)
        final=evaluate(model,server,bank.test,prior,setting,observed,device)
        row["test_global"]=macro(final,set(final))
        row["test_forgetting"]=sum(max(0.,peak[k]-v["accuracy"]) for k,v in final.items())/len(final)
        rows.append(row);atomic_json(output/"terminal_evaluation.json",final)
        torch.save({"server":server,"version":version,"stage":len(stages)},checkpoints/"latest.pt")
    else:
        final=test
    # Stable scalar round table; detailed panels in individual JSON files.
    fields=["sync","stage","round_in_stage","flush","history_before","history_after","aggregation_forgetting",
            "current_before","current_after","test_current","test_history","test_global","test_forgetting","shared_replay_size"]
    with (output/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    end_stages=[r for r in rows if "test_current" in r]
    values=[r["test_current"] for r in end_stages if r["test_current"] is not None]
    atomic_json(output/"result_summary.json",{"job_id":job["job_id"],"protocol":"matched_reference_v1",
        "learning_syncs":sum(not r["flush"] for r in rows),"terminal_flush":job["participation"]<1,
        "final_accuracy":macro(final,set(final)),"current_task_average_accuracy":sum(values)/len(values),
        "final_forgetting":rows[-1].get("test_forgetting"),"counts":dict(all_counts),
        "allocation_sha256":audit["allocation_sha256"],"replay_budget_per_class":budget,
        "evaluation":"global_seen_class or independent_semantic_task_head",
        "aggregation":"sample_weighted_full_model; stale parameter delta/(1+lag); stale buffers unchanged",
        "optimizer":"persistent SGD", "augmentation":"none", "subset_probes":subset_probes})
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True);p.add_argument("--output",required=True)
    p.add_argument("--data-root",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--download",action="store_true");p.add_argument("--subset-probes",action="store_true")
    a=p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    bank=get_data_bank("cifar100",a.data_root,download=a.download,seed=0)
    run_reference(json.loads(Path(a.job).read_text()),a.output,bank,a.device,subset_probes=a.subset_probes)


if __name__=="__main__":
    main()
