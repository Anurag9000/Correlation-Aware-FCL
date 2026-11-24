"""Independent stationary reference; not an archived continual-run reproduction."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from .data import get_data_bank
from .models import create_model, copy_cpu_state
from .diagnostic_campaign import atomic_json

CHECKPOINTS={1,2,5,10,20,50,100}


def check_data(root, download=False):
    bank=get_data_bank("cifar100",root,download=download,seed=0,class_subset=tuple(range(30)))
    if bank.num_classes!=30 or not all(len(d)>0 for d in (bank.train,bank.validation,bank.test)):
        raise ValueError("Expected 30-class train/validation/test splits")
    return bank


def partitions(targets,k,policy,seed):
    targets=list(map(int,targets))
    rng=random.Random(seed)
    parts=[[] for _ in range(k)]
    for c in range(30):
        ids=[i for i,y in enumerate(targets) if y==c]
        rng.shuffle(ids)
        if policy=="disjoint_labels":
            parts[c*k//30].extend(ids)
        else:
            for j,i in enumerate(ids):
                parts[j%k].append(i)
    assert sorted(i for part in parts for i in part)==list(range(len(targets)))
    assert all(parts)
    return parts


def average(states,sizes):
    total=sum(sizes)
    result={}
    for key,v in states[0].items():
        if v.is_floating_point():
            value=torch.zeros_like(v,dtype=torch.float64)
            for state,n in zip(states,sizes):
                value.add_(state[key].to(torch.float64),alpha=n/total)
            result[key]=value.to(v.dtype)
        else:
            result[key]=torch.stack([s[key] for s in states]).amax(dim=0)
    return result


def evaluate(model,dataset,device):
    model.eval()
    hits=0; total=0; loss=0.
    with torch.no_grad():
        for x,y in DataLoader(dataset,batch_size=64,shuffle=False):
            x,y=x.to(device),y.to(device)
            z=model(x)
            loss+=nn.functional.cross_entropy(z,y,reduction="sum").item()
            hits+=(z.argmax(1)==y).sum().item()
            total+=y.numel()
    return {"accuracy":hits/total,"loss":loss/total,"examples":total}


def stationary_probe(model,bank,parts,labels,objective,device,limit=128):
    """Fixed TRAIN panels; common and actual objective gradient estimates."""
    model.eval()
    params=list(model.parameters())
    global_grad=[torch.zeros_like(p,device="cpu",dtype=torch.float64) for p in params]
    actual_grad=[torch.zeros_like(p,device="cpu",dtype=torch.float64) for p in params]
    total=sum(map(len,parts));global_loss=actual_loss=0.
    for cid,ids in enumerate(parts):
        sample=random.Random(99437+cid).sample(ids,min(limit,len(ids)))
        batch=[bank.train[i] for i in sample]
        x=torch.stack([a for a,_ in batch]).to(device)
        y=torch.tensor([int(b) for _,b in batch],device=device)
        w=len(ids)/total
        z=model(x)
        loss=nn.functional.cross_entropy(z,y)
        g=torch.autograd.grad(loss,params)
        global_loss+=w*float(loss.detach())
        for a,b in zip(global_grad,g):a.add_(b.detach().cpu().double(),alpha=w)
        if objective=="restricted_local":
            active=torch.tensor(labels[cid],device=device)
            mapping=torch.full((30,),-1,dtype=torch.long,device=device)
            mapping[active]=torch.arange(len(active),device=device)
            loss=nn.functional.cross_entropy(model(x)[:,active],mapping[y])
            g=torch.autograd.grad(loss,params)
        actual_loss+=w*float(loss.detach())
        for a,b in zip(actual_grad,g):a.add_(b.detach().cpu().double(),alpha=w)
    return {"sampled_global_gradient_norm_squared":sum(float(x.square().sum()) for x in global_grad),
            "sampled_actual_objective_gradient_norm_squared":sum(float(x.square().sum()) for x in actual_grad),
            "sampled_global_loss":global_loss,"sampled_actual_objective_loss":actual_loss}


def train(job,output,root,device,download):
    if job["family"]!="FL" or job.get("binding")!="stationary_reference":
        raise ValueError("Unsupported job")
    seed=job["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    bank=check_data(root,download)
    k=max(1,job["clients"])
    parts=partitions(bank.train.targets,k,job["partition"],seed)
    labels=[sorted({int(bank.train.targets[i]) for i in part}) for part in parts]
    atomic_json(output/"partition.json",{"classes":list(range(30)),"indices":parts,"sizes":list(map(len,parts)),"class_counts":[dict(Counter(int(bank.train.targets[i]) for i in part)) for part in parts],"partition_sha256":hashlib.sha256(json.dumps(parts).encode()).hexdigest()})
    model=create_model("resnet18_cifar32",30,learning_setting="class_il").to(device)
    # Central reference uses the same 100 blocks of ten epochs as one-client FL.
    # Optimizers reset each communication block for all arms, deliberately.
    rounds=job["rounds_per_stage"] or 100
    steps=0; processed=0
    rows=[]
    for r in range(1,rounds+1):
        start=copy_cpu_state(model)
        states=[]
        for cid,ids in enumerate(parts):
            model.load_state_dict(start)
            model.train()
            opt=torch.optim.SGD(model.parameters(),lr=.03,momentum=.9,weight_decay=5e-4)
            generator=torch.Generator().manual_seed(seed+100003*r+cid)
            loader=DataLoader(Subset(bank.train,ids),batch_size=64,shuffle=True,generator=generator)
            active=torch.tensor(labels[cid],device=device)
            remap=torch.full((30,),-1,dtype=torch.long,device=device)
            remap[active]=torch.arange(len(active),device=device)
            for _ in range(job["local_epochs"]):
                for x,y in loader:
                    x,y=x.to(device),y.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits=model(x)
                    if job["objective"]=="restricted_local":
                        logits,y=logits[:,active],remap[y]
                    loss=nn.functional.cross_entropy(logits,y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    loss.backward(); opt.step()
                    steps+=1; processed+=len(y)
            states.append(copy_cpu_state(model))
        model.load_state_dict(average(states,list(map(len,parts))))
        val=evaluate(model,bank.validation,device)
        row={"round":r,"validation_accuracy":val["accuracy"],"validation_loss":val["loss"],"optimizer_steps":steps,"processed_examples":processed}
        if r in CHECKPOINTS or r==rounds:
            test=evaluate(model,bank.test,device)
            row.update(test_accuracy=test["accuracy"],test_loss=test["loss"])
            row.update(stationary_probe(model,bank,parts,labels,job["objective"],device))
            checkpoints=output/"checkpoints";checkpoints.mkdir(exist_ok=True)
            torch.save({"round":r,"state":copy_cpu_state(model)},checkpoints/f"round_{r:03d}.pt")
        rows.append(row)
        with (output/"metrics.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=["round","validation_accuracy","validation_loss","test_accuracy","test_loss","optimizer_steps","processed_examples","sampled_global_gradient_norm_squared","sampled_actual_objective_gradient_norm_squared","sampled_global_loss","sampled_actual_objective_loss"])
            writer.writeheader();writer.writerows(rows)
        print(json.dumps(row),flush=True)
    atomic_json(output/"result_summary.json",{"job_id":job["job_id"],"final":rows[-1],"rounds":rounds,"protocol":"stationary_reference","optimizer_policy":"reset_each_block","augmentation":"none (shared deterministic repository preprocessing)","bn_policy":"sample_weighted_floating_buffers_max_integer_buffers"})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--data-root",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--download",action="store_true")
    a=p.parse_args()
    train(json.loads(Path(a.job).read_text()),Path(a.output),a.data_root,a.device,a.download)


if __name__=="__main__":
    main()
