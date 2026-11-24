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
from .diagnostic_designs import (compile_matched, compile_reconstructed, batch_stream,
    plan_audit, stable_seed, head_labels, model_head)
from .stationary_diagnostic import average
from .joint.mechanistic import alignment
from .joint.update_diagnostics import update_geometry


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


def capped_memory(items,labels,seed,budget=50,by_occurrence=False):
    by=defaultdict(dict)
    for i,h in items:
        key=(int(labels[i]),tuple(h)) if by_occurrence else (int(labels[i]),)
        by[key][int(i)]=(int(i),tuple(h))
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
        allowed=torch.tensor(head_labels(h),device=x.device)
        truth=y[idx,None]==allowed[None,:]
        if not truth.any(1).all():
            raise ValueError("Semantic head provenance mismatch")
        total=total+F.cross_entropy(model.classifier(features[idx],model_head(h)),truth.long().argmax(1),reduction="sum")
    return total/len(y)


def train_block(model,state,opt_state,units,replay,bank,job,stage,cid,r,observed,device):
    reconstructed=job["binding"]=="historical_reconstruction_v1"
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
        for rows,real,current,history in batch_stream(group,replay,bank.train.targets,seed,epochs,r*epochs,paired_padding=reconstructed):
            x,y,heads=load_records(bank.train,rows,device)
            if reconstructed:
                x=crop_flip(x,stable_seed(seed,r,counts["optimizer_steps"]))
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
    labels={c for h in heads for c in head_labels(h)}
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
                classes=torch.tensor(head_labels(h),device=device)
                mask=(y[:,None]==classes[None,:]).any(1)
                if not mask.any():
                    continue
                logits=model.classifier(features[mask],model_head(h))
                truth=(y[mask,None]==classes[None,:]).long().argmax(1)
                a=accum[head_key(h)]
                a[0]+=int((logits.argmax(1)==truth).sum());a[1]+=int(mask.sum())
                a[2]+=float(F.cross_entropy(logits,truth,reduction="sum"))
    return {k:{"accuracy":a[0]/a[1],"loss":a[2]/a[1],"correct":a[0],"total":a[1]} for k,a in accum.items() if a[1]}


def head_key(h):
    return ":".join(map(str,h))


def targets(heads,setting):
    return {str(c) for h in heads for c in head_labels(h)} if setting=="class_il" else {head_key(h) for h in heads}


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


def aggregate(base,clients,cohort,version,param_names,beta=1.,fresh_only=False):
    uploads=[c for c in cohort if clients[c]["pending"] and (not fresh_only or clients[c]["version"]==version)]
    if not uploads:
        return base,[]
    candidates=[candidate(base,clients[c]["state"],clients[c]["anchor"],
                          version-clients[c]["version"],param_names,beta) for c in uploads]
    weights=[len(clients[c]["pending"]) for c in uploads]
    return average(candidates,weights),[{"client":c,"lag":version-clients[c]["version"],
               "unique_pending_samples":w,"weight":w/sum(weights),"damping":(1+version-clients[c]["version"])**(-beta)}
               for c,w in zip(uploads,weights)]


def grad_probe(model,base,bank,units,setting,observed,device,limit=32,competitors=None,replay=None):
    """Sampled TRAIN current-objective gradient at identical server weights.

    Per-client sample count weights estimate global gradient; not a claim
    of exact full-dataset stationarity. Eval-mode normalization is explicit.
    """
    params=[p for p in model.parameters() if p.requires_grad]
    grads=[];weights=[];loss_values=[];active_ids=[]
    model.load_state_dict(base);model.eval()
    for cid,us in enumerate(units):
        rr=records(us)
        if not rr:
            continue
        rng=random.Random(stable_seed("gradient",cid,len(rr)))
        sample=rng.sample(rr,min(limit,len(rr)))
        if replay is not None and replay[cid]:
            by=defaultdict(list)
            for row in replay[cid]:by[int(bank.train.targets[row[0]])].append(row)
            sample=sample+[rng.choice(by[rng.choice(sorted(by))]) for _ in range(len(sample))]
        x,y,h=load_records(bank.train,sample,device)
        loss=classification_loss(model,x,y,h,setting,competitors[cid] if competitors is not None else observed)
        g=torch.autograd.grad(loss,params,allow_unused=True)
        grads.append([v.detach().cpu() if v is not None else None for v in g]);weights.append(len(rr))
        loss_values.append(float(loss.detach()));active_ids.append(cid)
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
        pairs[f"{active_ids[i]}:{active_ids[j]}"]=dot/math.sqrt(ni*nj) if ni and nj else None
    return {"sampled_weighted_current_objective_gradient_norm_squared":norm,"pairwise_backbone_cosine":pairs,
            "sampled_weighted_objective_loss":sum(w*l for w,l in zip(weights,loss_values))/total,
            "data_objective":"current_plus_class_uniform_replay" if replay is not None else "current_only",
            "split":"train","objective":"local_competitor_union" if competitors is not None else "common_global","max_examples_per_client":limit,"bn_mode":"eval","no_feedback":True}


def historical_gradient(model,state,dataset,heads,setting,observed,device,limit=4):
    """Fixed validation historical panel, no influence on training/selection."""
    if not heads:return {}
    by=defaultdict(list)
    for i,y in enumerate(dataset.targets):by[int(y)].append(i)
    if setting=="class_il":
        rr=[(i,next(h for h in sorted(heads) if c in head_labels(h)))
            for c in sorted({c for h in heads for c in head_labels(h)}) for i in by[c][:limit]]
    else:
        rr=[(i,h) for h in sorted(heads) for c in head_labels(h) for i in by[c][:limit]]
    if not rr:return {}
    model.load_state_dict(state);model.eval()
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    result={}
    for start in range(0,len(rr),64):
        rows=rr[start:start+64]
        x,y,h=load_records(dataset,rows,device)
        loss=classification_loss(model,x,y,h,setting,observed)*len(rows)/len(rr)
        grads=torch.autograd.grad(loss,[p for _,p in params],allow_unused=True)
        for (n,_),g in zip(params,grads):
            if g is not None:
                v=g.detach().cpu()
                if n in result:result[n].add_(v)
                else:result[n]=v.clone()
    return result


def weighted_parameter_branch(base,weighted_candidates,param_names):
    """Linear parameter-only branch with fixed base buffers."""
    return {k:(v+sum((w*(s[k]-v) for w,s in weighted_candidates),torch.zeros_like(v))
               if k in param_names else v) for k,v in base.items()}


def run_reference(job,output,bank,device="cpu",model_name="resnet18_cifar32",subset_probes=False,resume_from=None):
    if job.get("binding") not in ("matched_reference_v1","historical_reconstruction_v1"):
        raise ValueError("No supported reference binding")
    reconstructed=job["binding"]=="historical_reconstruction_v1"
    intervention=job.get("intervention","none")
    beta=0. if intervention=="no_damping_beta0" else 2. if intervention=="strong_damping_beta2" else 1.
    fresh_only=intervention=="fresh_only"
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    seed=job["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    stages=(compile_reconstructed if reconstructed else compile_matched)(job,bank.train.targets,bank.num_classes)
    all_heads={tuple(u["head"]) for s in stages for us in s["units"] for u in us}
    setting=job["il"]
    model=create_model(model_name,bank.num_classes,allow_test_model=model_name.startswith("test_"),
        task_class_sets={model_head(h) if isinstance(model_head(h),str) else "task_"+"_".join(f"{c:03d}" for c in head_labels(h)):head_labels(h) for h in all_heads} if setting=="task_il" else None,learning_setting=setting).to(device)
    param_names={n for n,p in model.named_parameters()}
    server=copy_cpu_state(model);version=0
    clients=[{"state":server,"anchor":server,"version":0,"optimizer":None,"memory":[],"shared":[],"pending":{},"known":set()}
             for _ in range(job["clients"])]
    archive=[];prior=set();observed=set();integrated=set();peak={};rows=[];all_counts=defaultdict(int)
    audit=plan_audit(job,stages,bank.train.targets)
    atomic_json(output/"partition.json",{**audit,"split":data_partition_manifest(bank,"cifar100" if bank.num_classes==100 else "synthetic",.1,1.)})
    atomic_json(output/"schedule.json",stages)
    diagnostics=output/"per_round_metrics";diagnostics.mkdir(exist_ok=True)
    checkpoints=output/"checkpoints";checkpoints.mkdir(exist_ok=True)
    budget=job.get("replay_budget_per_class",50)
    def cap(items):
        return capped_memory(items,bank.train.targets,seed,budget,by_occurrence=reconstructed)
    def own_classes(us):
        return {c for u in us for c in head_labels(u["head"])}
    def competitors(c,us):
        if reconstructed and setting=="class_il" and job["objective"]=="original":
            return c["known"]|own_classes(us)
        return observed
    ctr=0
    first_stage=0
    test=None
    from .identity import runtime_identity,dataset_identity
    from .joint.checkpoint_io import durable_torch_save
    from .diagnostic_campaign import signature,sha_file
    import shutil
    identity={'job_signature':signature(job),'runtime':runtime_identity(),
              'dataset':dataset_identity(bank,allow_tensor_fixture=model_name.startswith('test_')),'allocation_sha256':audit['allocation_sha256'],
              'model':model_name,'subset_probes':bool(subset_probes),
              'device_type':torch.device(device).type}
    def save_resume(next_stage):
        artifacts={str(p.relative_to(output)):sha_file(p)
                   for p in sorted(diagnostics.glob('sync_*.json'))}
        artifacts.update({str(p.relative_to(output)):sha_file(p)
                          for p in sorted(output.glob('stage_*_evaluation.json'))})
        fork=checkpoints/'fork_reference.pt'
        if fork.is_file():artifacts[str(fork.relative_to(output))]=sha_file(fork)
        durable_torch_save({'identity':identity,'next_stage':next_stage,
            'server':server,'version':version,'clients':clients,'archive':archive,
            'prior':prior,'observed':observed,'integrated':integrated,'peak':peak,
            'rows':rows,'all_counts':dict(all_counts),'ctr':ctr,'last_test':test,
            'artifacts':artifacts,'python_rng':random.getstate(),
            'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
            'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
            checkpoints/'resume.pt')
    if resume_from is not None:
        source=Path(resume_from)
        saved=torch.load(source,map_location='cpu',weights_only=False)
        if saved['identity']!=identity:raise RuntimeError('Matched checkpoint identity mismatch')
        first_stage=int(saved['next_stage'])
        if not 1<=first_stage<=len(stages):raise ValueError('Invalid matched checkpoint stage')
        server=saved['server'];version=saved['version'];clients=saved['clients']
        archive=saved['archive'];prior=saved['prior'];observed=saved['observed']
        integrated=saved['integrated'];peak=saved['peak'];rows=saved['rows']
        all_counts=defaultdict(int,saved['all_counts']);ctr=int(saved['ctr']);test=saved['last_test']
        if ctr!=len(rows) or [x['sync'] for x in rows]!=list(range(1,ctr+1)):
            raise ValueError('Matched checkpoint history mismatch')
        for name,expected in saved['artifacts'].items():
            src=source.parent.parent/name;dest=output/name
            if not src.is_file() or sha_file(src)!=expected:
                raise ValueError('Missing or corrupt checkpoint evidence: '+name)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if src.resolve()!=dest.resolve():shutil.copy2(src,dest)
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng']:torch.cuda.set_rng_state_all(saved['cuda_rng'])
        durable_torch_save(saved,checkpoints/'resume.pt')
        durable_torch_save({'server':server,'version':version,'stage':first_stage-1},checkpoints/'latest.pt')
    def eval_panel(state,heads,limit=4):
        return evaluate(model,state,bank.validation,heads,setting,observed,device,limit)
    def commit(cohort,stage,current,flush=False,train_counts=None,local=None,gradient=None):
        nonlocal server,version,archive,ctr,integrated
        ctr+=1
        base=server
        history_keys=targets(prior,setting);current_keys=targets(current,setting)
        before=eval_panel(base,prior|current)
        updated,uploads=aggregate(base,clients,cohort,version,param_names,beta,fresh_only)
        if intervention=="bn_recalibration" and uploads:
            received=[i for u in uploads for i in clients[u["client"]]["pending"]]
            updated=recalibrate_bn(model,updated,bank.train,received+[i for i,_ in archive],device,stable_seed(seed,ctr))
        after=eval_panel(updated,prior|current)
        hist_before=macro(before,history_keys);hist_after=macro(after,history_keys)
        current_before=macro(before,current_keys);current_after=macro(after,current_keys)
        g_history=historical_gradient(model,base,bank.validation,prior,setting,observed,device)
        candidate_uploads=[{"client_id":u["client"],"examples":u["unique_pending_samples"],
            "weights":candidate(base,clients[u["client"]]["state"],clients[u["client"]]["anchor"],u["lag"],param_names,beta)} for u in uploads]
        geometry=update_geometry(model,base,candidate_uploads) if candidate_uploads else None
        history_alignment=alignment(g_history,base,updated,sorted(param_names)) if g_history else None
        individual_alignment={str(u["client_id"]):alignment(g_history,base,u["weights"],sorted(param_names)) for u in candidate_uploads} if g_history else {}
        subsets=[];interactions=[]
        if subset_probes and len(clients)<=5 and not flush:
            for n in range(1,len(clients)+1):
                for ids in combinations(range(len(clients)),n):
                    alt,alt_uploads=aggregate(base,clients,ids,version,param_names,beta,fresh_only)
                    if intervention=="bn_recalibration" and alt_uploads:
                        received=[i for u in alt_uploads for i in clients[u["client"]]["pending"]]
                        alt=recalibrate_bn(model,alt,bank.train,received+[i for i,_ in archive],device,stable_seed(seed,ctr))
                    ev=eval_panel(alt,prior|current)
                    subsets.append({"cohort":list(ids),"history_accuracy":macro(ev,history_keys),
                        "current_accuracy":macro(ev,current_keys),"history_loss":macro(ev,history_keys,"loss"),
                        "history_gain":difference(macro(ev,history_keys),hist_before),"current_gain":difference(macro(ev,current_keys),current_before)})
            if g_history:
                for u,v in combinations(candidate_uploads,2):
                    total=u["examples"]+v["examples"]
                    wi,wj=u["examples"]/total,v["examples"]/total
                    a,b=weighted_parameter_branch(base,[(wi,u["weights"])],param_names),weighted_parameter_branch(base,[(wj,v["weights"])],param_names)
                    both=weighted_parameter_branch(base,[(wi,u["weights"]),(wj,v["weights"])],param_names)
                    losses=[macro(eval_panel(st,prior),history_keys,"loss") for st in (a,b,both)]
                    baseline=macro(before,history_keys,"loss")
                    interactions.append({"pair":[u["client_id"],v["client_id"]],"weights":[wi,wj],
                        "historical_loss_nonadditivity":losses[2]-losses[0]-losses[1]+baseline,"buffers":"fixed_base"})
        server=updated
        if uploads:version+=1
        actual=[u["client"] for u in uploads]
        for c in actual:
            archive=cap(archive+list(clients[c]["pending"].values()))
            integrated.update(int(bank.train.targets[i]) for i in clients[c]["pending"])
        sync=[]
        rejected=[c for c in cohort if fresh_only and clients[c]["pending"] and c not in actual]
        for c in actual+rejected:
            state=clients[c]
            pre=eval_panel(state["state"],prior) if prior else {}
            sync.append({"client":c,"download_history_drop":difference(macro(pre,history_keys),hist_after)})
            state.update(state=server,anchor=server,version=version,shared=list(archive),pending={})
            state["known"].update(integrated)
            if intervention=="reset_momentum":state["optimizer"]=None
        row={"sync":ctr,"stage":stage,"flush":flush,"uploaders":actual,
             "history_before":hist_before,"history_after":hist_after,
             "aggregation_forgetting":difference(hist_before,hist_after),
             "current_before":current_before,"current_after":current_after,
             "history_loss_before":macro(before,history_keys,"loss"),"history_loss_after":macro(after,history_keys,"loss"),
             "current_loss_before":macro(before,current_keys,"loss"),"current_loss_after":macro(after,current_keys,"loss"),
             "shared_replay_size":len(archive),"uploads":uploads,"rejected_stale_clients":rejected,
             "integrated_classes":sorted(integrated),"pending_classes":sorted(observed-integrated),"local":local or [],
             "download":sync,"counts":train_counts or {},"gradient":gradient,"counterfactual_subsets":subsets,
             "update_geometry":geometry,"historical_gradient_alignment":history_alignment,
             "individual_historical_alignment":individual_alignment,"pair_interactions":interactions,
             "diagnostic_split":"validation","panel_examples_per_class":4}
        atomic_json(diagnostics/f"sync_{ctr:04d}.json",row)
        return row
    for s in stages[first_stage:]:
        t=s["stage"];current={tuple(u["head"]) for us in s["units"] for u in us}
        observed.update(s["classes"])
        # Freeze replay at parent-stage start: compute matched arms process
        # identical current and replay draws under different sync frequency.
        frozen=[cap(c["memory"]+c["shared"]) for c in clients]
        for rr in range(job["rounds_per_stage"]):
            local=[];counts=defaultdict(int)
            if intervention=="download_before_new_training":
                for cid in s["cohort"]:
                    c=clients[cid]
                    c.update(state=server,anchor=server,version=version,shared=list(archive),pending={})
                    c["known"].update(integrated)
            if reconstructed:
                # With additional communications, use new archive receipts on
                # subsequent rounds; the original single-round case is unchanged.
                # Historical replay contains completed prior stages only.
                frozen=[cap([(i,h) for i,h in c["memory"]+c["shared"] if tuple(h) not in current]) for c in clients]
            competitor_sets=[competitors(c,us) for c,us in zip(clients,s["units"])]
            gradient=grad_probe(model,server,bank,s["units"],setting,observed,device,
                competitors=competitor_sets if reconstructed and setting=="class_il" and job["objective"]=="original" else None)
            gradient_replay=grad_probe(model,server,bank,s["units"],setting,observed,device,
                competitors=competitor_sets if reconstructed and setting=="class_il" and job["objective"]=="original" else None,replay=frozen)
            for cid,us in enumerate(s["units"]):
                if not us:continue
                c=clients[cid];hist_keys=targets(prior,setting)
                own_heads={tuple(u["head"]) for u in us}
                own_keys=targets(own_heads,setting)
                pre=eval_panel(c["state"],prior|own_heads)
                new,opt,cc=train_block(model,c["state"],c["optimizer"],us,frozen[cid],bank,job,t,cid,rr,competitor_sets[cid],device)
                post=eval_panel(new,prior|own_heads)
                local.append({"client":cid,"history_before":macro(pre,hist_keys),"history_after":macro(post,hist_keys),
                              "local_forgetting":difference(macro(pre,hist_keys),macro(post,hist_keys)),
                              "current_before":macro(pre,own_keys),"current_after":macro(post,own_keys),
                              "classification_competitor_classes":sorted(competitor_sets[cid])})
                c.update(state=new,optimizer=opt)
                c["known"].update(own_classes(us))
                c["pending"].update({i:(i,h) for i,h in records(us)})
                for key,v in cc.items():counts[key]+=v;all_counts[key]+=v
            if rr==job["rounds_per_stage"]-1:
                for cid,us in enumerate(s["units"]):
                    c=clients[cid]
                    c["memory"]=cap(c["memory"]+records(us))
            row=commit(s["cohort"],t,current,train_counts=dict(counts),local=local,gradient=gradient)
            row["round_in_stage"]=rr+1
            row["pre_current_replay_gradient"]=gradient_replay
            row["post_current_gradient"]=grad_probe(model,server,bank,s["units"],setting,observed,device,
                competitors=competitor_sets if reconstructed and setting=="class_il" and job["objective"]=="original" else None)
            row["post_current_replay_gradient"]=grad_probe(model,server,bank,s["units"],setting,observed,device,
                competitors=competitor_sets if reconstructed and setting=="class_il" and job["objective"]=="original" else None,replay=frozen)
            atomic_json(diagnostics/f"sync_{ctr:04d}.json",row)
            rows.append(row)
        # Official test is diagnostic only. Fixed historical keys distinguish
        # forgetting from the newly expanded evaluation denominator.
        test=evaluate(model,server,bank.test,prior|current,setting,observed,device)
        val=evaluate(model,server,bank.validation,prior|current,setting,observed,device)
        keys=targets(prior,setting)
        forgetting=[max(0.,peak[k]-test[k]["accuracy"]) for k in keys if k in peak and k in test]
        for k,a in test.items():peak[k]=max(peak.get(k,0.),a["accuracy"])
        task_slots=[macro(test,targets({tuple(u["head"])},setting)) for us in s["units"] for u in us]
        task_slots=[x for x in task_slots if x is not None]
        rows[-1].update(test_current=sum(task_slots)/len(task_slots),test_history=macro(test,keys),
                       test_global=macro(test,set(test)),test_forgetting=sum(forgetting)/len(forgetting) if forgetting else None)
        legacy=evaluate_union(model,server,bank.test,prior|current,device) if reconstructed and setting=="task_il" else None
        if legacy is not None:rows[-1]["legacy_union_accuracy"]=macro(legacy,set(legacy))
        atomic_json(output/f"stage_{t:02d}_evaluation.json",{"stage":t,"test":test,"validation":val,"current":sorted(targets(current,setting)),"historical":sorted(keys),"class_membership_union_test":legacy})
        atomic_json(diagnostics/f"sync_{ctr:04d}.json",rows[-1])
        torch.save({"server":server,"version":version,"stage":t},checkpoints/"latest.pt")
        if (job["family"]=="MC" and job["clients"]==2 and job["partition"]=="iid"
                and job["rounds_per_stage"]==1 and t==4):
            # A common learned checkpoint and identical replay IDs for all
            # subsequent short client-count forgetting forks.
            torch.save({"server":server,"version":version,"stage":t,
                        "prior_heads":sorted(prior|current),"observed":sorted(observed),
                        "archive":archive,"job_id":job["job_id"],"allocation_sha256":audit["allocation_sha256"]},
                       checkpoints/"fork_reference.pt")
        prior.update(current)
        save_resume(t+1)
    if job["participation"]<1 or reconstructed:
        row=commit(list(range(len(clients))),len(stages),set(),flush=True)
        final=evaluate(model,server,bank.test,prior,setting,observed,device)
        row["test_global"]=macro(final,set(final))
        if reconstructed and setting=="task_il":
            union=evaluate_union(model,server,bank.test,prior,device)
            row["legacy_union_accuracy"]=macro(union,set(union))
            atomic_json(output/"terminal_union_evaluation.json",union)
        row["test_forgetting"]=sum(max(0.,peak[k]-v["accuracy"]) for k,v in final.items())/len(final)
        rows.append(row);atomic_json(output/"terminal_evaluation.json",final)
        atomic_json(diagnostics/f"sync_{ctr:04d}.json",row)
        torch.save({"server":server,"version":version,"stage":len(stages)},checkpoints/"latest.pt")
    else:
        final=test
    # Stable scalar round table; detailed panels in individual JSON files.
    fields=["sync","stage","round_in_stage","flush","history_before","history_after","aggregation_forgetting",
            "current_before","current_after","test_current","test_history","test_global","test_forgetting","legacy_union_accuracy","shared_replay_size"]
    with (output/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    end_stages=[r for r in rows if "test_current" in r]
    values=[r["test_current"] for r in end_stages if r["test_current"] is not None]
    atomic_json(output/"result_summary.json",{"job_id":job["job_id"],"protocol":job["binding"],"historical_reproduction_verified":False if reconstructed else None,
        "learning_syncs":sum(not r["flush"] for r in rows),"terminal_flush":job["participation"]<1 or reconstructed,
        "final_accuracy":macro(final,set(final)),"legacy_union_accuracy":rows[-1].get("legacy_union_accuracy"),"current_task_average_accuracy":sum(values)/len(values),
        "final_forgetting":rows[-1].get("test_forgetting"),"counts":dict(all_counts),
        "allocation_sha256":audit["allocation_sha256"],"replay_budget_per_class":budget,
        "evaluation":"global_seen_class or independent_semantic_task_head",
        "aggregation":"sample_weighted_full_model; stale parameter delta/(1+lag); stale buffers unchanged",
        "optimizer":"persistent SGD except explicit reset intervention", "augmentation":"crop_flip_reference" if reconstructed else "none", "subset_probes":subset_probes,
        "memory_budget_scope":"per_class_occurrence" if reconstructed else "per_class","intervention":intervention,"stale_beta":beta})
    return rows


def crop_flip(x,seed):
    rng=random.Random(seed)
    padded=F.pad(x,(4,4,4,4))
    result=[]
    for i in range(len(x)):
        a,b=rng.randrange(9),rng.randrange(9)
        z=padded[i,:,a:a+32,b:b+32]
        result.append(z.flip(-1) if rng.random()<.5 else z)
    return torch.stack(result)


@torch.no_grad()
def recalibrate_bn(model,state,dataset,indices,device,seed):
    """Only TRAIN examples accepted from uploaders or shared replay."""
    model.load_state_dict(state)
    layers=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
    if not layers or not indices:
        return state
    ids=sorted(set(indices))
    ids=random.Random(seed).sample(ids,min(512,len(ids)))
    model.eval()
    for m in layers:
        m.reset_running_stats()
        m.train()
    for x,_ in DataLoader(Subset(dataset,ids),batch_size=64,shuffle=False):
        model.features(x.to(device))
    return copy_cpu_state(model)


@torch.no_grad()
def evaluate_union(model,state,dataset,heads,device):
    """Class-membership oracle union, separate from true task-ID accuracy.

    This implements the documented duplicate-class probability rule, not
    a claim of bitwise historical evaluator equivalence.
    """
    model.load_state_dict(state);model.eval()
    by_class=defaultdict(list)
    for h in sorted(heads):
        for c in head_labels(h):by_class[c].append(h)
    ids=[i for i,y in enumerate(dataset.targets) if int(y) in by_class]
    accum=defaultdict(lambda:[0,0,0.])
    for x,y in DataLoader(Subset(dataset,ids),batch_size=64,shuffle=False):
        y=y.to(device);features=model.features(x.to(device))
        for label in y.unique().tolist():
            mask=y==label
            group=by_class[label]
            scores=torch.zeros((int(mask.sum()),model.num_classes),device=device)
            count=torch.zeros(model.num_classes,device=device)
            for h in group:
                c=torch.tensor(head_labels(h),device=device)
                scores[:,c]+=model.classifier(features[mask],model_head(h)).softmax(1)
                count[c]+=1
            present=count>0
            scores[:,present]/=count[present]
            scores[:,present]/=scores[:,present].sum(1,keepdim=True)
            a=accum[str(label)];a[0]+=int((scores.argmax(1)==label).sum());a[1]+=int(mask.sum())
            a[2]+=float(-scores[:,label].clamp_min(1e-12).log().sum())
    return {k:{"accuracy":a[0]/a[1],"loss":a[2]/a[1],"correct":a[0],"total":a[1]} for k,a in accum.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job",required=True);p.add_argument("--output",required=True)
    p.add_argument("--data-root",required=True);p.add_argument("--device",default="cuda:0")
    p.add_argument("--download",action="store_true");p.add_argument("--subset-probes",action="store_true")
    p.add_argument("--resume-from")
    a=p.parse_args()
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    bank=get_data_bank("cifar100",a.data_root,download=a.download,seed=0)
    run_reference(json.loads(Path(a.job).read_text()),a.output,bank,a.device,subset_probes=a.subset_probes,resume_from=a.resume_from)


if __name__=="__main__":
    main()
