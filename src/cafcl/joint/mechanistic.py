"""Bounded validation-only gradient/representation probes on fixed panels.

The loss gradient is evaluated at the actual pre-aggregation server weights.
A positive g dot delta predicts increased loss to first order. Stale candidate
deltas are the rebased/damped updates, not raw client gradients. Historical
and current panels use the same competitor universe at both checkpoints.
"""
from collections import defaultdict
from itertools import combinations
import math
import random

import torch
from torch.nn import functional as F
from .curriculum import stable_seed


def pick_targets(values,limit,seed):
    values=sorted(set(values))
    return sorted(random.Random(seed).sample(values,min(limit,len(values))))


def build_panels(dataset,setting,observed,prior_classes,prior_heads,current,heads,
                 seed,interval,limit=8,samples=4):
    by_class=defaultdict(list)
    for i,y in enumerate(dataset.targets):by_class[int(y)].append(i)
    if setting=='task_il':
        current_keys={x['head_id'] for x in current}
        categories={'current':current_keys,'historical_noncurrent':set(prior_heads)-current_keys,
                    'recurrent':set(prior_heads)&current_keys,'new':current_keys-set(prior_heads)}
        labels=lambda k:heads[k]
    else:
        current_keys={c for x in current for c in x['classes']}
        categories={'current':current_keys,'historical_noncurrent':set(prior_classes)-current_keys,
                    'recurrent':set(prior_classes)&current_keys,'not_previously_integrated':current_keys-set(prior_classes)}
        labels=lambda k:(k,)
    panels={}
    for name,values in categories.items():
        targets=pick_targets(values,limit,stable_seed(seed,interval,name))
        records=[]
        for target in targets:
            # Stable sample choice per class across cohorts at this interval.
            ids=[]
            for c in labels(target):
                available=by_class[int(c)]
                ids.extend(sorted(random.Random(stable_seed(seed,interval,int(c),'probe')).sample(
                    available,min(samples,len(available)))))
            if ids:records.append((target,ids))
        panels[name]=records
    return panels


def panel_loss(model,dataset,records,setting,observed,device,need_grad):
    params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    grads={};stats={};losses=[]
    if not records:return None,{},{}
    for target,indices in records:
        # Chunk to keep wide tasks bounded. Use eval-mode BN, no padding.
        sum_loss=0.;count=0;sum_feature=None;rows=[]
        for start in range(0,len(indices),32):
            ids=indices[start:start+32]
            batch=[dataset[i] for i in ids]
            x=torch.stack([v[0] for v in batch]).to(device)
            y=torch.tensor([int(v[1]) for v in batch],device=device)
            with torch.set_grad_enabled(need_grad):
                features=model.features(x)
                if setting=='task_il':
                    allowed=torch.as_tensor(model.classifier.task_classes[target],device=device)
                    logits=model.classifier(features,target)
                else:
                    allowed=torch.as_tensor(sorted(observed),device=device)
                    logits=model.classifier(features).index_select(1,allowed)
                truth=(y[:,None]==allowed[None,:])
                if not truth.any(1).all():raise ValueError('Probe label outside competitor set')
                target_ids=truth.long().argmax(1)
                loss=F.cross_entropy(logits,target_ids,reduction='sum')
                if need_grad:
                    gs=torch.autograd.grad(loss/(len(indices)*len(records)),[p for _,p in params],allow_unused=True)
                    for (name,_),g in zip(params,gs):
                        if g is not None:
                            g=g.detach().cpu()
                            if name in grads:grads[name].add_(g)
                            else:grads[name]=g.clone()
                prob=logits.detach().softmax(1)
                pred=allowed[prob.argmax(1)]
                true_prob=prob.gather(1,target_ids[:,None]).squeeze(1)
                for i,label,predicted,p in zip(ids,y.tolist(),pred.tolist(),true_prob.tolist()):
                    rows.append({'validation_index':i,'label':label,'prediction':predicted,'true_probability':p})
                vec=features.detach().float().sum(0).cpu()
                sum_feature=vec if sum_feature is None else sum_feature+vec
                sum_loss+=float(loss.detach());count+=len(ids)
        losses.append(sum_loss/count)
        stats[str(target)]={'loss':sum_loss/count,'accuracy':sum(r['label']==r['prediction'] for r in rows)/count,
                           'samples':rows,'centroid':sum_feature/count}
    return sum(losses)/len(losses),grads,stats


def alignment(gradient,base,candidate,param_names):
    result={}
    for group in ('all','backbone','classifier'):
        dot=gn=dn=0.
        for n in param_names:
            if group!='all' and (n.startswith('classifier.')!=(group=='classifier')):continue
            a=base[n].detach().cpu().reshape(-1)
            b=candidate[n].detach().cpu().reshape(-1)
            g=gradient.get(n)
            for offset in range(0,a.numel(),131072):
                end=offset+131072
                d=b[offset:end].double()-a[offset:end].double()
                dn+=float(d@d)
                if g is not None:
                    v=g.reshape(-1)[offset:end].double()
                    dot+=float(v@d);gn+=float(v@v)
        result[group]={'first_order_loss_change':dot,'gradient_norm':math.sqrt(gn),'delta_norm':math.sqrt(dn),
                       'cosine':dot/math.sqrt(gn*dn) if gn and dn else None}
    return result


def gradient_cosine(a,b):
    dot=gn=hn=0.
    for name in set(a)|set(b):
        x=a.get(name);y=b.get(name)
        if x is not None:gn+=float(x.double().square().sum())
        if y is not None:hn+=float(y.double().square().sum())
        if x is not None and y is not None:dot+=float((x.double()*y.double()).sum())
    return dot/math.sqrt(gn*hn) if gn and hn else None


def feature_stats(a,b):
    x=a['centroid'].double();y=b['centroid'].double()
    norm=float(x.norm()*y.norm())
    return {'centroid_l2_drift':float((x-y).norm()),
            'centroid_cosine':float(x@y)/norm if norm else None,
            'before_norm':float(x.norm()),'after_norm':float(y.norm())}


def probe_aggregation(model,bank,device,base,committed,uploads,setting,observed,prior_classes,
                      prior_heads,current,heads,*,seed=0,interval=0,limit=8,samples=4):
    panels=build_panels(bank.validation,setting,observed,prior_classes,prior_heads,current,heads,
                        seed,interval,limit,samples)
    old_mode=model.training;records={};pair_grads={}
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    try:
        model.eval()
        for name,panel in panels.items():
            if not panel:continue
            model.load_state_dict(base)
            before,g,pre=panel_loss(model,bank.validation,panel,setting,observed,device,True)
            model.load_state_dict(committed)
            after,_,post=panel_loss(model,bank.validation,panel,setting,observed,device,False)
            records[name]={'pre_loss':before,'post_loss':after,'actual_loss_change':after-before,
                'committed_alignment':alignment(g,base,committed,names),
                'candidate_alignment':{str(u['client_id']):alignment(g,base,u['weights'],names) for u in uploads},
                'targets':{k:{'before':{a:v for a,v in pre[k].items() if a!='centroid'},
                              'after':{a:v for a,v in post[k].items() if a!='centroid'},
                              **feature_stats(pre[k],post[k])} for k in pre}}
            if name in ('current','historical_noncurrent'):pair_grads[name]=g
        cosine=None
        if len(pair_grads)==2:cosine=gradient_cosine(pair_grads['current'],pair_grads['historical_noncurrent'])
        return {'interval':interval,'split':'validation','no_feedback':True,'setting':setting,
                'competitor_classes':sorted(observed),'samples_per_class':samples,'target_limit':limit,
                'current_historical_gradient_cosine':cosine,'panels':records,
                'scope':'Fixed sampled validation panels; first-order loss predictions are not causal attribution.'}
    finally:
        model.load_state_dict(committed);model.train(old_mode)
