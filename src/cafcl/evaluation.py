"""All-observed global evaluation; never removes inactive clients' classes."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset

@torch.inference_mode()
def evaluate(model, bank, observed, batch_size,device,peak=None,*, integrated=None, old=None, new=None,
             task_class_groups=None, split='test', feat_ecc=None):
    if split not in {'validation','test'}:
        raise ValueError(f'Unsupported evaluation split: {split}')
    dataset=getattr(bank,split,None)
    if dataset is None:
        if split=='validation':
            raise ValueError('Validation evaluation requested without a validation split')
        dataset=bank.test
    observed=set(map(int,observed))
    if not observed:
        return {'accuracy':None,'macro_accuracy':None,'per_class':{},
                'confusion_matrix':{'labels':list(range(bank.num_classes)),
                                    'counts':[[0]*bank.num_classes for _ in range(bank.num_classes)]},
                'forgetting':0.,'unlearned':[],'evaluation_split':split,
                'evaluation_samples':0,'test_samples':0},peak or {}
    ids=[i for i,y in enumerate(dataset.targets) if int(y) in observed]
    test_samples=sum(1 for y in bank.test.targets if int(y) in observed)
    loader=DataLoader(Subset(dataset,ids),batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval()
    correct={c:0 for c in observed};count={c:0 for c in observed}
    class_losses={c:0. for c in observed}
    class_margins={c:0. for c in observed}
    confusion=torch.zeros((bank.num_classes,bank.num_classes),dtype=torch.int64)
    allowed_class_ids=torch.as_tensor(sorted(observed),dtype=torch.long,device=device)
    for x,y in loader:
        # Evaluation runs the identical physical batch of 64, too. The last
        # group is padded ONLY for forward computation; duplicated predictions
        # are discarded before loss, confusion-matrix and accuracy accounting.
        valid=len(y)
        evaluation_y=y
        if valid<batch_size:
            indices=torch.arange(batch_size-valid)%valid
            x=torch.cat((x,x[indices]),dim=0)
            evaluation_y=torch.cat((y,y[indices]),dim=0)
        if task_class_groups is None:
            if feat_ecc is None:
                logits=model(x.to(device))[:valid]
            else:
                from .feat_inference_correction import paper_reconstructed_feat_logits
                logits=paper_reconstructed_feat_logits(model,x.to(device),feat_ecc)[:valid]
            mask=torch.ones(logits.shape[1],dtype=torch.bool,device=device)
            mask[allowed_class_ids]=False
            logits=logits.masked_fill(mask.unsqueeze(0),-1e9)
            pred=logits.argmax(1).cpu()
            task_probabilities=None
        else:
            # Task IDs are client-task occurrences, not just class tuples. All
            # known occurrences that have trained a true class are queried.
            from .models import task_head_key
            if isinstance(task_class_groups,dict):
                task_map={task_head_key(task_id):tuple(map(int,classes))
                          for task_id,classes in task_class_groups.items()}
            else:
                task_map={task_head_key(group):tuple(sorted(set(map(int,group))))
                          for group in task_class_groups}
            task_ids=[tuple(task_id for task_id,classes in sorted(task_map.items())
                            if int(label) in set(classes))
                      for label in evaluation_y.tolist()]
            if any(not ids_for_sample for ids_for_sample in task_ids):
                raise ValueError('Task-IL evaluation encountered labels without a task')
            task_probabilities=model.task_union_scores(x.to(device),task_ids)[:valid]
            pred=task_probabilities.argmax(1).cpu()
            logits=task_probabilities.clamp_min(torch.finfo(task_probabilities.dtype).tiny).log()
            logits=torch.where(task_probabilities>=0,logits,torch.full_like(logits,-1.e9))
        confusion+=torch.bincount(
            y*bank.num_classes+pred,
            minlength=bank.num_classes*bank.num_classes).reshape(
                bank.num_classes,bank.num_classes)
        # Cross-entropy is measured under the same prediction universe as
        # the official accuracy: task IDs are given only for task-IL.
        if task_class_groups is None:
            per_sample_loss=F.cross_entropy(logits,y.to(device),reduction='none').cpu()
        else:
            per_sample_loss=-task_probabilities.gather(1,y.to(device).unsqueeze(1)).squeeze(1).clamp_min(
                torch.finfo(task_probabilities.dtype).tiny).log().cpu()
        true_logit=logits.gather(1,y.to(device).unsqueeze(1)).squeeze(1)
        competing=logits.clone()
        competing.scatter_(1,y.to(device).unsqueeze(1),-1.e9)
        margins=(true_logit-competing.max(dim=1).values).detach().cpu()
        for cls in y.unique().tolist():
            mask=y==cls
            correct[int(cls)]+=int((pred[mask]==y[mask]).sum())
            count[int(cls)]+=int(mask.sum())
            class_losses[int(cls)]+=float(per_sample_loss[mask].sum())
            class_margins[int(cls)]+=float(margins[mask].sum())
    per_class={int(c):correct[c]/count[c] if count[c] else 0. for c in sorted(observed)}
    micro=sum(correct.values())/max(1,sum(count.values()))
    micro_loss=sum(class_losses.values())/max(1,sum(count.values()))
    per_class_loss={c:class_losses[c]/max(1,count[c]) for c in sorted(observed)}
    per_class_margin={c:class_margins[c]/max(1,count[c]) for c in sorted(observed)}
    macro=float(np.mean(list(per_class.values())))
    historical=dict(peak or {})
    # All observed labels contribute to official global accuracy. Forgetting
    # only considers a class's PREVIOUS peak, never its current first score.
    prior={c:max(0.,historical.get(c,0.)-a) for c,a in per_class.items() if c in historical}
    # Distinguish decline on ALL observed classes from forgetting of knowledge
    # that had actually been learned after contributing to global training.
    observed_peak_drop=float(np.mean(list(prior.values()))) if prior else 0.
    integrated=set(map(int, integrated or ()))
    chance=1.0/bank.num_classes
    eligible={c:drop for c,drop in prior.items()
              if c in integrated and historical.get(c,0.) > chance+0.02}
    forgetting=float(np.mean(list(eligible.values()))) if eligible else 0.
    for c,a in per_class.items():historical[c]=max(a,historical.get(c,0.))
    # The official accuracy always includes every class that has arrived,
    # regardless of whether any selected client has trained on that class.
    never_integrated=sorted(observed-integrated)
    integrated_prior={c:v for c,v in prior.items() if c in integrated}
    old_observed=sorted(observed & set(old or ()))
    new_observed=sorted(observed & set(new or ()))
    group_accuracy=lambda group: (float(np.mean([per_class[c] for c in group])) if group else None)
    unlearned=[c for c in per_class if historical[c] <= chance + 0.02]
    return {'accuracy':micro,'macro_accuracy':macro,'per_class':per_class,
            'confusion_matrix':{'labels':list(range(bank.num_classes)),
                                'counts':confusion.tolist()},
            'cross_entropy_loss':micro_loss,'per_class_loss':per_class_loss,
            'per_class_true_logit_margin':per_class_margin,
            'forgetting':forgetting, 'all_observed_peak_drop':observed_peak_drop,
            'integrated_forgetting':float(np.mean(list(integrated_prior.values()))) if integrated_prior else 0.,
            'historically_learned_class_count':len(eligible),
            'old_accuracy':group_accuracy(old_observed),
            'new_accuracy':group_accuracy(new_observed),
            'old_classes':old_observed,'new_classes':new_observed,
            'globally_integrated_classes':sorted(integrated),
            'never_integrated_classes':never_integrated,
            'never_integrated_accuracy':group_accuracy(never_integrated),
            'observed_classes':sorted(observed),'test_samples':test_samples,
            'evaluation_samples':len(ids),'evaluation_split':split,
            'task_id_at_inference':task_class_groups is not None,
            'task_id_assignment':('schedule_class_membership_oracle'
                                  if task_class_groups is not None else 'none'),
            'unlearned':unlearned},historical
