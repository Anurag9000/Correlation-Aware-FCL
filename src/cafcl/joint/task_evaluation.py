"""Standard task-conditioned accuracy, separate from class-membership oracle."""
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, Subset


@torch.inference_mode()
def evaluate_task_conditioned(model,bank,groups,batch_size,device,split='validation'):
    """Evaluate each supplied head on examples in its own class set.

    Backbone features are computed once per held-out example, not once per
    head. The task ID is supplied by the evaluation episode; this is not
    a Class-IL prediction rule. No evaluation result feeds client selection.
    """
    if split not in ('validation','test'):
        raise ValueError(split)
    if not groups:
        return {'per_head':{},'macro_head_accuracy':None}
    dataset=getattr(bank,split)
    if dataset is None:raise ValueError('Missing evaluation split')
    by_class=defaultdict(list)
    for head,labels in groups.items():
        for c in labels:by_class[int(c)].append(head)
    ids=[i for i,c in enumerate(dataset.targets) if int(c) in by_class]
    loader=DataLoader(Subset(dataset,ids),batch_size=batch_size,shuffle=False,
                      num_workers=0,generator=torch.Generator().manual_seed(0))
    correct=defaultdict(int);total=defaultdict(int)
    class_correct=defaultdict(int);class_total=defaultdict(int)
    model.eval()
    for x,y in loader:
        y=y.to(device);features=model.features(x.to(device))
        active={head for c in y.unique().tolist() for head in by_class[int(c)]}
        for head in sorted(active):
            classes=torch.as_tensor(groups[head],device=device,dtype=torch.long)
            mask=(y[:,None]==classes[None,:]).any(dim=1)
            logits=model.classifier(features[mask],head)
            pred=classes[logits.argmax(dim=1)]
            correct[head]+=int((pred==y[mask]).sum())
            total[head]+=int(mask.sum())
            targets=y[mask]
            for c in targets.unique().tolist():
                subset=targets==c
                class_total[c]+=int(subset.sum())
                class_correct[c]+=int((pred[subset]==c).sum())
    records={head:{'accuracy':correct[head]/total[head] if total[head] else None,
                   'correct':correct[head],'total':total[head],
                   'classes':list(map(int,groups[head]))}
             for head in sorted(groups)}
    values=[x['accuracy'] for x in records.values() if x['accuracy'] is not None]
    return {'per_head':records,'macro_head_accuracy':sum(values)/len(values) if values else None,
            'per_class':{c:class_correct[c]/n for c,n in sorted(class_total.items()) if n},
            'micro_episode_accuracy':sum(correct.values())/sum(total.values()) if sum(total.values()) else None,
            'evaluation_split':split,'evaluation_protocol':'task_conditioned'}
