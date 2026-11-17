"""All-observed global evaluation; never removes inactive clients' classes."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

@torch.inference_mode()
def evaluate(model, bank, observed, batch_size,device,peak=None,*, integrated=None, old=None, new=None):
    observed=set(map(int,observed))
    if not observed:return {'accuracy':None,'macro_accuracy':None,'per_class':{},'forgetting':0.,'unlearned':[]},peak or {}
    ids=[i for i,y in enumerate(bank.test.targets) if int(y) in observed]
    loader=DataLoader(Subset(bank.test,ids),batch_size=batch_size,shuffle=False,num_workers=0)
    model.eval()
    correct={c:0 for c in observed};count={c:0 for c in observed}
    for x,y in loader:
        pred=model(x.to(device)).argmax(1).cpu()
        for cls in y.unique().tolist():
            mask=y==cls
            correct[int(cls)]+=int((pred[mask]==y[mask]).sum())
            count[int(cls)]+=int(mask.sum())
    per_class={int(c):correct[c]/count[c] if count[c] else 0. for c in sorted(observed)}
    micro=sum(correct.values())/max(1,sum(count.values()))
    macro=float(np.mean(list(per_class.values())))
    historical=dict(peak or {})
    # All observed labels contribute to official global accuracy. Forgetting
    # only considers a class's PREVIOUS peak, never its current first score.
    prior={c:max(0.,historical.get(c,0.)-a) for c,a in per_class.items() if c in historical}
    forgetting=float(np.mean(list(prior.values()))) if prior else 0.
    for c,a in per_class.items():historical[c]=max(a,historical.get(c,0.))
    # The official accuracy always includes every class that has arrived,
    # regardless of whether any selected client has trained on that class.
    integrated=set(map(int, integrated or ()))
    never_integrated=sorted(observed-integrated)
    integrated_prior={c:v for c,v in prior.items() if c in integrated}
    old_observed=sorted(observed & set(old or ()))
    new_observed=sorted(observed & set(new or ()))
    group_accuracy=lambda group: (float(np.mean([per_class[c] for c in group])) if group else None)
    chance=1.0/bank.num_classes
    unlearned=[c for c in per_class if historical[c] <= chance + 0.02]
    return {'accuracy':micro,'macro_accuracy':macro,'per_class':per_class,
            'forgetting':forgetting,
            'integrated_forgetting':float(np.mean(list(integrated_prior.values()))) if integrated_prior else 0.,
            'old_accuracy':group_accuracy(old_observed),
            'new_accuracy':group_accuracy(new_observed),
            'old_classes':old_observed,'new_classes':new_observed,
            'globally_integrated_classes':sorted(integrated),
            'never_integrated_classes':never_integrated,
            'never_integrated_accuracy':group_accuracy(never_integrated),
            'observed_classes':sorted(observed),'test_samples':len(ids),
            'unlearned':unlearned},historical
