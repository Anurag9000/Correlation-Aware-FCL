"""Controlled deterministic receiver-domain image shifts.

Source/archive pixels remain canonical. A client transforms current and
replayed raw images in its active domain at each local experience. This is
a receiver-domain replay control, not preservation of source-domain pixels.
"""
from dataclasses import replace
import torch
from torch.utils.data import Dataset
from .curriculum import stable_seed


def domain_transform(x,client,experience,mode,index=0):
    if mode=='none':return x
    if mode=='client_corruption':
        generator=torch.Generator().manual_seed(stable_seed('domain_noise',client,index)%(2**63))
        noise=torch.randn(x.shape,generator=generator,dtype=x.dtype).to(x.device)
        return x+noise*(.05+.05*(client%5))
    if mode not in ('client_fixed','client_periodic'):raise ValueError(mode)
    domain=client%5 if mode=='client_fixed' else (client+experience//5)%5
    if domain==0:return x
    if domain==1:return x.flip(-1)
    if domain==2:return x.roll(1,dims=-3)
    if domain==3:return x*.65+.2
    return x*.85-.25


class DomainView(Dataset):
    def __init__(self,base,client,experience,mode):
        self.base=base;self.client=client;self.experience=experience;self.mode=mode
        self.targets=base.targets
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        x,y=self.base[i]
        return domain_transform(x,self.client,self.experience,self.mode,int(i)),y


def terminal_domain_metrics(model,bank,cfg,clients,observed,task_class_map,device):
    from ..evaluation import evaluate
    from .task_evaluation import evaluate_task_conditioned
    labels=set(observed)
    groups=task_class_map if cfg.learning_setting=='task_il' else None
    records={}
    for cid,client in sorted(clients.items()):
        view=replace(bank,test=DomainView(bank.test,cid,client.local_experience,cfg.joint_domain_mode))
        result,_=evaluate(model,view,labels,cfg.batch_size,device,task_class_groups=groups,
                          split='test',loader_seed=0)
        if groups is not None:
            result['task_conditioned']=evaluate_task_conditioned(model,view,groups,cfg.batch_size,device,'test')
        records[cid]=result
    return {'domain_mode':cfg.joint_domain_mode,'replay_domain':'receiver_active_domain',
            'evaluation':'committed_global_model_on_all_observed_classes_in_each_terminal_client_domain',
            'test_only_no_feedback':True,'client_domains':records}
