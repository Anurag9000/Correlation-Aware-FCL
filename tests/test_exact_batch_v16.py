"""Assert that training and diagnostic forwards use physical batches of 64."""
import torch
import pytest
from cafcl.config import RunConfig
from cafcl.continual import LocalState,local_train,_FixedBatchSampler,_reference_gradient
from cafcl.models import create_model
from cafcl.evaluation import evaluate

class Bank:
    num_classes=4
    class Train:
        targets=[0,0,0,1,1,1,2,2,2,3,3,3]
        def __len__(self):return 12
        def __getitem__(self,i):
            return torch.ones(3,32,32)*(.1+.07*i),self.targets[i]
    train=Train()
    test=Train()
    validation=Train()

def test_full_batch_sampler_never_drops_unique_current_examples():
    groups=list(_FixedBatchSampler(70,64,12,True))
    assert len(groups)==2 and all(len(group)==64 for group in groups)
    assert set(range(70)).issubset(set(groups[0]+groups[1]))

def test_replay_reads_every_stored_example_for_each_update_in_batch64_microbatches():
    bank=Bank()
    class TrackingTrain:
        targets=Bank.Train.targets
        def __init__(self):self.reads=[]
        def __len__(self):return 12
        def __getitem__(self,i):
            self.reads.append(int(i))
            return torch.ones(3,32,32)*(.1+.07*i),self.targets[i]
    bank.train=TrackingTrain()
    model=create_model('test_mlp',4,allow_test_model=True,task_class_sets=((0,1),(2,3)))
    client=LocalState(0)
    client.receive(model.state_dict(),0)
    client.memory={0:[0],1:[3]}
    client.private_memory={0:[0],1:[3]}
    client.first_seen_local_experience={0:0,1:0}
    client.observed_classes={0,1}
    client.local_experience=1
    client.arrival([6,7,8,9,10,11],[2,3],stage=1)
    shapes=[]
    handle=model.features.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
    try:
        cfg=RunConfig(dataset='synthetic',model='test_mlp',num_clients=2,
            method='replay',replay_update_policy='exhaustive',batch_size=64,epochs=1,memory_per_class=2,lr=.01)
        result=local_train(model,client,bank,cfg,torch.device('cpu'),1,
            task_classes=(2,3),task_id=(2,3),
            known_task_ids=((0,1),(2,3)),
            task_ids_by_index={**{0:((0,1),),3:((0,1),)},
                               **{i:((2,3),) for i in range(6,12)}},
            experience_end=True,active_pair_classes=(2,3))
    finally:
        handle.remove()
    assert result['steps']==1
    assert shapes and all(n==64 for n in shapes)
    # Both incoming batches are training steps; the complete two-item archive
    # is read once for each step, with padding performed after data retrieval.
    assert bank.train.reads.count(0)==result['steps']
    assert bank.train.reads.count(3)==result['steps']


def test_replay_accumulates_archive_larger_than_one_microbatch_once():
    class WideTrain:
        targets=[0]*35+[1]*35+[2]*3+[3]*3
        def __init__(self):self.reads=[]
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            self.reads.append(int(i))
            return torch.ones(3,32,32)*(.01*i),self.targets[i]
    class WideBank:
        num_classes=4
        def __init__(self):self.train=WideTrain()
    bank=WideBank();model=create_model('test_mlp',4,allow_test_model=True,task_class_sets=((0,1),(2,3)))
    client=LocalState(0);client.receive(model.state_dict(),0)
    client.memory={0:list(range(35)),1:list(range(35,70))}
    client.private_memory={k:list(v) for k,v in client.memory.items()}
    client.observed_classes={0,1}
    client.first_seen_local_experience={0:0,1:0}
    client.local_experience=1
    client.arrival(list(range(70,76)),[2,3],1)
    shapes=[]
    handle=model.features.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
    try:
        cfg=RunConfig(dataset='synthetic',model='test_mlp',num_clients=2,
            method='replay',replay_update_policy='exhaustive',batch_size=64,epochs=1,memory_per_class=40,lr=.01)
        result=local_train(model,client,bank,cfg,torch.device('cpu'),1,
            task_classes=(2,3),task_id=(2,3),
            known_task_ids=((0,1),(2,3)),
            task_ids_by_index={**{i:((0,1),) for i in range(70)},
                               **{i:((2,3),) for i in range(70,76)}},
            experience_end=False,active_pair_classes=(2,3))
    finally:
        handle.remove()
    assert result['steps']==1
    assert all(bank.train.reads.count(i)==result['steps'] for i in range(70))
    assert shapes and all(size==64 for size in shapes)


@pytest.mark.parametrize('method',('replay','er_ace','der','derpp','gem','agem'))
def test_every_replay_capable_local_method_reads_each_stored_example_per_update(method):
    from collections import Counter
    class TrackingTrain:
        targets=[0,0,1,1,2,2,3,3]
        def __init__(self):self.reads=Counter()
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):
            self.reads[int(i)]+=1
            return torch.ones(3,32,32)*(.05*int(i)),self.targets[i]
    class TrackingBank:
        num_classes=4
        def __init__(self):self.train=TrackingTrain()
    bank=TrackingBank();model=create_model('test_mlp',4,allow_test_model=True,task_class_sets=((0,1),(2,3)))
    client=LocalState(0);client.receive(model.state_dict(),0)
    client.memory={0:[0,1],1:[2,3]}
    client.private_memory={0:[0,1],1:[2,3]}
    client.observed_classes={0,1}
    client.first_seen_local_experience={0:0,1:0}
    client.local_experience=1
    client.arrival([4,5,6,7],[2,3],1)
    cfg=RunConfig(dataset='synthetic',model='test_mlp',num_clients=2,
        method=method,replay_update_policy='exhaustive',batch_size=64,epochs=2,lr=.01,memory_per_class=2,
        replay_ratio=1.)
    result=local_train(model,client,bank,cfg,torch.device('cpu'),1,
        task_classes=(2,3),task_id=(2,3),known_task_ids=((0,1),(2,3)),
        task_ids_by_index={**{i:((0,1),) for i in range(4)},
                           **{i:((2,3),) for i in range(4,8)}},
        experience_end=False,active_pair_classes=(2,3))
    assert result['steps']==2
    assert all(bank.train.reads[i]==result['steps'] for i in (0,1,2,3))

def test_reference_gradient_requires_one_full_physical_forward():
    model=create_model('test_mlp',4,allow_test_model=True,task_class_sets=((0,1),))
    shapes=[]
    handle=model.features.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
    try:
        grad=_reference_gradient(model,list(model.parameters()),Bank.train,[0,3],64,
            torch.device('cpu'),12,reference_classes=(0,1),task_ids=((0,1),))
    finally:
        handle.remove()
    assert shapes==[64] and torch.isfinite(grad).all()

def test_eval_padding_never_changes_real_sample_denominator_or_accuracy():
    class AlwaysZero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bias=torch.nn.Parameter(torch.tensor([3.,0.,0.,0.]))
        def forward(self,x):
            assert len(x)==64
            return self.bias.unsqueeze(0).expand(len(x),-1)
    metrics,_=evaluate(AlwaysZero(),Bank(),{0,1,2,3},64,torch.device('cpu'))
    assert metrics['evaluation_samples']==12
    assert metrics['accuracy']==3/12
