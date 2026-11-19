"""Assert that training and diagnostic forwards use physical batches of 64."""
import torch
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

def test_replay_even_with_two_old_samples_only_forwards_batch64():
    bank=Bank()
    model=create_model('test_mlp',4,allow_test_model=True)
    client=LocalState(0)
    client.receive(model.state_dict(),0)
    client.memory={0:[0],1:[3]}
    client.private_memory={0:[0],1:[3]}
    client.first_seen_local_experience={0:0,1:0}
    client.observed_classes={0,1}
    client.local_experience=1
    client.arrival([6,7,8,9,10,11],[2,3],stage=1)
    shapes=[]
    handle=model.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
    try:
        cfg=RunConfig(dataset='synthetic',model='test_mlp',num_clients=2,
            method='replay',batch_size=64,epochs=1,memory_per_class=2,lr=.01)
        result=local_train(model,client,bank,cfg,torch.device('cpu'),1,
            experience_end=True,active_pair_classes=(2,3))
    finally:
        handle.remove()
    assert result['steps']==2
    assert shapes and all(n==64 for n in shapes)

def test_reference_gradient_requires_one_full_physical_forward():
    model=create_model('test_mlp',4,allow_test_model=True)
    shapes=[]
    handle=model.register_forward_pre_hook(lambda _mod,args:shapes.append(len(args[0])))
    try:
        grad=_reference_gradient(model,list(model.parameters()),Bank.train,[0,3],64,
            torch.device('cpu'),12,reference_classes=(0,1))
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
