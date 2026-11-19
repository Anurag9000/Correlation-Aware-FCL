"""v12 paper-mechanism regression: each CL method under unchanged task stream."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from torch import nn

from cafcl.cl_estimators import estimate_diagonal_importance
from cafcl.continual import LocalState, local_train, gem_project, agem_project
from cafcl.config import RunConfig
from cafcl.models import create_model


class ScalarBank:
    class Train:
        targets=[0,1]
        xs=[1.,3.]
        def __len__(self):return 2
        def __getitem__(self,index):
            return torch.tensor([self.xs[index]],dtype=torch.float32),self.targets[index]
    train=Train()


def test_ewc_predictive_fisher_averages_per_example_squares():
    model=nn.Linear(1,2,bias=False)
    with torch.no_grad():model.weight.zero_()
    results=estimate_diagonal_importance(model,ScalarBank.train,[0,1],
        'cpu',kind='ewc',max_samples=2,seed=15,active_classes=(0,1))
    # Uniform predictive probabilities. Each class-row derivative magnitude
    # is |x|/2, so the mean square = (1²+3²)/2 /4 = 1.25.
    torch.testing.assert_close(results['weight'],torch.full((2,1),1.25))


def test_mas_averages_absolute_individual_output_sensitivities():
    model=nn.Linear(1,1,bias=False)
    with torch.no_grad():model.weight.fill_(2.)
    results=estimate_diagonal_importance(model,ScalarBank.train,[0,1],
        'cpu',kind='mas',max_samples=2,seed=9)
    # f(x)=2x; d/dw ||f(x)||²=2*w*x²=4*x²; average is (4+36)/2=20.
    torch.testing.assert_close(results['weight'],torch.tensor([[20.]]))


def test_ewc_mas_measurements_preserve_bn_stats_and_mode():
    model=nn.Sequential(nn.BatchNorm1d(1),nn.Linear(1,2))
    model.train()
    before=model[0].running_mean.clone(),model[0].running_var.clone()
    for kind in ('ewc','mas'):
        values=estimate_diagonal_importance(model,ScalarBank.train,[0,1],
            'cpu',kind=kind,max_samples=2,seed=42,active_classes=(0,1))
        assert model.training
        assert all(torch.isfinite(v).all() for v in values.values())
        torch.testing.assert_close(model[0].running_mean,before[0])
        torch.testing.assert_close(model[0].running_var,before[1])


def test_importance_rejects_validation_and_negative_budget():
    model=nn.Linear(1,2)
    with pytest.raises(ValueError):
        estimate_diagonal_importance(model,ScalarBank.train,[],'cpu',
            kind='ewc',max_samples=2,seed=1)
    with pytest.raises(ValueError):
        estimate_diagonal_importance(model,ScalarBank.train,[0,1],'cpu',
            kind='ewc',max_samples=0,seed=1)


@pytest.mark.parametrize('method',('gem','agem'))
def test_gradient_constraint_projection_matches_primal_halfspaces(method):
    g=torch.tensor([-1.,-1.])
    refs=[torch.tensor([1.,0.]),torch.tensor([0.,1.])]
    if method=='gem':
        z,n,v=gem_project(g,refs)
        assert v==2
        assert n>0
        assert all(torch.dot(z,r)>=-1.e-5 for r in refs)
        torch.testing.assert_close(z,torch.zeros_like(z),atol=2e-5,rtol=0)
    else:
        z,n,v=agem_project(g,refs[0])
        assert v==1
        assert n>0
        assert torch.dot(z,refs[0])>=-1.e-6
        torch.testing.assert_close(z,torch.tensor([0.,-1.]),atol=1e-5,rtol=0)


def test_nonfractional_memory_uses_true_classwise_reservoir_across_arrivals():
    labels=[0]*100+[1]*100
    first,second=LocalState(0),LocalState(1)
    a=np.random.default_rng(731);b=np.random.default_rng(731)
    first.update_memory(range(200),labels,10,a)
    for i in range(0,200,7):
        second.update_memory(range(i,min(i+7,200)),labels,10,b)
    assert first.private_memory==second.private_memory
    assert first.private_reservoir_seen_counts=={0:100,1:100}
    assert all(len(v)==10 for v in first.private_memory.values())
    first.update_memory(range(50),labels,10,a)
    assert first.private_reservoir_seen_counts=={0:100,1:100}


class ImageBank:
    class Train:
        targets=[0,0,1,1,2,2,3,3]
        def __len__(self):return 8
        def __getitem__(self,i):
            x=torch.zeros(3,32,32)
            x[self.targets[i]%3,self.targets[i],:]=0.5+i*.05
            return x,self.targets[i]
    train=Train()


def client_model():
    torch.manual_seed(103)
    model=create_model('test_linear',4,allow_test_model=True)
    state={name:value.detach().clone() for name,value in model.state_dict().items()}
    client=LocalState(0)
    client.receive(state,0)
    return model,client


def cfg(method,**kwargs):
    return replace(RunConfig(dataset='synthetic',model='test_linear',num_clients=2,
        method=method,aggregator='fedavg',epochs=1,lr=.03,
        batch_size=16,memory_per_class=2,
        regularization_strength=0.1,importance_samples_per_experience=4),
        **kwargs)


@pytest.mark.parametrize('method',('der','derpp'))
def test_der_remembers_preoptimizer_online_trajectory_logits(method):
    model,client=client_model()
    client.local_experience=0
    client.arrival([0,1,2,3],(0,1),0)
    initial={}
    model.eval()
    with torch.no_grad():
        for i in range(4):
            x,_=ImageBank.train[i]
            initial[i]=model(x.unsqueeze(0)).squeeze(0).clone()
    local_train(model,client,ImageBank(),cfg(method),torch.device('cpu'),
        0,experience_end=True,active_pair_classes=(0,1))
    assert set(client.private_memory)=={0,1}
    assert len(client.memory_logits)==4
    for i,z in client.memory_logits.items():
        torch.testing.assert_close(z,initial[i],atol=1e-5,rtol=1e-5)
    final=model.state_dict()
    assert any(not torch.equal(final[name],client.sync_anchor[name]) for name in final)
    assert all(len(v)<=2 for v in client.private_memory.values())


@pytest.mark.parametrize('method',('ewc','mas','si','lwf','gem','agem','replay','er_ace','der','derpp','sgd'))
def test_every_registered_method_stays_in_original_task_protocol(method):
    model,client=client_model()
    client.local_experience=0
    client.arrival([0,1,2,3],(0,1),0)
    result=local_train(model,client,ImageBank(),cfg(method),
        torch.device('cpu'),0,experience_end=True,active_pair_classes=(0,1))
    assert result['method_used']==method
    assert result['active_pair_classes']==[0,1]
    assert result['steps']==1
    # Old/future classifier rows are frozen even when a regularizer exists.
    rows=model.state_dict()['1.weight']
    torch.testing.assert_close(rows[2:],client.sync_anchor['1.weight'][2:],rtol=0,atol=0)
    assert torch.isfinite(rows).all()
    if method in ('ewc','mas'):
        assert method in client.pending_importance
        assert client.pending_importance[method]['precision']
    if method in ('gem','agem','replay','er_ace','der','derpp'):
        assert sum(map(len,client.private_memory.values()))>0
    else:
        assert not client.private_memory


def test_ewc_and_mas_per_experience_budget_in_config():
    assert RunConfig(dataset='synthetic').importance_samples_per_experience==32
    with pytest.raises(AssertionError):
        replace(RunConfig(dataset='synthetic'),
                importance_samples_per_experience=0).validate()


def test_gem_historical_reference_gradient_uses_its_original_pair_loss():
    from cafcl.continual import _reference_gradient
    import torch.nn.functional as F
    model=nn.Linear(1,4,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2],[-.5],[.8],[1.2]]))
    params=list(model.parameters())
    actual=_reference_gradient(model,params,ScalarBank.train,[0,1],2,
        torch.device('cpu'),13,active_pair_classes=None,reference_classes=(0,1))
    model.zero_grad()
    x=torch.tensor([[1.],[3.]])
    y=torch.tensor([0,1])
    F.cross_entropy(model(x)[:,[0,1]],y).backward()
    expected=torch.cat([p.grad.reshape(-1) for p in params])
    torch.testing.assert_close(actual,expected,atol=1e-7,rtol=1e-6)


def test_si_integral_uses_unregularized_task_gradient_even_with_old_importance():
    import torch.nn.functional as F
    model=create_model('test_mlp',4,allow_test_model=True)
    client=LocalState(0)
    client.receive({k:v.detach().clone() for k,v in model.state_dict().items()},0)
    conf=replace(cfg('si'),model='test_mlp',regularization_strength=20.,
                 lr=.07,si_xi=.1)
    client.local_experience=0
    client.arrival([0,1,2,3],(0,1),0)
    first=local_train(model,client,ImageBank(),conf,torch.device('cpu'),0,
                      experience_end=True,active_pair_classes=(0,1))
    client.shared_weights=first['weights']
    old_prec={n:v.clone() for n,v in client.pending_importance['si']['precision'].items()}
    client.local_experience=1
    client.arrival([4,5,6,7],(2,3),1)
    model.load_state_dict(client.shared_weights)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    # The training optimizer always forwards a physical batch of 64.
    # Reconstruct the exact deterministic final-batch replacement draws.
    from cafcl.continual import _FixedBatchSampler
    positions=next(iter(_FixedBatchSampler(4,conf.batch_size,conf.seed,True)))
    x=torch.stack([ImageBank.train[4+int(i)][0] for i in positions])
    labels=torch.tensor([2,2,3,3])[positions]
    logits=model(x)
    # New data now competes against every class previously observed by the
    # client (including classes 0 and 1), not just its latest task pair.
    old_task_loss=F.cross_entropy(logits[:,[0,1,2,3]],labels)
    params=dict(model.named_parameters())
    gradients=torch.autograd.grad(old_task_loss,list(params.values()),
                                   allow_unused=True)
    expected_task_grad={name:(g.detach().clone() if g is not None
                              else torch.zeros_like(p))
                        for (name,p),g in zip(params.items(),gradients)}
    local_train(model,client,ImageBank(),conf,torch.device('cpu'),0,
                experience_end=True,active_pair_classes=(2,3))
    fresh_delta={n:p.detach()-before[n] for n,p in model.named_parameters()}
    for name,delta in fresh_delta.items():
        newly_added=(client.pending_importance['si']['precision'][name]-old_prec[name])
        expected=(-expected_task_grad[name]*delta/
                  (delta.square()+conf.si_xi)).clamp(min=0)
        torch.testing.assert_close(newly_added,expected,atol=2e-5,rtol=1e-4)


@pytest.mark.parametrize('method',('der','derpp'))
def test_der_old_trajectory_logits_remain_immutable_after_new_task(method):
    model,client=client_model()
    client.local_experience=0
    client.arrival([0,1,2,3],(0,1),0)
    first=local_train(model,client,ImageBank(),cfg(method),
                      torch.device('cpu'),0,experience_end=True,
                      active_pair_classes=(0,1))
    targets={i:v.clone() for i,v in client.memory_logits.items()}
    client.shared_weights=first['weights']
    client.local_experience=1
    client.arrival([4,5,6,7],(2,3),1)
    local_train(model,client,ImageBank(),cfg(method),
                torch.device('cpu'),0,experience_end=True,
                active_pair_classes=(2,3))
    assert set(targets).issubset(client.memory_logits)
    for i,v in targets.items():
        torch.testing.assert_close(client.memory_logits[i],v,rtol=0,atol=0)
    assert len(client.memory_logits)==8


def test_inactive_rows_are_bitwise_frozen_even_with_optimizer_momentum():
    from cafcl.class_indexed import (
        freeze_other_class_rows,snapshot_frozen_rows,restore_frozen_rows)
    model=nn.Linear(3,4)
    optimizer=torch.optim.SGD(model.parameters(),lr=.1,momentum=.9)
    x=torch.randn(3,3)
    y=torch.tensor([0,1,2])
    # Populate momentum buffers on ALL classifier rows.
    nn.functional.cross_entropy(model(x),y).backward()
    optimizer.step()
    old_w=model.weight.detach().clone()
    old_b=model.bias.detach().clone()
    optimizer.zero_grad()
    nn.functional.cross_entropy(model(x),y).backward()
    freeze_other_class_rows(model,(0,1))
    snapshot=snapshot_frozen_rows(model,(0,1))
    optimizer.step()
    restore_frozen_rows(model,optimizer,snapshot)
    assert torch.equal(model.weight[2:],old_w[2:])
    assert torch.equal(model.bias[2:],old_b[2:])
    assert torch.equal(optimizer.state[model.weight]['momentum_buffer'][2:],
                       torch.zeros_like(model.weight[2:]))
    assert torch.equal(optimizer.state[model.bias]['momentum_buffer'][2:],
                       torch.zeros_like(model.bias[2:]))


@pytest.mark.parametrize('method,expected_replay_draws',(('der',1),('derpp',2)))
def test_derpp_independently_samples_logit_and_label_replay_minibatches(
    method,expected_replay_draws,monkeypatch
):
    from cafcl import continual as cl
    model,client=client_model()
    client.local_experience=0
    client.arrival([0,1,2,3],(0,1),0)
    client.first_seen_local_experience.update({0:0,1:0})
    first=local_train(model,client,ImageBank(),cfg(method),
        torch.device('cpu'),0,experience_end=True,active_pair_classes=(0,1))
    client.shared_weights=first['weights']
    client.local_experience=1
    client.arrival([4,5,6,7],(2,3),1)
    original=cl._example_loader
    recorded=[]
    def traced(*args,**kwargs):
        if kwargs.get('shuffle') is False:
            recorded.append(tuple(args[1]))
        return original(*args,**kwargs)
    monkeypatch.setattr(cl,'_example_loader',traced)
    local_train(model,client,ImageBank(),cfg(method),
                torch.device('cpu'),0,experience_end=True,active_pair_classes=(2,3))
    assert len(recorded)==expected_replay_draws
