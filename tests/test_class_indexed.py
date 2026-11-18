import pytest
import torch
from cafcl.models import create_model
from cafcl.class_indexed import (canonical_pair_loss,freeze_other_class_rows,
    classifier_parameter_names,classwise_merge)

@pytest.mark.parametrize('name',['smallcnn','mlp','linear'])
def test_only_two_classifier_rows_receive_gradients(name):
    model=create_model(name,10)
    x=torch.rand(4,3,32,32)
    y=torch.tensor([3,7,3,7])
    loss=canonical_pair_loss(model(x),y,(7,3))
    loss.backward()
    freeze_other_class_rows(model,{3,7})
    w,b=classifier_parameter_names(model)
    params=dict(model.named_parameters())
    assert torch.all(params[w].grad[[0,1,2,4,5,6,8,9]]==0)
    assert params[w].grad[[3,7]].abs().sum()>0
    if b is not None:assert torch.all(params[b].grad[[0,1,2,4,5,6,8,9]]==0)


def test_same_class_number_maps_to_one_server_row():
    base={'classifier.weight':torch.zeros(3,2),'classifier.bias':torch.zeros(3)}
    cand={k:v.clone() for k,v in base.items()}
    u=[]
    for weight,n in [(2.,1),(6.,3)]:
        w={k:v.clone() for k,v in base.items()}
        w['classifier.weight'][1]=weight
        w['classifier.bias'][1]=weight
        u.append({'weights':w,'class_update_steps':{1:n}})
    result=classwise_merge(base,cand,u,{1:4},('classifier.weight','classifier.bias'))
    assert torch.allclose(result['classifier.weight'][1],torch.tensor([2.5,2.5]))
    assert result['classifier.bias'][1].item()==pytest.approx(2.5)
    assert torch.all(result['classifier.weight'][0]==0)


def test_pair_loss_rejects_single_label_and_mismatch():
    with pytest.raises(ValueError,match='two distinct'):
        canonical_pair_loss(torch.randn(2,10),torch.tensor([2,2]),[2])
    with pytest.raises(ValueError,match='outside'):
        canonical_pair_loss(torch.randn(2,10),torch.tensor([2,5]),[2,3])

@pytest.mark.parametrize('method',['sgd','gem','agem','replay','ewc','si','er_ace','der','derpp','mas','lwf'])
def test_all_methods_two_binary_episodes_same_round_while_offline(method,tmp_path):
    from cafcl.config import RunConfig
    from cafcl.simulator import run_experiment
    holders={str(c):((0,) if c<4 else (1,)) for c in range(10)}
    cfg=RunConfig(dataset='synthetic',num_clients=2,stages=1,rounds_per_stage=1,
        model='linear',participation=.5,participation_overrides={'0':(1,)},
        method=method,aggregator='fedavg',holders_override=holders,
        class_pair_protocol='strict',checkpoint_every=1,
        memory_per_class=4,job_id='pairs_'+method)
    assert run_experiment(cfg,str(tmp_path),verbose=False)['status']=='completed'
    check=torch.load(tmp_path/cfg.job_id/'model_final.pt',weights_only=False)
    c=check['clients'][0]
    assert [e['classes'] for e in c['local_training_events']]==[[0,1],[2,3]]
    assert c['pending_upload_steps']>0 and c['model_version']==0
    assert set(c['pending_class_update_steps'])=={0,1,2,3}
    if method in {'gem','agem','replay','der','derpp','er_ace'}:
        assert set(c['memory'])=={0,1,2,3}
    assert set(check['clients'][1]['activated_labels'])=={4,5,6,7,8,9}


def test_strict_pair_protocol_rejects_singleton_arrival(tmp_path):
    from cafcl.config import RunConfig
    from cafcl.simulator import run_experiment
    cfg=RunConfig(dataset='synthetic',num_clients=10,stages=1,rounds_per_stage=1,
        scenario='disjoint',model='linear',class_pair_protocol='strict',job_id='strict_odd')
    with pytest.raises(ValueError,match='insufficient distinct class'):
        run_experiment(cfg,str(tmp_path),verbose=False)
