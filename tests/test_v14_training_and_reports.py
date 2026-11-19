"""v14: end-to-end class-IL calibration, full SGD and stage accuracy receipts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import csv
import json
import torch
import torch.nn.functional as F
import pytest

from cafcl.config import RunConfig
from cafcl.class_indexed import canonical_active_loss
from cafcl.continual import LocalState, local_train
from cafcl.models import create_model, VIT_ID, CNN_ID
from cafcl.stage_report import render_sync_report
from cafcl.simulator import run_experiment


def test_known_class_softmax_old_vs_new_competition_and_future_masking():
    logits=torch.tensor([[6.,0.,3.,2.,-1.,5.]],requires_grad=True)
    target=torch.tensor([2])
    pair=canonical_active_loss(logits,target,(2,3))
    multiclass=canonical_active_loss(logits,target,(0,1,2,3))
    assert float(multiclass)>float(pair)+2.
    grad=torch.autograd.grad(multiclass,logits)[0][0]
    assert grad[0]>0 and grad[2]<0
    assert torch.equal(grad[4:],torch.zeros_like(grad[4:]))
    pairgrad=torch.autograd.grad(pair,logits)[0][0]
    assert torch.equal(pairgrad[:2],torch.zeros_like(pairgrad[:2]))


class TinyBank:
    class Train:
        targets=[0,0,1,1,2,2,3,3]
        def __len__(self):return 8
        def __getitem__(self,i):
            x=torch.zeros(3,32,32)
            c=self.targets[i]
            x[c%3,c,:]=.25+i*.07
            return x,c
    train=Train()


@pytest.mark.parametrize('method',('replay','ewc','si','mas','lwf','gem','agem','derpp','sgd'))
def test_every_method_can_update_known_rows_but_never_future_rows(method):
    from cafcl.models import copy_cpu_state
    torch.manual_seed(36)
    model=create_model('test_linear',6,allow_test_model=True)
    c=LocalState(0)
    c.receive(copy_cpu_state(model),0)
    config=RunConfig(dataset='synthetic',model='test_linear',method=method,
                     num_clients=2,batch_size=16,micro_batch_size=4,
                     lr=.03,epochs=1,memory_per_class=2,
                     importance_samples_per_experience=2,
                     momentum=.9,weight_decay=5e-4)
    c.local_experience=0
    c.arrival([0,1,2,3],(0,1),0)
    first=local_train(model,c,TinyBank(),config,torch.device('cpu'),0,
                      active_pair_classes=(0,1),experience_end=True)
    c.shared_weights=first['weights']
    c.local_experience=1
    c.arrival([4,5,6,7],(2,3),1)
    prior=first['weights']['1.bias'].clone()
    result=local_train(model,c,TinyBank(),config,torch.device('cpu'),1,
                       active_pair_classes=(2,3),experience_end=True)
    after=result['weights']['1.bias']
    if method in ('replay','sgd'):
        assert not torch.equal(after[:2],prior[:2]), (
            'Old class rows must be allowed to participate in global seen-class CE')
    assert torch.equal(after[4:],prior[4:]),'Unknown future classifier rows must stay unchanged'
    assert result['classification_competitor_classes']==[0,1,2,3]
    assert result['sgd_momentum']==pytest.approx(.9)
    assert result['sgd_weight_decay']==pytest.approx(.0005)


def test_er_ace_keeps_original_asymmetric_current_stream_loss():
    from cafcl.class_indexed import canonical_active_loss
    logits=torch.tensor([[9.,8.,4.,3.]],requires_grad=True)
    label=torch.tensor([2])
    stream=canonical_active_loss(logits,label,(2,3))
    full=canonical_active_loss(logits,label,(0,1,2,3))
    assert stream.item()<full.item()
    assert torch.all(torch.autograd.grad(stream,logits)[0][0,:2]==0)


def test_sgd_defaults_and_loss_are_documented():
    cfg=RunConfig(dataset='synthetic')
    assert cfg.local_optimizer=='sgd'
    assert cfg.momentum==pytest.approx(.9)
    assert cfg.weight_decay==pytest.approx(.0005)
    assert cfg.lr>0


def test_active_suite_uses_standard_vit_b_16_not_vit_tiny():
    from cafcl.models import resolve_backbone,ViTBaseFeature
    assert resolve_backbone('vit')==VIT_ID=='vit_base_patch16_32'
    assert resolve_backbone('cnn')==CNN_ID=='resnet18_cifar32'
    assert ViTBaseFeature.num_features==768
    from cafcl.models import FEATURE_FACTORIES
    assert 'vit_tiny_patch16_32' not in FEATURE_FACTORIES


def test_per_sync_report_tracks_known_class_history_and_no_test_feedback(tmp_path):
    arrivals=[SimpleNamespace(round_id=0,client_id=0,labels=(0,1)),
              SimpleNamespace(round_id=0,client_id=0,labels=(2,3)),
              SimpleNamespace(round_id=0,client_id=1,labels=(4,5)),
              SimpleNamespace(round_id=1,client_id=0,labels=(6,7))]
    mat=[[0]*8 for _ in range(8)]
    for c in range(8):mat[c][c]=3
    mat[7][7]=0;mat[7][0]=3
    metrics=dict(accuracy=21/24,per_class={c:mat[c][c]/3 for c in range(8)},
                 confusion_matrix={'labels':list(range(8)),'counts':mat})
    vals=dict(accuracy=.83)
    record=render_sync_report(tmp_path,round_number=2,model_version=2,
        num_clients=2,arrivals=arrivals,test_metrics=metrics,
        validation_metrics=vals,selected_clients=[0,1],backbone='test_linear')
    assert record['seen_class_count']==8
    d=Path(tmp_path)/'per_sync_accuracy'
    for suffix in ('.json','.csv','.md','.png'):
        assert (d/('sync_002'+suffix)).is_file()
    j=json.loads((d/'sync_002.json').read_text())
    assert j['test_only_no_feedback'] is True
    assert len(j['entries'])==3
    assert j['zero_prediction_classes']==[7]
    with (d/'sync_002.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==8
    assert 'Client 0' in (d/'sync_002.md').read_text()


def test_synthetic_run_exports_all_round_tables_without_changing_communication(tmp_path):
    cfg=RunConfig(dataset='synthetic',model='test_linear',
        num_clients=2,stream_protocol=True,class_pair_protocol='strict',
        method='replay',aggregator='fedavg',batch_size=64,micro_batch_size=4,
        epochs=1,stages=1,rounds_per_stage=1,
        momentum=.9,weight_decay=5e-4,
        participation=1.,evaluate_final_only=True,job_id='stage_v14')
    result=run_experiment(cfg,str(tmp_path),verbose=False)
    assert result['status']=='completed'
    run=tmp_path/'stage_v14'
    metrics=json.loads((run/'metrics.json').read_text())
    assert len(metrics)==3
    assert [len(x['task_barriers']) for x in metrics]==[2,1,0]
    assert [len(x['per_sync_test_per_class']) for x in metrics]==[8,10,10]
    for i,row in enumerate(metrics,1):
        assert row['per_sync_test_accuracy'] is not None
        assert row['per_sync_validation_accuracy'] is not None
        assert (run/'per_sync_accuracy'/f'sync_{i:03d}.png').is_file()
        assert (run/'per_sync_accuracy'/f'sync_{i:03d}.csv').is_file()
        assert row['sync_accuracy_report']['test_accuracy']==row['per_sync_test_accuracy']
    assert metrics[0]['evaluation_deferred'] is True
    assert metrics[1]['evaluation_deferred'] is True
    assert metrics[-1]['evaluation_split']=='test'


def test_agem_historical_reference_does_not_compete_with_unknown_future_classes():
    from cafcl.continual import _reference_gradient
    from torch import nn

    class ReferenceDataset:
        targets=[0,1]
        def __len__(self):return 2
        def __getitem__(self,i):
            return torch.tensor([1.+i]),self.targets[i]

    model=nn.Linear(1,4,bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.2],[-.6],[.5],[9.]]))
    actual=_reference_gradient(
        model,list(model.parameters()),ReferenceDataset(),[0,1],2,
        torch.device('cpu'),seed=12,
        active_pair_classes=(0,1,2),reference_classes=None)
    model.zero_grad(set_to_none=True)
    y=torch.tensor([0,1])
    F.cross_entropy(model(torch.tensor([[1.],[2.]]))[:,[0,1,2]],y).backward()
    expected=model.weight.grad.flatten()
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)
    assert actual[3]==0  # genuinely unknown future classifier output


def test_final_only_diagnostics_still_update_validation_forgetting_history(tmp_path):
    cfg=RunConfig(dataset='synthetic',model='test_linear',
        num_clients=2,stream_protocol=True,class_pair_protocol='strict',
        method='sgd',aggregator='fedavg',batch_size=64,micro_batch_size=4,
        epochs=1,stages=1,rounds_per_stage=1,
        momentum=.9,weight_decay=5e-4,participation=1.,
        evaluate_final_only=True,job_id='validate_forgetting_peaks_v14')
    run_experiment(cfg,str(tmp_path),verbose=False)
    run=tmp_path/cfg.job_id
    metrics=json.loads((run/'metrics.json').read_text())
    ck=torch.load(run/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
    peaks={int(k):float(v) for k,v in ck['peak'].items()}
    assert peaks, 'Validation history must not remain empty when test is deferred'
    first={int(k):float(v) for k,v in metrics[0]['selection_validation_per_class'].items()}
    assert first
    for cls,score in first.items():
        assert peaks[cls] >= score
    assert metrics[0]['evaluation_deferred'] is True
    assert metrics[0]['selection_validation_per_class']
    # Audit TEST and selection VALIDATION predictions remain segregated.
    assert 'per_sync_test_per_class' in metrics[0]
    assert metrics[0]['per_class']=={}


def test_unarrived_future_outputs_not_false_flagged_as_prediction_collapse(tmp_path):
    mat=[[0]*100 for _ in range(100)]
    mat[0][0]=10
    mat[1][1]=10
    report=render_sync_report(tmp_path,round_number=1,model_version=1,
        num_clients=2,
        arrivals=[SimpleNamespace(round_id=0,client_id=0,labels=(0,1))],
        test_metrics={'accuracy':1.,'per_class':{0:1.,1:1.},
                      'confusion_matrix':{'labels':list(range(100)),'counts':mat}},
        selected_clients=[0],integrated_classes={0,1},backbone='test_linear')
    assert report['never_predicted_classes']==0
    result=json.loads((tmp_path/'per_sync_accuracy'/'sync_001.json').read_text())
    assert result['zero_prediction_classes']==[]
    assert result['observed_class_count']==2
    assert len(result['predicted_class_counts'])==2
