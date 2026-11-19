"""Production native 32x32 standard ViT-Base/16 and CIFAR-stem ResNet-18 contracts."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from cafcl.models import VIT_ID,CNN_ID,create_model,ViTBaseFeature,CIFARResNet18Feature,resolve_backbone
from cafcl.config import RunConfig
from cafcl.class_indexed import classifier_parameter_names,canonical_active_loss
from cafcl.full_matrix import expand
from cafcl.simulator import aggregate


def test_both_architectures_are_native_32_and_canonical():
    for model_id,dim in ((VIT_ID,768),(CNN_ID,512)):
        model=create_model(model_id,100)
        assert model.features.num_features==dim
        assert model.classifier.out_features==100
        assert classifier_parameter_names(model)==('classifier.weight','classifier.bias')
        assert model.backbone_description['input_size']==32
        assert model.backbone_description['implementation']=='native_torch'
        assert model.backbone_description['pretrained'] is False
        assert model(torch.randn(2,3,32,32)).shape==(2,100)
        with pytest.raises(ValueError,match='32'):
            model(torch.randn(1,3,224,224))


def test_native_vit_architecture_exact_patch_grid():
    m=create_model('vit_base_patch16_32',100)
    assert isinstance(m.features,ViTBaseFeature)
    assert m.features.patch_embed.kernel_size==(16,16)
    assert m.features.patch_embed.stride==(16,16)
    assert m.features.pos_embed.shape==(1,5,768) # 1 CLS + four 16x16 patches
    assert len(m.features.blocks)==12
    assert m.features.blocks[0].attn.num_heads==12


def test_native_resnet18_cifar_stem_and_four_stages():
    m=create_model('resnet18_cifar32',100)
    assert isinstance(m.features,CIFARResNet18Feature)
    assert m.features.conv1.kernel_size==(3,3)
    assert m.features.conv1.stride==(1,1)
    assert not hasattr(m.features,'maxpool')
    assert [len(getattr(m.features,f'layer{i}')) for i in range(1,5)]==[2,2,2,2]


def test_backward_uses_only_active_classifier_rows_both_backbones():
    for model_id in (VIT_ID,CNN_ID):
        m=create_model(model_id,100)
        m.train()
        y=m(torch.randn(2,3,32,32))
        canonical_active_loss(y,torch.tensor([6,25]),(6,25)).backward()
        assert torch.count_nonzero(m.classifier.weight.grad[[0,1,99]])==0
        assert torch.count_nonzero(m.classifier.weight.grad[[6,25]])>0
        assert torch.count_nonzero(next(m.features.parameters()).grad)>0


def test_production_batch1024_and_swappable_backbone_guard():
    from cafcl.full_matrix import base
    for model_id in (VIT_ID,CNN_ID):
        cfg=replace(base(0),model=model_id,batch_size=64)
        cfg.validate()
    with pytest.raises(AssertionError,match='1024'):
        replace(RunConfig(),batch_size=512).validate()
    with pytest.raises(AssertionError,match='Research requires'):
        replace(RunConfig(),model='test_smallcnn').validate()
    with pytest.raises(ValueError):resolve_backbone('deit_tiny_patch16_224.fb_in1k')
    for choice in ('vit','cnn'):
        jobs=list(expand(seeds=(0,),families=('balanced',),model=choice))
        assert 0<len(jobs)<=11*9*6*3
        assert {cfg.model for _,cfg in jobs}=={resolve_backbone(choice)}
        assert {cfg.batch_size for _,cfg in jobs}=={64}


def test_resnet_bn_integer_counters_with_nonuniform_offline_updates():
    a={'bn.num_batches_tracked':torch.tensor(3,dtype=torch.int64),'w':torch.tensor([1.])}
    b={'bn.num_batches_tracked':torch.tensor(11,dtype=torch.int64),'w':torch.tensor([3.])}
    c={'bn.num_batches_tracked':torch.tensor(0,dtype=torch.int64),'w':torch.tensor([0.])}
    updates=[dict(weights=a,examples=3),dict(weights=b,examples=1)]
    merged=aggregate(updates,c,'fedavg')
    assert merged['bn.num_batches_tracked'].item()==11
    assert merged['w'].item()==pytest.approx(1.5)
