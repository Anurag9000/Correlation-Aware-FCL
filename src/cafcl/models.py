"""Native, swappable 32x32 research backbones.

ViT-Base/16 (standard ViT-B): 12 transformer blocks, 768 hidden dimensions, 12 heads.
Legacy ViT-Tiny/16 (not part of current suites): 192 dimensions, 3 heads,
16x16 patches (four image tokens at 32x32) with a learned class token.
ResNet-18: canonical [2,2,2,2] BasicBlock stages with CIFAR 3x3/stride-1
stem and *no* stem maxpool. Both have a single class-indexed final Linear.

All research runs initialize BOTH architectures from scratch by default.
This avoids treating unequal, adapted ImageNet transfer initialization as a
controlled model-comparison experiment. No silent pretrained fallback.
"""
from __future__ import annotations
import math
import hashlib
import torch
from torch import nn
from torch.nn import functional as F

VIT_ID='vit_base_patch16_32'
LEGACY_VIT_TINY_ID='vit_tiny_patch16_32'
CNN_ID='resnet18_cifar32'
BACKBONE_ID=VIT_ID  # backward-compatible default-model constant
INPUT_SIZE=32
NORMALIZATION_MEAN=(.5071,.4867,.4408)
NORMALIZATION_STD=(.2675,.2565,.2761)


def _state_sha256(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0')
        digest.update(str(tuple(tensor.shape)).encode()+b'\0')
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

class TransformerBlock(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4):
        super().__init__()
        self.norm1=nn.LayerNorm(dim,eps=1e-6)
        self.attn=nn.MultiheadAttention(dim,heads,dropout=0.,batch_first=True,bias=True)
        self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*mlp_ratio),nn.GELU(),nn.Linear(dim*mlp_ratio,dim))
    def forward(self,x):
        y=self.norm1(x)
        x=x+self.attn(y,y,y,need_weights=False)[0]
        return x+self.mlp(self.norm2(x))

class ViTTinyFeature(nn.Module):
    num_features=192
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Conv2d(3,192,kernel_size=16,stride=16)
        self.cls_token=nn.Parameter(torch.zeros(1,1,192))
        self.pos_embed=nn.Parameter(torch.zeros(1,5,192))  # 1 CLS + 2x2 patches
        self.blocks=nn.Sequential(*(TransformerBlock() for _ in range(12)))
        self.norm=nn.LayerNorm(192,eps=1e-6)
        nn.init.trunc_normal_(self.cls_token,std=.02)
        nn.init.trunc_normal_(self.pos_embed,std=.02)
    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,32,32):
            raise ValueError('ViT-Tiny/16 requires (N,3,32,32) images')
        tokens=self.patch_embed(x).flatten(2).transpose(1,2)
        tokens=torch.cat((self.cls_token.expand(x.size(0),-1,-1),tokens),dim=1)
        tokens=self.blocks(tokens+self.pos_embed)
        return self.norm(tokens)[:,0]

class ViTBaseFeature(nn.Module):
    """Standard ViT-Base/16 width, depth and heads adapted to native 32x32.

    Four 16x16 image patches plus CLS token. This is RANDOMLY initialized
    ViT-B/16, not a pretrained ImageNet ViT or an ImageNet-resolution model.
    """
    num_features=768
    def __init__(self):
        super().__init__()
        self.patch_embed=nn.Conv2d(3,768,kernel_size=16,stride=16)
        self.cls_token=nn.Parameter(torch.zeros(1,1,768))
        self.pos_embed=nn.Parameter(torch.zeros(1,5,768))
        self.blocks=nn.Sequential(*(TransformerBlock(768,12) for _ in range(12)))
        self.norm=nn.LayerNorm(768,eps=1e-6)
        nn.init.trunc_normal_(self.cls_token,std=.02)
        nn.init.trunc_normal_(self.pos_embed,std=.02)
    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,32,32):
            raise ValueError('ViT-Base/16 requires (N,3,32,32) images')
        patches=self.patch_embed(x).flatten(2).transpose(1,2)
        tokens=torch.cat((self.cls_token.expand(x.size(0),-1,-1),patches),dim=1)
        tokens=self.blocks(tokens+self.pos_embed)
        return self.norm(tokens)[:,0]


class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,inplanes,planes,stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(inplanes,planes,3,stride,1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(planes,planes,3,1,1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        if stride!=1 or inplanes!=planes:
            self.downsample=nn.Sequential(nn.Conv2d(inplanes,planes,1,stride,bias=False),nn.BatchNorm2d(planes))
        else:self.downsample=None
    def forward(self,x):
        identity=x if self.downsample is None else self.downsample(x)
        x=self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(x))+identity)

class CIFARResNet18Feature(nn.Module):
    num_features=512
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.relu=nn.ReLU(inplace=True)
        self.layer1=self._layer(64,64,2,1)
        self.layer2=self._layer(64,128,2,2)
        self.layer3=self._layer(128,256,2,2)
        self.layer4=self._layer(256,512,2,2)
        self.pool=nn.AdaptiveAvgPool2d(1)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):nn.init.ones_(m.weight);nn.init.zeros_(m.bias)
    @staticmethod
    def _layer(input_channels,output_channels,num_blocks,stride):
        return nn.Sequential(BasicBlock(input_channels,output_channels,stride),
                             *(BasicBlock(output_channels,output_channels) for _ in range(num_blocks-1)))
    def forward(self,x):
        if x.ndim!=4 or tuple(x.shape[1:])!=(3,32,32):
            raise ValueError('ResNet-18 CIFAR stem requires (N,3,32,32) images')
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.pool(x).flatten(1)

# Feature-builder registry: registering a new CNN or ViT is a single, explicit
# factory addition. Public CLI aliases remain intentionally limited to the
# requested two production variants until another architecture is approved.
# Native GLFC/LGA/FEAT backbones are loaded from frozen, byte-identical
# vendored upstream files. No modifications to those upstream directories.
class NativePublishedFeature(nn.Module):
    num_features=512
    def __init__(self,method):
        super().__init__()
        import importlib.util
        from pathlib import Path
        directory={'native_glfc_cbam':'GLFC/src/ResNet.py',
                   'native_lga_cbam':'LGA/ResNet.py',
                   'native_feat_resnet':'FEAT/convs/resnet.py'}[method]
        path=Path(__file__).resolve().parents[2]/directory
        if not path.is_file():raise FileNotFoundError(f'Original upstream source unavailable: {path}')
        name='_cafcl_upstream_'+method
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.method=method
        self.original=(module.resnet18(pretrained=False) if method=='native_feat_resnet'
                       else module.resnet18_cbam(pretrained=False))

    def forward(self,x):
        out=self.original(x)
        return out['features'] if self.method=='native_feat_resnet' else out

FEATURE_FACTORIES={VIT_ID:ViTBaseFeature,CNN_ID:CIFARResNet18Feature,
    'native_glfc_cbam':lambda:NativePublishedFeature('native_glfc_cbam'),
    'native_lga_cbam':lambda:NativePublishedFeature('native_lga_cbam'),
    'native_feat_resnet':lambda:NativePublishedFeature('native_feat_resnet')}
MODEL_IDS=tuple(FEATURE_FACTORIES)

class CanonicalClassifier(nn.Module):
    def __init__(self, backbone_name:str, classes:int):
        super().__init__()
        if backbone_name not in MODEL_IDS:raise ValueError(f'Unknown research backbone: {backbone_name}')
        self.features=FEATURE_FACTORIES[backbone_name]()
        self.classifier=nn.Linear(self.features.num_features,classes)
        self.backbone_description=dict(model_id=backbone_name,model_family='vit' if backbone_name==VIT_ID else 'cnn',
            input_size=32,classes=classes,pretrained=False,initialization='seeded_random',
            normalization_mean=NORMALIZATION_MEAN,normalization_std=NORMALIZATION_STD,
            patch_size=16 if backbone_name==VIT_ID else None,
            vit_variant='ViT-Base/16' if backbone_name==VIT_ID else None,
            classifier='canonical_class_indexed',implementation='native_torch')
    def forward(self,x):return self.classifier(self.features(x))

# One registry, intentionally easy to extend with requested ViT/ResNet variants later.
BACKBONES={'vit':VIT_ID,'cnn':CNN_ID}

def resolve_backbone(name:str)->str:
    if name in BACKBONES:return BACKBONES[name]
    if name in MODEL_IDS:return name
    raise ValueError(f'Unsupported research backbone {name}; choose vit or cnn')

def create_model(name:str,classes:int,*,allow_test_model=False):
    if name in MODEL_IDS:return CanonicalClassifier(name,classes)
    if not allow_test_model:raise ValueError(f'Production requires one of {MODEL_IDS}, got {name}')
    if name=='test_linear':return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,classes))
    if name=='test_mlp':return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,256),nn.ReLU(),nn.Linear(256,classes))
    if name=='test_smallcnn':return nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(8,classes))
    raise ValueError(name)

def copy_cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def check_finite(state):
    return all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
