"""Native, swappable 32x32 research backbones.

ViT-Base/16 (standard ViT-B): 12 transformer blocks, 768 hidden dimensions, 12 heads.
Legacy ViT-Tiny/16 (not part of current suites): 192 dimensions, 3 heads,
16x16 patches (four image tokens at 32x32) with a learned class token.
ResNet-18: canonical [2,2,2,2] BasicBlock stages with CIFAR 3x3/stride-1
stem and *no* stem maxpool. Task-IL models share a backbone and have one classifier head per task class set.

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
        self.original=(module.resnet18(pretrained=False,args={'dataset':'cifar100','init_cls':2,'increment':2}) if method=='native_feat_resnet'
                       else module.resnet18_cbam(pretrained=False))

    def forward(self,x):
        out=self.original(x)
        if self.method=='native_feat_resnet':
            return out['features']
        # LGA's upstream feature extractor deliberately returns [N,512,4,4];
        # its original myNetwork.network applies AvgPool2d(4) and flattening
        # before the classifier. The port must reproduce that projection.
        if out.ndim==4:
            out=torch.nn.functional.adaptive_avg_pool2d(out,1).flatten(1)
        if out.ndim!=2 or out.shape[1]!=self.num_features:
            raise RuntimeError(f'Unexpected published backbone features: {out.shape}')
        return out

class FedCLASSResNet8Feature(nn.Module):
    """Unmodified FedCLASS/convs/resnet_client.py ResNet-8, CIFAR 32x32.

    FedCLASS's CENet wraps this feature extractor and SimpleLinear.
    We keep both upstream modules byte-identical; only their loading and
    100-class / sparse-class scheduling are adapted here.
    """
    num_features=64

    def __init__(self):
        super().__init__()
        import importlib.util
        from pathlib import Path
        source=Path(__file__).resolve().parents[2]/'FedCLASS/convs/resnet_client.py'
        if not source.is_file():
            raise FileNotFoundError(f'Vendored FedCLASS architecture missing: {source}')
        spec=importlib.util.spec_from_file_location('_cafcl_fedclass_resnet_client',source)
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.original=module.resnet8(pretrained=False)
        if self.original.out_dim!=self.num_features:
            raise RuntimeError('FedCLASS ResNet8 feature dimension changed')

    def forward(self,x):
        features=self.original(x)
        if features.ndim!=2 or features.shape[1]!=self.num_features:
            raise RuntimeError(f'FedCLASS ResNet8 unexpected features: {tuple(features.shape)}')
        return features


def _fedclass_original_linear(in_features,out_features):
    """Construct the original upstream SimpleLinear, including zero bias init."""
    import importlib.util
    from pathlib import Path
    source=Path(__file__).resolve().parents[2]/'FedCLASS/convs/linears.py'
    if not source.is_file():
        raise FileNotFoundError(f'Vendored FedCLASS classifier missing: {source}')
    spec=importlib.util.spec_from_file_location('_cafcl_fedclass_linears',source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleLinear(in_features,out_features,bias=True)


class FixedSimplexETFHead(nn.Module):
    """Fixed C-class simplex ETF using a d-dimensional orthonormal embedding.

    This reproduces the *paper's geometry*, not the unpublished FEAT
    IncrementalNet_CausalETF source. Each row has unit norm; pairwise
    inner products equal -1/(C-1). The prototypes stay fixed across local
    tasks and federated synchronizations as FEAT requires.

    A requires_grad=False Parameter preserves the existing canonical
    classifier introspection and leaves the fixed prototype tensor in
    model.state_dict for full state synchronization.
    """
    def __init__(self,features,classes):
        super().__init__()
        if not (2<=classes<=features):
            raise ValueError('Nondegenerate simplex ETF needs 2<=C<=d')
        basis,_=torch.linalg.qr(torch.randn(features,classes),mode='reduced')
        center=torch.eye(classes)-torch.ones(classes,classes)/classes
        etf=(classes/(classes-1))**.5*(basis@center).T.contiguous()
        self.in_features=features
        self.out_features=classes
        self.weight=nn.Parameter(etf,requires_grad=False)
        self.register_buffer('source_model_reconstruction',torch.tensor(1,dtype=torch.int8))

    def forward(self,features):
        return F.linear(features,self.weight,None)


def task_head_key(classes):
    if isinstance(classes,str):
        if not classes.startswith('task_') or '/' in classes or '.' in classes:
            raise ValueError(f'Invalid task ID {classes!r}')
        return classes
    values=tuple(sorted(set(map(int,classes))))
    if not values or values[0]<0:
        raise ValueError('Task heads need nonnegative global class IDs')
    return 'task_'+'_'.join(f'{c:03d}' for c in values)


def task_class_sets_from_state_dict(state):
    """Recover occurrence-specific task IDs and class maps from head keys."""
    prefix='classifier.heads.'
    keys=sorted({name[len(prefix):].split('.',1)[0]
                 for name in state if name.startswith(prefix)})
    if not keys:
        raise ValueError('Checkpoint has no task-specific classifier heads')
    result={}
    for key in keys:
        if not key.startswith('task_'):
            raise ValueError(f'Unrecognized task-head key {key!r}')
        class_part=key[5:].split('__',1)[0]
        values=tuple(int(value) for value in class_part.split('_') if value)
        if not values or task_head_key(key)!=key:
            raise ValueError(f'Invalid task-head key {key!r}')
        if tuple(sorted(set(values)))!=values:
            raise ValueError(f'Noncanonical class order in task-head key {key!r}')
        result[key]=values
    return result


class TaskHeadBank(nn.Module):
    """Independent classifier per canonical task class set.

    The task ID is the sorted tuple of global labels. Shared labels in two
    tasks therefore have separate parameters in the two heads.
    """
    def __init__(self,in_features,task_class_sets,*,head_kind='linear'):
        super().__init__()
        if isinstance(task_class_sets,dict):
            entries=sorted((task_head_key(task_id),tuple(sorted(set(map(int,classes)))) )
                           for task_id,classes in task_class_sets.items())
            if len({key for key,_ in entries})!=len(entries):
                raise ValueError('Task IDs must be unique')
        else:
            unique=sorted({tuple(sorted(set(map(int,c)))) for c in task_class_sets})
            entries=[(task_head_key(classes),classes) for classes in unique]
        if not entries or any(not classes for _,classes in entries):
            raise ValueError('At least one nonempty task class set is required')
        self.task_classes=dict(entries)
        self.heads=nn.ModuleDict()
        self.active_task_key=next(iter(self.task_classes))
        for key,classes in entries:
            if head_kind=='fedclass':
                head=_fedclass_original_linear(in_features,len(classes))
            elif head_kind=='etf':
                head=FixedSimplexETFHead(in_features,len(classes))
            else:
                head=nn.Linear(in_features,len(classes))
            self.heads[key]=head

    def set_task_id(self,task_id):
        key=task_id if isinstance(task_id,str) and task_id in self.heads else task_head_key(task_id)
        if key not in self.heads:
            raise KeyError(f'Unknown task ID {task_id!r}; registered={tuple(self.task_classes.values())}')
        self.active_task_key=key

    def classes_for(self,task_id=None):
        key=self._key(task_id)
        return self.task_classes[key]

    def _key(self,task_id=None):
        if task_id is None:
            key=self.active_task_key
            if key is None and len(self.heads)==1:key=next(iter(self.heads))
        else:
            key=task_id if isinstance(task_id,str) and task_id in self.heads else task_head_key(task_id)
        if key not in self.heads:
            raise ValueError('Prediction/training requires a valid task ID')
        return key

    def forward(self,features,task_id=None):
        return self.heads[self._key(task_id)](features)

    @property
    def weight(self):
        return self.heads[self._key()].weight

    @property
    def bias(self):
        return getattr(self.heads[self._key()],'bias',None)

    @property
    def out_features(self):
        return len(self.task_classes[self._key()])


class TaskIncrementalClassifier(nn.Module):
    def __init__(self,features,classes,task_class_sets,*,model_id=None):
        super().__init__()
        self.features=features
        kind=('fedclass' if model_id=='fedclass_resnet8' else
              'etf' if model_id=='native_feat_etf' else 'linear')
        self.classifier=TaskHeadBank(features.num_features,task_class_sets,head_kind=kind)
        self.num_classes=int(classes)
        self.task_class_sets=tuple(sorted(self.classifier.task_classes.values()))
        self.backbone_description=dict(model_id=model_id,task_incremental=True,
            num_global_classes=self.num_classes,task_head_class_sets=[list(x) for x in self.task_class_sets],
            task_id_to_classes={key:list(value) for key,value in sorted(self.classifier.task_classes.items())},
            initialization='seeded_random')

    def set_active_task_id(self,task_id):
        self.classifier.set_task_id(task_id)

    def forward(self,x,task_id=None):
        return self.classifier(self.features(x),task_id)

    def loss_for_labels(self,x,labels,task_ids=None,task_ids_by_example=None,features=None):
        """Replay loss using each sample's originating task ID when supplied.

        Task IDs are sets of global labels, so a sample can have multiple
        valid IDs only when the scenario intentionally reuses that same raw
        example. Its loss is averaged over those source tasks, not every task
        head that happens to contain its class.
        """
        features=self.features(x) if features is None else features
        labels=labels.to(features.device,dtype=torch.long)
        if task_ids_by_example is not None and len(task_ids_by_example)!=len(labels):
            raise ValueError('One source task-ID set is required per replay example')
        losses=torch.zeros(len(labels),device=features.device)
        counts=torch.zeros(len(labels),device=features.device)
        selected_keys=(tuple(self.classifier._key(task_id) for task_id in task_ids)
                       if task_ids is not None else tuple(self.classifier.task_classes))
        for key in sorted(set(selected_keys)):
            classes=self.classifier.task_classes[key]
            lookup={c:i for i,c in enumerate(classes)}
            mask=torch.zeros_like(labels,dtype=torch.bool)
            for c in classes:mask|=(labels==c)
            if task_ids_by_example is not None:
                allowed=torch.as_tensor(
                    [key in {self.classifier._key(task_id) for task_id in ids}
                     for ids in task_ids_by_example],
                    dtype=torch.bool,device=features.device)
                mask&=allowed
            if not mask.any():continue
            selected=self.classifier(features[mask],key)
            targets=torch.tensor([lookup[int(y)] for y in labels[mask].tolist()],
                                 dtype=torch.long,device=features.device)
            losses[mask]+=F.cross_entropy(selected,targets,reduction='none')
            counts[mask]+=1
        if (counts==0).any():
            missing=labels[counts==0].unique().tolist()
            raise ValueError(f'Replay labels have no task head: {missing}')
        return (losses/counts).mean()

    @torch.inference_mode()
    def task_union_scores(self,x,task_id_sets):
        """Average per-head probabilities for duplicated global labels.

        Returns normalized global-label probabilities restricted to the union
        of requested task heads for each input row.
        """
        if len(task_id_sets)!=len(x):
            raise ValueError('One task-ID set is required per prediction row')
        features=self.features(x)
        result=torch.full((len(x),self.num_classes),-1.,device=features.device)
        for row,ids in enumerate(task_id_sets):
            sums=torch.zeros(self.num_classes,device=features.device)
            counts=torch.zeros(self.num_classes,device=features.device)
            keys=sorted({self.classifier._key(tid) for tid in ids})
            if not keys:raise ValueError('Task-ID union cannot be empty')
            for key in keys:
                classes=self.classifier.task_classes[key]
                probs=torch.softmax(self.classifier(features[row:row+1],key),dim=1)[0]
                idx=torch.as_tensor(classes,dtype=torch.long,device=features.device)
                sums[idx]+=probs
                counts[idx]+=1
            present=counts>0
            scores=sums[present]/counts[present]
            scores=scores/scores.sum().clamp_min(torch.finfo(scores.dtype).tiny)
            result[row,present]=scores
        return result


class SyntheticTaskIncrementalClassifier(TaskIncrementalClassifier):
    def __init__(self,model,task_class_sets,classes):
        layers=list(model.children())
        final=layers[-1]
        if not isinstance(final,nn.Linear):
            raise ValueError('Synthetic task model needs a final Linear layer')
        features=nn.Sequential(*layers[:-1])
        features.num_features=int(final.in_features)
        super().__init__(features,classes,task_class_sets,model_id='synthetic')


FEATURE_FACTORIES={VIT_ID:ViTBaseFeature,CNN_ID:CIFARResNet18Feature,
    'native_glfc_cbam':lambda:NativePublishedFeature('native_glfc_cbam'),
    'native_lga_cbam':lambda:NativePublishedFeature('native_lga_cbam'),
    'native_feat_resnet':lambda:NativePublishedFeature('native_feat_resnet'),
    'native_feat_etf':lambda:NativePublishedFeature('native_feat_resnet'),
    'fedclass_resnet8':FedCLASSResNet8Feature}
MODEL_IDS=tuple(FEATURE_FACTORIES)

class CanonicalClassifier(TaskIncrementalClassifier):
    def __init__(self,backbone_name:str,classes:int,task_class_sets):
        if backbone_name not in MODEL_IDS:raise ValueError(f'Unknown research backbone: {backbone_name}')
        features=FEATURE_FACTORIES[backbone_name]()
        super().__init__(features,classes,task_class_sets,model_id=backbone_name)
        self.backbone_description.update(model_family='vit' if backbone_name==VIT_ID else 'cnn',
            input_size=32,pretrained=False,normalization_mean=NORMALIZATION_MEAN,
            normalization_std=NORMALIZATION_STD,
            patch_size=16 if backbone_name==VIT_ID else None,
            vit_variant='ViT-Base/16' if backbone_name==VIT_ID else None,
            classifier='independent_task_heads',
            upstream_source=('FedCLASS/convs/resnet_client.py:resnet8 + FedCLASS/convs/linears.py:SimpleLinear'
                             if backbone_name=='fedclass_resnet8' else None))

# One registry, intentionally easy to extend with requested ViT/ResNet variants later.
BACKBONES={'vit':VIT_ID,'cnn':CNN_ID}

def resolve_backbone(name:str)->str:
    if name in BACKBONES:return BACKBONES[name]
    if name in MODEL_IDS:return name
    raise ValueError(f'Unsupported research backbone {name}; choose vit or cnn')

class JointClassIncrementalClassifier(nn.Module):
    """Explicit one-head global-label classifier for the joint protocol."""
    def __init__(self, features, classes, model_id):
        super().__init__()
        self.features=features
        self.num_classes=int(classes)
        self.classifier=nn.Linear(features.num_features,self.num_classes)
        self.backbone_description=dict(model_id=model_id,task_incremental=False,
            num_global_classes=self.num_classes,classifier='shared_class_indexed_linear',
            initialization='seeded_random',input_size=32,pretrained=False)

    def forward(self,x):
        return self.classifier(self.features(x))


def create_model(name:str,classes:int,*,allow_test_model=False,task_class_sets=None,learning_setting='task_il'):
    if learning_setting=='class_il':
        if task_class_sets is not None:
            raise ValueError('Class-IL cannot receive task heads')
        if name in FEATURE_FACTORIES:
            return JointClassIncrementalClassifier(FEATURE_FACTORIES[name](),classes,name)
        if not allow_test_model:
            raise ValueError('Unknown Class-IL research backbone')
        # Synthetic fixtures only; no implicit production fallback.
        if name=='test_linear':
            features=nn.Flatten();features.num_features=3*32*32
        elif name=='test_mlp':
            features=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,256),nn.ReLU());features.num_features=256
        elif name=='test_smallcnn':
            features=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten());features.num_features=8
        else:
            raise ValueError(name)
        return JointClassIncrementalClassifier(features,classes,name)
    if learning_setting!='task_il':
        raise ValueError('Unknown learning setting')
    if task_class_sets is None:
        task_class_sets=(tuple(range(int(classes))),)
    if name in MODEL_IDS:return CanonicalClassifier(name,classes,task_class_sets)
    if not allow_test_model:raise ValueError(f'Production requires one of {MODEL_IDS}, got {name}')
    if name=='test_linear':base=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,classes))
    elif name=='test_mlp':base=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,256),nn.ReLU(),nn.Linear(256,classes))
    elif name=='test_smallcnn':base=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(8,classes))
    else:raise ValueError(name)
    return SyntheticTaskIncrementalClassifier(base,task_class_sets,classes)

def copy_cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def check_finite(state):
    return all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
