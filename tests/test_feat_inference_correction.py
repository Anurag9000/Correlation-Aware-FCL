"""Active, no-oracle FEAT projector correction with precise provenance."""
import torch
import torch.nn.functional as F
import pytest
from cafcl.feat_inference_correction import paper_reconstructed_feat_logits
from cafcl.evaluation import evaluate


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*4*4,4))
        self.classifier=torch.nn.Linear(4,4,bias=False)
        self.backbone_description={'model_id':'native_feat_etf'}
    def forward(self,x):return self.classifier(self.features(x))

def receipt():
    return dict(PH=torch.diag(torch.tensor([1.,0.,0.,0.])),
                PT=torch.diag(torch.tensor([0.,1.,0.,0.])),
                r_H=1,r_T=1,energy={'bar_eH_G':0.,'nT_sum':4})

def test_corrected_output_matches_third_party_formula():
    m=Tiny().eval()
    x=torch.randn(4,3,4,4)
    r=receipt()
    z=paper_reconstructed_feat_logits(m,x,r)
    f=F.normalize(m.features(x),dim=1)
    h=f@r['PH'];t=f@r['PT']
    eh=h.square().sum(1);et=t.square().sum(1)
    gate=(eh/(eh+et+1e-8)).clamp_min(0)
    true=m.classifier(F.normalize(f-gate[:,None]*h+gate[:,None]*t,dim=1))
    torch.testing.assert_close(z,true)
    assert not torch.equal(z,m(x))

def test_missing_prior_keeps_original_output():
    m=Tiny().eval()
    x=torch.randn(2,3,4,4)
    torch.testing.assert_close(paper_reconstructed_feat_logits(m,x,None),m(x))

def test_wrong_etf_and_projector_rejected():
    m=Tiny()
    x=torch.randn(2,3,4,4)
    bad=receipt();bad['PH']=torch.eye(3)
    with pytest.raises(ValueError,match='shape mismatch'):
        paper_reconstructed_feat_logits(m,x,bad)
    m.backbone_description['model_id']='native_glfc_cbam'
    with pytest.raises(RuntimeError,match='ETF'):
        paper_reconstructed_feat_logits(m,x,receipt())

def test_evaluation_uses_corrected_inference_without_task_id():
    class Set:
        targets=[0,1,2,3]*2
        def __len__(self):return 8
        def __getitem__(self,i):return torch.full((3,4,4),float(i)/8),self.targets[i]
    class Bank:
        num_classes=4;test=Set();validation=Set()
    r,_=evaluate(Tiny(),Bank(),set(range(4)),8,torch.device('cpu'),
                 feat_ecc=receipt(),integrated=set(range(4)))
    assert r['task_id_at_inference'] is False
    assert r['evaluation_samples']==8
