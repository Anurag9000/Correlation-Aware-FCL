"""Global classifiers with normalization safe for small per-client batches."""
import torch
from torch import nn

class SmallCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),nn.GroupNorm(4,32),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.GroupNorm(8,64),nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.GroupNorm(8,128),nn.ReLU(),
            nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),
            nn.Linear(512,128),nn.ReLU()
        )
        self.classifier=nn.Linear(128,classes)
    def forward(self,x): return self.classifier(self.features(x))

def create_model(name:str, classes:int):
    if name == 'smallcnn': return SmallCNN(classes)
    if name == 'linear': return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,classes))
    if name == 'mlp': return nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,256),nn.ReLU(),nn.Linear(256,classes))
    raise ValueError(name)

def copy_cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def check_finite(state):
    return all(torch.isfinite(v).all().item() for v in state.values() if v.is_floating_point())
