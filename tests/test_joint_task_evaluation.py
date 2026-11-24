import unittest
import torch
from types import SimpleNamespace
from cafcl.models import create_model
from cafcl.joint.task_evaluation import evaluate_task_conditioned


class Dataset(torch.utils.data.Dataset):
    targets=[0,1,2,0,1,2]
    def __len__(self):return len(self.targets)
    def __getitem__(self,i):return torch.zeros(3,32,32),self.targets[i]


class TaskEvaluationTest(unittest.TestCase):
    def test_known_task_denominators(self):
        groups={'task_000_001':(0,1),'task_001_002':(1,2)}
        model=create_model('test_linear',3,allow_test_model=True,task_class_sets=groups)
        with torch.no_grad():
            for p in model.parameters():p.zero_()
        result=evaluate_task_conditioned(model,SimpleNamespace(validation=Dataset()),
                                         groups,2,torch.device('cpu'))
        self.assertEqual(result['macro_head_accuracy'],.5)
        self.assertEqual(result['per_head']['task_000_001']['total'],4)
        self.assertEqual(result['per_head']['task_001_002']['total'],4)
