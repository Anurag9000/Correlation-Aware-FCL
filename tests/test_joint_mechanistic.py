import unittest
from types import SimpleNamespace
import torch
from cafcl.models import create_model
from cafcl.joint.mechanistic import probe_aggregation


class Dataset(torch.utils.data.Dataset):
    targets=[0,1,2,0,1,2]
    def __len__(self):return len(self.targets)
    def __getitem__(self,i):return torch.full((3,32,32),i/10),self.targets[i]


class MechanisticTests(unittest.TestCase):
    def test_task_and_class_probes_restore_model(self):
        groups={'task_000_001':(0,1),'task_001_002':(1,2)}
        for setting in ('task_il','class_il'):
            with self.subTest(setting=setting):
                model=create_model('test_linear',3,allow_test_model=True,
                    task_class_sets=groups if setting=='task_il' else None,learning_setting=setting)
                base={k:v.clone() for k,v in model.state_dict().items()}
                after={k:v.clone() for k,v in base.items()}
                result=probe_aggregation(model,SimpleNamespace(validation=Dataset()),torch.device('cpu'),
                    base,after,[],setting,{0,1,2},{0,1},{'task_000_001'},
                    [{'head_id':'task_001_002','classes':[1,2]}],groups)
                self.assertEqual(result['split'],'validation')
                self.assertAlmostEqual(result['panels']['current']['actual_loss_change'],0.)
                self.assertEqual(result['panels']['current']['committed_alignment']['all']['first_order_loss_change'],0.)
                self.assertTrue(all(torch.equal(v,after[k]) for k,v in model.state_dict().items()))
                self.assertTrue(all(p.grad is None for p in model.parameters()))
