import unittest
import torch
from cafcl.joint.update_diagnostics import update_geometry


class UpdateGeometryTests(unittest.TestCase):
    def test_exact_cancellation(self):
        model=torch.nn.Linear(2,1,bias=False)
        base={'weight':torch.zeros(1,2)}
        u=[{'client_id':0,'examples':1,'weights':{'weight':torch.tensor([[1.,0.]])}},
           {'client_id':1,'examples':1,'weights':{'weight':torch.tensor([[-1.,0.]])}}]
        result=update_geometry(model,base,u)
        self.assertEqual(result['all']['cosines']['0:1'],-1.)
        self.assertEqual(result['all']['reference_cancellation_ratio'],0.)
