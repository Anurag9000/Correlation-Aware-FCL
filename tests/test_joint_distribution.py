import unittest
from collections import Counter
import torch
from cafcl.joint.distribution import domain_transform
from cafcl.joint.distribution_designs import semantic_stream
from cafcl.joint.curriculum import compile_plan


class DistributionTests(unittest.TestCase):
    def test_semantic_matched_counts(self):
        coarse={c:c//5 for c in range(100)}
        a=semantic_stream(coarse,True);b=semantic_stream(coarse,False)
        self.assertEqual(Counter(c for _,_,labels in a for c in labels),
                         Counter(c for _,_,labels in b for c in labels))
        self.assertTrue(all(coarse[x]==coarse[y] for _,_,(x,y) in a))
        self.assertTrue(all(coarse[x]!=coarse[y] for _,_,(x,y) in b))
        for events in (a,b):
            plan=compile_plan(events,[c for c in range(100) for _ in range(450)],
                              num_clients=4,num_classes=100,local_depth=2)
            self.assertEqual(plan.allocation_audit['unique_assigned'],45000)

    def test_deterministic_domain(self):
        x=torch.arange(3*32*32,dtype=torch.float32).reshape(3,32,32)/1000
        a=domain_transform(x,2,0,'client_corruption',7)
        b=domain_transform(x,2,3,'client_corruption',7)
        self.assertTrue(torch.equal(a,b))
        self.assertFalse(torch.equal(a,x))
        self.assertTrue(torch.equal(domain_transform(x,0,0,'client_fixed'),x))
        self.assertFalse(torch.equal(domain_transform(x,0,5,'client_periodic'),x))
