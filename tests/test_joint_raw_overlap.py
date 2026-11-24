import unittest
from cafcl.joint.curriculum import compile_plan,exact_groups


class RawOverlapTests(unittest.TestCase):
    def test_fixed_draws_and_realized_overlap(self):
        labels=[c for c in range(10) for _ in range(40)]
        events=exact_groups((2,),width=2,steps=5,num_classes=10)
        for fraction in (0,.25,.5,.75,1):
            p=compile_plan(events,labels,num_clients=2,num_classes=10,
                raw_sample_overlap=fraction,head_policy='semantic_shared')
            self.assertEqual(p.allocation_audit['assignments'],400)
            a,b=p.tasks[:2]
            self.assertEqual(len(a.train_indices),40)
            self.assertEqual(len(b.train_indices),40)
            self.assertEqual(len(set(a.train_indices)&set(b.train_indices)),int(40*fraction))
            self.assertEqual(a.head_id,b.head_id)
            self.assertEqual(p.allocation_audit['unique_assigned'],400-int(200*fraction))

    def test_reject_future_head_ambiguity(self):
        with self.assertRaises(ValueError):
            compile_plan([(0,0,(0,1)),(0,1,(1,2))],[c for c in range(3) for _ in range(40)],
                num_clients=2,num_classes=3,head_policy='semantic_shared',raw_sample_overlap=.5)
