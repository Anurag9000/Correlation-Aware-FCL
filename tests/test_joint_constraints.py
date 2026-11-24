import unittest
from itertools import combinations
from collections import Counter
from cafcl.joint.constraints import solve_overlap,recurrence_gap
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.extensions import extension_designs


class ConstraintTests(unittest.TestCase):
    def test_hyperedges_make_clique_feasible(self):
        edges={e:1 for e in combinations(range(5),2)}
        tasks=solve_overlap((2,)*5,edges,6)
        self.assertEqual(len(set.intersection(*(set(x) for x in tasks))),1)
        self.assertEqual(len(set.union(*(set(x) for x in tasks))),6)
        with self.assertRaises(ValueError):
            solve_overlap((2,)*5,edges,5)

    def test_weighted_allocation(self):
        events=[(0,0,(0,1)),(0,1,(0,1))]
        labels=[0]*40+[1]*40
        p=compile_plan(events,labels,num_clients=2,num_classes=2,
            allocation_weights={'0:0:0':3,'0:0:1':3})
        a,b=p.tasks
        self.assertEqual(len(a.train_indices)+len(b.train_indices),80)
        self.assertGreater(len(a.train_indices),len(b.train_indices))
        self.assertFalse(set(a.train_indices)&set(b.train_indices))
        self.assertEqual(p.allocation_audit['unique_assigned'],80)

    def test_recurrence_gap(self):
        self.assertEqual(recurrence_gap([(0,0,(0,1)),(3,1,(1,2))],(0,0),(3,1),exact=False,shared=1),2)

    def test_extension_designs(self):
        targets=[c for c in range(100) for _ in range(450)]
        designs=list(extension_designs())
        self.assertEqual(len({d.name for d in designs}),len(designs))
        for d in designs:
            with self.subTest(d=d.name):
                p=compile_plan(d.events,targets,num_clients=d.num_clients,num_classes=100,
                    local_depth=d.local_depth,head_policy=d.head_policy,
                    allocation_weights=d.allocation_weights)
                self.assertEqual(p.allocation_audit['unique_assigned'],45000)
