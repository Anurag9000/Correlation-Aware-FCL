import unittest
from collections import Counter
from cafcl.joint.interactions import interaction_designs,short_designs
from cafcl.joint.curriculum import compile_plan


class InteractionTests(unittest.TestCase):
    def test_nominal_counts(self):
        designs=list(interaction_designs())
        self.assertEqual(Counter(x.family for x in designs),
            {'X1':54,'X2':36,'X3':18,'X4':24,'X5':24,'X6':16,'X7':18,'X8':27})
        self.assertEqual(len({x.name for x in designs}),217)

    def test_short_compiles(self):
        targets=[c for c in range(100) for _ in range(450)]
        for d in short_designs():
            with self.subTest(d=d.name):
                p=compile_plan(d.events,targets,num_clients=d.num_clients,num_classes=100,
                    local_depth=d.local_depth,head_policy=d.head_policy,
                    require_all_classes=False,allocation_order=d.allocation_order)
                self.assertLess(len(p.overlap_audit['class_owners']),100)
