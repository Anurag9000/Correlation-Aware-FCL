import unittest

from cafcl.joint.curriculum import (
    compile_plan, exact_groups, explicit_sequences, foundational_registry,
    foundational_stream, shared_anchor_stream,
)


class JointCurriculumTests(unittest.TestCase):
    def setUp(self):
        self.targets = [c for c in range(100) for _ in range(450)]

    def test_foundational_matching(self):
        for s in ("S1", "S2", "S3"):
            for o in (0, 1):
                events = foundational_stream(s, o)
                plans = [compile_plan(events, self.targets, num_clients=5,
                                      num_classes=100, local_depth=d,
                                      head_policy="semantic_shared")
                         for d in (1, 2, 3, 5)]
                self.assertEqual(len({p.to_dict()["atomic_hash"] for p in plans}), 1)
                self.assertEqual([len(p.intervals) for p in plans], [101, 51, 35, 21])
                for p in plans:
                    self.assertEqual(p.allocation_audit["unique_assigned"], 45000)
                    self.assertEqual(set(p.overlap_audit["class_occurrences"].values()), {10})
                    self.assertEqual(p.overlap_audit["adjacent_local_intersection_histogram"],
                                     {str(o): 495})
                    self.assertTrue(p.intervals[-1].terminal_flush)
                    self.assertFalse(p.intervals[-1].occurrences)

    def test_exact_and_partial(self):
        p = compile_plan(exact_groups((2, 3), width=2, steps=1), self.targets,
                         num_clients=5, num_classes=100, require_all_classes=False,
                         head_policy="semantic_shared")
        self.assertEqual(p.overlap_audit["classifier_heads"], 2)
        p = compile_plan(shared_anchor_stream(num_clients=5, width=5, shared=2,
                                              steps=1), self.targets,
                         num_clients=5, num_classes=100, require_all_classes=False)
        self.assertEqual(p.overlap_audit["spatial_intersection_histogram"], {"2": 10})

    def test_delayed_sync_and_heterogeneous_depth(self):
        events = explicit_sequences({0: [(0,1),(1,2)], 1: [(0,1)]})
        p = compile_plan(events, self.targets, num_clients=2, num_classes=100,
                         local_depth=2, require_all_classes=False,
                         upload_cohorts=[(0,)])
        self.assertEqual(len(p.intervals), 2)
        self.assertEqual(len(p.intervals[0].occurrences), 3)
        self.assertEqual(p.intervals[0].upload_clients, (0,))
        self.assertEqual(p.intervals[-1].upload_clients, (0,1))

    def test_reject_disjoint_or_insufficient_support(self):
        with self.assertRaises(ValueError):
            compile_plan([(0,0,(0,1)),(0,1,(2,3))], self.targets,
                         num_clients=2,num_classes=100,require_all_classes=False)
        with self.assertRaises(ValueError):
            compile_plan([(0,0,(0,1)),(0,1,(0,1))], [0,1],
                         num_clients=2,num_classes=100,require_all_classes=False)

    def test_registry(self):
        self.assertEqual(len(list(foundational_registry())), 48)


if __name__ == "__main__":
    unittest.main()
