import unittest
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.family_catalog import screening_designs,sequence_order,topology_stream


class JointCatalogTests(unittest.TestCase):
    def test_catalog_compilation(self):
        targets=[c for c in range(100) for _ in range(450)]
        designs=list(screening_designs())
        self.assertEqual(len({d.name for d in designs}),len(designs))
        for d in designs:
            with self.subTest(d=d.name):
                p=compile_plan(d.events,targets,num_clients=d.num_clients,num_classes=100,
                    local_depth=d.local_depth,interval_boundaries=d.boundaries,
                    upload_cohorts=d.cohorts,offline_clients=d.offline_clients,
                    allocation_order=d.allocation_order,raw_sample_overlap=d.raw_sample_overlap,
                    head_policy=d.head_policy,allow_cross_semantic_reuse=d.allow_cross_semantic_reuse)
                self.assertEqual(p.allocation_audit['assignments'],45000)
                if not d.raw_sample_overlap:
                    self.assertEqual(p.allocation_audit['unique_assigned'],45000)

    def test_order_matched_samples(self):
        targets=[c for c in range(100) for _ in range(450)]
        refs=[]
        for policy in ('aligned','reversed','cyclic','random'):
            events,ends=sequence_order(policy,5)
            p=compile_plan(events,targets,num_clients=5,num_classes=100,
                           interval_boundaries=ends,allocation_order='client_semantic')
            refs.append({(t.client_id,t.semantic_task_id):t.train_indices for t in p.tasks})
        self.assertTrue(all(r==refs[0] for r in refs))

    def test_topology(self):
        edges={(0,1),(0,2),(0,3),(0,4)}
        events=topology_stream(edges,4,steps=1)
        classes={i:set(labels) for _,i,labels in events}
        for i in range(5):
            for j in range(i+1,5):
                self.assertEqual(len(classes[i]&classes[j]),int((i,j) in edges))
