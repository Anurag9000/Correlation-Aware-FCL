import unittest
from types import SimpleNamespace
from cafcl.joint.provenance import expose,visible_source_heads,prepare_receipts,install_receipts,download_provenance
from cafcl.joint.curriculum import compile_plan


def client(cid):
    return SimpleNamespace(client_id=cid,joint_local_source_heads={},joint_received_source_heads={},
        joint_pending_occurrences=[],local_training_events=[],received_memory={})


class ProvenanceTests(unittest.TestCase):
    def test_future_and_offline_heads(self):
        a,b=client(0),client(1)
        state=SimpleNamespace(joint_source_heads={},joint_occurrence_receipts={})
        expose(a,[7],'task_000_001')
        expose(b,[7],'task_001_002')
        self.assertEqual(visible_source_heads(a)[7],('task_000_001',))
        b.joint_pending_occurrences=['B']
        b.local_training_events=[dict(occurrence_id='B',atomic_step=2,semantic_task_id='B',
            task_id='task_001_002',classes=[1,2],round=1)]
        receipts=prepare_receipts(state,{0:a,1:b},[{'client_id':1}],1,2)
        install_receipts(state,receipts,{0:a,1:b},[{'client_id':1}])
        self.assertEqual(visible_source_heads(a)[7],('task_000_001',))
        a.received_memory={1:[7]};download_provenance(state,a)
        self.assertEqual(visible_source_heads(a)[7],('task_000_001','task_001_002'))
        with self.assertRaises(RuntimeError):
            prepare_receipts(state,{0:a,1:b},[{'client_id':1}],2,3)

    def test_cross_semantic_explicit_opt_in(self):
        p=compile_plan([(0,0,(0,1)),(0,1,(1,2))],
            [c for c in range(3) for _ in range(40)],num_clients=2,num_classes=3,
            raw_sample_overlap=.5,allow_cross_semantic_reuse=True,head_policy='semantic_shared')
        self.assertEqual(len(set(p.tasks[0].train_indices)&set(p.tasks[1].train_indices)),10)
        self.assertTrue(p.allocation_audit['allow_cross_semantic_reuse'])
