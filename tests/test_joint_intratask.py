import json
import tempfile
import unittest
from pathlib import Path
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.intratask import intratask_designs


class IntrataskTests(unittest.TestCase):
    def test_parent_identity_and_allocation(self):
        labels=[c for c in range(10) for _ in range(40)]
        events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3))]
        base=compile_plan(events,labels,num_clients=2,num_classes=10,
                          head_policy='semantic_shared',require_all_classes=False).to_dict()
        split=compile_plan(events,labels,num_clients=2,num_classes=10,
                          head_policy='semantic_shared',require_all_classes=False,epoch_segments=2).to_dict()
        meta=split['allocation_audit']['epoch_slicing']
        self.assertEqual(meta['parent_atomic_hash'],base['atomic_hash'])
        parents={x['occurrence_id']:x for x in base['tasks']}
        for frag in split['tasks']:
            parent=parents[meta['fragment_metadata'][frag['occurrence_id']]['parent_occurrence_id']]
            self.assertEqual(frag['train_indices'],parent['train_indices'])
            self.assertEqual(frag['head_id'],parent['head_id'])
        self.assertEqual(len(split['intervals']),5)
        self.assertEqual(len(list(intratask_designs())),24)

    def test_no_memory_insertion_until_parent_end(self):
        from cafcl.config import RunConfig
        from cafcl.data import get_data_bank
        from cafcl.simulator import run_experiment
        from cafcl.joint.knowledge_audit import audit_knowledge
        bank=get_data_bank('synthetic','data',download=False,seed=0)
        for setting in ('task_il','class_il'):
            with self.subTest(setting=setting),tempfile.TemporaryDirectory() as tmp:
                events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3))]
                plan=compile_plan(events,bank.train.targets,num_clients=2,num_classes=10,
                    require_all_classes=False,head_policy='semantic_shared',epoch_segments=2).to_dict()
                path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan))
                cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=2,
                    stream_protocol=True,local_tasks_per_round=1,batch_size=8,epochs=1,
                    stages=1,rounds_per_stage=1,method='replay',participation=1.,
                    client_training_concurrency=1,memory_per_class=2,
                    learning_setting=setting,joint_plan_path=str(path),joint_plan_hash=plan['plan_hash'],
                    joint_epoch_segments=2,job_id='intra')
                run_experiment(cfg,tmp,download=False,verbose=False)
                root=Path(tmp)/'intra'
                first=json.loads((root/'joint_interval_00000.json').read_text())['local_occurrences']
                second=json.loads((root/'joint_interval_00001.json').read_text())['local_occurrences']
                self.assertTrue(all(x['memory_samples']==0 for x in first))
                self.assertTrue(all(x['memory_samples']>0 for x in second))
                self.assertTrue(all(x['local_experience']==0 for x in first+second))
                self.assertEqual(audit_knowledge(root,plan)['acknowledged'],8)
