import json
import tempfile
import unittest
from pathlib import Path
from cafcl.joint.curriculum import compile_plan


class PooledTests(unittest.TestCase):
    def test_both_learning_settings(self):
        from cafcl.config import RunConfig
        from cafcl.data import get_data_bank
        from cafcl.simulator import run_experiment
        from cafcl.joint.knowledge_audit import audit_knowledge
        bank=get_data_bank('synthetic','data',download=False,seed=0)
        events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3)),
                (2,0,(0,1)),(2,1,(0,1)),(3,0,(1,2)),(3,1,(2,3))]
        for setting in ('task_il','class_il'):
            with self.subTest(setting=setting),tempfile.TemporaryDirectory() as tmp:
                plan=compile_plan(events,bank.train.targets,num_clients=2,num_classes=10,
                    local_depth=2,require_all_classes=False,head_policy='semantic_shared',
                    control_kind='simultaneous_access').to_dict()
                path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan))
                cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=2,
                    stream_protocol=True,local_tasks_per_round=2,batch_size=8,epochs=1,
                    stages=1,rounds_per_stage=1,method='replay',participation=1.,
                    client_training_concurrency=1,memory_per_class=2,
                    learning_setting=setting,joint_plan_path=str(path),joint_plan_hash=plan['plan_hash'],
                    joint_control_kind='simultaneous_access',joint_local_evaluation=True,job_id='pooled')
                run_experiment(cfg,tmp,download=False,verbose=False)
                root=Path(tmp)/'pooled'
                trace=json.loads((root/'joint_interval_00000.json').read_text())
                self.assertEqual(len(trace['atomic_barriers']),1)
                self.assertEqual(trace['atomic_barriers'][0]['atomic_steps'],[0,1])
                groups=[x['pooled_group_training'] for x in trace['local_occurrences'] if x['pooled_group_training']]
                self.assertEqual(len(groups),2)
                self.assertTrue(all(g['replay_sampled_examples']==0 for g in groups))
                if setting=='task_il':
                    self.assertTrue(all(len(g['task_head_examples'])==2 for g in groups))
                self.assertEqual(audit_knowledge(root,plan)['acknowledged'],8)
