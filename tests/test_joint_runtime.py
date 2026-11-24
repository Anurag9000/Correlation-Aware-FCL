"""CPU integration contracts. Run in the repository's torch environment."""
import json
from pathlib import Path
import tempfile
import unittest

from cafcl.joint.curriculum import compile_plan


class JointRuntimeTests(unittest.TestCase):
    def test_plan_json_roundtrip(self):
        from cafcl.joint.adapter import load_plan
        targets=[c for c in range(10) for _ in range(20)]
        plan=compile_plan([(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3))],
                          targets,num_clients=2,num_classes=10,local_depth=2,
                          require_all_classes=False,head_policy='semantic_shared').to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan))
            loaded=load_plan(path,plan['plan_hash'],targets)
            self.assertEqual(loaded['plan_hash'],plan['plan_hash'])
            with self.assertRaises(ValueError):
                load_plan(path,'0'*64,targets)
            with self.assertRaises(ValueError):
                load_plan(path,plan['plan_hash'],list(reversed(targets)))

    def test_explicit_class_head(self):
        import torch
        from cafcl.models import create_model
        model=create_model('test_smallcnn',10,allow_test_model=True,learning_setting='class_il')
        self.assertEqual(tuple(model(torch.zeros(2,3,32,32)).shape),(2,10))
        self.assertFalse(hasattr(model.classifier,'task_classes'))

    def test_synthetic_two_local_tasks(self):
        from cafcl.config import RunConfig
        from cafcl.data import get_data_bank
        from cafcl.simulator import run_experiment
        bank=get_data_bank('synthetic','data',download=False,seed=0)
        events=[(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3))]
        for setting in ('task_il','class_il'):
            with self.subTest(setting=setting), tempfile.TemporaryDirectory() as tmp:
                plan=compile_plan(events,bank.train.targets,num_clients=2,num_classes=10,
                                  local_depth=2,require_all_classes=False,
                                  head_policy='semantic_shared').to_dict()
                path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan))
                cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=2,
                              stream_protocol=True,local_tasks_per_round=2,
                              batch_size=8,epochs=1,stages=1,rounds_per_stage=1,
                              method='replay',aggregator='fedavg',participation=1.,
                              memory_per_class=2,client_training_concurrency=1,
                              learning_setting=setting,joint_plan_path=str(path),
                              joint_plan_hash=plan['plan_hash'],joint_local_evaluation=True,
                              job_id='smoke',checkpoint_every=1)
                run_experiment(cfg,tmp,download=False,verbose=False)
                self.assertTrue((Path(tmp)/'smoke'/'summary.json').exists())
                self.assertTrue((Path(tmp)/'smoke'/'joint_plan.json').exists())


if __name__=='__main__':
    unittest.main()
