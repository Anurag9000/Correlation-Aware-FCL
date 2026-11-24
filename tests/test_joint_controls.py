import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.controls import control_designs


class ControlTests(unittest.TestCase):
    def test_control_plan_semantics(self):
        targets=[c for c in range(100) for _ in range(450)]
        designs=list(control_designs())
        self.assertEqual(len(designs),36)
        for d in designs:
            p=compile_plan(d.events,targets,num_clients=d.num_clients,num_classes=100,
                local_depth=d.local_depth,upload_cohorts=d.cohorts,terminal_flush=d.terminal_flush,
                control_kind=d.control_kind)
            self.assertEqual(p.allocation_audit['unique_assigned'],45000)
            if d.control_kind=='local_only':
                self.assertTrue(all(not r.upload_clients for r in p.intervals))
            elif d.control_kind=='centralized_sequential':
                self.assertTrue(all(not r.upload_clients for r in p.intervals[:-1]))
                self.assertEqual(p.intervals[-1].upload_clients,(0,))

    def test_central_and_local_runtime(self):
        from cafcl.config import RunConfig
        from cafcl.data import get_data_bank
        from cafcl.simulator import run_experiment
        bank=get_data_bank('synthetic','data',download=False,seed=0)
        for kind,events,k,flush in [
            ('local_only',[(0,0,(0,1)),(0,1,(0,1)),(1,0,(1,2)),(1,1,(2,3))],2,False),
            ('centralized_sequential',[(0,0,(0,1)),(1,0,(1,2))],1,True)]:
            with self.subTest(kind=kind),TemporaryDirectory() as tmp:
                plan=compile_plan(events,bank.train.targets,num_clients=k,num_classes=10,
                    local_depth=2,upload_cohorts=[()],terminal_flush=flush,
                    require_all_classes=False,control_kind=kind,head_policy='semantic_shared').to_dict()
                path=Path(tmp)/'plan.json';path.write_text(json.dumps(plan))
                cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=k,
                    stream_protocol=True,local_tasks_per_round=2,batch_size=8,epochs=1,
                    stages=1,rounds_per_stage=1,method='replay',participation=1.,
                    client_training_concurrency=1,memory_per_class=2,learning_setting='class_il',
                    joint_plan_path=str(path),joint_plan_hash=plan['plan_hash'],
                    joint_control_kind=kind,stream_terminal_flush=flush,job_id='control')
                run_experiment(cfg,tmp,download=False,verbose=False)
                final=json.loads((Path(tmp)/'control'/'joint_final_private_test.json').read_text())
                self.assertEqual(final['server_version'],int(flush))
                self.assertEqual(len(final['client_metrics']),k)
