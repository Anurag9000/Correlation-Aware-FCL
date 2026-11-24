import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cafcl.joint.curriculum import compile_plan
from cafcl.joint.runner import run_job


class RunnerEntryTests(unittest.TestCase):
    def test_public_entry_and_configuration_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plan=compile_plan([(0,0,(0,1)),(0,1,(0,1))],
                [0]*40+[1]*40,num_clients=2,num_classes=100,
                require_all_classes=False,head_policy='semantic_shared').to_dict()
            (root/'plan.json').write_text(json.dumps(plan))
            job={'job_id':'entry','plan_file':'plan.json','plan_hash':plan['plan_hash'],
                'learning_setting':'class_il','seed':0,'epochs':1,'allow_partial_coverage':True}
            manifest=root/'manifest.json';manifest.write_text(json.dumps({'jobs':[job]}))
            def fake_run(cfg,output,data_root,download):
                cfg.validate()
                target=Path(output)/cfg.job_id
                (target/'config.json').write_text(json.dumps(cfg.to_dict()))
                return {'status':'completed'}
            with patch('cafcl.simulator.run_experiment',side_effect=fake_run) as mock:
                result=run_job(manifest,'entry',root/'results','data',False,
                               checkpoint_policy='latest')
                self.assertEqual(result['status'],'completed')
                self.assertEqual(mock.call_args.args[0].joint_checkpoint_policy,'latest')
                with self.assertRaisesRegex(RuntimeError,'configuration differs'):
                    run_job(manifest,'entry',root/'results','data',False)
