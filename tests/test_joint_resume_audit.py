import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from cafcl.joint.curriculum import compile_plan


class ResumeAuditTests(unittest.TestCase):
    def test_resume_offline_cross_semantic(self):
        import torch
        from cafcl.config import RunConfig
        from cafcl.data import get_data_bank
        from cafcl import simulator
        from cafcl.joint.knowledge_audit import audit_knowledge
        from cafcl.joint.analysis import analyze_run
        bank=get_data_bank('synthetic','data',download=False,seed=0)
        # Client 1 carries one stale update, and two labels recur in distinct tasks.
        events=[(0,0,(0,1)),(0,1,(1,2)),(1,0,(2,3)),(2,0,(0,2)),(2,1,(0,1))]
        plan=compile_plan(events,bank.train.targets,num_clients=2,num_classes=10,
            interval_boundaries=[1,3],upload_cohorts=[(0,),(0,1)],
            require_all_classes=False,head_policy='semantic_shared',
            raw_sample_overlap=.5,allow_cross_semantic_reuse=True).to_dict()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'plan.json';path.write_text(json.dumps(plan))
            cfg=RunConfig(dataset='synthetic',model='test_smallcnn',num_clients=2,
                stream_protocol=True,local_tasks_per_round=2,batch_size=8,epochs=1,
                stages=1,rounds_per_stage=1,method='replay',participation=1.,
                sample_overlap_fraction=.5,client_training_concurrency=1,memory_per_class=2,
                joint_plan_path=str(path),joint_plan_hash=plan['plan_hash'],
                joint_local_evaluation=True,job_id='resume',checkpoint_every=1)
            simulator.run_experiment(cfg,str(root/'continuous'),download=False,verbose=False)
            original=simulator.save_round_state
            def interrupted(*args,**kwargs):
                original(*args,**kwargs)
                if Path(args[0]).name=='checkpoint_latest.pt':raise RuntimeError('injected interruption')
            with patch.object(simulator,'save_round_state',side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError,'injected interruption'):
                    simulator.run_experiment(cfg,str(root/'interrupted'),download=False,verbose=False)
            simulator.run_experiment(cfg,str(root/'interrupted'),download=False,verbose=False)
            a=root/'continuous'/'resume';b=root/'interrupted'/'resume'
            left=torch.load(a/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            right=torch.load(b/'checkpoint_latest.pt',map_location='cpu',weights_only=False)
            for key in left['global_state']:
                self.assertTrue(torch.equal(left['global_state'][key],right['global_state'][key]),key)
            self.assertEqual(left['server_cl_state'].joint_occurrence_receipts,right['server_cl_state'].joint_occurrence_receipts)
            self.assertEqual(audit_knowledge(a,plan),audit_knowledge(b,plan))
            summary=analyze_run(b,figures=False)
            self.assertEqual(summary['knowledge_ledger']['acknowledged'],5)
            self.assertEqual(summary['knowledge_ledger']['pending'],0)
