import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from types import SimpleNamespace
import json
from cafcl.joint.runner import plan_campaign
from cafcl.joint.essential import essential_designs
from cafcl.joint.campaign import Ledger


class EssentialTests(unittest.TestCase):
    def test_exact_60_unique_seed_zero_jobs(self):
        bank=SimpleNamespace(train=SimpleNamespace(targets=[c for c in range(100) for _ in range(450)]),
                             coarse_by_fine={c:c//5 for c in range(100)})
        with TemporaryDirectory() as tmp,patch('cafcl.data.get_data_bank',return_value=bank):
            jobs=plan_campaign(tmp,'data',0,False,suite='essential')
            self.assertEqual(len(jobs),60)
            self.assertEqual(Counter(j['family'] for j in jobs),
                             {'smoke':8,'H':36,'D':2,'E':2,'N':4,'controls':8})
            self.assertEqual(Counter(j['learning_setting'] for j in jobs),
                             {'task_il':30,'class_il':30})
            self.assertEqual({j['seed'] for j in jobs},{0})
            with self.assertRaises(ValueError):plan_campaign(tmp,'data',1,False,suite='essential')

    def test_ledger_transitions(self):
        with TemporaryDirectory() as tmp:
            job=dict(job_id='x',family='smoke',learning_setting='task_il',seed=0,
                     epochs=1,plan_hash='x')
            ledger=Ledger(tmp,[job])
            ledger.update({'job_id':'x','status':'running','attempt':1})
            ledger.update({'job_id':'x','status':'artifact_verified','test_accuracy':.5})
            result=json.loads((Path(tmp)/'job_ledger.json').read_text())
            self.assertEqual(result['counts'],{'artifact_verified':1})
            self.assertEqual(result['jobs'][0]['test_accuracy'],.5)
            self.assertTrue((Path(tmp)/'job_ledger.csv').exists())
