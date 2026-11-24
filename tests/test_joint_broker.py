import tempfile
import unittest
from pathlib import Path
from cafcl.joint.broker import lease,gate,select_jobs


class BrokerTests(unittest.TestCase):
    def test_exclusive_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lease'
            with lease(path):
                with self.assertRaises(RuntimeError):
                    with lease(path):pass
            with lease(path):pass

    def test_hardware_gate(self):
        self.assertEqual(gate({'free_mib':100,'compute_pids':[]},200,False),'insufficient_gpu_memory')
        self.assertEqual(gate({'free_mib':300,'compute_pids':[5]},200,False),'unrelated_gpu_processes')
        self.assertIsNone(gate({'free_mib':300,'compute_pids':[5]},200,True))

    def test_priority_and_alias_family(self):
        a=dict(job_id='a',families=['A','H'],communication_intervals=10,task_occurrences=20)
        b=dict(job_id='b',families=['H'],communication_intervals=2,task_occurrences=4)
        self.assertEqual(select_jobs({'jobs':[a,b]},'H',1),[b])
        self.assertEqual(select_jobs({'jobs':[a,b]},'A'),[a])
        with self.assertRaises(ValueError):select_jobs({'jobs':[a,a]})
