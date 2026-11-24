import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from cafcl.joint.checkpoint_io import durable_torch_save,job_lease


class CheckpointIOTests(unittest.TestCase):
    def test_failed_serialization_preserves_last_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'checkpoint.pt'
            durable_torch_save({'version':1},path)
            def fail(payload,handle):
                handle.write(b'incomplete');raise OSError('injected disk failure')
            with patch('cafcl.joint.checkpoint_io.torch.save',side_effect=fail):
                with self.assertRaises(OSError):durable_torch_save({'version':2},path)
            self.assertEqual(torch.load(path,weights_only=True),{'version':1})
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
            durable_torch_save({'version':2},path)
            self.assertEqual(torch.load(path,weights_only=True),{'version':2})

    def test_job_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            with job_lease(tmp):
                with self.assertRaises(RuntimeError):
                    with job_lease(tmp):pass
            with job_lease(tmp):pass
