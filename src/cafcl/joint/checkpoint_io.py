"""Durable checkpoint replacement and exclusive runner job lease."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import os
import tempfile

import torch


@contextmanager
def job_lease(directory):
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    with (root/'.joint_job.lock').open('a+') as handle:
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('Joint job already running: '+str(root)) from exc
        try:yield
        finally:fcntl.flock(handle,fcntl.LOCK_UN)


def durable_torch_save(payload,path):
    """Keep old complete checkpoint if serialization fails before replace.

    Atomicity assumes filesystem rename/fsync semantics. This is not a
    distributed transaction or a guarantee against arbitrary FUSE failure.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            torch.save(payload,handle)
            handle.flush();os.fsync(handle.fileno())
        os.replace(tmp,path)
        dfd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(dfd)
        finally:os.close(dfd)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
