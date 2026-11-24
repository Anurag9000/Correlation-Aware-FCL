"""Reproducible source, dependency and raw dataset receipts."""
from hashlib import sha256
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import json
import platform
import subprocess
import numpy as np
import torch


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def runtime_identity():
    package=Path(__file__).resolve().parent
    source={str(p.relative_to(package)):sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*.py'))}
    result={'python':platform.python_version(),'source_sha256':digest(source),
            'dependencies':{},'cuda_build':torch.version.cuda}
    for name in ('torch','torchvision','numpy','scipy','pillow'):
        try:result['dependencies'][name]=version(name)
        except PackageNotFoundError:result['dependencies'][name]=None
    try:
        result['git_revision']=subprocess.check_output(['git','rev-parse','HEAD'],
            cwd=package,stderr=subprocess.DEVNULL,text=True).strip()
        diff=subprocess.check_output(['git','diff','HEAD','--binary'],
            cwd=package,stderr=subprocess.DEVNULL)
        result['tracked_diff_sha256']=sha256(diff).hexdigest()
    except (OSError,subprocess.CalledProcessError):
        result['git_revision']=None
        result['tracked_diff_sha256']=None
    return result


def dataset_identity(bank):
    from .data import source_indices
    bases={}
    splits={}
    for name in ('train','validation','test'):
        view=getattr(bank,name)
        if view is None:
            splits[name]=None
            continue
        base=view
        while hasattr(base,'underlying') and hasattr(base,'indices'):
            base=base.underlying
        key=id(base)
        if key not in bases:
            raw=getattr(base,'data',None)
            if raw is None:
                raise ValueError('Dataset identity requires immutable raw data')
            if torch.is_tensor(raw):raw=raw.detach().cpu().numpy()
            raw=np.ascontiguousarray(raw)
            h=sha256()
            h.update(str(raw.dtype).encode())
            h.update(str(raw.shape).encode())
            h.update(memoryview(raw).cast('B'))
            bases[key]={'raw_sha256':h.hexdigest(),
                        'raw_targets_sha256':digest(list(map(int,base.targets))),
                        'transform':repr(getattr(base,'transform',None))}
        splits[name]={**bases[key],
                      'source_indices_sha256':digest(source_indices(view)),
                      'targets_sha256':digest(list(map(int,view.targets))),
                      'samples':len(view)}
    return splits
