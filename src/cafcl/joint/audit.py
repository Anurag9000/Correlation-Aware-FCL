"""Read-only communication boundary verification, including tensor payloads."""
from hashlib import sha256
import json
import torch


def _hash_value(h,value):
    def frame(blob):
        h.update(len(blob).to_bytes(8,'big'))
        h.update(blob)
    frame(type(value).__name__.encode())
    if torch.is_tensor(value):
        tensor=value.detach().cpu().contiguous()
        frame(str(tensor.dtype).encode())
        frame(str(tuple(tensor.shape)).encode())
        frame(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value,dict):
        frame(str(len(value)).encode())
        for key in sorted(value,key=repr):
            _hash_value(h,key)
            _hash_value(h,value[key])
    elif isinstance(value,(tuple,list,set,frozenset)):
        frame(str(len(value)).encode())
        for item in (sorted(value,key=repr) if isinstance(value,(set,frozenset)) else value):
            _hash_value(h,item)
    elif value is None or isinstance(value,(str,int,float,bool)):
        frame(json.dumps(value,allow_nan=False).encode())
    else:
        raise TypeError(f'Unsupported audited server-state value: {type(value)!r}')


def server_boundary_digest(weights,version,state):
    h=sha256()
    _hash_value(h,int(version))
    _hash_value(h,weights)
    _hash_value(h,vars(state))
    return h.hexdigest()
