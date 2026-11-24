"""Read-only communication boundary verification."""
from hashlib import sha256
import json


def server_boundary_digest(weights,version,state):
    h=sha256()
    h.update(str(int(version)).encode())
    for name,value in sorted(weights.items()):
        tensor=value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    metadata={
        'archive_version':state.version,
        'memory':state.memory,
        'acknowledged_indices':sorted(state.acknowledged_train_indices),
        'integrated_classes':sorted(state.integrated_classes),
        'class_sample_exposure':state.class_sample_exposure,
        'joint_source_heads':{i:sorted(v) for i,v in state.joint_source_heads.items()},
        'joint_occurrence_receipts':state.joint_occurrence_receipts,
    }
    h.update(json.dumps(metadata,sort_keys=True).encode())
    return h.hexdigest()
