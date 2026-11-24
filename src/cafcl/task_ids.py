"""Stable identities for client task-head occurrences."""
from __future__ import annotations


def task_instance_id(client_id, round_id, ordinal, classes):
    """Identify one client task occurrence, including its class set.

    Identical class sets from different clients or timesteps remain distinct
    task IDs, as required by the task-incremental protocol.
    """
    values=tuple(sorted(set(map(int,classes))))
    if not values or values[0]<0:
        raise ValueError('Task IDs require nonnegative class labels')
    if int(client_id)<0 or int(round_id)<0 or int(ordinal)<0:
        raise ValueError('Task ID coordinates must be nonnegative')
    class_part='_'.join(f'{value:03d}' for value in values)
    return (f'task_{class_part}__client_{int(client_id):03d}'
            f'__round_{int(round_id):05d}__ordinal_{int(ordinal):03d}')
