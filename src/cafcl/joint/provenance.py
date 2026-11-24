"""Event-causal sample/head and occurrence provenance.

Only locally delivered events and acknowledged downloaded receipts authorize
replay heads. Planned future associations are never consulted by local_train.
"""
from collections import defaultdict


def expose(client, indices, head):
    if not head:
        raise ValueError('Task-IL source exposure needs an active head')
    for index in indices:
        client.joint_local_source_heads.setdefault(int(index),set()).add(head)


def visible_source_heads(client):
    merged=defaultdict(set)
    for table in (client.joint_local_source_heads,client.joint_received_source_heads):
        for index,heads in table.items():
            merged[int(index)].update(heads)
    return {i:tuple(sorted(heads)) for i,heads in merged.items()}


def prepare_receipts(state,clients,accepted,interval,version):
    receipts={}
    for upload in accepted:
        cid=upload['client_id'];client=clients[cid]
        events={e['occurrence_id']:e for e in client.local_training_events}
        for oid in client.joint_pending_occurrences:
            if oid in state.joint_occurrence_receipts or oid in receipts:
                raise RuntimeError('Duplicate occurrence acknowledgement: '+oid)
            if oid not in events:
                raise RuntimeError('Unrecognized pending occurrence: '+oid)
            event=events[oid]
            receipts[oid]={'client':cid,'atomic_step':event['atomic_step'],
                'semantic_task_id':event['semantic_task_id'],'head_id':event['task_id'],
                'classes':event['classes'],'trained_interval':event['round'],
                'accepted_interval':interval,'accepted_server_version':version,
                **{k:event[k] for k in ('parent_occurrence_id','parent_atomic_step','segment_index','segment_count','experience_end') if k in event}}
        if not client.joint_pending_occurrences:
            raise RuntimeError('Upload has no pending atomic occurrence')
    return receipts


def install_receipts(state,receipts,clients,accepted):
    for upload in accepted:
        client=clients[upload['client_id']]
        for idx,heads in client.joint_local_source_heads.items():
            state.joint_source_heads.setdefault(idx,set()).update(heads)
    state.joint_occurrence_receipts.update(receipts)


def download_provenance(state,client):
    # Only transfer source heads for raw examples in the acknowledged archive.
    # A raw index that reoccurs under a new task cannot acquire that head until
    # the client encounters it or a server receipt really arrives.
    ids={int(i) for values in client.received_memory.values() for i in values}
    client.joint_received_source_heads={i:set(state.joint_source_heads[i])
        for i in ids if i in state.joint_source_heads}
