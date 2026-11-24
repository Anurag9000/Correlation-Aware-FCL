"""Independent verification of persisted joint occurrence/communication ledgers."""
from pathlib import Path
import json


def audit_knowledge(directory,plan):
    root=Path(directory)
    expected={t['occurrence_id']:t for t in plan['tasks']}
    trained=set();acknowledged=set();pending=set();version=0
    for interval in plan['intervals']:
        r=interval['interval_id']
        row=json.loads((root/f'joint_knowledge_{r:05d}.json').read_text())
        local=row['locally_trained_occurrences']
        if len(local)!=len(set(local)) or set(local)!=set(interval['occurrences']) or set(local)&trained:
            raise ValueError(f'Incorrect trained occurrence set at interval {r}')
        trained.update(local);pending.update(local)
        cohort=set(interval['upload_clients'])
        if set(row['broadcast_clients'])!=cohort:raise ValueError('Broadcast cohort differs from plan')
        uploaded=set();senders=set()
        for upload in row['uploads']:
            cid=upload['client'];oids=upload['occurrences']
            if cid not in cohort or cid in senders or upload['steps']<=0 or upload['anchor_version']>version:
                raise ValueError('Invalid upload receipt')
            if len(oids)!=len(set(oids)) or uploaded.intersection(oids):
                raise ValueError('Duplicate upload occurrence')
            if not set(oids)<=pending or any(expected[oid]['client_id']!=cid for oid in oids):
                raise ValueError('Upload contains untrained or foreign occurrence')
            uploaded.update(oids);senders.add(cid)
        version+=int(bool(uploaded))
        receipts=row['acknowledged_occurrences']
        if set(receipts)!=uploaded or acknowledged&uploaded:
            raise ValueError('Acknowledgement does not match unique accepted uploads')
        for oid,receipt in receipts.items():
            task=expected[oid]
            if (receipt['client']!=task['client_id'] or receipt['atomic_step']!=task['atomic_step']
                    or receipt['semantic_task_id']!=task['semantic_task_id']
                    or receipt['accepted_server_version']!=version
                    or receipt['accepted_interval']!=r
                    or receipt['trained_interval']>r):
                raise ValueError('Invalid occurrence acknowledgement metadata')
        acknowledged.update(uploaded);pending-=uploaded
        pending_by_client=row['pending_occurrences']
        actual=[]
        for cid,oids in pending_by_client.items():
            if any(expected[oid]['client_id']!=int(cid) for oid in oids):
                raise ValueError('Pending occurrence attributed to wrong client')
            actual.extend(oids)
        if len(actual)!=len(set(actual)) or set(actual)!=pending:
            raise ValueError('Lost or duplicated pending occurrence')
        if row['committed_version']!=version or row['server_acknowledged_total']!=len(acknowledged):
            raise ValueError('Server commit/acknowledgement counter mismatch')
    if trained!=set(expected):raise ValueError('Incomplete training event coverage')
    if plan['intervals'][-1]['terminal_flush'] and pending:
        raise ValueError('Terminal flush left unacknowledged occurrences')
    return {'trained':len(trained),'acknowledged':len(acknowledged),
            'pending':len(pending),'commits':version}
