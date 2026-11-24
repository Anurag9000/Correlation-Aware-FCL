import hashlib
import json
from pathlib import Path

root = Path('/dev/shm/cafcl-ag-recovery-v4')
queue_path = root / 'queue.json'
queue = json.loads(queue_path.read_text())
row = next(x for x in queue['jobs'] if x['cell_id'] == 'B_recur_gap_5_seed0')
report_path = root / 'specs/B_recur_gap_5_seed0/runtime_plan_validation.json'
report_bytes = report_path.read_bytes()
report = json.loads(report_bytes)
assert report['status'] == 'runner_plan_validated'
assert report['run_id'] == row['run_id']
assert report['learning_syncs'] == 18 and report['arrivals'] == 90
row['runtime_plan_validation'] = report
row['runtime_plan_validation_sha256'] = hashlib.sha256(report_bytes).hexdigest()
queue_path.write_text(json.dumps(queue, indent=2, sort_keys=True) + '\n')
print(json.dumps({'updated_cell': row['cell_id'], 'run_id': row['run_id'],
                  'runtime_plan_validation_sha256': row['runtime_plan_validation_sha256'],
                  'queue_sha256': hashlib.sha256(queue_path.read_bytes()).hexdigest()}, indent=2))
