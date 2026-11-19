"""Completed CIFAR runs must retain valid per-sync accuracy evidence."""
import json
from pathlib import Path
import pytest
from cafcl.stage_report import assert_sync_reports_complete

def add(tmp,stage,version,seen,integrated,accuracy):
    p=tmp/'per_sync_accuracy'
    p.mkdir(parents=True,exist_ok=True)
    name=f'sync_{stage:03d}'
    content=dict(round=stage,version=version,observed_classes=seen,
                 server_integrated_classes=integrated,test_accuracy=accuracy,
                 test_only_no_feedback=True)
    (p/(name+'.json')).write_text(json.dumps(content))
    (p/(name+'.csv')).write_text('class_id,test_accuracy\n0,0.5\n')
    (p/(name+'.md')).write_text('# Stage\n')
    (p/(name+'.png')).write_bytes(b'PNG fixture')
    return dict(model_version=version,observed_classes=seen,
                globally_integrated_classes=integrated,per_sync_test_accuracy=accuracy)

def test_all_stages_and_terminal_flush_are_required(tmp_path):
    rows=[add(tmp_path,1,1,[0,1],[0,1],.5),
          add(tmp_path,2,2,[0,1,2],[0,1],.3),
          add(tmp_path,3,3,[0,1,2],[0,1,2],.4)]
    assert assert_sync_reports_complete(tmp_path,rows)==3
    (tmp_path/'per_sync_accuracy'/'sync_003.csv').unlink()
    with pytest.raises(RuntimeError,match='Missing or empty'):
        assert_sync_reports_complete(tmp_path,rows)

@pytest.mark.parametrize('field,value',[
    ('round',22),('version',7),('observed_classes',[1]),
    ('server_integrated_classes',[1]),('test_accuracy',.8),
    ('test_only_no_feedback',False)])
def test_corrupt_stage_evidence_is_rejected(tmp_path,field,value):
    rows=[add(tmp_path,1,1,[0,1],[0],.4)]
    p=tmp_path/'per_sync_accuracy'/'sync_001.json'
    record=json.loads(p.read_text())
    record[field]=value
    p.write_text(json.dumps(record))
    with pytest.raises(RuntimeError):
        assert_sync_reports_complete(tmp_path,rows)

def test_empty_history_and_bad_json_fail_closed(tmp_path):
    with pytest.raises(RuntimeError,match='No committed'):
        assert_sync_reports_complete(tmp_path,[])
    rows=[add(tmp_path,1,1,[0,1],[0,1],.5)]
    (tmp_path/'per_sync_accuracy'/'sync_001.json').write_text('{broken')
    with pytest.raises(RuntimeError,match='Invalid per-sync'):
        assert_sync_reports_complete(tmp_path,rows)
