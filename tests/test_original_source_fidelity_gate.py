"""Fail-closed source-to-runtime paper-fidelity contracts."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from cafcl.upstream_fidelity import (
    METHODS, ROOT, COMPONENTS, fidelity_report, require_verified,
    require_all_verified,
)


def test_all_four_methods_declared_and_no_misleading_certification():
    assert set(METHODS)=={'glfc','lga','feat','fedclass'}
    for method in METHODS:
        report=fidelity_report(method)
        assert report['verified_paper_faithful'] is False
        assert report['verified_full_method_coverage'] is False
        assert report['required_original_entries']==len(COMPONENTS[method])
        assert all(item['source_path'] and item['source_symbol'] and item['pilot_status']
                   for item in report['components'])


def test_author_upstream_is_present_but_feat_critical_imports_are_missing():
    for method in ('glfc','lga','fedclass'):
        assert not fidelity_report(method)['missing_original_dependencies']
    feat=fidelity_report('feat')
    assert set(feat['missing_original_dependencies'])=={
        'FEAT/utils/inc_net.py','FEAT/methods/GSA.py','FEAT/methods/EGC.py'}
    assert (ROOT/'FEAT/methods/FEAT.py').is_file()
    assert (ROOT/'FedCLASS/models/ours.py').is_file()


def test_new_paper_fidelity_claims_fail_closed_but_legacy_runs_are_explicit():
    for method in METHODS:
        with pytest.raises(RuntimeError,match='INCOMPLETE adapted pilot'):
            require_verified(method)
        assert require_verified(method,permit_legacy=True)['verified_paper_faithful'] is False
    with pytest.raises(RuntimeError,match='INCOMPLETE'):
        require_all_verified()
    require_all_verified(permit_legacy=True)


def test_default_launch_prevented_before_any_cuda_or_data_download(tmp_path):
    from cafcl.native_primary import main as native
    from cafcl.fedclass_primary import main as fedclass
    for method in ('glfc','lga','feat'):
        with pytest.raises(ValueError,match='no task-head adaptation'):
            native(['--suite',method,'--output-dir',str(tmp_path)])
    with pytest.raises(RuntimeError,match='INCOMPLETE adapted pilot'):
        fedclass(['--output-dir',str(tmp_path)])
    # Native Task-IL plans are not emitted while their method adapter is
    # unsupported; source fidelity remains a separate launch gate.
    with pytest.raises(ValueError,match='no task-head adaptation'):
        native(['--suite','glfc','--plan-only','--output-dir',str(tmp_path)])
