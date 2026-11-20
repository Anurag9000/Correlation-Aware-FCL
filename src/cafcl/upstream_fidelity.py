"""Fail-closed scientific source-method fidelity gate.

Importing a backbone or finding source code on disk does not establish that an
experiment executes the upstream method. These contracts name exact core
entry points. They are deliberately NOT automatically upgraded by a passing
unit test or a comment in a source file: only a reviewed source-to-runner
integration plus independent tests may change their status.

The four 2026-11-20 pilots use different algorithms from the papers. Running
them as paper comparisons without explicit opt-in is now disallowed.
"""
from __future__ import annotations

from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

# Each item is (upstream source file, upstream symbol, actual state of pilot).
# 'imported' means only the published backbone/classifier is reused.
# 'adapted' means an altered replacement is executed.
# 'omitted' means the published algorithmic function is not invoked.
# 'upstream_missing' means the authors' repository does not ship a dependency.
COMPONENTS={
 'glfc':(
  ('GLFC/src/ResNet.py','resnet18_cbam','imported'),
  ('GLFC/src/GLFC.py','GLFC_model.beforeTrain','omitted'),
  ('GLFC/src/GLFC.py','GLFC_model.entropy_signal','omitted'),
  ('GLFC/src/GLFC.py','GLFC_model.update_new_set','omitted'),
  ('GLFC/src/GLFC.py','GLFC_model._compute_loss','original_invoked'),
  ('GLFC/src/GLFC.py','GLFC_model.efficient_old_class_weight','original_invoked'),
  ('GLFC/src/GLFC.py','GLFC_model._construct_exemplar_set','omitted'),
  ('GLFC/src/GLFC.py','GLFC_model.proto_grad_sharing','omitted'),
  ('GLFC/src/GLFC.py','GLFC_model.prototype_mask','omitted'),
  ('GLFC/src/ProxyServer.py','proxyServer.reconstruction','omitted'),
  ('GLFC/src/ProxyServer.py','proxyServer.monitor','omitted'),
  ('GLFC/src/ProxyServer.py','proxyServer.model_back','omitted'),
  ('GLFC/src/Fed_utils.py','FedAvg','original_invoked'),
 ),
 'lga':(
  ('LGA/ResNet.py','resnet18_cbam','imported'),
  ('LGA/LGA.py','LGA.beforeTrain','omitted'),
  ('LGA/LGA.py','LGA.entropy_signal','omitted'),
  ('LGA/LGA.py','LGA.update_new_set','omitted'),
  ('LGA/LGA.py','LGA._compute_loss','original_invoked'),
  ('LGA/LGA.py','LGA.efficient_old_class_weight','original_invoked'),
  ('LGA/LGA.py','LGA._construct_exemplar_set','omitted'),
  ('LGA/LGA.py','LGA.proto_grad_sharing','omitted'),
  ('LGA/LGA.py','LGA.prototype_mask','omitted'),
  ('LGA/myNetwork.py','network.compute_radius','omitted'),
  ('LGA/ProxyServer.py','proxyServer.reconstruction','omitted'),
  ('LGA/ProxyServer.py','proxyServer.f_monitor','omitted'),
  ('LGA/ProxyServer.py','proxyServer.model_back','omitted'),
  ('LGA/Fed_utils.py','FedAvg','original_invoked'),
 ),
 'feat':(
  ('FEAT/convs/resnet.py','resnet18','imported'),
  ('FEAT/methods/FEAT.py','gsa_distill_loss','original_invoked'),
  ('FEAT/utils/inc_net.py','fixed_ETF_geometry','paper_reconstruction'),
  ('FEAT/methods/FEAT.py','FLDataSelector.mask_local_data','omitted'),
  ('FEAT/methods/FEAT.py','FLDataSelector.compute_local_leverage_scores','omitted'),
  ('FEAT/methods/FEAT.py','FLDataSelector.aggregate_leverage_scores','omitted'),
  ('FEAT/methods/FEAT.py','FLDataSelector.sample_data','omitted'),
  ('FEAT/methods/FEAT.py','FEAT._select_data_for_retention','omitted'),
  ('FEAT/methods/FEAT.py','FEAT._local_update','omitted'),
  ('FEAT/methods/FEAT.py','FEAT._local_finetune','omitted'),
  ('FEAT/methods/FEAT.py','estimate_ecc_priors','original_invoked'),
  ('FEAT/methods/FEAT.py','aggregate_ecc_priors_from_clients','original_invoked'),
  ('FEAT/methods/FEAT.py','_build_projector_from_W','original_invoked'),
  ('FEAT/methods/FEAT.py','FEAT._fl_train','omitted'),
  ('FEAT/utils/inc_net.py','IncrementalNet_CausalETF','upstream_missing'),
  ('FEAT/methods/GSA.py','GSA','upstream_missing'),
  ('FEAT/methods/EGC.py','EGC','upstream_missing'),
 ),
 'fedclass':(
  ('FedCLASS/convs/resnet_client.py','resnet8','imported'),
  ('FedCLASS/convs/linears.py','SimpleLinear','imported'),
  ('FedCLASS/models/ce.py','CE._train','original_invoked'),
  ('FedCLASS/models/base.py','BaseLearner.gen_cur_classes','adapted'),
  ('FedCLASS/models/base.py','BaseLearner.build_rehearsal_memory','adapted'),
  ('FedCLASS/models/base.py','BaseLearner._construct_exemplar_unified','original_invoked'),
  ('FedCLASS/utils/inc_net.py','CENet.update_fc','omitted'),
  ('FedCLASS/utils/fed_algorithm.py','FedAvg.update','original_invoked'),
  ('FedCLASS/fl_trainer.py','FlTrainer.fl_train','omitted'),
 ),
}

SOURCE_KIND={
 'glfc':'authors_published_GLFC',
 'lga':'authors_published_LGA',
 'feat':'authors_published_FEAT_incomplete_dependencies',
 'fedclass':'authors_published_CE_FedAvg_baseline_not_FedCLASS_OURS',
}
METHODS=tuple(COMPONENTS)


def fidelity_report(suite):
    """Machine-readable honest ledger; does not certify model accuracy."""
    if suite not in COMPONENTS:
        raise ValueError(f'Unknown suite: {suite}')
    entries=[]
    for path,symbol,status in COMPONENTS[suite]:
        present=(ROOT/path).is_file()
        entries.append(dict(source_path=path,source_symbol=symbol,
                            pilot_status=status,source_present=present))
    missing=[e['source_path'] for e in entries if not e['source_present']]
    return dict(
        suite=suite,reference=SOURCE_KIND[suite],
        verified_paper_faithful=False,
        implemented_original_entries=sum(e['pilot_status']=='original_invoked' for e in entries),
        required_original_entries=len(entries),
        verified_full_method_coverage=False,
        missing_original_dependencies=missing,
        components=entries,
        remediation='Execute all original method-specific components, not merely imported source/backbones; audit source-to-runtime call graph and numerical/behavioral parity before certification.',
    )


def require_verified(suite, *, permit_legacy=False):
    report=fidelity_report(suite)
    if report['verified_paper_faithful']:
        return report
    if permit_legacy:
        return report
    missing=', '.join(report['missing_original_dependencies'])
    reason=f'; source missing: {missing}' if missing else ''
    raise RuntimeError(
        f'{suite.upper()} is an INCOMPLETE adapted pilot, not an original-paper-'
        f'faithful implementation{reason}. Refusing to launch new scientifically '
        'mislabelled training. Read docs/ORIGINAL_SOURCE_FIDELITY_AUDIT.md. '
        'Only for explicitly labelled legacy ablation runs, use '
        '--permit-adapted-legacy (does NOT make it faithful).')


def require_all_verified(*, permit_legacy=False):
    for suite in METHODS:
        require_verified(suite,permit_legacy=permit_legacy)
