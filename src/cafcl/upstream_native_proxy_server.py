"""Server lifecycle for the authors' original GLFC/LGA proxy machinery.

This module is called ONLY after accepted-client model uploads and source
gradient receipts have been committed by FederatedCLState.merge_clients.

The published proxyServer.dataloader performs 250-step LBFGS inversion
for every accepted prototype-gradient receipt, reconstructs the original
20 images/receipt, monitors the current global model and updates the
two original best-model teacher slots. This module uses that EXACT
original function body, via the source-AST proxy class.

Original proxy data construction assumes at least 64 reconstructed images
for the drop_last=True monitor. The user's 2-class stream can yield only
40. We buffer ACKNOWLEDGED gradients until >=4 receipts; this is a
necessary explicitly noncanonical class-stream protocol guard, and
the fidelity gate remains CLOSED. An offline client's unacknowledged
gradients never enter this buffer. All results are resumable.
"""
from __future__ import annotations

import copy
import torch


def source_proxy_server_step(state,method,server_model,candidate,cfg,device,round_id):
    if method not in {'native_glfc','native_lga'}:return None
    from .upstream_native_proxy import (
        construct_source_proxy,source_gradient_labels,proxy_snapshot)
    newly_accepted=[copy.deepcopy(state.native_prototype_gradients[key])
                    for key in state.native_latest_accepted_prototypes]
    # Validate every newly accepted original source gradient BEFORE it
    # enters durable proxy history; unselected hidden clients are absent.
    if newly_accepted:
        source_gradient_labels(method,newly_accepted,device)
    pool=copy.deepcopy(state.native_proxy_pending_receipts)+newly_accepted
    for receipt in pool:
        if receipt.get('method')!=method:
            raise RuntimeError('Foreign method in original proxy gradient backlog')
    if pool:
        source_gradient_labels(method,pool,device)

    previous=state.native_proxy_state
    eligible=len(pool)>=4 # original reconstruction yields 20 images/receipt
    has_monitor=bool(previous and
         int(previous.get('reconstructed_images',0))>=64)

    if not eligible and not has_monitor:
        # The original drop_last monitor has no batch for <64 images.
        # Do NOT claim to have executed the original reconstruction.
        state.native_proxy_pending_receipts=pool
        state.native_proxy_history.append(dict(
            round=int(round_id),newly_accepted=len(newly_accepted),
            buffered=len(pool),ran_source_dataloader=False,
            reason='original_monitor_drop_last_64_needs_4_prototypes'))
        return None

    model=copy.deepcopy(server_model)
    model.load_state_dict(candidate)
    model.to(device).eval()
    proxy=construct_source_proxy(
        method,model,cfg.seed,device,
        prior=previous,radius=state.native_lga_radius,
        ep_g=round_id,lr=cfg.lr)

    if eligible:
        # NOTE: these are actual original author encoder gradients from
        # source_proto_gradients, never raw unaccepted images.
        grads=[[g.detach().to(device) for g in receipt['gradients']]
               for receipt in pool]
        if method=='native_glfc':
            proxy.dataloader(grads)
        else:
            proxy.dataloader(grads,proxy.model)
        state.native_proxy_pending_receipts=[]
    else:
        # The source's empty-pool branch re-monitors the existing
        # reconstructed proxy dataset against the new global model.
        if method=='native_glfc':
            proxy.dataloader([])
        else:
            proxy.dataloader([],proxy.model)
        state.native_proxy_pending_receipts=pool

    result=proxy_snapshot(proxy)
    if result['reconstructed_images']<64:
        raise RuntimeError('Authors proxy monitor received fewer than 64 images')
    result['source_method']=method
    result['server_round']=int(round_id)
    result['source_provenance']='authors_original_proxyServer.dataloader'
    result['experimental_buffered_receipts']=int(len(pool)) if eligible else 0
    state.native_proxy_state=result
    state.native_proxy_history.append(dict(
        round=int(round_id),newly_accepted=len(newly_accepted),
        buffered=len(state.native_proxy_pending_receipts),
        ran_source_dataloader=True,
        source_reconstruction=bool(eligible),
        reconstructed_images=int(result['reconstructed_images']),
        best_perf=float(result['best_perf'])))
    return result


def original_source_teacher_slots(client,model,device,order,active_pair):
    """Build the two ORIGINAL proxy/model_back teacher slots for source.train.

    This replaces the earlier [local_teacher, local_teacher] placeholder.
    A slot can be None, as in the original source before proxy monitoring
    has selected one. The actual GLFC/LGA original train() function chooses
    between these exact slots according to the entropy trigger.
    """
    from .upstream_glfc_lga_memory import _CompactLogits
    saved=client.received_native_proxy
    if not saved:return [None,None]
    known=set(int(v) for v in client.native_teacher_classes)
    novel=set(map(int,active_pair or ()))
    prior=tuple(int(c) for c in order if int(c) in known and int(c) not in novel)
    teachers=[]
    for name in ('best_model_1','best_model_2'):
        weights=saved.get(name)
        if weights is None or not prior:
            teachers.append(None)
            continue
        teacher=copy.deepcopy(model)
        teacher.load_state_dict(weights)
        teacher.to(device).eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        teachers.append(_CompactLogits(teacher,prior).to(device).eval())
    return teachers
