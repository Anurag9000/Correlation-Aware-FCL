"""Invoke authors' GLFC_model.train and LGA.train source AST directly.

Their original SGD initialization, per-epoch LR schedule, BCE/teacher loss,
model/device conversion and zero_grad/backward/step instructions are
unchanged. Only the stated 2-class sparse local stream, fixed-batch64 source
loader, actual local/received replay, and pre-existing teacher bridge are
adapted. The original proxy server/model_back is still missing; a stored
prior local teacher is used as an explicitly NON-faithful interim branch,
and the complete experiment remains fail-closed.
"""
from __future__ import annotations
from collections import Counter
import copy

import torch

from .upstream_glfc_lga_memory import original_before_local_task, _CompactLogits


def train_original_glfc_lga(model,client,bank,cfg,device,round_id,task_classes,
                            *,experience_end=False,active_pair_classes=None):
    if client.shared_weights is None:
        raise RuntimeError('Client source model has not synchronized initially')
    model.load_state_dict(client.shared_weights)
    model.to(device)
    client.last_learning_method=cfg.method
    _,source=original_before_local_task(
        cfg.method,model,client,bank,cfg,device,active_pair_classes,
        return_source=True)
    order=tuple(source.source_global_order)
    teacher=None
    old_classes=tuple(sorted(set(client.native_teacher_classes) & set(order)))
    if old_classes and client.native_teacher is not None:
        teacher_core=copy.deepcopy(model)
        teacher_core.load_state_dict(client.native_teacher)
        teacher_core.to(device).eval()
        for param in teacher_core.parameters():
            param.requires_grad_(False)
        teacher=_CompactLogits(teacher_core,old_classes).to(device).eval()
    # model_back is original proxy server's two selected teacher snapshots;
    # that mechanism is NOT yet wired. Both slots use the last legit local
    # teacher as an explicitly labelled protocol-compatibility approximation.
    source.train(round_id,[teacher,teacher])
    if experience_end and source.signal:
        # The actual 50-step author mask and original auxiliary encoder
        # gradients are generated only when its unchanged entropy trigger
        # fires. Disconnected clients hold receipts privately until upload.
        from .upstream_native_prototypes import source_prototype_gradients
        receipts=source_prototype_gradients(
            cfg.method,source,cfg,client,model,device)
        client.pending_native_prototypes.extend(receipts)
    steps=int(cfg.epochs)*len(source.train_loader)
    if steps<=0:
        raise RuntimeError('Original GLFC/LGA train performed no updates')
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if any(not torch.isfinite(v.float()).all() for v in state.values()):
        raise RuntimeError('Authors GLFC/LGA original SGD produced nonfinite weights')
    client.native_teacher=state.copy()
    client.native_teacher_classes=set(order)
    client.optimizer_state=None  # published SGD fresh per source train invocation
    client.train_steps+=steps
    client.trained_classes.update(int(bank.train.targets[i]) for i in client.current_indices)
    client.activated_labels.update(order)
    counts=dict(Counter(int(bank.train.targets[i]) for i in client.current_indices))
    return dict(
        weights=state,examples=len(source.source_train_identifiers),
        class_counts=counts,class_update_steps={int(c):steps for c in order},
        active_pair_classes=list(active_pair_classes or ()),
        steps=steps,loss=None,
        constraint_violations=0,qp_calls=0,projection_norm_last=0.,
        memory_samples=sum(map(len,client.memory.values())),
        method_used=cfg.method,fedprox_mu=0.,
        local_old_loss_before=None,local_old_loss_after=None,
        optimizer_state_policy='authors_fresh_per_task',
        sgd_momentum=0.0,sgd_weight_decay=.00001,
        classification_competitor_classes=list(order),
        source_method=('GLFC/src/GLFC.py:GLFC_model.train'
                       if cfg.method=='native_glfc'
                       else 'LGA/LGA.py:LGA.train'),
        source_memory='authors.update_new_set',
        source_teacher='LOCAL_TEACHER_APPROX_NOT_ORIGINAL_PROXY_MODEL_BACK',
    )
