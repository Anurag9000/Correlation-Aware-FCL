"""Continuous on-device continual learning with conditional federated communication.

Every client trains locally every logical round after its data arrive. A selected
client uploads accumulated private learning, then receives the committed server
model; inactive clients never synchronize unless a full-sync ablation requests
it explicitly. All client state is durable across checkpoints.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from .config import RunConfig
from .data import get_data_bank,compile_scenario
from .streaming import compile_stream_scenario,select_offline_cohort
from .models import create_model,copy_cpu_state,check_finite
from .continual import LocalState,local_train
from .class_indexed import classwise_merge,classifier_parameter_names
from .federated_cl import FederatedCLState
from concurrent.futures import ThreadPoolExecutor
from .parallel_clients import create_client_runtimes,train_round_task_barriers
from .selection import select_clients
from .evaluation import evaluate
from .causal import aggregation_leave_one_out,aggregation_subset_report
from collections import Counter
from .diagnostics import (diagnostic_gradient_relationships,
    class_gradient_representations, arrived_feature_means, pairwise_vector_cosines)
from .task_ids import task_instance_id


# Changing the semantics of an existing job must never silently reuse its old
# success markers or checkpoints, even though its scientific config is unchanged.
PROTOCOL_VERSION = 'cifar100_native32_v27_chain20_random_offline_parallel5_classuniform_sampledmixed_source_task_replay'

def protocol_version_for(cfg):
    if getattr(cfg,'joint_plan_path',''):
        return 'joint_spatiotemporal_fcl_v1_'+cfg.learning_setting
    suffix=[]
    if cfg.learning_setting=='task_il':suffix.append('taskhead_source_provenance')
    if cfg.stream_sync_every_batch:suffix.append('stream_batch_sync_padpoolfix')
    if cfg.stream_partial_batch_policy!='include_padding':suffix.append('streampartial_'+cfg.stream_partial_batch_policy)
    if cfg.replay_padding_policy!='repeat_chunk':suffix.append('replaypad_'+cfg.replay_padding_policy)
    return PROTOCOL_VERSION + (('_'+'_'.join(suffix)) if suffix else '')


def _normalize_saved_config(config):
    """Migrate config fields whose new defaults preserve old semantics."""
    normalized=dict(config)
    normalized.setdefault('replay_buffer_order_policy','archive_order')
    normalized.setdefault('replay_update_policy','exhaustive')
    for k,v in (('replay_padding_policy','repeat_chunk'),('stream_partial_batch_policy','include_padding'),('stream_sync_every_batch',False)):normalized.setdefault(k,v)
    return normalized   # legacy defaults


def _assign_task_ids(scenario,bank,cfg):
    """Give every client task occurrence a unique head ID and label map."""
    from collections import defaultdict
    task_class_map={}
    task_first_round={}
    arrivals=list(scenario.arrivals)
    if cfg.stream_protocol:
        assigned=[]
        for arrival in arrivals:
            if not arrival.labels:continue
            classes=tuple(sorted(set(map(int,arrival.labels))))
            task_id=(arrival.task_id or task_instance_id(
                arrival.client_id,arrival.round_id,arrival.stage,classes))
            task_class_map[task_id]=classes
            task_first_round[task_id]=min(task_first_round.get(task_id,int(arrival.round_id)),
                                          int(arrival.round_id))
            assigned.append(replace(arrival,task_id=task_id))
        arrivals=assigned
    else:
        groups=defaultdict(list)
        for position,arrival in enumerate(arrivals):
            groups[(int(arrival.round_id),int(arrival.client_id))].append((position,arrival))
        assigned=[None]*len(arrivals)
        for (round_id,client_id),events in sorted(groups.items()):
            ordered=sorted(events,key=lambda item:(int(item[1].stage),item[1].labels,
                                                   item[1].indices[:1]))
            for ordinal,start in enumerate(range(0,len(ordered),2)):
                pair=ordered[start:start+2]
                classes=tuple(sorted({int(cls) for _,event in pair for cls in event.labels}))
                if len(classes)<2 and cfg.class_pair_protocol=='strict':
                    raise ValueError(f'Task-IL replay provenance contains singleton task {classes}')
                task_id=task_instance_id(client_id,round_id,ordinal,classes)
                task_class_map[task_id]=classes
                task_first_round[task_id]=min(task_first_round.get(task_id,round_id),round_id)
                for position,event in pair:
                    assigned[position]=replace(event,task_id=task_id)
        arrivals=assigned
    annotated=replace(scenario,arrivals=arrivals)
    provenance=defaultdict(set)
    for arrival in arrivals:
        if not arrival.task_id:continue
        for index in arrival.indices:
            provenance[int(index)].add(arrival.task_id)
    return (annotated,{index:tuple(sorted(tasks)) for index,tasks in provenance.items()},
            task_class_map,task_first_round)


def _integrated_task_class_sets(task_ids_by_index, acknowledged_train_indices,
                                task_class_map):
    """Return unique task heads represented by server-acknowledged examples."""
    if task_ids_by_index is None:return {}
    ids={task_id
        for index in acknowledged_train_indices
        for task_id in task_ids_by_index.get(int(index),())}
    return {task_id:task_class_map[task_id] for task_id in sorted(ids)}


def _classwise_fedavg_contribution(uploads):
    """Example-weight fractions for class-specific local updates.

    This is an exact explanation of canonical FedAvg weights for fresh
    synchronized inputs. It is a REFERENCE diagnostic, not the weighting
    of repository-specific custom server methods. 'Direct' refers to
    examples with that true class; 'head' includes seen-class softmax
    competition and replay-driven classifier row updates.
    """
    total=sum(int(u['examples']) for u in uploads)
    if total<=0:return {}
    classes={int(c) for u in uploads
             for c in set(u.get('class_counts',{}))|set(u.get('class_update_steps',{}))}
    result={}
    for cls in sorted(classes):
        direct=sum(int(u['examples']) for u in uploads
                   if int(u.get('class_counts',{}).get(cls,0))>0)
        head=sum(int(u['examples']) for u in uploads
                 if int(u.get('class_update_steps',{}).get(cls,0))>0)
        result[str(cls)]={'direct_new_class_example_weight_fraction':direct/total,
                          'updated_classifier_row_weight_fraction':head/total,
                          'uploader_count':sum(1 for u in uploads
                              if int(u.get('class_update_steps',{}).get(cls,0))>0)}
    return result


def _normalization_aggregation_diagnostics(previous,committed,uploads):
    """Read-only BN-buffer spread and classifier row drift.

    BatchNorm running statistics are not optimizer-gradient parameters;
    client-specific disjoint-class BN estimates may diverge substantially.
    These statistics do not modify FedAvg, synchronization or any predictor.
    """
    bn_names=[key for key in committed
              if key.endswith('running_mean') or key.endswith('running_var')]
    spread=[]
    drift=[]
    for key in bn_names:
        after=committed[key].detach().float().cpu()
        before=previous[key].detach().float().cpu()
        drift.append(float(torch.linalg.vector_norm(after-before)))
        for u in uploads:
            value=u['weights'][key].detach().float().cpu()
            spread.append(float(torch.linalg.vector_norm(value-after)))
    name='classifier.bias' if 'classifier.bias' in committed else None
    classifier_shift=None
    if name is not None:
        classifier_shift=float(torch.linalg.vector_norm(
            committed[name].float().cpu()-previous[name].float().cpu()))
    return dict(batchnorm_running_buffer_keys=len(bn_names),
                bn_mean_client_to_server_l2=(
                    sum(spread)/len(spread) if spread else None),
                bn_mean_server_drift_l2=(sum(drift)/len(drift) if drift else None),
                classifier_bias_update_l2=classifier_shift)


def anchored_upload(client, global_state, *, current_version=None):
    """Rebase delayed parameter updates without treating buffers as parameters.

    For a client last synchronized at anchor A with current private model L,
    a delayed learned-parameter contribution is G+alpha*(L-A), where alpha
    decays with server-version lag. BatchNorm running moments and other buffers
    are state, not learned parameters, so a stale client cannot rebase them;
    retain the current server values for those keys. Fresh uploads remain the
    exact local model and follow canonical FedAvg aggregation.
    Neither model weights nor privacy-sensitive raw memory are sent separately.
    """
    if client.shared_weights is None or client.sync_anchor is None:
        return None
    if client.pending_upload_steps <= 0:
        return None
    stale=((current_version is not None and client.model_version<current_version) or
           (current_version is None and any(not torch.equal(client.sync_anchor[key],gv)
                                           for key,gv in global_state.items())))
    if current_version is not None and client.model_version>current_version:
        raise RuntimeError('Client model version ahead of server')
    staleness=max(0,int(current_version-client.model_version)) if current_version is not None else int(stale)
    stale_scale=1.0/(1.0+staleness) if stale else 1.0
    if not stale and not all(torch.equal(client.sync_anchor[key],gv)
                             for key,gv in global_state.items()):
        raise RuntimeError('Fresh FedAvg client anchor differs from current global model')
    candidate={}
    for key,gv in global_state.items():
        lv=client.shared_weights[key]
        anchor=client.sync_anchor[key]
        if lv.shape != gv.shape or anchor.shape != gv.shape:
            raise RuntimeError('Client/server state shape mismatch')
        if not stale:
            candidate[key]=lv.detach().cpu().clone()   # exact canonical FedAvg model
        elif key.endswith('running_mean') or key.endswith('running_var') or key.endswith('num_batches_tracked'):
            # BN running moments are distribution estimates rather than
            # optimizer parameters. Never rebase them as G+(L-A).
            candidate[key]=gv.detach().cpu().clone()
        elif gv.is_floating_point():
            # Delayed FedAvg adaptation, attenuated by server-version lag.
            candidate[key]=(gv.to(torch.float64)+stale_scale*(lv.to(torch.float64)
                            -anchor.to(torch.float64))).to(gv.dtype)
        else:
            candidate[key]=gv.detach().cpu().clone() if stale else lv.detach().cpu().clone()
    if not check_finite(candidate):
        raise RuntimeError('Nonfinite accumulated client update')
    task_head_examples=dict(client.pending_task_head_examples)
    return {'weights':candidate,'examples':client.pending_upload_examples,
            'class_counts':dict(client.pending_upload_class_counts),
            'class_update_steps':dict(client.pending_class_update_steps),
            'task_head_examples':task_head_examples,
            'accumulated_steps':client.pending_upload_steps,
            'accumulated_classes':sorted(client.pending_upload_classes),
            'anchor_version':client.model_version,
             'stale_rebased':bool(stale),
             'staleness_versions':staleness,
             'stale_update_scale':stale_scale,
             'aggregation_input_semantics':('stale_delta_damped_bn_safe' if stale else 'fresh_local_model')}


def _guard_single_stale_terminal_flush(candidate, base, uploads, cfg, *, terminal_flush):
    """Limit a sole stale uploader's influence at the communication-only flush."""
    if not (terminal_flush and cfg.aggregator=='fedavg' and len(uploads)==1
            and uploads[0].get('stale_rebased',False)):
        return candidate,1.0
    scale=1.0/max(1,int(cfg.num_clients))
    guarded={}
    for key,previous in base.items():
        if previous.is_floating_point():
            guarded[key]=(previous.to(torch.float64)+scale*(candidate[key].to(torch.float64)
                         -previous.to(torch.float64))).to(previous.dtype)
        else:
            guarded[key]=previous.detach().clone()
    if not check_finite(guarded):
        raise RuntimeError('Nonfinite terminal stale-update guard result')
    return guarded,scale


def _merge_task_heads_fedavg(updates,base,candidate):
    # Average each task head only across clients that trained that head.
    import re
    out={k:v.clone() for k,v in candidate.items()}
    head_pattern=re.compile(r'^classifier\.heads\.(task_[A-Za-z0-9_]+)\.')
    heads=sorted({m.group(1) for key in base if (m:=head_pattern.match(key))})
    for head in heads:
        contributors=[u for u in updates if float(u.get('task_head_examples',{}).get(head,0.))>0]
        prefix=f'classifier.heads.{head}.'
        keys=[key for key in base if key.startswith(prefix)]
        if not contributors:
            for key in keys:out[key]=base[key].clone()
            continue
        denom=sum(float(u['task_head_examples'][head]) for u in contributors)
        if denom<=0:continue
        for key in keys:
            if base[key].is_floating_point():
                value=sum((u['weights'][key].to(torch.float64)*float(u['task_head_examples'][head])
                           for u in contributors),torch.zeros_like(base[key],dtype=torch.float64))
                out[key]=(value/denom).to(base[key].dtype)
            else:
                chosen=max(contributors,key=lambda u:float(u['task_head_examples'][head]))
                out[key]=chosen['weights'][key].clone()
    return out


def aggregate(updates,base,method,*,clients=None,round_id=0,old_gradient=None,class_history_counts=None,source_method=None):
    if not updates:raise RuntimeError('No client updates to aggregate')
    if method=='fedavg' and source_method is not None:
        from .original_aggregation import original_aggregate
        return original_aggregate(source_method,updates,base)
    if method=='uniform':
        weights=[1./len(updates)]*len(updates)
    else:
        counts=[u['examples'] for u in updates]
        total=sum(counts)
        if total<=0:raise RuntimeError('FedAvg has no contributing examples')
        weights=[n/total for n in counts]
    if method=='class_balanced':
        # Repository-specific history-aware algorithm, NOT canonical FedAvg.
        # Historical counts only include actually accepted uploads; a hidden
        # client's unsynchronized classes are excluded from the server ledger.
        from collections import Counter
        round_counts=Counter()
        for u in updates:round_counts.update(u['class_counts'])
        prior=class_history_counts or {}
        scores=[sum(v/(round_counts[c]+int(prior.get(c,0)))
                    for c,v in u['class_counts'].items() if v>0)
                for u in updates]
        norm=sum(scores)
        if norm<=0:raise RuntimeError('Class-balanced aggregation lacks class evidence')
        weights=[x/norm for x in scores]
    if method=='inverse_participation':
        if clients is None:raise ValueError('Client histories are required')
        # Divide by the total selection count *including this communication*.
        scores=[u['examples']/(1+clients[u['client_id']].selection_count)
                for u in updates]
        weights=[score/sum(scores) for score in scores]
    if method in {'conflict_aware','retention_aware','conflict_projection'}:
        floating=[k for k,v in base.items() if v.is_floating_point()]
        delta=[torch.cat([(u['weights'][k].detach().cpu().float()-base[k].float()).reshape(-1)
                          for k in floating]) for u in updates]
        if method=='conflict_aware':
            penalties=[]
            for i,di in enumerate(delta):
                conflict=0.
                for j,dj in enumerate(delta):
                    if i!=j:
                        denom=float(torch.linalg.vector_norm(di)*torch.linalg.vector_norm(dj))
                        if denom>1.e-12:
                            conflict+=max(0.,-float(torch.dot(di,dj))/denom)
                penalties.append(conflict)
            scores=[w/(1.+penalty) for w,penalty in zip(weights,penalties)]
            weights=[score/sum(scores) for score in scores]
        elif method=='retention_aware' and old_gradient is not None:
            g=old_gradient.detach().cpu().float()
            scores=[]
            for w,d in zip(weights,delta):
                denom=float(torch.linalg.vector_norm(g)*torch.linalg.vector_norm(d))
                conflict=float(torch.dot(g,d))/denom if denom>1.e-12 else 0.
                scores.append(w*np.exp(-max(-20.,min(20.,conflict))))
            weights=[score/sum(scores) for score in scores]
        elif method=='conflict_projection':
            # Deterministic pairwise PCGrad-style projection of client deltas.
            # Unlike reweighting, this changes client update directions.
            projected=[]
            for i,d in enumerate(delta):
                z=d.clone()
                for j,other in enumerate(delta):
                    if i==j:continue
                    dot=torch.dot(z,other)
                    denom=torch.dot(other,other)
                    if dot<0 and denom>1.e-12:z-=dot/denom*other
                projected.append(z)
            shared_update=sum(w*d for w,d in zip(weights,projected))
            out={};offset=0
            for key,baseval in base.items():
                if baseval.is_floating_point():
                    n=baseval.numel()
                    out[key]=(baseval.float().reshape(-1)+shared_update[offset:offset+n]).reshape_as(baseval).to(baseval.dtype)
                    offset+=n
                else:
                    out[key]=baseval.clone()
            if not check_finite(out):raise RuntimeError('Nonfinite gradient-corrected aggregate')
            return out
    out={}
    for key,baseval in base.items():
        values=[u['weights'][key] for u in updates]
        if not baseval.is_floating_point():
            if key.endswith('num_batches_tracked'):
                # ResNet BatchNorm counters legitimately differ across independently
                # training and intermittently disconnected clients. Keep a
                # monotone counter rather than rejecting valid updates.
                out[key]=torch.stack([v.to(torch.int64) for v in values]).max(dim=0).values.to(baseval.dtype)
                continue
            if not all(torch.equal(v,values[0]) for v in values):
                raise RuntimeError(f'Nonfloating buffer disagrees: {key}')
            out[key]=values[0].clone()
        else:
            accum=torch.zeros_like(baseval,dtype=torch.float64)
            for weight,v in zip(weights,values):
                if v.shape!=baseval.shape:raise RuntimeError(f'Shape mismatch {key}')
                accum+=weight*v.to(torch.float64)
            out[key]=accum.to(baseval.dtype)
    if not check_finite(out):raise RuntimeError('Nonfinite server aggregate')
    return out


def server_optimizer_step(base,mean,method,cfg,state):
    """FedAvgM/FedAdam server optimization applied to ALL model float tensors.

    FedAvgM: Hsu et al. Nesterov server momentum over aggregated model deltas.
    FedAdam: Reddi et al. FedOpt uncorrected moments, v_0>=tau² and
    update eta * m / (sqrt(v)+tau); not standard PyTorch Adam bias correction.
    Other aggregation methods have no adaptive server optimizer.
    """
    if method not in {'fedadam','fedavgm'}:
        return mean,state
    step=int(state.get('step',0))+1
    state={'step':step,'m':dict(state.get('m',{})),
           'v':dict(state.get('v',{}))}
    out={}
    beta1,beta2=cfg.server_beta1,cfg.server_beta2
    for name,old in base.items():
        if not old.is_floating_point():
            out[name]=mean[name].clone()
            continue
        delta=mean[name].to(torch.float64)-old.to(torch.float64)
        old_m=state['m'].get(name,torch.zeros_like(delta))
        if method=='fedavgm':
            velocity=beta1*old_m+delta
            state['m'][name]=velocity
            # Original FedAvgM's Nesterov acceleration.
            update=cfg.server_lr*(delta+beta1*velocity)
        else:
            moment=beta1*old_m+(1-beta1)*delta
            old_v=state['v'].get(name,torch.full_like(delta,float(cfg.fedadam_epsilon)**2))
            variance=beta2*old_v+(1-beta2)*delta.square()
            state['m'][name]=moment
            state['v'][name]=variance
            # FedOpt paper uses uncorrected moment estimates.
            update=cfg.server_lr*moment/(variance.sqrt()+cfg.fedadam_epsilon)
        out[name]=(old.to(torch.float64)+update).to(old.dtype)
    if not check_finite(out):raise RuntimeError('Nonfinite server update')
    return out,state

def save_round_state(path,cfg,round_id,global_state,version,clients,observed,peak,
                     integrated,history,rng,server_optimizer_state,server_class_update_counts=None,server_cl_state=None):
    """Durable full-state checkpoint for exact logical-round resumption."""
    payload={'config':cfg.to_dict(),'protocol_version':protocol_version_for(cfg),'last_completed_round':round_id,
             'global_state':global_state,'version':version,
             'clients':{cid:asdict(s) for cid,s in clients.items()},
             'observed':observed,'peak':peak,'integrated':integrated,
             'history':history,'selection_rng':rng.bit_generator.state,
             'server_optimizer_state':server_optimizer_state,
             'server_class_update_counts':dict(server_class_update_counts or {}),
             'server_cl_state':server_cl_state,
             'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
             'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)


def atomic_json(path,obj):
    path=Path(path)
    # Multiple independent CIFAR jobs may publish the same architecture
    # reference concurrently. A shared fixed temporary path lets one writer
    # replace or truncate another writer's in-progress JSON.
    tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    with tmp.open('w') as fh:json.dump(obj,fh,indent=2,sort_keys=True,default=str)
    os.replace(tmp,path)


def save_checkpoint(path,weights,cfg,version,states=None,full_state=False,server_cl_state=None):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    payload={'model_state':weights,'model_version':version,'protocol_version':protocol_version_for(cfg),
             'configuration':cfg.to_dict()}
    if full_state:
        payload['clients']={cid:asdict(s) for cid,s in states.items()}
        payload['server_cl_state']=server_cl_state
        # Retain resumable client memory and model state at full-state checkpoints.
    torch.save(payload,tmp)
    os.replace(tmp,path)


def run_experiment(cfg:RunConfig,output_root:str,data_root:str='data',download=True,verbose=True,*,fork_from_checkpoint=None):
    cfg.validate()
    if cfg.dataset!='synthetic' and (cfg.dataset!='cifar100' or not cfg.stream_protocol):
        raise ValueError('Only the CIFAR-100 v3 task-stream protocol is an active experiment; retired CIFAR-10/20 and stage-based job runs are disabled')
    job=cfg.job_id or cfg.signature()
    run_dir=Path(output_root)/job
    run_dir.mkdir(parents=True,exist_ok=True)
    config_file=run_dir/'config.json'
    status_file=run_dir/'status.json'
    if status_file.exists():
        old=json.loads(status_file.read_text())
        if old.get('status')=='completed':
            if old.get('protocol_version')!=protocol_version_for(cfg):
                raise RuntimeError('Completed run uses incompatible local-training protocol; use a new output root')
            normalized_config=json.loads(json.dumps(cfg.to_dict(),default=str))
            if not config_file.exists() or json.loads(config_file.read_text())!=normalized_config:
                raise RuntimeError(f'Completed job ID collision: {job}')
            if any(not (run_dir/name).exists() for name in
                    ('model_final.pt','summary.json','metrics.json')):
                raise RuntimeError('Completion marker exists but output artifacts are missing')
            if cfg.stream_protocol:
                from .stage_report import assert_sync_reports_complete
                assert_sync_reports_complete(
                    run_dir,json.loads((run_dir/'metrics.json').read_text()))
            return {'status':'already_completed','run_dir':str(run_dir)}
    atomic_json(config_file,cfg.to_dict())
    atomic_json(status_file,{'status':'running','protocol_version':protocol_version_for(cfg),'started_at':time.time()})
    try:
        # Stable independent randomness for independent processes.
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg.seed)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bank=get_data_bank(cfg.dataset,data_root,download=download,seed=cfg.seed,class_subset=cfg.class_subset)
        scenario=(compile_stream_scenario(bank,cfg) if cfg.stream_protocol
                  else compile_scenario(bank,cfg))
        if cfg.learning_setting=='task_il':
            (scenario,task_ids_by_index,task_class_map,
             task_head_first_round)=_assign_task_ids(scenario,bank,cfg)
            task_class_sets=task_class_map
        else:
            task_ids_by_index=None;task_class_map={};task_head_first_round={}
            task_class_sets=None
        if cfg.learning_setting=='task_il' and not task_class_sets:
            raise RuntimeError('Task-IL scenario has no task heads')
        torch.save({'arrivals':[asdict(a) for a in scenario.arrivals],
                    'stages':scenario.stages,
                    'original_labels':bank.original_labels,
                    'task_ids_by_training_index':task_ids_by_index,
                    'task_id_to_classes':task_class_map,
                    'duplicated_training_samples':scenario.duplicate_sample_ids,
                    'unused_training_samples':scenario.unused_sample_ids},run_dir/'scenario.pt')
        clients={cid:LocalState(cid) for cid in range(cfg.num_clients)}
        model=create_model(cfg.model,bank.num_classes,allow_test_model=(cfg.dataset=='synthetic'),task_class_sets=task_class_sets,learning_setting=cfg.learning_setting if cfg.joint_plan_path else 'task_il').to(device)
        if cfg.joint_plan_path:
            atomic_json(run_dir/'joint_plan.json',scenario.joint_plan)
        backbone_description=getattr(model,'backbone_description',{'test_only':True})
        if cfg.dataset=='cifar100':
            # All clients and the server start from the identical configured
            # seeded-random 32x32 backbone; no pretrained-weight assertion.
            reference=(run_dir/'architecture_reference.json' if cfg.joint_plan_path else Path(output_root)/(cfg.model+'_architecture_reference.json'))
            normalized_description=json.loads(json.dumps(backbone_description))
            if reference.exists():
                if json.loads(reference.read_text())!=normalized_description:
                    raise RuntimeError('Backbone definition differs from prior jobs; use a new experiment root')
            else:
                atomic_json(reference,normalized_description)
        global_state=copy_cpu_state(model)
        # Initial model is shared at round zero. Subsequent downloads require
        # explicit communication, except deliberate full-sync ablations.
        for client in clients.values():
            client.receive(global_state,version=0)
        observed=set();peak={};test_peak={};history=[];version=0
        integrated=set();server_optimizer_state={}
        server_class_update_counts={}
        server_cl_state=FederatedCLState()
        first_round=0
        rng=np.random.default_rng(cfg.seed+888_777)
        total_rounds=(len(scenario.stages) if cfg.stream_protocol else cfg.stages*cfg.rounds_per_stage)
        class_first_stage={}
        for stage,labels in enumerate(scenario.stages):
            for c in labels:class_first_stage.setdefault(c,stage)
        latest_checkpoint=run_dir/'checkpoint_latest.pt'
        source_checkpoint=(Path(fork_from_checkpoint) if fork_from_checkpoint else
                           (latest_checkpoint if cfg.resume and latest_checkpoint.exists() else None))
        if source_checkpoint is not None:
            saved=torch.load(source_checkpoint,map_location='cpu',weights_only=False)
            if saved.get('protocol_version')!=protocol_version_for(cfg):
                raise RuntimeError('Cannot resume or fork a checkpoint from an incompatible training protocol')
            if fork_from_checkpoint:
                # Forks may only change the participation intervention and
                # destination identity. Dataset/model/training must be identical.
                allowed={'job_id','force_target','target_client','intervention_round',
                         'causal_diagnostics','gradient_diagnostics','checkpoint_every',
                         'fork_source_job','fork_checkpoint_round'}
                old={k:v for k,v in _normalize_saved_config(saved['config']).items()
                     if k not in allowed}
                new={k:v for k,v in cfg.to_dict().items() if k not in allowed}
                if old!=new:
                    changed=[k for k in old if old[k]!=new.get(k)]
                    raise RuntimeError(f'Fork changes confounded settings: {changed}')
            elif _normalize_saved_config(saved['config'])!=cfg.to_dict():
                raise RuntimeError('Resumption configuration does not match saved checkpoint')
            first_round=int(saved['last_completed_round'])+1
            global_state=saved['global_state'];version=saved['version']
            clients={int(cid):LocalState(**state) for cid,state in saved['clients'].items()}
            observed=set(saved['observed']);peak=saved['peak'];integrated=set(saved['integrated'])
            history=saved['history'];server_optimizer_state=saved['server_optimizer_state']
            # Rebuild audit-only held-out peaks from prior persisted metrics.
            # The test history is never an input to selection/training.
            for previous in history:
                for cls,acc in previous.get('per_sync_test_per_class',{}).items():
                    c=int(cls)
                    test_peak[c]=max(test_peak.get(c,0.),float(acc))
            server_class_update_counts=saved.get('server_class_update_counts',{})
            server_cl_state=saved.get('server_cl_state')
            if not isinstance(server_cl_state,FederatedCLState):
                raise RuntimeError('Missing v8 federated continual-learning checkpoint state')
            rng.bit_generator.state=saved['selection_rng']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if torch.cuda.is_available() and saved['cuda_rng']:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        # Keep one *live independent* model per client and configured streams.
        # The global server remains a sixth, separately held model object.
        # A single-GPU pilot launches two independent simulations as two
        # simultaneous processes; each process retains five client models.
        client_runtimes=create_client_runtimes(cfg,bank,clients,device,task_class_sets)
        client_executor=ThreadPoolExecutor(
            max_workers=(cfg.client_training_concurrency or cfg.num_clients),
            thread_name_prefix='fcl-client')
        for round_id in range(first_round,total_rounds):
            active_task_ids=tuple(sorted(head for head,first in task_head_first_round.items()
                                          if first<=round_id))
            active_task_sets={task_id:task_class_map[task_id]
                              for task_id in active_task_ids}
            round_start=time.monotonic()
            if (cfg.method=='native_lga' and round_id==1
                    and server_cl_state.native_lga_radius is None):
                # Official LGA computes a source covariance radius once
                # at its first class-stage boundary. Use only TRAIN samples
                # already integrated by the server after interval one.
                from .upstream_lga_radius import original_lga_radius
                model.load_state_dict(global_state)
                model.to(device)
                server_cl_state.native_lga_radius=original_lga_radius(
                    model,bank,server_cl_state.acknowledged_train_indices,device)
            
            # Queue all client arrivals but do NOT expose later class labels to
            # the local learner before it has processed earlier class events.
            # This is essential for ER-ACE masking and true within-round CL.
            new_events={cid:[] for cid in clients}
            joint_events=[]
            if cfg.joint_plan_path:
                lookup={x['occurrence_id']:x for x in scenario.joint_plan['tasks']}
                joint_events=[lookup[oid] for oid in scenario.joint_plan['intervals'][round_id]['occurrences']]
            for arrival_index,arrival in enumerate(scenario.arrivals_at(round_id)):
                if cfg.stream_protocol:
                    new_events[arrival.client_id].append({
                        'indices':list(map(int,arrival.indices)),
                        'padding_indices':list(map(int,arrival.padding_indices)),
                        'class_ids':tuple(map(int,arrival.labels)),
                        'task_id':arrival.task_id,
                        'global_stage':int(arrival.stage),
                        'is_recurrence':bool(arrival.recurrence)})
                    if cfg.joint_plan_path:
                        atomic=joint_events[arrival_index]
                        assert atomic['client_id']==arrival.client_id and tuple(atomic['classes'])==arrival.labels
                        new_events[arrival.client_id][-1].update(atomic_step=atomic['atomic_step'],
                            occurrence_id=atomic['occurrence_id'],semantic_task_id=atomic['semantic_task_id'])
                    observed.update(arrival.labels)
                    continue
                for cls in arrival.labels:
                    ids=[int(i) for i in arrival.indices
                         if int(bank.train.targets[i])==int(cls)]
                    if ids:
                        new_events[arrival.client_id].append({
                            'indices':ids,'class_id':int(cls),
                            'task_id':arrival.task_id,
                            'global_stage':int(arrival.stage),
                            'is_recurrence':bool(arrival.recurrence)})
                # Global test exposure is independent of participation, and
                # includes labels from all arrivals in this communication round.
                observed.update(arrival.labels)
            current_stage=(round_id if cfg.stream_protocol else
                           min(cfg.stages-1,round_id//cfg.rounds_per_stage))
            old={c for c in observed if class_first_stage[c]<current_stage}
            new=set(observed)-old
            # All local clients, including those selected offline later, own
            # separate GPU models. Every client's first task is submitted
            # concurrently; only after all complete does any Task 2 begin.
            # Federation is forbidden until the final task barrier finishes.
            if cfg.joint_plan_path:
                from .joint.audit import server_boundary_digest
                boundary_before=server_boundary_digest(global_state,version,server_cl_state)
            local_details,local_trained_clients,task_barriers=(
                train_round_task_barriers(
                    runtimes=client_runtimes,executor=client_executor,
                    clients=clients,bank=bank,cfg=cfg,device=device,
                    round_id=round_id,current_stage=current_stage,
                    scenario=scenario,new_events=new_events,
                    known_task_sets=(() if cfg.joint_plan_path else active_task_ids),
                    task_ids_by_index=task_ids_by_index))
            if cfg.joint_plan_path:
                boundary_after=server_boundary_digest(global_state,version,server_cl_state)
                if boundary_before!=boundary_after:
                    raise RuntimeError('Server changed during private local continual learning')
                if [b['atomic_step'] for b in task_barriers]!=scenario.joint_plan['intervals'][round_id]['atomic_steps']:
                    raise RuntimeError('Executed atomic barriers differ from the plan')
                atomic_json(run_dir/f'joint_interval_{round_id:05d}.json',{
                    'interval':round_id,'server_version_during_local_learning':version,
                    'server_state_digest_before':boundary_before,'server_state_digest_after':boundary_after,
                    'atomic_barriers':task_barriers,'local_occurrences':local_details})
            if cfg.joint_plan_path and cfg.joint_local_evaluation and local_details:
                # Use the same planned, all-observed evaluation context before
                # and after aggregation. This cannot affect selection.
                model.load_state_dict(global_state)
                joint_global_before,_=evaluate(model,bank,observed,cfg.batch_size,device,
                    task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                    split='validation',loader_seed=0)
            else:joint_global_before=None
            class_relationships=None
            if cfg.selector in {'feature_max','feature_min','gradient_align','gradient_conflict'}:
                # Simulation-privileged: historical data that have *arrived*,
                # never future or test data. Feature and gradient selectors
                # are oracle diagnostics, not privacy-preserving methods.
                model.load_state_dict(global_state)
                model.to(device)
                arrived=set(server_cl_state.acknowledged_train_indices)
                if cfg.selector.startswith('feature'):
                    vectors=arrived_feature_means(model,bank,observed,device,
                                                  allowed_indices=arrived)
                else:
                    vectors=class_gradient_representations(model,bank,observed,device,
                                                           allowed_indices=arrived)
                class_relationships=pairwise_vector_cosines(vectors)
            risks={}
            if history:
                # Never feed held-out TEST information into client selection.
                # Prior published server-sync tables are diagnostic only.
                observed_validation=history[-1].get('selection_validation_per_class',
                                                     history[-1].get('per_class',{}))
                risks={int(c):max(0.,peak.get(int(c),0.)-float(v))
                       for c,v in observed_validation.items()}
            selected=(list(scenario.joint_plan['intervals'][round_id]['upload_clients'])
                      if cfg.joint_plan_path else select_clients(clients,cfg,round_id,rng,old_classes=old,new_classes=new,
                                    coarse_by_fine=bank.coarse_by_fine,
                                    relationships=class_relationships,class_forgetting=risks))
            if cfg.stream_protocol and not cfg.joint_plan_path:
                is_flush=bool(cfg.stream_terminal_flush and round_id==total_rounds-1)
                if is_flush:
                    # One clearly marked communication-only closing round:
                    # every client with unsent updates can finally contribute.
                    selected=sorted(clients)
                else:
                    selected=select_offline_cohort(cfg,round_id,total_rounds,selected)
            selected_set=set(selected)
            for cid in selected:
                for event in clients[cid].local_training_events:
                    if event['round']==round_id:
                        event['selected_for_upload']=True
            for entry in local_details:
                entry['is_selected']=entry['client'] in selected_set
            accepted=[]
            base_global_state=global_state
            for cid in selected:
                client=clients[cid]
                uploaded=anchored_upload(client,global_state,current_version=version)
                if uploaded is None:
                    continue
                uploaded['client_id']=cid
                integrated.update(uploaded['accumulated_classes'])
                accepted.append(uploaded)
            # Nonselected clients have not downloaded, uploaded or reset their
            # pending updates. Their local trajectory remains intact.
            evaluation_bank=bank
            if cfg.scenario in {'domain','gradual_drift','shifted_recurrence'}:
                from .data import DataBank
                from .domains import GlobalDomainTest
                domain_labels={}
                for arrival in scenario.arrivals:
                    if arrival.round_id>round_id:break
                    key=(arrival.client_id,arrival.stage)
                    domain_labels.setdefault(key,set()).update(arrival.labels)
                slices=[(cid,stage,sorted(labels))
                        for (cid,stage),labels in sorted(domain_labels.items())]
                domain_test=GlobalDomainTest(bank.test,slices,cfg.scenario)
                domain_validation=(GlobalDomainTest(bank.validation,slices,cfg.scenario)
                                   if bank.validation is not None else None)
                evaluation_bank=DataBank(bank.train,domain_test,bank.num_classes,
                                         bank.coarse_by_fine,bank.original_labels,
                                         domain_validation)
            if cfg.spurious_strength:
                from .data import DataBank
                from .domains import SpuriousTest
                evaluation_bank=DataBank(bank.train,
                    SpuriousTest(evaluation_bank.test,cfg.spurious_strength),
                    bank.num_classes,bank.coarse_by_fine,bank.original_labels,
                    (SpuriousTest(evaluation_bank.validation,cfg.spurious_strength)
                     if evaluation_bank.validation is not None else None))
            pre_aggregation_metrics=None
            pre_representation=None
            # A fixed historical cohort is required for meaningful before/after
            # representation-drift comparisons; don't add new accepted IDs mid-measure.
            pre_representation_ids=set(server_cl_state.acknowledged_train_indices)
            if accepted and cfg.prepost_diagnostics:
                model.load_state_dict(global_state)
                model.to(device)
                # Diagnostics only; never fed back to training or selection.
                pre_aggregation_metrics,_=evaluate(model,evaluation_bank,observed,cfg.batch_size,
                                                    device,peak.copy(),integrated=integrated,
                                                    old=old,new=new,
                                                    task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                                                    split='validation',feat_ecc=(server_cl_state.feat_ecc if cfg.method=='native_feat' else None))
            if accepted and cfg.representation_diagnostics and old:
                model.load_state_dict(global_state)
                model.to(device)
                arrived=set(server_cl_state.acknowledged_train_indices)
                pre_representation=arrived_feature_means(model,bank,old,device,
                                                          allowed_indices=arrived)
            aggregation_norm_diagnostics=None
            custom_head_fusion=False
            if accepted:
                old_gradient=None
                if cfg.aggregator in {'retention_aware'} and old:
                    model.load_state_dict(global_state)
                    arrived=set(server_cl_state.acknowledged_train_indices)
                    vectors=class_gradient_representations(model,bank,old,device,
                                                           allowed_indices=arrived)
                    if vectors:old_gradient=torch.stack(list(vectors.values())).mean(0)
                mean=aggregate(accepted,global_state,cfg.aggregator,
                               clients=clients,round_id=round_id,old_gradient=old_gradient,
                               class_history_counts=server_cl_state.class_sample_exposure,
                               source_method=(None if cfg.learning_setting=='task_il' else cfg.method if cfg.method in {'native_glfc','native_lga','native_feat'} else
                                              'fedclass_resnet8' if cfg.model=='fedclass_resnet8' else None))
                candidate,new_server_state=server_optimizer_step(
                    global_state,mean,cfg.aggregator,cfg,server_optimizer_state)
                if cfg.learning_setting=='task_il' and cfg.aggregator=='fedavg':
                    # The per-head contributor average is the Task-IL form of
                    # canonical FedAvg. Do not apply it after FedAdam/FedAvgM
                    # or a custom server rule: doing so would erase that
                    # aggregator's head update and momentum/adaptive state.
                    candidate=_merge_task_heads_fedavg(accepted,global_state,candidate)
                is_terminal_flush=bool(cfg.stream_protocol and cfg.stream_terminal_flush and round_id==total_rounds-1)
                candidate,terminal_stale_scale=_guard_single_stale_terminal_flush(
                    candidate,global_state,accepted,cfg,terminal_flush=is_terminal_flush)
                # Canonical methods MUST aggregate every classifier row with
                # the same rule as every other model parameter.
                custom_head_fusion=bool(cfg.classwise_head_merge and cfg.aggregator in
                    {'class_balanced','inverse_participation','conflict_aware',
                     'conflict_projection','retention_aware'})
                if custom_head_fusion and cfg.learning_setting!='task_il':
                    candidate=classwise_merge(global_state,candidate,accepted,
                        server_class_update_counts,classifier_parameter_names(model),
                        method=cfg.aggregator)
                for u in accepted:
                    for cls,n in u.get('class_update_steps',{}).items():
                        server_class_update_counts[int(cls)]=(
                            server_class_update_counts.get(int(cls),0)+int(n))

                # Accepted client-private historical state is merged before
                # the model/state acknowledgement; inactive clients are untouched.
                if cfg.share_continual_state:
                    server_cl_state.merge_clients([clients[u['client_id']] for u in accepted],
                        bank,cfg.method,candidate,version+1,
                        task_ids_by_index=task_ids_by_index)
                    if cfg.method in {'native_glfc','native_lga'}:
                        # Execute original source proxy inversion, monitor,
                        # and model_back on accepted gradients only.
                        # This precedes selected-client broadcast; hidden
                        # clients cannot receive it before reconnection.
                        from .upstream_native_proxy_server import source_proxy_server_step
                        source_proxy_server_step(server_cl_state,cfg.method,
                            model,candidate,cfg,device,round_id)
                    for upload in accepted:
                        for cls,n in upload['class_counts'].items():
                            cls=int(cls)
                            server_cl_state.class_sample_exposure[cls]=(
                                server_cl_state.class_sample_exposure.get(cls,0)+int(n))
                # Candidate is built without mutating the committed server state.
                for cid in selected:
                    clients[cid].receive(candidate,version+1,
                        clear_memory=cfg.reset_memory_on_sync)
                    if cfg.share_continual_state:
                        server_cl_state.broadcast_to(
                            clients[cid],clients[cid].last_learning_method or cfg.method)
                        server_cl_state.assert_complete_download(clients[cid])
                    assert all(torch.equal(clients[cid].shared_weights[k],v) for k,v in candidate.items())
                global_state=candidate
                server_optimizer_state=new_server_state
                version+=1
            else:
                # All active clients still receive the current committed model.
                for cid in selected:
                    clients[cid].receive(global_state,version,
                        clear_memory=cfg.reset_memory_on_sync)
                    if cfg.share_continual_state:
                        server_cl_state.broadcast_to(
                            clients[cid],clients[cid].last_learning_method or cfg.method)
                        server_cl_state.assert_complete_download(clients[cid])
            # Capture server-level backbone/head compatibility diagnostics
            # after the commit, without changing any global or local weights.
            aggregation_norm_diagnostics=_normalization_aggregation_diagnostics(
                base_global_state,global_state,accepted)
            if cfg.full_sync:
                for cid in clients:
                    clients[cid].receive(global_state,version,
                        clear_memory=cfg.reset_memory_on_sync)
                    if cfg.share_continual_state:
                        server_cl_state.broadcast_to(
                            clients[cid],clients[cid].last_learning_method or cfg.method)
                        server_cl_state.assert_complete_download(clients[cid])
            for cid in selected:
                clients[cid].last_selected=round_id
                clients[cid].selection_count+=1
            model.load_state_dict(global_state)
            model.to(device)
            if cfg.joint_plan_path and cfg.joint_local_evaluation:
                joint_global_after,_=evaluate(model,bank,observed,cfg.batch_size,device,
                    task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                    split='validation',loader_seed=0)
                if cfg.learning_setting=='task_il':
                    from .joint.task_evaluation import evaluate_task_conditioned
                    joint_global_after['task_conditioned']=evaluate_task_conditioned(
                        model,bank,active_task_sets,cfg.batch_size,device)
                # The final local measurement in each interval is the pre-upload
                # model. Selected clients received exactly the committed state;
                # evaluate it with their unchanged pre-upload competitor set.
                last_local={d['client']:d for d in local_details}
                broadcast_metrics={}
                for cid in selected:
                    prior=last_local.get(cid,{}).get('local_validation_post_task')
                    if prior is None:continue
                    labels=set(prior['observed_classes'])
                    groups=prior.get('evaluation_task_heads') if cfg.learning_setting=='task_il' else None
                    after,_=evaluate(model,bank,labels,cfg.batch_size,device,
                        task_class_groups=groups,split='validation',loader_seed=0)
                    broadcast_metrics[cid]={'pre_upload':prior,'post_broadcast':after,
                        'accuracy_change':after['accuracy']-prior['accuracy']}
                atomic_json(run_dir/f'joint_communication_{round_id:05d}.json',{
                    'interval':round_id,'server_version':version,'selected_clients':selected,
                    'accepted_clients':[u['client_id'] for u in accepted],
                    'global_validation_before':joint_global_before,'global_validation_after':joint_global_after,
                    'broadcast_validation':broadcast_metrics})
            causal=None
            if cfg.causal_diagnostics and len(accepted)>=2 and cfg.aggregator in {'fedavg','uniform'}:
                # Use the pre-aggregation base model and the exact accepted
                # client updates. All candidates are from the same model version.
                causal=aggregation_leave_one_out(
                    model,evaluation_bank,observed,accepted,base_global_state,
                    'uniform' if cfg.aggregator=='uniform' else 'fedavg',
                    cfg.batch_size,device,limit=cfg.causal_max_clients)
            subset_analysis=None
            if cfg.subset_diagnostics and accepted and cfg.aggregator in {'fedavg','uniform'}:
                subset_analysis=aggregation_subset_report(model,evaluation_bank,observed,
                    accepted,base_global_state,cfg.aggregator,cfg.batch_size,device,
                    limit=cfg.causal_max_clients)
            representation_drift=None
            if pre_representation is not None:
                arrived=set(pre_representation_ids)
                after=arrived_feature_means(model,bank,old,device,allowed_indices=arrived)
                representation_drift={str(c):float(torch.linalg.vector_norm(
                    after[c].float()-v.float())) for c,v in pre_representation.items()
                    if c in after}
            gradient_relationships=None
            if cfg.gradient_diagnostics:
                gradient_relationships=diagnostic_gradient_relationships(
                    model,bank,observed,device,samples_per_class=8,max_pairs=32)
            client_update_cosines=None
            old_update_alignment=None
            if cfg.gradient_diagnostics and accepted:
                floating=[k for k,v in base_global_state.items() if v.is_floating_point()]
                deltas={u['client_id']:torch.cat([
                    (u['weights'][k].float()-base_global_state[k].float()).reshape(-1)
                    for k in floating]) for u in accepted}
                from .diagnostics import vector_cosine
                from itertools import combinations
                client_update_cosines={f'{a}:{b}':vector_cosine(deltas[a],deltas[b])
                    for a,b in combinations(sorted(deltas),2)}
                if old:
                    arrived=set(server_cl_state.acknowledged_train_indices)
                    model.load_state_dict(base_global_state)
                    vectors=class_gradient_representations(model,bank,old,device,
                        allowed_indices=arrived)
                    if vectors:
                        mean_old_grad=torch.stack(list(vectors.values())).mean(0)
                        old_update_alignment={str(cid):vector_cosine(mean_old_grad,delta)
                            for cid,delta in deltas.items()}
                    model.load_state_dict(global_state)
            # Some optional gradient/causal diagnostics temporarily load
            # candidate or pre-aggregation weights. Restore the committed
            # server model explicitly before ANY scored evaluation.
            model.load_state_dict(global_state)
            model.to(device)
            if cfg.evaluate_final_only and round_id!=total_rounds-1:
                # The pilot's official test split is read exactly ONCE, after
                # the terminal flush. Earlier rounds store communication and
                # client-state diagnostics but never evaluate held-out labels.
                result={'accuracy':None,'macro_accuracy':None,'forgetting':None,
                        'per_class':{},'cross_entropy_loss':None,'evaluation_deferred':True,
                        'observed_classes':sorted(observed),
                        'globally_integrated_classes':sorted(integrated),
                        'evaluation_split':None,'evaluation_samples':0}
            else:
                # If terminal-only, do not mix held-out TEST peaks into
                # the validation peak history used by client selectors.
                evaluation_peaks=(peak.copy() if cfg.evaluate_final_only else peak)
                result,new_eval_peaks=evaluate(model,evaluation_bank,observed,cfg.batch_size,device,evaluation_peaks,
                                     integrated=integrated,old=old,new=new,
                                     task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                                     split=('test' if cfg.evaluate_final_only else 'validation'),feat_ecc=(server_cl_state.feat_ecc if cfg.method=='native_feat' else None))
                if not cfg.evaluate_final_only:
                    peak=new_eval_peaks
                result['evaluation_deferred']=False
            # Every communication interval, including the all-client closing
            # flush, gets a fresh server-global evaluation over every currently
            # observed class. Held-out TEST scores are audit-only: do not use
            # them in client selection, training or hyperparameter search.
            from .stage_report import render_sync_report
            validation_report=None
            if evaluation_bank.validation is not None:
                # Selection/forgetting peaks are always VALIDATION-only.
                # This must happen even when the official test is deferred
                # until the last round, otherwise forgetting-based selectors
                # silently see zero history at every intermediate sync.
                validation_report,peak=evaluate(model,evaluation_bank,observed,
                    cfg.batch_size,device,peak=peak,integrated=integrated,
                    old=old,new=new,
                    task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                    split='validation',feat_ecc=(server_cl_state.feat_ecc if cfg.method=='native_feat' else None))
            per_sync_test,test_peak=evaluate(model,evaluation_bank,observed,
                cfg.batch_size,device,peak=test_peak,integrated=integrated,
                old=old,new=new,
                task_class_groups=active_task_sets if cfg.learning_setting=='task_il' else None,
                split='test',feat_ecc=(server_cl_state.feat_ecc if cfg.method=='native_feat' else None))
            # Report the previously defined all-arrived metric AND the
            # accepted-only server capability. Offline unsent classes must
            # never be presented as already learned by the server.
            integrated_test=None
            integrated_task_sets={}
            if cfg.learning_setting=='task_il':
                integrated_task_sets=_integrated_task_class_sets(
                    task_ids_by_index,server_cl_state.acknowledged_train_indices,
                    task_class_map)
            if integrated and not (cfg.learning_setting=='task_il' and
                                   not integrated_task_sets):
                integrated_test,_=evaluate(model,evaluation_bank,set(integrated),
                    cfg.batch_size,device,peak={},integrated=integrated,
                    old=old,new=new,
                    task_class_groups=integrated_task_sets if cfg.learning_setting=='task_il' else None,
                    split='test',feat_ecc=(server_cl_state.feat_ecc if cfg.method=='native_feat' else None))
            sync_report=render_sync_report(run_dir,round_number=round_id+1,
                model_version=version,num_clients=cfg.num_clients,
                arrivals=scenario.arrivals,test_metrics=per_sync_test,
                validation_metrics=validation_report,selected_clients=selected,
                terminal_flush=bool(cfg.stream_protocol and cfg.stream_terminal_flush and round_id==total_rounds-1),
                backbone=cfg.model,
                optimizer=dict(name='sgd',lr=cfg.lr,momentum=cfg.momentum,weight_decay=cfg.weight_decay),
                classifier_bias=(global_state[classifier_parameter_names(model)[1]]
                                 if classifier_parameter_names(model)[1] is not None else None),
                integrated_classes=integrated,integrated_metrics=integrated_test,
                integrated_task_heads=integrated_task_sets)
            # Used by validation-based selectors and never by the test audit.
            result['selection_validation_per_class']=(
                validation_report['per_class'] if validation_report else {})
            result['per_sync_test_accuracy']=per_sync_test['accuracy']
            result['per_sync_server_integrated_only_accuracy']=(
                integrated_test['accuracy'] if integrated_test else None)
            result['per_sync_server_integrated_task_heads']=[
                {'task_id':task_id,'classes':list(classes)}
                for task_id,classes in sorted(integrated_task_sets.items())]
            result['per_sync_pending_classes']=sorted(set(observed)-set(integrated))
            result['per_sync_test_per_class']=per_sync_test['per_class']
            result['per_sync_validation_accuracy']=(
                validation_report['accuracy'] if validation_report else None)
            result['sync_accuracy_report']=sync_report
            result['per_sync_forgetting']=per_sync_test['forgetting']
            if cfg.evaluate_final_only and round_id==total_rounds-1:
                # Final public TEST score is the same held-out evaluation.
                # Reuse its peak-based forgetting rather than falsely claiming
                # zero forgetting because historical testing was deferred.
                result['forgetting']=per_sync_test['forgetting']
                result['all_observed_peak_drop']=per_sync_test['all_observed_peak_drop']
            if cfg.spurious_ood_eval and cfg.spurious_strength:
                from .data import DataBank
                from .domains import SpuriousTest
                shifted=DataBank(bank.train,SpuriousTest(bank.test,cfg.spurious_strength,reverse=True),
                                 bank.num_classes,bank.coarse_by_fine,bank.original_labels)
                shifted_metrics,_=evaluate(model,shifted,observed,cfg.batch_size,
                                           device,peak={},integrated=integrated)
                result['spurious_reversed_accuracy']=shifted_metrics['accuracy']
            if pre_aggregation_metrics is not None:
                result['pre_aggregation_accuracy']=pre_aggregation_metrics['accuracy']
                result['pre_aggregation_cross_entropy']=pre_aggregation_metrics['cross_entropy_loss']
                result['aggregation_accuracy_delta']=result['accuracy']-pre_aggregation_metrics['accuracy']
                result['aggregation_loss_delta']=result['cross_entropy_loss']-pre_aggregation_metrics['cross_entropy_loss']
                result['per_class_aggregation_loss_delta']={str(c):
                    loss-pre_aggregation_metrics['per_class_loss'][c]
                    for c,loss in result['per_class_loss'].items()}
            transmitted_bytes=sum(sum(t.numel()*t.element_size()
                    for t in u['weights'].values()) for u in accepted)
            result.update({'round':round_id,'model_version':version,
                           'terminal_communication_only_flush':bool(cfg.stream_protocol and cfg.stream_terminal_flush and round_id==total_rounds-1),
                           'stage':current_stage,'selected_clients':selected,
                           'selected_count':len(selected),'uploaded_count':len(accepted),
                           'locally_trained_clients':sorted(set(local_trained_clients)),
                           'locally_trained_count':len(set(local_trained_clients)),
                           'uploaded_accumulated_classes':{str(u['client_id']):u['accumulated_classes'] for u in accepted},
                           'uploaded_since_version':{str(u['client_id']):u['anchor_version'] for u in accepted},
                           'uploaded_accumulated_steps':{str(u['client_id']):u['accumulated_steps'] for u in accepted},
                           'client_unsent_steps':{str(cid):c.pending_upload_steps for cid,c in clients.items()},
                           'protocol_version':protocol_version_for(cfg),
                           'aggregation_algorithm':cfg.aggregator,
                           'aggregation_family':('stale_delta_damped_bn_safe' if any(u['stale_rebased'] for u in accepted)
                                else ('uniform_mean_ablation' if cfg.aggregator=='uniform'
                                else ('canonical_full_model' if cfg.aggregator in
                                      {'fedavg','fedavgm','fedadam'} else 'repository_specific_custom'))),
                           'classwise_head_merge_applied':custom_head_fusion,
                           'canonical_fedavg_reference_class_contributions':_classwise_fedavg_contribution(accepted),
                           'aggregation_norm_diagnostics':aggregation_norm_diagnostics,
                           'stale_rebased_client_ids':[u['client_id'] for u in accepted if u['stale_rebased']],
                           'stale_upload_lag_versions':{str(u['client_id']):u.get('staleness_versions',0) for u in accepted},
                           'stale_upload_scales':{str(u['client_id']):u.get('stale_update_scale',1.0) for u in accepted},
                           'terminal_stale_flush_scale':terminal_stale_scale if accepted else 1.0,
                           'upload_model_origins':{str(u['client_id']):u['aggregation_input_semantics'] for u in accepted},
                           'server_class_sample_exposure':{str(k):int(v) for k,v in sorted(server_cl_state.class_sample_exposure.items())},
                           'server_integrated_class_count':len(server_cl_state.integrated_classes),
                           'server_acknowledged_train_examples':len(server_cl_state.acknowledged_train_indices),
                           'server_cl_state':server_cl_state.statistics(),
                           'client_shared_memory_sizes':{str(cid):sum(map(len,c.received_memory.values())) for cid,c in clients.items()},
                           'client_private_memory_sizes':{str(cid):sum(map(len,c.private_memory.values())) for cid,c in clients.items()},
                           'client_private_replay_by_class':{str(cid):{str(cls):len(indices) for cls,indices in client.private_memory.items()} for cid,client in clients.items()},
                           'client_federated_cl_versions':{str(cid):c.federated_cl_version for cid,c in clients.items()},
                           'local_details':local_details,
                           'client_execution_mode':(
                               'parallel_task_barriers'
                               if not cfg.client_training_concurrency
                               else 'bounded_concurrent_task_barriers'),
                           'client_training_concurrency':(
                               cfg.client_training_concurrency or cfg.num_clients),
                           'task_barriers':task_barriers,
                           'independent_live_client_models':len(client_runtimes),
                           'seconds':time.monotonic()-round_start,
                           'global_test_includes_inactive':True,
                           'global_evaluation_includes_inactive':True,
                           'domain_evaluation_slices':(len(slices) if cfg.scenario in {'domain','gradual_drift','shifted_recurrence'} else None),
                           'causal_diagnostics':causal,
                           'subset_diagnostics':subset_analysis,
                           'representation_drift':representation_drift,
                           'update_transmitted_bytes':transmitted_bytes,
                           'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)
                               if device.type=='cuda' else 0,
                           'gradient_relationships':gradient_relationships,
                           'client_update_cosines':client_update_cosines,
                           'old_gradient_vs_client_updates':old_update_alignment,
                           'client_training_versions':{str(cid):clients[cid].model_version for cid in selected},
                           'participation_rate_realized':len(selected)/cfg.num_clients})
            history.append(result)
            # Durable intermediate results, so an interrupted run remains auditable.
            atomic_json(run_dir/'metrics.json',history)
            if cfg.checkpoint_every and (round_id+1)%cfg.checkpoint_every==0 or \
               (cfg.stream_protocol or (round_id+1)%cfg.rounds_per_stage==0):
                save_round_state(latest_checkpoint,cfg,round_id,global_state,version,
                                 clients,observed,peak,integrated,history,rng,
                                 server_optimizer_state,server_class_update_counts,server_cl_state)
                if cfg.checkpoint_every:
                    save_round_state(run_dir/f'checkpoint_round_{round_id+1:04d}.pt',
                                     cfg,round_id,global_state,version,
                                     clients,observed,peak,integrated,history,rng,
                                     server_optimizer_state,server_class_update_counts,server_cl_state)
            if (cfg.stream_protocol or (round_id+1)%cfg.rounds_per_stage==0):
                save_checkpoint(run_dir/f'model_stage_{current_stage+1:02d}.pt',global_state,
                                cfg,version,clients,full_state=True,server_cl_state=server_cl_state)
            if verbose:
                acc=('deferred' if result['accuracy'] is None else f'{result["accuracy"]:.4f}')
                forgetting=('deferred' if result['forgetting'] is None else f'{result["forgetting"]:.4f}')
                print(f'[{job}] round {round_id+1}/{total_rounds} acc={acc} '
                      f'forgetting={forgetting} clients={len(selected)}',flush=True)
        client_executor.shutdown(wait=True)
        if cfg.stream_protocol:
            from .stage_report import assert_sync_reports_complete
            assert_sync_reports_complete(run_dir,history)
        save_checkpoint(run_dir/'model_final.pt',global_state,cfg,version,clients,full_state=True,server_cl_state=server_cl_state)
        atomic_json(run_dir/'summary.json',{'job_id':job,'dataset':cfg.dataset,
                    'fork_source':str(fork_from_checkpoint) if fork_from_checkpoint else None,
                    'device':str(device),'rounds':total_rounds,'backbone':backbone_description,'final_metrics':history[-1],
                    'model_final':str(run_dir/'model_final.pt'),
                    'shared_continual_state':server_cl_state.statistics(),
                    'per_sync_accuracy_report_dir':str(run_dir/'per_sync_accuracy'),
                    'per_sync_report_count':total_rounds})
        atomic_json(status_file,{'status':'completed','protocol_version':protocol_version_for(cfg),
                                'finished_at':time.time(),
                                'rounds':total_rounds})
        return {'status':'completed','run_dir':str(run_dir),'final_accuracy':history[-1]['accuracy']}
    except Exception as exc:
        if 'client_executor' in locals():
            client_executor.shutdown(wait=True,cancel_futures=True)
        atomic_json(status_file,{'status':'failed','protocol_version':protocol_version_for(cfg),
                                  'error':repr(exc),'traceback':traceback.format_exc()})
        raise
