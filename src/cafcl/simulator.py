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
from .selection import select_clients
from .evaluation import evaluate
from .causal import aggregation_leave_one_out,aggregation_subset_report
from collections import Counter
from .diagnostics import (diagnostic_gradient_relationships,
    class_gradient_representations, arrived_feature_means, pairwise_vector_cosines)


# Changing the semantics of an existing job must never silently reuse its old
# success markers or checkpoints, even though its scientific config is unchanged.
PROTOCOL_VERSION = 'cifar100_deit_tiny_v4'


def anchored_upload(client, global_state):
    """Rebase *all* locally accumulated learning since the last download.

    For a client last synchronized at anchor A with current private model L,
    the current-server contribution is G+(L-A). If A==G (no staleness), this
    equals ordinary FedAvg weight aggregation. Otherwise it is explicitly a
    stale-delta FedAvg adaptation, NOT identical to synchronous FedAvg.
    Neither model weights nor privacy-sensitive raw memory are sent separately.
    """
    if client.shared_weights is None or client.sync_anchor is None:
        return None
    if client.pending_upload_steps <= 0:
        return None
    candidate={}
    for key,gv in global_state.items():
        lv=client.shared_weights[key]
        anchor=client.sync_anchor[key]
        if lv.shape != gv.shape or anchor.shape != gv.shape:
            raise RuntimeError('Client/server state shape mismatch')
        if gv.is_floating_point():
            candidate[key]=(gv.to(torch.float64)+lv.to(torch.float64)
                           -anchor.to(torch.float64)).to(gv.dtype)
        else:
            candidate[key]=lv.clone()
    if not check_finite(candidate):
        raise RuntimeError('Nonfinite accumulated client update')
    return {'weights':candidate,'examples':client.pending_upload_examples,
            'class_counts':dict(client.pending_upload_class_counts),
            'class_update_steps':dict(client.pending_class_update_steps),
            'accumulated_steps':client.pending_upload_steps,
            'accumulated_classes':sorted(client.pending_upload_classes),
            'anchor_version':client.model_version}


def aggregate(updates,base,method,*,clients=None,round_id=0,old_gradient=None):
    if not updates:raise RuntimeError('No client updates to aggregate')
    if method=='uniform':
        weights=[1./len(updates)]*len(updates)
    else:
        counts=[u['examples'] for u in updates]
        total=sum(counts)
        if total<=0:raise RuntimeError('FedAvg has no contributing examples')
        weights=[n/total for n in counts]
    if method=='class_balanced':
        from collections import Counter
        total_by_class=Counter()
        for u in updates:total_by_class.update(u['class_counts'])
        scores=[sum(v/total_by_class[c] for c,v in u['class_counts'].items())
                for u in updates]
        weights=[x/sum(scores) for x in scores]
    if method=='inverse_participation':
        if clients is None:raise ValueError('Client histories are required')
        scores=[u['examples']/max(1,clients[u['client_id']].selection_count)
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
    """FedAvgM and FedAdam using the FedAvg mean model delta.

    For FedAvg and uniform weighting, the returned mean model is unchanged.
    Non-floating buffers follow the chosen aggregate and are not optimized.
    """
    if method not in {'fedadam','fedavgm'}:
        return mean,state
    state={'step':state.get('step',0)+1,
           'm':dict(state.get('m',{})),
           'v':dict(state.get('v',{}))}
    out={}
    for name, old in base.items():
        if not old.is_floating_point():
            out[name]=mean[name].clone();continue
        delta=(mean[name].to(torch.float64)-old.to(torch.float64))
        prev_m=state['m'].get(name,torch.zeros_like(delta))
        momentum=(cfg.server_beta1*prev_m+delta if method=='fedavgm'
                  else cfg.server_beta1*prev_m+(1-cfg.server_beta1)*delta)
        state['m'][name]=momentum
        if method=='fedavgm':
            update=cfg.server_lr*momentum
        else:
            prev_v=state['v'].get(name,torch.zeros_like(delta))
            variance=cfg.server_beta2*prev_v+(1-cfg.server_beta2)*delta.square()
            state['v'][name]=variance
            # Bias correction makes the first server step well-conditioned.
            m_hat=momentum/(1-cfg.server_beta1**state['step'])
            v_hat=variance/(1-cfg.server_beta2**state['step'])
            update=cfg.server_lr*m_hat/(v_hat.sqrt()+cfg.fedadam_epsilon)
        out[name]=(old.to(torch.float64)+update).to(old.dtype)
    if not check_finite(out):raise RuntimeError('Nonfinite adaptive server update')
    return out,state


def save_round_state(path,cfg,round_id,global_state,version,clients,observed,peak,
                     integrated,history,rng,server_optimizer_state,server_class_update_counts=None):
    """Durable full-state checkpoint for exact logical-round resumption."""
    payload={'config':cfg.to_dict(),'protocol_version':PROTOCOL_VERSION,'last_completed_round':round_id,
             'global_state':global_state,'version':version,
             'clients':{cid:asdict(s) for cid,s in clients.items()},
             'observed':observed,'peak':peak,'integrated':integrated,
             'history':history,'selection_rng':rng.bit_generator.state,
             'server_optimizer_state':server_optimizer_state,
             'server_class_update_counts':dict(server_class_update_counts or {}),
             'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
             'torch_rng':torch.get_rng_state(),
             'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,tmp);os.replace(tmp,path)


def atomic_json(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as fh:json.dump(obj,fh,indent=2,sort_keys=True,default=str)
    os.replace(tmp,path)


def save_checkpoint(path,weights,cfg,version,states=None,full_state=False):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+'.tmp')
    payload={'model_state':weights,'model_version':version,'protocol_version':PROTOCOL_VERSION,
             'configuration':cfg.to_dict()}
    if full_state:
        payload['clients']={cid:asdict(s) for cid,s in states.items()}
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
            if old.get('protocol_version')!=PROTOCOL_VERSION:
                raise RuntimeError('Completed run uses incompatible local-training protocol; use a new output root')
            if not config_file.exists() or json.loads(config_file.read_text())!=cfg.to_dict():
                raise RuntimeError(f'Completed job ID collision: {job}')
            if any(not (run_dir/name).exists() for name in
                   ('model_final.pt','summary.json','metrics.json')):
                raise RuntimeError('Completion marker exists but output artifacts are missing')
            return {'status':'already_completed','run_dir':str(run_dir)}
    atomic_json(config_file,cfg.to_dict())
    atomic_json(status_file,{'status':'running','protocol_version':PROTOCOL_VERSION,'started_at':time.time()})
    try:
        # Stable independent randomness for independent processes.
        random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg.seed)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bank=get_data_bank(cfg.dataset,data_root,download=download,seed=cfg.seed,class_subset=cfg.class_subset)
        scenario=(compile_stream_scenario(bank,cfg) if cfg.stream_protocol
                  else compile_scenario(bank,cfg))
        torch.save({'arrivals':[asdict(a) for a in scenario.arrivals],
                    'stages':scenario.stages,
                    'original_labels':bank.original_labels,
                    'duplicated_training_samples':scenario.duplicate_sample_ids,
                    'unused_training_samples':scenario.unused_sample_ids},run_dir/'scenario.pt')
        clients={cid:LocalState(cid) for cid in range(cfg.num_clients)}
        model=create_model(cfg.model,bank.num_classes,allow_test_model=(cfg.dataset=='synthetic')).to(device)
        backbone_description=getattr(model,'backbone_description',{'test_only':True})
        if cfg.dataset=='cifar100':
            # Every job in a sequential research campaign must start from
            # the identical published pretrained feature extractor.
            reference=Path(output_root)/'deit_tiny_pretrained_reference.json'
            if reference.exists():
                if json.loads(reference.read_text())!=backbone_description:
                    raise RuntimeError('Pretrained ViT reference differs from prior jobs; use a new experiment root')
            else:
                atomic_json(reference,backbone_description)
        global_state=copy_cpu_state(model)
        # Initial model is shared at round zero. Subsequent downloads require
        # explicit communication, except deliberate full-sync ablations.
        for client in clients.values():
            client.receive(global_state,version=0)
        observed=set();peak={};history=[];version=0
        integrated=set();server_optimizer_state={}
        server_class_update_counts={}
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
            if saved.get('protocol_version')!=PROTOCOL_VERSION:
                raise RuntimeError('Cannot resume or fork a checkpoint from an incompatible training protocol')
            if fork_from_checkpoint:
                # Forks may only change the participation intervention and
                # destination identity. Dataset/model/training must be identical.
                allowed={'job_id','force_target','target_client','intervention_round',
                         'causal_diagnostics','gradient_diagnostics','checkpoint_every',
                         'fork_source_job','fork_checkpoint_round'}
                old={k:v for k,v in saved['config'].items() if k not in allowed}
                new={k:v for k,v in cfg.to_dict().items() if k not in allowed}
                if old!=new:
                    changed=[k for k in old if old[k]!=new.get(k)]
                    raise RuntimeError(f'Fork changes confounded settings: {changed}')
            elif saved['config']!=cfg.to_dict():
                raise RuntimeError('Resumption configuration does not match saved checkpoint')
            first_round=int(saved['last_completed_round'])+1
            global_state=saved['global_state'];version=saved['version']
            clients={int(cid):LocalState(**state) for cid,state in saved['clients'].items()}
            observed=set(saved['observed']);peak=saved['peak'];integrated=set(saved['integrated'])
            history=saved['history'];server_optimizer_state=saved['server_optimizer_state']
            server_class_update_counts=saved.get('server_class_update_counts',{})
            rng.bit_generator.state=saved['selection_rng']
            random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
            torch.set_rng_state(saved['torch_rng'])
            if torch.cuda.is_available() and saved['cuda_rng']:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        for round_id in range(first_round,total_rounds):
            round_start=time.monotonic()
            # Queue all client arrivals but do NOT expose later class labels to
            # the local learner before it has processed earlier class events.
            # This is essential for ER-ACE masking and true within-round CL.
            new_events={cid:[] for cid in clients}
            for arrival in scenario.arrivals_at(round_id):
                if cfg.stream_protocol:
                    new_events[arrival.client_id].append({
                        'indices':list(map(int,arrival.indices)),
                        'class_ids':tuple(map(int,arrival.labels)),
                        'global_stage':int(arrival.stage),
                        'is_recurrence':bool(arrival.recurrence)})
                    observed.update(arrival.labels)
                    continue
                for cls in arrival.labels:
                    ids=[int(i) for i in arrival.indices
                         if int(bank.train.targets[i])==int(cls)]
                    if ids:
                        new_events[arrival.client_id].append({
                            'indices':ids,'class_id':int(cls),
                            'global_stage':int(arrival.stage),
                            'is_recurrence':bool(arrival.recurrence)})
                # Global test exposure is independent of participation, and
                # includes labels from all arrivals in this communication round.
                observed.update(arrival.labels)
            current_stage=(round_id if cfg.stream_protocol else
                           min(cfg.stages-1,round_id//cfg.rounds_per_stage))
            old={c for c in observed if class_first_stage[c]<current_stage}
            new=set(observed)-old
            # Local training is unconditional on federation. Every client has
            # an independent persistent model, private CL state and unsent
            # update accumulator. Train arrivals sequentially, then retain the
            # final current experience for the next round when no new data arrive.
            local_details=[]
            local_trained_clients=[]
            for cid,client in clients.items():
                local_cfg=cfg
                if cfg.client_heterogeneity:
                    algorithms=('sgd','gem','agem','replay')
                    local_cfg=replace(cfg,method=algorithms[cid%len(algorithms)])
                raw_events=new_events[cid]
                events=[]
                if cfg.stream_protocol:
                    events=list(raw_events)
                    if len(events)>cfg.local_tasks_per_round:
                        raise AssertionError('Exceeded local task budget per server interval')
                    if any(len(set(e['class_ids']))<2 for e in events):
                        raise AssertionError('Singleton local task in strict stream')
                else:
                  for start in range(0,len(raw_events),2):
                      group=raw_events[start:start+2]
                      if len(group)==2 and group[0]['class_id']!=group[1]['class_id']:
                          events.append({'indices':group[0]['indices']+group[1]['indices'],
                              'class_ids':(group[0]['class_id'],group[1]['class_id']),
                              'global_stage':max(g['global_stage'] for g in group),
                              'is_recurrence':any(g['is_recurrence'] for g in group)})
                      else:
                          if cfg.class_pair_protocol=='strict':
                              raise ValueError(f'Client {cid} has insufficient distinct class '
                                  f'labels for a two-class episode at round {round_id}: {group}')
                          events.extend({**g,'class_ids':(g['class_id'],)} for g in group)
                if cfg.stream_protocol and not events:
                    continue   # No fabricated training events after stream exhaustion.
                if not events and not client.current_indices:
                    continue
                # With retain-all, the union is the active data outside of new
                # local arrivals. In all cases, individual new class events are
                # learned separately and in arrival order.
                jobs_to_train=events if events else [{'indices':list(client.current_indices),
                    'class_id':None,'class_ids':(),
                    'global_stage':current_stage,'is_recurrence':False}]
                all_available=list(client.current_indices)
                for event_index,event in enumerate(jobs_to_train):
                    if events:
                        classes=tuple(int(c) for c in event['class_ids'])
                        client.arrival(event['indices'],classes,event['global_stage'])
                        for cls in classes:
                            if event['is_recurrence']:
                                client.recurrence_rounds[cls]=round_id
                        all_available.extend(event['indices'])
                        client.local_experience+=1
                        for cls in classes:
                            client.first_seen_local_experience.setdefault(cls,client.local_experience)
                        client.local_training_events.append({'round':round_id,
                            'local_experience':client.local_experience,
                            'class':classes[0] if len(classes)==1 else None,
                            'classes':list(classes),
                            'global_stage':event['global_stage'],
                            'recurrence':event['is_recurrence'],
                            'selected_for_upload':False})
                        client.current_indices=list(event['indices'])
                        if cfg.pending_policy=='bounded' and cfg.max_pending_samples>0:
                            client.current_indices=client.current_indices[-cfg.max_pending_samples:]
                    if cfg.learning_setting=='task_il':
                        current_event_stage=event['global_stage']
                        task_labels=scenario.stages[min(current_event_stage,len(scenario.stages)-1)]
                    else:
                        task_labels=None
                    result=local_train(model,client,bank,local_cfg,device,round_id,
                        task_classes=task_labels,experience_end=bool(events),
                        active_pair_classes=(tuple(event['class_ids']) if events and
                                             len(event['class_ids'])>=2 and not cfg.label_noise_rate else None))
                    if result is None:
                        continue
                    # Advance the private weights *without* downloading the
                    # current global model or exposing raw private memory.
                    client.shared_weights=result['weights']
                    client.pending_upload_examples+=result['examples']
                    client.pending_upload_steps+=result['steps']
                    for cls,count in result['class_update_steps'].items():
                        client.pending_class_update_steps[int(cls)]=(
                            client.pending_class_update_steps.get(int(cls),0)+int(count))
                    client.pending_upload_classes.update(result['class_counts'])
                    for cls,count in result['class_counts'].items():
                        client.pending_upload_class_counts[int(cls)]=(
                            client.pending_upload_class_counts.get(int(cls),0)+count)
                    local_trained_clients.append(cid)
                    local_details.append({'client':cid,'local_experience':client.local_experience,
                        'is_selected':False,
                        **{k:v for k,v in result.items() if k!='weights'}})
                # The next round's current training distribution must respect
                # configured data-retention semantics, not the event queue.
                if events and cfg.pending_policy=='retain_all':
                    client.current_indices=list(dict.fromkeys(all_available))
                if events and cfg.pending_policy=='bounded' and cfg.max_pending_samples>0:
                    client.current_indices=client.current_indices[-cfg.max_pending_samples:]
            class_relationships=None
            if cfg.selector in {'feature_max','feature_min','gradient_align','gradient_conflict'}:
                # Simulation-privileged: historical data that have *arrived*,
                # never future or test data. Feature and gradient selectors
                # are oracle diagnostics, not privacy-preserving methods.
                model.load_state_dict(global_state)
                model.to(device)
                arrived=set().union(*(c.seen_indices for c in clients.values()))
                if cfg.selector.startswith('feature'):
                    vectors=arrived_feature_means(model,bank,observed,device,
                                                  allowed_indices=arrived)
                else:
                    vectors=class_gradient_representations(model,bank,observed,device,
                                                           allowed_indices=arrived)
                class_relationships=pairwise_vector_cosines(vectors)
            risks={}
            if history:
                risks={int(c):max(0.,peak.get(int(c),0.)-float(v))
                       for c,v in history[-1]['per_class'].items()}
            selected=select_clients(clients,cfg,round_id,rng,old_classes=old,new_classes=new,
                                    coarse_by_fine=bank.coarse_by_fine,
                                    relationships=class_relationships,class_forgetting=risks)
            if cfg.stream_protocol:
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
                uploaded=anchored_upload(client,global_state)
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
                evaluation_bank=DataBank(bank.train,domain_test,bank.num_classes,
                                         bank.coarse_by_fine,bank.original_labels)
            if cfg.spurious_strength:
                from .data import DataBank
                from .domains import SpuriousTest
                evaluation_bank=DataBank(bank.train,
                    SpuriousTest(evaluation_bank.test,cfg.spurious_strength),
                    bank.num_classes,bank.coarse_by_fine,bank.original_labels)
            pre_aggregation_metrics=None
            pre_representation=None
            if accepted and cfg.prepost_diagnostics:
                model.load_state_dict(global_state)
                model.to(device)
                # Diagnostics only; never fed back to training or selection.
                pre_aggregation_metrics,_=evaluate(model,evaluation_bank,observed,cfg.batch_size,
                                                    device,peak.copy(),integrated=integrated,
                                                    old=old,new=new,
                                                    task_class_groups=scenario.stages if cfg.learning_setting=='task_il' else None)
            if accepted and cfg.representation_diagnostics and old:
                model.load_state_dict(global_state)
                model.to(device)
                arrived=set().union(*(c.seen_indices for c in clients.values()))
                pre_representation=arrived_feature_means(model,bank,old,device,
                                                          allowed_indices=arrived)
            if accepted:
                old_gradient=None
                if cfg.aggregator in {'retention_aware'} and old:
                    model.load_state_dict(global_state)
                    arrived=set().union(*(c.seen_indices for c in clients.values()))
                    vectors=class_gradient_representations(model,bank,old,device,
                                                           allowed_indices=arrived)
                    if vectors:old_gradient=torch.stack(list(vectors.values())).mean(0)
                mean=aggregate(accepted,global_state,cfg.aggregator,
                               clients=clients,round_id=round_id,old_gradient=old_gradient)
                candidate,new_server_state=server_optimizer_step(
                    global_state,mean,cfg.aggregator,cfg,server_optimizer_state)
                if cfg.classwise_head_merge:
                    candidate=classwise_merge(global_state,candidate,accepted,
                        server_class_update_counts,classifier_parameter_names(model),
                        method=cfg.aggregator)
                for u in accepted:
                    for cls,n in u.get('class_update_steps',{}).items():
                        server_class_update_counts[int(cls)]=(
                            server_class_update_counts.get(int(cls),0)+int(n))

                # Candidate is built without mutating the committed server state.
                for cid in selected:
                    clients[cid].receive(candidate,version+1,
                        clear_memory=cfg.reset_memory_on_sync)
                    assert all(torch.equal(clients[cid].shared_weights[k],v) for k,v in candidate.items())
                global_state=candidate
                server_optimizer_state=new_server_state
                version+=1
            else:
                # All active clients still receive the current committed model.
                for cid in selected:clients[cid].receive(global_state,version,
                    clear_memory=cfg.reset_memory_on_sync)
            if cfg.full_sync:
                for cid in clients:
                    clients[cid].receive(global_state,version,
                        clear_memory=cfg.reset_memory_on_sync)
            for cid in selected:
                clients[cid].last_selected=round_id
                clients[cid].selection_count+=1
            model.load_state_dict(global_state)
            model.to(device)
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
                arrived=set().union(*(c.seen_indices for c in clients.values()))
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
                    arrived=set().union(*(c.seen_indices for c in clients.values()))
                    model.load_state_dict(base_global_state)
                    vectors=class_gradient_representations(model,bank,old,device,
                        allowed_indices=arrived)
                    if vectors:
                        mean_old_grad=torch.stack(list(vectors.values())).mean(0)
                        old_update_alignment={str(cid):vector_cosine(mean_old_grad,delta)
                            for cid,delta in deltas.items()}
                    model.load_state_dict(global_state)
            result,peak=evaluate(model,evaluation_bank,observed,cfg.batch_size,device,peak,
                                 integrated=integrated,old=old,new=new,
                                 task_class_groups=scenario.stages if cfg.learning_setting=='task_il' else None)
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
                           'protocol_version':PROTOCOL_VERSION,
                           'local_details':local_details,
                           'seconds':time.monotonic()-round_start,
                           'global_test_includes_inactive':True,
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
                                 server_optimizer_state,server_class_update_counts)
                if cfg.checkpoint_every:
                    save_round_state(run_dir/f'checkpoint_round_{round_id+1:04d}.pt',
                                     cfg,round_id,global_state,version,
                                     clients,observed,peak,integrated,history,rng,
                                     server_optimizer_state,server_class_update_counts)
            if (cfg.stream_protocol or (round_id+1)%cfg.rounds_per_stage==0):
                save_checkpoint(run_dir/f'model_stage_{current_stage+1:02d}.pt',global_state,
                                cfg,version,clients,full_state=True)
            if verbose:
                print(f'[{job}] round {round_id+1}/{total_rounds} acc={result["accuracy"]:.4f} '
                      f'forgetting={result["forgetting"]:.4f} clients={len(selected)}',flush=True)
        save_checkpoint(run_dir/'model_final.pt',global_state,cfg,version,clients,full_state=True)
        atomic_json(run_dir/'summary.json',{'job_id':job,'dataset':cfg.dataset,
                    'fork_source':str(fork_from_checkpoint) if fork_from_checkpoint else None,
                    'device':str(device),'rounds':total_rounds,'backbone':backbone_description,'final_metrics':history[-1],
                    'model_final':str(run_dir/'model_final.pt')})
        atomic_json(status_file,{'status':'completed','protocol_version':PROTOCOL_VERSION,
                                'finished_at':time.time(),
                                'rounds':total_rounds})
        return {'status':'completed','run_dir':str(run_dir),'final_accuracy':history[-1]['accuracy']}
    except Exception as exc:
        atomic_json(status_file,{'status':'failed','protocol_version':PROTOCOL_VERSION,
                                  'error':repr(exc),'traceback':traceback.format_exc()})
        raise
