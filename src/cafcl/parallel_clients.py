"""Client learning with hard barriers between task ordinals.

Five independent client models persist on the GPU. With concurrency set to
five, they submit work simultaneously on independent CUDA streams. A bounded concurrency setting runs
clients in waves; concurrency one shares a CUDA stream to reuse allocator pools.
CUDA decides how kernels from concurrent experiment processes overlap
physically; the campaign contract is simultaneous live workers and intact
per-client batch, optimizer, and synchronization semantics.

FedAvg and CL-state synchronization remain in simulator.py *after* this function
returns, so disconnected clients still train but cannot upload or download.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Barrier
from time import perf_counter
import torch

from .continual import local_train
from .models import create_model


@dataclass
class ClientRuntime:
    model: torch.nn.Module
    stream: object | None


def create_client_runtimes(cfg, bank, clients, device, task_class_sets):
    """Create one persistent model per client and the configured CUDA streams.

    Default execution uses a distinct CUDA stream per client. Bounded
    single-client waves share one stream so the caching allocator can reuse
    blocks between independent clients.

    New random model instances have their weights replaced with each client's
    own persistent weights. Preserve the simulator's global RNG to keep other
    seed-controlled steps unaffected by worker-model construction.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    runtimes = {}
    shared_stream = None
    if device.type == 'cuda' and cfg.client_training_concurrency == 1:
        shared_stream = torch.cuda.Stream(device=device)
    try:
        for cid, client in sorted(clients.items()):
            model = create_model(
                cfg.model, bank.num_classes,
                allow_test_model=(cfg.dataset == 'synthetic'),
                task_class_sets=task_class_sets).to(device)
            if client.shared_weights is None:
                raise RuntimeError(f'Client {cid} has no initialized local weights')
            model.load_state_dict(client.shared_weights)
            stream = (shared_stream if shared_stream is not None else
                      torch.cuda.Stream(device=device) if device.type == 'cuda' else None)
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(device))
            runtimes[cid] = ClientRuntime(model, stream)
        if len({id(v.model) for v in runtimes.values()}) != len(clients):
            raise RuntimeError('Client models must be independent PyTorch objects')
        return runtimes
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def _train_one(runtime, client, bank, cfg, device, round_id, event,
               task_classes, task_id, barrier,known_task_sets=(),task_ids_by_index=None):
    # All clients with a task in the current ordinal must reach this barrier.
    barrier.wait(timeout=300)
    start = perf_counter()
    if cfg.learning_setting=='task_il' and task_id:
        runtime.model.set_active_task_id(task_id)
    if device.type == 'cuda':
        with torch.cuda.device(device), torch.cuda.stream(runtime.stream):
            result = local_train(
                runtime.model, client, bank, cfg, device, round_id,
                task_classes=task_classes, task_id=task_id,
                experience_end=event['is_actual'],
                active_pair_classes=event['active_pair'],
                padding_indices=event.get('padding_indices'),
                known_task_ids=known_task_sets,
                task_ids_by_index=task_ids_by_index)
            # Never expose an unfinished client update to the server or Task 2.
            runtime.stream.synchronize()
    else:
        result = local_train(
            runtime.model, client, bank, cfg, device, round_id,
            task_classes=task_classes, task_id=task_id,
            experience_end=event['is_actual'],
            active_pair_classes=event['active_pair'],
            padding_indices=event.get('padding_indices'),
            known_task_ids=known_task_sets,
            task_ids_by_index=task_ids_by_index)
    return result, start, perf_counter()


def _prepare_events(cfg, client, cid, raw_events, current_stage):
    if cfg.stream_protocol:
        events = list(raw_events)
        expected=cfg.local_tasks_per_round
        if len(events)>expected:
            raise AssertionError('More client tasks arrived than configured for this sync')
        if any(len(set(e['class_ids'])) < 2 for e in events):
            raise AssertionError('Singleton local task in strict stream')
    else:
        # Keep task-head construction, per-example provenance and local event
        # grouping on the same canonical ordering, independent of arrival-list
        # iteration order or a semantic class-order override.
        raw_events=sorted(raw_events,key=lambda e:(int(e['global_stage']),
            tuple(map(int,e['class_ids'])),tuple(map(int,e['indices'][:1]))))
        events = []
        for start in range(0, len(raw_events), 2):
            group = raw_events[start:start + 2]
            if len(group) == 2 and group[0]['class_id'] != group[1]['class_id']:
                events.append({
                    'indices': group[0]['indices'] + group[1]['indices'],
                    'class_ids': (group[0]['class_id'], group[1]['class_id']),
                    'task_id': group[0]['task_id'],
                    'global_stage': max(g['global_stage'] for g in group),
                    'is_recurrence': any(g['is_recurrence'] for g in group)})
            else:
                if cfg.class_pair_protocol == 'strict':
                    raise ValueError(f'Client {cid} has insufficient distinct class labels for a two-class episode: {group}')
                events.extend({**g, 'class_ids': (g['class_id'],)}
                              for g in group)
    if cfg.stream_protocol and not events:
        return [], False
    if not events and not client.current_indices:
        return [], False
    if events:
        return events, True
    previous_classes=()
    if cfg.learning_setting=='task_il' and client.local_training_events:
        previous_classes=tuple(client.local_training_events[-1].get('classes',()))
    return ([{
        'indices': list(client.current_indices), 'class_id': None,
        'class_ids': previous_classes,
        'task_id':client.local_training_events[-1].get('task_id'),
        'global_stage': current_stage, 'is_recurrence': False
    }], False)


def train_round_task_barriers(*, runtimes, executor, clients, bank, cfg, device,
                              round_id, current_stage, scenario, new_events,
                              known_task_sets=(),task_ids_by_index=None):
    """Train all clients' Task 1 concurrently, join; then Task 2, join; etc.

    The server is not involved during either task. Every client has its own
    private model and memory, including clients selected offline later.
    """
    prepared = {}
    has_new_events = {}
    all_available = {}
    for cid, client in sorted(clients.items()):
        events, has_new = _prepare_events(cfg, client, cid, new_events[cid], current_stage)
        if events:
            prepared[cid] = events
            has_new_events[cid] = has_new
            all_available[cid] = list(client.current_indices)
    local_details = []
    locally_trained = []
    barriers = []
    for task_index in range(max(map(len, prepared.values()), default=0)):
        active = [cid for cid in sorted(prepared) if task_index < len(prepared[cid])]
        if not active:
            continue
        concurrency = cfg.client_training_concurrency or len(active)
        outcomes = []
        execution_waves = []
        # Client updates are independent until the round barrier. Running them
        # in bounded waves preserves the same per-client batch/optimizer work
        # while limiting simultaneously live activations on a shared GPU.
        for offset in range(0, len(active), concurrency):
            wave = active[offset:offset + concurrency]
            execution_waves.append(wave)
            start_barrier = Barrier(len(wave))
            pending = []
            for cid in wave:
                client = clients[cid]
                raw = prepared[cid][task_index]
                has_new = has_new_events[cid]
                if has_new:
                    classes = tuple(int(c) for c in raw['class_ids'])
                    client.arrival(raw['indices'], classes, raw['global_stage'])
                    for cls in classes:
                        if raw['is_recurrence']:
                            client.recurrence_rounds[cls] = round_id
                    all_available[cid].extend(raw['indices'])
                    client.local_experience += 1
                    for idx in raw['indices']:
                        client.sample_first_local_experience.setdefault(
                            int(idx), client.local_experience)
                    for cls in classes:
                        client.first_seen_local_experience.setdefault(
                            cls, client.local_experience)
                    client.local_training_events.append({
                        'round': round_id, 'local_experience': client.local_experience,
                        'class': classes[0] if len(classes) == 1 else None,
                        'classes': list(classes), 'task_id':raw['task_id'],
                        'global_stage': raw['global_stage'],
                        'recurrence': raw['is_recurrence'],
                        'selected_for_upload': False})
                    client.current_indices = list(raw['indices'])
                    if cfg.pending_policy == 'bounded' and cfg.max_pending_samples > 0:
                        client.current_indices = client.current_indices[-cfg.max_pending_samples:]
                if cfg.learning_setting == 'task_il':
                    task_labels = tuple(sorted(set(map(int,raw['class_ids']))))
                    task_id=raw['task_id']
                else:
                    task_labels = None
                    task_id=None
                local_cfg = replace(cfg, epochs=1) if cfg.stream_sync_every_batch else cfg
                if cfg.client_heterogeneity:
                    methods = ('sgd', 'gem', 'agem', 'replay')
                    client_method=methods[cid % len(methods)]
                    policy=(cfg.replay_update_policy if cfg.method=='replay'
                            else 'sampled_mixed')
                    local_cfg = replace(local_cfg, method=client_method,
                        replay_update_policy=(policy if client_method=='replay'
                                              else local_cfg.replay_update_policy))
                event = {
                    'is_actual': bool(has_new),
                    'active_pair': (tuple(raw['class_ids'])
                                    if has_new and len(raw['class_ids']) >= 2
                                    and not cfg.label_noise_rate else None),
                    'padding_indices': list(map(int,raw.get('padding_indices',()))),
                }
                pending.append((cid, executor.submit(
                    _train_one, runtimes[cid], client, bank, local_cfg, device,
                    round_id, event, task_labels, task_id, start_barrier,known_task_sets,
                    task_ids_by_index)))
            # Every active client completes Task N before any Task N+1 starts.
            for cid, future in pending:
                result, start, end = future.result()
                outcomes.append((cid, result, start, end))
        begins, ends = [], []
        for cid, result, start, end in outcomes:
            begins.append(start)
            ends.append(end)
            if result is None:
                continue
            client = clients[cid]
            client.shared_weights = result['weights']
            client.pending_upload_examples += result['examples']
            client.pending_upload_steps += result['steps']
            for cls, count in result['class_update_steps'].items():
                cls = int(cls)
                client.pending_class_update_steps[cls] = (
                    client.pending_class_update_steps.get(cls, 0) + int(count))
            for head,count in result.get('task_head_examples',{}).items():
                client.pending_task_head_examples[head]=(
                    client.pending_task_head_examples.get(head,0.)+float(count))
            client.pending_upload_classes.update(result['class_counts'])
            for cls, count in result['class_counts'].items():
                cls = int(cls)
                client.pending_upload_class_counts[cls] = (
                    client.pending_upload_class_counts.get(cls, 0) + count)
            locally_trained.append(cid)
            local_details.append({
                'client': cid, 'local_experience': client.local_experience,
                'task_ordinal': task_index + 1,
                'is_selected': False,
                **{k: v for k, v in result.items() if k != 'weights'}})
        barriers.append({
            'task_ordinal': task_index + 1,
            'client_ids': active,
            'completed_count': len(outcomes),
            'execution_waves': execution_waves,
            'start_spread_seconds': max(begins) - min(begins),
            'overlap_window_seconds': max(0., min(ends) - max(begins)),
            'wall_seconds': max(ends) - min(begins),
        })
    for cid in prepared:
        client = clients[cid]
        if has_new_events[cid] and cfg.pending_policy == 'retain_all':
            client.current_indices = list(dict.fromkeys(all_available[cid]))
        elif (has_new_events[cid] and cfg.pending_policy == 'bounded'
              and cfg.max_pending_samples > 0):
            client.current_indices = client.current_indices[-cfg.max_pending_samples:]
    return local_details, locally_trained, barriers
