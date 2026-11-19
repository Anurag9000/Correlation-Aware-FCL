"""Concurrent client learning with a hard barrier between task ordinals.

Five independent client model instances can submit work simultaneously on one
GPU using independent CUDA streams. In the two-arm probe, two Python processes
share that same GPU. CUDA decides whether individual kernels overlap physically;
the contract guaranteed here is concurrent client execution and task barriers.

FedAvg and CL-state synchronization remain in simulator.py *after* this function
returns, so disconnected clients still train but cannot upload or download.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier
from time import perf_counter
import torch

from .continual import local_train
from .models import create_model


@dataclass
class ClientRuntime:
    model: torch.nn.Module
    stream: object | None


def create_client_runtimes(cfg, bank, clients, device):
    """Create a distinct persistent model object and CUDA stream per client.

    New random model instances have their weights replaced with each client's
    own persistent weights. Preserve the simulator's global RNG to keep other
    seed-controlled steps unaffected by worker-model construction.
    """
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
    runtimes = {}
    try:
        for cid, client in sorted(clients.items()):
            model = create_model(
                cfg.model, bank.num_classes,
                allow_test_model=(cfg.dataset == 'synthetic')).to(device)
            if client.shared_weights is None:
                raise RuntimeError(f'Client {cid} has no initialized local weights')
            model.load_state_dict(client.shared_weights)
            stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
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
               task_classes, barrier):
    # All clients with a task in the current ordinal must reach this barrier.
    barrier.wait(timeout=300)
    start = perf_counter()
    if device.type == 'cuda':
        with torch.cuda.device(device), torch.cuda.stream(runtime.stream):
            result = local_train(
                runtime.model, client, bank, cfg, device, round_id,
                task_classes=task_classes, experience_end=event['is_actual'],
                active_pair_classes=event['active_pair'])
            # Never expose an unfinished client update to the server or Task 2.
            runtime.stream.synchronize()
    else:
        result = local_train(
            runtime.model, client, bank, cfg, device, round_id,
            task_classes=task_classes, experience_end=event['is_actual'],
            active_pair_classes=event['active_pair'])
    return result, start, perf_counter()


def _prepare_events(cfg, client, cid, raw_events, current_stage):
    if cfg.stream_protocol:
        events = list(raw_events)
        if len(events) > cfg.local_tasks_per_round:
            raise AssertionError('Exceeded local task budget per server interval')
        if any(len(set(e['class_ids'])) < 2 for e in events):
            raise AssertionError('Singleton local task in strict stream')
    else:
        events = []
        for start in range(0, len(raw_events), 2):
            group = raw_events[start:start + 2]
            if len(group) == 2 and group[0]['class_id'] != group[1]['class_id']:
                events.append({
                    'indices': group[0]['indices'] + group[1]['indices'],
                    'class_ids': (group[0]['class_id'], group[1]['class_id']),
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
    return ([{
        'indices': list(client.current_indices), 'class_id': None,
        'class_ids': (), 'global_stage': current_stage, 'is_recurrence': False
    }], False)


def train_round_task_barriers(*, runtimes, executor, clients, bank, cfg, device,
                              round_id, current_stage, scenario, new_events):
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
        start_barrier = Barrier(len(active))
        pending = []
        for cid in active:
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
                    'classes': list(classes), 'global_stage': raw['global_stage'],
                    'recurrence': raw['is_recurrence'],
                    'selected_for_upload': False})
                client.current_indices = list(raw['indices'])
                if cfg.pending_policy == 'bounded' and cfg.max_pending_samples > 0:
                    client.current_indices = client.current_indices[-cfg.max_pending_samples:]
            if cfg.learning_setting == 'task_il':
                stage = raw['global_stage']
                task_labels = scenario.stages[min(stage, len(scenario.stages) - 1)]
            else:
                task_labels = None
            local_cfg = cfg
            if cfg.client_heterogeneity:
                from dataclasses import replace
                methods = ('sgd', 'gem', 'agem', 'replay')
                local_cfg = replace(cfg, method=methods[cid % len(methods)])
            event = {
                'is_actual': bool(has_new),
                'active_pair': (tuple(raw['class_ids'])
                                if has_new and len(raw['class_ids']) >= 2
                                and not cfg.label_noise_rate else None),
            }
            pending.append((cid, executor.submit(
                _train_one, runtimes[cid], client, bank, local_cfg, device,
                round_id, event, task_labels, start_barrier)))
        # Hard barrier: all Task N workers finish before any Task N+1 starts.
        outcomes = []
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
