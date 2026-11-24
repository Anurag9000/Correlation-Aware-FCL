"""Read-only diagnostic reconstruction and compact per-run figures."""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math


def average(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def accuracy_for_task(metrics,task,setting):
    if not metrics:return None
    if setting=='task_il':
        return metrics.get('task_conditioned',{}).get('per_head',{}).get(task['head_id'],{}).get('accuracy')
    per=metrics.get('per_class',{})
    return average([per.get(str(c),per.get(c)) for c in task['classes']])


def compare_task_metrics(before,after,task,setting):
    a=accuracy_for_task(before,task,setting);b=accuracy_for_task(after,task,setting)
    if a is None or b is None:return None
    if setting=='class_il':
        # Do not call differences caused by extra competitors pure forgetting.
        if set(before['observed_classes'])!=set(after['observed_classes']):
            return None
    return b-a


def prediction_stats(counts,observed):
    counts=[int(counts.get(str(c),counts.get(c,0))) for c in observed]
    total=sum(counts)
    if not total:return {'coverage':None,'dominance':None,'entropy':None,'effective_classes':None}
    p=[n/total for n in counts if n]
    h=-sum(x*math.log(x) for x in p)
    return {'coverage':len(p)/len(observed),'dominance':max(p),
            'entropy':h,'effective_classes':math.exp(h)}


def analyze_run(directory,output=None,figures=True):
    root=Path(directory);out=Path(output) if output else root/'joint_analysis'
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'config.json').read_text())
    plan=json.loads((root/'joint_plan.json').read_text())
    setting=cfg['learning_setting']
    tasks={x['occurrence_id']:x for x in plan['tasks']}
    local_rows=[];global_rows=[];latency=[];class_rows=[];task_rows=[];local_history=[]
    peaks={};first_integrated={};local_peak={};history_peaks={};global_head_peaks={};integrated_heads=set();seen_classes=set()
    from .knowledge_audit import audit_knowledge
    ledger=audit_knowledge(root,plan)
    for interval in plan['intervals']:
        r=interval['interval_id']
        trace=json.loads((root/f'joint_interval_{r:05d}.json').read_text())
        knowledge=json.loads((root/f'joint_knowledge_{r:05d}.json').read_text())
        comm_path=root/f'joint_communication_{r:05d}.json'
        comm=json.loads(comm_path.read_text()) if comm_path.exists() else {}
        local_final=comm.get('private_pre_upload_validation',{})
        for d in trace['local_occurrences']:
            task=tasks[d['occurrence_id']];cid=task['client_id']
            pre=d.get('local_validation_pre_task');post=d.get('local_validation_post_task')
            end=local_final.get(str(cid))
            a=accuracy_for_task(post,task,setting)
            change=compare_task_metrics(pre,post,task,setting)
            retention=compare_task_metrics(post,end,task,setting)
            # Task-conditioned historical head scores have fixed competitors.
            # For Class-IL, only identical observed-label universes are compared.
            peak_drop=None
            if a is not None and setting=='task_il':
                key=(cid,task['head_id'])
                if key in local_peak:peak_drop=max(0.,local_peak[key]-a)
                local_peak[key]=max(a,local_peak.get(key,0.))
            # Every locally visible historical head is scored after each local
            # experience, not merely when its own task recurs. Class-IL keys
            # include the competitor universe to prevent invalid comparisons.
            if post and (not d.get('simultaneous_access') or d.get('pooled_group_training')):
                if setting=='task_il':
                    values={h:v['accuracy'] for h,v in post.get('task_conditioned',{}).get('per_head',{}).items()}
                    context='task_conditioned'
                else:
                    values=post.get('per_class',{});context=','.join(map(str,sorted(post['observed_classes'])))
                for target,value in values.items():
                    if value is None:continue
                    key=(cid,target,context);prior=history_peaks.get(key)
                    local_history.append({'interval':r,'atomic_step':task['atomic_step'],'client':cid,
                        'target':target,'evaluation_context':context,'accuracy':value,
                        'prior_peak_drop':None if prior is None else max(0.,prior-value)})
                    history_peaks[key]=max(value,prior if prior is not None else 0.)
            local_rows.append({'interval':r,'atomic_step':task['atomic_step'],'client':cid,
                'occurrence_id':task['occurrence_id'],'semantic_task':task['semantic_task_id'],
                'current_accuracy_after':a,'acquisition':change,
                'signed_retention_drop_to_upload':None if retention is None else -retention,
                'same_head_prior_peak_drop':peak_drop,'optimizer_steps':d['steps'],
                'simultaneous_access':d.get('simultaneous_access',False),
                'pooled_group_id':d.get('pooled_group_id'),
                'pooled_group_optimizer_steps':(d.get('pooled_group_training') or {}).get('steps'),
                'fresh_unique_examples':d['examples'],'replay_draws':d.get('replay_sampled_examples')})
        for oid,receipt in knowledge['acknowledged_occurrences'].items():
            if receipt.get('head_id'):integrated_heads.add(receipt['head_id'])
            latency.append({'occurrence_id':oid,'client':receipt['client'],
                'atomic_step':receipt['atomic_step'],'trained_interval':receipt['trained_interval'],
                'accepted_interval':r,'server_version':receipt['accepted_server_version'],
                'interval_latency':r-receipt['trained_interval'],
                'atomic_step_latency':sum(len(x['atomic_steps']) for x in plan['intervals'][:r+1])-1-receipt['atomic_step']})
        report=json.loads((root/'per_sync_accuracy'/f'sync_{r+1:03d}.json').read_text())
        counts=report['predicted_class_counts'];observed=report['observed_classes']
        integrated=set(report['server_integrated_classes'])
        per={}
        for entry in report['entries']:per.update({int(c):float(a) for c,a in entry['per_class'].items()})
        drops=[]
        for c in sorted(integrated):
            if c not in per:continue
            first_integrated.setdefault(c,r)
            previous=peaks.get(c)
            drop=None if previous is None else max(0.,previous-per[c])
            if drop is not None:drops.append(drop)
            peaks[c]=max(per[c],peaks.get(c,0.))
            class_rows.append({'interval':r,'class_id':c,'accuracy':per[c],
                'peak_drop_after_integration':drop,'first_integrated_interval':first_integrated[c],
                'distinct_client_owners':len(plan['overlap_audit']['class_owners'][str(c)])})
        current_tasks=[tasks[oid] for oid in interval['occurrences']]
        current_classes={c for t in current_tasks for c in t['classes']}
        new_classes=current_classes-seen_classes;historical_only=seen_classes-current_classes
        seen_classes.update(current_classes)
        task_test=comm.get('global_task_conditioned_test') or {}
        head_scores=task_test.get('per_head',{})
        current_task_scores=[]
        for t in current_tasks:
            score=(head_scores.get(t['head_id'],{}).get('accuracy') if setting=='task_il' else
                   average([per.get(c) for c in t['classes']]))
            current_task_scores.append(score)
            task_rows.append({'interval':r,'occurrence_id':t['occurrence_id'],'client':t['client_id'],
                'semantic_task_id':t['semantic_task_id'],'head_id':t['head_id'],'current_task_test_accuracy':score,
                'evaluation_protocol':'task_conditioned' if setting=='task_il' else 'class_il_seen_class_recall'})
        head_drops=[]
        for h in integrated_heads:
            value=head_scores.get(h,{}).get('accuracy')
            if value is None:continue
            prior=global_head_peaks.get(h)
            if prior is not None:head_drops.append(max(0.,prior-value))
            global_head_peaks[h]=max(value,prior if prior is not None else 0.)
        resource_path=root/f'joint_resources_{r:05d}.json'
        resource=json.loads(resource_path.read_text()) if resource_path.exists() else {}
        before=comm.get('global_validation_before');after=comm.get('global_validation_after')
        global_rows.append({'interval':r,'server_version':report['version'],
            'atomic_steps_completed':sum(len(x['atomic_steps']) for x in plan['intervals'][:r+1]),
            'test_accuracy':report['test_accuracy'],'validation_accuracy':report['validation_accuracy'],
            'integrated_only_test_accuracy':report['integrated_only_test_accuracy'],
            'integrated_class_peak_drop':average(drops),
            'integrated_task_head_peak_drop':average(head_drops),
            'current_task_average_test_accuracy':average(current_task_scores),
            'current_class_macro_test_accuracy':average([per.get(c) for c in current_classes]),
            'new_class_macro_test_accuracy':average([per.get(c) for c in new_classes]),
            'historical_noncurrent_class_macro_test_accuracy':average([per.get(c) for c in historical_only]),
            'optimizer_steps':resource.get('work_totals',{}).get('optimizer_steps'),
            'model_upload_tensor_bytes':resource.get('model_upload_tensor_bytes'),
            'model_download_tensor_bytes':resource.get('model_download_tensor_bytes'),
            'global_validation_gain':None if not before or not after else after['accuracy']-before['accuracy'],
            'mean_broadcast_gain':average([b['accuracy_change'] for b in comm.get('broadcast_validation',{}).values()]),
            **prediction_stats(counts,observed)})
    def write_csv(name,rows):
        keys=sorted({k for row in rows for k in row})
        with (out/name).open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=keys);w.writeheader();w.writerows(rows)
    write_csv('local_acquisition_retention.csv',local_rows)
    write_csv('global_trajectory.csv',global_rows)
    write_csv('integration_latency.csv',latency)
    write_csv('class_retention.csv',class_rows)
    write_csv('current_task_test.csv',task_rows)
    write_csv('local_historical_accuracy_matrix.csv',local_history)
    learning_rows=[row for row,interval in zip(global_rows,plan['intervals']) if interval['occurrences']]
    last_learning=learning_rows[-1] if learning_rows else {}
    final_private_path=root/'joint_final_private_test.json'
    final_private=json.loads(final_private_path.read_text()) if final_private_path.exists() else {}
    private_scores=[]
    for value in final_private.get('client_metrics',{}).values():
        if value is not None:
            private_scores.append(value.get('task_conditioned',{}).get('macro_head_accuracy') if setting=='task_il' else value.get('accuracy'))
    totals={}
    for key in ('optimizer_steps','model_upload_tensor_bytes','model_download_tensor_bytes'):
        values=[r[key] for r in global_rows]
        totals[key]=sum(values) if all(v is not None for v in values) else None
    summary={'learning_setting':setting,'control_kind':cfg.get('joint_control_kind','federated'),
             'last_learning_interval':last_learning,'resource_totals':totals,
             'mean_current_task_test_accuracy':average([r['current_task_average_test_accuracy'] for r in learning_rows]),
             'mean_historical_noncurrent_class_test_accuracy':average([r['historical_noncurrent_class_macro_test_accuracy'] for r in learning_rows]),
             'terminal_private_macro_client_accuracy':average(private_scores),
             'knowledge_ledger':ledger,'mean_local_acquisition':average([x['acquisition'] for x in local_rows]),
             'mean_signed_local_retention_drop':average([x['signed_retention_drop_to_upload'] for x in local_rows]),
             'mean_integration_latency_intervals':average([x['interval_latency'] for x in latency]),
             'mean_integration_latency_atomic_steps':average([x['atomic_step_latency'] for x in latency]),
             'mean_occurrence_weighted_current_task_test_accuracy':average([x['current_task_test_accuracy'] for x in task_rows]),
             'terminal':global_rows[-1],'scope':'Descriptive single-run diagnostics; unmatched Class-IL competitor sets excluded'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if figures:render_figures(plan,global_rows,local_rows,out)
    return summary


def render_figures(plan,global_rows,local_rows,out):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    fig=Figure(figsize=(10,4));FigureCanvasAgg(fig);ax=fig.add_subplot()
    for key in ('test_accuracy','current_task_average_test_accuracy','historical_noncurrent_class_macro_test_accuracy','integrated_class_peak_drop','coverage'):
        points=[(x['atomic_steps_completed'],x[key]) for x in global_rows if x[key] is not None]
        if points:ax.plot(*zip(*points),label=key)
    ax.set(xlabel='Atomic steps completed',ylabel='Fraction',ylim=(0,1));ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'global_trajectory.png',dpi=140)
    horizon=plan['overlap_audit']['atomic_steps'];k=plan['num_clients']
    matrix=np.full((k,horizon),np.nan)
    for t in plan['tasks']:matrix[t['client_id'],t['atomic_step']]=len(t['classes'])
    fig=Figure(figsize=(min(20,max(8,horizon*.12)),max(3,k*.3)));FigureCanvasAgg(fig);ax=fig.add_subplot()
    im=ax.imshow(matrix,aspect='auto',interpolation='nearest',cmap='viridis')
    for interval in plan['intervals']:
        if interval['atomic_steps'] and interval['upload_clients']:
            ax.axvline(max(interval['atomic_steps'])+.5,color='white',alpha=.45,linewidth=.6)
    fig.colorbar(im,ax=ax,label='Classes per local task')
    ax.set(xlabel='Atomic step',ylabel='Client',title='Curriculum widths and actual communication opportunities')
    fig.tight_layout();fig.savefig(out/'curriculum.png',dpi=140)
    if local_rows:
        fig=Figure(figsize=(10,4));FigureCanvasAgg(fig);ax=fig.add_subplot()
        for cid in sorted({x['client'] for x in local_rows}):
            points=[(x['atomic_step'],x['current_accuracy_after']) for x in local_rows
                    if x['client']==cid and x['current_accuracy_after'] is not None]
            if points:ax.plot(*zip(*points),label=f'C{cid}',alpha=.7)
        ax.set(xlabel='Atomic step',ylabel='Current-task validation accuracy',ylim=(0,1))
        if ax.lines:ax.legend(fontsize=7,ncol=3)
        fig.tight_layout();fig.savefig(out/'local_acquisition.png',dpi=140)
