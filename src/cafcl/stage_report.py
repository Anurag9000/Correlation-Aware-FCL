"""Immutable, post-sync, server-model class-retention diagnostics.

Each communication produces a complete seen-class table. The test split is
read only for reporting and NEVER used by client selection, training, growth,
hyperparameters, or control decisions. Validation drives any online feedback.
Per-stage outputs are independent of the terminal-only official test score.
"""
from __future__ import annotations

import csv
import json
from hashlib import sha256
from collections import Counter
from pathlib import Path

def _entries(arrivals, current_round, per_class, num_clients):
    introduced={}
    for a in arrivals:
        rd=int(a.round_id)
        if rd>current_round:continue
        cid=int(a.client_id)
        for cls in a.labels:
            introduced.setdefault((cid,rd),set()).add(int(cls))
    entries=[]
    for cid in range(num_clients):
        for rd in range(current_round+1):
            classes=sorted(int(c) for c in introduced.get((cid,rd),()) if int(c) in per_class)
            if not classes:continue
            entries.append(dict(client_id=cid,first_round=rd+1,
                classes=classes,mean_accuracy=sum(float(per_class[c]) for c in classes)/len(classes),
                per_class={c:float(per_class[c]) for c in classes}))
    return entries


def render_sync_report(run_dir, *, round_number, model_version, num_clients,
                       arrivals, test_metrics, validation_metrics=None,
                       selected_clients=(), terminal_flush=False, backbone='',
                       optimizer=None,classifier_bias=None,
                       integrated_classes=None,integrated_metrics=None,
                       integrated_task_heads=()):
    """Save a new PNG, Markdown, CSV and JSON after EVERY server synchronization."""
    root=Path(run_dir)/'per_sync_accuracy'
    root.mkdir(parents=True,exist_ok=True)
    stem=f'sync_{int(round_number):03d}'
    report_split='test' if test_metrics is not None else 'validation'
    report_metrics=test_metrics if test_metrics is not None else validation_metrics
    if report_metrics is None:raise ValueError('A test or validation report is required')
    per_class={int(k):float(v) for k,v in report_metrics['per_class'].items()}
    entries=_entries(arrivals,int(round_number)-1,per_class,int(num_clients))
    integrated=set(map(int,integrated_classes if integrated_classes is not None else per_class))
    for entry in entries:
        entry['pending_not_accepted_classes']=[int(c) for c in entry['classes'] if int(c) not in integrated]
    cells={(r['client_id'],r['first_round']):r for r in entries}
    confusion=report_metrics['confusion_matrix']['counts']
    n_classes=len(confusion)
    predicted=[sum(int(row[c]) for row in confusion) for c in range(n_classes)]
    seen=sorted(per_class)
    # Exclude unarrived output slots: they are deliberately masked at
    # inference and MUST NOT be described as collapsed learned classes.
    zero_labels=[int(c) for c in seen if predicted[c]==0]
    accuracy=float(report_metrics['accuracy'] or 0.0)
    bias_summary={}
    if classifier_bias is not None:
        biases=classifier_bias.detach().cpu().flatten().tolist()
        if len(biases)>=n_classes:
            new_ids=sorted({c for e in entries if e['first_round']==int(round_number)
                            for c in e['classes']})
            old_ids=sorted(set(seen)-set(new_ids))
            bias_summary={'bias_by_class':{int(c):float(biases[c]) for c in seen},
                          'new_classes':new_ids,'old_classes':old_ids,
                          'mean_new_bias':(sum(biases[c] for c in new_ids)/len(new_ids)
                                           if new_ids else None),
                          'mean_old_bias':(sum(biases[c] for c in old_ids)/len(old_ids)
                                           if old_ids else None)}
            if new_ids and old_ids:
                bias_summary['new_minus_old_mean_bias']=(
                    bias_summary['mean_new_bias']-bias_summary['mean_old_bias'])
    metadata=dict(round=int(round_number),version=int(model_version),
        terminal_communication_only_flush=bool(terminal_flush),
        report_split=report_split,score_scope='global_server_scores_indexed_by_client_task',
        evaluation_protocol=report_metrics.get('task_id_assignment','class_il'),
        test_accuracy=accuracy if report_split=='test' else None,validation_accuracy=(
            None if validation_metrics is None else validation_metrics['accuracy']),
        observed_classes=seen,observed_class_count=len(seen),
        server_integrated_classes=sorted(integrated),
        server_integrated_class_count=len(integrated),
        server_integrated_task_heads=(
            [{'task_id':str(task_id),'classes':list(map(int,classes))}
             for task_id,classes in sorted(integrated_task_heads.items())]
            if isinstance(integrated_task_heads,dict) else
            [{'task_id':None,'classes':list(map(int,task))}
             for task in integrated_task_heads]),
        pending_unintegrated_classes=sorted(set(seen)-integrated),
        integrated_only_test_accuracy=(integrated_metrics['accuracy'] if integrated_metrics else None),
        selected_clients=list(map(int,selected_clients)),zero_prediction_classes=zero_labels,
        predicted_class_counts={int(c):int(v) for c,v in enumerate(predicted) if v},
        backbone=backbone,optimizer=optimizer or {},classifier_bias_diagnostics=bias_summary,
        per_class_true_logit_margin=report_metrics.get('per_class_true_logit_margin',{}),
        test_only_no_feedback=True,entries=entries)
    (root/(stem+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
    with (root/(stem+'.csv')).open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=[
            'server_sync','client_id','introduced_sync','class_id',report_split+'_correct',
            report_split+'_total',report_split+'_accuracy','seen_class_global_accuracy'])
        writer.writeheader()
        for entry in entries:
            for cls in entry['classes']:
                totals=sum(int(x) for x in confusion[cls])
                writer.writerow(dict(server_sync=int(round_number),
                    client_id=entry['client_id'],introduced_sync=entry['first_round'],
                    class_id=cls,seen_class_global_accuracy=f'{accuracy:.6f}',
                    **{report_split+'_correct':int(confusion[cls][cls]),report_split+'_total':totals,
                       report_split+'_accuracy':f'{per_class[cls]:.6f}'}))
    latest=max([e['first_round'] for e in entries],default=0)
    md=[f'# Server sync {round_number} — global model v{model_version}',
        '',f'Backbone: {backbone}; held-out {report_split.upper()} server diagnostic.',
        'Rows index historical client task placements; values are GLOBAL SERVER scores, not private-client accuracy.',
        f"Evaluation protocol: {metadata['evaluation_protocol']}. Class-membership oracle scores are diagnostic, not principal Task-IL scores.",
        f'Seen-class global accuracy: {accuracy:.2%} on {len(seen)} observed labels.',
        f'Validation accuracy: {metadata["validation_accuracy"]}; seen labels never predicted: {len(zero_labels)}.',
        f'Server-accepted classes: {len(integrated)}; pending private classes: {len(set(seen)-integrated)}.',
        f'Accepted-only TEST accuracy: {metadata["integrated_only_test_accuracy"]}.',
        f'Terminal flush: {bool(terminal_flush)}. Selected clients: {list(selected_clients)}.',
        '**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**',
        '', '| Client | '+' | '.join(f'U{i}' for i in range(1,latest+1))+' |',
        '|:--|'+'|'.join(':--:' for _ in range(latest))+'|']
    for cid in range(num_clients):
        row=[]
        for rd in range(1,latest+1):
            e=cells.get((cid,rd))
            if not e:
                row.append('—')
                continue
            suffix=' [PENDING upload]' if e['pending_not_accepted_classes'] else ''
            row.append(f'**{e["mean_accuracy"]:.1%}** ('+', '.join(
                f'{c}: {e["per_class"][c]:.0%}' for c in e['classes'])+')'+suffix)
        md.append(f'| Client {cid} | '+' | '.join(row)+' |')
    (root/(stem+'.md')).write_text('\n'.join(md)+'\n')

    # PNG is a presentation of the exact same immutable JSON/CSV data.
    # Matplotlib uses a noninteractive backend and never touches model state.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    # Keep full JSON/CSV/Markdown history; bound raster rendering to the latest 30 columns.
    plot_rounds=range(max(1,latest-29),latest+1)
    cols=len(plot_rounds)+1
    height=max(2.8,1.5+0.82*num_clients)
    width=max(9,2.3*cols)
    fig=Figure(figsize=(width,height),dpi=130,facecolor='#f6f8fc')
    FigureCanvasAgg(fig)
    ax=fig.add_axes((.02,.02,.96,.80));ax.axis('off')
    data=[]
    for cid in range(num_clients):
        row=[f'Client {cid}']
        for rd in plot_rounds:
            e=cells.get((cid,rd))
            if e:
                row.append(f'{e["mean_accuracy"]:.1%}\n'+
                           '  '.join(f'{c}: {e["per_class"][c]:.0%}' for c in e['classes'])+
                           ('\nPENDING upload' if e['pending_not_accepted_classes'] else ''))
            else:row.append('—')
        data.append(row)
    headers=['CLIENT (global scores)']+[f'U{n}' for n in plot_rounds]
    table=ax.table(cellText=data,colLabels=headers,cellLoc='center',colLoc='center',
                   loc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.5 if cols<9 else 7)
    for (ri,ci),cell in table.get_celld().items():
        cell.set_edgecolor('#b7c7dc');cell.set_linewidth(.55)
        cell.set_facecolor('#e3ebf7' if ri==0 else
                           ('#f5f7fb' if ri%2==0 else '#ffffff'))
        if ri==0 or ci==0:cell.get_text().set_fontweight('bold')
    fig.text(.028,.95,f'Server sync {round_number} · {len(seen)} seen classes · {backbone}',
             fontsize=15,fontweight='bold',color='#152846')
    fig.text(.028,.885,
             f'Global seen-class {report_split.upper()} accuracy {accuracy:.2%}  |  '
             f'Never-predicted output classes {len(zero_labels)}  |  '
             f'Online clients {list(selected_clients)}',
             fontsize=9,color='#374862')
    fig.savefig(root/(stem+'.png'),dpi=130,facecolor=fig.get_facecolor())
    artifact_sha256={ext:sha256((root/(stem+ext)).read_bytes()).hexdigest() for ext in ('.json','.csv','.md','.png')}
    return {'stem':stem,'relative_dir':'per_sync_accuracy','artifact_sha256':artifact_sha256,
            'test_accuracy':accuracy if report_split=='test' else None,'seen_class_count':len(seen),
            'report_split':report_split,
            'never_predicted_classes':len(zero_labels)}


def assert_sync_reports_complete(run_dir, round_metrics):
    """Reject incomplete/corrupt per-sync evidence before accepting a run."""
    root=Path(run_dir)/'per_sync_accuracy'
    if not round_metrics:
        raise RuntimeError('No committed server-sync metrics to validate')
    for stage,metrics in enumerate(round_metrics,1):
        stem=f'sync_{stage:03d}'
        for ext in ('.json','.csv','.md','.png'):
            artifact=root/(stem+ext)
            if not artifact.is_file() or artifact.stat().st_size==0:
                raise RuntimeError(f'Missing or empty server-sync artifact: {artifact}')
        try:
            saved=json.loads((root/(stem+'.json')).read_text())
        except (ValueError,OSError) as exc:
            raise RuntimeError(f'Invalid per-sync JSON: {stem}') from exc
        if int(saved.get('round',-1))!=stage:
            raise RuntimeError(f'Per-sync report round mismatch: {stem}')
        if int(saved.get('version',-1))!=int(metrics['model_version']):
            raise RuntimeError(f'Per-sync model version mismatch: {stem}')
        for report_key,metric_key in (
                ('observed_classes','observed_classes'),
                ('server_integrated_classes','globally_integrated_classes')):
            if sorted(map(int,saved.get(report_key,())))!=sorted(map(int,metrics[metric_key])):
                raise RuntimeError(f'Per-sync class coverage mismatch: {stem}/{report_key}')
        if saved.get('test_only_no_feedback') is not True:
            raise RuntimeError(f'Per-sync TEST feedback invariant violated: {stem}')
        if saved.get('test_accuracy')!=metrics.get('per_sync_test_accuracy'):
            raise RuntimeError(f'Per-sync accuracy mismatch: {stem}')
        if ('per_sync_server_integrated_only_accuracy' in metrics and
                saved.get('integrated_only_test_accuracy')!=metrics.get(
                    'per_sync_server_integrated_only_accuracy')):
            raise RuntimeError(f'Integrated-only per-sync accuracy mismatch: {stem}')
        receipt=metrics.get('sync_accuracy_report',{}).get('artifact_sha256')
        if receipt is not None:
            extensions=('.json','.csv','.md','.png')
            if set(receipt)!=set(extensions):raise RuntimeError(f'Incomplete artifact checksum receipt: {stem}')
            for ext in extensions:
                if sha256((root/(stem+ext)).read_bytes()).hexdigest()!=receipt[ext]:
                    raise RuntimeError(f'Per-sync artifact checksum mismatch: {stem}{ext}')
            split=saved['report_split']
            if split not in ('validation','test'):raise RuntimeError('Invalid report split')
            accuracy=saved[split+'_accuracy']
            with (root/(stem+'.csv')).open(newline='') as handle:
                rows=list(csv.DictReader(handle))
            expected={}
            for entry in saved['entries']:
                for cls in entry['classes']:
                    key=(entry['client_id'],entry['first_round'],cls)
                    if key in expected:raise RuntimeError('Duplicate report key')
                    expected[key]=entry['per_class'][str(cls)]
            actual={}
            for row in rows:
                key=(int(row['client_id']),int(row['introduced_sync']),int(row['class_id']))
                total=int(row[split+'_total']);correct=int(row[split+'_correct'])
                if key in actual or int(row['server_sync'])!=stage or not 0<=correct<=total:
                    raise RuntimeError('Invalid per-sync CSV identity/counts')
                if total and abs(float(row[split+'_accuracy'])-correct/total)>5.1e-7:
                    raise RuntimeError('Per-sync CSV accuracy/count mismatch')
                if accuracy is not None and abs(float(row['seen_class_global_accuracy'])-accuracy)>5.1e-7:
                    raise RuntimeError('Per-sync CSV global accuracy mismatch')
                actual[key]=float(row[split+'_accuracy'])
            if set(actual)!=set(expected) or any(abs(actual[k]-expected[k])>5.1e-7 for k in expected):
                raise RuntimeError('Per-sync CSV differs from JSON entries')
            md=(root/(stem+'.md')).read_text()
            if f'# Server sync {stage} — global model v{saved["version"]}' not in md:
                raise RuntimeError('Per-sync Markdown identity mismatch')
            from PIL import Image
            with Image.open(root/(stem+'.png')) as im:
                if im.format!='PNG' or min(im.size)<=0:raise RuntimeError('Invalid sync PNG')
                im.verify()
        expected_heads=metrics.get('per_sync_server_integrated_task_heads')
        if expected_heads is not None:
            actual_heads=saved.get('server_integrated_task_heads',())
            if actual_heads!=expected_heads:
                raise RuntimeError(f'Integrated task-head mismatch: {stem}')
    return len(round_metrics)

