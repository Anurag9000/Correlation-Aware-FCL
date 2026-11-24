"""Evaluate a private model snapshot without mutating client state."""
from ..evaluation import evaluate
from .provenance import visible_source_heads
from .task_evaluation import evaluate_task_conditioned


def private_metrics(model,client,bank,cfg,device,split='validation'):
    model.load_state_dict(client.shared_weights)
    groups=None
    labels=set(client.observed_classes)
    if cfg.learning_setting=='task_il':
        heads={h for values in visible_source_heads(client).values() for h in values}
        groups={h:model.classifier.task_classes[h] for h in sorted(heads)}
        labels={c for values in groups.values() for c in values}
    if not labels:return None
    result,_=evaluate(model,bank,labels,cfg.batch_size,device,
                      task_class_groups=groups,split=split,loader_seed=0)
    if groups is not None:
        result['evaluation_task_heads']=groups
        result['task_conditioned']=evaluate_task_conditioned(
            model,bank,groups,cfg.batch_size,device,split)
    return result
