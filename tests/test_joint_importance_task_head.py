import torch
from cafcl.data import get_data_bank
from cafcl.models import create_model
from cafcl.cl_estimators import estimate_diagonal_importance

def test_importance_uses_task_local_columns_for_high_global_labels():
    bank=get_data_bank('synthetic','data',download=False)
    head='task_008_009'
    model=create_model('test_linear',10,allow_test_model=True,task_class_sets={head:(8,9)})
    model.set_active_task_id(head)
    ids=[i for i,y in enumerate(bank.train.targets) if y in (8,9)]
    for kind in ('ewc','mas'):
        result=estimate_diagonal_importance(model,bank.train,ids,'cpu',kind=kind,max_samples=2,seed=1,active_classes=(8,9))
        assert all(torch.isfinite(v).all() for v in result.values())
        assert any(v.abs().sum()>0 for v in result.values())
