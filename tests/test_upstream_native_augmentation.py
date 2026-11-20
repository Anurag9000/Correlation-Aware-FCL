"""Official original GLFC/LGA train transforms on CIFAR-100 bytes."""
import torch
from cafcl.upstream_native_augmentation import (
    original_glfc_lga_train_transform,original_glfc_lga_augment_batch,
    entropy_source_collate)


def test_source_train_transform_ast_has_authors_original_ops():
    for name in ('native_glfc','native_lga'):
        tr=original_glfc_lga_train_transform(name)
        types=[type(x).__name__ for x in tr.transforms]
        assert types==['RandomCrop','RandomHorizontalFlip','ColorJitter',
                       'ToTensor','Normalize']
        assert tr.transforms[0].size==(32,32)
        assert tr.transforms[0].padding==4
        assert tr.transforms[2].brightness==(1-0.24705882352941178,
                                            1+0.24705882352941178)


def test_source_transform_batch_is_deterministic_and_train_only():
    from cafcl.models import NORMALIZATION_MEAN,NORMALIZATION_STD
    mean=torch.tensor(NORMALIZATION_MEAN).view(1,3,1,1)
    std=torch.tensor(NORMALIZATION_STD).view(1,3,1,1)
    x=torch.linspace(0.,1.,3*32*32).reshape(1,3,32,32).repeat(64,1,1,1)
    normalized=(x-mean)/std
    for method in ('native_glfc','native_lga'):
        before=torch.get_rng_state().clone()
        a=original_glfc_lga_augment_batch(method,normalized,123)
        b=original_glfc_lga_augment_batch(method,normalized,123)
        assert torch.equal(a,b)
        assert torch.equal(torch.get_rng_state(),before)
        assert a.shape==(64,3,32,32)
        assert torch.isfinite(a).all()
        assert not torch.equal(a,normalized)


def test_original_entropy_collator_retains_ids_and_compact_labels():
    tr=entropy_source_collate('native_glfc',123)
    batch=[(i,torch.zeros(3,32,32),i%2) for i in range(64)]
    ids,x,y=tr(batch)
    assert len(ids)==len(x)==len(y)==64
    assert set(y.tolist())=={0,1}
