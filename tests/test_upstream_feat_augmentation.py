"""Execute pinned FEAT original training preprocessing and assert isolation."""
import torch
import pytest
from cafcl.upstream_feat_augmentation import (
    original_feat_cifar100_transform,original_feat_augment_batch)


def test_original_feat_source_pipeline_is_imported_verbatim():
    t=original_feat_cifar100_transform()
    names=[type(x).__name__ for x in t.transforms]
    assert names==['RandomCrop','RandomHorizontalFlip','ColorJitter',
                  'ToTensor','Normalize']
    assert t.transforms[0].size==(32,32)
    assert t.transforms[0].padding==4
    assert t.transforms[2].brightness==pytest.approx((1-63/255,1+63/255))


def test_source_random_augmentation_repeats_with_same_seed_and_restores_rng():
    source=torch.arange(0,64*3*32*32,dtype=torch.int64).reshape(64,3,32,32)
    raw=(source%256).to(torch.float32)/255.
    mean=torch.tensor((.5071,.4867,.4408)).view(1,3,1,1)
    std=torch.tensor((.2675,.2565,.2761)).view(1,3,1,1)
    normalized=(raw-mean)/std
    state=torch.get_rng_state()
    first=original_feat_augment_batch(normalized,seed=987)
    assert torch.equal(torch.get_rng_state(),state)
    second=original_feat_augment_batch(normalized,seed=987)
    assert first.shape==(64,3,32,32)
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    assert not torch.equal(first,normalized)
    assert torch.isfinite(first).all()


def test_source_augmentation_rejects_invalid_batch_shape():
    with pytest.raises(ValueError,match='32,32'):
        original_feat_augment_batch(torch.zeros((64,3,8,8)),seed=1)
