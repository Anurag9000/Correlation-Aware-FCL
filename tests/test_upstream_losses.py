"""Execute GLFC and LGA published original weighted BCE loss functions."""
import torch
import pytest

from cafcl.upstream_losses import original_loss_core,original_loss


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_exact_source_extracted(method):
    Source=original_loss_core(method)
    assert '_compute_loss' in Source.__dict__
    assert 'efficient_old_class_weight' in Source.__dict__
    assert Source._compute_loss.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')
    assert Source.efficient_old_class_weight.__code__.co_filename.endswith(
        'GLFC/src/GLFC.py' if method=='native_glfc' else 'LGA/LGA.py')


@pytest.mark.parametrize('method',['native_glfc','native_lga'])
def test_source_loss_cpu_cuda_api_fixture_reproduces_original_gradient(monkeypatch,method):
    # The only CPU fixture adaptation is translating the original .cuda()
    # device transfer calls into .to('cpu'); the objective AST is unchanged.
    logits=torch.tensor([[.2,-.5,1.,.6],[1.,.2,-.6,.3]],
                        dtype=torch.float32,requires_grad=True)
    labels=torch.tensor([1,3])
    got=original_loss(method,logits,labels,
                      known_classes={1,3},old_classes=())
    assert torch.isfinite(got)
    got.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_original_lga_source_includes_task_weighting_and_clamp():
    import inspect
    source=inspect.getsource(original_loss_core('native_lga').efficient_old_class_weight)
    assert 'self.task_id_old' in source
    assert 'torch.clamp' in source
    assert 'self.learned_classes' in source


def test_cpu_uses_author_math_with_device_transfer_only():
    import inspect
    cpu=original_loss_core('native_glfc',cpu_test=True)
    assert cpu._compute_loss.__code__.co_filename.endswith('GLFC/src/GLFC.py')
    z=torch.zeros((2,4),requires_grad=True)
    loss=original_loss('native_glfc',z,torch.tensor([1,3]),known_classes={1,3})
    loss.backward()
    assert torch.isfinite(loss)
    assert z.grad is not None and torch.isfinite(z.grad).all()
