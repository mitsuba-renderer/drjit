import drjit as dr
import pytest

try:
    import torch

    pytorch_missing = False
except ImportError:
    pytorch_missing = True

needs_pytorch = pytest.mark.skipif(pytorch_missing, reason="Test requires PyTorch")


@needs_pytorch
@pytest.test_arrays('is_diff,float,shape=(*)')
def test01_simple(t):
    @dr.wrap_ad(source="drjit", target="torch")
    def foo(x):
        return x * 2

    x = dr.arange(t, 3)
    dr.enable_grad(x)
    y = foo(x)

    y.grad = [10, 20, 30]
    dr.backward_to(x)
    assert dr.all(x.grad == [20, 40, 60])
