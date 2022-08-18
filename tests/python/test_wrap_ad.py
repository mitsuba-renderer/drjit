import drjit as dr

import sys
import importlib
import pytest

try:
    import torch
except:
    pass

@pytest.fixture(scope="module", params=['drjit.cuda.ad', 'drjit.llvm.ad'])
def m(request):
    if not 'torch' in sys.modules:
        pytest.skip('PyTorch is not installed on this system')
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_to_torch(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    dr.enable_grad(a)

    def func(a):
        return a * 4

    b = dr.wrap_ad(func, 'torch', a)
    dr.backward(dr.sum(b))

    assert dr.allclose(b, [4, 8, 12])
    assert dr.allclose(dr.grad(a), [4, 4, 4])


def test02_from_torch(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
    a.requires_grad = True

    def func(a):
        return a * 4

    b = dr.wrap_ad(func, 'drjit', a)

    b.sum().backward()

    assert dr.allclose(m.Float(b), [4, 8, 12])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])
