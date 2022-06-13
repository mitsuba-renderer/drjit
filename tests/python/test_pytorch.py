import drjit as dr
import torch

import importlib
import pytest

@pytest.fixture(scope="module", params=['drjit.cuda.ad', 'drjit.llvm.ad'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_to_torch(m):
    a = m.Float([0.0, 1.0, 2.0])
    dr.enable_grad(a)
    b = 4.0 * a
    c = dr.to_torch(b)
    d = 4.0 * c
    e = d.sum()
    e.backward()
    assert dr.allclose(dr.grad(a), [16, 16, 16])

    a = m.Array3f([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    dr.enable_grad(a)
    b = m.Array3f([1, 2], [2, 3], [4, 5]) * a
    c = dr.to_torch(b)
    d = 4.0 * c
    e = d.sum()
    e.backward()
    assert dr.allclose(dr.grad(a), [[4, 8], [8, 12], [16, 20]])

    a = m.TensorXf([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], shape=(2, 3))
    dr.enable_grad(a)
    b = 4.0 * a
    c = dr.to_torch(b)
    d = 4.0 * c
    e = d.sum()
    e.backward()
    assert a.shape == dr.grad(a).shape
    assert dr.allclose(dr.grad(a), 16)


def test02_from_torch(m):
    with pytest.raises(TypeError) as ei:
        _ = dr.from_torch(float, torch.tensor([1, 2, 3]))
    assert "expected a differentiable Dr.Jit array type!" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        _ = dr.from_torch(dr.llvm.Float, torch.tensor([1, 2, 3]))
    assert "expected a differentiable Dr.Jit array type!" in str(ei.value)

    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = 4.0 * a
    c = dr.from_torch(m.Float, b)
    d = 4.0 * c
    e = dr.sum(d)
    dr.backward(e)
    assert dr.allclose(a.grad, [16, 16, 16])

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = torch.tensor([[1, 2, 3], [3, 4, 5]]) * a
    c = dr.from_torch(m.Array3f, b)
    d = 4.0 * c
    e = dr.sum(d)
    dr.backward(e)
    assert dr.allclose(a.grad, [[4, 8, 12], [12, 16, 20]])

    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    b = torch.tensor([[1, 2, 3], [3, 4, 5]]) * a
    c = dr.from_torch(m.TensorXf, b)
    d = 4.0 * c
    e = dr.sum(d)
    dr.backward(e)
    assert dr.allclose(a.grad, [[4, 8, 12], [12, 16, 20]])


def test03_torch_drjit_roundtrip(m):
    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = 4.0 * a
    c = dr.from_torch(m.Float, b)
    d = 4.0 * c
    e = dr.to_torch(d)
    f = e.sum()
    f.backward()
    assert dr.allclose(a.grad, [16, 16, 16])


def test04_drjit_torch_roundtrip(m):
    a = m.Float([0.0, 1.0, 2.0])
    dr.enable_grad(a)
    b = 4.0 * a
    c = dr.to_torch(b)
    d = 4.0 * c
    e = dr.from_torch(m.Float, d)
    f = dr.sum(e)
    dr.backward(f)
    assert dr.allclose(dr.grad(a), [16, 16, 16])
