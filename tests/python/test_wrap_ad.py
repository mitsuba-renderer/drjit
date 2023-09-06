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


def test01_to_torch_single_arg(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    dr.enable_grad(a)

    @dr.wrap_ad(source='drjit', target='torch')
    def func(a):
        return a * 4

    b = func(a)
    dr.backward(dr.sum(b))

    assert dr.allclose(b, [4, 8, 12])
    assert dr.allclose(dr.grad(a), [4, 4, 4])

    a = m.Float([1.0, 2.0, 3.0])
    dr.enable_grad(a)
    with pytest.raises(TypeError, match='should be Dr.Jit tensor'):
        b = func(a)


def test02_to_torch_two_args(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    b = m.TensorXf(m.Float([4.0, 5.0, 6.0]), shape=[3])
    dr.enable_grad(a, b)

    @dr.wrap_ad(source='drjit', target='torch')
    def func(a, b):
        return a * 4 + b * 3

    c = func(a, b)
    dr.backward(dr.sum(c))

    assert dr.allclose(c, [16, 23, 30])
    assert dr.allclose(dr.grad(a), [4, 4, 4])
    assert dr.allclose(dr.grad(b), [3, 3, 3])


def test03_to_torch_non_diff_arg_and_kwargs(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    dr.enable_grad(a)

    @dr.wrap_ad(source='drjit', target='torch')
    def func2(a, c: int = 4, s: str = '', d: int = 1):
        print(s)
        return a * c * d

    b = func2(a, s='test', c=4)
    dr.backward(dr.sum(b))

    assert dr.allclose(b, [4, 8, 12])
    assert dr.allclose(dr.grad(a), [4, 4, 4])


def test04_to_torch_two_args_two_outputs(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    b = m.TensorXf(m.Float([4.0, 5.0, 6.0]), shape=[3])
    dr.enable_grad(a, b)

    @dr.wrap_ad(source='drjit', target='torch')
    def func(a, b):
        return a * 4, b * 3

    c, d = func(a, b)
    dr.backward(dr.sum(c + d))

    assert dr.allclose(c, [4, 8, 12])
    assert dr.allclose(d, [12, 15, 18])
    assert dr.allclose(dr.grad(a), [4, 4, 4])
    assert dr.allclose(dr.grad(b), [3, 3, 3])


def test05_from_torch_single_arg(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
    a.requires_grad = True

    @dr.wrap_ad(source='torch', target='drjit')
    def func(a):
        return a * 4

    b = func(a)

    b.sum().backward()

    assert dr.allclose(m.Float(b), [4, 8, 12])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])


def test06_from_torch_two_args(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
        b = b.cuda()
    a.requires_grad = True
    b.requires_grad = True

    @dr.wrap_ad(source='torch', target='drjit')
    def func(a, b):
        return a * 4 + b * 3

    c = func(a, b)

    c.sum().backward()

    assert dr.allclose(m.Float(c), [16, 23, 30])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])
    assert dr.allclose(m.Float(b.grad), [3, 3, 3])


def test07_from_torch_non_diff_args_and_kwargs(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
    a.requires_grad = True

    @dr.wrap_ad(source='torch', target='drjit')
    def func2(a, c: int = 4, s: str = '', d: int = 1):
        print(s)
        return a * c * d

    b = func2(a, s='test', c=4)

    b.sum().backward()

    assert dr.allclose(m.Float(b), [4, 8, 12])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])


def test08_from_torch_two_args_two_outputs(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4.0, 5.0, 6.0])
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
        b = b.cuda()
    a.requires_grad = True
    b.requires_grad = True

    @dr.wrap_ad(source='torch', target='drjit')
    def func(a, b):
        return a * 4, b * 3

    c, d = func(a, b)

    e = (c + d).sum()
    e.backward()

    assert dr.allclose(m.Float(c), [4, 8, 12])
    assert dr.allclose(m.Float(d), [12, 15, 18])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])
    assert dr.allclose(m.Float(b.grad), [3, 3, 3])


def test09_to_torch_list_of_tensors_as_args(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    b = m.TensorXf(m.Float([4.0, 5.0, 6.0]), shape=[3])
    l = [a,b]
    dr.enable_grad(*l)

    @dr.wrap_ad(source='drjit', target='torch')
    def func(l):
        return l[0] * 4, l[1] * 3

    c, d = func(l)
    dr.backward(dr.sum(c + d))

    assert dr.allclose(c, [4, 8, 12])
    assert dr.allclose(d, [12, 15, 18])
    assert dr.allclose(dr.grad(a), [4, 4, 4])
    assert dr.allclose(dr.grad(b), [3, 3, 3])


def test10_to_torch_list_of_tensors_as_args_and_return_nested_stucture(m):
    a = m.TensorXf(m.Float([1.0, 2.0, 3.0]), shape=[3])
    b = m.TensorXf(m.Float([4.0, 5.0, 6.0]), shape=[3])
    l = [a,b]
    dr.enable_grad(*l)

    @dr.wrap_ad(source='drjit', target='torch')
    def func(l):
        return {
            "first": l[0] * 4,
            "second": {
                "real_second": [l[1] * 3]
            }
        }

    dictionary = func(l)
    c, d = dictionary["first"], dictionary["second"]["real_second"][0]
    dr.backward(dr.sum(c + d))

    assert dr.allclose(c, [4, 8, 12])
    assert dr.allclose(d, [12, 15, 18])
    assert dr.allclose(dr.grad(a), [4, 4, 4])
    assert dr.allclose(dr.grad(b), [3, 3, 3])


def test11_from_torch_integer_tensors(m):
    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor([4, 5, 6], dtype=torch.int32)
    if dr.is_cuda_v(m.Float):
        a = a.cuda()
        b = b.cuda()
    a.requires_grad = True

    @dr.wrap_ad(source='torch', target='drjit')
    def func(a, b):
        return a * 4, b * 3

    c, d = func(a, b)

    e = (c + d).sum()
    e.backward()

    assert dr.allclose(m.Float(c), [4, 8, 12])
    assert dr.allclose(d, [12, 15, 18])
    assert dr.allclose(m.Float(a.grad), [4, 4, 4])
    assert b.grad is None
