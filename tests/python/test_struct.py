import drjit as dr
import pytest
import importlib


def prepare(pkg):
    if 'cuda' in pkg:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in pkg:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    return importlib.import_module(pkg)


@pytest.mark.parametrize("package", ['drjit.cuda', 'drjit.cuda.ad',
                                     'drjit.llvm', 'drjit.llvm.ad'])
def test_zero_initialization(package):
    package = prepare(package)
    Float, Array3f = package.Float, package.Array3f

    class MyStruct:
        DRJIT_STRUCT = { 'a' : Array3f, 'b' : Float }

        def __init__(self):
            self.a = Array3f()
            self.b = Float()

        # Custom zero initialize callback
        def zero_(self, size):
            self.a += 1

    foo = dr.zeros(MyStruct, 4)
    assert dr.width(foo) == 4
    assert foo.a == 1
    assert foo.b == 0

    foo = dr.zeros(MyStruct, 1)
    dr.resize(foo, 8)
    assert dr.width(foo) == 8


@pytest.mark.parametrize("package", ['drjit.cuda.ad', 'drjit.llvm.ad'])
def test_ad_operations(package):
    package = prepare(package)
    Float, Array3f = package.Float, package.Array3f

    class MyStruct:
        DRJIT_STRUCT = { 'a' : Array3f, 'b' : Float }

        def __init__(self):
            self.a = Array3f()
            self.b = Float()

    foo = dr.zeros(MyStruct, 4)
    assert not dr.grad_enabled(foo.a)
    assert not dr.grad_enabled(foo.b)
    assert not dr.grad_enabled(foo)

    dr.enable_grad(foo)
    assert dr.grad_enabled(foo.a)
    assert dr.grad_enabled(foo.b)
    assert dr.grad_enabled(foo)

    foo_detached = dr.detach(foo)
    assert not dr.grad_enabled(foo_detached.a)
    assert not dr.grad_enabled(foo_detached.b)
    assert not dr.grad_enabled(foo_detached)

    x = Float(4.0)
    dr.enable_grad(x)
    foo.a += x
    foo.b += x*x
    dr.forward(x)
    foo_grad = dr.grad(foo)
    assert foo_grad.a == 1
    assert foo_grad.b == 8

    dr.set_grad(foo, 5.0)
    foo_grad = dr.grad(foo)
    assert foo_grad.a == 5.0
    assert foo_grad.b == 5.0

    dr.accum_grad(foo, 5.0)
    foo_grad = dr.grad(foo)
    assert foo_grad.a == 10.0
    assert foo_grad.b == 10.0
