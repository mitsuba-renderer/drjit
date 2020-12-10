import enoki as ek
import pytest

def prepare(pkg):
    if 'cuda' in pkg.__name__:
        if not ek.has_cuda():
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in pkg.__name__:
        if not ek.has_llvm():
            pytest.skip('LLVM mode is unsupported')


@pytest.mark.parametrize("package", [ek.cuda, ek.cuda.ad,
                                     ek.llvm, ek.llvm.ad])
def test_zero_initialization(package):
    Float, Array3f = package.Float, package.Array3f
    prepare(package)

    class MyStruct:
        ENOKI_STRUCT = { 'a' : Array3f, 'b' : Float }

        def __init__(self):
            self.a = Array3f()
            self.b = Float()

        # Custom zero initialize callback
        def zero_(self, size):
            self.a += 1

    foo = ek.zero(MyStruct, 4)
    assert ek.width(foo) == 4
    assert foo.a == 1
    assert foo.b == 0

    foo = ek.zero(MyStruct, 1)
    ek.resize(foo, 8)
    assert ek.width(foo) == 8


@pytest.mark.parametrize("package", [ek.cuda.ad, ek.llvm.ad])
def test_ad_operations(package):
    Float, Array3f = package.Float, package.Array3f
    prepare(package)

    class MyStruct:
        ENOKI_STRUCT = { 'a' : Array3f, 'b' : Float }

        def __init__(self):
            self.a = Array3f()
            self.b = Float()

    foo = ek.zero(MyStruct, 4)
    assert not ek.grad_enabled(foo.a)
    assert not ek.grad_enabled(foo.b)
    assert not ek.grad_enabled(foo)

    ek.enable_grad(foo)
    assert ek.grad_enabled(foo.a)
    assert ek.grad_enabled(foo.b)
    assert ek.grad_enabled(foo)

    foo_detached = ek.detach(foo)
    assert not ek.grad_enabled(foo_detached.a)
    assert not ek.grad_enabled(foo_detached.b)
    assert not ek.grad_enabled(foo_detached)

    x = Float(4.0)
    ek.enable_grad(x)
    foo.a += x
    foo.b += x*x
    ek.forward(x)
    foo_grad = ek.grad(foo)
    assert foo_grad.a == 1
    assert foo_grad.b == 8

    ek.set_grad(foo, 5.0)
    foo_grad = ek.grad(foo)
    assert foo_grad.a == 5.0
    assert foo_grad.b == 5.0

    ek.accum_grad(foo, 5.0)
    foo_grad = ek.grad(foo)
    assert foo_grad.a == 10.0
    assert foo_grad.b == 10.0
