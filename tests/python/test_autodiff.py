import drjit as dr
import pytest
import importlib

@pytest.fixture(scope="module", params=['drjit.cuda.ad', 'drjit.llvm.ad'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def struct_class(m):
    class MyStruct:
        def __init__(self) -> None:
            self.x = m.Float()
            self.y = m.Float()
        DRJIT_STRUCT = { 'x': m.Float, 'y': m.Float }
    return MyStruct

def test01_enable_grad(m):
    # Test on single float variable
    a = m.Float(4.0)
    assert not dr.grad_enabled(a)
    dr.enable_grad(a)
    assert dr.grad_enabled(a)
    dr.disable_grad(a)
    assert not dr.grad_enabled(a)

    # Test on non-float variable
    b = m.UInt(1)
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert not dr.grad_enabled(b)

    # Test with sequence and mapping
    c = m.Float(2.0)

    l = [a, b, c]
    assert not dr.grad_enabled(l)
    dr.enable_grad(c)
    assert dr.grad_enabled(l)
    dr.disable_grad(l)
    assert not dr.grad_enabled(l)

    d = { 'a': a, 'c': b, 'c': c }
    assert not dr.grad_enabled(d)
    dr.enable_grad(c)
    assert dr.grad_enabled(d)
    dr.disable_grad(d)
    assert not dr.grad_enabled(d)

    # Test with multiple arguments
    dr.enable_grad(c)
    assert dr.grad_enabled(a, b, c)

    # Test with static array
    a = m.Array3f(1.0, 2.0, 3.0)
    assert not dr.grad_enabled(a)
    dr.enable_grad(a)
    assert dr.grad_enabled(a)
    assert dr.grad_enabled(a[0])

    s = struct_class(m)()
    assert not dr.grad_enabled(s)
    dr.enable_grad(s.x)
    assert dr.grad_enabled(s)
    dr.enable_grad(s)
    assert dr.grad_enabled(s.y)


def test02_detach(m):
    a = m.Float([1, 2, 3])
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert not type(a) == type(b)
    assert type(a) == type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    a = m.Array3f()
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert not type(a) == type(b)
    assert type(a) == type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    a = struct_class(m)()
    dr.enable_grad(a)
    c = dr.detach(a)
    assert type(a) == type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(c)


def test03_set_accum_grad(m):
    a = m.Float([1, 2, 3])

    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)

    dr.set_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)

    dr.accum_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 4.0)

    dr.set_grad(a, [3.0, 2.0, 1.0])
    assert dr.allclose(dr.grad(a), m.Float([3.0, 2.0, 1.0]))

def test04_forward_from(m):
    a = m.Float(1.0)
    dr.enable_grad(a)
    b = a * a * 2
    dr.forward(a)
    assert dr.allclose(dr.grad(b), 4.0)

def test05_backward_from(m):
    a = m.Float(1.0)
    dr.enable_grad(a)
    b = a * a * 2
    dr.backward(b)
    assert dr.allclose(dr.grad(a), 4.0)

    a = m.Array3f(1, 2, 3)
    dr.enable_grad(a)
    b = a * 2
    dr.backward(b)
    assert dr.allclose(dr.grad(a), 2.0)

def test47_replace_grad(m):
    x = m.Array3f(1, 2, 3)
    y = m.Array3f(3, 2, 1)
    dr.enable_grad(x, y)
    x2 = x*x
    y2 = y*y

    z = dr.replace_grad(x2, y2)

    assert z.x.index_ad == y2.x.index_ad
    assert z.y.index_ad == y2.y.index_ad

    z2 = z*z
    assert dr.allclose(z2, [1, 16, 81])

    dr.backward(z2)

    assert dr.allclose(dr.grad(x), 0)
    assert dr.allclose(dr.grad(y), [12, 32, 36])


if __name__ == "__main__":
    from drjit.llvm.ad import Float

    a = Float(4.0)

    print(a)

    print(dr.grad_enabled(a))

    dr.enable_grad(a)

    print(dr.grad_enabled(a))


    dr.disable_grad(a)

    print(dr.grad_enabled(a))

    # assert True