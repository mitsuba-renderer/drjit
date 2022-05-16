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


def instantiate_struct(m):
    class MyStruct:
        def __init__(self) -> None:
            self.x = m.Float()
            self.y = m.Float()
        DRJIT_STRUCT = { 'x': m.Float, 'y': m.Float }
    return MyStruct()

def test01_enable_grad(m):
    # Test on single float variable
    a = m.Float([4.0])
    assert not dr.grad_enabled(a)
    dr.enable_grad(a)
    assert dr.grad_enabled(a)
    dr.disable_grad(a)
    assert not dr.grad_enabled(a)

    # Test on non-float variable
    b = m.UInt([1])
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert not dr.grad_enabled(b)

    # Test with sequence and mapping
    c = m.Float([2.0])

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
    a = m.Array3f([1.0], [2.0], [3.0])
    assert not dr.grad_enabled(a)
    dr.enable_grad(a)
    assert dr.grad_enabled(a)
    assert dr.grad_enabled(a[0])

    s = instantiate_struct(m)
    assert not dr.grad_enabled(s)
    dr.enable_grad(s.x)
    assert dr.grad_enabled(s)
    dr.enable_grad(s)
    assert dr.grad_enabled(s.y)


def test02_set_accum_grad(m):
    a = m.Float([4.0])
    assert dr.grad(a) == [0.0]

    dr.set_grad(a, 2.0)
    assert dr.grad(a) == [2.0]

    dr.accum_grad(a, 2.0)
    assert dr.grad(a) == [4.0]



if __name__ == "__main__":
    from drjit.llvm.ad import Float

    a = Float([4.0])

    print(a)

    print(dr.grad_enabled(a))

    dr.enable_grad(a)

    print(dr.grad_enabled(a))


    dr.disable_grad(a)

    print(dr.grad_enabled(a))

    # assert True