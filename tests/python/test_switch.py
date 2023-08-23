import drjit as dr
import pytest


def get_module(name):
    """Resolve a package+class name into the corresponding type"""
    if 'cuda' in name:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in name:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    elif 'packet' in name and not hasattr(dr, 'packet'):
        pytest.skip('Packet mode is unsupported')

    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    return value


@pytest.mark.parametrize("modname", ["drjit.cuda", "drjit.llvm"])
@pytest.mark.parametrize("recorded", [True, False])
def test01_switch(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    def f(a, b):
        return a * 4.0, b

    def g(a, b):
        return a * 8.0, -b

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    b = m.Float(1.0)

    result = dr.switch(idx, [f, g], a, b)

    assert dr.allclose(result, [[4, 8, 24, 32], [1, 1, -1, -1]])

    # Masked case
    active = m.Bool([True, False, True, False])

    def f(a, b, active):
        return a * 4.0, b

    def g(a, b, active):
        return a * 8.0, -b

    result = dr.switch(idx, [f, g], a, b, active)

    assert dr.allclose(result, [[4, 0, 24, 0], [1, 0, -1, 0]])
    assert dr.all(dr.eq(active, [True, False, True, False]))


@pytest.mark.parametrize("modname", ["drjit.cuda.ad", "drjit.llvm.ad"])
@pytest.mark.parametrize("recorded", [True, False])
def test02_switch_autodiff_forward(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    def f(a, b):
        return a * 4.0, b

    def g(a, b):
        return a * 8.0, -b

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    b = m.Float(1.0)

    dr.enable_grad(a, b)

    result = dr.switch(idx, [f, g], a, b)

    assert dr.allclose(result, [[4, 8, 24, 32], [1, 1, -1, -1]])

    dr.forward(a)
    grad = dr.grad(result)

    assert dr.allclose(grad, [[4, 4, 8, 8], [0, 0, 0, 0]])


@pytest.mark.parametrize("modname", ["drjit.cuda.ad", "drjit.llvm.ad"])
@pytest.mark.parametrize("recorded", [True, False])
def test03_switch_autodiff_forward_implicit(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    data = m.Float([1.0, 2.0, 3.0, 4.0])
    dr.enable_grad(data)

    def f(a, i):
        return a + dr.gather(m.Float, data, i)

    def g(a, i):
        return a + 4 * dr.gather(m.Float, data, i)

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    i = m.UInt([3, 2, 1, 0])

    result = dr.switch(idx, [f, g], a, i)

    assert dr.allclose(result, [5, 5, 11, 8])

    dr.forward(data)

    assert dr.allclose(dr.grad(result), [1, 1, 4, 4])

    # Test implicit dependency of a un-modified variable

    value = m.Float(4.0)
    dr.enable_grad(value)

    def f2(a):
        return value

    def g2(a):
        return 4.0 * a

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])

    result = dr.switch(idx, [f2, g2], a)

    assert dr.allclose(result, [4, 4, 12, 16])

    dr.forward(value)

    assert dr.allclose(dr.grad(result), [1, 1, 0, 0])


@pytest.mark.parametrize("modname", ["drjit.cuda.ad", "drjit.llvm.ad"])
@pytest.mark.parametrize("recorded", [True, False])
def test04_switch_autodiff_backward(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    def f(a, b):
        return a * 4.0, b

    def g(a, b):
        return a * 8.0, -b

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    b = m.Float([1.0, 1.0, 1.0, 1.0])

    dr.enable_grad(a, b)

    result = dr.switch(idx, [f, g], a, b)

    assert dr.allclose(result, [[4, 8, 24, 32], [1, 1, -1, -1]])

    dr.backward(dr.sum(result[0] + result[1]))

    assert dr.allclose(dr.grad(a), [4, 4, 8, 8])
    assert dr.allclose(dr.grad(b), [1, 1, -1, -1])


@pytest.mark.parametrize("modname", ["drjit.cuda.ad", "drjit.llvm.ad"])
@pytest.mark.parametrize("recorded", [True, False])
def test05_switch_autodiff_backward_implicit(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    data = m.Float([1.0, 2.0, 3.0, 4.0])
    dr.enable_grad(data)

    def f(a, i):
        return a + dr.gather(m.Float, data, i)

    def g(a, i):
        return a + 4 * dr.gather(m.Float, data, i)

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    i = m.UInt([3, 2, 1, 0])

    result = dr.switch(idx, [f, g], a, i)

    assert dr.allclose(result, [5, 5, 11, 8])

    dr.backward(result)

    assert dr.allclose(dr.grad(data), [4, 4, 1, 1])


@pytest.mark.parametrize("modname", ["drjit.cuda.ad", "drjit.llvm.ad"])
@pytest.mark.parametrize("recorded", [True])
def test06_switch_failure(modname, recorded):
    m = get_module(modname)

    dr.set_flag(dr.JitFlag.VCallRecord, recorded)

    def f(a, b):
        return a * 4.0, b

    def g(a, b):
        raise RuntimeError("foo")
        return a * 8.0, -b

    idx = m.UInt([0, 0, 1, 1])
    a = m.Float([1.0, 2.0, 3.0, 4.0])
    b = m.Float([1.0, 1.0, 1.0, 1.0])

    with pytest.raises(RuntimeError) as ei:
        result = dr.switch(idx, [f, g], a, b)
    assert "foo" in str(ei.value)

    pytest.skip("TODO when an exception is thrown in a CustomOp, AD edges aren't cleaned properly!")

    dr.enable_grad(a, b)

    class TestOp(dr.CustomOp):
        def eval(self, a, b):
            return a + 1.0, b + 2.0

        def forward(self):
            raise RuntimeError("bar")

    def h(a, b):
        return dr.custom(TestOp, a, b)

    result = dr.switch(idx, [f, h], a, b)

    with pytest.raises(RuntimeError) as ei:
        dr.forward(a)
    assert "bar" in str(ei.value)
