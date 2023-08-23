import drjit as dr
import pytest
import sys

def make_mystruct(t):
    class MyStruct:
        def __init__(self) -> None:
            self.x = t(1)
            self.y = t(2)
        DRJIT_STRUCT = { 'x': t, 'y': t }
    return MyStruct

@pytest.test_arrays("is_diff,-mask,-shape=()")
def test01_enable_grad(t):
    a = t(1)

    assert not dr.grad_enabled(a)
    dr.enable_grad(a)

    if not dr.is_float_v(t):
        assert not dr.grad_enabled(a)
        return

    assert dr.grad_enabled(a)

    b = (t(1), t(2), t(3))
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert dr.grad_enabled(b) and \
           dr.grad_enabled(b[0]) and \
           dr.grad_enabled(b[1]) and \
           dr.grad_enabled(b[2])

    b = {'a': t(1), 'b': t(2), 'c': t(3)}
    assert not dr.grad_enabled(b)
    dr.enable_grad(b)
    assert dr.grad_enabled(b) and \
           dr.grad_enabled(b['a']) and \
           dr.grad_enabled(b['b']) and \
           dr.grad_enabled(b['c'])

    a, b, c = t(1), t(2), t(3)
    assert not dr.grad_enabled(a, b, c)
    dr.enable_grad(a, b, c)
    assert dr.grad_enabled(a, b, c) and \
           dr.grad_enabled(a) and \
           dr.grad_enabled(b) and \
           dr.grad_enabled(c)

    if a.ndim > 1:
        assert dr.grad_enabled(a[0])


    MyStruct = make_mystruct(t)
    a = MyStruct()
    assert not dr.grad_enabled(a) and \
           not dr.grad_enabled(a.x) and \
           not dr.grad_enabled(a.y)

    dr.enable_grad(a)

    assert dr.grad_enabled(a) and \
           dr.grad_enabled(a.x) and \
           dr.grad_enabled(a.y)

@pytest.test_arrays("is_diff,float,-shape=()")
def test02_detach(t):
    a = t(1)
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert type(a) is not type(b)
    assert dr.detached_t(type(a)) is type(b)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    MyStruct = make_mystruct(t)
    a = MyStruct()
    dr.enable_grad(a)
    c = dr.detach(a)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(c)

@pytest.test_arrays("is_diff,float,shape=(*)")
def test03_set_grad(t):
    a = t([1, 2, 3])
    dr.set_grad(a, 2.0) # AD tracking not yet enabled
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 0.0)
    dr.enable_grad(a)
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, 0.0)
    dr.set_grad(a, 2.0)
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, 2.0)

    dr.set_grad(a, t(3, 4, 5))
    g = dr.grad(a)
    assert len(g) == 3 and dr.allclose(g, [3, 4, 5])

    with pytest.raises(RuntimeError, match="attempted to store a gradient of size 2 into AD variable"):
        dr.set_grad(a, t(1, 2))

    a = t(1)
    dr.enable_grad(a)
    dr.set_grad(a, t(3, 4, 5))
    g = dr.grad(a)
    assert len(g) == 1 and dr.allclose(g, 3+4+5)

    assert type(dr.grad(a)) is type(a)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    Array3f = getattr(sys.modules[t.__module__], 'Array3f')
    a = Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.set_grad(a, 1.0)
    assert dr.allclose(dr.grad(a), [[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    v = Array3f([2, 3, 4], [5, 6, 7], [8, 9, 10])
    dr.set_grad(a, v)
    assert dr.allclose(dr.grad(a), v)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    MyStruct = make_mystruct(t)
    a, b = MyStruct(), MyStruct()
    a.x = t(0, 0, 0)
    a.y = t(0, 0, 0)
    dr.enable_grad(a)
    b.x = t(1, 2, 3)
    b.y = t(4, 5, 6)
    dr.set_grad(a, b)

    c = dr.grad(a)
    assert dr.all(b.x == c.x)
    assert dr.all(b.y == c.y)

@pytest.test_arrays("is_diff,float,shape=(*)")
def test04_accum_grad(t):
    a = t([1, 2, 3])
    dr.accum_grad(a, 2) # AD tracking not yet enabled
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 0)
    dr.enable_grad(a)
    dr.accum_grad(a, 2)
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 2)
    dr.accum_grad(a, 2)
    g = dr.grad(a)
    assert len(a) == 3 and dr.allclose(g, 4)

    a = t([1])
    dr.enable_grad(a)
    dr.accum_grad(a, [1, 2, 3])
    g = dr.grad(a)
    assert len(a) == 1 and dr.allclose(g, 6)


@pytest.test_arrays("is_diff,float,shape=(*)")
def test05_set_label(t):
    a = t(1.0)
    b = [t(1.0), t(2.0)]
    Array3f = getattr(sys.modules[t.__module__], 'Array3f')
    c = Array3f(1.0, 2.0, 3.0)
    d = make_mystruct(t)()

    assert dr.label(a) is None
    dr.enable_grad(a, b, c, d)
    assert dr.label(a) is None

    dr.set_label(a, 'aa')
    assert dr.label(a) == 'aa'

    dr.set_label(a=a, b=b, c=c, d=d)
    assert dr.label(a) == 'a'
    assert dr.label(b[0]) == 'b_0'
    assert dr.label(b[1]) == 'b_1'
    assert dr.label(c.x) == 'c_0'
    assert dr.label(c.y) == 'c_1'
    assert dr.label(c.z) == 'c_2'
    assert dr.label(d.x) == 'd_x'
    assert dr.label(d.y) == 'd_y'

    with pytest.raises(TypeError, match="incompatible function arguments"):
        dr.set_label(a, 'aa', b=b)

#@pytest.test_arrays("is_diff,float,shape=(*)")
#def test06_forward_to(t):
#    a = t(1.0)
#    dr.enable_grad(a)
#    b = a * a * 2
#    c = a * 2
#    dr.set_grad(a, 1.0)
#    d = t(4.0) # some detached variable
#    grad_b, grad_c, grad_d = dr.forward_to(b, c, d)
#    assert dr.allclose(dr.grad(a), 0.0)
#    assert dr.allclose(grad_b, 4.0)
#    assert dr.allclose(grad_c, 2.0)
#    assert dr.allclose(grad_d, 0.0)
#
#    with pytest.raises(TypeError, matches="AD flags should be passed via the"):
#        dr.forward_to(b, c, dr.ADFlag.Default)
#
#    # Error because the input isn't attached to the AD graph
#    with pytest.raises(TypeError) as ei:
#        dr.forward_to(m.Float(2.0))
#    assert "the argument does not depend on the input" in str(ei.value)
#
#    # Error because the input isn't a diff array
#    with pytest.raises(TypeError) as ei:
#        dr.forward_to(dr.detached_t(m.Float)(2.0))
#    assert "expected a differentiable array type" in str(ei.value)
#
#    # Error because the input isn't a drjit array
#    with pytest.raises(TypeError) as ei:
#        dr.forward_to([2.0])
#    assert "expected a Dr.JIT array type" in str(ei.value)
#
#    # Trying to call with a different flag
#    dr.set_grad(a, 1.0)
#    b = a * a * 2
#    grad_b = dr.forward_to(b, flags=dr.ADFlag.ClearInterior)
#    assert dr.allclose(dr.grad(a), 1.0)
#    assert dr.allclose(grad_b, 4.0)
