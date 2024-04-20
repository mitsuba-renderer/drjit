import drjit as dr
import pytest
import sys
import math
import re

def make_mystruct(t):
    class MyStruct:
        def __init__(self) -> None:
            self.x = t(1)
            self.y = t(2)
        DRJIT_STRUCT = { 'x': t, 'y': t }
    return MyStruct

@pytest.test_arrays('is_diff,-mask,-shape=()')
def test001_enable_grad(t):
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

@pytest.test_arrays('is_diff,float,-shape=()')
def test002_detach(t):
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


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test003_set_grad(t):
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

    with pytest.raises(RuntimeError, match=re.escape("attempted to store a gradient of size 2 into AD variable")):
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


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test004_accum_grad(t):
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


@pytest.test_arrays('is_diff,float,shape=(*)')
def test005_set_label(t):
    a = t(1.0)
    b = [t(1.0), t(2.0)]
    Array3f = getattr(sys.modules[t.__module__], 'Array3f')
    c = Array3f(1.0, 2.0, 3.0)
    d = make_mystruct(t)()

    assert a.label is None
    dr.enable_grad(a, b, c, d)
    assert a.label is None

    dr.set_label(a, 'aa')
    assert a.label == 'aa'

    dr.set_label(a=a, b=b, c=c, d=d)
    assert a.label == 'a'
    assert b[0].label == 'b_0'
    assert b[1].label == 'b_1'
    assert c.x.label == 'c_0'
    assert c.y.label == 'c_1'
    assert c.z.label == 'c_2'
    assert d.x.label == 'd_x'
    assert d.y.label == 'd_y'

    with pytest.raises(TypeError, match='incompatible function arguments'):
        dr.set_label(a, 'aa', b=b)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test006_add_bwd(t):
    a, b = t(1), t(2)
    dr.enable_grad(a, b)
    c = 2 * a + b
    dr.backward(c)
    assert dr.grad(a) == 2
    assert dr.grad(b) == 1


@pytest.test_arrays('is_diff,float,shape=(*)')
def test007_add_fwd(t):
    if True:
        a, b = t(1), t(2)
        dr.enable_grad(a, b)
        c = 2 * a + b
        dr.forward(a, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 2
        dr.set_grad(c, 101)
        dr.forward(b)
        assert dr.grad(c) == 102

    if True:
        a, b = t(1), t(2)
        dr.enable_grad(a, b)
        c = 2 * a + b
        dr.set_grad(a, 1.0)
        dr.enqueue(dr.ADMode.Forward, a)
        dr.traverse(dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 2
        assert dr.grad(a) == 0
        dr.set_grad(a, 1.0)
        dr.enqueue(dr.ADMode.Forward, a)
        dr.traverse(dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 4


@pytest.test_arrays('is_diff,float,shape=(*)')
def test008_branch_fwd(t):
    a = t(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.forward(a)
    assert dr.grad(d) == 2


@pytest.test_arrays('is_diff,float,shape=(*)')
def test009_branch_ref(t):
    a = t(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.backward(d)
    assert dr.grad(a) == 2


@pytest.test_arrays('is_diff,float,shape=(*)')
def test010_forward_to(t):
  a = t(1.0)
  dr.enable_grad(a)
  b = a * a * 2
  c = a * 2
  dr.set_grad(a, 1.0)
  d = t(4.0) # some detached variable
  grad_b, grad_c, grad_d = dr.forward_to(b, c, d)
  assert dr.allclose(dr.grad(a), 0.0)
  assert dr.allclose(grad_b, 4.0)
  assert dr.allclose(grad_c, 2.0)
  assert dr.allclose(grad_d, 0.0)
  dr.forward_to(b, c, d, dr.ADFlag.Default)

  # Error because the input isn't attached to the AD graph
  with pytest.raises(RuntimeError, match='argument does not depend on the input'):
      dr.forward_to(t(1.0))

  dr.forward_to(t(1.0), flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)

  # Error because the input isn't a diff array
  with pytest.raises(RuntimeError, match='argument does not depend on the input'):
      dr.forward_to(dr.detached_t(t)(1.0))

  # Trying to call with a different flag
  dr.set_grad(a, 1.0)
  b = a * a * 2
  grad_b = dr.forward_to(b, flags=dr.ADFlag.ClearInterior)
  assert dr.allclose(dr.grad(a), 1.0)
  assert dr.allclose(grad_b, 4.0)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test011_forward_from(t):
    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.forward_from(a)

    dr.enable_grad(a)
    b = a * a * 2
    dr.forward_from(a)
    assert dr.allclose(dr.grad(a), 0.0)
    assert dr.allclose(dr.grad(b), 4.0)

    b = a * a * 2
    dr.forward_from(a, flags=dr.ADFlag.ClearInterior)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 4.0)

    # Interior gradients are cleared, forwarding again will accumulate gradients
    dr.forward_from(a, flags=dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(b), 8.0)

    # Edges are cleared, forwarding again will do nothing
    dr.forward_from(a, flags=dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 8.0)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test012_backward_to(t):
    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_to(1.0)

    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_to(a)

    b = t(3.0)
    dr.enable_grad(a, b)
    c = a * b * 2

    dr.set_grad(c, 1.0)
    dr.backward_to(a, flags=dr.ADFlag.ClearVertices)
    assert dr.allclose(dr.grad(a), 6.0)
    assert dr.allclose(dr.grad(b), 0.0)
    assert dr.allclose(dr.grad(c), 0.0)

    dr.set_grad(c, 1.0)
    dr.backward_to(a, b, flags=dr.ADFlag.ClearVertices)
    assert dr.allclose(dr.grad(a), 12.0) # accumulates
    assert dr.allclose(dr.grad(b), 2.0)
    assert dr.allclose(dr.grad(c), 0.0)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test013_backward_from(t):
    a = t(1.0)

    with pytest.raises(RuntimeError, match='argument does not depend on the input'):
        dr.backward_from(a)

    dr.enable_grad(a)
    b = a * a * 2
    dr.backward_from(b)
    assert dr.allclose(dr.grad(a), 4.0)
    assert dr.allclose(dr.grad(b), 0.0)

    a = t(1.0)
    dr.enable_grad(a)
    if '64' in t.__name__:
        Array3f = getattr(sys.modules[t.__module__], 'Array3f64')
    elif '16' in t.__name__:
        Array3f = getattr(sys.modules[t.__module__], 'Array3f16')
    else:
        Array3f = getattr(sys.modules[t.__module__], 'Array3f')

    c = Array3f(a)
    dr.backward_from(c)
    assert dr.allclose(dr.grad(a), 3.0)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test014_forward_to_reuse(t):
    a, b, c = t(1), t(2), t(3)
    dr.enable_grad(a, b, c)
    dr.set_grad(a, 10)
    dr.set_grad(b, 100)
    dr.set_grad(c, 1000)

    d, e, f = a + b, a + c, b + c
    g, h, i = d*d, e*e, f*f

    for k in range(2):
        for j, v in enumerate([g, h, i]):
            dr.set_grad(v, 0)
            dr.forward_to(v, flags=dr.ADFlag.ClearInterior)
            assert v == [9, 16, 25][j]
            assert dr.grad(v) == [660, 8080, 11000][j]
            assert dr.grad([g, h, i][(j + 1)%3]) == 0
            assert dr.grad([g, h, i][(j + 2)%3]) == 0
            dr.set_grad(v, t())

@pytest.test_arrays('is_diff,float,shape=(*)')
def test015_backward_to_reuse(t):
    a, b, c = t(1), t(2), t(3)
    dr.enable_grad(a, b, c)

    d, e, f = a + b, a + c, b + c
    g, h, i = d*d, e*e, f*f

    dr.set_grad(g, 10)
    dr.set_grad(h, 100)
    dr.set_grad(i, 1000)

    for k in range(2):
        for j, v in enumerate([a, b, c]):
            dr.backward_to(v, flags=dr.ADFlag.ClearInterior)
            assert dr.grad(v) == [860, 10060, 10800][j]
            assert dr.grad([a, b, c][(j + 1)%3]) == 0
            assert dr.grad([a, b, c][(j + 2)%3]) == 0
            dr.set_grad(v, t())

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test016_mixed_precision_bwd(t):
    t2 = dr.float64_array_t(t)

    a = t(1)
    dr.enable_grad(a)
    b = dr.sin(a)
    c = dr.sin(t2(a))
    d = t2(t(t2(a)))
    e = b + c + d + a
    dr.backward_from(e)
    assert dr.allclose(dr.grad(a), dr.cos(1) * 2 + 2)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test017_mixed_precision_fwd(t):
    t2 = dr.float64_array_t(t)

    a = t(1)
    dr.enable_grad(a)
    b = dr.sin(a)
    c = dr.sin(t2(a))
    d = t2(t(t2(a)))
    e = b + c + d + a
    dr.forward_from(a)
    assert dr.allclose(dr.grad(e), dr.cos(1) * 2 + 2)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test018_select_fwd(t):
    a = t(1)
    b = t(2)
    dr.enable_grad(a, b)
    dr.set_grad(a, 3)
    dr.set_grad(b, 4)
    c = dr.select(dr.mask_t(a)(True, False), a, b)
    gc = dr.forward_to(c)
    assert dr.allclose(c, t(1, 2))
    assert dr.allclose(gc, t(3, 4))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test019_nan_propagation(t):
    for i in range(2):
        x = dr.arange(t, 10)
        dr.enable_grad(x)
        f0 = t(0)
        y = dr.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        dr.backward(y)
        g = dr.grad(x)
        if i == 0:
            assert dr.allclose(g, 1)
        else:
            assert dr.all(dr.isnan(g))

    for i in range(2):
        x = dr.arange(t, 10)
        dr.enable_grad(x)
        f0 = t(0)
        y = dr.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        dr.forward(x)
        g = dr.grad(y)
        if i == 0:
            assert dr.allclose(g, 1)
        else:
            assert dr.all(dr.isnan(g))

@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.mark.parametrize("f1", [0, int(dr.ADFlag.ClearEdges)])
@pytest.mark.parametrize("f2", [0, int(dr.ADFlag.ClearInterior)])
@pytest.mark.parametrize("f3", [0, int(dr.ADFlag.ClearInput)])
def test020_ad_flags(t, f1, f2, f3):
    v0 = t(2)
    dr.enable_grad(v0)
    v1 = v0 * t(0.5)
    v2 = v0 + v1

    for i in range(2):
        dr.accum_grad(v0, 1 if i == 0 else 100)
        dr.enqueue(dr.ADMode.Forward, v0)
        dr.traverse(dr.ADMode.Forward, flags=(f1 | f2 | f3))

    if f1 == 0:
        if f2 == 0:
            if f3 == 0:
                assert dr.grad(v0) == 101
                assert dr.grad(v1) == 51
                assert dr.grad(v2) == 153.5
            else:
                assert dr.grad(v0) == 0
                assert dr.grad(v1) == 50.5
                assert dr.grad(v2) == 152
        else:
            if f3 == 0:
                assert dr.grad(v0) == 101
                assert dr.grad(v1) == 0
                assert dr.grad(v2) == 153
            else:
                assert dr.grad(v0) == 0
                assert dr.grad(v1) == 0
                assert dr.grad(v2) == 151.5
    else:
        if f2 == 0:
            if f3 == 0:
                assert dr.grad(v0) == 101
                assert dr.grad(v1) == 0.5
                assert dr.grad(v2) == 1.5
            else:
                assert dr.grad(v0) == 100
                assert dr.grad(v1) == 0.5
                assert dr.grad(v2) == 1.5
        else:
            if f3 == 0:
                assert dr.grad(v0) == 101
                assert dr.grad(v1) == 0
                assert dr.grad(v2) == 1.5
            else:
                assert dr.grad(v0) == 100
                assert dr.grad(v1) == 0
                assert dr.grad(v2) == 1.5


@pytest.test_arrays('is_diff,float,shape=(*)')
def test021_sum_0_bwd(t):
    x = dr.linspace(t, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.backward(y)
    assert len(y) == 1 
    assert dr.allclose(y, t(95.0/27.0))
    assert dr.allclose(dr.grad(x), 2 * dr.detach(x))


@pytest.test_arrays('is_diff,float,shape=(*)')
def test022_sum_0_fwd(t):
    x = dr.linspace(t, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(dr.detach(y), t(95.0/27.0))
    assert len(dr.grad(y)) == 1 and dr.allclose(dr.grad(y), 10)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test023_sum_1_bwd(t):
    x = dr.linspace(t, 0, 1, 9)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.backward(y)
    assert dr.allclose(dr.grad(x), t(9))


@pytest.test_arrays('is_diff,float,shape=(*)')
def test024_sum_1_fwd(t):
    x = dr.linspace(t, 0, 1, 9)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.forward(x)
    assert dr.allclose(dr.grad(y), 81)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test025_sum_2_bwd(t):
    x = dr.linspace(t, 0, 1, 9)
    dr.enable_grad(x)
    z = dr.sum(dr.sum(x*x)*x*x)
    dr.backward(z)
    assert dr.allclose(dr.grad(x), 
        [0, 1.59375, 3.1875, 4.78125, 6.375, 7.96875, 9.5625, 11.1562, 12.75])


@pytest.test_arrays('is_diff,float,shape=(*)')
def test026_sum_2_fwd(t):
    x = dr.linspace(t, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x*x)*dr.sum(x*x))
    dr.forward(x)
    assert dr.allclose(dr.grad(y), t(1900.0 / 27.0))


@pytest.test_arrays('is_diff,float,shape=(*)')
def test027_prod(t):
    x = t(1, 2, 5, 8)
    dr.enable_grad(x)
    y = dr.prod(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 80)
    assert dr.allclose(dr.grad(x), [80, 40, 16, 10])


@pytest.test_arrays('is_diff,float,shape=(*)')
def test028_max_bwd(t):
    x = t(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(x), [0, 0, 1, 0, 1])


@pytest.test_arrays('is_diff,float,shape=(*)')
def test029_max_fwd(t):
    x = t(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(y), [2])  # Approximation


@pytest.test_arrays('is_diff,float,shape=(*)')
def test030_gather_bwd(t):
    x = dr.linspace(t, -1, 1, 9)
    dr.enable_grad(x)
    y = dr.gather(t, x*x, dr.uint32_array_t(t)(1, 1, 2, 3))
    z = dr.sum(y)
    dr.backward(z)
    ref = [0, -3, -1, -0.5, 0, 0, 0, 0, 0]
    assert dr.allclose(dr.grad(x), ref)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test031_gather_fwd(t):
    x = dr.linspace(t, -1, 1, 9)
    dr.enable_grad(x)
    y = dr.gather(t, x*x, dr.uint32_array_t(t)(1, 1, 2, 3))
    dr.forward(x)
    ref = [-1.5, -1.5, -1, -0.5]
    assert dr.allclose(dr.grad(y), ref)


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test032_scatter_bwd(t):
    m = sys.modules[t.__module__]
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 4) + 3

        x = dr.linspace(t, 0, 1, 5)
        y = dr.linspace(t, 1, 2, 4)
        buf = dr.zeros(t, 10)

        if i % 2 == 0:
            dr.enable_grad(buf)
        if i // 2 == 0:
            dr.enable_grad(x, y)

        dr.set_label(x, "x")
        dr.set_label(y, "y")
        dr.set_label(buf, "buf")

        buf2 = t(buf)
        dr.scatter(buf2, x, idx1)
        dr.eval(buf2)
        dr.scatter(buf2, y, idx2)

        ref_buf = t(0.0000, 0.2500, 0.5000, 1.0000, 1.3333,
                    1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        assert dr.allclose(ref_buf, buf2, atol=1e-4)

        s = dr.dot(buf2, buf2)

        dr.backward(s)

        ref_x = t(0.0000, 0.5000, 1.0000, 0.0000, 0.0000)
        ref_y = t(2.0000, 2.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert dr.allclose(dr.grad(y), dr.detach(ref_y), atol=1e-4)
            assert dr.allclose(dr.grad(x), dr.detach(ref_x), atol=1e-4)
        else:
            assert dr.all(dr.grad(x) == 0)
            assert dr.all(dr.grad(y) == 0)

        if i % 2 == 0:
            assert dr.allclose(dr.grad(buf), 0, atol=1e-4)
        else:
            assert dr.all(dr.grad(buf) == 0)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test033_scatter_fwd(t):
    m = sys.modules[t.__module__]
    x = t(4.0)
    dr.enable_grad(x)

    values = x * x * dr.linspace(t, 1, 4, 4)
    idx = 2 * dr.arange(m.UInt, 4)

    buf = dr.zeros(t, 10)
    dr.scatter(buf, values, idx)

    assert dr.grad_enabled(buf)

    ref = [16.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert dr.allclose(buf, ref)

    dr.forward(x, flags=dr.ADFlag.ClearVertices)
    grad = dr.grad(buf)

    ref_grad = [8.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert dr.allclose(grad, ref_grad)

    # Overwrite first value with non-diff value, resulting gradient entry should be 0
    y = t(3)
    idx = m.UInt(0)
    dr.scatter(buf, y, idx)

    ref = [3.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert dr.allclose(buf, ref)

    dr.forward(x)
    grad = dr.grad(buf)

    ref_grad = [0.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert dr.allclose(grad, ref_grad)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test034_scatter_fwd_permute(t):
    m = sys.modules[t.__module__]
    x = t(4.0)
    dr.enable_grad(x)

    values_0 = x * dr.linspace(t, 1, 9, 5)
    values_1 = x * dr.linspace(t, 11, 19, 5)

    buf = dr.zeros(t, 10)

    idx_0 = dr.arange(m.UInt, 5)
    idx_1 = dr.arange(m.UInt, 5) + 5

    dr.scatter(buf, values_0, idx_0, mode=dr.ReduceMode.Permute)
    dr.scatter(buf, values_1, idx_1, mode=dr.ReduceMode.Permute)

    ref = [4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0]
    assert dr.allclose(buf, ref)

    dr.forward(x)
    grad = dr.grad(buf)

    ref_grad = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
    assert dr.allclose(grad, ref_grad)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test035_scatter_add_bwd(t):
    m = sys.modules[t.__module__]
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 5) + 3

        x = dr.linspace(t, 0, 1, 5)
        y = dr.linspace(t, 1, 2, 5)
        buf = dr.zeros(t, 10)

        if i % 2 == 0:
            dr.enable_grad(buf)
        if i // 2 == 0:
            dr.enable_grad(x, y)

        dr.set_label(x, "x")
        dr.set_label(y, "y")
        dr.set_label(buf, "buf")

        buf2 = t(buf)
        dr.scatter_add(buf2, x, idx1)
        dr.scatter_add(buf2, y, idx2)

        ref_buf = t(0.0000, 0.2500, 0.5000, 1.7500, 2.25,
                          1.5, 1.75, 2.0000, 0.0000, 0.0000)

        assert dr.allclose(ref_buf, buf2)

        s = dr.dot(buf2, buf2)

        dr.backward(s)

        ref_x = t(0.0000, 0.5000, 1.0000, 3.5000, 4.5)
        ref_y = t(3.5000, 4.5, 3, 3.5, 4.000)

        if i // 2 == 0:
            assert dr.allclose(dr.grad(y), dr.detach(ref_y))
            assert dr.allclose(dr.grad(x), dr.detach(ref_x))
        else:
            assert dr.all(dr.grad(x) == 0)
            assert dr.all(dr.grad(y) == 0)

        if i % 2 == 0:
            assert dr.allclose(dr.grad(buf), dr.detach(ref_buf) * 2)
        else:
            assert dr.all(dr.grad(buf) == 0)


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test036_scatter_add_fwd(t):
    m = sys.modules[t.__module__]
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 4) + 3

        x = dr.linspace(t, 0, 1, 5)
        y = dr.linspace(t, 1, 2, 4)
        buf = dr.zeros(t, 10)

        if i % 2 == 0:
            dr.enable_grad(buf)
            dr.set_grad(buf, 1)
        if i // 2 == 0:
            dr.enable_grad(x, y)
            dr.set_grad(x, 1)
            dr.set_grad(y, 1)

        dr.set_label(x, "x")
        dr.set_label(y, "y")
        dr.set_label(buf, "buf")

        buf2 = t(buf)
        dr.scatter_add(buf2, x, idx1)
        dr.scatter_add(buf2, y, idx2)

        s = dr.dot(buf2, buf2)

        if i % 2 == 0:
            dr.enqueue(dr.ADMode.Forward, buf)
        if i // 2 == 0:
            dr.enqueue(dr.ADMode.Forward, x, y)

        dr.traverse(dr.ADMode.Forward)

        # Verified against Mathematica
        assert dr.allclose(dr.detach(s), t(15.5972))
        assert dr.allclose(dr.grad(s), t((25.1667 if i // 2 == 0 else 0)
                           + (17 if i % 2 == 0 else 0)))

counter = 37

def std_test(name, func, f_in, f_out, grad_out):
    global counter

    if not isinstance(f_in, tuple):
        f_in = (f_in, )

    if not isinstance(grad_out, tuple):
        grad_out = (grad_out, )

    def test_func(t):
        args = tuple(t(v) for v in f_in)
        dr.enable_grad(args)
        rv = func(*args)
        assert dr.allclose(rv, t(f_out))
        dr.backward_from(rv, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
        g_out = tuple(t(v) for v in grad_out)
        assert dr.allclose(dr.grad(args), g_out)

    query = pytest.test_arrays('is_diff,float,shape=(*)')
    test_func_param = query(test_func)

    globals()[f'test{counter:02}_op_{name}'] = test_func_param
    counter += 1

# Spot-check various builtin operations
std_test('neg', lambda a: -a, 3, -3, (-1))
std_test('add', lambda a, b: a + b, (2, 3), 5, (1, 1))
std_test('sub', lambda a, b: a - b, (2, 3), -1, (1, -1))
std_test('mul', lambda a, b: a * b, (2, 3), 6, (3, 2))
std_test('div', lambda a, b: a / b, (2, 3), 2 / 3, (1 / 3, -2 / 9))
std_test('fma', lambda a, b, c: dr.fma(a, b, c), (3, 7, 11), 32, (7, 3, 1))
std_test('abs_pos', lambda a: abs(a), 3, 3, 1)
std_test('abs_neg', lambda a: abs(a), -3, 3, -1)
std_test('rcp', lambda a: dr.rcp(a), 3, 1/3, -1/9)
std_test('rsqrt', lambda a: dr.rsqrt(a), 2, 1/dr.sqrt(2), -1/(4*dr.sqrt(2)))
std_test('cbrt', lambda a: dr.cbrt(a), 8, 8**(1/3), 1/(3 * 8 **(2/3)))
std_test('erf', lambda a: dr.erf(a), .2, math.erf(.2), 2*math.exp(-.2**2)/math.sqrt(math.pi))
std_test('log', lambda a: dr.log(a), 2, math.log(2), .5)
std_test('log2', lambda a: dr.log2(a), 2, math.log2(2), 1/(2*math.log(2)))
std_test('exp', lambda a: dr.exp(a), 2, math.exp(2), math.exp(2))
std_test('exp2', lambda a: dr.exp2(a), 2, 2**2, 4*math.log(2))

std_test('sin', lambda a: dr.sin(a), 1, math.sin(1), math.cos(1))
std_test('cos', lambda a: dr.cos(a), 1, math.cos(1), -math.sin(1))
std_test('tan', lambda a: dr.tan(a), .5, math.tan(.5), 1/math.cos(.5)**2)
std_test('asin', lambda a: dr.asin(a), .5, math.asin(.5), 1/math.sqrt(1 - .5**2))
std_test('acos', lambda a: dr.acos(a), .5, math.acos(.5), -1/math.sqrt(1 - .5**2))
std_test('atan', lambda a: dr.atan(a), .5, math.atan(.5), 1/(1 + .5**2))

std_test('sinh', lambda a: dr.sinh(a), 1, math.sinh(1), math.cosh(1))
std_test('cosh', lambda a: dr.cosh(a), 1, math.cosh(1), math.sinh(1))
std_test('tanh', lambda a: dr.tanh(a), 1, math.tanh(1), 1/math.cosh(1)**2)
std_test('asinh', lambda a: dr.asinh(a), .5, math.asinh(.5), 1/math.sqrt(1 + .5**2))
std_test('acosh', lambda a: dr.acosh(a), 1.5, math.acosh(1.5), 1/math.sqrt(1.5**2 - 1))
std_test('atanh', lambda a: dr.atanh(a), .5, math.atanh(.5), 1/(1 - .5**2))

std_test('sincos_s', lambda a: dr.sincos(a)[0], 1, math.sin(1), math.cos(1))
std_test('sincos_c', lambda a: dr.sincos(a)[1], 1, math.cos(1), -math.sin(1))
std_test('sincosh_s', lambda a: dr.sincosh(a)[0], 1, math.sinh(1), math.cosh(1))
std_test('sincosh_c', lambda a: dr.sincosh(a)[1], 1, math.cosh(1), math.sinh(1))

std_test('atan2_1', lambda a, b: dr.atan2(a, b), (1, 2), math.atan2(1, 2), (2/5, -1/5))
std_test('atan2_2', lambda a, b: dr.atan2(a, b), (-1, 2), math.atan2(-1, 2), (2/5, 1/5))

std_test('round', lambda a: dr.round(a), 1.6, 2.0, 0.0)
std_test('trunc', lambda a: dr.trunc(a), 1.6, 1.0, 0.0)
std_test('ceil', lambda a: dr.ceil(a), 1.6, 2.0, 0.0)
std_test('floor', lambda a: dr.floor(a), 1.6, 1.0, 0.0)



@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test075_exp(t):
    x = dr.linspace(t, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.exp(x * x)
    dr.backward(y)
    exp_x = dr.exp(dr.square(dr.detach(x)))
    assert dr.allclose(y, exp_x)
    assert dr.allclose(dr.grad(x), 2 * dr.detach(x) * exp_x)


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test076_log(t):
    x = dr.linspace(t, 0.01, 1, 10)
    dr.enable_grad(x)
    y = dr.log(x * x)
    dr.backward(y)
    log_x = dr.log(dr.square(dr.detach(x)))
    assert dr.allclose(y, log_x)
    assert dr.allclose(dr.grad(x), 2 / dr.detach(x))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test077_pow(t):
    x = dr.linspace(t, 1, 10, 10)
    y = dr.full(t, 2.0, 10)
    dr.enable_grad(x, y)
    z = dr.power(x, y)
    dr.backward(z)
    assert dr.allclose(dr.grad(x), dr.detach(x)*2)
    assert dr.allclose(dr.grad(y),
                       t(0., 2.77259, 9.88751, 22.1807, 40.2359,
                               64.5033, 95.3496, 133.084, 177.975, 230.259))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test078_tan(t):
    x = dr.linspace(t, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.tan(x * x)
    dr.backward(y)
    tan_x = dr.tan(dr.square(dr.detach(x)))
    assert dr.allclose(y, tan_x)
    assert dr.allclose(dr.grad(x),
                       t(0., 0.222256, 0.44553, 0.674965, 0.924494,
                               1.22406, 1.63572, 2.29919, 3.58948, 6.85104))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test079_asin(t):
    x = dr.linspace(t, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.asin(x * x)
    dr.backward(y)
    asin_x = dr.asin(dr.square(dr.detach(x)))
    assert dr.allclose(y, asin_x)
    assert dr.allclose(dr.grad(x),
                       t(-2.08232, -1.3497, -0.906755, -0.534687,
                               -0.177783, 0.177783, 0.534687, 0.906755,
                               1.3497, 2.08232))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test080_acos(t):
    x = dr.linspace(t, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.acos(x * x)
    dr.backward(y)
    acos_x = dr.acos(dr.square(dr.detach(x)))
    assert dr.allclose(y, acos_x)
    assert dr.allclose(dr.grad(x),
                       t(2.08232, 1.3497, 0.906755, 0.534687, 0.177783,
                               -0.177783, -0.534687, -0.906755, -1.3497,
                               -2.08232))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test081_atan(t):
    x = dr.linspace(t, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.atan(x * x)
    dr.backward(y)
    atan_x = dr.atan(dr.square(dr.detach(x)))
    assert dr.allclose(y, atan_x)
    assert dr.allclose(dr.grad(x),
                       t(-1.13507, -1.08223, -0.855508, -0.53065,
                               -0.177767, 0.177767, 0.53065, 0.855508, 1.08223,
                               1.13507))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test082_atan2(t):
    x = dr.linspace(t, -.8, .8, 10)
    Int = getattr(sys.modules[t.__module__], 'Int')
    y = t(dr.arange(Int, 10) & 1) * 1 - .5
    dr.enable_grad(x, y)
    z = dr.atan2(y, x)
    dr.backward(z)
    assert dr.allclose(z, t(-2.58299, 2.46468, -2.29744, 2.06075,
                            -1.74674, 1.39486, -1.08084, 0.844154,
                            -0.676915, 0.558599))
    assert dr.allclose(dr.grad(x),
                       t(0.561798, -0.784732, 1.11724, -1.55709, 1.93873,
                         -1.93873, 1.55709, -1.11724, 0.784732,
                         -0.561798))
    assert dr.allclose(dr.grad(y),
                       t(-0.898876, -0.976555, -0.993103, -0.83045,
                         -0.344663, 0.344663, 0.83045, 0.993103,
                          0.976555, 0.898876))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test083_cbrt(t):
    x = dr.linspace(t, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.cbrt(x)
    dr.backward(y)
    assert dr.allclose(y, t(-0.928318, -0.853719, -0.763143, -0.64366,
                                  -0.446289, 0.446289, 0.64366, 0.763143,
                                  0.853719, 0.928318))
    assert dr.allclose(dr.grad(x),
                       t(0.386799, 0.45735, 0.572357, 0.804574, 1.67358,
                               1.67358, 0.804574, 0.572357, 0.45735, 0.386799))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test084_sinh(t):
    x = dr.linspace(t, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.sinh(x)
    dr.backward(y)
    assert dr.allclose(
        y, t(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                   0.11134, 0.339541, 0.584578, 0.858602, 1.1752))
    assert dr.allclose(
        dr.grad(x),
        t(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test085_cosh(t):
    x = dr.linspace(t, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.cosh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        t(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))
    assert dr.allclose(
        dr.grad(x),
        t(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                0.11134, 0.339541, 0.584578, 0.858602, 1.1752))


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test086_tanh(t):
    x = dr.linspace(t, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.tanh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        t(-0.761594, -0.651429, -0.504672, -0.321513, -0.110656,
                0.110656, 0.321513, 0.504672, 0.651429, 0.761594))
    assert dr.allclose(
        dr.grad(x),
        t(0.419974, 0.57564, 0.745306, 0.89663, 0.987755, 0.987755,
                0.89663, 0.745306, 0.57564, 0.419974)
    )


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test087_asinh(t):
    x = dr.linspace(t, -.9, .9, 10)
    dr.enable_grad(x)
    y = dr.asinh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        t(-0.808867, -0.652667, -0.481212, -0.295673, -0.0998341,
                0.0998341, 0.295673, 0.481212, 0.652667, 0.808867))
    assert dr.allclose(
        dr.grad(x),
        t(0.743294, 0.819232, 0.894427, 0.957826, 0.995037,
                0.995037, 0.957826, 0.894427, 0.819232, 0.743294)
    )


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test088_acosh(t):
    x = dr.linspace(t, 1.01, 2, 10)
    dr.enable_grad(x)
    y = dr.acosh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        t(0.141304, 0.485127, 0.665864, 0.802882, 0.916291,
                1.01426, 1.10111, 1.17944, 1.25098, 1.31696))
    assert dr.allclose(
        dr.grad(x),
        t(7.05346, 1.98263, 1.39632, 1.12112, 0.952381,
                0.835191, 0.747665, 0.679095, 0.623528, 0.57735)
    )


@pytest.test_arrays('is_diff,float,-float16,shape=(*)')
def test089_atanh(t):
    x = dr.linspace(t, -.99, .99, 10)
    dr.enable_grad(x)
    y = dr.atanh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        t(-2.64665, -1.02033, -0.618381, -0.342828, -0.110447, 0.110447,
                0.342828, 0.618381, 1.02033, 2.64665))
    assert dr.allclose(
        dr.grad(x),
        t(50.2513, 2.4564, 1.43369, 1.12221, 1.01225, 1.01225, 1.12221,
                1.43369, 2.4564, 50.2513)
    )


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test090_replace_grad(t):
    m = sys.modules[t.__module__]

    with pytest.raises(RuntimeError) as ei:
       dr.replace_grad(t(1,2,3),t(1,2,3,4))
    assert "incompatible input sizes" in str(ei.value)

    with pytest.raises(RuntimeError) as ei:
       dr.replace_grad((1.0,), (m.UInt(1),))
    assert "incompatible input types." in str(ei.value.__cause__)

    x = m.Array3f(1, 2, 3)
    y = m.Array3f(3, 2, 1)
    dr.enable_grad(x, y)
    x2 = x*x
    y2 = y*y
    z = dr.replace_grad(x2, y2)
    assert z.x.index_ad == y2.x.index_ad
    assert z.y.index_ad == y2.y.index_ad
    assert z.z.index_ad == y2.z.index_ad
    z2 = z*z
    assert dr.allclose(z2, [1, 16, 81])
    dr.backward(z2)
    assert dr.allclose(dr.grad(x), 0)
    assert dr.allclose(dr.grad(y), [12, 32, 36])

    x = m.Array3f(1, 2, 3)
    y = t(1)
    dr.enable_grad(x, y)
    z = dr.replace_grad(x, y)
    assert z.x.index_ad == y.index_ad
    assert z.y.index_ad == y.index_ad
    assert z.z.index_ad == y.index_ad

    a = t(1.0)
    dr.enable_grad(a)
    b = dr.replace_grad(4.0, a)
    assert type(b) is type(a)
    assert b.index_ad == a.index_ad
    assert dr.allclose(b, 4.0)

    a = m.ArrayXf(1, 2, 3)
    y = t(1)
    dr.enable_grad(x, y)
    z = dr.replace_grad(x, y)
    assert z[0].index_ad == y.index_ad
    assert z[1].index_ad == y.index_ad
    assert z[2].index_ad == y.index_ad


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test091_safe_functions(t):
    x = dr.linspace(t, 0, 1, 10)
    y = dr.linspace(t, -1, 1, 10)
    z = dr.linspace(t, -1, 1, 10)
    dr.enable_grad(x, y, z)
    x2 = dr.safe_sqrt(x)
    y2 = dr.safe_acos(y)
    z2 = dr.safe_asin(z)
    dr.backward(x2)
    dr.backward(y2)
    dr.backward(z2)
    assert dr.grad(x)[0] == 0
    assert dr.allclose(dr.grad(x)[1], .5 / dr.sqrt(1 / 9))
    assert x[0] == 0
    assert dr.all(dr.isfinite(dr.grad(x)))
    assert dr.all(dr.isfinite(dr.grad(y)))
    assert dr.all(dr.isfinite(dr.grad(z)))


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test092_prefix_sum_fwd(t):
    x = t([1,2,3,4])
    dr.enable_grad(x)
    y = dr.prefix_sum(x)
    assert dr.allclose(y, [0, 1, 3, 6])

    dr.set_grad(x, [1.1, 1.2, 1.3, 1.4])
    dr.forward_to(y)
    assert dr.allclose(dr.grad(y), [0, 1.1, 2.3, 3.6])

    x = t([1,2,3,4])
    dr.enable_grad(x)
    y = dr.prefix_sum(x, False)
    assert dr.allclose(y, [1, 3, 6, 10])
    dr.set_grad(x, [1.1, 1.2, 1.3, 1.4])
    dr.forward_to(y)
    assert dr.allclose(dr.grad(y), [1.1, 2.3, 3.6, 5])

    x = t([1,2,3,4])
    dr.enable_grad(x)
    y = dr.prefix_sum(x, False)
    dr.forward_from(x)
    assert dr.allclose(dr.grad(y), [1, 2, 3, 4])


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test093_prefix_sum_bwd(t):
    x = t([1, 2, 3, 4])
    dr.enable_grad(x)
    y = dr.prefix_sum(x)
    assert dr.allclose(y, [0, 1, 3, 6])
    dr.set_grad(y, [1.1, 1.2, 1.3, 1.4])
    dr.backward_to(x)
    assert dr.allclose(dr.grad(x), [1.2+1.3+1.4, 1.3+1.4, 1.4, 0])

    x = t([1, 2, 3, 4])
    dr.enable_grad(x)
    y = dr.prefix_sum(x, False)
    dr.set_grad(y, 1)
    dr.backward_to(x)
    assert dr.allclose(dr.grad(x), [4, 3, 2, 1])


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test094_suspend_resume(t):
    a = t(1)
    b = t(1)
    c = t(1)
    dr.enable_grad(a, b, c)

    with dr.suspend_grad():
        assert not dr.grad_enabled(a) and \
               not dr.grad_enabled(b) and \
               not dr.grad_enabled(c) and \
               not dr.grad_enabled(a, b, c)
        d = a + b + c
        assert not dr.grad_enabled(d)

        with dr.resume_grad():
            assert dr.grad_enabled(a) and \
                   dr.grad_enabled(b) and \
                   dr.grad_enabled(c) and \
                   dr.grad_enabled(a, b, c)
            e = a + b + c
            assert dr.grad_enabled(e)
            pass

        # dr.enable_grad() is ignored in a full dr.suspend_grad() session
        e = t(1)
        dr.enable_grad(e)
        assert not dr.grad_enabled(a, b, c, d, e)

        # Replicating suspended variables creates detached copies
        f = t(a)
        with dr.resume_grad():
            assert dr.grad_enabled(a) and \
                   not dr.grad_enabled(f)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test095_suspend_resume_selective(t):
    a = t(1)
    b = t(1)
    c = t(1)
    d = t(1)
    dr.enable_grad(a, b, c)

    with dr.suspend_grad():
        with dr.resume_grad(a, b):
            dr.enable_grad(d)
            assert dr.grad_enabled(a) and \
                   dr.grad_enabled(b) and \
                   not dr.grad_enabled(c) and \
                   dr.grad_enabled(d)
            with dr.suspend_grad(b):
                assert dr.grad_enabled(a) and \
                       not dr.grad_enabled(b) and \
                       not dr.grad_enabled(c)

    with dr.suspend_grad(a, b):
        assert not dr.grad_enabled(a) and \
               not dr.grad_enabled(b) and \
               dr.grad_enabled(c)
        with dr.resume_grad(b):
            assert not dr.grad_enabled(a) and \
                   dr.grad_enabled(b) and \
                   dr.grad_enabled(c)
        with dr.resume_grad():
            assert dr.grad_enabled(a) and \
                   dr.grad_enabled(b) and \
                   dr.grad_enabled(c)

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test096_isolate_bwd(t):
    a = t(1)
    dr.enable_grad(a)

    b = a * 2

    with dr.isolate_grad():
        c = b * 2

        with dr.isolate_grad():
            d = c * 2
            dr.backward(d)

            assert dr.grad(d) == 0 and \
                   dr.grad(c) == 2 and \
                   dr.grad(b) == 0 and \
                   dr.grad(a) == 0

        assert dr.grad(d) == 0 and \
               dr.grad(c) == 0 and \
               dr.grad(b) == 4 and \
               dr.grad(a) == 0

    assert dr.grad(d) == 0 and \
           dr.grad(c) == 0 and \
           dr.grad(b) == 0 and \
           dr.grad(a) == 8

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test097_isolate_fwd(t):
    # Tests that repeatedly forward-propagating through a shared subgraph
    # leaves the part outside of the isolation boundary intact (Not really sure
    # if we need this, but that's how the operation behaves)

    if True:
        a = t(0)
        dr.enable_grad(a)
        dr.set_grad(a, 2)

        b = a * 2
        db = dr.forward_to(b)
        assert db == 4

        c = a * 3
        dc = dr.forward_to(c)
        assert dc == 0

    if True:
        a = t(0)
        dr.enable_grad(a)
        dr.set_grad(a, 2)

        with dr.isolate_grad():
            b = a * 2
            db = dr.forward_to(b)
            assert db == 4

            c = a * 3
            dc = dr.forward_to(c)
            assert dc == 6

class Normalize(dr.CustomOp):
    def eval(self, value):
        self.value = value
        self.inv_norm = dr.rcp(dr.norm(value))
        return value * self.inv_norm

    def forward(self):
        grad_in = self.grad_in('value')
        grad_out = grad_in * self.inv_norm
        grad_out -= self.value * (dr.dot(self.value, grad_out) * dr.square(self.inv_norm))
        self.set_grad_out(grad_out)

    def backward(self):
        grad_out = self.grad_out()
        grad_in = grad_out * self.inv_norm
        grad_in -= self.value * (dr.dot(self.value, grad_in) * dr.square(self.inv_norm))
        self.set_grad_in('value', grad_in)

    def name(self):
        return "Normalize"

class Swap(dr.CustomOp):
    def eval(self, x, y):
        return y, x

    def forward(self):
        self.set_grad_out((self.grad_in('y'), self.grad_in('x')))

    def backward(self):
        grad_out = self.grad_out()
        self.set_grad_in('x', grad_out[1])
        self.set_grad_in('y', grad_out[0])

    def name(self):
        return "Swap"


class Copy(dr.CustomOp):
    def eval(self, x):
        return x, x

    def forward(self):
        self.set_grad_out((self.grad_in('x'), self.grad_in('x')))

    def backward(self):
        grad_out = self.grad_out()
        self.set_grad_in('x', grad_out[0] + grad_out[1])

    def name(self):
        return "Copy"

class ArgsKwargs(dr.CustomOp):
    def eval(self, *args, **kwargs):
        return args[0], kwargs['key']

    def forward(self):
        self.set_grad_out((self.grad_in('args')[0], self.grad_in('kwargs')['key']))

    def name(self):
        return "ArgsKWargs"

@pytest.test_arrays('is_diff,float,-float16,shape=(3, *)')
def test097_custom_op_fwd_1(t):
    d = t(1, 2, 3)
    dr.enable_grad(d)
    d2 = dr.custom(Normalize, d)
    dr.set_grad(d, t(5, 6, 7))
    dr.enqueue(dr.ADMode.Forward, d)
    dr.traverse(dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
    assert dr.all(dr.grad(d) == 0)
    dr.set_grad(d, t(5, 6, 7))
    assert dr.allclose(dr.grad(d2), t(0.610883, 0.152721, -0.305441))
    dr.enqueue(dr.ADMode.Forward, d)
    dr.traverse(dr.ADMode.Forward)
    assert dr.allclose(dr.grad(d2), t(0.610883, 0.152721, -0.305441)*2)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test098_custom_op_fwd_2(t):
    x, y = t(1), t(2)
    dr.enable_grad(x, y)
    dr.set_grad(x, 10)
    dr.set_grad(y, 20)
    a, b = dr.custom(Swap, x, y)
    ga, gb = dr.forward_to(a, b)
    assert dr.allclose((ga, gb), [20, 10])


@pytest.test_arrays('is_diff,float,shape=(*)')
def test099_custom_op_fwd_3(t):
    x = t(1)
    dr.enable_grad(x)
    dr.set_grad(x, 10)
    a, b = dr.custom(Copy, x)
    ga, gb = dr.forward_to(a, b)
    assert dr.allclose((ga, gb), [10, 10])


@pytest.test_arrays('is_diff,float,-float16,shape=(3, *)')
def test100_custom_op_bwd_1(t):
    d = t(1, 2, 3)
    dr.enable_grad(d)
    d2 = dr.custom(Normalize, d)
    dr.set_grad(d2, t(5, 6, 7))
    dr.backward_to(d)
    assert dr.allclose(dr.grad(d), t(0.610883, 0.152721, -0.305441))

@pytest.test_arrays('is_diff,float,shape=(*)')
def test101_custom_op_bwd_2(t):
    x, y = t(1), t(2)
    dr.enable_grad(x, y)
    a, b = dr.custom(Swap, x, y)
    dr.set_grad(a, 10)
    dr.set_grad(b, 20)
    gx, gy = dr.backward_to(x, y)
    assert dr.allclose((gx, gy), [20, 10])

@pytest.test_arrays('is_diff,float,shape=(*)')
def test102_custom_op_bwd_3(t):
    x = t(1)
    dr.enable_grad(x)
    a, b = dr.custom(Copy, x)
    dr.set_grad(a, 10)
    dr.set_grad(b, 20)
    gx = dr.backward_to(x)
    assert dr.allclose(gx, 30)

@pytest.test_arrays('is_diff,float,shape=(*)')
def test103_custom_forward_external_dependency(t):
    theta = t(2)
    dr.enable_grad(theta)

    param = theta * 3.0

    class GlobalOp(dr.CustomOp):
        def eval(self, value):
            self.add_input(param)
            self.value = value
            return value * param

        def forward(self):
            grad_value = self.grad_in('value')
            grad_param = dr.grad(param)
            value = param * grad_value + self.value * grad_param
            self.set_grad_out(value)

        def name(self):
            return "global-op"

    dr.set_grad(theta, 1.0)
    out = dr.custom(GlobalOp, 123)
    dr.forward_to(out)

    assert out == 123*6
    assert dr.grad(out) == 3*123

@pytest.test_arrays('is_diff,float,shape=(*)')
def test104_suspend_resume_custom_fwd(t):
    v_implicit, v_input = t(1), t(1)
    check = [0]

    class TestOp(dr.CustomOp):
        def eval(self, value):
            self.add_input(v_implicit)
            assert dr.grad_enabled(v_implicit) == check[0]
            return value + dr.detach(v_implicit)

        def forward(self):
            assert dr.grad_enabled(v_implicit) == check[0]
            self.set_grad_out(self.grad_in('value') + dr.grad(v_implicit))

    for i in range(4):
        dr.disable_grad(v_implicit, v_input)
        dr.enable_grad(v_implicit, v_input)
        dr.set_grad(v_implicit, 1)
        dr.set_grad(v_input, 1)
        check[0] = (i & 1) == 0

        with dr.suspend_grad(
                v_implicit if i & 1 else None,
                v_input if i & 2 else None):
            output = dr.custom(TestOp, v_input)
            assert dr.detach(output) == 2
            assert (i == 3 and not dr.grad_enabled(output)) or \
                   (i <  3 and dr.grad_enabled(output))

            dr.enqueue(dr.ADMode.Forward, v_implicit, v_input)
            dr.traverse(dr.ADMode.Forward)
            assert dr.grad(output) == 2 - (i & 1) - ((i & 2) >> 1)


@pytest.test_arrays('is_diff,float,shape=(*)')
def test105_suspend_resume_custom_bwd(t):
    v_implicit, v_input = t(1), t(1)
    check = [0]

    class TestOp(dr.CustomOp):
        def eval(self, value):
            self.add_input(v_implicit)
            assert dr.grad_enabled(v_implicit) == check[0]
            return value + dr.detach(v_implicit)

        def backward(self):
            assert dr.grad_enabled(v_implicit) == check[0]
            g = self.grad_out()
            dr.accum_grad(v_implicit, g)
            self.set_grad_in('value', g)

    for i in range(4):
        dr.disable_grad(v_implicit, v_input)
        dr.enable_grad(v_implicit, v_input)
        check[0] = (i & 1) == 0

        with dr.suspend_grad(
                v_implicit if i & 1 else None,
                v_input if i & 2 else None):
            output = dr.custom(TestOp, v_input)
            assert dr.detach(output) == 2
            assert (i == 3 and not dr.grad_enabled(output)) or \
                   (i <  3 and dr.grad_enabled(output))

            dr.enqueue(dr.ADMode.Backward, output)
            dr.set_grad(output, 1)
            dr.traverse(dr.ADMode.Backward)
            assert dr.grad(v_implicit) == ((i & 1) == 0)
            assert dr.grad(v_input) == ((i & 2) == 0)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test106_custom_op_leak(t):
    # Ensure that a non-traversed custom op is correctly GCed
    x = t(1)
    dr.enable_grad(x)
    dr.custom(Copy, x)


class Fail(dr.CustomOp):
    def eval(self, x):
        return x

    def forward(self):
        raise RuntimeError("Forward traversal failed")

    def backward(self):
        raise RuntimeError("Backward traversal failed")

    def name(self):
        return "Fail"


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test107_custom_op_fail_fwd(t):
    x = t(1)
    dr.enable_grad(x)
    y = dr.custom(Fail, x)
    with pytest.raises(RuntimeError, match='Forward traversal failed'):
        dr.forward_from(x)


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test108_custom_op_fail_bwd(t):
    x = t(1)
    dr.enable_grad(x)
    y = dr.custom(Fail, x)
    with pytest.raises(RuntimeError, match='Backward traversal failed'):
        dr.backward_from(y)

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test109_infinite_recursion(t):
    x = [ t(1) ]
    x.append(x)
    with pytest.raises(RuntimeError) as e:
        dr.enable_grad(x)
    e = e.value
    while e.__cause__ is not None:
        e = e.__cause__
    assert type(e) is RecursionError


@pytest.test_arrays('float,-float16,shape=(*),is_diff')
def test110_scatter_add_kahan(t):
    buf1 = dr.zeros(t, 2)
    buf2 = dr.zeros(t, 2)
    value = t(1, 2, 3)
    dr.enable_grad(buf1, buf2, value)
    ti = dr.uint32_array_t(t)

    dr.scatter_add_kahan(
        buf1,
        buf2,
        value,
        ti(0, 1, 1)
    )
    dr.set_grad(buf1, [10, 20])
    dr.set_grad(buf2, [100, 100]) # Ignored
    value_d = dr.backward_to(value)
    assert dr.all(buf1 + buf2 == [1, 5])
    assert dr.all(value_d == [10, 20, 20])


@pytest.test_arrays('is_diff,float32,shape=(*)')
def test111_custom_op_args_kwargs(t):
    x, y = t(1), t(2)
    dr.enable_grad(x, y)
    a, b = dr.custom(ArgsKwargs, x, key=y)
    x.grad = 100
    y.grad = 200
    dr.forward_to(a, b)
    assert a.grad == 100 and b.grad == 200

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test112_shrink_fwd(t):
    x = t(1, 2, 3, 4, 5)
    dr.enable_grad(x)
    y = dr.reshape(t, x, 3, shrink=True)
    x.grad = [10, 20, 30, 40, 50]
    dr.forward_to(y)
    assert dr.all(y.grad == [10, 20, 30])

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test113_shrink_bwd(t):
    x = t(1, 2, 3, 4, 5)
    dr.enable_grad(x)
    y = dr.reshape(t, x, 3, shrink=True)
    y.grad = [10, 20, 30]
    dr.backward_to(x)
    assert dr.all(x.grad == [10, 20, 30, 0, 0])


@pytest.test_arrays('is_diff,float32,shape=(*)')
@pytest.mark.parametrize("mode", ['symbolic', 'evaluated'])
def test114_block_sum_fwd(t, mode):
    x = t(1, 2, 3, 4, 5, 6)
    dr.enable_grad(x)
    y = dr.block_sum(x, 2, mode=mode)
    assert dr.all(y == [3, 7, 11])
    x.grad = [10, 20, 30, 40, 50, 60]
    dr.forward_to(y)
    assert dr.all(y.grad == [30, 70, 110])


@pytest.test_arrays('is_diff,float32,shape=(*)')
@pytest.mark.parametrize("mode", ['symbolic', 'evaluated'])
def test114_block_sum_rev(t, mode):
    x = t(1, 2, 3, 4, 5, 6)
    dr.enable_grad(x)
    y = dr.block_sum(x, 2, mode=mode)
    assert dr.all(y == [3, 7, 11])
    y.grad = [10, 20, 30]
    dr.backward_to(x)
    assert dr.all(x.grad == [10, 10, 20, 20, 30, 30])

@pytest.test_arrays('is_diff,float32,shape=(*)')
def test115_dot_ad(t):
    a = dr.arange(t, 102400)
    b = t(dr.arange(dr.uint32_array_t(t), 102400)%4)
    dr.enable_grad(a, b)
    dr.eval(a, b)

    c0 = dr.sum(a * b)
    dr.backward_from(c0)
    g0 = a.grad, b.grad
    dr.clear_grad(a)
    dr.clear_grad(b)

    c1 = dr.dot(a, b)
    dr.backward_from(c1)
    g1 = a.grad, b.grad

    assert dr.all((c0 == c1) & (g0[0] == g1[0]) & (g0[1] == g1[1]))

@pytest.test_arrays('is_diff,float32,shape=(*, *)')
def test116_arrayxf_backprop(t):
    a = t([[1,2,3], [4,5,6]])
    dr.enable_grad(a)
    c = 2 * a
    dr.backward(c)
    assert dr.all(a.grad == t([2,2,2], [2,2,2]), axis=None)


@pytest.mark.parametrize('op', [dr.ReduceOp.Max, dr.ReduceOp.Min])
@pytest.test_arrays('is_diff,llvm,float32,shape=(*)')
def test117_scatter_reduce_minmax_fwd(t, op):
    x = t(2, 4, 6)
    dr.enable_grad(x)
    x.grad = [.01, .1, 1]
    if op == dr.ReduceOp.Max:
        val = t(1, 3, 5, 4, 0, 0)
    else:
        val = t(1, 3, 5, 4, 10, 10)

    dr.enable_grad(val)
    index = dr.uint32_array_t(t)(0, 0, 1, 1, 2, 2)
    val.grad = [1, 10, 100, 1000, 10000, 10000]
    dr.scatter_reduce(op, x, val, index)

    xg = dr.forward_to(x)
    if op == dr.ReduceOp.Max:
        assert dr.all(x == [3, 5, 6])
        assert dr.all(xg == [10, 100, 1])
    else:
        assert dr.all(x == [1, 4, 6])
        assert dr.all(xg == [1, 1000, 1])


@pytest.mark.parametrize('op', [dr.ReduceOp.Max, dr.ReduceOp.Min])
@pytest.test_arrays('is_diff,llvm,float32,shape=(*)')
def test118_scatter_reduce_minmax_bwd(t, op):
    x = t(2, 4, 6)
    dr.enable_grad(x)
    if op == dr.ReduceOp.Max:
        val = t(1, 3, 5, 3, 0, 0)
    else:
        val = t(1, 3, 5, 3, 10, 10)

    dr.enable_grad(val)
    index = dr.uint32_array_t(t)(0, 0, 1, 1, 2, 2)
    xn = t(x)
    dr.scatter_reduce(op, xn, val, index)

    xn.grad=[1, 10, 100]
    xg, valg = dr.backward_to(x, val)

    if op == dr.ReduceOp.Max:
        assert dr.all(valg == [0, 1, 10, 0, 0, 0])
    else:
        assert dr.all(valg == [1, 0, 0, 10, 0, 0])
    print(xg)
    assert dr.all(xg == [0, 0, 100])
