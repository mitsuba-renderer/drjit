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
            self.x = m.Float(1.0)
            self.y = m.Float(2.0)
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
    assert dr.detached_t(type(a)) is type(b)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    a = m.Array3f()
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert dr.detached_t(type(a)) is type(b)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(b)
    assert not dr.grad_enabled(c)

    a = m.ArrayXf(1, 2, 3)
    dr.enable_grad(a)
    b = dr.detach(a, preserve_type=False)
    c = dr.detach(a, preserve_type=True)
    assert dr.detached_t(type(a)) is type(b)
    assert type(a) is type(c)
    assert type(a) is not type(b)

    a = struct_class(m)()
    dr.enable_grad(a)
    c = dr.detach(a)
    assert type(a) is type(c)
    assert dr.grad_enabled(a)
    assert not dr.grad_enabled(c)

    with pytest.raises(TypeError) as ei:
        dr.detach(a, preserve_type=False)
    assert "preserve_type=True is required" in str(ei.value)


def test03_set_grad(m):
    a = m.Float([1, 2, 3])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.set_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)
    dr.set_grad(a, m.Float([3.0]))
    assert dr.allclose(dr.grad(a), 3.0)
    with pytest.raises(RuntimeError) as ei:
        dr.set_grad(a, m.Float([1, 2, 3, 4]))
    assert "attempted to assign a gradient of size 4" in str(ei.value)
    dr.set_grad(a, [3.0, 2.0, 1.0])
    assert dr.allclose(dr.grad(a), [3.0, 2.0, 1.0])
    assert type(dr.grad(a)) is type(a)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    a = m.Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.set_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)
    dr.set_grad(a, m.Float([3.0]))
    assert dr.allclose(dr.grad(a), 3.0)
    with pytest.raises(RuntimeError) as ei:
        dr.set_grad(a, m.Float([1, 2, 3, 4]))
    assert "attempted to assign a gradient of size 4" in str(ei.value)
    dr.set_grad(a, m.Array3f([3, 2, 1]))
    assert dr.allclose(dr.grad(a.y), [2, 2, 2])
    dr.set_grad(a, m.Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(dr.grad(a), [[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    assert type(dr.grad(a)) is type(a)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    a = m.ArrayXf(1, 2, 3)
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.set_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)
    dr.set_grad(a, m.ArrayXf([3, 2, 1]))
    assert dr.allclose(dr.grad(a[1]), [2, 2, 2])
    assert type(dr.grad(a)) is type(a)
    assert type(dr.grad(a, False)) is dr.detached_t(type(a))

    args = [m.Float(1.0), m.Float(2.0), m.Float(3.0)]
    dr.enable_grad(args)
    dr.set_grad(args, 1.0)
    assert dr.allclose(dr.grad(args), 1.0)
    dr.set_grad(args, [3.0, 2.0, 1.0])
    assert dr.allclose(dr.grad(args), [3.0, 2.0, 1.0])
    with pytest.raises(RuntimeError) as ei:
        dr.set_grad(args, [3.0, 2.0])
    assert "argument sizes are not matching" in str(ei.value)

    args = {'a': m.Float(1.0), 'b': m.Float(2.0)}
    dr.enable_grad(args)
    dr.set_grad(args, 1.0)
    assert dr.allclose(dr.grad(args)['a'], 1.0)
    assert dr.allclose(dr.grad(args)['b'], 1.0)
    dr.set_grad(args, {'a': 2.0, 'b': 3.0})
    assert dr.allclose(dr.grad(args)['a'], 2.0)
    assert dr.allclose(dr.grad(args)['b'], 3.0)

    struct_class_type = struct_class(m)
    a = struct_class_type()
    dr.enable_grad(a)
    dr.set_grad(a, 1.0)
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 1.0)
    dr.set_grad(a, struct_class_type())
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 2.0)

    with pytest.raises(TypeError) as ei:
        dr.grad(a, preserve_type=False)
    assert "preserve_type=True is required" in str(ei.value)


def test04_broadcasting_set_grad(m):
    theta = m.Float(1.0)
    dr.enable_grad(theta)
    x = 4.0 * theta
    y = m.Array3f(x)
    dr.backward(y)
    assert dr.grad(theta) == 12.0


def test05_accum_grad(m):
    a = m.Float([1, 2, 3])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.accum_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)
    dr.accum_grad(a, m.Float([3.0]))
    assert dr.allclose(dr.grad(a), 5.0)
    with pytest.raises(RuntimeError) as ei:
        dr.accum_grad(a, m.Float([1, 2, 3, 4]))
    assert "attempted to accumulate a gradient of size 4" in str(ei.value)
    dr.accum_grad(a, [3.0, 2.0, 1.0])
    assert dr.allclose(dr.grad(a), [8.0, 7.0, 6.0])

    a = m.Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5])
    dr.enable_grad(a)
    assert dr.allclose(dr.grad(a), 0.0)
    dr.accum_grad(a, 2.0)
    assert dr.allclose(dr.grad(a), 2.0)
    dr.accum_grad(a, m.Float([3.0]))
    assert dr.allclose(dr.grad(a), 5.0)
    with pytest.raises(RuntimeError) as ei:
        dr.accum_grad(a, m.Float([1, 2, 3, 4]))
    assert "attempted to accumulate a gradient of size 4" in str(ei.value)
    dr.accum_grad(a, m.Array3f([3, 2, 1]))
    assert dr.allclose(dr.grad(a.y), [7, 7, 7])
    dr.accum_grad(a, m.Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(dr.grad(a), [[9, 10, 11], [9, 10, 11], [9, 10, 11]])

    args = [m.Float(1.0), m.Float(2.0), m.Float(3.0)]
    dr.enable_grad(args)
    dr.accum_grad(args, 1.0)
    assert dr.allclose(dr.grad(args), 1.0)
    dr.accum_grad(args, [3.0, 2.0, 1.0])
    assert dr.allclose(dr.grad(args), [4.0, 3.0, 2.0])
    with pytest.raises(RuntimeError) as ei:
        dr.accum_grad(args, [3.0, 2.0])
    assert "argument sizes are not matching" in str(ei.value)

    args = {'a': m.Float(1.0), 'b': m.Float(2.0)}
    dr.enable_grad(args)
    dr.accum_grad(args, 1.0)
    assert dr.allclose(dr.grad(args)['a'], 1.0)
    assert dr.allclose(dr.grad(args)['b'], 1.0)
    dr.accum_grad(args, {'a': 2.0, 'b': 3.0})
    assert dr.allclose(dr.grad(args)['a'], 3.0)
    assert dr.allclose(dr.grad(args)['b'], 4.0)

    struct_class_type = struct_class(m)
    a = struct_class_type()
    dr.enable_grad(a)
    dr.accum_grad(a, 1.0)
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 1.0)
    dr.accum_grad(a, struct_class_type())
    assert dr.allclose(dr.grad(a).x, 2.0)
    assert dr.allclose(dr.grad(a).y, 3.0)


def test06_replace_grad(m):
    with pytest.raises(TypeError) as ei:
        dr.replace_grad(1.0, m.UInt(1))
    assert "unsupported input types" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.replace_grad(1.0, dr.detached_t(m.Float)(2.0))
    assert "unsupported input types" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.replace_grad(1.0, 2.0)
    assert "unsupported input types" in str(ei.value)

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
    y = m.Float(1)
    dr.enable_grad(x, y)
    z = dr.replace_grad(x, y)
    assert z.x.index_ad == y.index_ad
    assert z.y.index_ad == y.index_ad
    assert z.z.index_ad == y.index_ad

    a = m.Float(1.0)
    dr.enable_grad(a)
    b = dr.replace_grad(4.0, a)
    assert type(b) is type(a)
    assert b.index_ad == a.index_ad
    assert dr.allclose(b, 4.0)

    a = m.ArrayXf(1, 2, 3)
    y = m.Float(1)
    dr.enable_grad(x, y)
    z = dr.replace_grad(x, y)
    assert z[0].index_ad == y.index_ad
    assert z[1].index_ad == y.index_ad
    assert z[2].index_ad == y.index_ad


def test07_set_label(m):
    a = m.Float(1.0)
    b = [m.Float(1.0), m.Float(2.0)]
    c = m.Array3f(1.0, 2.0, 3.0)
    d = struct_class(m)()
    dr.enable_grad(a, b, c, d)


def test08_forward_to(m):
    a = m.Float(1.0)
    dr.enable_grad(a)
    b = a * a * 2
    c = a * 2
    dr.set_grad(a, 1.0)
    d = m.Float(4.0) # some detached variable
    grad_b, grad_c, grad_d = dr.forward_to(b, c, d)
    assert dr.allclose(dr.grad(a), 0.0)
    assert dr.allclose(grad_b, 4.0)
    assert dr.allclose(grad_c, 2.0)
    assert dr.allclose(grad_d, 0.0)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, dr.ADFlag.Default)
    assert "AD flags should be passed via the" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, flags=d)
    assert "incompatible function arguments" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, flags=dr.ADFlag.Default, test='test')
    assert "unexpected keyword argument" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, test='test')
    assert "unexpected keyword argument" in str(ei.value)

    # Error because the input isn't attached to the AD graph
    with pytest.raises(TypeError) as ei:
        dr.forward_to(m.Float(2.0))
    assert "the argument does not depend on the input" in str(ei.value)

    # Error because the input isn't a diff array
    with pytest.raises(TypeError) as ei:
        dr.forward_to(dr.detached_t(m.Float)(2.0))
    assert "expected a differentiable array type" in str(ei.value)

    # Error because the input isn't a drjit array
    with pytest.raises(TypeError) as ei:
        dr.forward_to([2.0])
    assert "expected a differentiable array type" in str(ei.value)

    # Trying to call with a different flag
    dr.set_grad(a, 1.0)
    b = a * a * 2
    grad_b = dr.forward_to(b, flags=dr.ADFlag.ClearInterior)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(grad_b, 4.0)


def test09_forward_to(m):
    a, b, c = m.Float(1), m.Float(2), m.Float(3)
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
            dr.set_grad(v, m.Float())


def test10_forward_from(m):
    with pytest.raises(TypeError) as ei:
        dr.forward_from(1.0)
    assert "expected a differentiable array type" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_from(dr.detached_t(m.Float)(1.0))
    assert "expected a differentiable array type" in str(ei.value)

    a = m.Float(1.0)

    with pytest.raises(TypeError) as ei:
        dr.forward_from(a)
    assert "the argument does not depend on the input" in str(ei.value)

    dr.enable_grad(a)
    b = a * a * 2
    dr.forward_from(a)
    assert dr.allclose(dr.grad(a), 0.0)
    assert dr.allclose(dr.grad(b), 4.0)

    b = a * a * 2
    dr.forward_from(a, dr.ADFlag.ClearInterior)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 4.0)

    # Interior gradients are cleared, forwarding again will accumulate gradients
    dr.forward_from(a, dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(b), 8.0)

    # Edges are cleared, forwarding again will do nothing
    dr.forward_from(a, dr.ADFlag.ClearEdges)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(dr.grad(b), 8.0)


def test11_backward_to(m):
    with pytest.raises(TypeError) as ei:
        dr.backward_to(1.0)
    assert "expected a differentiable array type" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.backward_from(dr.detached_t(m.Float)(1.0))
    assert "expected a differentiable array type" in str(ei.value)

    a = m.Float(1.0)

    with pytest.raises(TypeError) as ei:
        dr.backward_to(a)
    assert "the argument does not depend on the input" in str(ei.value)

    b = m.Float(3.0)
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


def test12_backward_to(m):
    a, b, c = m.Float(1), m.Float(2), m.Float(3)
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
            dr.set_grad(v, m.Float())


def test13_backward_from(m):
    with pytest.raises(TypeError) as ei:
        dr.backward_from(1.0)
    assert "expected a differentiable array type" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.backward_from(dr.detached_t(m.Float)(1.0))
    assert "expected a differentiable array type" in str(ei.value)

    a = m.Float(1.0)

    with pytest.raises(TypeError) as ei:
        dr.backward_from(a)
    assert "the argument does not depend on the input" in str(ei.value)

    dr.enable_grad(a)
    b = a * a * 2
    dr.backward_from(b)
    assert dr.allclose(dr.grad(a), 4.0)
    assert dr.allclose(dr.grad(b), 0.0)

    a = m.Float(1.0)
    dr.enable_grad(a)
    c = m.Array3f(a)
    dr.backward_from(c)
    assert dr.allclose(dr.grad(a), 3.0)


def test14_nan_propagation(m):
    for i in range(2):
        x = dr.arange(m.Float, 10)
        dr.enable_grad(x)
        f0 = m.Float(0)
        y = dr.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        dr.backward(y)
        g = dr.grad(x)
        if i == 0:
            assert dr.allclose(g, 1)
        else:
            assert dr.all(dr.isnan(g))

    for i in range(2):
        x = dr.arange(m.Float, 10)
        dr.enable_grad(x)
        f0 = m.Float(0)
        y = dr.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        dr.forward(x)
        g = dr.grad(y)
        if i == 0:
            assert dr.allclose(g, 1)
        else:
            assert dr.all(dr.isnan(g))


@pytest.mark.parametrize("f1", [0, int(dr.ADFlag.ClearEdges)])
@pytest.mark.parametrize("f2", [0, int(dr.ADFlag.ClearInterior)])
@pytest.mark.parametrize("f3", [0, int(dr.ADFlag.ClearInput)])
def test14_ad_flags(m, f1, f2, f3):
    v0 = m.Float(2)
    dr.enable_grad(v0)
    v1 = v0 * 0.5
    v2 = v0 + v1

    for i in range(2):
        dr.accum_grad(v0, 1 if i == 0 else 100)
        dr.enqueue(dr.ADMode.Forward, v0)
        dr.traverse(m.Float, dr.ADMode.Forward, flags=(f1 | f2 | f3))

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


# ------------------------------------------------------------------------------

def test15_add_bwd(m):
    a, b = m.Float(1), m.Float(2)
    dr.enable_grad(a, b)
    c = 2 * a + b
    dr.backward(c)
    assert dr.grad(a) == 2
    assert dr.grad(b) == 1


def test16_add_fwd(m):
    if True:
        a, b = m.Float(1), m.Float(2)
        dr.enable_grad(a, b)
        c = 2 * a + b
        dr.forward(a, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 2
        dr.set_grad(c, 101)
        dr.forward(b)
        assert dr.grad(c) == 102

    if True:
        a, b = m.Float(1), m.Float(2)
        dr.enable_grad(a, b)
        c = 2 * a + b
        dr.set_grad(a, 1.0)
        dr.enqueue(dr.ADMode.Forward, a)
        dr.traverse(m.Float, dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 2
        assert dr.grad(a) == 0
        dr.set_grad(a, 1.0)
        dr.enqueue(dr.ADMode.Forward, a)
        dr.traverse(m.Float, dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
        assert dr.grad(c) == 4


def test17_branch_fwd(m):
    a = m.Float(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.forward(a)
    assert dr.grad(d) == 2


def test18_branch_ref(m):
    a = m.Float(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.backward(d)
    assert dr.grad(a) == 2


def test19_sub_mul(m):
    a, b, c = m.Float(2), m.Float(3), m.Float(4)
    dr.enable_grad(a, b, c)
    d = a * b - c
    dr.backward(d)
    assert dr.grad(a) == dr.detach(b)
    assert dr.grad(b) == dr.detach(a)
    assert dr.grad(c) == -1


def test20_div(m):
    a, b = m.Float(2), m.Float(3)
    dr.enable_grad(a, b)
    d = a / b
    dr.backward(d)
    assert dr.allclose(dr.grad(a),  1.0 / 3.0)
    assert dr.allclose(dr.grad(b), -2.0 / 9.0)


def test21_sum_0_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y, 95.0/27.0)
    assert dr.allclose(dr.grad(x), 2 * dr.detach(x))


def test22_sum_0_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(dr.detach(y), 95.0/27.0)
    assert len(dr.grad(y)) == 1 and dr.allclose(dr.grad(y), 10)


def test23_sum_1_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 11)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.backward(y)
    assert dr.allclose(dr.grad(x), 11)


def test24_sum_1_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.forward(x)
    assert dr.allclose(dr.grad(y), 100)


def test25_sum_2_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 11)
    dr.enable_grad(x)
    z = dr.sum(dr.sum(x*x)*x*x)
    dr.backward(z)
    assert dr.allclose(dr.grad(x),
                       [0., 1.54, 3.08, 4.62, 6.16, 7.7,
                        9.24, 10.78, 12.32, 13.86, 15.4])


def test26_sum_2_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x*x)*dr.sum(x*x))
    dr.forward(x)
    assert dr.allclose(dr.grad(y), 1900.0 / 27.0)


def test27_prod(m):
    x = m.Float(1, 2, 5, 8)
    dr.enable_grad(x)
    y = dr.prod(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 80)
    assert dr.allclose(dr.grad(x), [80, 40, 16, 10])


def test28_max_bwd(m):
    x = m.Float(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(x), [0, 0, 1, 0, 1])


def test29_max_fwd(m):
    x = m.Float(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(y), [2])  # Approximation


def test30_sqrt(m):
    x = m.Float(1, 4, 16)
    dr.enable_grad(x)
    y = dr.sqrt(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [1, 2, 4])
    assert dr.allclose(dr.grad(x), [.5, .25, .125])


def test31_rsqrt(m):
    x = m.Float(1, .25, 0.0625)
    dr.enable_grad(x)
    y = dr.rsqrt(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [1, 2, 4])
    assert dr.allclose(dr.grad(x), [-.5, -4, -32])


def test32_abs(m):
    x = m.Float(-2, 2)
    dr.enable_grad(x)
    y = dr.abs(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [2, 2])
    assert dr.allclose(dr.grad(x), [-1, 1])


def test33_sin(m):
    x = dr.linspace(m.Float, 0, 10, 10)
    dr.enable_grad(x)
    y = dr.sin(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), dr.sin(dr.detach(x)))
    assert dr.allclose(dr.grad(x), dr.cos(dr.detach(x)))


def test34_cos(m):
    x = dr.linspace(m.Float, 0.01, 10, 10)
    dr.enable_grad(x)
    y = dr.cos(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), dr.cos(dr.detach(x)))
    assert dr.allclose(dr.grad(x), -dr.sin(dr.detach(x)))


def test35_gather(m):
    x = dr.linspace(m.Float, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.gather(m.Float, x*x, m.UInt(1, 1, 2, 3))
    z = dr.sum(y)
    dr.backward(z)
    ref = [0, -1.55556*2, -1.11111, -0.666667, 0, 0, 0, 0, 0, 0]
    assert dr.allclose(dr.grad(x), ref)


def test36_gather_fwd(m):
    x = dr.linspace(m.Float, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.gather(m.Float, x*x, m.UInt(1, 1, 2, 3))
    dr.forward(x)
    ref = [-1.55556, -1.55556, -1.11111, -0.666667]
    assert dr.allclose(dr.grad(y), ref)


@pytest.mark.skip("TODO")
def test37_scatter_reduce_bwd(m):
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 4) + 3

        x = dr.linspace(m.Float, 0, 1, 5)
        y = dr.linspace(m.Float, 1, 2, 4)
        buf = dr.zeros(m.Float, 10)

        if i % 2 == 0:
            dr.enable_grad(buf)
        if i // 2 == 0:
            dr.enable_grad(x, y)

        dr.set_label(x, "x")
        dr.set_label(y, "y")
        dr.set_label(buf, "buf")

        buf2 = m.Float(buf)
        dr.scatter_reduce(dr.ReduceOp.Add, buf2, x, idx1)
        dr.scatter_reduce(dr.ReduceOp.Add, buf2, y, idx2)

        ref_buf = m.Float(0.0000, 0.2500, 0.5000, 1.7500, 2.3333,
                          1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        assert dr.allclose(ref_buf, buf2, atol=1e-4)

        s = dr.dot(buf2, buf2)

        dr.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 3.5000, 4.6667)
        ref_y = m.Float(3.5000, 4.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert dr.allclose(dr.grad(y), dr.detach(ref_y), atol=1e-4)
            assert dr.allclose(dr.grad(x), dr.detach(ref_x), atol=1e-4)
        else:
            assert dr.grad(x) == 0
            assert dr.grad(y) == 0

        if i % 2 == 0:
            assert dr.allclose(dr.grad(buf), dr.detach(ref_buf) * 2, atol=1e-4)
        else:
            assert dr.grad(buf) == 0


@pytest.mark.skip("TODO")
def test38_scatter_reduce_fwd(m):
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 4) + 3

        x = dr.linspace(m.Float, 0, 1, 5)
        y = dr.linspace(m.Float, 1, 2, 4)
        buf = dr.zeros(m.Float, 10)

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

        buf2 = m.Float(buf)
        dr.scatter_reduce(dr.ReduceOp.Add, buf2, x, idx1)
        dr.scatter_reduce(dr.ReduceOp.Add, buf2, y, idx2)

        s = dr.dot(buf2, buf2)

        if i % 2 == 0:
            dr.enqueue(dr.ADMode.Forward, buf)
        if i // 2 == 0:
            dr.enqueue(dr.ADMode.Forward, x, y)

        dr.traverse(m.Float, dr.ADMode.Forward)

        # Verified against Mathematica
        assert dr.allclose(dr.detach(s), 15.5972)
        assert dr.allclose(dr.grad(s), (25.1667 if i // 2 == 0 else 0)
                           + (17 if i % 2 == 0 else 0))


def test39_scatter_bwd(m):
    for i in range(3):
        idx1 = dr.arange(m.UInt, 5)
        idx2 = dr.arange(m.UInt, 4) + 3

        x = dr.linspace(m.Float, 0, 1, 5)
        y = dr.linspace(m.Float, 1, 2, 4)
        buf = dr.zeros(m.Float, 10)

        if i % 2 == 0:
            dr.enable_grad(buf)
        if i // 2 == 0:
            dr.enable_grad(x, y)

        dr.set_label(x, "x")
        dr.set_label(y, "y")
        dr.set_label(buf, "buf")

        buf2 = m.Float(buf)
        dr.scatter(buf2, x, idx1)
        dr.eval(buf2)
        dr.scatter(buf2, y, idx2)

        ref_buf = m.Float(0.0000, 0.2500, 0.5000, 1.0000, 1.3333,
                          1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        assert dr.allclose(ref_buf, buf2, atol=1e-4)

        s = dr.dot(buf2, buf2)

        dr.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 0.0000, 0.0000)
        ref_y = m.Float(2.0000, 2.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert dr.allclose(dr.grad(y), dr.detach(ref_y), atol=1e-4)
            assert dr.allclose(dr.grad(x), dr.detach(ref_x), atol=1e-4)
        else:
            assert dr.grad(x) == 0
            assert dr.grad(y) == 0

        if i % 2 == 0:
            assert dr.allclose(dr.grad(buf), 0, atol=1e-4)
        else:
            assert dr.grad(buf) == 0


def test40_scatter_fwd(m):
    x = m.Float(4.0)
    dr.enable_grad(x)

    values = x * x * dr.linspace(m.Float, 1, 4, 4)
    idx = 2 * dr.arange(m.UInt, 4)

    buf = dr.zeros(m.Float, 10)
    dr.scatter(buf, values, idx)

    assert dr.grad_enabled(buf)

    ref = [16.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert dr.allclose(buf, ref)

    dr.forward(x, flags=dr.ADFlag.ClearVertices)
    grad = dr.grad(buf)

    ref_grad = [8.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert dr.allclose(grad, ref_grad)

    # Overwrite first value with non-diff value, resulting gradient entry should be 0
    y = m.Float(3)
    idx = m.UInt(0)
    dr.scatter(buf, y, idx)

    ref = [3.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert dr.allclose(buf, ref)

    dr.forward(x)
    grad = dr.grad(buf)

    ref_grad = [0.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert dr.allclose(grad, ref_grad)


def test41_scatter_fwd_permute(m):
    x = m.Float(4.0)
    dr.enable_grad(x)

    values_0 = x * dr.linspace(m.Float, 1, 9, 5)
    values_1 = x * dr.linspace(m.Float, 11, 19, 5)

    buf = dr.zeros(m.Float, 10)

    idx_0 = dr.arange(m.UInt, 5)
    idx_1 = dr.arange(m.UInt, 5) + 5

    dr.scatter(buf, values_0, idx_0)
    dr.scatter(buf, values_1, idx_1)

    ref = [4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0]
    assert dr.allclose(buf, ref)

    dr.forward(x)
    grad = dr.grad(buf)

    ref_grad = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
    assert dr.allclose(grad, ref_grad)


def test42_exp(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.exp(x * x)
    dr.backward(y)
    exp_x = dr.exp(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, exp_x)
    assert dr.allclose(dr.grad(x), 2 * dr.detach(x) * exp_x)


def test43_log(m):
    x = dr.linspace(m.Float, 0.01, 1, 10)
    dr.enable_grad(x)
    y = dr.log(x * x)
    dr.backward(y)
    log_x = dr.log(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, log_x)
    assert dr.allclose(dr.grad(x), 2 / dr.detach(x))


def test44_pow(m):
    x = dr.linspace(m.Float, 1, 10, 10)
    y = dr.full(m.Float, 2.0, 10)
    dr.enable_grad(x, y)
    z = dr.power(x, y)
    dr.backward(z)
    assert dr.allclose(dr.grad(x), dr.detach(x)*2)
    assert dr.allclose(dr.grad(y),
                       m.Float(0., 2.77259, 9.88751, 22.1807, 40.2359,
                               64.5033, 95.3496, 133.084, 177.975, 230.259))


def test45_tan(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.tan(x * x)
    dr.backward(y)
    tan_x = dr.tan(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, tan_x)
    assert dr.allclose(dr.grad(x),
                       m.Float(0., 0.222256, 0.44553, 0.674965, 0.924494,
                               1.22406, 1.63572, 2.29919, 3.58948, 6.85104))


def test46_asin(m):
    x = dr.linspace(m.Float, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.asin(x * x)
    dr.backward(y)
    asin_x = dr.asin(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, asin_x)
    assert dr.allclose(dr.grad(x),
                       m.Float(-2.08232, -1.3497, -0.906755, -0.534687,
                               -0.177783, 0.177783, 0.534687, 0.906755,
                               1.3497, 2.08232))


def test47_acos(m):
    x = dr.linspace(m.Float, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.acos(x * x)
    dr.backward(y)
    acos_x = dr.acos(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, acos_x)
    assert dr.allclose(dr.grad(x),
                       m.Float(2.08232, 1.3497, 0.906755, 0.534687, 0.177783,
                               -0.177783, -0.534687, -0.906755, -1.3497,
                               -2.08232))


def test48_atan(m):
    x = dr.linspace(m.Float, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.atan(x * x)
    dr.backward(y)
    atan_x = dr.atan(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, atan_x)
    assert dr.allclose(dr.grad(x),
                       m.Float(-1.13507, -1.08223, -0.855508, -0.53065,
                               -0.177767, 0.177767, 0.53065, 0.855508, 1.08223,
                               1.13507))


def test49_atan2(m):
    x = dr.linspace(m.Float, -.8, .8, 10)
    y = m.Float(dr.arange(m.Int, 10) & 1) * 1 - .5
    dr.enable_grad(x, y)
    z = dr.atan2(y, x)
    dr.backward(z)
    assert dr.allclose(z, m.Float(-2.58299, 2.46468, -2.29744, 2.06075,
                                  -1.74674, 1.39486, -1.08084, 0.844154,
                                  -0.676915, 0.558599))
    assert dr.allclose(dr.grad(x),
                       m.Float(0.561798, -0.784732, 1.11724, -1.55709, 1.93873,
                               -1.93873, 1.55709, -1.11724, 0.784732,
                               -0.561798))
    assert dr.allclose(dr.grad(y),
                       m.Float(-0.898876, -0.976555, -0.993103, -0.83045,
                               -0.344663, 0.344663, 0.83045, 0.993103,
                               0.976555, 0.898876))


def test50_cbrt(m):
    x = dr.linspace(m.Float, -.8, .8, 10)
    dr.enable_grad(x)
    y = dr.cbrt(x)
    dr.backward(y)
    assert dr.allclose(y, m.Float(-0.928318, -0.853719, -0.763143, -0.64366,
                                  -0.446289, 0.446289, 0.64366, 0.763143,
                                  0.853719, 0.928318))
    assert dr.allclose(dr.grad(x),
                       m.Float(0.386799, 0.45735, 0.572357, 0.804574, 1.67358,
                               1.67358, 0.804574, 0.572357, 0.45735, 0.386799))


def test51_sinh(m):
    x = dr.linspace(m.Float, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.sinh(x)
    dr.backward(y)
    assert dr.allclose(
        y, m.Float(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                   0.11134, 0.339541, 0.584578, 0.858602, 1.1752))
    assert dr.allclose(
        dr.grad(x),
        m.Float(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))


def test52_cosh(m):
    x = dr.linspace(m.Float, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.cosh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        m.Float(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))
    assert dr.allclose(
        dr.grad(x),
        m.Float(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                0.11134, 0.339541, 0.584578, 0.858602, 1.1752))


def test53_tanh(m):
    x = dr.linspace(m.Float, -1, 1, 10)
    dr.enable_grad(x)
    y = dr.tanh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        m.Float(-0.761594, -0.651429, -0.504672, -0.321513, -0.110656,
                0.110656, 0.321513, 0.504672, 0.651429, 0.761594))
    assert dr.allclose(
        dr.grad(x),
        m.Float(0.419974, 0.57564, 0.745306, 0.89663, 0.987755, 0.987755,
                0.89663, 0.745306, 0.57564, 0.419974)
    )


def test54_asinh(m):
    x = dr.linspace(m.Float, -.9, .9, 10)
    dr.enable_grad(x)
    y = dr.asinh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        m.Float(-0.808867, -0.652667, -0.481212, -0.295673, -0.0998341,
                0.0998341, 0.295673, 0.481212, 0.652667, 0.808867))
    assert dr.allclose(
        dr.grad(x),
        m.Float(0.743294, 0.819232, 0.894427, 0.957826, 0.995037,
                0.995037, 0.957826, 0.894427, 0.819232, 0.743294)
    )


def test55_acosh(m):
    x = dr.linspace(m.Float, 1.01, 2, 10)
    dr.enable_grad(x)
    y = dr.acosh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        m.Float(0.141304, 0.485127, 0.665864, 0.802882, 0.916291,
                1.01426, 1.10111, 1.17944, 1.25098, 1.31696))
    assert dr.allclose(
        dr.grad(x),
        m.Float(7.05346, 1.98263, 1.39632, 1.12112, 0.952381,
                0.835191, 0.747665, 0.679095, 0.623528, 0.57735)
    )


def test56_atanh(m):
    x = dr.linspace(m.Float, -.99, .99, 10)
    dr.enable_grad(x)
    y = dr.atanh(x)
    dr.backward(y)
    assert dr.allclose(
        y,
        m.Float(-2.64665, -1.02033, -0.618381, -0.342828, -0.110447, 0.110447,
                0.342828, 0.618381, 1.02033, 2.64665))
    assert dr.allclose(
        dr.grad(x),
        m.Float(50.2513, 2.4564, 1.43369, 1.12221, 1.01225, 1.01225, 1.12221,
                1.43369, 2.4564, 50.2513)
    )


def test57_safe_functions(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    y = dr.linspace(m.Float, -1, 1, 10)
    z = dr.linspace(m.Float, -1, 1, 10)
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


# ------------------------------------------------------------------------------


def test58_custom_error(m):
    v = m.Float(1.0)
    dr.enable_grad(v)

    class CustomOp01(dr.CustomOp):
        def eval(self, value):
            self.add_input(v)
            return value * v

        def forward(self):
            pass

        def backward(self):
            pass

    a = m.Float(1.0)
    dr.enable_grad(a)

    with pytest.raises(RuntimeError) as ei:
        b = dr.custom(CustomOp01, a)
    assert "should not be attached to the AD graph" in str(ei.value)


class Normalize(dr.CustomOp):
    def eval(self, value):
        self.value = value
        self.inv_norm = dr.rcp(dr.norm(value))
        return value * self.inv_norm

    def forward(self):
        grad_in = self.grad_in('value')
        grad_out = grad_in * self.inv_norm
        grad_out -= self.value * (dr.dot(self.value, grad_out) * dr.sqr(self.inv_norm))
        self.set_grad_out(grad_out)

    def backward(self):
        grad_out = self.grad_out()
        grad_in = grad_out * self.inv_norm
        grad_in -= self.value * (dr.dot(self.value, grad_in) * dr.sqr(self.inv_norm))
        self.set_grad_in('value', grad_in)

    def name(self):
        return "normalize"


def test59_custom_backward(m):
    d = m.Array3f(1, 2, 3)
    dr.enable_grad(d)
    d2 = dr.custom(Normalize, d)
    dr.set_grad(d2, m.Array3f(5, 6, 7))
    dr.enqueue(dr.ADMode.Backward, d2)
    dr.traverse(m.Float, dr.ADMode.Backward)
    assert dr.allclose(dr.grad(d), m.Array3f(0.610883, 0.152721, -0.305441))


def test60_custom_forward(m):
    d = m.Array3f(1, 2, 3)
    dr.enable_grad(d)
    d2 = dr.custom(Normalize, d)
    dr.set_grad(d, m.Array3f(5, 6, 7))
    dr.enqueue(dr.ADMode.Forward, d)
    dr.traverse(m.Float, dr.ADMode.Forward, flags=dr.ADFlag.ClearVertices)
    assert dr.allclose(dr.grad(d), 0)
    dr.set_grad(d, m.Array3f(5, 6, 7))
    assert dr.allclose(dr.grad(d2), m.Array3f(0.610883, 0.152721, -0.305441))
    dr.enqueue(dr.ADMode.Forward, d)
    dr.traverse(m.Float, dr.ADMode.Forward)
    assert dr.allclose(dr.grad(d2), m.Array3f(0.610883, 0.152721, -0.305441)*2)


def test61_custom_forward_external_dependency(m):
    class BuggyOp(dr.CustomOp):
        def eval(self, value):
            self.add_input(param)
            self.value = value
            return value * dr.detach(param)

        def forward(self):
            grad_in = self.grad_in('value')

            value = param * 4.0

            dr.enqueue(dr.ADMode.Forward, param)
            dr.traverse(m.Float, dr.ADMode.Forward, dr.ADFlag.ClearEdges)

            self.set_grad_out(dr.grad(value))

        def backward(self):
            pass

        def name(self):
            return "buggy-op"

    theta = m.Float(2)
    dr.enable_grad(theta)

    param = theta * 3.0

    dr.set_grad(theta, 1.0)
    dr.enqueue(dr.ADMode.Forward, theta)
    dr.traverse(m.Float, dr.ADMode.Forward, dr.ADFlag.ClearEdges)

    v3 = dr.custom(BuggyOp, 123)

    dr.enqueue(dr.ADMode.Forward, param)
    dr.traverse(m.Float, dr.ADMode.Forward, dr.ADFlag.ClearEdges)

    assert dr.allclose(dr.grad(v3), 12)


def test62_implicit_dep_customop(m):
    v0 = m.Float(2)
    dr.enable_grad(v0)
    v1 = v0 * 3

    class ImplicitDep(dr.CustomOp):
        def eval(self, value):
            self.add_input(v1)
            self.value = value
            return value * dr.detach(v1)

        def forward(self):
            grad_in = self.grad_in('value')
            assert grad_in is None
            self.set_grad_out(self.value * dr.grad(v1))

        def backward(self):
            grad_out = self.grad_out()
            self.set_grad_in('value', grad_out * dr.detach(v1))
            dr.accum_grad(v1, grad_out * self.value)

        def name(self):
            return "implicit-dep"

    v3 = dr.custom(ImplicitDep, 123)
    assert v3[0] == 123*6

    dr.forward(v0, flags=dr.ADFlag.ClearVertices)
    assert dr.grad(v3) == 123*3

    v3 = dr.custom(ImplicitDep, 123)
    assert v3[0] == 123*6
    dr.backward(v3)
    assert dr.grad(v0) == 123*3


# ------------------------------------------------------------------------------


def test63_suspend_resume(m):
    a = m.Float(1)
    b = m.Float(1)
    c = m.Float(1)
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
        e = m.Float(1)
        dr.enable_grad(e)
        assert not dr.grad_enabled(a, b, c, d, e)

        # Replicating suspended variables creates detached copies
        f = m.Float(a)
        with dr.resume_grad():
            assert dr.grad_enabled(a) and \
                   not dr.grad_enabled(f)


def test64_suspend_resume_selective(m):
    a = m.Float(1)
    b = m.Float(1)
    c = m.Float(1)
    d = m.Float(1)
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


def test65_suspend_resume_custom_fwd(m):
    v_implicit, v_input = m.Float(1), m.Float(1)
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
            dr.traverse(m.Float, dr.ADMode.Forward)
            assert dr.grad(output) == 2 - (i & 1) - ((i & 2) >> 1)


def test66_suspend_resume_custom_bwd(m):
    v_implicit, v_input = m.Float(1), m.Float(1)
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
            dr.traverse(m.Float, dr.ADMode.Backward)
            assert dr.grad(v_implicit) == ((i & 1) == 0)
            assert dr.grad(v_input) == ((i & 2) == 0)


def test67_isolate(m):
    a = m.Float(1)
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


def test68_isolate_fwd(m):
    # Tests the impact of repeatedly propagating
    # when an isolation boundary is present

    if True:
        a = m.Float(0)
        dr.enable_grad(a)
        dr.set_grad(a, 2)

        b = a * 2
        db = dr.forward_to(b)
        assert db == 4

        c = a * 3
        dc = dr.forward_to(c)
        assert dc == 0

    if True:
        a = m.Float(0)
        dr.enable_grad(a)
        dr.set_grad(a, 2)

        with dr.isolate_grad():
            b = a * 2
            db = dr.forward_to(b)
            assert db == 4

            c = a * 3
            dc = dr.forward_to(c)
            assert dc == 6



# ------------------------------------------------------------------------------

@pytest.mark.skip("TODO")
def test69_diff_loop(m, do_record):
    def mcint(a, b, f, sample_count=100000):
        rng = m.PCG32()
        i = m.UInt(0)
        result = m.Float(0)
        l = m.Loop("test45", lambda: (i, rng, result))
        while l(i < sample_count):
            result += f(dr.lerp(a, b, rng.next_float32()))
            i += 1
        return result * (b - a) / sample_count

    class EllipticK(dr.CustomOp):
        # --- Internally used utility methods ---

        # Integrand of the 'K' function
        def K(self, x, m_):
            return dr.rsqrt(1 - m_ * dr.sqr(dr.sin(x)))

        # Derivative of the above with respect to 'm'
        def dK(self, x, m_):
            m_ = m.Float(m_) # Convert 'm' to differentiable type
            dr.enable_grad(m_)
            y = self.K(x, m_)
            dr.forward(m_)
            return dr.grad(y)

        # Monte Carlo integral of dK, used in forward/backward pass
        def eval_grad(self):
            return mcint(a=0, b=dr.Pi/2, f=lambda x: self.dK(x, self.m_))

        # --- CustomOp interface ---

        def eval(self, m_):
            self.m_ = m_ # Stash 'm' for later
            return mcint(a=0, b=dr.Pi/2, f=lambda x: self.K(x, self.m_))

        def forward(self):
            self.set_grad_out(self.grad_in('m_') * self.eval_grad())

    def elliptic_k(m_):
        return dr.custom(EllipticK, m_)

    x = m.Float(0.5)
    dr.enable_grad(x)
    y = elliptic_k(x)
    dr.forward(x, flags=dr.ADFlag.ClearVertices)
    del x
    assert dr.allclose(y, 1.85407, rtol=5e-4)
    assert dr.allclose(dr.grad(y), 0.847213, rtol=5e-4)


@pytest.mark.skip("TODO")
def test70_loop_ballistic(m, do_record):
    class Ballistic(dr.CustomOp):
        def timestep(self, pos, vel, dt=0.02, mu=.1, g=9.81):
            acc = -mu*vel*dr.norm(vel) - m.Array2f(0, g)
            pos_out = pos + dt * vel
            vel_out = vel + dt * acc
            return pos_out, vel_out

        def eval(self, pos, vel):
            pos, vel = m.Array2f(pos), m.Array2f(vel)

            # Run for 100 iterations
            it, max_it = m.UInt(0), 100

            # Allocate scratch space
            n = max(dr.width(pos), dr.width(vel))
            self.temp_pos = dr.empty(m.Array2f, n * max_it)
            self.temp_vel = dr.empty(m.Array2f, n * max_it)

            loop = m.Loop("eval", lambda: (pos, vel, it))
            while loop(it < max_it):
                # Store current loop variables
                index = it * n + dr.arange(m.UInt, n)
                dr.scatter(self.temp_pos, pos, index)
                dr.scatter(self.temp_vel, vel, index)

                # Update loop variables
                pos_out, vel_out = self.timestep(pos, vel)
                pos.assign(pos_out)
                vel.assign(vel_out)

                it += 1

            # Ensure output and temp. arrays are evaluated at this point
            dr.eval(pos, vel)

            return pos, vel

        def backward(self):
            grad_pos, grad_vel = self.grad_out()

            # Run for 100 iterations
            it = m.UInt(100)

            loop = m.Loop("backward", lambda: (it, grad_pos, grad_vel))
            n = dr.width(grad_pos)
            while loop(it > 0):
                # Retrieve loop variables, backward chronological order
                it -= 1
                index = it * n + dr.arange(m.UInt, n)
                pos = dr.gather(m.Array2f, self.temp_pos, index)
                vel = dr.gather(m.Array2f, self.temp_vel, index)

                # Differentiate loop body in backward mode
                dr.enable_grad(pos, vel)
                pos_out, vel_out = self.timestep(pos, vel)
                dr.set_grad(pos_out, grad_pos)
                dr.set_grad(vel_out, grad_vel)
                dr.enqueue(dr.ADMode.Backward, pos_out, vel_out)
                dr.traverse(m.Float, dr.ADMode.Backward)

                # Update loop variables
                grad_pos.assign(dr.grad(pos))
                grad_vel.assign(dr.grad(vel))

            self.set_grad_in('pos', grad_pos)
            self.set_grad_in('vel', grad_vel)

    pos_in = m.Array2f([1, 2, 4], [1, 2, 1])
    vel_in = m.Array2f([10, 9, 4], [5, 3, 6])

    for i in range(20):
        dr.enable_grad(vel_in)
        dr.eval(vel_in, pos_in)
        pos_out, vel_out = dr.custom(Ballistic, pos_in, vel_in)
        loss = dr.squared_norm(pos_out - m.Array2f(5, 0))
        dr.backward(loss)

        vel_in = m.Array2f(dr.detach(vel_in) - 0.2 * dr.grad(vel_in))

    assert dr.allclose(loss, 0, atol=1e-4)
    assert dr.allclose(vel_in.x, [3.3516, 2.3789, 0.79156], rtol=1e-3)


@pytest.mark.skip("TODO")
def test71_loop_ballistic_2(m, do_record):
    class Ballistic2(dr.CustomOp):
        def timestep(self, pos, vel, dt=0.02, mu=.1, g=9.81):
            acc = -mu*vel*dr.norm(vel) - m.Array2f(0, g)
            pos_out = pos + dt * vel
            vel_out = vel + dt * acc
            return pos_out, vel_out

        def eval(self, pos, vel):
            pos, vel = m.Array2f(pos), m.Array2f(vel)

            # Run for 100 iterations
            it, max_it = m.UInt(0), 100

            loop = m.Loop("eval", lambda: (pos, vel, it))
            while loop(it < max_it):
                # Update loop variables
                pos, vel = self.timestep(pos, vel)
                it += 1

            self.pos = pos
            self.vel = vel

            return pos, vel

        def backward(self):
            grad_pos, grad_vel = self.grad_out()
            pos, vel = self.pos, self.vel

            # Run for 100 iterations
            it = m.UInt(0)

            loop = m.Loop("backward", lambda: (it, pos, vel, grad_pos, grad_vel))
            while loop(it < 100):
                # Take backward step in time
                pos, vel = self.timestep(pos, vel, dt=-0.02)

                # Take a forward step in time, keep track of derivatives
                pos_bwd, vel_bwd = m.Array2f(pos), m.Array2f(vel)
                dr.enable_grad(pos_bwd, vel_bwd)
                pos_fwd, vel_fwd = self.timestep(pos_bwd, vel_bwd, dt=0.02)

                dr.set_grad(pos_fwd, grad_pos)
                dr.set_grad(vel_fwd, grad_vel)
                dr.enqueue(dr.ADMode.Backward, pos_fwd, vel_fwd)
                dr.traverse(m.Float, dr.ADMode.Backward)

                grad_pos = dr.grad(pos_bwd)
                grad_vel = dr.grad(vel_bwd)
                it += 1

            self.set_grad_in('pos', grad_pos)
            self.set_grad_in('vel', grad_vel)

    pos_in = m.Array2f([1, 2, 4], [1, 2, 1])
    vel_in = m.Array2f([10, 9, 4], [5, 3, 6])

    for i in range(20):
        dr.enable_grad(vel_in)
        dr.eval(vel_in, pos_in)
        pos_out, vel_out = dr.custom(Ballistic2, pos_in, vel_in)
        loss = dr.squared_norm(pos_out - m.Array2f(5, 0))
        dr.backward(loss)

        vel_in = m.Array2f(dr.detach(vel_in) - 0.2 * dr.grad(vel_in))

    assert dr.allclose(loss, 0, atol=1e-4)
    assert dr.allclose(vel_in.x, [3.3516, 2.3789, 0.79156], atol=1e-3)


@pytest.mark.skip("TODO")
def test72_diffloop_simple_fwd(m, no_record):
    fi, fo = m.Float(1, 2, 3), m.Float(0, 0, 0)
    dr.enable_grad(fi)

    loop = m.Loop("MyLoop", lambda: fo)
    while loop(fo < 10):
        fo += fi
    dr.forward(fi)
    assert dr.grad(fo) == m.Float(10, 5, 4)


@pytest.mark.skip("TODO")
def test73_diffloop_simple_bwd(m, no_record):
    fi, fo = m.Float(1, 2, 3), m.Float(0, 0, 0)
    dr.enable_grad(fi)

    loop = m.Loop("MyLoop", lambda: fo)
    while loop(fo < 10):
        fo += fi
    dr.backward(fo)
    assert dr.grad(fi) == m.Float(10, 5, 4)


@pytest.mark.skip("TODO")
def test74_diffloop_masking_fwd(m, no_record):
    fo = dr.zeros(m.Float, 10)
    fi = m.Float(1, 2)
    i = m.UInt(0, 5)
    dr.enable_grad(fi)
    loop = m.Loop("MyLoop", lambda: i)
    while loop(i < 5):
        dr.scatter_reduce(dr.ReduceOp.Add, fo, fi, i)
        i += 1
    dr.forward(fi)
    assert fo == m.Float(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)
    assert dr.grad(fo) == m.Float(1, 1, 1, 1, 1, 0, 0, 0, 0, 0)


@pytest.mark.skip("TODO")
def test75_diffloop_masking_bwd(m, no_record):
    fo = dr.zeros(m.Float, 10)
    fi = m.Float(1, 2)
    i = m.UInt(0, 5)
    dr.enable_grad(fi)
    loop = m.Loop("MyLoop", lambda: i)
    while loop(i < 5):
        dr.scatter_reduce(dr.ReduceOp.Add, fo, fi, i)
        i += 1
    dr.backward(fo)
    assert dr.grad(fi) == m.Float(5, 0)


def test76_custom_op_non_array_types(m):
    class NonArrayType():
        def __init__(self):
            self.constant = 0.5

    v_input = m.Float(1)
    other_input = NonArrayType()
    dr.enable_grad(v_input)

    class TestOp(dr.CustomOp):
        def eval(self, value, other_input):
            result1 = value + other_input.constant
            result2 = 2 * result1
            other_output1 = NonArrayType()
            other_output2 = m.UInt32([1, 2, 3, 4])
            return result1, result2, other_output1, other_output2

        def forward(self):
            grad_in = self.grad_in('value')
            self.set_grad_out((grad_in, 2 * grad_in, None, None))

        def backward(self):
            g = self.grad_out()
            assert len(g) == 4
            self.set_grad_in('value', g[0] + 2 * g[1])

    # Backward mode
    result1, _, other1, other2 = dr.custom(TestOp, v_input, other_input)
    assert isinstance(other1, NonArrayType)
    assert isinstance(other2, m.UInt32)
    dr.backward(result1)
    assert dr.all(dr.grad(v_input) == 1)

    dr.set_grad(v_input, m.Float(0.))
    _, result2, other1, other2 = dr.custom(TestOp, v_input, other_input)
    dr.backward(result2)
    assert dr.all(dr.grad(v_input) == 2)

    dr.set_grad(v_input, m.Float(0.))
    result1, result2, other1, other2 = dr.custom(TestOp, v_input, other_input)
    dr.backward(result1 + result2)
    assert dr.all(dr.grad(v_input) == 3)

    # Forward mode
    dr.set_grad(v_input, m.Float(3.5))
    result1, result2, other1, other2 = dr.custom(TestOp, v_input, other_input)
    dr.forward_to(result1)
    # Pushes gradients to both outputs
    assert dr.all(dr.grad(result1) == 3.5)
    assert dr.all(dr.grad(result2) == 2 * 3.5)


def test77_forward_from_quaternion(m):
    # Propagating from a Quaternion, propagates from all its components
    a = m.Quaternion4f(1.0)
    dr.enable_grad(a.y) # Gradients on `j` component
    b = 2 * a
    dr.forward_from(a)
    assert dr.allclose(dr.grad(b), m.Quaternion4f(0, 2, 0, 0))


def test78_backward_from_quaternion(m):
    # Propagating from a Quaternion, propagates from all its components
    a = m.Quaternion4f(1.0)
    dr.enable_grad(a.y) # Gradients on `j` component
    b = 2 * a
    dr.backward_from(b)
    assert dr.allclose(dr.grad(a), m.Quaternion4f(0, 2, 0, 0))


def test79_forward_from_complex(m):
    # Propagating from a Quaternion, propagates from all its components
    a = m.Complex2f(1.0)
    dr.enable_grad(a.imag) # Gradients on imaginary component
    b = 2 * a
    dr.forward_from(a)
    assert dr.allclose(dr.grad(b), m.Complex2f(0, 2))


def test80_backward_from_complex(m):
    # Propagating from a Quaternion, propagates from all its components
    a = m.Complex2f(1.0)
    dr.enable_grad(a.imag) # Gradients on imaginary component
    b = 2 * a
    dr.backward_from(b)
    assert dr.allclose(dr.grad(a), m.Complex2f(0, 2))
