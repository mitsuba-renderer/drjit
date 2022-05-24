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

    a = struct_class(m)()
    dr.enable_grad(a)
    dr.set_grad(a, 1.0)
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 1.0)
    dr.set_grad(a, struct_class(m)())
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 2.0)

    with pytest.raises(TypeError) as ei:
        dr.grad(a, preserve_type=False)
    assert "preserve_type=True is required" in str(ei.value)


def test04_accum_grad(m):
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

    a = struct_class(m)()
    dr.enable_grad(a)
    dr.accum_grad(a, 1.0)
    assert dr.allclose(dr.grad(a).x, 1.0)
    assert dr.allclose(dr.grad(a).y, 1.0)
    dr.accum_grad(a, struct_class(m)())
    assert dr.allclose(dr.grad(a).x, 2.0)
    assert dr.allclose(dr.grad(a).y, 3.0)


def test05_replace_grad(m):
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

def test06_set_label(m):
    a = m.Float(1.0)
    b = [m.Float(1.0), m.Float(2.0)]
    c = m.Array3f(1.0, 2.0, 3.0)
    d = struct_class(m)()
    dr.enable_grad(a, b, c, d)

    assert dr.label(a) == 'unnamed'

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

    with pytest.raises(TypeError) as ei:
        dr.set_label(a, 'aa', b=b)
    assert "incompatible function arguments" in str(ei.value)


def test07_forward_to(m):
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
    assert "AD flags should be passed via the" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, flags=dr.ADFlag.Default, test='test')
    assert "only AD flags should be passed" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.forward_to(b, c, test='test')
    assert "only AD flags should be passed" in str(ei.value)

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
    assert "expected a Dr.JIT array type" in str(ei.value)

    # Trying to call with a different flag
    dr.set_grad(a, 1.0)
    b = a * a * 2
    grad_b = dr.forward_to(b, flags=dr.ADFlag.ClearInterior)
    assert dr.allclose(dr.grad(a), 1.0)
    assert dr.allclose(grad_b, 4.0)


def test08_forward_from(m):
    with pytest.raises(TypeError) as ei:
        dr.forward_from(1.0)
    assert "expected a Dr.JIT array type" in str(ei.value)

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


def test09_backward_to(m):
    with pytest.raises(TypeError) as ei:
        dr.backward_to(1.0)
    assert "expected a Dr.JIT array type" in str(ei.value)

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


def test10_backward_from(m):
    with pytest.raises(TypeError) as ei:
        dr.backward_from(1.0)
    assert "expected a Dr.JIT array type" in str(ei.value)

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

# ------------------------------------------------------------------------------

def test_ad_01_add_bwd(m):
    a, b = m.Float(1), m.Float(2)
    dr.enable_grad(a, b)
    c = 2 * a + b
    dr.backward(c)
    assert dr.grad(a) == 2
    assert dr.grad(b) == 1


def test_ad_02_add_fwd(m):
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


def test_ad_03_branch_fwd(m):
    a = m.Float(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.forward(a)
    assert dr.grad(d) == 2


def test_ad_04_branch_ref(m):
    a = m.Float(1)
    dr.enable_grad(a)

    b = a + 1
    c = a + 1
    d = b + c

    del b, c

    dr.backward(d)
    assert dr.grad(a) == 2


def test_ad_05_sub_mul(m):
    a, b, c = m.Float(2), m.Float(3), m.Float(4)
    dr.enable_grad(a, b, c)
    d = a * b - c
    dr.backward(d)
    assert dr.grad(a) == dr.detach(b)
    assert dr.grad(b) == dr.detach(a)
    assert dr.grad(c) == -1


def test_ad_06_div(m):
    a, b = m.Float(2), m.Float(3)
    dr.enable_grad(a, b)
    d = a / b
    dr.backward(d)
    assert dr.allclose(dr.grad(a),  1.0 / 3.0)
    assert dr.allclose(dr.grad(b), -2.0 / 9.0)

def test07_sum_0_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y, 95.0/27.0)
    assert dr.allclose(dr.grad(x), 2 * dr.detach(x))


def test08_sum_0_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(x*x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(dr.detach(y), 95.0/27.0)
    assert len(dr.grad(y)) == 1 and dr.allclose(dr.grad(y), 10)


def test09_sum_1_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 11)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.backward(y)
    assert dr.allclose(dr.grad(x), 11)


def test10_sum_1_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x)*x)
    dr.forward(x)
    assert dr.allclose(dr.grad(y), 100)


def test11_sum_2_bwd(m):
    x = dr.linspace(m.Float, 0, 1, 11)
    dr.enable_grad(x)
    z = dr.sum(dr.sum(x*x)*x*x)
    dr.backward(z)
    assert dr.allclose(dr.grad(x),
                       [0., 1.54, 3.08, 4.62, 6.16, 7.7,
                        9.24, 10.78, 12.32, 13.86, 15.4])


def test12_sum_2_fwd(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.sum(dr.sum(x*x)*dr.sum(x*x))
    dr.forward(x)
    assert dr.allclose(dr.grad(y), 1900.0 / 27.0)


def test13_prod(m):
    x = m.Float(1, 2, 5, 8)
    dr.enable_grad(x)
    y = dr.prod(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 80)
    assert dr.allclose(dr.grad(x), [80, 40, 16, 10])


def test14_max_bwd(m):
    x = m.Float(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.backward(y)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(x), [0, 0, 1, 0, 1])


def test15_max_fwd(m):
    x = m.Float(1, 2, 8, 5, 8)
    dr.enable_grad(x)
    y = dr.max(x)
    dr.forward(x)
    assert len(y) == 1 and dr.allclose(y[0], 8)
    assert dr.allclose(dr.grad(y), [2])  # Approximation


def test_ad_16_sqrt(m):
    x = m.Float(1, 4, 16)
    dr.enable_grad(x)
    y = dr.sqrt(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [1, 2, 4])
    assert dr.allclose(dr.grad(x), [.5, .25, .125])


def test_ad_17_rsqrt(m):
    x = m.Float(1, .25, 0.0625)
    dr.enable_grad(x)
    y = dr.rsqrt(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [1, 2, 4])
    assert dr.allclose(dr.grad(x), [-.5, -4, -32])


def test_ad_18_abs(m):
    x = m.Float(-2, 2)
    dr.enable_grad(x)
    y = dr.abs(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), [2, 2])
    assert dr.allclose(dr.grad(x), [-1, 1])


def test_ad_19_sin(m):
    x = dr.linspace(m.Float, 0, 10, 10)
    dr.enable_grad(x)
    y = dr.sin(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), dr.sin(dr.detach(x)))
    assert dr.allclose(dr.grad(x), dr.cos(dr.detach(x)))


def test_ad_20_cos(m):
    x = dr.linspace(m.Float, 0.01, 10, 10)
    dr.enable_grad(x)
    y = dr.cos(x)
    dr.backward(y)
    assert dr.allclose(dr.detach(y), dr.cos(dr.detach(x)))
    assert dr.allclose(dr.grad(x), -dr.sin(dr.detach(x)))


def test_ad_29_log(m):
    x = dr.linspace(m.Float, 0.01, 1, 10)
    dr.enable_grad(x)
    y = dr.log(x * x)
    dr.backward(y)
    log_x = dr.log(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, log_x)
    assert dr.allclose(dr.grad(x), 2 / dr.detach(x))


def test_ad_30_pow(m):
    x = dr.linspace(m.Float, 1, 10, 10)
    y = dr.full(m.Float, 2.0, 10)
    dr.enable_grad(x, y)
    z = x**y
    dr.backward(z)
    assert dr.allclose(dr.grad(x), dr.detach(x)*2)
    assert dr.allclose(dr.grad(y),
                       m.Float(0., 2.77259, 9.88751, 22.1807, 40.2359,
                               64.5033, 95.3496, 133.084, 177.975, 230.259))


def test_ad_33_tan(m):
    x = dr.linspace(m.Float, 0, 1, 10)
    dr.enable_grad(x)
    y = dr.tan(x * x)
    dr.backward(y)
    tan_x = dr.tan(dr.sqr(dr.detach(x)))
    assert dr.allclose(y, tan_x)
    assert dr.allclose(dr.grad(x),
                       m.Float(0., 0.222256, 0.44553, 0.674965, 0.924494,
                               1.22406, 1.63572, 2.29919, 3.58948, 6.85104))


def test_ad_35_asin(m):
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


def test_ad_36_acos(m):
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


def test_ad_37_atan(m):
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


def test_ad_38_atan2(m):
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


def test_ad_39_cbrt(m):
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


def test_ad_40_sinh(m):
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


def test_ad_41_cosh(m):
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


def test_ad_42_tanh(m):
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


def test_ad_43_asinh(m):
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


def test_ad_44_acosh(m):
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


def test_ad_45_atanh(m):
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

# ------------------------------------------------------------------------------

class Normalize(dr.CustomOp):
    def eval(self, value):
        print('eval in')
        self.value = value
        self.inv_norm = dr.rcp(dr.norm(value))
        # print('eval out')
        return value * self.inv_norm
        # return value

    def forward(self):
        print('forward!')
        grad_in = self.grad_in('value')
        grad_out = grad_in * self.inv_norm
        grad_out -= self.value * (dr.dot(self.value, grad_out) * dr.sqr(self.inv_norm))
        self.set_grad_out(grad_out)

    def backward(self):
        print('backward!')
        grad_out = self.grad_out()
        grad_in = grad_out * self.inv_norm
        grad_in -= self.value * (dr.dot(self.value, grad_in) * dr.sqr(self.inv_norm))
        self.set_grad_in('value', grad_in)

    def name(self):
        return "normalize"


def test48_custom_backward(m):
    d = m.Array3f(1, 2, 3)
    dr.enable_grad(d)
    d2 = dr.custom(Normalize, d)
    dr.set_grad(d2, m.Array3f(5, 6, 7))
    dr.enqueue(dr.ADMode.Backward, d2)
    dr.traverse(m.Float, dr.ADMode.Backward)
    assert dr.allclose(dr.grad(d), m.Array3f(0.610883, 0.152721, -0.305441))


def test49_custom_forward(m):
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


def test50_custom_forward_external_dependency(m):
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


def test60_implicit_dep_customop(m):
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
            self.set_grad_out(grad_in * dr.detach(v1) + self.value * dr.grad(v1))

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