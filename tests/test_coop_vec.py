import drjit as dr
import drjit.nn as nn
import pytest
import sys

def skip_if_coopvec_not_supported(t):
    if dr.backend_v(t) == dr.JitBackend.CUDA:
        if dr.detail.cuda_version() < (12, 8):
            pytest.skip("CUDA driver does not support cooperative vectors (Driver R570) or later is required")

@pytest.test_arrays('jit,float16,shape=(3, *),-diff', 'jit,float32,shape=(3, *),-diff')
def test01_pack_unpack(t):
    skip_if_coopvec_not_supported(t)

    # Test coop vector creation and unpacking
    m = sys.modules[t.__module__]
    v = dr.full(dr.value_t(t), 7, 32)
    x = nn.CoopVec(t(1, 2, 3), t(4, 5, 6), v, 8)
    assert len(x) == 8
    assert len(nn.CoopVec(*x, 2, (4, 5), *x)) == 19
    y = list(x)
    z = m.ArrayXf(x)
    assert len(y) == 8 and len(z) == 8
    result_ok = True
    for i in range(8):
        result_ok &= dr.all(y[i] == i+1)
        result_ok &= dr.all(z[i] == i+1)
    assert result_ok


@pytest.mark.parametrize('size', [0, 20, 10])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test02_add_sub(t, size):
    skip_if_coopvec_not_supported(t)

    # Test addition and subtraction
    x = nn.CoopVec(dr.full(t, 5, 32), 6, *tuple(range(size)))
    y = x + 15
    z = y - 2
    r0, r1 = list(z)[0:2]
    dr.schedule(r0, r1)
    assert dr.all((r0 == 18) & (r1 == 19))

@pytest.mark.parametrize('size', [0, 20, 100])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test03_add_min_max_fma(t, size):
    skip_if_coopvec_not_supported(t)

    # Test min/max/FMA operations
    x = nn.CoopVec(t(5), 8, *tuple(range(size)))
    x_min = dr.minimum(x, 6)
    x_max = dr.maximum(x, 7)
    # zero addition needed to work around a constant propagation bug in R570 driver..
    zero = dr.opaque(t, 0)
    z = dr.fma(x_min, x_max, 1+zero)
    r0, r1 = list(z)[0:2]
    dr.schedule(r0, r1)
    assert r0 == 36 and r1 == 49


@pytest.mark.parametrize('sub_slice', [False, True])
@pytest.test_arrays('jit,float16,shape=(*),-diff')
def test04_pack_unpack(t, sub_slice):
    skip_if_coopvec_not_supported(t)

    # Test the nn.pack() and nn.unpack() memory operations
    m = sys.modules[t.__module__]
    extra = 2 if sub_slice else 0
    X = m.TensorXf16(dr.arange(t, 24*(32+extra)), (24, 32+extra))
    Xv = nn.view(X)

    assert Xv.dtype == dr.VarType.Float16
    assert Xv.offset == 0
    assert Xv.size == 24*(32+extra)
    assert Xv.shape == (24, 32+extra)
    assert Xv.stride == 32+extra
    assert Xv.buffer is X.array

    Xv1 = Xv[0:16, 0:32]
    Xv2 = Xv[16:, 0:32]
    X1 = X[0:16, 0:32]
    X2 = X[16:, 0:32]

    assert Xv1.dtype == dr.VarType.Float16
    assert Xv1.offset == 0
    assert Xv1.shape == (16, 32)
    assert Xv1.stride == 32+extra
    assert Xv1.size == (Xv1.shape[0] - 1) * Xv1.stride + Xv1.shape[1]
    assert Xv1.buffer is X.array

    assert Xv2.dtype == dr.VarType.Float16
    assert Xv2.offset == 16*(32+extra)
    assert Xv2.size == (Xv2.shape[0] - 1) * Xv2.stride + Xv2.shape[1]
    assert Xv2.shape == (8, 32)
    assert Xv2.stride == 32+extra
    assert Xv2.buffer is X.array

    for layout in ('inference', 'training'):
        _, *Pa = nn.pack(
            Xv1, Xv2,
            layout=layout
        )

        _, X1a, X2a = nn.unpack(*Pa)
        assert dr.all(m.TensorXf16(X1a) == X1[:, 0:32], axis=None)
        assert dr.all(m.TensorXf16(X2a) == X2[:, 0:32], axis=None)


@pytest.mark.parametrize('shape', [(2, 8), (5, 2), (16, 16)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('pack', [False, True])
@pytest.test_arrays('jit,tensor,float16,-diff', 'jit,tensor,float32,-diff')
def test05_matvec(t, shape, transpose, bias, pack):
    skip_if_coopvec_not_supported(t)

    # Test matrix multiplication for various sizes and configurations (primal)
    m = sys.modules[t.__module__]
    Tensor = t
    Float = dr.array_t(t)

    if dr.backend_v(t) == dr.JitBackend.CUDA:
        if (not pack and shape[1] == 2) or \
           (not pack and transpose) or \
           dr.type_v(t) == dr.VarType.Float32:
            pytest.skip("Unsupported configuration")

    output_size = shape[1] if transpose else shape[0]
    input_size = shape[0] if transpose else shape[1]

    A = Tensor(m.PCG32(dr.prod(shape), 1).next_float_normal(Float), shape)
    A_n = A.numpy()

    if bias:
        b = Tensor(m.PCG32(output_size, 2).next_float_normal(Float))
        b_n = b.numpy()
    else:
        b = b_n = None

    if pack:
        if bias:
            _, A, b = nn.pack(A, b)
            assert A.buffer is b.buffer
        else:
            _, A = nn.pack(A)
    else:
        A = nn.view(A)
        if bias:
            b = nn.view(b)

    rng_3 = m.PCG32(32, 3)
    x = [rng_3.next_float_normal(Float) for _ in range(input_size)]
    x_n = Tensor(x).numpy()

    x = nn.CoopVec(x)
    r = nn.matvec(A, x, b, transpose=transpose)
    r_n = Tensor(r).numpy()

    if transpose:
        A_n = A_n.T
    ref = A_n @ x_n

    if bias:
        ref += b_n[:, None]

    assert dr.allclose(r_n, ref)


@pytest.test_arrays('jit,shape=(*),float16,-diff', 'jit,shape=(*),float32,-diff')
@pytest.mark.parametrize('op', ['exp2', 'log2', 'tanh'])
def test06_unary(t, op):
    skip_if_coopvec_not_supported(t)

    # Test some special unary operations that are supported by coop vectors
    func = getattr(dr, op)
    x = nn.CoopVec(t(0.1), t(0.2), t(0.3))
    r = func(x)
    x, y, z = r
    dr.schedule(x, y, z)
    assert dr.allclose(x[0], func(0.1), rtol=1e-3)
    assert dr.allclose(y[0], func(0.2), rtol=1e-3)
    assert dr.allclose(z[0], func(0.3), rtol=1e-3)


@pytest.test_arrays('jit,shape=(*),float16,-diff', 'jit,shape=(*),float32,-diff')
def test07_step(t):
    skip_if_coopvec_not_supported(t)

    # Test the dr.step() function on coop vectors
    x = nn.CoopVec(t(0.1), t(0.2))
    y = nn.CoopVec(t(0.15), t(0.15))
    z = dr.step(x, y)
    r0, r1 = z
    dr.schedule(r0, r1)
    assert r0 == 0 and r1 == 1


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test08_fwd_grad_unpack(t):
    skip_if_coopvec_not_supported(t)

    # Test that forward gradients correctly propagate through coop vector creation and unpacking
    a, b = t(1), t(2)
    dr.enable_grad(a, b)
    z = nn.CoopVec(a, b) # pack
    assert dr.grad_enabled(z)
    assert not dr.grad_enabled(dr.detach(z))
    x, y = z # unpack
    a.grad = 4
    b.grad = 5
    dr.forward_to(x, y)
    dr.schedule(x.grad, y.grad)
    assert x.grad == 4
    assert y.grad == 5
    assert dr.grad_enabled(z)
    assert not dr.grad_enabled(dr.detach(z))
    dr.disable_grad(z)
    assert not dr.grad_enabled(z)


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test09_bwd_grad_unpack(t):
    skip_if_coopvec_not_supported(t)

    # Test that backward gradients correctly propagate through coop vector creation and unpacking
    a, b = t(1), t(2)

    with pytest.raises(RuntimeError, match="to create a differentiable cooperative vector, construct it from grad-enabled components"):
        z = nn.CoopVec(a, b)
        dr.enable_grad(z)

    dr.enable_grad(a, b)
    z = nn.CoopVec(a, b) # pack
    x, y = z # unpack
    x.grad = 4
    y.grad = 5
    dr.backward_to(a, b)
    dr.schedule(a.grad, b.grad)
    assert a.grad == 4
    assert b.grad == 5


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test10_fwd_addition(t):
    skip_if_coopvec_not_supported(t)

    # Propagate forward gradients through an addition
    a, b = t(1), t(1)
    c, d = t(1), t(1)
    dr.enable_grad(a, b, c, d)
    x0 = nn.CoopVec(a, b)
    x1 = nn.CoopVec(c, d)
    x2 = x0 + x1
    r0, r1 = x2
    a.grad = 1
    b.grad = 2
    c.grad = 100
    d.grad = 200
    dr.forward_to(r0, r1)
    dr.schedule(r0.grad, r1.grad)
    assert r0.grad == 101 and r1.grad == 202


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test11_bwd_mul(t):
    skip_if_coopvec_not_supported(t)

    # Propagate forward gradients through a multiplication
    a, b = t(8), t(9)
    c, d = t(3), t(2)
    dr.enable_grad(a, b, c, d)
    x0 = nn.CoopVec(a, b)
    x1 = nn.CoopVec(c, d)
    x2 = x0 * x1
    r0, r1 = x2
    r0.grad = 1
    r1.grad = 10
    dr.backward_to(a, b, c, d)
    dr.schedule(a.grad, b.grad, c.grad, d.grad)
    assert a.grad == 3 and b.grad == 20
    assert c.grad == 8 and d.grad == 90


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test12_bwd_min_max_fma(t):
    skip_if_coopvec_not_supported(t)

    # Check derivatives of supported binary/ternary operations
    x = [ t(1), t(2), t(3), t(4) ]
    y = t(5)
    z = t(6)
    minval = t(25)
    maxval = t(12)
    dr.enable_grad(x, y, z, minval, maxval)
    q = nn.CoopVec(x)

    q = dr.fma(q, y, z)
    q = dr.minimum(q, minval)
    q = dr.maximum(q, maxval)

    a, b, c, d = q
    dr.backward_from(a+b*2 + c*4 + d*8)
    dr.schedule(x[0].grad, x[1].grad, x[2].grad, x[3].grad, y.grad,
                z.grad, minval.grad, maxval.grad, a, b, c, d)
    assert a[0] == 12 and b[0] == 16 and c[0] == 21 and d[0] == 25
    assert x[0].grad[0] == 0 and x[1].grad[0] == 10 and x[2].grad[0] == 20 and x[3].grad[0] == 0
    assert minval.grad[0] == 8 and maxval.grad[0] == 1

@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test13_exp2_tanh_fwd(t):
    skip_if_coopvec_not_supported(t)

    # Check derivatives of supported unary transcendental operations
    x = t(2)
    dr.enable_grad(x)
    y = nn.CoopVec(x)
    r0 = dr.exp2(y)
    r1 = dr.tanh(y)
    r0, = r0; r1, = r1
    dr.forward_from(x)
    dr.schedule(r0, r1, r0.grad, r1.grad)
    assert dr.allclose(r0[0], 4)
    assert dr.allclose(r1[0], 0.9640275800758169, rtol=1e-3)
    assert dr.allclose(r0.grad[0], 2.77259, rtol=1e-3)
    assert dr.allclose(r1.grad[0], 0.0706508, rtol=1e-2)


@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('has_A_grad', [False, True])
@pytest.mark.parametrize('has_x_grad', [False, True])
@pytest.mark.parametrize('has_b_grad', [None, False, True])
@pytest.mark.parametrize('layout', ['training', 'inference'])
@pytest.test_arrays('jit,tensor,float16,diff')
def test14_matvec_fwd(t, transpose, has_A_grad, has_x_grad, has_b_grad, layout):
    skip_if_coopvec_not_supported(t)

    # Test forward-propagation of derivatives from input through matrix multiplication
    m = sys.modules[t.__module__]
    Tensor = t
    Float = dr.array_t(t)
    Matrix2f = m.Matrix2f16
    Array2f = m.Array2f16

    if not has_A_grad and not has_x_grad and not has_b_grad:
        pytest.skip("Trivial configuration")
    if dr.backend_v(Float) == dr.JitBackend.LLVM and layout == 'training':
        pytest.skip("Layout not used in LLVM backend")

    # Set up 'A' matrix
    A      = [[4, 2], [5, 1]]
    A_grad = [[2, 1], [1, -1]]
    _, A_v = nn.pack(Tensor(A), layout=layout)
    A_ref = Matrix2f(A)
    if has_A_grad:
        _, A_grad_v = nn.pack(Tensor(A_grad))
        assert not dr.grad_enabled(A_v)
        dr.enable_grad(A_v)
        assert dr.grad_enabled(A_v)
        assert not dr.grad_enabled(dr.detach(A_v))
        A_v.buffer.grad = A_grad_v.buffer
        dr.enable_grad(A_ref)
        dr.set_grad(A_ref, A_grad)

    # Set up 'x' vector
    x = Array2f(1, 2)
    if has_x_grad:
        dr.enable_grad(x)
        x.grad = [2, 1]
    x_v = nn.CoopVec(x)

    # Set up 'b' vector
    b_v = None
    b_ref = Array2f(0)
    if has_b_grad is not None:
        b1, b2 = Float(-1), Float(1)
        b_ref = Array2f(b1, b2)
        _, b_v = nn.pack(Tensor([-1, 1]))

        if has_b_grad is True:
            dr.enable_grad(b_ref)
            b_ref.grad = [1, -1]
            _, b_grad_v = nn.pack(Tensor([1, -1]))
            dr.enable_grad(b_v.buffer)
            b_v.buffer.grad = b_grad_v.buffer

    # Compute the reference
    if transpose:
        A_ref = A_ref.T
    y_ref = A_ref @ x + b_ref

    y = Array2f(nn.matvec(A_v, x_v, b_v, transpose))
    dr.forward_to(y, y_ref)
    dr.schedule(y, y.grad, y_ref, y_ref.grad)

    # print(f"primal: y={y} vs ref={y_ref}")
    # print(f"grad:   y={y.grad} vs ref={y_ref.grad}")

    assert dr.all((y == y_ref) & (y.grad == y_ref.grad))


@pytest.mark.parametrize('transpose', [False, True])
@pytest.test_arrays('jit,tensor,float16,-diff')
def test15_matvec_in_vcall(t, transpose):
    skip_if_coopvec_not_supported(t)

    # Check that mat-vec products still work as expected when done from a callable
    Float = dr.array_t(t)
    UInt32 = dr.uint32_array_t(Float)
    size = 64
    rng = dr.rng()
    A = rng.normal(t, (size, size))
    b = rng.normal(t, size)
    _, A, b = nn.pack(A, b)

    def mult_it():
        x = nn.CoopVec(
            Float(i/(size-1) - 0.5) for i in range(size)
        )
        return list(nn.matvec(A, x, b, transpose=transpose))[0]

    r0 = mult_it()
    r1 = dr.switch(UInt32(0), [mult_it])

    dr.schedule(r0, r1)
    assert dr.allclose(r0[0], r1[0])

    # Try again without bias vector
    b = None

    r0 = mult_it()
    r1 = dr.switch(UInt32(0), [mult_it])

    dr.schedule(r0, r1)
    assert r0[0] == r1[0]


@pytest.mark.parametrize('in_vcall', [False, True])
@pytest.mark.parametrize('grad_lit', [False, True])
@pytest.mark.parametrize('separate_buffers', [False, True])
@pytest.test_arrays('jit,tensor,float16,diff')
def test16_matvec_bwd(t, in_vcall, grad_lit, separate_buffers):
    skip_if_coopvec_not_supported(t)

    # Test the reverse-mode derivative of a matrix-vector product
    # (potentially in a vcall)

    m = sys.modules[t.__module__]
    UInt32 = m.UInt32
    A = t([[1, 3], [-2, 4], [3, -2]])
    b = t([0, 0, 0])
    if separate_buffers:
        buffer, Av= nn.pack(A, layout='training')
        buffer_bias, bv = nn.pack(b, layout='training')
    else:
        buffer, Av, bv = nn.pack(A, b, layout='training')
    x = m.Array2f16(2, 4)
    dr.enable_grad(x, buffer)

    if separate_buffers:
        dr.enable_grad(x, buffer_bias)
    dr.enable_grad(x, buffer)

    if grad_lit:
        dr.set_grad(buffer, dr.zeros(type(buffer), dr.width(buffer)))
        if separate_buffers:
            dr.set_grad(buffer_bias, dr.zeros(type(buffer_bias), dr.width(buffer_bias)))

    def do_mul(x):
        xv = nn.CoopVec(x)
        yv = nn.matvec(Av, xv, bv)
        return m.Array3f16(yv)

    if in_vcall:
        y = dr.switch(UInt32(0), [do_mul], x)
    else:
        y = do_mul(x)

    z = dr.opaque(dr.array_t(t), 0)

    y.grad = (-2+z, 5+z, 10+z)
    dr.backward_from(y)
    grad_x = x.grad

    # print(f"{y=}")
    # print(f"{grad_x=}")

    grad_x_ref = m.Array2f16(18, -6)
    assert dr.all(grad_x_ref == grad_x)

    dr.schedule(grad_x)
    _, grad_A = nn.unpack(Av.grad)
    _, grad_b = nn.unpack(bv.grad)

    grad_A = t(grad_A)
    grad_b = t(grad_b)[:, 0]
    grad_A_ref = t([[-4, -8], [10, 20], [20, 40]])
    grad_b_ref = t([-2, 5, 10])
    assert dr.all(grad_A_ref == grad_A)
    assert dr.all(grad_b_ref == grad_b)


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test17_cast(t):
    skip_if_coopvec_not_supported(t)

    z = dr.opaque(t, 0)
    a = nn.CoopVec(
        z + 1,
        z + 2,
        z + 3
    )
    b = nn.cast(a, dr.float32_array_t(t))
    c = nn.cast(b, dr.float16_array_t(t))
    x, y, z = c
    dr.eval(x, y, z)
    assert x[0] == 1 and y[0] == 2 and z[0] == 3

    # Test gradient propagation through cast
    x0, x1 = t(1), t(2)
    dr.enable_grad(x0, x1)
    a = nn.CoopVec(x0, x1)
    b = nn.cast(a, dr.float32_array_t(t))
    c = nn.cast(b, dr.float16_array_t(t))
    y0, y1 = c
    dr.set_grad(y0, 10)
    dr.set_grad(y1, 20)
    dr.backward_to(x0, x1)
    assert x0.grad == 10
    assert x1.grad == 20



@pytest.test_arrays('jit,shape=(*),float32,-diff')
@dr.syntax
def test18_symbolic_loop_if_stmt(t):
    skip_if_coopvec_not_supported(t)

    # Test that cooperative vectors can be passed through
    # symbolic loops and conditionals
    UInt32 = dr.uint32_array_t(t)
    a = nn.CoopVec(t(1), t(2))
    i = UInt32(0)

    while i < 10:
        if i > 5:
            a += a
        i += 1

    x, y = a
    dr.schedule(x, y, i)
    assert x[0] == 16 and y[0] == 32


@pytest.test_arrays('jit,shape=(*),float32,-diff')
@dr.syntax
def test19_no_eval(t):
    skip_if_coopvec_not_supported(t)

    # Cooperative vectors cannot be evaluted via dr.eval()
    a = nn.CoopVec(t(1), t(2))
    with pytest.raises(RuntimeError, match="Cooperative vectors cannot be evaluated"):
        dr.schedule(a)
    with pytest.raises(RuntimeError, match="Cooperative vectors cannot be evaluated"):
        dr.eval(a)
    with pytest.raises(RuntimeError, match="Cooperative vectors cannot be evaluated"):
        dr.make_opaque(a)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('jit,shape=(*),float16,diff')
@dr.syntax
def test20_matvec_in_loop(t, mode):
    # Check that derivative inference works when
    # cooperative vectors are used inside loops
    skip_if_coopvec_not_supported(t)

    m = sys.modules[t.__module__]
    Float16 = t
    TensorXf16 = m.TensorXf16
    Float32 = m.Float32
    UInt32 = m.UInt32

    A = dr.ones(TensorXf16, shape=(2, 2))
    b = dr.zeros(Float16, shape=(2))

    _, A_view, b_view = nn.pack(A, b, layout='inference')

    cnt = UInt32(0)
    res = Float32(0)

    while dr.hint(cnt < 3, mode=mode):
        x = nn.CoopVec(Float16([0.5]), Float16([0.5]))
        a, b = nn.matvec(A_view, x, b_view)
        res += Float32(a)
        cnt += 1

    assert res == 3


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('jit,shape=(*),float16,diff')
@dr.syntax
def test21_optimize_in_loop_bwd(t, mode):
    # Check that derivative backpropagation occurs when
    # cooperative vectors are used inside loops
    skip_if_coopvec_not_supported(t)

    m = sys.modules[t.__module__]
    Float16 = t
    TensorXf16 = m.TensorXf16
    Float32 = m.Float32
    UInt32 = m.UInt32

    A = dr.ones(TensorXf16, shape=(2, 2))
    b = dr.zeros(Float16, shape=(2))

    buf, A_view, b_view = nn.pack(A, b, layout='training')
    dr.enable_grad(buf)

    cnt = dr.zeros(UInt32, 2)
    res = dr.zeros(Float32, 2)

    while dr.hint(cnt < 3, max_iterations=-1, mode=mode):
        x = nn.CoopVec(Float16(0.5), Float16(0.5))
        a, _ = nn.matvec(A_view, x, b_view)
        res += Float32(a)
        cnt += 1

    dr.backward(res)

    _, A_view, b_view = nn.unpack(A_view.grad, b_view.grad)
    A = TensorXf16(A_view)
    b = TensorXf16(b_view)
    assert dr.all(A == TensorXf16([[3, 3], [0, 0]]))
    assert dr.all(b == TensorXf16([[6], [0]]))


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('jit,shape=(*),float16,diff')
@dr.syntax
def test22_optimize_in_loop_bwd_v2(t, mode):
    # Check that derivative backpropagation occurs when
    # cooperative vectors are used inside loops, and the
    # backprop call is placed there as well

    skip_if_coopvec_not_supported(t)

    m = sys.modules[t.__module__]
    Float16 = t
    TensorXf16 = m.TensorXf16
    Float32 = m.Float32
    UInt32 = m.UInt32

    A = dr.ones(TensorXf16, shape=(2, 2))
    b = dr.zeros(Float16, shape=(2))

    buf, A_view, b_view = nn.pack(A, b, layout='training')
    dr.enable_grad(buf)

    cnt = dr.zeros(UInt32, 2)
    res = dr.zeros(Float32, 2)

    while dr.hint(cnt < 3, mode=mode, exclude=[A_view, b_view]):
        x = nn.CoopVec(Float16(0.5), Float16(0.5))
        a, _ = nn.matvec(A_view, x, b_view)
        res = Float32(a)
        dr.backward(res)
        cnt += 1

    _, A_view, b_view = nn.unpack(A_view.grad, b_view.grad)
    A = TensorXf16(A_view)
    b = TensorXf16(b_view)
    assert dr.all(A == TensorXf16([[3, 3], [0, 0]]))
    assert dr.all(b == TensorXf16([[6], [0]]))


@pytest.test_arrays('jit,shape=(*),float16,diff')
def test23_linear_layer_dtype(t):
    m = sys.modules[t.__module__]
    Float16 = t
    TensorXf16 = m.TensorXf16

    net = nn.Sequential(
        nn.Linear(-1, 32, bias=False),
        nn.ReLU(),
        nn.Linear(-1, 3)
    )

    _ = net.alloc(TensorXf16, 2)
    with pytest.raises(TypeError, match="Linear layer requires a Tensor type"):
        _ = net.alloc(Float16, 2)
