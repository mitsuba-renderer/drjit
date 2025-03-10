import drjit as dr
import pytest
import sys

@pytest.test_arrays('jit,float16,shape=(3, *),-diff', 'jit,float32,shape=(3, *),-diff')
def test01_pack_unpack(t):
    # Test coop vector creation and unpacking
    m = sys.modules[t.__module__]
    v = dr.full(dr.value_t(t), 7, 32)
    x = dr.coop.Vector(t(1, 2, 3), t(4, 5, 6), v, 8)
    assert len(x) == 8
    y = list(x)
    z = m.ArrayXf(x)
    result_ok = True
    for i in range(8):
        result_ok &= dr.all(y[i] == i+1)
        result_ok &= dr.all(z[i] == i+1)
    assert result_ok

@pytest.mark.parametrize('size', [64, 65])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test02_basic(t, size):
    # Simple arithmetic to investigate a bug with OptiX compilation
    x = dr.coop.Vector(*tuple(dr.full(t, i, 32) for i in range(size)))
    m = sys.modules[t.__module__]
    o = dr.coop.Vector(tuple(t(1) for _ in range(size)))
    x = x + o
    x = x + o
    x = m.TensorXf(x)
    assert dr.all((x[0] == 2) & (x[1] == 3))

@pytest.mark.parametrize('size', [0, 20, 10])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test02_add_sub(t, size):
    # Test addition and subtraction
    x = dr.coop.Vector(dr.full(t, 5, 32), 6, *tuple(range(size)))
    y = x + 15
    z = y - 2
    r0, r1 = list(z)[0:2]
    dr.schedule(r0, r1)
    assert dr.all((r0 == 18) & (r1 == 19))

@pytest.mark.parametrize('size', [0, 20, 100])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test03_add_min_max_fma(t, size):
    # Test min/max/FMA operations
    x = dr.coop.Vector(t(5), 8, *tuple(range(size)))
    x_min = dr.minimum(x, 6)
    x_max = dr.maximum(x, 7)
    z = dr.fma(x_min, x_max, 1)
    r0, r1 = list(z)[0:2]
    dr.schedule(r0, r1)
    assert r0 == 36 and r1 == 49

@pytest.mark.parametrize('sub_slice', [False, True])
@pytest.test_arrays('jit,float16,shape=(*),-diff')
def test04_pack_unpack(t, sub_slice):
    # Test the dr.coop.pack() and dr.coop.unpack() memory operations
    m = sys.modules[t.__module__]
    extra = 2 if sub_slice else 0
    X = m.TensorXf16(dr.arange(t, 24*(32+extra)), (24, 32+extra))
    Xv = dr.coop.view(X)

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

    for i in range(2):
        Pa = dr.coop.pack(
            Xv1, Xv2,
            layout='inference' if i == 0 else 'training'
        )

        X1a, X2a = dr.coop.unpack(*Pa)
        assert dr.all(m.TensorXf16(X1a) == X1[:, 0:32], axis=None)
        assert dr.all(m.TensorXf16(X2a) == X2[:, 0:32], axis=None)


@pytest.mark.parametrize('shape', [(2, 8), (5, 2), (16, 16)])
@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('bias', [False, True])
@pytest.mark.parametrize('pack', [False, True])
@pytest.test_arrays('jit,tensor,float16,-diff', 'jit,tensor,float32,-diff')
def test05_vecmat(t, shape, transpose, bias, pack):
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
            A, b = dr.coop.pack(A, b)
            assert A.buffer is b.buffer
        else:
            A = dr.coop.pack(A)
    else:
        A = dr.coop.view(A)
        if bias:
            b = dr.coop.view(b)

    rng_3 = m.PCG32(32, 3)
    x = [rng_3.next_float_normal(Float) for _ in range(input_size)]
    x_n = Tensor(x).numpy()

    x = dr.coop.Vector(x)
    r = dr.coop.matvec(A, x, b, transpose=transpose)
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
    # Test some special unary operations that are supported by coop vectors
    func = getattr(dr, op)
    x = dr.coop.Vector(t(0.1), t(0.2), t(0.3))
    r = func(x)
    x, y, z = r
    dr.schedule(x, y, z)
    assert dr.allclose(x[0], func(0.1), rtol=1e-3)
    assert dr.allclose(y[0], func(0.2), rtol=1e-3)
    assert dr.allclose(z[0], func(0.3), rtol=1e-3)


@pytest.test_arrays('jit,shape=(*),float16,-diff', 'jit,shape=(*),float32,-diff')
def test07_step(t):
    # Test the dr.step() function on coop vectors
    x = dr.coop.Vector(t(0.1), t(0.2))
    y = dr.coop.Vector(t(0.15), t(0.15))
    z = dr.step(x, y)
    r0, r1 = z
    dr.schedule(r0, r1)
    assert r0 == 0 and r1 == 1


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test09_fwd_grad_unpack(t):
    # Test that forward gradients correctly propagate through coop vector creation and unpacking
    a, b = t(1), t(2)
    dr.enable_grad(a, b)
    z = dr.coop.Vector(a, b) # pack
    x, y = z # unpack
    a.grad = 4
    b.grad = 5
    dr.forward_to(x, y)
    dr.schedule(x.grad, y.grad)
    assert x.grad == 4
    assert y.grad == 5


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test10_bwd_grad_unpack(t):
    # Test that backward gradients correctly propagate through coop vector creation and unpacking
    a, b = t(1), t(2)
    dr.enable_grad(a, b)
    z = dr.coop.Vector(a, b) # pack
    x, y = z # unpack
    x.grad = 4
    y.grad = 5
    dr.backward_to(a, b)
    dr.schedule(a.grad, b.grad)
    assert a.grad == 4
    assert b.grad == 5


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test11_fwd_addition(t):
    # Propagate forward gradients through an addition
    a, b = t(1), t(1)
    c, d = t(1), t(1)
    dr.enable_grad(a, b, c, d)
    x0 = dr.coop.Vector(a, b)
    x1 = dr.coop.Vector(c, d)
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
def test12_bwd_mul(t):
    # Propagate forward gradients through an addition
    a, b = t(8), t(9)
    c, d = t(3), t(2)
    dr.enable_grad(a, b, c, d)
    x0 = dr.coop.Vector(a, b)
    x1 = dr.coop.Vector(c, d)
    x2 = x0 * x1
    r0, r1 = x2
    r0.grad = 1
    r1.grad = 10
    dr.backward_to(a, b, c, d)
    dr.schedule(a.grad, b.grad, c.grad, d.grad)
    assert a.grad == 3 and b.grad == 20
    assert c.grad == 8 and d.grad == 90


@pytest.test_arrays('jit,shape=(*),float16,diff', 'jit,shape=(*),float32,diff')
def test13_bwd_min_max_fma(t):
    # Check derivatives of supported binary/ternary operations
    x = [ t(1), t(2), t(3), t(4) ]
    y = t(5)
    z = t(6)
    minval = t(25)
    maxval = t(12)
    dr.enable_grad(x, y, z, minval, maxval)
    q = dr.coop.Vector(x)

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
def test14_exp2_tanh_fwd(t):
    # Check derivatives of supported unary transcendental operations
    x = t(2)
    dr.enable_grad(x)
    y = dr.coop.Vector(x)
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
def test15_matvec_fwd(t, transpose, has_A_grad, has_x_grad, has_b_grad, layout):
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
    A_v = dr.coop.pack(Tensor(A), layout=layout)
    A_ref = Matrix2f(A)
    if has_A_grad:
        A_grad_v = dr.coop.pack(Tensor(A_grad))
        dr.enable_grad(A_v.buffer)
        A_v.buffer.grad = A_grad_v.buffer
        dr.enable_grad(A_ref)
        dr.set_grad(A_ref, A_grad)

    # Set up 'x' vector
    x = Array2f(1, 2)
    if has_x_grad:
        dr.enable_grad(x)
        x.grad = [2, 1]
    x_v = dr.coop.Vector(x)

    # Set up 'b' vector
    b_v = None
    b_ref = Array2f(0)
    if has_b_grad is not None:
        b1, b2 = Float(-1), Float(1)
        b_ref = Array2f(b1, b2)
        b_v = dr.coop.pack(Tensor([-1, 1]))

        if has_b_grad is True:
            dr.enable_grad(b_ref)
            b_ref.grad = [1, -1]
            b_grad_v = dr.coop.pack(Tensor([1, -1]))
            dr.enable_grad(b_v.buffer)
            b_v.buffer.grad = b_grad_v.buffer

    # Compute the reference
    if transpose:
        A_ref = A_ref.T
    y_ref = A_ref @ x + b_ref

    y = Array2f(dr.coop.matvec(A_v, x_v, b_v, transpose))
    dr.forward_to(y, y_ref)
    dr.schedule(y, y.grad, y_ref, y_ref.grad)

    print(f"primal: y={y} vs ref={y_ref}")
    print(f"grad:   y={y.grad} vs ref={y_ref.grad}")

    assert dr.all((y == y_ref) & (y.grad == y_ref.grad))
