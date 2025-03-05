import drjit as dr
import pytest
import sys

@pytest.mark.parametrize('size', [0, 20, 100])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test01_init_get_set(t, size):
    x = dr.coop.Vector(t(5), 6, *tuple(range(size)))
    assert len(x) == 2 + size
    x[1] = 11
    r0, r1 = x[0], x[1]
    dr.schedule(r0, r1)
    assert r0 == 5 and r1 == 11

@pytest.mark.parametrize('size', [64, 65])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test02_basic(t, size):
    dr.set_flag(dr.JitFlag.PrintIR, True)
    x = dr.coop.Vector(*tuple(t(i) for i in range(size)))
    m = sys.modules[t.__module__]
    q = m.TensorXf(x)
    o = dr.coop.Vector(tuple(t(1) for _ in range(size)))
    x = x + o
    x = x + o
    x = m.TensorXf(x)
    print(x)
    print(q)
    assert x[0] == 2 and x[1] == 3

@pytest.mark.parametrize('size', [0, 20, 100])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test02_add_sub(t, size):
    dr.set_flag(dr.JitFlag.PrintIR, True)
    x = dr.coop.Vector(t(5), 6, *tuple(range(size)))
    y = x + 15
    z = y - 2
    r0, r1 = z[0], z[1]
    dr.schedule(r0, r1)
    assert r0 == 18 and r1 == 19

@pytest.mark.parametrize('size', [0, 20, 100])
@pytest.test_arrays('jit,float16,shape=(*),-diff', 'jit,float32,shape=(*),-diff')
def test03_add_min_max_fma(t, size):
    x = dr.coop.Vector(t(5), 8, *tuple(range(size)))
    x_min = dr.minimum(x, 6)
    x_max = dr.maximum(x, 7)
    z = dr.fma(x_min, x_max, 1)
    r0, r1 = z[0], z[1]
    dr.schedule(r0, r1)
    assert r0 == 36 and r1 == 49

@pytest.mark.parametrize('sub_slice', [False, True])
@pytest.test_arrays('jit,float16,shape=(*),-diff')
def test04_pack_unpack(t, sub_slice):
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
    m = sys.modules[t.__module__]
    Tensor = t
    Float = dr.array_t(t)

    if dr.backend_v(t) == dr.JitBackend.CUDA and not pack:
        pytest.skip("Skip for now, crashes on driver R570")

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
    x = dr.coop.Vector(t(0.1), t(0.2))
    y = dr.coop.Vector(t(0.15), t(0.15))
    z = dr.step(x, y)
    r0, r1 = z
    dr.schedule(r0, r1)
    assert r0 == 0 and r1 == 1
