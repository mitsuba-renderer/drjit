import drjit as dr
import pytest

@pytest.test_arrays('-bool,shape=(*)')
def test01_gather_simple(t):
    assert dr.all(dr.gather(
        dtype=t,
        source=dr.arange(t, 10),
        index=dr.uint32_array_t(t)(0, 5, 3)
    ) == t(0, 5, 3))

    assert dr.all(dr.gather(
        dtype=t,
        source=dr.arange(t, 10),
        index=dr.uint32_array_t(t)(0, 5, 3),
        active=dr.mask_t(t)(True, False, True)
    ) == t(0, 0, 3))

    with pytest.raises(TypeError, match="unsupported dtype"):
        dr.gather(
            dtype=str,
            source=dr.arange(t, 10),
            index=dr.uint32_array_t(t)(0, 5, 3),
            active=dr.mask_t(t)(True, False, True)
        )


@pytest.test_arrays('-bool,shape=(*)')
def test02_scatter_simple(t):
    target = dr.empty(t, 3)
    dr.scatter(target, 1, [0, 1])
    dr.scatter(target, 2, 2)
    assert dr.all(target == t([1, 1, 2]))

    target = dr.zeros(t, 3)
    dr.scatter(target=target, value=1, index=[0, 1, 2], active=[True, True, False])
    assert dr.all(target == t([1, 1, 0]))


@pytest.test_arrays('-bool,shape=(*, *)')
def test03_gather_nested(t):
    vt = dr.value_t(t)
    v = t(dr.arange(vt, 10), dr.arange(vt, 10)+10)
    r = t([0, 5, 3], [10, 15, 13])
    active = dr.mask_t(vt)(True, False, True)
    idx = dr.uint32_array_t(vt)(0, 5, 3)

    assert dr.all(dr.gather(
        dtype=t,
        source=v,
        index=idx) == r, axis=None)

    r = t([0, 0, 3], [10, 0, 13])

    assert dr.all(dr.gather(
        dtype=t,
        source=v,
        index=idx,
        active=active) == r, axis=None)


@pytest.test_arrays('-bool,shape=(*, *)')
def test04_scatter_nested(t):
    vt = dr.value_t(t)

    buf = t(dr.arange(vt, 10), dr.arange(vt, 10)+10)
    val = t([3, 4, 5], [30, 40, 50])

    idx = dr.uint32_array_t(vt)(0, 5, 3)

    dr.scatter(
        buf,
        val,
        idx
    )

    assert dr.all(buf[0] == [3,  1, 2, 5,  4, 4,  6, 7, 8, 9])
    assert dr.all(buf[1] == [30, 11, 12, 50, 14, 40, 16, 17, 18, 19])

    dr.scatter(
        buf,
        val+1,
        idx,
        dr.mask_t(vt)(True, False, True)
    )

    assert dr.all(buf[0] == [4,  1, 2, 6,  4, 4,  6, 7, 8, 9])
    assert dr.all(buf[1] == [31, 11, 12, 51, 14, 40, 16, 17, 18, 19])


@pytest.test_arrays('-bool,shape=(3, *)')
def test05_gather_nested_2(t):
    vt = dr.value_t(t)
    v = dr.arange(vt, 30)
    idx = dr.uint32_array_t(vt)(0, 5, 3)
    active = dr.mask_t(vt)(True, False, True)
    r = t(
        vt(0, 15, 9),
        vt(1, 16, 10),
        vt(2, 17, 11)
    )
    assert dr.all(dr.gather(t, v, idx) == r, axis=None)
    r = t(
        vt(0, 0, 9),
        vt(1, 0, 10),
        vt(2, 0, 11)
    )
    assert dr.all(dr.gather(t, v, idx, active) == r, axis=None)


@pytest.test_arrays('-bool,shape=(3, *)')
def test06_scatter_nested_2(t):
    vt = dr.value_t(t)
    buf = dr.zeros(vt, 9)
    dr.scatter(
        buf,
        t([1, 2], [3, 4], [5, 6]),
        dr.uint32_array_t(vt)(0, 2)
    )

    assert dr.all(buf == [1, 3, 5, 0, 0, 0, 2, 4, 6])
    dr.scatter(
        buf,
        t([8, 2], [9, 4], [10, 6]),
        dr.uint32_array_t(vt)(1, 2),
        dr.mask_t(vt)(True, False)
    )
    assert dr.all(buf == [1, 3, 5, 8, 9, 10, 2, 4, 6])


@pytest.test_arrays('-bool,shape=(*)')
def test07_gather_pytree(t):
    x = t([1, 2, 3, 4])
    y = t([5, 6, 7, 8])
    i = dr.uint32_array_t(t)([1, 0])
    r = dr.gather(tuple, (x, y), i)
    assert type(r) is tuple and len(r) == 2
    assert dr.all(r[0] == t([2, 1]))
    assert dr.all(r[1] == t([6, 5]))

    class MyStruct:
        DRJIT_STRUCT = { 'a' : t }
        def __init__(self, a: t):
            self.a = a

    x = MyStruct(x)
    r = dr.gather(MyStruct, x, i)
    assert type(r) is MyStruct
    assert dr.all(r.a == t([2, 1]))


@pytest.test_arrays('-bool,shape=(*)')
def test08_scatter_pytree(t):
    x = dr.zeros(t, 4)
    y = dr.zeros(t, 4)
    i = dr.uint32_array_t(t)([1, 0])
    trg = (x, y)
    dr.scatter(
        trg,
        (1, [2, 3]),
        [1, 3]
    )
    assert dr.all(x == [0, 1, 0, 1])
    assert dr.all(y == [0, 2, 0, 3])

    class MyStruct:
        DRJIT_STRUCT = { 'a' : t }
        def __init__(self, a: t):
            self.a = a

    x = MyStruct(x)
    x.a = dr.zeros(t, 4)
    dr.scatter(
        x,
        MyStruct(t(1, 2)),
        (1, 0),
        (True, False)
    )
    assert dr.all(x.a == [0, 1, 0, 0])


def test09_ravel_scalar():
    s = dr.scalar
    v = s.ArrayXf(1, 2, 3)
    assert dr.ravel(v) is v
    assert dr.all(dr.ravel(s.Array3f(1, 2, 3), order='C') == s.ArrayXf([1, 2, 3]))
    assert dr.all(dr.ravel(s.Array3f(1, 2, 3), order='F') == s.ArrayXf([1, 2, 3]))
    assert dr.all(dr.ravel(s.Array22f([1, 2], [3, 4]), order='C') == s.ArrayXf([1, 2, 3, 4]))
    assert dr.all(dr.ravel(s.Array22f([1, 2], [3, 4]), order='F') == s.ArrayXf([1, 3, 2, 4]))


@pytest.test_arrays('-bool,shape=(3, *)')
def test10_ravel_vec(t, drjit_verbose, capsys):
    vt = dr.value_t(t)
    v = vt(1, 2, 3)
    assert dr.unravel(vt, v) is v
    assert dr.all(dr.ravel(t([1, 2, 3], [4, 5, 6], 7), order='C') == [1, 2, 3, 4, 5, 6, 7, 7, 7])
    assert dr.all(dr.ravel(t([1, 2, 3], [4, 5, 6], 7), order='F') == [1, 4, 7, 2, 5, 7, 3, 6, 7])
    assert dr.all(dr.ravel(t([1, 2, 3], [4, 5, 6], 7)) == [1, 4, 7, 2, 5, 7, 3, 6, 7])

    transcript = capsys.readouterr().out
    assert transcript.count('jit_var_scatter') == 9

    assert dr.all(dr.ravel(t()) == [])

    with pytest.raises(RuntimeError, match="order parameter must equal"):
        dr.ravel(t(), order='Q')

    with pytest.raises(RuntimeError, match="ragged"):
        dr.ravel(t([1,2], [3, 4], [5, 6, 7]))


def test11_unravel_scalar():
    s = dr.scalar
    v = s.ArrayXf(1, 2, 3)
    assert dr.unravel(type(v), v) is v
    assert dr.all(dr.unravel(dtype=s.Array3f, array=s.ArrayXf(1, 2, 3), order='C') == s.Array3f([1, 2, 3]))
    assert dr.all(dr.unravel(dtype=s.Array3f, array=s.ArrayXf(1, 2, 3), order='F') == s.Array3f([1, 2, 3]))
    assert dr.all(dr.unravel(s.Array22f, s.ArrayXf(1, 2, 3, 4), order='C') == s.Array22f([[1, 2], [3, 4]]), axis=None)
    assert dr.all(dr.unravel(s.Array22f, s.ArrayXf(1, 3, 2, 4), order='F') == s.Array22f([[1, 2], [3, 4]]), axis=None)


@pytest.test_arrays('-bool,shape=(3, *)')
def test12_unravel_vec(t, drjit_verbose, capsys):
    vt = dr.value_t(t)

    v0 = vt(1, 2, 3, 4, 5, 6, 7, 7, 7)
    v1 = vt(1, 4, 7, 2, 5, 7, 3, 6, 7)

    assert dr.all(dr.unravel(t, v0, order='C') == t([1, 2, 3], [4, 5, 6], 7), axis=None)
    assert dr.all(dr.unravel(t, v1, order='F') == t([1, 2, 3], [4, 5, 6], 7), axis=None)
    assert dr.all(dr.unravel(t, v1) == t([1, 2, 3], [4, 5, 6], 7), axis=None)
    transcript = capsys.readouterr().out
    assert transcript.count('jit_var_gather') == 9

    assert dr.all(dr.unravel(t, vt()) == t(), axis=None)


    with pytest.raises(TypeError, match="expected array of type"):
        dr.unravel(t, t(), order='C')

    with pytest.raises(RuntimeError, match="order parameter must equal"):
        dr.unravel(t, vt(), order='Q')

    with pytest.raises(RuntimeError, match="not divisible"):
        dr.unravel(t, vt(1, 2, 3, 4))


@pytest.test_arrays('shape=(3, *), float32', 'shape=(*, *), float32')
def test11_reverse(t):
    vt = dr.value_t(t)
    assert dr.all(dr.reverse(vt(1, 2, 3)) == vt(3, 2, 1))
    assert dr.all(dr.reverse([1, 2, 3]) == [3, 2, 1])
    assert dr.all(dr.reverse((1, 2, 3)) == (3, 2, 1))
    assert dr.all(dr.reverse(t(1, 2, [3, 4])) == t([3, 4], 2, 1), axis=None)


@pytest.test_arrays('shape=(*), uint32, is_jit')
def test13_scatter_inc(t):
    # Intense test of the dr.scatter_inc() operation to catch any issues caused
    # by the local pre-accumulation phase). The code below performs random
    # atomic increments to locations in a small array.

    try:
        import numpy as np
    except ImportError:
        pytest.skip('NumPy is not installed')

    np.random.seed(0)
    size = 10000
    for i in range(2, 17):
        index = np.random.randint(0, i, size)
        hist = np.histogram(index, bins=np.arange(i+1))[0]
        assert hist.sum() == size
        counter = dr.zeros(t, i)
        offset = dr.scatter_inc(counter, t(index))
        dr.eval(offset)
        assert np.all(np.array(counter) == hist)
        a = np.column_stack((index, np.array(offset)))
        for j in range(i):
            g = a[a[:, 0] == j, 1]
            g = np.sort(g)
            assert len(g) == hist[j]
            assert np.all(g == np.arange(len(g)))


@pytest.test_arrays('uint32,shape=(*),jit')
def test14_out_of_bounds_gather(t, capsys):
    buf = dr.opaque(t, 0, 10)
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        dr.eval(dr.gather(t, buf, dr.arange(t, 11)))
    transcript = capsys.readouterr().err
    assert "out-of-bounds read from position 10 in an array of size 10." in transcript


@pytest.test_arrays('uint32,shape=(*),jit')
def test14_out_of_bounds_scatter(t, capsys):
    buf = dr.opaque(t, 0, 10)
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        dr.scatter(buf, index=dr.arange(t, 11), value=5)
        dr.eval()
    transcript = capsys.readouterr().err
    assert "out-of-bounds write to position 10 in an array of size 10." in transcript

@pytest.test_arrays('float,-float16,shape=(*),jit')
def test15_scatter_add_kahan(t):
    buf1 = dr.zeros(t, 2)
    buf2 = dr.zeros(t, 2)
    buf3 = dr.zeros(t, 2)
    ti = dr.uint32_array_t(t)
    dr.scatter_add_kahan(
        buf1,
        buf2,
        t(1, dr.epsilon(t), dr.epsilon(t)),
        dr.full(ti, 1, 3)
    )
    with dr.scoped_set_flag(dr.JitFlag.ScatterReduceLocal, False):
        dr.scatter_add(
            buf3,
            t(1, dr.epsilon(t), dr.epsilon(t)),
            dr.full(ti, 1, 3)
        )
    assert dr.all(buf1 + buf2 == [0, 1+dr.epsilon(t)*2])
    assert dr.all(buf3 == [0, 1])


@pytest.test_arrays('int32,shape=(*)')
def test16_meshgrid(t):
    assert dr.all(dr.meshgrid(t(1, 2), indexing='ij') == t(1, 2))
    assert dr.all(dr.meshgrid(t(1, 2), indexing='xy') == t(1, 2))

    a, b = dr.meshgrid(t(1, 2), t(3, 4, 5))
    assert dr.all(a == t(1, 2, 1, 2, 1, 2)) and dr.all(b == t(3, 3, 4, 4, 5, 5))

    a, b = dr.meshgrid(t(1, 2), t(3, 4, 5), indexing='ij')
    assert dr.all(a == t(1, 1, 1, 2, 2, 2)) and dr.all(b == t(3, 4, 5, 3, 4, 5))

    a, b, c = dr.meshgrid(t(1, 2), t(3, 4, 5), t(5, 6), indexing='xy')
    assert dr.all(a == t(1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2)) and \
           dr.all(b == t(3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5)) and \
           dr.all(c == t(5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6))

    a, b, c = dr.meshgrid(t(1, 2), t(3, 4, 5), t(5, 6), indexing='ij')
    assert dr.all(a == t(1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2)) and \
           dr.all(b == t(3, 3, 4, 4, 5, 5, 3, 3, 4, 4, 5, 5)) and \
           dr.all(c == t(5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6))

    # Ensure consistency with NumPy
    np = pytest.importorskip("numpy")

    a, b = dr.meshgrid(t(1, 2), t(3, 4, 5))
    a_np, b_np = np.meshgrid((1, 2), (3, 4, 5))
    assert dr.all(a == a_np.ravel()) and dr.all(b == b_np.ravel())
    a, b = dr.meshgrid(t(1, 2), t(3, 4, 5), indexing='ij')
    a_np, b_np = np.meshgrid((1, 2), (3, 4, 5), indexing='ij')
    assert dr.all(a == a_np.ravel()) and dr.all(b == b_np.ravel())

    a, b, c = dr.meshgrid(t(1, 2), t(3, 4, 5), t(5, 6))
    a_np, b_np, c_np = np.meshgrid((1, 2), (3, 4, 5), t(5, 6))
    assert dr.all(a == a_np.ravel()) and dr.all(b == b_np.ravel()) and dr.all(c == c_np.ravel())
    a, b, c = dr.meshgrid(t(1, 2), t(3, 4, 5), t(5, 6), indexing='ij')
    a_np, b_np, c_np = np.meshgrid((1, 2), (3, 4, 5), t(5, 6), indexing='ij')
    assert dr.all(a == a_np.ravel()) and dr.all(b == b_np.ravel()) and dr.all(c == c_np.ravel())

@pytest.test_arrays('int32,shape=(*)')
def test17_slice(t):
    v = t([1,2,3])
    v2 = dr.slice(v, 2)
    assert type(v2) is int and v2 == 3

    v2 = dr.slice(v, [1, 2])
    assert type(v2) is t and dr.all(v2 == [2, 3])

    # No-op for non-array types
    v = [1,2,3]
    v2 = dr.slice(v, 2)
    assert type(v2) is list and v2 == v

    # Pytree traversal
    v = [t(1,2,3), t(3,4,5)]
    v2 = dr.slice(v, 2)
    assert type(v2) is list and len(v2) == 2 and type(v2[0]) is int and type(v2[1]) is int
    assert v2[0] == 3 and v2[1] == 5

    # Dimensionality reduction

    if dr.is_jit_v(t):
        import sys
        mod = sys.modules[t.__module__]
        v = mod.Array3i([[1,2], [3, 4], [5]])
        v2 = dr.slice(v, 1)
        assert type(v2) is dr.scalar.Array3i
        assert dr.all(v2 == dr.scalar.Array3i(2, 4, 5))

        v2 = dr.slice(v, [1,0])
        assert type(v2) is mod.Array3i
        ref = mod.Array3i([2, 1], [4, 3], [5, 5])
        assert dr.all(v2 == ref, axis=None)

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

@pytest.mark.parametrize('op',
    [int(dr.ReduceOp.Add), int(dr.ReduceOp.Min), int(dr.ReduceOp.Max),
     int(dr.ReduceOp.And), int(dr.ReduceOp.Or)])
@pytest.test_arrays('-bool,jit,-diff,shape=(*)')
def test_scatter_reduce(t, op):
    import sys
    mod = sys.modules[t.__module__]
    op = dr.ReduceOp(op)
    size = 100000

    if op == dr.ReduceOp.And or \
       op == dr.ReduceOp.Or:
        if dr.is_float_v(t):
            return
        size = 100

    backend = dr.backend_v(t)
    tp = dr.type_v(t)

    if backend == dr.JitBackend.LLVM and \
       (op == dr.ReduceOp.Min or
        op == dr.ReduceOp.Max) and \
       versiontuple(dr.detail.llvm_version()) < (15, 0, 0):
        return # unsupported in older LLVM versions

    if backend == dr.JitBackend.CUDA and \
       (tp == dr.VarType.Float32 or
        tp == dr.VarType.Float64) and \
       (op == dr.ReduceOp.Min or
        op == dr.ReduceOp.Max):
        return # unsupported in hardware

    if tp == dr.VarType.Float16:
        # Support for float16 atomics is still kind of spotty
        size = 100

        if backend == dr.JitBackend.LLVM:
            # Don't test float16 LLVM atomics on older LLVM versions
            if versiontuple(dr.detail.llvm_version()) < (16, 0, 0) or \
               op != dr.ReduceOp.Add:
                return

        if backend == dr.JitBackend.CUDA:
            ccap = dr.detail.cuda_compute_capability()

            if op == dr.ReduceOp.Min or op == dr.ReduceOp.Max and ccap < 90:
                return

    identity = dr.detail.reduce_identity(backend, tp, op)

    for k in range(0, 10):
        k = 2**k

        rng = mod.PCG32(size)
        j = t(rng.next_uint32())
        i = rng.next_uint32_bounded(k)
        l = dr.arange(t, size)
        if dr.type_v(t) == dr.VarType.Float16:
            j = dr.full(t, 1, size)

        buf_1 = dr.full(t, identity[0], k)
        dr.scatter_reduce(op, buf_1, index=i, value=j, mode=dr.ReduceMode.Direct)
        dr.eval(buf_1)

        #  dr.set_flag(dr.JitFlag.PrintIR ,True)
        buf_2 = dr.full(t, identity[0], k)
        dr.scatter_reduce(op, buf_2, index=i, value=j, mode=dr.ReduceMode.Expand)
        dr.eval(buf_2)

        if dr.is_float_v(t):
            assert dr.allclose(buf_1, buf_2)
        else:
            assert dr.all(buf_1 == buf_2)

        if k == 1:
            v = None
            if op == dr.ReduceOp.Add:
                v = dr.sum(j)
            elif op == dr.ReduceOp.Min:
                v = dr.min(j)
            elif op == dr.ReduceOp.Max:
                v = dr.max(j)
            if v is not None:
                if dr.is_float_v(t):
                    assert dr.allclose(buf_1, v)
                else:
                    assert dr.all(buf_1 == v)
