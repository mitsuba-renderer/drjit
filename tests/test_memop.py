import drjit as dr
import pytest
import sys
from dataclasses import dataclass
import re

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
        def __init__(self, a: t = t()):
            self.a = a

    s = MyStruct(x)
    r = dr.gather(MyStruct, s, i)
    assert type(r) is MyStruct
    assert dr.all(r.a == t([2, 1]))

    @dataclass
    class MyDataclass:
        a : t

    s = MyDataclass(x)
    r = dr.gather(MyDataclass, s, i)
    assert type(r) is MyDataclass
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

    s = MyStruct(x)
    s.a = dr.zeros(t, 4)
    dr.scatter(
        s,
        MyStruct(t(1, 2)),
        (1, 0),
        (True, False)
    )
    assert dr.all(s.a == [0, 1, 0, 0])

    @dataclass
    class MyDataclass:
        a : t

    s = MyDataclass(x)
    s.a = dr.zeros(t, 4)
    dr.scatter(
        s,
        MyDataclass(t(1, 2)),
        (1, 0),
        (True, False)
    )
    assert dr.all(s.a == [0, 1, 0, 0])


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

    # PyTree traversal
    v = [t(1,2,3), t(3,4,5)]
    v2 = dr.slice(v, 2)
    assert type(v2) is list and len(v2) == 2 and type(v2[0]) is int and type(v2[1]) is int
    assert v2[0] == 3 and v2[1] == 5

    # Dimensionality reduction

    if dr.is_jit_v(t):
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
    [dr.ReduceOp.Add, dr.ReduceOp.Min, dr.ReduceOp.Max,
     dr.ReduceOp.And, dr.ReduceOp.Or])
@pytest.test_arrays('-bool,jit,-diff,shape=(*)')
def test18_scatter_reduce(t, op):
    mod = sys.modules[t.__module__]
    size = 100000

    if dr.type_v(t) == dr.VarType.Float16:
        size = 100

    if not dr.detail.can_scatter_reduce(t, op):
        pytest.skip(f"Unsupported scatter combination: backend={dr.backend_v(t)}, type={dr.type_v(t)}, op={op}")
    identity = dr.detail.reduce_identity(t, op)

    for k in range(0, 10):
        k = 2**k

        rng = mod.PCG32(size)
        j = t(rng.next_uint32())
        i = rng.next_uint32_bounded(k)
        if dr.type_v(t) == dr.VarType.Float16:
            j = dr.full(t, 1, size)

        buf_1 = dr.full(t, identity[0], k)
        dr.scatter_reduce(op, buf_1, index=i, value=j, mode=dr.ReduceMode.Direct)
        dr.eval(buf_1)

        buf_2 = dr.full(t, identity[0], k)
        dr.scatter_reduce(op, buf_2, index=i, value=j, mode=dr.ReduceMode.Local)
        dr.eval(buf_2)

        if dr.is_float_v(t):
            assert dr.allclose(buf_1, buf_2)
        else:
            assert dr.all(buf_1 == buf_2)

        if dr.backend_v(t) == dr.JitBackend.LLVM:
            buf_3 = dr.full(t, identity[0], k)
            dr.scatter_reduce(op, buf_3, index=i, value=j, mode=dr.ReduceMode.Expand)
            dr.eval(buf_3)

            if dr.is_float_v(t):
                assert dr.allclose(buf_1, buf_3)
            else:
                assert dr.all(buf_1 == buf_3)

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

    np = pytest.importorskip("numpy")
    UInt32 = dr.uint32_array_t(t)
    perm = UInt32(np.random.permutation(size))

    rng = mod.PCG32(size)
    j = t(rng.next_uint32())
    if dr.type_v(t) == dr.VarType.Float16:
        j = dr.full(t, 1, size)
    buf_1 = dr.full(t, identity[0], size)
    dr.scatter_reduce(op, buf_1, index=perm, value=j, mode=dr.ReduceMode.NoConflicts)

    buf_2 = dr.full(t, identity[0], size)
    dr.scatter_reduce(op, buf_2, index=perm, value=j, mode=dr.ReduceMode.Direct)

    if dr.is_float_v(t):
        assert dr.allclose(buf_1, buf_2)
    else:
        assert dr.all(buf_1 == buf_2)


@pytest.test_arrays('jit,tensor,float32')
def test19_reshape_tensor(t):
    value = dr.arange(t, 6)
    value2 = dr.reshape(dtype=t, value=value, shape=(3, -1))
    value3 = dr.reshape(dtype=t, value=value, shape=(-1, 2))
    ref = t([[0, 1], [2, 3], [4, 5]])
    assert dr.all(value2 == ref, axis=None)
    assert dr.all(value3 == ref, axis=None)

    with pytest.raises(RuntimeError, match="only a single 'shape' entry may be equal to -1"):
        dr.reshape(dtype=t, value=value, shape=(-1, -1))

    with pytest.raises(RuntimeError, match="cannot infer a compatible shape"):
        dr.reshape(dtype=t, value=value, shape=(4, -1))

    with pytest.raises(RuntimeError, match=r"mismatched array sizes \(input: 6, target: 20\)"):
        dr.reshape(dtype=t, value=value, shape=(4, 5))

    pytree_in = { 'a' : (value, )}
    pytree_out = dr.reshape(dtype=dict, value=pytree_in, shape=(3, -1))
    assert type(pytree_out) is dict and len(pytree_out) == 1 and 'a' in pytree_out
    val = pytree_out["a"]
    assert type(val) is tuple and len(val) == 1
    val = val[0]
    assert type(val) is t and dr.all(val == value2, axis=None)


@pytest.test_arrays('jit,float32,shape=(2, *)')
def test20_reshape_nested(t):
    mod = sys.modules[t.__module__]
    t2 = mod.Array3f

    value = t([1, 2, 3], [4, 5, 6])
    value_c = dr.reshape(dtype=t2, value=value, shape=(3, -1), order='C')
    value_f = dr.reshape(dtype=t2, value=value, shape=(3, -1), order='F')
    value_a = dr.reshape(dtype=t2, value=value, shape=(3, -1), order='A')

    assert dr.all(value_a == value_f, axis=None)
    assert dr.all(value_c == t2([1, 2], [3, 4], [5, 6]), axis=None)
    assert dr.all(value_f == t2([1, 5], [4, 3], [2, 6]), axis=None)

    with pytest.raises(RuntimeError, match="only a single 'shape' entry may be equal to -1"):
        dr.reshape(dtype=t2, value=value, shape=(-1, -1))

    with pytest.raises(RuntimeError, match="cannot infer a compatible shape"):
        dr.reshape(dtype=t2, value=value, shape=(4, -1))

    with pytest.raises(RuntimeError, match=r"mismatched array sizes \(input: 6, target: 20\)"):
        dr.reshape(dtype=t2, value=value, shape=(4, 5))


@pytest.test_arrays('jit,float32,shape=(2, *)')
def test21_reshape_shrink(t):
    t2 = dr.value_t(t)
    v = dr.arange(t2, 10)
    dr.eval(v)
    vs = dr.reshape(t2, v, 5, shrink=True)
    assert len(vs) == 5
    assert dr.all(vs == [0, 1, 2, 3, 4])

    v = t(v, v)
    vs = dr.reshape(t, v, 5, shrink=True)
    assert len(vs) == 2 and len(vs[0]) == 5 and len(vs[1]) == 5
    assert dr.all(vs[0] == [0, 1, 2, 3, 4])


@pytest.test_arrays('jit,float32,shape=(*)')
def test22_tile(t):
    x = t(1, 2)
    y = dr.tile(x, 3)
    assert dr.all(y == [1, 2, 1, 2, 1, 2])

@pytest.test_arrays('jit,float32,shape=(*)')
def test22_repeat(t):
    x = t(1, 2)
    y = dr.repeat(x, 3)
    assert dr.all(y == [1, 1, 1, 2, 2, 2])

@pytest.test_arrays('shape=(*),-bool,-int8')
def test23_block_sum(t):
    x = t(1, 2, 3, 4, 5, 6)
    assert dr.all(dr.block_sum(x, 1) == x)
    assert dr.all(dr.block_sum(x, 2) == [3, 7, 11])
    assert dr.all(dr.block_sum(x, 3) == [6, 15])
    assert dr.all(dr.block_sum(x, 4) == [10, 11])
    assert dr.all(dr.block_sum(x, 5) == [15, 6])
    assert dr.all(dr.block_sum(x, 6) == [21])

@pytest.test_arrays('shape=(*),-bool,-int8')
def test24_block_prefix_sum(t):
    x = t(1, 2, 3, 4, 5, 6)
    assert dr.all(dr.block_prefix_sum(x, 1) == [0, 0, 0, 0, 0, 0])
    assert dr.all(dr.block_prefix_sum(x, 2) == [0, 1, 0, 3, 0, 5])
    assert dr.all(dr.block_prefix_sum(x, 3) == [0, 1, 3, 0, 4, 9])
    assert dr.all(dr.block_prefix_sum(x, 4) == [0, 1, 3, 6, 0, 5])
    assert dr.all(dr.block_prefix_sum(x, 5) == [0, 1, 3, 6, 10, 0])
    assert dr.all(dr.block_prefix_sum(x, 6) == [0, 1, 3, 6, 10, 15])

@pytest.mark.parametrize('op',
    [dr.ReduceOp.Add, dr.ReduceOp.Min, dr.ReduceOp.Max,
     dr.ReduceOp.And, dr.ReduceOp.Or])
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
@pytest.test_arrays('shape=(*), uint32, jit')
def test25_block_reduce_intense(t, op):
    size = 4096*1024
    mod = sys.modules[t.__module__]
    rng = mod.PCG32(size)

    value = rng.next_uint32()
    dr.eval(value)

    for i in range(0, 23):
        block_size = 1 << i
        sum_1 = dr.block_reduce(
            op=dr.ReduceOp(op),
            value=value,
            block_size=block_size,
            mode='evaluated'
        )
        sum_2 = dr.block_reduce(
            op=dr.ReduceOp(op),
            value=value,
            block_size=block_size,
            mode='symbolic'
        )
        assert dr.all(sum_1 == sum_2)


@pytest.mark.parametrize('variant', [0, 1])
@pytest.test_arrays('diff, float32, shape=(*)')
def test26_elide_scatter(t, variant):
    # Test that scatters are not performed when their result is not used
    UInt = dr.uint32_array_t(t)
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        v = dr.arange(t, 1000)
        dr.enable_grad(v)

        out = dr.zeros(t, 1000)
        dr.scatter(out, index=dr.arange(UInt, 1000), value=v)

        dr.backward_from(out)
        if variant == 0:
            del out
        assert dr.all(v.grad == 1)
        dr.eval()

    hist = dr.kernel_history((dr.KernelType.JIT,))
    if variant == 0:
        assert len(hist) == 0
    else:
        ir = hist[0]['ir'].getvalue()
        if dr.backend_v(t) is dr.JitBackend.CUDA:
            assert ir.count('st.global.b32') == 1
        else:
            assert ir.count('call void @llvm.masked.scatter') == 1


@pytest.mark.parametrize('variant', [0, 1])
@pytest.test_arrays('diff, float32, shape=(*)')
def test27_elide_scatter_in_call(t, variant):
    # Test that scatters are not performed when their result is not used
    UInt = dr.uint32_array_t(t)
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        v = dr.arange(t, 1000)
        i = dr.arange(UInt, 1000)
        k = dr.opaque(UInt, 0, 1000)
        dr.enable_grad(v)
        out = dr.zeros(t, 1000)

        def f(i,v):
            dr.scatter(out, index=i, value=v)

        dr.switch(k, (f,), i, v)
        if variant == 0:
            del out
        dr.eval()

    hist = dr.kernel_history((dr.KernelType.JIT,))
    assert len(hist) == 1
    ir = hist[0]['ir'].getvalue()
    if dr.backend_v(t) is dr.JitBackend.CUDA:
        assert ir.count('st.global.b32') == variant
    else:
        assert ir.count('call void @llvm.masked.scatter') == variant

@pytest.test_arrays('-bool, -diff, shape=(*)')
def test28_scalar_reductions(t):
    x = dr.full(t, 3, 4)
    assert dr.sum(x) == 12
    assert dr.prod(x) == 81
    assert dr.all(dr.block_reduce(dr.ReduceOp.Add, x, 2) == [6, 6])
    assert dr.all(dr.block_reduce(dr.ReduceOp.Mul, x, 2) == [9, 9])

@pytest.mark.parametrize('psize', [2, 4, 8, 16])
@pytest.test_arrays('-diff, jit, shape=(*, *)')
def test29_packet_gather(t, psize):
    mod = sys.modules[t.__module__]

    size = 1024
    pcg = mod.PCG32(size*psize)
    tp = dr.type_v(t)
    vt = dr.value_t(t)

    if tp == dr.VarType.Bool:
        buf = pcg.next_uint32() & 1 == 0
    elif tp in (dr.VarType.Float16, dr.VarType.Float32, dr.VarType.Float64):
        buf = vt(pcg.next_float32())
    elif tp in (dr.VarType.Int32, dr.VarType.UInt32, dr.VarType.UInt8, dr.VarType.Int8):
        buf = vt(pcg.next_uint32())
    elif tp in (dr.VarType.Int64, dr.VarType.UInt64):
        buf = vt(pcg.next_uint64())
    else:
        raise Exception("Unknown type")

    dr.eval(buf)
    i = mod.PCG32(1024*1024).next_uint32_bounded(size)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, False):
        y1 = dr.gather(t, source=buf, index=i, active = i & 128 != 0, shape=(psize, size))
        dr.eval(y1)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, True):
        y2 = dr.gather(t, source=buf, index=i, active = i & 128 != 0, shape=(psize, size))
        dr.eval(y2)

    assert dr.all(y1 == y2, axis=None)


@pytest.mark.parametrize('psize', [2, 4, 8, 16])
@pytest.test_arrays('-diff, jit, shape=(*, *)')
def test30_packet_scatter(t, psize):
    np = pytest.importorskip("numpy")
    mod = sys.modules[t.__module__]

    tp = dr.type_v(t)
    vt = dr.value_t(t)
    UInt32 = dr.uint32_array_t(vt)

    value = []
    size = 1024
    pcg = mod.PCG32(size)
    for _ in range(psize):
        if tp == dr.VarType.Bool:
            v = pcg.next_uint32() & 1 == 0
        elif tp in (dr.VarType.Float16, dr.VarType.Float32, dr.VarType.Float64):
            v = vt(pcg.next_float32())
        elif tp in (dr.VarType.Int32, dr.VarType.UInt32, dr.VarType.UInt8, dr.VarType.Int8):
            v = vt(pcg.next_uint32())
        elif tp in (dr.VarType.Int64, dr.VarType.UInt64):
            v = vt(pcg.next_uint64())
        else:
            raise Exception("Unknown type")
        value.append(v)

    value_arr = t(value)
    perm = UInt32(np.random.permutation(1024))
    target_1 = dr.empty(vt, size*psize)
    target_2 = dr.empty(vt, size*psize)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, False):
        dr.scatter(target_1, value_arr, perm)
    dr.eval(target_1)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, True):
        dr.scatter(target_2, value_arr, perm)
    dr.eval(target_2)

    assert dr.all(target_1 == target_2)

@pytest.mark.parametrize('psize', [2, 4, 8, 16])
@pytest.test_arrays('-diff, jit, int, shape=(*, *), -int8')
def test31_packet_scatter_add(t, psize):
    np = pytest.importorskip("numpy")
    mod = sys.modules[t.__module__]

    tp = dr.type_v(t)
    vt = dr.value_t(t)
    UInt32 = dr.uint32_array_t(vt)

    value = []
    size = 1024
    pcg = mod.PCG32(size)
    for _ in range(psize):
        if tp in (dr.VarType.Int32, dr.VarType.UInt32):
            v = vt(pcg.next_uint32())
        elif tp in (dr.VarType.Int64, dr.VarType.UInt64):
            v = vt(pcg.next_uint64())
        else:
            raise Exception("Unknown type")
        value.append(v)

    value_arr = t(value)
    perm = UInt32(np.random.permutation(1024))
    target_1 = dr.zeros(vt, size*psize)
    target_2 = dr.zeros(vt, size*psize)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, False):
        dr.scatter_add(target_1, value_arr, perm)
    dr.eval(target_1)

    with dr.scoped_set_flag(dr.JitFlag.PacketOps, True):
        dr.scatter_add(target_2, value_arr, perm)
    dr.eval(target_2)

    assert dr.all(target_1 == target_2)


@pytest.test_arrays('jit, -bool, -quat, -diff, shape=(4, *)')
def test32_packet_ravel_unravel(t, capsys, drjit_verbose):
    # Test that packet memory operations are used in ravel/unravel

    q = t([0,1],[0,1],[1,0],[1,0])
    q2 = dr.ravel(q)
    dr.eval(q2)
    q3 = dr.unravel(t, q2)
    dr.eval(q3)
    assert dr.all(q == q3, axis=None)
    transcript = capsys.readouterr().out
    assert transcript.count('jit_var_gather_packet') != 0
    assert transcript.count('jit_var_scatter_packet') != 0


@pytest.mark.parametrize('mode', [dr.ReduceMode.Local, dr.ReduceMode.Expand,
    dr.ReduceMode.Direct, dr.ReduceMode.Auto])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test33_scatter_reduce_loop(t, mode, optimize):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = dr.ones(t, 64)
        v = dr.zeros(t, 64)
        unused = t(0, 0) # Dummy variable to cause 2 x recording

        while dr.hint(i < 10, exclude=[v], include=[unused]):
            i += 1
            dr.scatter_add(target=v, index=0, value=1, mode=mode)

        assert v[0] == 64 * 9
        assert dr.all(i == 10)

@pytest.mark.parametrize("order", ["C", "F"])
@pytest.test_arrays("is_jit, shape=(*)")
def test34_ravel_builtin(t, order):

    # Get value with builtin type of t
    x = dr.ones(t, 1)[0]

    y = dr.ravel(x, order = order)

    assert type(x) is type(y)
    assert x == y


@pytest.test_arrays("is_tensor, float32")
def test35_take(t):
    np = pytest.importorskip("numpy")
    np.random.seed(0)
    for ndim in range(1, 5):
        shape = tuple(np.random.randint(2, 6, size=ndim))
        arr = np.float32(np.random.randn(*shape))
        arr2 = t(arr)

        for axis in range(ndim):
            axis_size = shape[axis]
            # Beginning, middle, end
            test_indices = (0, axis_size // 2, axis_size - 1)

            for index in test_indices:
                test1 = np.take(arr, index, axis)
                test2 = dr.take(arr2, int(index), axis)

                assert np.all(test2.numpy() == test1)


@pytest.test_arrays("is_tensor, float32")
def test36_take_interp(t):
    np = pytest.importorskip("numpy")
    np.random.seed(0)
    for ndim in range(1, 5):
        shape = tuple(np.random.randint(2, 6, size=ndim))
        arr = np.float32(np.random.randn(*shape))
        arr2 = t(arr)

        for axis in range(ndim):
            axis_size = shape[axis]
            np.random.rand()

            for pos in np.float32(np.random.rand(3)*(axis_size-1)):
                v0 = np.take(arr, int(pos), axis)
                v1 = np.take(arr, int(pos)+1, axis)
                w1 = pos - int(pos)
                w0 = 1-w1
                test1 = v0*w0 + v1*w1
                test2 = dr.take_interp(arr2, float(pos), axis)

                assert np.allclose(test2.numpy(), test1)

@pytest.test_arrays("is_tensor, float32")
def test37_take_multiple(t):
    np = pytest.importorskip("numpy")
    Index = dr.uint32_array_t(dr.array_t(t))
    arr = np.float32(np.random.randn(3, 4, 5))
    arr2 = t(arr)

    test1 = np.take(arr, (2, 1), 1)
    test2 = dr.take(arr2, Index(2, 1), 1)
    assert np.all(test1 == test2.numpy(), axis=None)

@pytest.mark.parametrize("packet_size", [1, 2, 3, 4, 5, 6, 12, 16])
@pytest.mark.parametrize("reduce_op", ["Add", "Mul", "Max", "Min"])
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
@pytest.test_arrays("is_jit, float, shape=(*)")
@pytest.mark.parametrize("force_optix", [True, False])
def test35_scatter_packet_reduce(t, reduce_op, packet_size, force_optix):
    """
    Tests that packeted scatter reduce operations behave correctly.
    """

    tp = dr.type_v(t)

    if (
        dr.backend_v(t) == dr.JitBackend.LLVM
        and dr.detail.llvm_version()[0] < 16
        and tp == dr.VarType.Float16
    ):
        pytest.skip("Half precision atomics too spotty on LLVM before v16.0.0")

    if (
        dr.backend_v(t) == dr.JitBackend.CUDA
        and force_optix
        and tp == dr.VarType.Float16
        and reduce_op != "Add"
    ):
        pytest.skip(
            "Only scatter add reductions are supported for Float16 types on the OptiX backend"
        )

    mod = sys.modules[t.__module__]
    if tp == dr.VarType.Float16:
        ArrayXf = mod.ArrayXf16
    elif tp == dr.VarType.Float32:
        ArrayXf = mod.ArrayXf
    elif tp == dr.VarType.Float64:
        ArrayXf = mod.ArrayXf64

    n = 3

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        with dr.scoped_set_flag(dr.JitFlag.ForceOptiX, force_optix):

            target = dr.zeros(t, n * packet_size)
            index = mod.UInt32(0, 1, 1, 2)
            src = dr.rng().uniform(ArrayXf, (packet_size, dr.width(index)))

            op = getattr(dr.ReduceOp, reduce_op)

            dr.scatter_reduce(op, target, src, index)

            dr.kernel_history_clear()
            dr.eval(target)
            history = dr.kernel_history((dr.KernelType.JIT,))

    # Manually construct a reference, by scattering into a python list.
    ref = dr.zeros(t, n * packet_size)
    for i in range(dr.width(index)):
        for j in range(packet_size):
            if reduce_op == "Add":
                def op(x, y):
                    return x + y
            if reduce_op == "Mul":
                def op(x, y):
                    return x * y
            if reduce_op == "Max":
                def op(x, y):
                    return max(x, y)
            if reduce_op == "Min":
                def op(x, y):
                    return min(x, y)

            ref[index[i] * packet_size + j] = op(
                ref[index[i] * packet_size + j], src[j][i]
            )

    assert dr.allclose(target, ref)

    # Test that we are actually using vector instructions on CUDA and LLVM
    ir = history[0]["ir"].getvalue()
    if dr.backend_v(t) is dr.JitBackend.CUDA:
        compute_capability = dr.detail.cuda_compute_capability()
        if compute_capability >= 90 and not force_optix:
            if tp == dr.VarType.Float16:
                n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
                n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]
                assert ir.count(f"red.global.v{n_regs}.f16.{reduce_op.lower()}.noftz") == n_inst
            if tp == dr.VarType.Float32:
                n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 4}[packet_size]
                n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 4}[packet_size]
                assert ir.count(f"red.global.v{n_regs}.f32.{reduce_op.lower()}") == n_inst
        else:
            if tp == dr.VarType.Float16:
                n_inst = {1: 1, 2: 1, 3: 3, 4: 2, 5: 5, 6: 3, 12: 6, 16: 8}[packet_size]
                assert ir.count("red.global.add.noftz.f16x2") == n_inst
    elif dr.backend_v(t) is dr.JitBackend.LLVM and reduce_op == "Add":
        # Compute maximum supported vector width for this architecture
        target_features = re.search('"target-features"=".*"', ir).string
        if "+see4.2" in target_features:
            llvm_vector_width = 4
        if "+avx" in target_features:
            llvm_vector_width = 8
        if "+avx512vl" in target_features:
            llvm_vector_width = 16
        if "+neon" in target_features:
            llvm_vector_width = 4

        if llvm_vector_width == 4:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 4}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 4}[packet_size]
        elif llvm_vector_width == 8:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]
        elif llvm_vector_width == 16:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]

        type_str = {
            dr.VarType.Float16: "f16",
            dr.VarType.Float32: "f32",
            dr.VarType.Float64: "f64",
        }[tp]

        assert ir.count(f"call fastcc void @scatter_add_{n_regs}x{type_str}") == n_inst


@pytest.mark.parametrize("packet_size", [1, 2, 3, 4, 5, 6, 12, 16])
@pytest.test_arrays("is_jit, float, shape=(*)")
@pytest.mark.parametrize("force_optix", [True, False])
def test36_gather_packet(t, packet_size, force_optix):
    """
    Tests that packeted gather operations behave correctly and use vector instructions.
    """
    tp = dr.type_v(t)
    if (
        dr.backend_v(t) == dr.JitBackend.LLVM
        and dr.detail.llvm_version()[0] < 16
        and tp == dr.VarType.Float16
    ):
        pytest.skip("Half precision vectorization too spotty on LLVM before v16.0.0")

    mod = sys.modules[t.__module__]
    if tp == dr.VarType.Float16:
        ArrayXf = mod.ArrayXf16
    elif tp == dr.VarType.Float32:
        ArrayXf = mod.ArrayXf
    elif tp == dr.VarType.Float64:
        ArrayXf = mod.ArrayXf64

    # Make n divisible by packet_size to avoid gather packet errors
    n = 16 * 16  # 256 - divisible by all packet sizes

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        with dr.scoped_set_flag(dr.JitFlag.ForceOptiX, force_optix):
            # Create source data, large enough for the largest packet size
            source = dr.arange(t, n)

            # Create indices for gathering - ensure they don't go out of bounds
            max_index = n // packet_size - 1
            index = mod.UInt32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
            index = index % max(1, max_index)  # Ensure indices are within bounds

            # Perform packeted gather
            result = dr.gather(
                ArrayXf, source=source, index=index, shape=(packet_size, dr.width(index))
            )

            dr.kernel_history_clear()
            dr.eval(result)
            history = dr.kernel_history((dr.KernelType.JIT,))

    # Manual verification - gather the same values using regular indexing
    ref = dr.zeros(ArrayXf, (packet_size, dr.width(index)))
    for i in range(dr.width(index)):
        for j in range(packet_size):
            ref[j, i] = source[index[i] * packet_size + j]

    assert dr.allclose(result, ref)

    ir = history[0]["ir"].getvalue()

    if dr.backend_v(t) is dr.JitBackend.CUDA:
        if tp == dr.VarType.Float16:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]
            vec_str = f".v{n_regs//2}" if n_regs > 2 else ""
            assert ir.count(f"ld.global.nc{vec_str}.b32") == n_inst
        elif tp == dr.VarType.Float32:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 4}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 4}[packet_size]
            assert ir.count(f"ld.global.nc.v{n_regs}.b32") == n_inst
        elif tp == dr.VarType.Float64:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 2, 5: 0, 6: 2, 12: 2, 16: 2}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 2, 5: 0, 6: 3, 12: 6, 16: 8}[packet_size]
            assert ir.count(f"ld.global.nc.v{n_regs}.b64") == n_inst

    elif dr.backend_v(t) is dr.JitBackend.LLVM:
        # Compute maximum supported vector width for this architecture
        target_features = re.search('"target-features"=".*"', ir).string
        if "+see4.2" in target_features:
            llvm_vector_width = 4
        if "+avx" in target_features:
            llvm_vector_width = 8
        if "+avx512vl" in target_features:
            llvm_vector_width = 16
        if "+neon" in target_features:
            llvm_vector_width = 4

        if llvm_vector_width == 4:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 4}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 4}[packet_size]
        elif llvm_vector_width == 8:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]
        elif llvm_vector_width == 16:
            n_regs = {1: 0, 2: 2, 3: 0, 4: 4, 5: 0, 6: 2, 12: 4, 16: 8}[packet_size]
            n_inst = {1: 0, 2: 1, 3: 0, 4: 1, 5: 0, 6: 3, 12: 3, 16: 2}[packet_size]

        type_str = {
            dr.VarType.Float16: "f16",
            dr.VarType.Float32: "f32",
            dr.VarType.Float64: "f64",
        }[tp]

        assert len(re.findall(f"call fastcc \\[.*\\] @gather_{n_regs}x{type_str}", ir)) == n_inst


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
@pytest.mark.parametrize("opaque", [True, False])
def test36_scatter_cas(t, opaque):
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target =     t(  1,  2,   3,  4,  5, 6, 7)
    compare =    t(999,  1, 999,  3,  5)
    value =      t( 10, 20,  30, 40, 50)
    index = UInt32(  1,  0,   3,  2,  4)
    mask = dr.full(Mask, True, 5)

    if opaque:
        dr.make_opaque(target, compare, value, index, mask)

    old, swapped = dr.scatter_cas(target, compare, value, index, mask)
    dr.eval(old, swapped)

    assert dr.allclose(old, [2, 1, 4, 3, 5])
    assert dr.all(swapped == [False, True, False, True, True])
    assert dr.allclose(target, [20, 2, 40, 4, 50, 6, 7])


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
@pytest.mark.parametrize("opaque", [True, False])
def test37_scatter_cas_masked(t, opaque):
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target =     t(  1,  2,   3,  4,  5, 6, 7)
    compare =    t(999,  1, 999,  3,  5)
    value =      t( 10, 20,  30, 40, 50)
    index = UInt32(  1,  0,   3,  2,  4)
    mask = Mask(True, True, True, True, False)

    if opaque:
        dr.make_opaque(target, compare, value, index, mask)

    old, swapped = dr.scatter_cas(target, compare, value, index, mask)
    dr.eval(old, swapped)

    assert dr.allclose(old, [2, 1, 4, 3, 0])
    assert dr.all(swapped == [False, True, False, True, False])
    assert dr.allclose(target, [20, 2, 40, 4, 5, 6, 7])


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
def test38_scatter_cas_vcall(t):
    # Test `dr.scatter_cas()` when nested in a vcall
    UInt32 = dr.uint32_array_t(t)

    buffer1 = t(0, 1, 2, 0, 4)
    buffer2 = t(0, 2, 0, 3, 4)

    # `f()` and `g()` will be merged
    def f(idx):
        dr.scatter_cas(buffer1, 0, 1, idx)
    def g(idx):
        dr.scatter_cas(buffer2, 0, 1, idx)
    def h(buffer):
        pass

    funcs = [f, g, h]
    idx = UInt32(0, 0, 1, 1, 2)

    dr.switch(idx, funcs, dr.arange(UInt32, 5))
    dr.eval(buffer1, buffer2)

    assert dr.allclose(buffer1, [1, 1, 2, 0, 4])
    assert dr.allclose(buffer2, [0, 2, 1, 3, 4])


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
@pytest.mark.parametrize("opaque", [True, False])
def test39_scatter_exch(t, opaque):
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target =     t(  1,  2,   3,  4,  5, 6, 7)
    value =      t( 10, 20,  30, 40, 50)
    index = UInt32(  1,  0,   3,  2,  4)
    mask = dr.full(Mask, True, 5)

    if opaque:
        dr.make_opaque(target, value, index, mask)

    old = dr.scatter_exch(target, value, index, mask)
    dr.eval(old)

    assert dr.allclose(old, [2, 1, 4, 3, 5])
    assert dr.allclose(target, [20, 10, 40, 30, 50, 6, 7])


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
@pytest.mark.parametrize("opaque", [True, False])
def test40_scatter_exch_masked(t, opaque):
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target =     t(  1,  2,   3,  4,  5, 6, 7)
    value =      t( 10, 20,  30, 40, 50)
    index = UInt32(  1,  0,   3,  2,  4)
    mask = Mask(True, True, True, False, False)

    if opaque:
        dr.make_opaque(target, value, index, mask)

    old = dr.scatter_exch(target, value, index, mask)
    dr.eval(old)

    assert dr.allclose(old, [2, 1, 4, 0, 0])
    assert dr.allclose(target, [20, 10, 3, 30, 5, 6, 7])


@pytest.test_arrays('float32,jit,shape=(*)', 'uint32,jit,shape=(*)')
def test41_scatter_exch_vcall(t):
    # Test `dr.scatter_exch()` when nested in a vcall
    UInt32 = dr.uint32_array_t(t)

    buffer1 = t(0, 1, 2, 0, 4)
    buffer2 = t(0, 2, 0, 3, 4)

    # `f()` and `g()` will be merged
    def f(idx):
        dr.scatter_exch(buffer1, 8, idx)
    def g(idx):
        dr.scatter_exch(buffer2, 9, idx)
    def h(buffer):
        pass

    funcs = [f, g, h]
    idx = UInt32(0, 0, 1, 1, 2)

    dr.switch(idx, funcs, dr.arange(UInt32, 5))
    dr.eval(buffer1, buffer2)

    assert dr.allclose(buffer1, [8, 8, 2, 0, 4])
    assert dr.allclose(buffer2, [0, 2, 9, 9, 4])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test42_masked_gather_loop(t, mode):
    # Symbolic (reindexed) gathers, should re-apply masks to previous gathers
    # even when the mask is implicit due to a loop

    Int32 = dr.int32_array_t(t)
    Mask = dr.mask_t(t)

    # Note: Use non-power of 2 sizes of array to make sure LLVM default masks work

    buf_0 = t(1, 2, 3, 4, 5, 6, 7, 8)
    mask_1 = (dr.arange(t, 5) % 4) != 0 # False, True, True, True, False
    index_1 = dr.arange(t, 5) * 2
    buf_1 = dr.gather(t, buf_0, index_1, mask_1) # 0, 3, 5, 7, 0

    active = Mask(True, True, False)
    out = dr.zeros(t, 3)
    counter = dr.zeros(t, 3)

    while dr.hint(active, exclude=[buf_1], mode=mode):
        index_2 = t(Int32(0, 2, -1))
        out += dr.gather(t, buf_1, index_2) # 0, 5, 0

        counter += 1
        active = counter == 0

    # Lane 0: outputs 0 because the gather on buf_0 masked it
    # Lane 1: outputs 5 because we lookup index 2 of buf_1, which is index 4 of buf_0
    # Lane 2: outputs 0 because the gather on buf_1 masked it

    assert dr.all(out == [0, 5, 0], axis=None)
