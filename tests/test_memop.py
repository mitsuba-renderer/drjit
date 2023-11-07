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
    with dr.scoped_set_flag(dr.JitFlag.AtomicReduceLocal, False):
        dr.scatter_add_kahan(
            buf1,
            buf2,
            t(1, dr.epsilon(t), dr.epsilon(t)),
            dr.full(ti, 1, 3)
        )
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
