import drjit as dr
import pytest
import sys

def test01_all_any_scalar():
    from drjit.scalar import Array2b, ArrayXb

    assert dr.all(True) == True
    assert dr.all(False) == False
    assert dr.any(True) == True
    assert dr.any(False) == False
    assert dr.any(()) == False
    assert dr.all(()) == True
    assert dr.all((True,)) == True
    assert dr.all((False,)) == False
    assert dr.any((True,)) == True
    assert dr.any((False,)) == False
    assert dr.all([True, True]) == True
    assert dr.all([True, False]) == False
    assert dr.all([False, False]) == False
    assert dr.any([True, True]) == True
    assert dr.any([True, False]) == True
    assert dr.any([False, False]) == False
    assert dr.none([True, True]) == False
    assert dr.none([True, False]) == False
    assert dr.none([False, False]) == True
    assert type(dr.all(Array2b(True, True))) is bool
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, False)) == False
    assert dr.all(Array2b(False, False)) == False
    assert dr.any(Array2b(True, True)) == True
    assert dr.any(Array2b(True, False)) == True
    assert dr.any(Array2b(False, False)) == False
    assert dr.none(Array2b(True, True)) == False
    assert dr.none(Array2b(True, False)) == False
    assert dr.none(Array2b(False, False)) == True
    assert type(dr.all(ArrayXb(True, True))) is bool
    assert dr.all(ArrayXb(True, True)) == True
    assert dr.all(ArrayXb(True, False)) == False
    assert dr.all(ArrayXb(False, False)) == False
    assert dr.any(ArrayXb(True, True)) == True
    assert dr.any(ArrayXb(True, False)) == True
    assert dr.any(ArrayXb(False, False)) == False
    assert dr.none(ArrayXb(True, True)) == False
    assert dr.none(ArrayXb(True, False)) == False
    assert dr.none(ArrayXb(False, False)) == True


# Tests dr.{any/all}[_nested] and implicit conversion to 'bool'
@pytest.test_arrays('shape=(1, *), bool')
def test02_any_all_nested(t):
    t0 = dr.value_t(t)
    v0 = t0([True, False, False])

    v = t(v0)
    assert len(v) == 1 and len(v0) == 3

    v0_all = dr.all(v0)
    v0_any = dr.any(v0)
    v0_none = dr.none(v0)
    assert type(v0_all) is t0 and len(v0_all) == 1 and \
           type(v0_any) is t0 and len(v0_any) == 1 and \
           type(v0_none) is t0 and len(v0_none) == 1

    assert bool(v0_all) == False
    assert bool(v0_any) == True
    assert bool(v0_none) == False

    va = dr.all(v)
    assert type(va) is t0 and len(va) == 3

    van = dr.all(v, axis=None)
    assert type(van) is t0 and len(van) == 1

    with pytest.raises(RuntimeError) as ei:
        dr.all((True, "hello"))

    assert "unsupported operand type(s)" in str(ei.value.__cause__)


@pytest.test_arrays('shape=(1, *)')
def test03_implicit_bool_conversion_failures(t):
    if not dr.is_mask_v(t):
        with pytest.raises(TypeError, match=r'implicit conversion to \'bool\' is only supported for scalar mask arrays.'):
            bool(t()[0])
    else:
        with pytest.raises(RuntimeError, match=r'implicit conversion to \'bool\' requires an array with at most 1 dimension \(this one has 2 dimensions\).'):
            bool(t())
        with pytest.raises(RuntimeError, match=r'implicit conversion to \'bool\' requires an array with at most 1 element \(this one has 2 elements\).'):
            bool(dr.value_t(t)(True, False))


@pytest.test_arrays('shape=(*), float32, jit')
def test04_sum(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.sum(6.0), 6)

    a = dr.sum(m.Float([1, 2, 3]), mode='evaluated')
    assert dr.allclose(a, 6)
    assert type(a) is m.Float

    a = dr.sum(m.Float([1, 2, 3]), mode='symbolic')
    assert dr.allclose(a, 6)
    assert type(a) is m.Float

    a = dr.sum([1.0, 2.0, 3.0])
    assert dr.allclose(a, 6)
    assert type(a) is float

    a = dr.sum(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Float

    a = dr.sum(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Int

    a = dr.sum(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, [6, 9, 12])
    assert type(a) is m.Float

    a = dr.sum(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]), axis=-1)
    assert dr.allclose(a, [[6], [9], [12]])
    assert type(a) is m.ArrayXf

    a = dr.sum(m.Array3f([1, 2, 3], [2, 3, 4], [3, 4, 5]), axis=1)
    assert dr.allclose(a, [[6], [9], [12]])
    assert type(a) is m.Array3f

    a = dr.sum(dr.scalar.Matrix2f([1, 2], [3, 4]), axis=0)
    assert dr.allclose(a, [4, 6])
    assert type(a) is dr.scalar.Array2f

    a = dr.sum(dr.scalar.Matrix2f([1, 2], [3, 4]), axis=1)
    assert dr.allclose(a, [3, 7])
    assert type(a) is dr.scalar.Array2f

    a = dr.sum(dr.scalar.Array2f([3, 7]), axis=0)
    assert dr.allclose(a, 10)
    assert type(a) is float

    with pytest.raises(RuntimeError, match="out-of-bounds axis"):
        a = dr.sum(dr.scalar.Matrix2f([1, 2], [3, 4]), axis=3)

    with pytest.raises(RuntimeError, match="out-of-bounds axis"):
        a = dr.sum(dr.scalar.Matrix2f([1, 2], [3, 4]), axis=-3)


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test05_prod(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.prod(6.0), 6.0)

    a = dr.prod(m.Float([1, 2, 3]))
    assert dr.allclose(a, 6)
    assert type(a) is m.Float

    a = dr.prod([1.0, 2.0, 3.0])
    assert dr.allclose(a, 6)
    assert type(a) is float

    a = dr.prod(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Float

    a = dr.prod(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Int

    a = dr.prod(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, [6, 24, 60])
    assert type(a) is m.Float


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test06_max(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.max(6.0), 6.0)

    a = dr.max(m.Float([1, 2, 3]))
    assert dr.allclose(a, 3)
    assert type(a) is m.Float

    a = dr.max([1.0, 2.0, 3.0])
    assert dr.allclose(a, 3)
    assert type(a) is float

    a = dr.max(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 3)
    assert type(a) is m.Float

    a = dr.max(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 3)
    assert type(a) is m.Int

    a = dr.max(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, [3, 4, 5])
    assert type(a) is m.Float


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test03_min(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.min(6.0), 6.0)

    a = dr.min(m.Float([1, 2, 3]))
    assert dr.allclose(a, 1)
    assert type(a) is m.Float

    a = dr.min([1.0, 2.0, 3.0])
    assert dr.allclose(a, 1)
    assert type(a) is float

    a = dr.min(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 1)
    assert type(a) is m.Float

    a = dr.min(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 1)
    assert type(a) is m.Int

    a = dr.min(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, [1, 2, 3])
    assert type(a) is m.Float


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test04_minimum(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.minimum(6.0, 4.0), 4.0)

    a = dr.minimum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Float

    a = dr.minimum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Float

    a = dr.minimum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Array3f

    a = dr.minimum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Array3f

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 3], [1, 1, 3], [1, 2, 3]])
    assert type(a) is m.ArrayXf

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 1, 1], [2, 1, 2], [3, 3, 3]])
    assert type(a) is m.ArrayXf


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test05_maximum(t):
    m = sys.modules[t.__module__]
    assert dr.allclose(dr.maximum(6.0, 4.0), 6.0)

    a = dr.maximum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Float

    a = dr.maximum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Float

    a = dr.maximum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Array3f

    a = dr.maximum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Array3f

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) is m.ArrayXf

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) is m.ArrayXf


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test06_dot(t):
    m = sys.modules[t.__module__]
    a = dr.dot(m.Float(1, 2, 3), m.Float(2, 1, 1))
    assert dr.allclose(a, 7)
    assert type(a) is m.Float

    a = dr.dot(m.Float(1, 2, 3), m.Float(2))
    assert dr.allclose(a, 12)
    assert type(a) is m.Float

    a = dr.dot([1, 2, 3], [2, 1, 1])
    assert dr.allclose(a, 7)
    assert type(a) is int

    a = dr.dot(m.Array3f(1, 2, 3), m.Array3f([2, 1, 1]))
    assert dr.allclose(a, 7)
    assert type(a) is m.Float

    a = dr.dot(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.ArrayXf([[1, 2, 5], [2, 2, 4], [3, 4, 3]]))
    assert dr.allclose(a, [14, 22, 50])
    assert type(a) is m.Float


@pytest.test_arrays('shape=(*), float, -float64, jit')
def test07_norm(t):
    m = sys.modules[t.__module__]
    a = dr.norm(m.Float(1, 2, 3))
    assert dr.allclose(a, 3.74166)
    assert type(a) is m.Float

    a = dr.norm(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 3.74166)
    assert type(a) is m.Float


@pytest.test_arrays('shape=(*), float32')
def test08_prefix_sum(t):
    m = sys.modules[t.__module__]
    assert dr.all(dr.prefix_sum(t(1, 2, 3)) == t(0, 1, 3))
    assert dr.all(dr.cumsum(t(1, 2, 3)) == t(1, 3, 6))
    assert dr.all(dr.prefix_sum(m.TensorXf([1, 2, 3])) == m.TensorXf([0, 1, 3]))
    assert dr.all(dr.cumsum(m.TensorXf([1, 2, 3])) == m.TensorXf([1, 3, 6]))
    assert dr.all(dr.prefix_sum(m.Array3f(1, 2, 3)) == m.Array3f(0, 1, 3), axis=None)
    assert dr.all(dr.cumsum(m.Array3f(1, 2, 3)) == m.Array3f(1, 3, 6), axis=None)

    # Prefix sum of literals
    x = dr.full(t, 3, 4)
    assert dr.all(dr.prefix_sum(x) == [0, 3, 6, 9])
    assert dr.all(dr.cumsum(x) == [3, 6, 9, 12])


@pytest.test_arrays('shape=(*), bool')
def test09_compress(t):
    a = t(False, True, True, False, False, True, False, True, True)
    i = dr.compress(a)
    assert dr.all(i == [1, 2, 5, 7, 8])
    assert dr.all(dr.compress(t(False, False, False)) == [])
    assert dr.all(dr.compress(t()) == [])


@pytest.test_arrays('shape=(3, *), float32')
def test10_sum_avg_mixed_size(t):
    assert dr.all(dr.sum(t([1,2],2,3)) == [6, 7])
    assert dr.all(dr.sum(t([1,2],2,3), axis=None) == [13])
    assert dr.allclose(dr.mean(t([1,2],2,3)), [2, 7/3])
    assert dr.allclose(dr.mean(t([1,2],2,3), axis=None), [13/6])
    assert dr.mean(3) == 3


@pytest.test_arrays('shape=(*), bool')
def test11_count(t):
    i = dr.uint32_array_t(t)

    assert dr.count(t(True, False)) == i(1)
    assert dr.count(t(True, False, True, True, False, False, False)) == i(3)
    assert dr.count(t()) == i(0)
    assert dr.count(False) == 0
    assert dr.count(True) == 1
    assert dr.count((True, False)) == 1


@pytest.mark.parametrize('op', [dr.ReduceOp.Add, dr.ReduceOp.Max, dr.ReduceOp.Min])
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
@pytest.test_arrays('int, tensor, -is_diff')
def test12_tensor_reduce(t, op):
    def check(y, axis, mode):
        import numpy as np

        #print(f"{axis=}: ", end="")
        ynp = y.numpy()
        if op is dr.ReduceOp.Add:
            a0 = ynp.sum(axis=axis)
        elif op is dr.ReduceOp.Min:
            a0 = ynp.min(axis=axis)
        elif op is dr.ReduceOp.Max:
            a0 = ynp.max(axis=axis)
        a1 = dr.reduce(op, y, axis, mode)
        match = np.all(a1.numpy() == a0, axis=None)
        #print(f"{'OK' if match else 'FAIL'}")


    def check_all(y):
        import itertools

        axes = []
        for i in range(0, y.ndim + 1):
            axes.extend(list(itertools.combinations(range(y.ndim), i)))

        for a in axes:
            check(y, a, 'symbolic')
            check(y, a, 'evaluated')


    if True:
        x = dr.arange(t, 256)
        y = dr.reshape(t, x, (4, 4, 4, 4))

        check_all(y)

    if True:
        x = dr.arange(t, 1155)
        y = dr.reshape(t, x, (3, 5, 7, 11))

        check_all(y)

@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('exclusive', [False, True])
@pytest.test_arrays('tensor, uint32, jit, -diff')
def test13_test_prefix_reduction(t, reverse, exclusive):
    try:
        import numpy as np
    except:
        pytest.skip(reason="NumPy is required")

    def randint(shape):
        np.random.seed(0)
        X = np.random.randint(low=0, high=0xFFFFFFFF, size=np.prod(shape), dtype=np.uint32)
        return np.reshape(X, shape)

    def test_red(shape, axis):
        X = randint(shape)

        Xt = t(X)
        if reverse:
            Xr1 = np.flip(np.cumsum(np.flip(X, axis), axis=axis), axis)
        else:
            Xr1 = np.cumsum(X, axis=axis)

        if exclusive:
            Xr1 = np.swapaxes(Xr1, 0, axis)
            if reverse:
                Xr1[0:-1, ...] =  Xr1[1:, ...]
                Xr1[-1, ...] = 0
            else:
                Xr1[1:, ...] =  Xr1[0:-1, ...]
                Xr1[0, ...] = 0
            Xr1 = np.swapaxes(Xr1, 0, axis)

        Xr2 = dr.prefix_reduce(op=dr.ReduceOp.Add, value=Xt,
                               axis=axis, exclusive=exclusive,
                               reverse=reverse)

        assert dr.all(t(Xr1) == Xr2, axis=None)

    test_red((1, ), 0)
    test_red((17, ), 0)
    test_red((1, 1), 0)
    test_red((1, 1), 1)
    test_red((13, 17), 1)
    test_red((13, 17), 0)
    test_red((9, 5, 7), 0)
    test_red((9, 5, 7), 1)
    test_red((9, 5, 7), 2)
    test_red((9, 5, 7), -1)

@pytest.test_arrays('tensor, bool')
def test14_any_all_multidimensional_tensors(t):
    v_all = t([[True, False, True],
               [True, True, True],
               [True, False, True],
               [True, True, True]
               ])
    assert list(dr.all(v_all, axis=0)) == [True, False, True]
    assert list(dr.all(v_all, axis=1)) == [False, True, False, True]

    v_any = t([[False, False, False],
               [False, True, False],
               [False, True, False],
               [False, True, False]
               ])

    assert list(dr.any(v_any, axis=0)) == [False, True, False]
    assert list(dr.any(v_any, axis=1)) == [False, True, True, True]
