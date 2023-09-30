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
    assert type(dr.all(Array2b(True, True))) is bool
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, False)) == False
    assert dr.all(Array2b(False, False)) == False
    assert dr.any(Array2b(True, True)) == True
    assert dr.any(Array2b(True, False)) == True
    assert dr.any(Array2b(False, False)) == False
    assert type(dr.all(ArrayXb(True, True))) is ArrayXb
    assert len(dr.all(ArrayXb(True, True))) == 1
    assert dr.all(ArrayXb(True, True))[0] == True
    assert dr.all(ArrayXb(True, False))[0] == False
    assert dr.all(ArrayXb(False, False))[0] == False
    assert dr.any(ArrayXb(True, True))[0] == True
    assert dr.any(ArrayXb(True, False))[0] == True
    assert dr.any(ArrayXb(False, False))[0] == False


# Tests dr.{any/all}[_nested] and implicit conversion to 'bool'
@pytest.test_arrays('shape=(1, *), bool')
def test02_any_all_nested(t):
    t0 = dr.value_t(t)
    v0 = t0([True, False, False])

    v = t(v0)
    assert len(v) == 1 and len(v0) == 3

    v0_all = dr.all(v0)
    v0_any = dr.any(v0)
    assert type(v0_all) is t0 and len(v0_all) == 1 and \
           type(v0_any) is t0 and len(v0_any) == 1

    assert bool(v0_all) == False
    assert bool(v0_any) == True

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

    a = dr.sum(m.Float([1, 2, 3]))
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


@pytest.test_arrays('shape=(*), float32, jit')
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


@pytest.test_arrays('shape=(*), float32, jit')
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


@pytest.test_arrays('shape=(*), float32, jit')
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


@pytest.test_arrays('shape=(*), float32, jit')
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

@pytest.test_arrays('shape=(*), float32, jit')
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


@pytest.test_arrays('shape=(*), float32, jit')
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


@pytest.test_arrays('shape=(*), float32, jit')
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
    assert dr.all(dr.prefix_sum(t(1, 2, 3), exclusive=False) == t(1, 3, 6))
    assert dr.all(dr.prefix_sum(m.TensorXf([1, 2, 3])) == m.TensorXf([0, 1, 3]))
    assert dr.all(dr.cumsum(m.TensorXf([1, 2, 3])) == m.TensorXf([1, 3, 6]))
    assert dr.all(dr.prefix_sum(m.Array3f(1, 2, 3)) == m.Array3f(0, 1, 3), axis=None)
    assert dr.all(dr.prefix_sum(m.Array3f(1, 2, 3), exclusive=False) == m.Array3f(1, 3, 6), axis=None)

    # Prefix sum of literals
    x = dr.full(t, 3, 4)
    assert dr.all(dr.prefix_sum(x) == [0, 3, 6, 9])
    assert dr.all(dr.prefix_sum(x, False) == [3, 6, 9, 12])
