import drjit as dr
import pytest
import importlib

@pytest.fixture(scope="module", params=['drjit.cuda', 'drjit.llvm'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test01_sum(m):
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

    a = dr.sum_nested(m.Array3f([1, 1], [2, 2], [3, 3]))
    assert dr.allclose(a, 12)
    assert type(a) is m.Float

    a = dr.sum_nested(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Int

    a = dr.sum_nested(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, 27)
    assert type(a) is m.Float


def test02_mean(m):
    assert dr.allclose(dr.sum(6.0), 6)

    a = dr.mean(m.Float([1, 2, 3]))
    assert dr.allclose(a, 6 / 3.0)
    assert type(a) is m.Float

    a = dr.mean([1.0, 2.0, 3.0])
    assert dr.allclose(a, 6 / 3.0)
    assert type(a) is float

    a = dr.mean(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 6 / 3.0)
    assert type(a) is m.Float

    a = dr.mean(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6 / 3.0)
    assert type(a) is m.Float

    a = dr.mean(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, [2, 3, 4])
    assert type(a) is m.Float

    a = dr.mean_nested(m.Array3f([1, 1], [2, 2], [3, 3]))
    assert dr.allclose(a, 12 / 6.0)
    assert type(a) is m.Float

    a = dr.mean_nested(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6 / 3.0)
    assert type(a) is m.Float

    a = dr.mean_nested(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, 27 / 9.0)
    assert type(a) is m.Float


def test03_prod(m):
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


    a = dr.prod_nested(m.Array3f([1, 1], [2, 2], [3, 3]))
    assert dr.allclose(a, 6 * 6)
    assert type(a) is m.Float

    a = dr.prod_nested(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 6)
    assert type(a) is m.Int

    a = dr.prod_nested(m.ArrayXf([1, 2, 3], [2, 3, 4], [3, 4, 5]))
    assert dr.allclose(a, 6 * 24 * 60)
    assert type(a) is m.Float


def test04_max(m):
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

    a = dr.max_nested(m.Array3f([1, 1], [2, 4], [3, 3]))
    assert dr.allclose(a, 4)
    assert type(a) is m.Float

    a = dr.max_nested(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 3)
    assert type(a) is m.Int

    a = dr.max_nested(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, 5)
    assert type(a) is m.Float


def test05_min(m):
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

    a = dr.min_nested(m.Array3f([1, 1], [2, 4], [3, 3]))
    assert dr.allclose(a, 1)
    assert type(a) is m.Float

    a = dr.min_nested(m.Array3i(1, 2, 3))
    assert dr.allclose(a, 1)
    assert type(a) is m.Int

    a = dr.min_nested(m.ArrayXf([1, 2, 5], [2, 3, 4], [3, 4, 3]))
    assert dr.allclose(a, 1)
    assert type(a) is m.Float


def test06_dot(m):
    a = dr.dot(m.Float(1, 2, 3), m.Float(2, 1, 1))
    assert dr.allclose(a, 7)
    assert type(a) is m.Float

    a = dr.dot(m.Float(1, 2, 3), m.Float(2))
    assert dr.allclose(a, 12)
    assert type(a) is m.Float

    a = dr.dot(m.Float(1, 2, 3), [2, 1, 1])
    assert dr.allclose(a, 7)
    assert type(a) is m.Float

    a = dr.dot(m.Array3f(1, 2, 3), m.Array3f([2, 1, 1]))
    assert dr.allclose(a, 7)
    assert type(a) is m.Float

    a = dr.dot(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.ArrayXf([[1, 2, 5], [2, 2, 4], [3, 4, 3]]))
    assert dr.allclose(a, [14, 22, 50])
    assert type(a) is m.Float


def test07_norm(m):
    a = dr.norm(m.Float(1, 2, 3))
    assert dr.allclose(a, 3.74166)
    assert type(a) is m.Float

    a = dr.norm(m.Array3f(1, 2, 3))
    assert dr.allclose(a, 3.74166)
    assert type(a) is m.Float

def test08_prefix_sum(m):
    # Heavy-duty tests for large arrays are already performed as part of the drjit-core test suite.
    # Here, we just exercise the bindings once for different types
    for t in [m.Float, m.Int32, m.UInt32, m.Float64]:
        assert dr.prefix_sum(t(1, 2, 3)) == t(0, 1, 3)
        assert dr.prefix_sum(t(1, 2, 3), exclusive=False) == t(1, 3, 6)
        assert dr.cumsum(t(1, 2, 3)) == t(1, 3, 6)

def test09_scatter_inc(m):
    try:
        import numpy as np
    except ImportError:
        pytest.skip('NumPy is not installed')
    n=10000
    counter = m.UInt32(0)
    index = dr.arange(m.UInt32, n)
    offset = dr.scatter_inc(counter, m.UInt32(0))

    out = dr.zeros(m.UInt32, n)
    dr.scatter(out, offset, index)
    out_np = np.array(out)
    out_np.sort()
    assert np.all(out_np == np.arange(n))
