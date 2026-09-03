"""
There are instances where functions that have Python-specific arguments such
as PyTrees use an independent code path relative to just the pure C++ interface.
These tests provide basic coverage to check consistency between the two
implementations.
"""

import drjit as dr
import pytest
import sys

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("py_cpp_consistency_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda
    elif backend == dr.JitBackend.Metal:
        return m.metal


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test01_tile(t):
    pkg = get_pkg(t)
    x = dr.arange(t, 10)
    assert dr.all(pkg.tile(x, 3) == dr.tile(x, 3))


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test02_repeat(t):
    pkg = get_pkg(t)
    x = dr.arange(t, 10)
    assert dr.all(pkg.repeat(x, 3) == dr.repeat(x, 3))


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test03_tile_ad(t):
    pkg = get_pkg(t)
    x = dr.arange(t, 10)
    x_tiled_dr = dr.tile(x, 3)
    x_tiled_pkg = pkg.tile(x, 3)

    dr.enable_grad(x_tiled_dr, x_tiled_pkg)

    x2_tiled_dr = x_tiled_dr * x_tiled_dr
    x2_tiled_pkg = x_tiled_pkg * x_tiled_pkg

    dr.backward(x2_tiled_dr)
    dr.backward(x2_tiled_pkg)

    assert dr.all(dr.grad(x_tiled_dr) == dr.grad(x_tiled_pkg))


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test04_repeat_ad(t):
    pkg = get_pkg(t)
    x = dr.arange(t, 10)
    x_repeated_dr = dr.repeat(x, 3)
    x_repeated_pkg = pkg.repeat(x, 3)

    dr.enable_grad(x_repeated_dr, x_repeated_pkg)

    x2_repeated_dr = x_repeated_dr * x_repeated_dr
    x2_repeated_pkg = x_repeated_pkg * x_repeated_pkg

    dr.backward(x2_repeated_dr)
    dr.backward(x2_repeated_pkg)

    assert dr.all(dr.grad(x_repeated_dr) == dr.grad(x_repeated_pkg))


@pytest.test_arrays('matrix,shape=(4, 4, *),float32')
def test05_transform_decompose(t):
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]

    v = [[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]]
    mtx = t(v)
    s_dr, q_dr, tr_dr = dr.transform_decompose(mtx)
    s_pkg, q_pkg, tr_pkg = pkg.transform_decompose(mtx)

    assert dr.all(dr.allclose(s_dr, s_pkg), axis=None)
    assert dr.all(dr.allclose(q_dr, q_pkg), axis=None)
    assert dr.all(tr_dr == tr_pkg, axis=None)


@pytest.test_arrays('matrix,shape=(4, 4, *),float32')
def test06_transform_compose(t):
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]

    v = [[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]]
    mtx = t(v)
    s , q, tr = dr.transform_decompose(mtx)

    m_comp_dr = dr.transform_compose(s, q, tr)
    m_comp_pkg = pkg.transform_compose(s, q, tr)

    assert dr.all(m_comp_dr == m_comp_pkg, axis=None)


@pytest.test_arrays('matrix,shape=(4, 4, *),float32')
def test07_translate(t):
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]
    Array3f  = dr.replace_type_t(m.Array3f, dr.type_v(t))

    v = [[1, 0, 0, 8], [0, 1, 0, 7], [0, 0, 1, 6], [0, 0, 0, 1]]
    mtx = t(v)
    tr = Array3f(8, 7, 6)
    mtx_tr = pkg.translate(tr)

    assert dr.all(mtx == mtx_tr, axis=None)


@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test08_idiv(t):
    pkg = get_pkg(t)
    m = sys.modules[t.__module__]

    divisors = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 17, 64, 100, 127, 128, 129,
                640, 641, 1000, 1023, 1024, 1025, 1920, 65535, 65536, 65537,
                0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]
    values = [0, 1, 2, 3, 7, 8, 9, 100, 4095, 4096, 65535, 65536, 1 << 20,
              0x7FFFFFFF, 0x80000000, 0xFFFFFFFE, 0xFFFFFFFF]

    n = t(values)
    for d in divisors:
        ref = t([v // d for v in values])
        assert dr.all(pkg.idiv_scalar(n, d) == ref)

        # Constants as opaque JIT variables, as if gathered from memory
        mul, shift = pkg.divisor_constants(d)
        mul, shift = dr.opaque(t, mul), dr.opaque(t, shift)
        assert dr.all(pkg.idiv_jit(n, mul, shift) == ref)

    Int32 = m.Int32
    values = [0, 1, -1, 7, -7, 100, -100, 0x7FFFFFFF, -0x80000000]
    n = Int32(values)
    for d in [1, 2, 3, 7, 16, 1000, -2, -3, -7, -16, -1000]:
        # Truncating division, as in C
        ref = Int32([abs(v) // abs(d) * (1 if (v < 0) == (d < 0) else -1) for v in values])
        assert dr.all(pkg.idiv_signed(n, d) == ref)
