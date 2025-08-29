"""
There are instances where functions that have Python-specific arguments such
as PyTrees use an independent code path relative to just the pure C++ interface.
These tests provide basic coverage to check consistency between the two 
implementations.
"""

import drjit as dr
import pytest

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("py_cpp_consistency_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda

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