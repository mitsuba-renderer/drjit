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
def test02_repeat(t):
    pkg = get_pkg(t)
    x = dr.arange(t, 10)
    assert dr.all(pkg.repeat(x, 3) == dr.repeat(x, 3))


@pytest.test_arrays('float32,cuda,is_diff,shape=(*)')
def test03_scatter_cas(t):
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)

    target = dr.arange(UInt32, 10) + 5
    dr.make_opaque(target)

    old_value = UInt32(20, 20, 20,  8,  9, 20)
    new_value = UInt32(30, 30, 30, 13, 14, 20)
    index =     UInt32( 1,  0,  4,  3,  4,  5)

    print()
    print(target)
    out = pkg.scatter_cas(target, old_value, new_value, index, True)
    #// 6, 5, 9, 8, 9, 10
    print(out)
    #// 5, 6, 7, 13, 14, 10, 11, 12, 13, 14
    print(target)
