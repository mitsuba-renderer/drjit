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

@pytest.test_arrays('float32,is_diff,shape=(*)')
def test05_scatter_cas(t):
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target = dr.arange(UInt32, 10) + 5
    dr.make_opaque(target)

    old_value = UInt32(20, 20, 20,  8,  9, 20)
    new_value = UInt32(30, 30, 30, 13, 14, 20)
    index =     UInt32( 1,  0,  4,  3,  4,  5)
    mask = Mask(True, True, True, True, True, True)
    dr.make_opaque(old_value, new_value, index, mask)

    old, swapped = pkg.scatter_cas(target, old_value, new_value, index, True)
    dr.eval(old, swapped)

    assert dr.allclose(old, [6, 5, 9, 8, 9, 10])
    assert dr.all(swapped == [False, False, False, True, True, False])
    assert dr.allclose(target, [5, 6, 7, 13, 14, 10, 11, 12, 13, 14])


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test06_scatter_cas_raise(t):
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    target = dr.full(UInt32, 5, 10)
    dr.make_opaque(target)

    old_value = UInt32(20, 20, 20,  5,  5, 20)
    new_value = UInt32(30, 30, 30, 13, 14, 20)
    index =     UInt32( 1,  0,  4,  3,  4,  5)
    mask = Mask(True, True, True, True, True, True)
    dr.make_opaque(old_value, new_value, index, mask)

    old, swapped = pkg.scatter_cas(target, old_value, new_value, index, True)
    dr.eval(old, swapped)

    # Test masking too

    assert dr.allclose(old, [5, 5, 5, 5, 5, 5])
    assert dr.all(swapped == [False, False, False, True, True, False])
    assert dr.allclose(target, [5, 5, 5, 13, 14, 5, 5, 5, 5, 5])

@pytest.test_arrays('float32,is_diff,shape=(*)')
def test07_scatter_cas(t):
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)

    target = dr.arange(UInt32, 10) + 5
    dr.make_opaque(target)

    old_value = UInt32(20, 20, 20,  8,  9, 20)
    new_value = UInt32(30, 30, 30, 13, 14, 20)
    index =     UInt32( 1,  0,  4,  3,  4,  5)

    old, swapped = pkg.scatter_cas(target, old_value, new_value, index, True)
    dr.eval(old)

    assert dr.allclose(old, [6, 5, 9, 8, 9, 10])
    assert dr.allclose(target, [5, 6, 7, 13, 14, 10, 11, 12, 13, 14])

    with pytest.raises(RuntimeError):
        dr.eval(swapped)
    # Check tracing time error instead #fixme



@pytest.test_arrays('float32,llvm,is_diff,shape=(*)')
def test08_scatter_exch(t):
    UInt32 = dr.uint32_array_t(t)
    Mask = dr.mask_t(t)

    dr.set_flag(dr.JitFlag.Debug, True)

    target = dr.full(UInt32, 5, 10)
    dr.make_opaque(target)

    new_value = UInt32(30, 30, 30, 13, 14, 20)
    index =     UInt32( 1,  0,  4,  3,  2,  5)
    mask = Mask(True, True, True, True, True, True)
    dr.make_opaque(new_value, index, mask)

    old = dr.scatter_exch(target, new_value, index, mask)
    dr.eval(old)

    # Test masking too

    assert dr.allclose(old, [5, 5, 5, 5, 5, 5])
    assert dr.allclose(target, [30, 30, 14, 13, 30, 20, 5, 5, 5, 5])

#@pytest.test_arrays('float32,cuda,is_diff,shape=(*)')
#def test06_scatter_cas(t):
#    pkg = get_pkg(t)
#    UInt32 = dr.uint32_array_t(t)
#
#    dr.set_flag(dr.JitFlag.Debug, True)
#    dr.set_flag(dr.JitFlag.ReuseIndices, False)
#
#    target = dr.arange(UInt32, 10) + 5
#    #dr.make_opaque(target)
#
#    dr.set_flag(dr.JitFlag.PrintIR, True)
#    index =     UInt32( 1,  0,  4,  3,  4,  5)
#
#    dr.scatter(target, 0, index)
#
#    dr.eval()
#
#    print(target)
#

@pytest.test_arrays('float32,cuda,shape=(*),-is_diff')
def test07_scatter_inc(t):
    pkg = get_pkg(t)
    UInt32 = dr.uint32_array_t(t)
    Float = t

    dr.set_flag(dr.JitFlag.Debug, True)
    dr.set_flag(dr.JitFlag.ReuseIndices, False)

    ctr = UInt32(0)
    active = dr.mask_t(t)([True, False, True])
    data_compact_1 = Float(0, 1)
    data_compact_2 = Float(10, 11)


    my_index = dr.scatter_inc(ctr, UInt32(0), active)
    dr.scatter(
        target=data_compact_1,
        value=Float(2, 3, 4),
        index=my_index,
        active=active
    )

    tmp = my_index + 2

    dr.eval(data_compact_1) # Run Kernel #1

    with pytest.raises(RuntimeError):
        dr.eval(tmp)
