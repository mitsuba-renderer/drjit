import drjit as dr
import sys
import pytest

@pytest.test_arrays('shape=(*), float32', 'shape=(*), float64')
def test01_basic(t):
    m = sys.modules[t.__module__]
    f32 = m.PCG32().next_float32()
    f64 = m.PCG32().next_float64()
    assert f32 == 0.10837864875793457
    assert f64 == 0.10837870510295033

    f32_n = m.PCG32().next_float32_normal()
    f64_n = m.PCG32().next_float64_normal()
    assert f32_n == -1.2351967096328735
    assert f64_n == -1.235196513088357

    if dr.is_jit_v(t):
        f = m.PCG32().next_float(t)
        f_n = m.PCG32().next_float_normal(t)

        if dr.type_v(t) == dr.VarType.Float32:
            assert f == f32
            assert f_n == f32_n
        else:
            assert f == f64
            assert f_n == f64_n
    else:
        f = m.PCG32().next_float(float)
        f_n = m.PCG32().next_float_normal(float)
        assert f == f64
        assert f_n == f64_n

@pytest.test_arrays('shape=(*), float32', 'shape=(*, *), float32', 'tensor, float32', 'shape=(3), float32')
def test02_rand_shape(t):
    if not dr.is_dynamic_v(t):
        x = dr.rand(t, 3)
        assert x.shape[-1] == 3
    elif dr.depth_v(t) > 1 and dr.size_v(t) == dr.Dynamic:
        x = dr.rand(t, (2, 15))
        assert x.shape == (2, 15)
    else:
        x = dr.rand(t, 15)
        assert x.shape[-1] == 15

    if dr.is_tensor_v(t):
        x = dr.rand(t, (15, 20))
        assert x.shape == (15, 20)

@pytest.test_arrays('shape=(3), float32')
def test02_rand_shape_v2(t):
    x = dr.rand(t, 3)
    x.shape == (3,) == 15

@pytest.test_arrays('shape=(*), float32')
def test03_seed(t):
    dr.seed(0)
    v0 = dr.rand(t, 10)
    v1 = dr.rand(t, 10)
    dr.seed(0)
    v2 = dr.rand(t, 10)
    v3 = dr.rand(t, 10, seed=5)
    v4 = dr.rand(t, 10, seed=5)

    assert not dr.all(v0 == v1) and dr.all(v0 == v2)
    assert not dr.all(v0 == v3) and not dr.all(v1 == v3) and dr.all(v3 == v4)
