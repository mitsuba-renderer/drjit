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
