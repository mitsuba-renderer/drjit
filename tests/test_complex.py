import math
import cmath
import drjit as dr
import pytest

@pytest.test_arrays('complex')
def test01_bcast(t):
    a = t()
    a.real = 5
    a.imag = 0
    b = t(dr.value_t(t)(5))
    assert dr.all(a == b)
    assert dr.all(dr.ones(t) == t(1, 0))


@pytest.test_arrays('complex')
def test02_sub(t):
    a = t(1, 3)
    b = t(5, 7)
    c = t(1-5, 3-7)
    assert dr.all(a-b == c)


@pytest.test_arrays('complex')
def test03_mul(t):
    a = t(1, 3)
    b = t(5, 7)
    c = a * b
    d = t(-16, 22)
    assert dr.all(c == d)
    assert dr.all(a * 2 == t(2, 6))
    assert dr.all(2 * a == t(2, 6))


@pytest.test_arrays('complex')
def test04_abs(t):
    a = dr.abs(t(1, 2))
    assert dr.allclose(a, dr.sqrt(5))
    assert type(a) is dr.value_t(t)


@pytest.test_arrays('complex')
def test05_div(t):
    a = t(1, 3)
    b = t(5, 7)
    c = a / b
    d = t(13 / 37, 4 / 37)
    assert dr.allclose(c, d)
    assert dr.all(a / 2 == t(1/2, 3/2))


@pytest.test_arrays('complex')
def test06_rcp(t):
    assert dr.allclose(dr.rcp(t(1, 3)), t(1/10, -3/10))


@pytest.test_arrays('complex')
def test07_from_builtin(t):
    assert dr.all(t(complex(3, 4)) == t(3, 4))


@pytest.test_arrays('complex')
def test08_to_numpy(t):
    np = pytest.importorskip("numpy")

    a = t(3, 4)
    b = np.array(a)
    assert a.ndim - 1 == b.ndim
    assert b.real == 3 and b.imag == 4

    if a.ndim == 2:
        a = t((3, 4), 5)
        b = np.array(a)
        assert np.all(b == np.array((3+5j, 4+5j), dtype=b.dtype))


@pytest.test_arrays('complex')
def test09_from_numpy(t):
    np = pytest.importorskip("numpy")

    if dr.depth_v(t) == 1:
        assert dr.all(t(np.array(3+5j)) == t(3, 5))
    elif dr.depth_v(t) == 2:
        print(t(np.array((3+5j, 4+5j))))
        assert dr.all(t(np.array((3+5j, 4+5j))) == t((3, 4), (5, 5)), axis=None)
    else:
        assert False


@pytest.test_arrays('complex')
def test10_fma(t):
    assert dr.all(dr.fma(t(2, 2), t(5, 5), t(5, 6)) == t(5, 26))


@pytest.test_arrays('complex')
def test11_sqrt_conj_abs_sqrt_rsqrt(t):
    def assert_close(value, ref, **kwargs):
        ref = t(ref)
        dr.make_opaque(ref)
        assert dr.allclose(value, ref)

    for i in range(-5, 5):
        for j in range(-5, 5):
            tij = t(i, j)
            dr.make_opaque(tij)

            a = dr.sqrt(tij)
            b = cmath.sqrt(complex(i, j))
            assert_close(a, b)

            assert_close(dr.conj(a), b.conjugate())
            assert_close(t(dr.abs(a)), abs(b))

            if i != 0 and j != 0:
                a = dr.rsqrt(tij)
                b = 1 / cmath.sqrt(complex(i, j))
                assert_close(a, b)


@pytest.test_arrays('complex')
def test12_math_explog(t):
    def assert_close(value, ref, **kwargs):
        ref = t(ref)
        dr.make_opaque(ref)
        assert dr.allclose(value, ref)

    for i in range(-5, 5):
        for j in range(-5, 5):
            tij = t(i, j)
            dr.make_opaque(tij)

            if i != 0 or j != 0:
                a = dr.log(tij)
                b = t(cmath.log(complex(i, j)))
                assert_close(a, b)

                a = dr.log2(tij)
                b = t(cmath.log(complex(i, j)) / cmath.log(2))
                assert_close(a, b)

            a = dr.exp(tij)
            b = t(cmath.exp(complex(i, j)))
            assert_close(a, b)

            a = dr.exp2(tij)
            b = t(cmath.exp(complex(i, j) * cmath.log(2)))
            assert_close(a, b)

            a = dr.power(t(2, 3), tij)
            b = t(complex(2, 3) ** complex(i, j))

            assert_close(a, b)

@pytest.test_arrays('complex')
def test09_trig(t):
    def assert_close(value, ref, **kwargs):
        ref = t(ref)
        dr.make_opaque(ref)
        assert dr.allclose(value, ref, **kwargs)

    for i in range(-5, 5):
        for j in range(-5, 5):
            tij = t(i, j)
            dr.make_opaque(tij)

            a = dr.sin(tij)
            b = cmath.sin(complex(i, j))
            assert_close(a, b)

            a = dr.cos(tij)
            b = cmath.cos(complex(i, j))
            assert_close(a, b)

            sa, ca = dr.sincos(tij)
            sb = cmath.sin(complex(i, j))
            cb = cmath.cos(complex(i, j))
            assert_close(sa, sb)
            assert_close(ca, cb)

            # Python appears to handle the branch cuts
            # differently from DrJit, t, and Mathematica..
            a = dr.asin(tij + t(0, 0.1))
            b = cmath.asin(complex(i, j+0.1))
            assert_close(a, b)

            a = dr.acos(tij + t(0, 0.1))
            b = cmath.acos(complex(i, j+0.1))
            assert_close(a, b)

            if abs(j) != 1 or i != 0:
                a = dr.atan(tij)
                b = cmath.atan(complex(i, j))
                assert_close(a, b, atol=1e-6)


@pytest.test_arrays('complex')
def test11_hyp(t):
    def assert_close(value, ref, **kwargs):
        ref = t(ref)
        dr.make_opaque(ref)
        assert dr.allclose(value, ref, **kwargs)

    for i in range(-5, 5):
        for j in range(-5, 5):
            tij = t(i, j)
            dr.make_opaque(tij)

            a = dr.sinh(tij)
            b = cmath.sinh(complex(i, j))
            assert_close(a, b)

            a = dr.cosh(tij)
            b = cmath.cosh(complex(i, j))
            assert_close(a, b)

            sa, ca = dr.sincosh(tij)
            sb = cmath.sinh(complex(i, j))
            cb = cmath.cosh(complex(i, j))
            assert_close(sa, sb)
            assert_close(ca, cb)

            # Python appears to handle the branch cuts
            # differently from DrJit and Mathematica..
            a = dr.asinh(tij + t(0.1, 0))
            b = cmath.asinh(complex(i + 0.1, j))
            assert_close(a, b)

            a = dr.acosh(tij)
            b = cmath.acosh(complex(i, j))
            assert_close(a, b, atol=1e-7)

            if abs(i) != 1 or j != 0:
                a = dr.atanh(tij)
                b = cmath.atanh(complex(i, j))
                assert_close(a, b, atol=1e-6)
