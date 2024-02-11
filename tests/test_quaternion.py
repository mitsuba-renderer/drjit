import math
import cmath
import drjit as dr
import pytest

@pytest.test_arrays('quaternion')
def test01_bcast(t):
    a = t()
    a.x = 0
    a.y = 0
    a.z = 0
    a.w = 5
    b = t(dr.value_t(t)(5))
    assert dr.all(a == b)
    assert dr.all(dr.ones(t) == t(0, 0, 0, 1))


@pytest.test_arrays('quaternion')
def test02_sub(t):
    a = t(1, 3, 5, 7)
    b = t(11, 12, 13, 14)
    c = t(1-11, 3-12, 5-13, 7-14)
    assert dr.all(a-b == c)


@pytest.test_arrays('quaternion')
def test03_mul(t):
    a = t(1, 3, 5, 7)
    b = t(11, 13, 19, 21)
    c = a * b
    d = t(90, 190, 218, 2)
    e = dr.fma(a, b, t(100,200,300,400))
    assert dr.all(c == d)
    assert dr.all(e == d + t(100,200,300,400))
    assert dr.all(a * 2 == t(2, 6, 10, 14))
    assert dr.all(2 * a == t(2, 6, 10, 14))


@pytest.test_arrays('quaternion')
def test04_div(t):
    a = t(1, 3, 5, 7)
    b = t(11, 13, 19, 21)
    c = a / b
    d = t(-4/91, -16/273, -2/273, 73/273)
    assert dr.allclose(c, d)


@pytest.test_arrays('quaternion')
def test05_abs(t):
    assert type(abs(t(0))) is dr.value_t(t)
    A = dr.array_t(t)
    assert dr.allclose(abs(t(1, 3, 5, 7)), A(math.sqrt(84)))


@pytest.test_arrays('quaternion')
def test06_sqrt(t):
    assert dr.allclose(dr.sqrt(t(1, 2, 3, 4)),
                       t(0.229691, 0.459382, 0.689074, 2.17684))

@pytest.test_arrays('quaternion')
def test07_rsqrt(t):
    assert dr.allclose(dr.sqrt(t(1, 2, 3, 4)),
                       t(0.229691, 0.459382, 0.689074, 2.17684))

@pytest.test_arrays('quaternion')
def test08_log(t):
    assert dr.allclose(dr.log(t(1, 2, 3, 4)),
                       t(0.200991, 0.401982, 0.602974, 1.7006))


@pytest.test_arrays('quaternion')
def test08_log2(t):
    assert dr.allclose(dr.log2(t(1, 2, 3, 4)),
                       t(0.289969, 0.579938, 0.869907, 2.45345))


@pytest.test_arrays('quaternion')
def test09_exp(t):
    assert dr.allclose(dr.exp(t(1, 2, 3, 4)),
                       t(-8.24003, -16.4801, -24.7201, -45.0598))

@pytest.test_arrays('quaternion')
def test09_exp2(t):
    assert dr.allclose(dr.exp2(t(1, 2, 3, 4)),
                       t(2.22808, 4.45615, 6.68423, -13.6565))

@pytest.test_arrays('quaternion')
def test10_pow(t):
    a = t(1, 2, 3, 4)
    assert dr.allclose(a ** t(0.11, 0.13, 0.19, 0.21),
                       t(0.253509, 0.372162, 0.481497, 0.982482))
    assert dr.allclose(a**3, a*a*a)
