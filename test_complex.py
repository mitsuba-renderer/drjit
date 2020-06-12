import cmath
import enoki as ek
from enoki.packet import Complex2f as C
from enoki.packet import Float, Array2f


def test01_bcast():
    a = C()
    a.real = 5
    a.imag = 0
    b = C(Float(5))
    assert a == b


def test02_add():
    a = C(1, 3)
    b = C(5, 7)
    c = C(1-5, 3-7)
    assert a-b == c


def test03_mul():
    a = C(1, 3)
    b = C(5, 7)
    c = a * b
    d = C(-16, 22)
    assert c == d
    assert a * 2 == C(2, 6)
    assert 2 * a == C(2, 6)


def test04_div():
    a = C(1, 3)
    b = C(5, 7)
    c = a / b
    d = C(13 / 37, 4 / 37)
    assert ek.allclose(c, d)
    assert a / 2 == C(1/2, 3/2)


def test05_rcp():
    assert ek.allclose(ek.rcp(C(1, 3)), C(1/10, -3/10))


def test06_from_builtin():
    assert C(complex(3, 4)) == C(3, 4)


def test07_from_builtin():
    # Fmadd should fallback to regular multiply (complex)-add
    assert ek.fmadd(C(2, 2), C(5, 5), C(5, 6)) == C(5, 26)


def test08_misc():
    for i in range(-5, 5):
        for j in range(-5, 5):
            a = ek.sqrt(C(i, j))
            b = cmath.sqrt(complex(i, j))
            assert ek.allclose(a, C(b))

            assert ek.allclose(ek.conj(a), C(b.conjugate()))
            assert ek.allclose(ek.abs(a), C(abs(b)))

            if i != 0 and j != 0:
                a = ek.rsqrt(C(i, j))
                b = C(1 / cmath.sqrt(complex(i, j)))
                assert ek.allclose(a, b)


def test09_trig():
    for i in range(-5, 5):
        for j in range(-5, 5):
            a = ek.sin(C(i, j))
            b = C(cmath.sin(complex(i, j)))
            assert ek.allclose(a, b)

            a = ek.cos(C(i, j))
            b = C(cmath.cos(complex(i, j)))
            assert ek.allclose(a, b)

            sa, ca = ek.sincos(C(i, j))
            sb = C(cmath.sin(complex(i, j)))
            cb = C(cmath.cos(complex(i, j)))
            assert ek.allclose(sa, sb)
            assert ek.allclose(ca, cb)

            # Python appears to handle the branch cuts
            # differently from Enoki, C, and Mathematica..
            a = ek.asin(C(i, j+0.1))
            b = C(cmath.asin(complex(i, j+0.1)))
            assert ek.allclose(a, b)

            a = ek.acos(C(i, j+0.1))
            b = C(cmath.acos(complex(i, j+0.1)))
            assert ek.allclose(a, b)

            if abs(j) != 1 or i != 0:
                a = ek.atan(C(i, j))
                b = C(cmath.atan(complex(i, j)))
                assert ek.allclose(a, b)


def test10_math_explog():
    for i in range(-5, 5):
        for j in range(-5, 5):
            if i != 0 or j != 0:
                a = ek.log(C(i, j))
                b = C(cmath.log(complex(i, j)))
                assert ek.allclose(a, b)

                a = ek.log2(C(i, j))
                b = C(cmath.log(complex(i, j)) / cmath.log(2))
                assert ek.allclose(a, b)

            a = ek.exp(C(i, j))
            b = C(cmath.exp(complex(i, j)))
            assert ek.allclose(a, b)

            a = ek.exp2(C(i, j))
            b = C(cmath.exp(complex(i, j) * cmath.log(2)))
            assert ek.allclose(a, b)

            a = ek.pow(C(2, 3), C(i, j))
            b = C(complex(2, 3) ** complex(i, j))
            assert ek.allclose(a, b)


def test11_hyp():
    for i in range(-5, 5):
        for j in range(-5, 5):
            a = ek.sinh(C(i, j))
            b = C(cmath.sinh(complex(i, j)))
            assert ek.allclose(a, b)

            a = ek.cosh(C(i, j))
            b = C(cmath.cosh(complex(i, j)))
            assert ek.allclose(a, b)

            sa, ca = ek.sincosh(C(i, j))
            sb = C(cmath.sinh(complex(i, j)))
            cb = C(cmath.cosh(complex(i, j)))
            assert ek.allclose(sa, sb)
            assert ek.allclose(ca, cb)

            # Python appears to handle the branch cuts
            # differently from Enoki, C, and Mathematica..
            a = ek.asinh(C(i + 0.1, j))
            b = C(cmath.asinh(complex(i + 0.1, j)))
            assert ek.allclose(a, b)

            a = ek.acosh(C(i, j))
            b = C(cmath.acosh(complex(i, j)))
            assert ek.allclose(a, b)

            if abs(i) != 1 or j != 0:
                a = ek.atanh(C(i, j))
                b = C(cmath.atanh(complex(i, j)))
                assert ek.allclose(a, b)


def test12_numpy():
    arr1 = C(1, 2)
    arr2 = arr1.numpy()
    assert 'j' in repr(arr2)
    arr3 = C(arr2)
    assert arr1 == arr3
