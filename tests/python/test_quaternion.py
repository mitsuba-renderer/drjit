import math
import enoki as ek
from enoki.scalar import Quaternion4f as Q
from enoki.scalar import Float, Array4f


def test01_bcast():
    a = Q()
    a.x = 0
    a.y = 0
    a.z = 0
    a.w = 5
    b = Q(Float(5))
    assert a == b


def test02_sub():
    a = Q(1, 3, 5, 7)
    b = Q(11, 12, 13, 14)
    c = Q(1-11, 3-12, 5-13, 7-14)
    assert a-b == c


def test03_mul():
    a = Q(1, 3, 5, 7)
    b = Q(11, 13, 19, 21)
    c = a * b
    d = Q(90, 190, 218, 2)
    assert c == d
    assert a * 2 == Q(2, 6, 10, 14)
    assert 2 * a == Q(2, 6, 10, 14)


def test03_div():
    a = Q(1, 3, 5, 7)
    b = Q(11, 13, 19, 21)
    c = a / b
    d = Q(-4/91, -16/273, -2/273, 73/273)
    assert ek.allclose(c, d)


def test04_abs():
    assert ek.allclose(abs(Q(1, 3, 5, 7)), math.sqrt(84))


def test05_sqrt():
    assert ek.allclose(ek.sqrt(Q(1, 2, 3, 4)),
                       Q(0.229691, 0.459382, 0.689074, 2.17684))


def test06_log():
    assert ek.allclose(ek.log(Q(1, 2, 3, 4)),
                       Q(0.200991, 0.401982, 0.602974, 1.7006))


def test06_exp():
    assert ek.allclose(ek.exp(Q(1, 2, 3, 4)),
                       Q(-8.24003, -16.4801, -24.7201, -45.0598))


def test07_pow():
    assert ek.allclose(Q(1, 2, 3, 4) ** Q(0.11, 0.13, 0.19, 0.21),
                       Q(0.253509, 0.372162, 0.481497, 0.982482))
