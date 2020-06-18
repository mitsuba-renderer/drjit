import enoki as ek
import pytest
from enoki.packet import Matrix2f as M2
from enoki.packet import Matrix3f as M3
from enoki.packet import Matrix4f as M4
from enoki.packet import Float
M = M4


def test01_init_broadcast_mul():
    m = M(*range(1, 17))
    m2 = m * m + 1
    m2[1, 2] = 3
    assert m2[1, 2] == 3

    assert m2 == M(
        91, 100, 110, 120,
        202, 229, 3, 280,
        314, 356, 399, 440,
        426, 484, 542, 601
    )

    assert m2[0] == m2.Value(91, 202, 314, 426)


def test02_transpose_diag():
    m = ek.transpose(M(*range(1, 17)))
    assert m == M(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    )

    assert ek.diag(m) == [1, 6, 11, 16]
    assert ek.diag(ek.diag(m)) == M(
        [1, 0, 0, 0],
        [0, 6, 0, 0],
        [0, 0, 11, 0],
        [0, 0, 0, 16]
    )


def test03_roundtrip():
    m = M(*range(1, 17)) + ek.full(M, ek.arange(ek.packet.Float))
    m2 = M(m.numpy())
    assert m == m2


def test04_trace_frob():
    m = M(*range(1, 17))
    assert ek.trace(m) == 34
    assert ek.frob(m) == 1496


def test05_allclose():
    m = ek.full(M, 1)
    assert ek.allclose(m, 1)


@pytest.mark.parametrize('M', [M2, M3, M4])
def test06_det_inverse(M):
    import numpy as np
    np.random.seed(1)
    for i in range(100):
        m1 = np.float32(np.random.normal(size=list(reversed(M.Shape))))
        m2 = M(m1)
        det1 = Float(np.linalg.det(m1))
        det2 = ek.det(m2)
        assert ek.allclose(det1, det2, atol=1e-6)
