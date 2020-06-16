import enoki as ek
from enoki.packet import Matrix4f as M


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
    assert ek.transpose(1) == 1
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
