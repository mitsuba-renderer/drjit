import enoki as ek
import numpy as np
import pytest
from enoki.packet import Matrix2f as M2
from enoki.packet import Matrix3f as M3
from enoki.packet import Matrix4f as M4
from enoki.packet import Float
M = M4

def prepare(pkg):
    if 'cuda' in pkg.__name__:
        if not ek.has_backend(ek.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in pkg.__name__:
        if not ek.has_backend(ek.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')


def test01_init_broadcast_mul():
    m = M(*range(1, 17))
    m2 = m @ m + 1
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
    pytest.importorskip("numpy")
    m = M(*range(1, 17)) + ek.full(M, ek.arange(ek.packet.Float))
    m2 = M(m.numpy())
    assert m == m2


def test04_trace_frob():
    m = M(*range(1, 17))
    assert ek.trace(m) == 34
    assert ek.frob(m) == 1496


def test05_allclose():
    m = ek.full(M, 1)
    assert ek.allclose(m, ek.full(M, 1))
    m[1, 0] = 0
    assert not ek.allclose(m, ek.full(M, 1))
    m = ek.identity(M)
    assert ek.allclose(m, 1)
    m[1, 0] = 1
    assert not ek.allclose(m, 1)


@pytest.mark.parametrize('M', [M2, M3, M4])
def test06_det(M):
    pytest.importorskip("numpy")
    import numpy as np
    np.random.seed(1)
    for i in range(100):
        m1 = np.float32(np.random.normal(size=list(reversed(M.Shape))))
        m2 = M(m1)
        det1 = Float(np.linalg.det(m1))
        det2 = ek.det(m2)
        assert ek.allclose(det1, det2, atol=1e-6)


@pytest.mark.parametrize('M', [M2, M3, M4])
def test07_inverse(M):
    pytest.importorskip("numpy")
    import numpy as np
    np.random.seed(1)
    for i in range(100):
        m1 = np.float32(np.random.normal(size=list(reversed(M.Shape))))
        m2 = M(m1)
        inv1 = M(np.linalg.inv(m1))@m2 - ek.identity(M)
        inv2 = ek.inverse(m2)@m2 - ek.identity(M)
        assert ek.allclose(inv1, 0, atol=1e-3)
        assert ek.allclose(inv2, 0, atol=1e-3)


def test08_polar():
    m = M(*range(1, 17)) + ek.identity(M)
    q, r = ek.polar_decomp(m)
    assert ek.allclose(q@r, m)
    assert ek.allclose(q@ek.transpose(q), ek.identity(M), atol=1e-6)


def test09_transform_decompose():
    m = ek.scalar.Matrix4f([[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]])
    s, q, t = ek.transform_decompose(m)

    assert ek.allclose(s, ek.scalar.Matrix3f(m))
    assert ek.allclose(q, ek.scalar.Quaternion4f(1))
    assert ek.allclose(t, [8, 7, 6])
    assert ek.allclose(m, ek.transform_compose(s, q, t))

    q2 = ek.rotate(ek.scalar.Quaternion4f, ek.scalar.Array3f(0, 0, 1), 15.0)
    m @= ek.quat_to_matrix(q2)
    s, q, t = ek.transform_decompose(m)

    assert ek.allclose(q, q2)


def test10_matrix_to_quat():
    q = ek.rotate(ek.scalar.Quaternion4f, ek.scalar.Array3f(0, 0, 1), 15.0)
    m = ek.quat_to_matrix(q)
    q2 = ek.matrix_to_quat(m)
    assert ek.allclose(q, q2)


@pytest.mark.parametrize("package", [ek.scalar, ek.cuda, ek.llvm])
def test11_constructor(package):
    """
    Check Matrix construction from Python array and Numpy array
    """
    Float, Matrix3f = package.Float, package.Matrix3f
    prepare(package)

    m1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    m2 = np.array(m1, dtype=np.float32)
    m3 = Matrix3f(m1)
    m4 = Matrix3f(m2)

    assert ek.allclose(m3, m1)
    assert ek.allclose(m3, m2)
    assert ek.allclose(m3, m4)
    assert ek.allclose(m4, m2)

    if ek.is_jit_array_v(Float):
        np.random.seed(1)
        for i in range(1, 4):
            values = np.random.random((i, 3, 3)).astype('float32')
            m5 = Matrix3f(values)
            assert ek.allclose(m5, values)


@pytest.mark.parametrize("package", [ek.scalar, ek.cuda, ek.llvm])
def test12_matrix_scale(package):
    Float, Matrix3f = package.Float, package.Matrix3f
    prepare(package)

    m = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    m2 = np.float32(2*m)

    assert package.Matrix3f(m) * 2 == package.Matrix3f(m2)
    assert package.Matrix3f(m) @ 2 == package.Matrix3f(m2)
    assert package.Matrix3f(m) * package.Float(2) == package.Matrix3f(m2)
    assert package.Matrix3f(m) @ package.Float(2) == package.Matrix3f(m2)
    assert 2 * package.Matrix3f(m) == package.Matrix3f(m2)
    assert package.Float(2) * package.Matrix3f(m) == package.Matrix3f(m2)


@pytest.mark.parametrize("package", [ek.scalar, ek.cuda, ek.llvm])
def test12_matrix_vector(package):
    m_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    Float, Matrix3f, Array3f = package.Float, package.Matrix3f, package.Array3f
    prepare(package)
    m = Matrix3f(m_)
    v1 = m @ Array3f(1, 0, 0)
    v2 = m @ Array3f(1, 1, 0)
    assert ek.allclose(v1, [0.1, 0.4, 0.7])
    assert ek.allclose(v2, [0.3, 0.9, 1.5])
    v1 = m @ ek.scalar.Array3f(1, 0, 0)
    v2 = m @ ek.scalar.Array3f(1, 1, 0)
    assert ek.allclose(v1, [0.1, 0.4, 0.7])
    assert ek.allclose(v2, [0.3, 0.9, 1.5])

    with pytest.raises(ek.Exception):
        m * Array3f(1, 0, 0)

    with pytest.raises(ek.Exception):
        m * ek.scalar.Array3f(1, 0, 0)

    with pytest.raises(ek.Exception):
        m * m

    with pytest.raises(ek.Exception):
        m * ek.scalar.Matrix3f(m_)


@pytest.mark.parametrize("package", [ek.scalar, ek.cuda, ek.llvm])
def test13_matmul_other(package):
    Float, Matrix3f, Array3f = package.Float, package.Matrix3f, package.Array3f
    prepare(package)

    m = Matrix3f(1,2,3,4,5,6,7,8,9)
    m2 = Matrix3f(30, 36, 42, 66, 81, 96, 102, 126, 150)
    v = Array3f(5,2,1)
    assert ek.allclose(m @ m, m2)
    assert ek.allclose(m @ v, Array3f(12, 36, 60))
    assert ek.allclose(v @ m, Array3f(20, 28, 36))
    assert ek.allclose(v @ v, Float(30))

@pytest.mark.parametrize("package", [ek.cuda, ek.llvm])
@pytest.mark.parametrize("dest", ['numpy', 'torch', 'jax'])
def test14_roundtrip(package, dest):
    pytest.importorskip(dest)
    Float, Array3f, Array4f = package.Float, package.Array3f, package.Array4f
    Matrix3f, Matrix4f, Matrix44f = package.Matrix3f, package.Matrix4f, package.Matrix44f
    prepare(package)

    def to_dest(a):
        if dest == 'numpy':
            return a.numpy()
        if dest == 'torch':
            return a.torch()
        if dest == 'jax':
            return a.jax()

    v = Array3f(
        (1.0 + ek.arange(Float, 5)),
        (1.0 + ek.arange(Float, 5)) * 2,
        (1.0 + ek.arange(Float, 5)) * 3,
    )
    assert(v == Array3f(to_dest(v)))

    m = Matrix3f([
        Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
        Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]) * 10,
        Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]) * 100
    ])
    assert(m ==Matrix3f(to_dest(m)))

    m = Matrix4f([
        Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]),
        Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 10,
        Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 100,
        Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 1000
    ])
    assert(m == Matrix4f(to_dest(m)))

    m = Matrix44f([
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0]
    ])
    assert(m == Matrix44f(to_dest(m)))

@pytest.mark.parametrize("package", [ek.scalar])
def test15_quat_to_euler(package):
    Quaternion4f, Array3f, Float = package.Quaternion4f, package.Array3f, package.Float
    prepare(package)

    # Gimbal lock at +pi/2
    q = Quaternion4f(0, 1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2))
    assert(ek.allclose(ek.quat_to_euler(q), Array3f(0, np.pi / 2, 0)))
    # Gimbal lock at -pi/2
    q = Quaternion4f(0, -1.0 / np.sqrt(2), 0, 1.0 / np.sqrt(2))
    assert(ek.allclose(ek.quat_to_euler(q), Array3f(0, -np.pi / 2, 0)))
    # Quaternion without gimbal lock
    q = Quaternion4f(0.15849363803863525, 0.5915063619613647, 0.15849363803863525, 0.7745190262794495)
    e = Array3f(np.pi / 3, np.pi / 3, np.pi / 3)
    assert(ek.allclose(ek.quat_to_euler(q), e))
    # Round trip
    assert(ek.allclose(e, ek.quat_to_euler(ek.euler_to_quat(e))))
    # Euler -> Quat
    assert(ek.allclose(q, ek.euler_to_quat(e)))
