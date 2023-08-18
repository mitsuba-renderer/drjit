import drjit as dr
import pytest
import importlib

from drjit.scalar import Matrix2f as M2
from drjit.scalar import Matrix3f as M3
from drjit.scalar import Matrix4f as M4
from drjit.scalar import Float
M = M4

def prepare(pkg):
    if 'cuda' in pkg:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in pkg:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    return importlib.import_module(pkg)


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
    m = dr.transpose(M(*range(1, 17)))
    assert m == M(
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    )

    assert dr.diag(m) == [1, 6, 11, 16]
    assert dr.diag(dr.diag(m)) == M(
        [1, 0, 0, 0],
        [0, 6, 0, 0],
        [0, 0, 11, 0],
        [0, 0, 0, 16]
    )


def test03_roundtrip():
    pytest.importorskip("numpy")
    m = M(*range(1, 17)) + dr.full(M, dr.arange(dr.scalar.Float))
    m2 = M(m.numpy())
    assert m == m2


def test04_trace_frob():
    m = M(*range(1, 17))
    assert dr.trace(m) == 34
    assert dr.frob(m) == 1496


def test05_allclose():
    m = dr.full(M, 1)
    assert dr.allclose(m, dr.full(M, 1))
    m[1, 0] = 0
    assert not dr.allclose(m, dr.full(M, 1))
    m = dr.identity(M)
    assert dr.allclose(m, 1)
    m[1, 0] = 1
    assert not dr.allclose(m, 1)


@pytest.mark.parametrize('M', [M2, M3, M4])
def test06_det(M):
    np = pytest.importorskip("numpy")

    np.random.seed(1)
    for i in range(100):
        m1 = np.float32(np.random.normal(size=list(reversed(M.Shape))))
        m2 = M(m1)
        det1 = Float(np.linalg.det(m1))
        det2 = dr.det(m2)
        assert dr.allclose(det1, det2, atol=1e-6)


@pytest.mark.parametrize('M', [M2, M3, M4])
def test07_inverse(M):
    np = pytest.importorskip("numpy")

    np.random.seed(1)
    for i in range(100):
        m1 = np.float32(np.random.normal(size=list(reversed(M.Shape))))
        m2 = M(m1)
        inv1 = M(np.linalg.inv(m1))@m2 - dr.identity(M)
        inv2 = dr.inverse(m2)@m2 - dr.identity(M)
        assert dr.allclose(inv1, 0, atol=1e-3)
        assert dr.allclose(inv2, 0, atol=1e-3)


def test08_polar():
    m = M(*range(1, 17)) + dr.identity(M)
    q, r = dr.polar_decomp(m)
    assert dr.allclose(q@r, m)
    assert dr.allclose(q@dr.transpose(q), dr.identity(M), atol=1e-6)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test09_transform_decompose(package):
    package = prepare(package)
    Quaternion4f, Array3f = package.Quaternion4f, package.Array3f
    Matrix3f, Matrix4f = package.Matrix3f, package.Matrix4f

    m = Matrix4f([[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]])
    s, q, t = dr.transform_decompose(m)

    assert dr.allclose(s, Matrix3f(m))
    assert dr.allclose(q, Quaternion4f(1))
    assert dr.allclose(t, [8, 7, 6])
    assert dr.allclose(m, dr.transform_compose(s, q, t))

    q2 = dr.rotate(Quaternion4f, Array3f(0, 0, 1), 15.0)
    m @= dr.quat_to_matrix(q2)
    s, q, t = dr.transform_decompose(m)

    assert dr.allclose(q, q2)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test10_matrix_to_quat(package):
    package = prepare(package)
    Quaternion4f, Array3f = package.Quaternion4f, package.Array3f

    q = dr.rotate(Quaternion4f, Array3f(0, 0, 1), 15.0)
    m = dr.quat_to_matrix(q)
    q2 = dr.matrix_to_quat(m)
    assert dr.allclose(q, q2)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test11_constructor(package):
    """
    Check Matrix construction from Python array and Numpy array
    """
    np = pytest.importorskip("numpy")

    package = prepare(package)
    Float, Matrix3f = package.Float, package.Matrix3f

    m1 = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    m2 = np.array(m1, dtype=np.float32)
    m3 = Matrix3f(m1)
    m4 = Matrix3f(m2)

    assert dr.allclose(m3, m1)
    assert dr.allclose(m3, m2)
    assert dr.allclose(m3, m4)
    assert dr.allclose(m4, m2)

    if dr.is_jit_v(Float):
        np.random.seed(1)
        for i in range(1, 4):
            values = np.random.random((i, 3, 3)).astype('float32')
            m5 = Matrix3f(values)
            assert dr.allclose(m5, values)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test12_matrix_scale(package):
    np = pytest.importorskip("numpy")

    package = prepare(package)
    Float, Matrix3f = package.Float, package.Matrix3f

    m = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    m2 = np.float32(2*m)

    assert Matrix3f(m) * 2 == Matrix3f(m2)
    assert Matrix3f(m) @ 2 == Matrix3f(m2)
    assert Matrix3f(m) * Float(2) == Matrix3f(m2)
    assert Matrix3f(m) @ Float(2) == Matrix3f(m2)
    assert 2 * Matrix3f(m) == Matrix3f(m2)
    assert Float(2) * Matrix3f(m) == Matrix3f(m2)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test12_matrix_vector(package):
    np = pytest.importorskip("numpy")

    package = prepare(package)
    Float, Matrix3f, Array3f = package.Float, package.Matrix3f, package.Array3f

    m_ = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32)
    m = Matrix3f(m_)

    v1 = m @ Array3f(1, 0, 0)
    v2 = m @ Array3f(1, 1, 0)
    assert dr.allclose(v1, [0.1, 0.4, 0.7])
    assert dr.allclose(v2, [0.3, 0.9, 1.5])
    v1 = m @ dr.scalar.Array3f(1, 0, 0)
    v2 = m @ dr.scalar.Array3f(1, 1, 0)
    assert dr.allclose(v1, [0.1, 0.4, 0.7])
    assert dr.allclose(v2, [0.3, 0.9, 1.5])

    with pytest.raises(dr.Exception):
        m * Array3f(1, 0, 0)

    with pytest.raises(dr.Exception):
        m * dr.scalar.Array3f(1, 0, 0)

    with pytest.raises(dr.Exception):
        m * m

    with pytest.raises(dr.Exception):
        m * dr.scalar.Matrix3f(m_)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test13_matmul_other(package):
    package = prepare(package)
    Float, Matrix3f, Array3f = package.Float, package.Matrix3f, package.Array3f

    m = Matrix3f(1,2,3,4,5,6,7,8,9)
    m2 = Matrix3f(30, 36, 42, 66, 81, 96, 102, 126, 150)
    v = Array3f(5,2,1)
    assert dr.allclose(m @ m, m2)
    assert dr.allclose(m @ v, Array3f(12, 36, 60))
    assert dr.allclose(v @ m, Array3f(20, 28, 36))
    assert dr.allclose(v @ v, Float(30))


@pytest.mark.parametrize("package", ["drjit.cuda", "drjit.llvm"])
@pytest.mark.parametrize("dest", ['numpy', 'torch', 'jax'])
def test14_roundtrip(package, dest):
    pytest.importorskip(dest)

    if dest == 'jax' and 'cuda' in package:
        try:
            from jax.lib import xla_bridge
            xla_bridge.get_backend('gpu')
        except:
            pytest.skip('Backend gpu failed to initialize on JAX')

    package = prepare(package)
    Float, Array3f, Array4f = package.Float, package.Array3f, package.Array4f
    Matrix3f, Matrix4f, Matrix44f = package.Matrix3f, package.Matrix4f, package.Matrix44f

    def to_dest(a):
        if dest == 'numpy':
            return a.numpy()
        if dest == 'torch':
            return a.torch()
        if dest == 'jax':
            return a.jax()

    v = Array3f(
      (1.0 + dr.arange(Float, 5)),
      (1.0 + dr.arange(Float, 5)) * 2,
      (1.0 + dr.arange(Float, 5)) * 3,
    )
    assert(v == Array3f(to_dest(v)))

    m = Matrix3f([
      Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]),
      Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]) * 10,
      Array3f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5]]) * 100
    ])
    assert(m == Matrix3f(to_dest(m)))

    m = Matrix4f([
      Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]),
      Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 10,
      Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 100,
      Array4f([[1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]]) * 1000
    ])
    assert(m == Matrix4f(to_dest(m)))

    m = Matrix44f([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ])
    m[0, 0].x = Float([1, 2])
    m2 = to_dest(m)
    m3 = Matrix44f(m2)
    assert(m == m3)


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test15_quat_to_euler(package):
    package = prepare(package)
    Quaternion4f, Array3f = package.Quaternion4f, package.Array3f

    # Gimbal lock at +pi/2
    q = Quaternion4f(0, 1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at -pi/2
    q = Quaternion4f(0, -1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at +pi/2, such that computed sinp > 1
    q = Quaternion4f(0, 1.0 / dr.sqrt(2) + 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at -pi/2, such that computed sinp < -1
    q = Quaternion4f(0, -1.0 / dr.sqrt(2) - 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0), atol=1e-3))
    # Quaternion without gimbal lock
    q = Quaternion4f(0.15849363803863525, 0.5915063619613647, 0.15849363803863525, 0.7745190262794495)
    e = Array3f(dr.pi / 3, dr.pi / 3, dr.pi / 3)
    assert(dr.allclose(dr.quat_to_euler(q), e))
    # Round trip
    assert(dr.allclose(e, dr.quat_to_euler(dr.euler_to_quat(e))))
    # Euler -> Quat
    assert(dr.allclose(q, dr.euler_to_quat(e)))


@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test16_nested(package):
    np = pytest.importorskip("numpy")
    package = prepare(package)
    Matrix41f = package.Matrix41f
    Matrix4f = package.Matrix4f
    Float = package.Float
    m1 = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    m2 = Matrix41f(m1)
    m3 = np.array(m1, dtype=np.float32)[None, None, :, :]
    m4 = np.array(m2)

    assert np.allclose(m3, m4)

@pytest.mark.parametrize("package", ["drjit.scalar", "drjit.cuda", "drjit.llvm"])
def test17_quat_to_matrix(package):
    np = pytest.importorskip("numpy")

    package = prepare(package)
    Quaternion4f, Matrix3f, Matrix4f = package.Quaternion4f, package.Matrix3f, package.Matrix4f

    # Identity
    q = Quaternion4f([ 0, 0, 0, 1 ])
    m3 = Matrix3f([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(q, size=3), m3))
    assert(dr.allclose(dr.quat_to_matrix(q, size=4), m4))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # pi/2 around z-axis
    q = Quaternion4f([ 0, 0, 1 / dr.sqrt(2), 1 / dr.sqrt(2) ])
    m3 = Matrix3f([ [0, -1, 0], [1, 0, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(q, size=3), m3, atol=2e-7))
    assert(dr.allclose(dr.quat_to_matrix(q, size=4), m4, atol=2e-7))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # Round trip "Random" quaternion
    q = Quaternion4f(0.72331658, 0.49242236, 0.31087897, 0.3710628)
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(q, size=3))))
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(q, size=4))))
