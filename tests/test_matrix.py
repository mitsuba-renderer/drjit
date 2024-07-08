import drjit as dr
import pytest
import sys


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test01_init_indexing_rowmajor(t):
    v = dr.value_t(t)
    a = t(1, 2, 3, 4)
    b = t(v(1, 2), v(3, 4))
    assert dr.all(a == b, axis=None)
    assert dr.all(a[0] == v(1, 2)) and dr.all(a[1] == v(3, 4))
    assert a[0, 0] == 1 and a[0, 1] == 2 and a[1, 0] == 3 and a[1, 1] == 4
    assert a[0][0] == 1 and a[0][1] == 2 and a[1][0] == 3 and a[1][1] == 4

    a[0, 1] = 4
    a[1][0] = 5
    assert dr.all(a == t(1, 4, 5, 4), axis=None)

@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test02_init_bcast(t):
    assert dr.all(t(2) == t(2, 0, 0, 2), axis=None)
    assert dr.all(t(2) + 1 == t(3, 0, 0, 3), axis=None)

@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test03_add_mul(t):
    a = dr.array_t(t)
    assert dr.all(t(1, 2, 3, 4) + t(0, 1, 0, 2) == t(1, 3, 3, 6), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) + a(t(0, 1, 0, 2)) == a(t(1, 3, 3, 6)), axis=None)
    assert dr.all(t(1, 2, 3, 4) @ t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)
    assert dr.all(t(1, 2, 3, 4) * t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) * a(t(0, 1, 0, 2)) == a(t(0, 2, 0, 8)), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) @ a(t(0, 1, 0, 2)) == a(t(0, 5, 0, 11)), axis=None)
    assert dr.all(dr.fma(t(1, 2, 3, 4), t(0, 1, 0, 2), t(100,100,100,100)) == t(100, 105, 100, 111), axis=None)


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test04_mat_vec(t):
    v = dr.value_t(t)
    assert dr.all(t(1, 2, 3, 4) @ v(1, 2) == v(5, 11))
    assert dr.all(v(1, 2) @ t(1, 2, 3, 4) == v(7, 10))
    assert dr.all(t(1, 2, 3, 4) * v(1, 2) == v(5, 11))
    assert dr.all(v(1, 2) * t(1, 2, 3, 4) == v(7, 10))


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test05_mat_scalar(t):
    v = dr.value_t(t)
    assert dr.all(t(1, 2, 3, 4) @ 2 == t(2, 4, 6, 8), axis=None)
    assert dr.all(t(1, 2, 3, 4) * 2 == t(2, 4, 6, 8), axis=None)
    assert dr.all(2 * t(1, 2, 3, 4) == t(2, 4, 6, 8), axis=None)
    assert dr.all(2 @ t(1, 2, 3, 4) == t(2, 4, 6, 8), axis=None)


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test06_mix_backends(t):
    assert dr.all(t(1, 2, 3, 4) @ dr.scalar.Array2f(1, 2) == dr.value_t(t)(5, 11))
    assert dr.all(dr.scalar.Matrix2f(1, 2, 3, 4) @ t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test07_transpose(t):
    a = t(1, 2, 3, 4)
    b = t(1, 3, 2, 4)
    assert dr.all(a.T == b, axis=None)


@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test08_invert_spot_check_2d(t):
    a = t(1, 2, 3, 4)
    b = t(-2, 1, 1.5, -0.5)
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -2)
    assert dr.allclose(1.0/a, b)
    assert dr.allclose(2/a, 2*b)


@pytest.test_arrays('matrix,shape=(3, 3, *)', 'matrix,shape=(3, 3)')
def test09_invert_spot_check_3d(t):
    a = t([[1, 2, 0], [3, 4, 0], [0, 1, 1]])
    b = t([[-2, 1,  0], [1.5, -0.5, 0], [-1.5, 0.5, 1]])
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -2)


@pytest.test_arrays('matrix,shape=(4, 4, *)', 'matrix,shape=(4, 4)')
def test10_invert_spot_check_4d(t):
    a = t([[1, 2, 0, 0], [3, 4, 0, 1], [0, 1, 1, 0], [1, 0, 1, 2]])
    b = t([[-9, 4, 2, -2], [5, -2, -1, 1], [-5, 2, 2, -1],  [7, -3, -2, 2]])
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -1)

@pytest.test_arrays('matrix,shape=(2, 2, *)', 'matrix,shape=(2, 2)')
def test11_diag_trace(t):
    v = dr.value_t(t)
    assert dr.all(dr.diag(t(1,2,3,4)) == dr.value_t(t)(1, 4))
    assert dr.trace(t(-1,2,3,4)) == 3
    assert dr.all(dr.diag(dr.value_t(t)(1, 3)) == t(1,0,0,3), axis=None)


@pytest.test_arrays('matrix,shape=(4, 4, *)', 'matrix,shape=(4, 4)')
def test12_frob(t):
    m = t(*range(1, 17))
    assert dr.frob(m) == 1496


@pytest.test_arrays('matrix,shape=(4, 4, *)', 'matrix,shape=(4, 4)')
def test13_polar(t):
    m = t(*range(1, 17)) + dr.identity(t)
    q, r = dr.polar_decomp(m)
    assert dr.allclose(q @ r, m)
    assert dr.allclose(q @ q.T, dr.identity(t))


@pytest.test_arrays('matrix,shape=(4, 4, *)', 'matrix,shape=(4, 4)')
def test14_transform_decompose(t):
    m = sys.modules[t.__module__]
    name = t.__name__
    Quat     = getattr(m, name.replace('Matrix4f', 'Quaternion4f'), None)
    Matrix3f = getattr(m, name.replace('Matrix4f', 'Matrix3f'), None)
    Array3f  = getattr(m, name.replace('Matrix4f', 'Array3f'), None)

    v = [[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]]
    mtx = t(v)
    s, q, tr = dr.transform_decompose(mtx)

    assert dr.allclose(s, Matrix3f([v[0][:3], v[1][:3], v[2][:3]]))
    assert dr.allclose(q, Quat(1))
    assert dr.allclose(tr, [8, 7, 6])
    assert dr.allclose(v, dr.transform_compose(s, q, tr))

    q2 = dr.rotate(Quat, Array3f(0, 0, 1), 15.0)
    mtx @= dr.quat_to_matrix(q2)
    s, q, tr = dr.transform_decompose(mtx)
    assert dr.allclose(q, q2)


@pytest.test_arrays('matrix,shape=(4, 4, *)', 'matrix,shape=(4, 4)')
def test15_matrix_to_quat(t):
    m = sys.modules[t.__module__]
    name    = t.__name__
    Quat    = getattr(m, name.replace('Matrix4f', 'Quaternion4f'), None)
    Array3f = getattr(m, name.replace('Matrix4f', 'Array3f'), None)

    q = dr.rotate(Quat, Array3f(0, 0, 1), 15.0)
    mtx = dr.quat_to_matrix(q)
    q2 = dr.matrix_to_quat(mtx)
    assert dr.allclose(q, q2)


@pytest.test_arrays('quaternion,-float16')
def test16_quat_to_euler(t):
    import sys
    m = sys.modules[t.__module__]
    name    = t.__name__
    Array3f = getattr(m, name.replace('Quaternion4f', 'Array3f'), None)

    # Gimbal lock at +pi/2
    q = t(0, 1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at -pi/2
    q = t(0, -1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at +pi/2, such that computed sinp > 1
    q = t(0, 1.0 / dr.sqrt(2) + 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0), atol=1e-3))
    # Gimbal lock at -pi/2, such that computed sinp < -1
    q = t(0, -1.0 / dr.sqrt(2) - 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0), atol=1e-3))
    # Quaternion without gimbal lock
    q = t(0.15849363803863525, 0.5915063619613647, 0.15849363803863525, 0.7745190262794495)
    e = Array3f(dr.pi / 3, dr.pi / 3, dr.pi / 3)
    assert(dr.allclose(dr.quat_to_euler(q), e))
    # Round trip
    assert(dr.allclose(e, dr.quat_to_euler(dr.euler_to_quat(e))))
    # Euler -> Quat
    assert(dr.allclose(q, dr.euler_to_quat(e)))


@pytest.test_arrays('quaternion,-float16')
def test17_quat_to_matrix(t):
    import sys
    m = sys.modules[t.__module__]
    name    = t.__name__
    Matrix3f = getattr(m, name.replace('Quaternion4f', 'Matrix3f'), None)
    Matrix4f = getattr(m, name.replace('Quaternion4f', 'Matrix4f'), None)

    # Identity
    q = t([ 0, 0, 0, 1 ])
    m3 = Matrix3f([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(q, size=3), m3))
    assert(dr.allclose(dr.quat_to_matrix(q, size=4), m4))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # pi/2 around z-axis
    q = t([ 0, 0, 1 / dr.sqrt(2), 1 / dr.sqrt(2) ])
    m3 = Matrix3f([ [0, -1, 0], [1, 0, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(q, size=3), m3, atol=2e-7))
    assert(dr.allclose(dr.quat_to_matrix(q, size=4), m4, atol=2e-7))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # Round trip "Random" quaternion
    q = t(0.72331658, 0.49242236, 0.31087897, 0.3710628)
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(q, size=3))))
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(q, size=4))))
