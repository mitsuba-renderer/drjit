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


@pytest.test_arrays('matrix,shape=(4, 4, *),-float16', 'matrix,shape=(4, 4),-float16')
def test14_transform_decompose(t):
    m = sys.modules[t.__module__]
    name = t.__name__
    Quat     = dr.replace_type_t(m.Quaternion4f, dr.type_v(t))
    Matrix3f = dr.replace_type_t(m.Matrix3f, dr.type_v(t))
    Matrix4f = dr.replace_type_t(m.Matrix4f, dr.type_v(t))
    Array3f  = dr.replace_type_t(m.Array3f, dr.type_v(t))

    v = [[1, 0, 0, 8], [0, 2, 0, 7], [0, 0, 9, 6], [0, 0, 0, 1]]
    mtx = t(v)
    s, q, tr = dr.transform_decompose(mtx)

    assert type(s) == Matrix3f
    assert dr.allclose(s, Matrix3f([v[0][:3], v[1][:3], v[2][:3]]))
    assert dr.allclose(q, Quat(1))
    assert dr.allclose(tr, [8, 7, 6])
    assert dr.allclose(v, dr.transform_compose(s, q, tr))

    q2 = dr.rotate(Quat, Array3f(0, 0, 1), 15.0)
    mtx @= dr.quat_to_matrix(Matrix4f, q2)
    s, q, tr = dr.transform_decompose(mtx)
    assert dr.allclose(q, q2)


@pytest.test_arrays('matrix,shape=(4, 4, *),-float16', 'matrix,shape=(4, 4),-float16')
def test15_matrix_to_quat(t):
    m = sys.modules[t.__module__]
    Quat    = dr.replace_type_t(m.Quaternion4f, dr.type_v(t))
    Array3f = dr.replace_type_t(m.Array3f, dr.type_v(t))
    Matrix3f = dr.replace_type_t(m.Matrix3f, dr.type_v(t))

    # Type checks
    o_t = type(dr.matrix_to_quat(t(1)))
    if t == m.Matrix4f:
        assert o_t == m.Quaternion4f
    else:
        assert o_t == m.Quaternion4f64

    q = dr.rotate(Quat, Array3f(0, 0, 1), 15.0)
    mtx = dr.quat_to_matrix(Matrix3f, q)
    q2 = dr.matrix_to_quat(mtx)
    assert dr.allclose(q, q2)


@pytest.test_arrays('quaternion,-float16')
def test16_quat_to_euler(t):
    import sys
    m = sys.modules[t.__module__]
    Array3f = dr.replace_type_t(m.Array3f, dr.type_v(t))

    # Type checks
    q = t([ 0, 0, 0, 1 ])
    o_t = type(dr.quat_to_euler(q))
    if t == m.Quaternion4f16:
        assert o_t == m.Array3f16
    elif t == m.Quaternion4f:
        assert o_t == m.Array3f
    else:
        assert o_t == m.Array3f64

    # Gimbal lock at +pi/2
    q = t(0, 1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0)))
    # Gimbal lock at -pi/2
    q = t(0, -1.0 / dr.sqrt(2), 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0)))
    # Gimbal lock at +pi/2, such that computed sinp > 1
    q = t(0, 1.0 / dr.sqrt(2) + 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, dr.pi / 2, 0)))
    # Gimbal lock at -pi/2, such that computed sinp < -1
    q = t(0, -1.0 / dr.sqrt(2) - 1e-6, 0, 1.0 / dr.sqrt(2))
    assert(dr.allclose(dr.quat_to_euler(q), Array3f(0, -dr.pi / 2, 0)))
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
    Matrix3f = dr.replace_type_t(m.Matrix3f, dr.type_v(t))
    Matrix4f = dr.replace_type_t(m.Matrix4f, dr.type_v(t))

    # Identity
    q = t([ 0, 0, 0, 1 ])
    m3 = Matrix3f([ [1, 0, 0], [0, 1, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(Matrix3f, q), m3))
    assert(dr.allclose(dr.quat_to_matrix(Matrix4f, q), m4))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # Type checks
    o_t = type(dr.quat_to_matrix(Matrix3f, q))
    if t == m.Quaternion4f16:
        assert o_t == m.Matrix3f16
    elif t == m.Quaternion4f:
        assert o_t == m.Matrix3f
    else:
        assert o_t == m.Matrix3f64

    # pi/2 around z-axis
    q = t([ 0, 0, 1 / dr.sqrt(2), 1 / dr.sqrt(2) ])
    m3 = Matrix3f([ [0, -1, 0], [1, 0, 0], [0, 0, 1] ])
    m4 = Matrix4f([ [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1] ])
    assert(dr.allclose(dr.quat_to_matrix(Matrix3f, q), m3))
    assert(dr.allclose(dr.quat_to_matrix(Matrix4f, q), m4))
    assert(dr.allclose(q, dr.matrix_to_quat(m3)))
    assert(dr.allclose(q, dr.matrix_to_quat(m4)))

    # Round trip "Random" quaternion
    q = t(0.72331658, 0.49242236, 0.31087897, 0.3710628)
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(Matrix3f, q))))
    assert(dr.allclose(q, dr.matrix_to_quat(dr.quat_to_matrix(Matrix4f, q))))

@pytest.test_arrays('-float16, matrix,shape=(4, 4, *)')
def test18_init_upcast(t):
    mod = sys.modules[t.__module__]
    Matrix43f = getattr(mod, 'Matrix43f')
    Matrix41f = getattr(mod, 'Matrix41f')
    assert dr.all(Matrix43f(2) == t(2), axis=None)
    assert dr.all(Matrix41f(2) == t(2), axis=None)
    with pytest.raises(TypeError):
        t(Matrix41f(2))


@pytest.test_arrays('matrix, shape=(2, 2), -jit')
def test19_matrix_convert_size(t):
    mod = sys.modules[t.__module__]
    Matrix2f = t
    Matrix3f = getattr(mod, 'Matrix3f')

    a = Matrix2f(1, 2,
                 3, 4)
    b = Matrix3f(a)
    assert dr.all(b == Matrix3f(1, 2, 0,
                                3, 4, 0,
                                0, 0, 1), axis=None)


    c = Matrix3f(1, 2, 3,
                 4, 5, 6,
                 7, 8, 9)
    d = Matrix2f(c)

    assert dr.all(d == Matrix2f(1, 2,
                                4, 5), axis=None)
