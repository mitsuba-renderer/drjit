import drjit as dr
import pytest

@pytest.test_arrays('matrix,shape=(2')
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

@pytest.test_arrays('matrix,shape=(2')
def test02_init_bcast(t):
    assert dr.all(t(2) == t(2, 0, 0, 2), axis=None)
    assert dr.all(t(2) + 1 == t(3, 0, 0, 3), axis=None)

@pytest.test_arrays('matrix,shape=(2')
def test03_add_mul(t):
    a = dr.array_t(t)
    assert dr.all(t(1, 2, 3, 4) + t(0, 1, 0, 2) == t(1, 3, 3, 6), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) + a(t(0, 1, 0, 2)) == a(t(1, 3, 3, 6)), axis=None)
    assert dr.all(t(1, 2, 3, 4) @ t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)
    assert dr.all(t(1, 2, 3, 4) * t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) * a(t(0, 1, 0, 2)) == a(t(0, 2, 0, 8)), axis=None)
    assert dr.all(a(t(1, 2, 3, 4)) @ a(t(0, 1, 0, 2)) == a(t(0, 5, 0, 11)), axis=None)


@pytest.test_arrays('matrix,shape=(2')
def test04_mat_vec(t):
    v = dr.value_t(t)
    assert dr.all(t(1, 2, 3, 4) @ v(1, 2) == v(5, 11))
    assert dr.all(v(1, 2) @ t(1, 2, 3, 4) == v(7, 10))
    assert dr.all(t(1, 2, 3, 4) * v(1, 2) == v(5, 11))
    assert dr.all(v(1, 2) * t(1, 2, 3, 4) == v(7, 10))


@pytest.test_arrays('matrix,shape=(2')
def test05_mat_scalar(t):
    v = dr.value_t(t)
    assert dr.all(t(1, 2, 3, 4) @ 2 == t(2, 4, 6, 8), axis=None)
    assert dr.all(t(1, 2, 3, 4) * 2 == t(2, 4, 6, 8), axis=None)
    assert dr.all(2 * t(1, 2, 3, 4) == t(2, 4, 6, 8), axis=None)
    assert dr.all(2 @ t(1, 2, 3, 4) == t(2, 4, 6, 8), axis=None)


@pytest.test_arrays('matrix,shape=(2')
def test06_mix_backends(t):
    assert dr.all(t(1, 2, 3, 4) @ dr.scalar.Array2f(1, 2) == dr.value_t(t)(5, 11))
    assert dr.all(dr.scalar.Matrix2f(1, 2, 3, 4) @ t(0, 1, 0, 2) == t(0, 5, 0, 11), axis=None)


@pytest.test_arrays('matrix,shape=(2')
def test07_transpose(t):
    a = t(1, 2, 3, 4)
    b = t(1, 3, 2, 4)
    assert dr.all(a.T == b, axis=None)


@pytest.test_arrays('matrix,shape=(2')
def test08_invert_spot_check_2d(t):
    a = t(1, 2, 3, 4)
    b = t(-2, 1, 1.5, -0.5)
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -2)
    assert dr.allclose(1.0/a, b)
    assert dr.allclose(2/a, 2*b)


@pytest.test_arrays('matrix,shape=(3')
def test09_invert_spot_check_3d(t):
    a = t([[1, 2, 0], [3, 4, 0], [0, 1, 1]])
    b = t([[-2, 1,  0], [1.5, -0.5, 0], [-1.5, 0.5, 1]])
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -2)


@pytest.test_arrays('matrix,shape=(4')
def test10_invert_spot_check_4d(t):
    a = t([[1, 2, 0, 0], [3, 4, 0, 1], [0, 1, 1, 0], [1, 0, 1, 2]])
    b = t([[-9, 4, 2, -2], [5, -2, -1, 1], [-5, 2, 2, -1],  [7, -3, -2, 2]])
    assert dr.allclose(dr.rcp(a), b)
    assert dr.allclose(dr.det(a), -1)

@pytest.test_arrays('matrix,shape=(2')
def test11_diag_trace(t):
    v = dr.value_t(t)
    assert dr.all(dr.diag(t(1,2,3,4)) == dr.value_t(t)(1, 4))
    assert dr.trace(t(-1,2,3,4)) == 3
    assert dr.all(dr.diag(dr.value_t(t)(1, 3)) == t(1,0,0,3), axis=None)
