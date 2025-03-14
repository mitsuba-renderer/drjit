import drjit as dr
import pytest

@pytest.test_arrays('shape=(3), -bool')
def test01_index_static(t):
    v = t(1, 2, 3)
    assert v.x == 1 and v.y == 2 and v.z == 3
    v.x, v.y, v.z = 4, 5, 6
    assert v.x == 4 and v.y == 5 and v.z == 6
    assert v[0] == 4 and v[1] == 5 and v[2] == 6
    assert v[-1] == 6 and v[-2] == 5 and v[-3] == 4
    assert len(v) == 3

    with pytest.raises(RuntimeError, match="does not have a"):
        v.w = 4
    with pytest.raises(TypeError, match="is not complex-valued"):
        v.imag = 4
    with pytest.raises(IndexError, match=r"entry 3 is out of bounds \(the array is of size 3\)."):
        v[3]
    with pytest.raises(IndexError, match=r"entry -1 is out of bounds \(the array is of size 3\)."):
        v[-4]

    assert v.shape == (3,)

@pytest.test_arrays('shape=(*), -bool')
def test02_index_dynamic(t):
    v = t(1, 2, 3)
    assert v[0] == 1 and v[1] == 2 and v[2] == 3
    v[0], v[1], v[2] = 4, 5, 6
    assert v[0] == 4 and v[1] == 5 and v[2] == 6
    assert v[-1] == 6 and v[-2] == 5 and v[-3] == 4
    assert len(v) == 3

    with pytest.raises(RuntimeError, match="does not have a"):
        v.x = 4
    with pytest.raises(TypeError, match="is not complex-valued"):
        v.imag = 4
    with pytest.raises(IndexError, match=r"entry 3 is out of bounds \(the array is of size 3\)."):
        v[3]
    with pytest.raises(IndexError, match=r"entry -1 is out of bounds \(the array is of size 3\)."):
        v[-4]

    assert v.shape == (3,)
    assert t(1).shape == (1,)

@pytest.test_arrays('shape=(*, *), -bool')
def test03_index_nested(t):
    v = t([1, 2, 3], [4, 5, 6], [7, 8, 9])

    assert dr.all(v[0] == [1, 2, 3])
    assert v[0, 0] == 1
    assert v[2, 1] == 8
    assert v[-1, -1] == 9

    v[0, 0] = 8
    v[2, 1] = 9
    v[-1, -1] = 1

    assert str(v) == "[[8, 4, 7],\n [2, 5, 9],\n [3, 6, 1]]"

    with pytest.raises(TypeError, match="Item retrieval failed."):
        v[-1, -1, 3]

    with pytest.raises(TypeError) as e:
        v[-1, ...]

    assert "Complex slicing operations involving 'None' / '...' are currently only supported on tensors" in str(e.value.__context__)

@pytest.test_arrays('shape=(*), -bool', 'tensor, -bool')
def test04_masked_assignment(t):
    v = dr.arange(t, 10)
    v2 = v
    v[(v<3) | (v >= 9)] = 3
    assert v2 is v

    assert dr.all(v == t([3, 3, 3, 3, 4, 5, 6, 7, 8, 3]))


@pytest.test_arrays('shape=(4, *), uint32')
def test05_swizzle_access(t):
    a = t(1,2,3,4)
    b = a.xywz
    assert type(b) is t
    assert dr.all(b == [1, 2, 4, 3])

    b = a.ww
    assert dr.size_v(b) == 2 and dr.all(b == [4, 4])

    b = a.wyzwwx
    assert dr.size_v(b) == -1 and len(b) == 6 and dr.all(b == [4, 2, 3, 4, 4, 1])

@pytest.test_arrays('shape=(4, *), uint32')
def test06_swizzle_assignment(t):
    a = t(1,2,3,4)
    a.yz = 5
    assert dr.all(a == [1, 5, 5, 4])

    a = t(1,2,3,4)
    a.yz = dr.value_t(t)(5)
    assert dr.all(a == [1, 5, 5, 4])

    a = t(1,2,3,4)
    a.xyzw = a.xxyy
    assert dr.all(a == [1, 1, 2, 2])

    a = t(1,2,3,4)
    a.xyw = a.yyx
    assert dr.all(a == [2, 2, 3, 1])


@pytest.test_arrays('shape=(3, *), uint32')
def test07_bad_swizzle(t):
    a = t(1,2,3)
    with pytest.raises(AttributeError):
        b = a.xw
    with pytest.raises(AttributeError):
        a.xw = 1
    with pytest.raises(IndexError):
        a = t(1,2,3)
        a.xyw = a.yyxx
    with pytest.raises(IndexError):
        a = t(1,2,3)
        a.xywx = a.yyx


@pytest.test_arrays('shape=(*), float32')
def test08_1d_slice_get(t):
    a = t(1,2,3,4,5)
    assert dr.all(a[1::2] == t(2, 4))
    assert dr.all(a[:-2] == t(1, 2, 3))
    assert a[:].index == a.index


@pytest.test_arrays('shape=(*), float32')
def test09_1d_slice_set(t):
    a = t(1,2,3,4,5)
    a[1::2] = t(8, 9)
    assert dr.all(a == t(1,8,3,9,5))


@pytest.test_arrays('shape=(*), float32')
def test10_1d_slice_get_uint32_index(t):
    a = t(1,2,3,4,5)
    u = dr.uint32_array_t(t)
    i = dr.int32_array_t(t)
    assert dr.all(a[u(1, 3)] == t(2, 4))
    assert dr.all(a[i(-1, -2)] == t(5, 4))


@pytest.test_arrays('shape=(*), float32')
def test11_1d_slice_set_uint32_index(t):
    a = t(1,2,3,4,5)
    u = dr.uint32_array_t(t)
    i = dr.int32_array_t(t)
    a[u(1, 3)] = t(8, 9)
    a[i(-1, -3)] = t(0, 1)
    assert dr.all(a == t(1,8,1,9,0))

@pytest.test_arrays('shape=(3, *), float32', 'shape=(*, *), float32')
def test12_2d_slice_get(t):
    v = dr.value_t(t)
    a = t(1,2,3)
    a1 = a[-1]
    assert type(a1) is v and a1 == 3
    a12 = a[1:3]
    assert len(a12) == 2 and dr.all(a12 == [2,3])
    a12 = a[:-1]
    assert len(a12) == 2 and dr.all(a12 == [1,2])

@pytest.test_arrays('shape=(3, *), float32', 'shape=(*, *), float32')
def test13_2d_slice_set(t):
    v = dr.value_t(t)
    a = t(1,2,3)
    a[1:3] = [10, 20]
    a[:-1] = [100, 200]
    assert dr.all(a == [100, 200, 20], axis=None)
    a[1:3] = 1
    assert dr.all(a == [100, 1, 1], axis=None)

@pytest.test_arrays('shape=(*), float32')
def test14_overwrite_1d(t):
    x = t([1, 2])
    x_id = x.index
    x_backup = x
    x[:] = t([2,3])
    assert dr.all(x == t([2,3]))
    if dr.is_jit_v(t):
        assert x.index != x_id and x is x_backup
    x[:] = 0
    assert dr.all(x == t([0,0]))

@pytest.test_arrays('tensor, float32')
def test15_overwrite_tensor(t):
    x = t([1, 2])
    x_id = x.array.index
    x_backup = x
    x[:] = t([2,3])
    assert dr.all(x == t([2,3]))
    if dr.is_jit_v(t):
        assert x.array.index != x_id and x is x_backup
    x[:] = 0
    assert dr.all(x == t([0,0]))


@pytest.test_arrays('shape=(3, *), float32, jit')
def test16_nested_slice_assignment(t):
    x = dr.ones(t, (3, 20))
    x[:-1, 1:100:2] = 2
    assert dr.any(x == 2, axis=None)
