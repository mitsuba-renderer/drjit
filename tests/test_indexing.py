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

    assert "Complex slicing operations are only supported on tensors." in str(e.value.__context__)

@pytest.test_arrays('shape=(*), -bool', 'tensor, -bool')
def test04_masked_assignment(t):
    v = dr.arange(t, 10)
    v2 = v
    v[(v<3) | (v >= 9)] = 3
    assert v2 is v

    assert dr.all(v == t([3, 3, 3, 3, 4, 5, 6, 7, 8, 3]))
