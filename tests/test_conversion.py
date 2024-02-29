import drjit as dr
import pytest

# Test conversions to/from numpy (tensors)
@pytest.test_arrays('tensor, -bool, -float16')
def test01_roundtrip_tensor_numpy(t):
    pytest.importorskip("numpy")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.numpy())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from numpy (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -float16')
def test02_roundtrip_vector_numpy(t):
    pytest.importorskip("numpy")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.numpy())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from torch (tensors)
@pytest.test_arrays('tensor, -bool, -float16, -uint64, -uint32')
def test03_roundtrip_tensor_torch(t):
    pytest.importorskip("torch")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.torch())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from torch (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -uint64, -uint32')
def test04_roundtrip_vector_torch(t):
    pytest.importorskip("torch")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.torch())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from tf (tensors)
@pytest.test_arrays('tensor, -bool, -float16')
def test05_roundtrip_tensor_tf(t):
    pytest.importorskip("tensorflow")
    arr = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(arr.tf())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == arr.shape
    assert dr.all(arr == roundtrip, axis=None)

# Test conversions to/from tf (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -float16')
def test06_roundtrip_vector_tf(t):
    pytest.importorskip("tensorflow")
    arr = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(arr.tf())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == arr.shape
    assert dr.all(arr == roundtrip, axis=None)

# Test conversions to/from jax (tensors)
@pytest.test_arrays('tensor, -bool, -uint64, -int64, -float64')
def test07_roundtrip_tensor_jax(t):
    pytest.importorskip("jax")
    arr = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(arr.jax())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == arr.shape
    assert dr.all(arr == roundtrip, axis=None)

# Test conversions to/from jax(vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -uint64, -int64, -float64')
def test08_roundtrip_vector_jax(t):
    pytest.importorskip("jax")
    arr = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(arr.jax())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == arr.shape
    assert dr.all(arr == roundtrip, axis=None)
