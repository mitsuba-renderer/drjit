import drjit as dr
import pytest

# Test conversions to/from numpy (tensors & dynamic arrays)
@pytest.test_arrays('is_tensor, -bool, -float16')
def test01_roundtrip_dynamic_numpy(t):
    pytest.importorskip("numpy")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.numpy())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

    flat_t = type(a.array)
    roundtrip = flat_t(a.array.numpy())
    assert dr.all(a.array == roundtrip, axis=None)

# Test conversions to/from numpy (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -float16')
def test02_roundtrip_vector_numpy(t):
    pytest.importorskip("numpy")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.numpy())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from torch (tensors & dynamic array)
@pytest.test_arrays('tensor, -bool, -float16, -uint64, -uint32')
def test03_roundtrip_dynamic_torch(t):
    pytest.importorskip("torch")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.torch())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

    flat_t = type(a.array)
    roundtrip = flat_t(a.array.numpy())
    assert dr.all(a.array == roundtrip, axis=None)

# Test conversions to/from torch (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -uint64, -uint32')
def test04_roundtrip_vector_torch(t):
    pytest.importorskip("torch")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.torch())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from tf (tensors & dynamic array)
@pytest.test_arrays('tensor, -bool, -float16')
def test05_roundtrip_dynamic_tf(t):
    pytest.importorskip("tensorflow")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.tf())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

    flat_t = type(a.array)
    roundtrip = flat_t(a.array.numpy())
    assert dr.all(a.array == roundtrip, axis=None)

# Test conversions to/from tf (vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -float16')
def test06_roundtrip_vector_tf(t):
    pytest.importorskip("tensorflow")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.tf())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test conversions to/from jax (tensors & dynamic array)
@pytest.test_arrays('tensor, -bool, -uint64, -int64, -float64')
def test07_roundtrip_dynamic_jax(t):
    pytest.importorskip("jax")
    a = t([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    roundtrip = t(a.jax())

    assert roundtrip.shape == (2, 2, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

    flat_t = type(a.array)
    roundtrip = flat_t(a.array.numpy())
    assert dr.all(a.array == roundtrip, axis=None)

# Test conversions to/from jax(vectors)
@pytest.test_arrays('vector, shape=(3, *), -bool, -uint64, -int64, -float64')
def test08_roundtrip_vector_jax(t):
    pytest.importorskip("jax")
    a = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    roundtrip = t(a.jax())

    assert roundtrip.shape == (3, 3) and roundtrip.shape == a.shape
    assert dr.all(a == roundtrip, axis=None)

# Test inplace modifications from numpy (tensors & dynamic array)
@pytest.test_arrays('tensor, -bool, -float16, -uint64, -uint32')
def test09_inplace_numpy(t):
    pytest.importorskip("numpy")
    a = dr.zeros(t, shape=(3, 3, 3))
    x = a.numpy()
    x[0,0,0] = 1

    backend = dr.backend_v(a)
    if backend == dr.JitBackend.LLVM or backend == dr.JitBackend.Invalid:
        assert a[0,0,0] == x[0,0,0]
    elif backend == dr.JitBackend.CUDA:
        assert a[0,0,0] == 0
        assert x[0,0,0] == 1

# Test inplace modifications from torch (tensors & dynamic array)
@pytest.test_arrays('tensor, -bool, -float16, -uint64, -uint32')
def test10_inplace_torch(t):
    pytest.importorskip("torch")
    a = dr.empty(t, shape=(3, 3, 3))
    x = a.torch()
    x[0,0,0] = 1

    assert a[0,0,0] == x[0,0,0]

# Test AD index preservation after conversion
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test11_conversion_ad(t):
    pytest.importorskip("numpy")
    x = dr.ones(t)
    dr.enable_grad(x)
    i = x.index_ad
    with dr.suspend_grad():
        y = x.numpy()
    assert dr.grad_enabled(x)
    assert i != 0
    assert i == x.index_ad
