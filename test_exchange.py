import drjit as dr
import pytest

def test01_construct_from_pytorch():
    # Nested arrays, copy PyTorch to CUDA and LLVM Dr.Jit array
    import drjit.cuda as c
    import drjit.llvm as l

    try:
        c.Float(1)
        import torch
        torch.tensor([1,2,3], device='cuda')
    except:
        pytest.skip('PyTorch/CUDA support missing')

    p = torch.tensor([[1, 2], [4, 5], [6, 7]], dtype=torch.float32, device='cuda')

    r = c.Array3f(p)
    assert dr.all_nested(r == c.Array3f([1, 2], [4, 5], [6, 7]))
    r = c.Array3i64(p)
    assert dr.all_nested(r == c.Array3i64([1, 2], [4, 5], [6, 7]))

    r = l.Array3f(p)
    assert dr.all_nested(r == l.Array3f([1, 2], [4, 5], [6, 7]))
    r = l.Array3i64(p)
    assert dr.all_nested(r == l.Array3i64([1, 2], [4, 5], [6, 7]))

    p = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cuda')
    r = c.Float(p)
    assert dr.all(r == c.Float([1, 2, 3, 4, 5]))
    p[2] = 5
    torch.cuda.synchronize()
    assert dr.all(r == c.Float([1, 2, 5, 4, 5]))

    p = torch.tensor([[1, 2], [4, 5], [6, 7]], dtype=torch.float32, device='cpu')
    r = c.Array3f(p)
    assert dr.all_nested(r == c.Array3f([1, 2], [4, 5], [6, 7]))
    r = c.Array3i64(p)
    assert dr.all_nested(r == c.Array3i64([1, 2], [4, 5], [6, 7]))

    r = l.Array3f(p)
    assert dr.all_nested(r == l.Array3f([1, 2], [4, 5], [6, 7]))
    r = l.Array3i64(p)
    assert dr.all_nested(r == l.Array3i64([1, 2], [4, 5], [6, 7]))

    p = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device='cpu')
    r = l.Float(p)
    assert dr.all(r == l.Float([1, 2, 3, 4, 5]))
    p[2] = 5
    assert dr.all(r == l.Float([1, 2, 5, 4, 5]))


def test02_construct_from_jax():
    import drjit.cuda as c
    import drjit.llvm as l

    try:
        c.Float(1)
        import jax.numpy as jnp
        jnp.array([1,2,3])
    except:
        pytest.skip('JAX support missing')

    p = jnp.array([[1, 2], [4, 5], [6, 7]], dtype=jnp.float32)

    r = c.Array3f(p)
    assert dr.all_nested(r == c.Array3f([1, 2], [4, 5], [6, 7]))
    r = c.Array3i(p)
    assert dr.all_nested(r == c.Array3i([1, 2], [4, 5], [6, 7]))

    r = l.Array3f(p)
    assert dr.all_nested(r == l.Array3f([1, 2], [4, 5], [6, 7]))
    r = l.Array3i(p)
    assert dr.all_nested(r == l.Array3i([1, 2], [4, 5], [6, 7]))


def test02_construct_from_tf():
    import drjit.cuda as c
    import drjit.llvm as l

    try:
        c.Float(1)
        import tensorflow as tf
        tf.constant([1,2,3])
    except:
        pytest.skip('Tensorflow support missing')

    p = tf.constant([[1, 2], [4, 5], [6, 7]], dtype=tf.float32)

    r = c.Array3f(p)
    assert dr.all_nested(r == c.Array3f([1, 2], [4, 5], [6, 7]))
    r = c.Array3i(p)
    assert dr.all_nested(r == c.Array3i([1, 2], [4, 5], [6, 7]))

    r = l.Array3f(p)
    assert dr.all_nested(r == l.Array3f([1, 2], [4, 5], [6, 7]))
    r = l.Array3i(p)
    assert dr.all_nested(r == l.Array3i([1, 2], [4, 5], [6, 7]))
