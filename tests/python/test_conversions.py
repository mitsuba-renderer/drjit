import drjit as dr
import pytest
import importlib

def prepare(name):
    if 'cuda' in name:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in name:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    elif 'packet' in name and not hasattr(dr, 'packet'):
        pytest.skip('Packet mode is unsupported')
    return importlib.import_module(name)

@pytest.mark.parametrize("package", ["drjit.cuda", "drjit.cuda.ad",
                                     "drjit.llvm", "drjit.llvm.ad"])
def test_roundtrip_dlpack_all(package):
    package = prepare(package)
    Float, Array3f = package.Float, package.Array3f
    a1 = Array3f(
        dr.arange(Float, 10),
        dr.arange(Float, 10)+100,
        dr.arange(Float, 10)+1000,
    )
    a2 = a1.__dlpack__()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.__dlpack__())


@pytest.mark.parametrize("package", ["drjit.cuda", "drjit.cuda.ad",
                                     "drjit.llvm", "drjit.llvm.ad",
                                     "drjit.packet"])
def test_roundtrip_numpy_all(package):
    pytest.importorskip("numpy")
    package = prepare(package)
    Float, Array3f = package.Float, package.Array3f
    a1 = Array3f(
        dr.arange(Float, 10),
        dr.arange(Float, 10)+100,
        dr.arange(Float, 10)+1000,
    )
    a2 = a1.numpy()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.numpy())

    if Float.IsDynamic:
        import numpy as np
        assert len(np.array(Float())) == 0
        assert len(Float(np.array([], dtype=np.float32))) == 0


def test_roundtrip_pytorch_cuda():
    pytest.importorskip("torch")
    prepare("drjit.cuda.ad")
    from drjit.cuda.ad import Float, Array3f
    a1 = Array3f(
        dr.arange(Float, 10),
        dr.arange(Float, 10)+100,
        dr.arange(Float, 10)+1000,
    )
    a2 = a1.torch()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.torch())


def test_roundtrip_pytorch_llvm():
    pytest.importorskip("torch")
    prepare("drjit.llvm.ad")
    from drjit.llvm.ad import Float, Array3f
    a1 = Array3f(
        dr.arange(Float, 10),
        dr.arange(Float, 10)+100,
        dr.arange(Float, 10)+1000,
    )
    a2 = a1.torch()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.torch())


def test_roundtrip_jax():
    pytest.importorskip("jax")

    try:
        from jax.lib import xla_bridge
        xla_bridge.get_backend('gpu')
    except:
        pytest.skip('Backend gpu failed to initialize on JAX')

    prepare("drjit.cuda.ad")
    from drjit.cuda.ad import Float, Array3f
    a1 = Array3f(
        dr.arange(Float, 10),
        dr.arange(Float, 10)+100,
        dr.arange(Float, 10)+1000,
    )
    a2 = a1.jax()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.jax())


def test_matrix_numpy_construction():
    pytest.importorskip("numpy")
    import numpy as np

    m_py = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    m_np = np.array(m_py)
    m_ek1 = dr.scalar.Matrix3f(m_py)
    m_ek2 = dr.scalar.Matrix3f(m_np)

    assert dr.allclose(m_ek1, m_ek2)
    assert dr.allclose(m_ek1, m_py)
    assert dr.allclose(m_ek1, m_np)


def test_matrix_3_to_4_conversion():
    m3 = dr.scalar.Matrix3f(*range(1, 10))
    m4 = dr.scalar.Matrix4f(m3)

    assert dr.allclose(m4, [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])
    assert dr.allclose(m3, dr.scalar.Matrix3f(m4))


def test_numpy_unit_dimension():
    pytest.importorskip("numpy")
    prepare("drjit.llvm")
    import numpy as np

    a = dr.llvm.TensorXf(dr.arange(dr.llvm.Float32, 3*4), shape=(1, 3, 1, 4, 1))
    b = a.numpy()

    assert dr.allclose(a.shape, b.shape)


def test_tensorflow_unit_dimension():
    pytest.importorskip("tensorflow")
    prepare("drjit.cuda")
    import tensorflow as tf

    a = dr.llvm.TensorXf(dr.arange(dr.llvm.Float32, 3*4), shape=(1, 3, 1, 4, 1))
    b = a.tf()

    assert dr.allclose(a.shape, b.shape)


def test_torch_matrix_unit_dimension():
    pytest.importorskip("torch")
    prepare("drjit.cuda.ad")
    import torch

    expected = dr.cuda.ad.Matrix4f(*range(0, 16))

    a = torch.arange(16).reshape((1, 4, 4)).float().cuda()
    b = dr.cuda.ad.Matrix4f(a)

    assert dr.allclose(b, expected)
