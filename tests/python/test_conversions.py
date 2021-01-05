import enoki
import enoki as ek
import pytest

def prepare(pkg):
    if 'cuda' in pkg.__name__:
        if not ek.has_backend(ek.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in pkg.__name__:
        if not ek.has_backend(ek.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')

@pytest.mark.parametrize("package", [ek.cuda, ek.cuda.ad,
                                     ek.llvm, ek.llvm.ad])
def test_roundtrip_dlpack_all(package):
    Float, Array3f = package.Float, package.Array3f
    prepare(package)
    a1 = Array3f(
        ek.arange(Float, 10),
        ek.arange(Float, 10)+100,
        ek.arange(Float, 10)+1000,
    )
    a2 = a1.dlpack()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.dlpack())


@pytest.mark.parametrize("package", [ek.cuda, ek.cuda.ad,
                                     ek.llvm, ek.llvm.ad,
                                     ek.packet])
def test_roundtrip_numpy_all(package):
    pytest.importorskip("numpy")
    Float, Array3f = package.Float, package.Array3f
    prepare(package)
    a1 = Array3f(
        ek.arange(Float, 10),
        ek.arange(Float, 10)+100,
        ek.arange(Float, 10)+1000,
    )
    a2 = a1.numpy()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.numpy())


def test_roundtrip_pytorch_cuda():
    pytest.importorskip("torch")
    prepare(enoki.cuda.ad)
    from enoki.cuda.ad import Float, Array3f
    a1 = Array3f(
        ek.arange(Float, 10),
        ek.arange(Float, 10)+100,
        ek.arange(Float, 10)+1000,
    )
    a2 = a1.torch()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.torch())


def test_roundtrip_pytorch_llvm():
    pytest.importorskip("torch")
    prepare(enoki.llvm.ad)
    from enoki.llvm.ad import Float, Array3f
    a1 = Array3f(
        ek.arange(Float, 10),
        ek.arange(Float, 10)+100,
        ek.arange(Float, 10)+1000,
    )
    a2 = a1.torch()
    a3 = Array3f(a2)
    assert a1 == a3
    assert a1.x == Float(a1.x.torch())


def test_roundtrip_pytorch_jax():
    pytest.importorskip("jax")
    prepare(enoki.cuda.ad)
    from enoki.cuda.ad import Float, Array3f
    a1 = Array3f(
        ek.arange(Float, 10),
        ek.arange(Float, 10)+100,
        ek.arange(Float, 10)+1000,
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
    m_ek1 = ek.scalar.Matrix3f(m_py)
    m_ek2 = ek.scalar.Matrix3f(m_np)

    assert ek.allclose(m_ek1, m_ek2)
    assert ek.allclose(m_ek1, m_py)
    assert ek.allclose(m_ek1, m_np)


def test_matrix_3_to_4_conversion():
    m3 = ek.scalar.Matrix3f(*range(1, 10))
    m4 = ek.scalar.Matrix4f(m3)

    assert ek.allclose(m4, [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 1]])
    assert ek.allclose(m3, ek.scalar.Matrix3f(m4))
