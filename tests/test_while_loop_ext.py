import drjit as dr
import pytest

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("while_loop_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test01_scalar_loop(t):
    pkg = pytest.importorskip("while_loop_ext")
    assert pkg.scalar_loop() == (5, 9)

@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test02_packet_loop(t):
    pkg = pytest.importorskip("while_loop_ext")
    assert pkg.packet_loop()


@pytest.mark.parametrize('symbolic', [True, False])
@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test03_jit_loop(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic):
        pkg = get_pkg(t)
        i, z = pkg.simple_loop()
        assert dr.all(i == t(5, 5, 5, 5, 5, 5, 6))
        assert dr.all(z == t(9, 9, 9, 9, 9, 0, 0))
