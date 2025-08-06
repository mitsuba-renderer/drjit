import drjit as dr
import pytest

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("local_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda
    elif backend == dr.JitBackend.Invalid:
        return m.scalar


@pytest.test_arrays('float32,shape=(*)')
def test01_lookup(t):
    pkg = get_pkg(t)

    assert dr.all(pkg.lookup(4, 5.0, 2) == 3.0)
    assert dr.all(pkg.lookup(4, 5.0, 4) == 5.0)
