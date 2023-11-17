import drjit as dr
import loop_ext as m
import pytest

def get_pkg(t):
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test01_scalar_loop(t):
    pkg = get_pkg(t)
    assert pkg.scalar_loop() == (5, 9)


@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test02_simple_loop(t):
    pkg = get_pkg(t)
    i, z = pkg.simple_loop()
    assert dr.all(i == t(5, 5, 5, 5, 5, 5, 6))
    assert dr.all(z == t(9, 9, 9, 9, 9, 0, 0))
