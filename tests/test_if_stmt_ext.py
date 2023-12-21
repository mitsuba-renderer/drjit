import drjit as dr
import if_stmt_ext as m
import pytest

def get_pkg(t):
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test01_scalar_cond(t):
    pkg = get_pkg(t)
    assert pkg.scalar_cond() == 5


@pytest.mark.parametrize('symbolic', [True, False])
@pytest.test_arrays('uint32,is_diff,shape=(*)')
def test02_simple_cond(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicConditionals, symbolic):
        pkg = get_pkg(t)
        i = pkg.simple_cond()
        assert dr.all(i == t(5, 4, 3, 2, 1, 0, 1, 2, 3, 4))

@pytest.mark.parametrize('symbolic', [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test03_simple_diff(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicConditionals, symbolic):
        pkg = get_pkg(t)
        x = t(-2, -1, 1, 2)
        dr.enable_grad(x)
        y = pkg.my_abs(x)
        dr.forward_from(x)
        assert dr.all(y.grad == [-1, -1, 1, 1])
