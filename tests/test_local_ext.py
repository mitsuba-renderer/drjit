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
    
def is_constant_valued(local, value):
    for i in range(len(local)):
        assert dr.allclose(local.read(i), value)


@pytest.test_arrays('float32,shape=(*)')
def test01_initialization(t):
    pkg = get_pkg(t)
    initial = 25.4

    l10 = pkg.Local10()
    assert len(l10) == 10

    with pytest.raises(AssertionError):
        is_constant_valued(l10, initial)

    l10 = pkg.Local10(initial)
    assert len(l10) == 10

    is_constant_valued(l10, initial)


@pytest.test_arrays('float32,is_jit,shape=(*)')
def test02_dynamic_initialization(t):
    pkg = get_pkg(t)
    initial = dr.full(t, 25.4, 15)

    ldyn = pkg.LocalDyn(initial)
    assert len(ldyn) == 1
    is_constant_valued(ldyn, initial)

    ldyn.resize(20)
    assert len(ldyn) == 20
    is_constant_valued(ldyn, initial)


@pytest.test_arrays('float32,shape=(*)')
def test03_write_read(t):
    pkg = get_pkg(t)
    width = 20

    local = pkg.Local10()

    for i in range(len(local)):
        value = dr.arange(t, i, i+width)
        if dr.backend_v(t) == dr.JitBackend.Invalid:
            value = value[0]
        local.write(i, value)

    sum = dr.zeros(t, width)
    for i in range(len(local)):
        sum += local.read(i)

    expected = dr.sum(dr.arange(t, len(local))) + (dr.arange(t, width) * len(local))
    
    if dr.backend_v(t) == dr.JitBackend.Invalid:
        sum = sum[0]
        expected = expected[0]
    
    assert dr.allclose(sum, expected)

