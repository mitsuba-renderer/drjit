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


@pytest.test_arrays('float32,shape=(*)')
def test04_struct(t):
    pkg = get_pkg(t)
    width = 20 if dr.backend_v(t) == dr.JitBackend else 1

    values = dr.ones(pkg.MyStruct, width)
    local = pkg.LocalStruct10(values)

    def validate_index(idx, value):
        struct = local.read(idx)
        assert dr.width(struct) == width
        assert dr.allclose(struct.value, value)
        assert dr.allclose(struct.priority, value)

    for i in range(10):
        validate_index(i, 1)

    values = dr.zeros(pkg.MyStruct, width)
    local.write(0, values)

    validate_index(0, 0)
    for i in range(1,10):
        validate_index(i, 1)

    if dr.backend_v(t) == dr.JitBackend:
        with pytest.raises(RuntimeError, match="out of bounds"):
            validate_index(10, 1)

@pytest.test_arrays('float32,shape=(*)')
def test05_loop(t):
    pkg = get_pkg(t)
    pkg.test_Local10_loop()
    pkg.test_Local10_loop_struct()

    if dr.backend_v(t) == dr.JitBackend:
        pkg.test_LocalDyn_loop()
        pkg.test_LocalDyn_loop_struct()

    pkg.test_LocalStruct10_loop()
    pkg.test_LocalStruct10_loop_struct()
