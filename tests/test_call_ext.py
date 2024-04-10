import drjit as dr
import pytest
import re
import gc

def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip('call_ext')
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda

def cleanup(s):
    """Remove memory addresses and backend names from a string """
    s = re.sub(r' at 0x[a-fA-F0-9]*',r'', s)
    s = re.sub(r'\.llvm\.',r'.', s)
    s = re.sub(r'\.cuda\.',r'.', s)
    return s

@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    dr.detail.clear_registry()

@pytest.test_arrays('float32,is_diff,shape=(*)')
def test01_array_operations(t):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    # Creating objects (scalar)
    BasePtr(A())
    BasePtr(None)

    # Creating objects (vectorial)
    c = dr.zeros(BasePtr, 2)
    assert(str(c) == '[None, None]')
    assert c[0] is None

    c = dr.full(BasePtr, a, 2)
    assert cleanup(str(c)) == '[<call_ext.A object>, <call_ext.A object>]'
    c = dr.full(BasePtr, b, 2)
    assert cleanup(str(c)) == '[<call_ext.B object>, <call_ext.B object>]'
    assert c[0] is b and c[1] is b
    c[0] = a
    assert cleanup(str(c)) == '[<call_ext.A object>, <call_ext.B object>]'

    c = BasePtr(a)
    assert cleanup(str(c)) == '[<call_ext.A object>]'
    assert c[0] is a

    c = BasePtr(a, b)
    assert cleanup(str(c)) == '[<call_ext.A object>, <call_ext.B object>]'
    assert c[0] is a and c[1] is b
    c[0] = b
    c[1] = a
    assert cleanup(str(c)) == '[<call_ext.B object>, <call_ext.A object>]'
    assert c[0] is b and c[1] is a

    with pytest.raises(TypeError, match=re.escape("unsupported operand type(s) for +")):
        c+c

    assert dr.all(c == c)
    assert dr.all((c == None) == [False, False])
    assert dr.all((c == b) == [True, False])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test02_array_call(t, symbolic):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = c.f(xi, yi)
    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test03_array_call_masked(t, symbolic):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, a, b, b)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)
    mi = dr.mask_t(t)(True, True, False, True, True)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = c.f_masked((xi, yi), mi)

    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))

    c.dummy()


@pytest.mark.parametrize("diff_p1", [True, False])
@pytest.mark.parametrize("diff_p2", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test04_forward_diff(t, symbolic, use_mask, diff_p1, diff_p2):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    if diff_p1:
        dr.enable_grad(xi)

    if diff_p2:
        dr.enable_grad(yi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = c.f_masked((xi, yi), mi)

    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))
    assert dr.grad_enabled(xo) == diff_p2
    assert dr.grad_enabled(yo) == diff_p1

    q, w = 0.0, 0.0
    if diff_p1:
        dr.set_grad(xi, dr.ones(t, 5))
        q = 1.0

    if diff_p2:
        dr.set_grad(yi, dr.full(t, 2, 5))
        w = 2.0

    xg, yg = dr.forward_to(xo, yo, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    dr.schedule(xg, yg)

    assert dr.all(xg == t(2*w, 2*w, 0, 3*w, 3*w))
    assert dr.all(yg == t(-q, -q, 0, q, q))


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test05_backward_diff(t, symbolic, use_mask):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    dr.enable_grad(xi)
    dr.enable_grad(yi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = c.f_masked((xi, yi), mi)

    dr.set_grad(xo, dr.ones(t, 5))
    dr.set_grad(yo, dr.full(t, 2, 5))
    xg, yg = dr.backward_to(xi, yi)
    assert dr.all(xg == [-2, -2, 0, 2, 2])
    assert dr.all(yg == [2, 2, 0, 3, 3])


@pytest.mark.parametrize("opaque", [True, False])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test06_forward_diff_implicit(t, symbolic, use_mask, optimize, opaque):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    x = t(1, 2, 8, 3, 4)
    av = t(1)
    bv = t(1)
    dr.enable_grad(x, av, bv)

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    a.value = av * 2
    b.value = bv * 4
    y = x*x

    if opaque:
        dr.make_opaque(a.value, b.value)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = c.g(y, mi)

    assert dr.all(xo == t(2, 2, 0, 4*9, 4*16))

    dr.set_grad(x, [1,2,3,4,5])
    dr.set_grad(av, 10)
    dr.set_grad(bv, 100)
    xg = dr.forward_to(xo)

    assert dr.all(xg == t([20, 20, 0, 9*400+24*4, 16*400+40*4]))


@pytest.mark.parametrize("opaque", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test07_backward_diff_implicit(t, symbolic, optimize, use_mask, opaque):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    x = t(1, 2, 8, 3, 4)
    av = t(1)
    bv = t(1)
    dr.enable_grad(x, av, bv)

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    a.value = av * 2
    b.value = bv * 4
    y = x*x

    if opaque:
        dr.make_opaque(a.value, b.value)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = c.g(y, mi)

    assert dr.all(xo == t(2, 2, 0, 4*9, 4*16))

    dr.set_grad(xo, [1,2,3,4,5])
    dr.backward_from(xo)

    assert dr.all(dr.grad(x) == t([0, 0, 0, 24, 32]))
    assert dr.all(dr.grad(av) == t([4]))
    assert dr.all(dr.grad(bv) == t([100]))


@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test08_getters(t, use_mask):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    arr0 = c.scalar_getter(mi)
    assert dr.all(arr0 == t([1, 1, 0, 2, 2]))

    arr1 = c.opaque_getter(mi)
    assert dr.all(arr1 == t([1, 1, 0, 2, 2]))

    arr3_a, arr3_b = c.complex_getter(mi)
    assert dr.all(arr3_a == t([1, 1, 0, 4, 4]))
    assert dr.all(arr3_b == dr.uint32_array_t(t)([5, 5, 0, 3, 3]))


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test09_constant_getter(t, drjit_verbose, capsys):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    c = BasePtr(a, a, a, b, b)
    d = c.constant_getter()
    transcript = capsys.readouterr().out
    assert d[0] == 123
    assert transcript.count('ad_call_getter') == 1
    assert transcript.count('jit_var_gather') == 1
    d = c.opaque_getter()
    transcript = capsys.readouterr().out
    assert transcript.count('ad_call_getter') == 1
    assert transcript.count('jit_var_gather') == 1


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test10_constant_getter_partial_registry(t, drjit_verbose, capsys):
    pkg = get_pkg(t)

    A, B, BasePtr = pkg.A, pkg.B, pkg.BasePtr
    a = A()
    b = B()

    del a
    gc.collect()
    gc.collect()

    c = BasePtr(b, b, b, b, b)
    d = c.constant_getter()
    transcript = capsys.readouterr().out
    assert d[0] == 123
    assert transcript.count('ad_call_getter') == 1
    assert transcript.count('jit_var_gather') == 0


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test11_getter_ad_fwd(t):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)
    dr.enable_grad(a.opaque)
    dr.enable_grad(b.opaque)
    dr.set_grad(a.opaque, 10)
    dr.set_grad(b.opaque, 20)

    arr1 = c.opaque_getter()
    assert dr.grad_enabled(arr1)
    assert dr.all(arr1 == t([1, 1, 0, 2, 2]))
    arr1_g = dr.forward_to(arr1)
    assert dr.all(arr1_g == t([10, 10, 0, 20, 20]))


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test12_getter_ad_bwd(t):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)
    dr.enable_grad(a.opaque)
    dr.enable_grad(b.opaque)

    arr1 = c.opaque_getter()
    arr1.grad = [1,2, 3,4,5]
    dr.backward_to(a.opaque, b.opaque)

    assert a.opaque.grad == 3
    assert b.opaque.grad == 9


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test13_array_call_instance_expired(t):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)
    del a
    gc.collect()
    gc.collect()

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    with pytest.raises(RuntimeError, match=re.escape("no longer exists")):
        with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, False):
            xo, yo = c.f(xi, yi)

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test14_array_call_self(t, symbolic, drjit_verbose, capsys):
    pkg = get_pkg(t)
    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        d = c.get_self()
    assert dr.all(c == d)
    transcript = capsys.readouterr().out
    if symbolic:
        if dr.backend_v(t) == dr.JitBackend.LLVM:
            assert transcript.count('%self') > 0
        else:
            assert transcript.count(', self;') > 0
    else:
        assert transcript.count('jit_var_gather') == 2

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test15_array_call_noinst(t, symbolic):
    pkg = get_pkg(t)
    A, B, BasePtr = pkg.A, pkg.B, pkg.BasePtr
    gc.collect()
    gc.collect()
    d = BasePtr(None, None)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        c = BasePtr()
        x = c.g(t())
        assert x.state == dr.VarState.Invalid
        assert len(x) == 0
        y = d.g(t(1,2))
        assert y.state == dr.VarState.Literal
        assert len(y) == 2

        a = A()
        b = B()
        a.value = 5
        b.value =6
        m = dr.mask_t(t)
        e = BasePtr(a, b)
        z = e.g(t(1, 2), m(False))
        assert z.state == dr.VarState.Literal

@pytest.mark.parametrize("diff_p1", [True, False])
@pytest.mark.parametrize("diff_p2", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test16_forward_diff_dispatch(t, symbolic, use_mask, diff_p1, diff_p2):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    def my_func(self, arg, mask):
        return self.f_masked(arg, mask)

    # Turn one element off, two different ways..
    if use_mask:
        mi = Mask(True, True, False, True, True)
        c = BasePtr(a, a, a, b, b)
    else:
        mi = dr.ones(Mask, 5)
        c = BasePtr(a, a, None, b, b)

    if diff_p1:
        dr.enable_grad(xi)

    if diff_p2:
        dr.enable_grad(yi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = dr.dispatch(c, my_func, (xi, yi), mi)

    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))
    assert dr.grad_enabled(xo) == diff_p2
    assert dr.grad_enabled(yo) == diff_p1

    q, w = 0.0, 0.0
    if diff_p1:
        dr.set_grad(xi, dr.ones(t, 5))
        q = 1.0

    if diff_p2:
        dr.set_grad(yi, dr.full(t, 2, 5))
        w = 2.0

    xg, yg = dr.forward_to(xo, yo, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
    dr.schedule(xg, yg)

    assert dr.all(xg == t(2*w, 2*w, 0, 3*w, 3*w))
    assert dr.all(yg == t(-q, -q, 0, q, q))

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test17_dispatch(t, symbolic):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, None, b, b)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = pkg.dispatch_f(c, xi, yi)
    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test18_test_ptr(t, symbolic):
    pkg = get_pkg(t)

    sampler_old = pkg.Sampler(3)
    sampler = pkg.Sampler(3)
    assert dr.all(sampler.rng-sampler_old.rng == [0, 0, 0])

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        c = BasePtr(a, b, None)
        sampler_out, value = c.sample(sampler)
        assert sampler_out is sampler

        diff = sampler.rng-sampler_old.rng
        sampler_old.rng.state[2] = 0
        sampler_old.rng.inc[2] = 0
        assert dr.all(sampler.rng-sampler_old.rng == [1, 0, 0])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test19_test_ptr_py_dispatch(t, symbolic):
    pkg = get_pkg(t)

    sampler_old = pkg.Sampler(3)
    sampler = pkg.Sampler(3)
    assert dr.all(sampler.rng-sampler_old.rng == [0, 0, 0])

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        c = BasePtr(a, b, None)

        def fn(inst, sampler):
            if inst is a:
                return sampler, sampler.next()
            else:
                return sampler, t(0)

        sampler_out, value = dr.dispatch(
            c,
            fn,
            sampler
        )

        assert sampler_out is sampler

        diff = sampler.rng-sampler_old.rng
        sampler_old.rng.state[2] = 0
        sampler_old.rng.inc[2] = 0
        assert dr.all(sampler.rng-sampler_old.rng == [1, 0, 0])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test20_test_ptr_py_loop(t, symbolic):
    # This test is technically about loops. It is located in test_call_ext.py
    # since reuses some of the infrastruture
    pkg = get_pkg(t)

    sampler_old = pkg.Sampler(3)
    sampler = pkg.Sampler(3)

    i = dr.arange(dr.int32_array_t(t), 3)

    def cond(i, sampler):
        return i < 3

    def body(i, sampler):
        i += 1
        sampler.next()
        return i, sampler

    i, sampler = dr.while_loop((i, sampler), cond, body)
    diff = sampler.rng-sampler_old.rng

    assert dr.all(i == [3,3,3])
    assert dr.all(diff == [3,2,1])


# Mask all elements, expect a zero-initialized return value
@pytest.mark.parametrize("opaque_mask", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test21_array_call_fully_masked(t, symbolic, opaque_mask):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    c = BasePtr(a, a, a, b, b)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)
    mi = dr.mask_t(t)(False)
    if opaque_mask:
        dr.make_opaque(mi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = c.f_masked((xi, yi), mi)

    assert dr.all(xo == t(0, 0, 0, 0, 0))
    assert dr.all(yo == t(0, 0, 0, 0, 0))

    c.dummy()

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("opaque_mask", [True, False])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test22_dispatch_fully_masked(t, symbolic, opaque_mask):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    def my_func(self, arg, mask):
        return self.f_masked(arg, mask)

    # Turn all element offs, two different ways..
    mi = Mask(False)
    if opaque_mask:
        dr.make_opaque(mi)
    c = BasePtr(a, a, a, b, b)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = dr.dispatch(c, my_func, (xi, yi), mi)

    assert dr.all(xo == t(0, 0, 0, 0, 0))
    assert dr.all(yo == t(0, 0, 0, 0, 0))

@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("variant", [0, 1])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test23_rev_correctness(t, symbolic, variant):
    # Check the reverse-mode derivative of a call does not overwrite
    # other derivatives flowing to an input argument
    pkg = get_pkg(t)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        A, B, BasePtr = pkg.A, pkg.B, pkg.BasePtr
        a, b = A(), B()

        c = BasePtr(a, a, b, b)

        xi = t(1, 2, 3, 4)
        yi = t(5, 6, 7, 8)
        dr.enable_grad(xi)
        dr.enable_grad(yi)

        if variant == 0:
            xo1 = 100 * xi
            yo1 = 100 * yi

        with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
            xo2, yo2 = c.f(xi, yi)

        if variant == 1:
            xo1 = 100 * xi
            yo1 = 100 * yi

        q = xo1 + xo2 + yo1 + yo2
        q.grad = 1
        xd, yd = dr.backward_to(xi, yi)
        assert dr.all(xd==[99, 99, 101, 101])
        assert dr.all(yd==[102, 102, 103, 103])

@pytest.mark.parametrize("symbolic", [False, True])
@pytest.mark.parametrize("with_evaluated_loop", [False, True])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test12_constant_broadcast(t, symbolic, with_evaluated_loop):
    # When the dispatch index is scalar, the return value should broadcast
    # to the size of the arguments even when they are not used.

    pkg = get_pkg(t)
    A, BasePtr = pkg.A, pkg.BasePtr
    a = A()
    a.value = t(123)

    arg = t(1, 2, 8, 3, 4)
    c = BasePtr(a)

    def cond(out, counter):
        # One iteration
        return counter == 0

    def body(out, counter):
        with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
            out += c.g(arg)
        counter += 1
        return out, counter

    out = t(0)
    counter = dr.uint32_array_t(t)(0)
    if with_evaluated_loop:
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
            out, _ = dr.while_loop((out, counter), cond, body)
    else:
        out += c.g(arg)

    assert dr.all(out == t(123, 123, 123, 123, 123))
