import drjit as dr
import pytest
import sys

# Scalar version, not backend-ependent
def test01_switch_scalar():
    c = [
        lambda x,active=True: x+1,
        lambda x,active=True: x*10
    ]

    assert dr.switch(0, c, 5) == 6
    assert dr.switch(1, c, 5) == 50
    assert dr.switch(0, c, x=5) == 6
    assert dr.switch(1, c, x=5) == 50
    assert dr.switch(1, c, active=True, x=5) == 50
    assert dr.switch(1, c, active=False, x=5) is None
    assert dr.switch(1, c, x=5, active=False) is None

# A simple call, nothing fancy
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test02_switch_vec_simple(t, symbolic, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)

        c = [
            lambda a, b: (a * 4, b+1),
            lambda a, b: (a * 8, -b)
        ]

        index = UInt32(0, 0, 1, 1)
        a = Int(1, 2, 3, 4)
        b = Int(1)

        result = dr.switch(index, c, a, b)
        assert dr.allclose(result, [[4, 8, 24, 32], [2, 2, -1, -1]])

        out = capsys.readouterr().out
        if symbolic:
            assert out.count("jit_var_call(") == 1
        else:
            # two kernel launches (not merged!) with 2 inputs/outputs and 2 side effects
            assert out.count("(n=2, in=4, out=0, se=2") == 2

# + Masking for some elements
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test03_switch_vec_masked(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)
        Bool = dr.mask_t(Int)

        def assert_literal(active, x):
            assert active.state == dr.VarState.Literal and active[0] is True
            return x

        c = [
            lambda a, b, active: assert_literal(active, (a * 4, Int(2))),
            lambda a, b, active: assert_literal(active, (a * 8, -b))
        ]

        index = UInt32(0, 0, 1, 1)
        a = Int(1, 2, 3, 4)
        b = Int(1)
        active = Bool(True, False, True, False)

        # Masked case
        result = dr.switch(index, c, a, b, active)
        assert dr.allclose(result, [[4, 0, 24, 0], [2, 0, -1, 0]])

        # Masked case, as keyword argument
        result = dr.switch(index, c, a, b, active=active)
        assert dr.allclose(result, [[4, 0, 24, 0], [2, 0, -1, 0]])

# Let's test a few failures -- dictionary key mismatch
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test04_failure_incompatible_dict(t, symbolic):
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            targets=(
                lambda a: dict(a=a),
                lambda a: dict(b=a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "inconsistent dictionary keys for field 'result' (['b'] and ['a'])" in str(e.value)


# Let's test a few failures -- dynamic arrays with mismatched sizes
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test05_failure_incompatible_shape(t, symbolic):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            targets=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXu(a, a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "inconsistent sizes for field 'result' (3 and 2)" in str(e.value)

    r = dr.switch(
        index=t(0, 0, 1, 1),
        targets=(
            lambda a: m.ArrayXu(a*1, a*2),
            lambda a: m.ArrayXu(a*3, a*4)
        ),
        a=t(1, 2, 3, 4)
    )
    assert dr.all(r == m.ArrayXu(
        t(1, 2, 9, 12),
        t(2, 4, 12, 16)
    ), axis=None)


# Let's test a few failures -- differently typed return values
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test06_failure_incompatible_type(t, symbolic):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            targets=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXf(a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "inconsistent types" in str(e.value)


# Let's test a few failures -- raising an exception in a callable
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test07_failure_incompatible_type(t, symbolic):
    def f0(x):
        return x
    def f1(x):
        raise RuntimeError("foobar")
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            t(0, 0, 1, 1),
            (f0, f1),
            t(1, 2, 3, 4)
        )
    assert "foobar" in str(e.value.__cause__)


# Forward-mode AD testcase
@pytest.test_arrays('float,shape=(*),jit,is_diff')
@pytest.mark.parametrize("symbolic", [True, False])
def test02_switch_autodiff_forward(t, symbolic):
    UInt32 = dr.uint32_array_t(t)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        def f(a, b):
            return a * 4.0, b

        def g(a, b):
            return a * 8.0, -b

        idx = UInt32([0, 0, 1, 1])
        a = t([1.0, 2.0, 3.0, 4.0])
        b = t(1.0)

        dr.enable_grad(a, b)
        result = dr.switch(idx, [f, g], a, b)
        assert dr.allclose(result, [[4, 8, 24, 32], [1, 1, -1, -1]])

        dr.forward(a)
        assert dr.allclose(dr.grad(result), [[4, 4, 8, 8], [0, 0, 0, 0]])


# Forward-mode AD testcase with an implicit dependence on another variable
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("variant", [0, 1])
@pytest.test_arrays('float,shape=(*),jit,is_diff')
def test03_switch_autodiff_forward_implicit(t, symbolic, variant):
    UInt32 = dr.uint32_array_t(t)
    idx = UInt32(0, 0, 1, 1)
    a = t(1.0, 2.0, 3.0, 4.0)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        # Implicit dependence on a variable accessed via `dr.gather`
        if variant == 0:
            data = t(1.0, 2.0, 3.0, 4.0)
            dr.enable_grad(data)
            data2 = dr.square(data)

            def f(a, i):
                return a + dr.gather(t, data2, i)

            def g(a, i):
                return a + 4 * dr.gather(t, data2, i)

            i = UInt32(3, 2, 1, 0)

            result = dr.switch(idx, [f, g], a, i)
            assert dr.allclose(result, [1+4**2, 2+3**2, 3+4*2**2, 4+4*1**1])

            dr.set_grad(data, [1, 2, 3, 4])
            g = dr.forward_to(result)
            assert dr.allclose(g, [8*4, 6*3, 16*2, 8*1])

        # Implicit dependence on a scalar variable accessed directly
        if variant == 1:
            value = t(4.0)
            dr.enable_grad(value)
            value2 = 2*value

            def f2(a):
                return value2

            def g2(a):
                return 4 * a

            idx = UInt32(0, 0, 1, 1)
            a = t(1.0, 2.0, 3.0, 4.0)

            result = dr.switch(idx, [f2, g2], a)
            assert dr.allclose(result, [8, 8, 12, 16])

            dr.forward(value)
            assert dr.allclose(dr.grad(result), [2, 2, 0, 0])


@pytest.test_arrays('float,shape=(*),jit,is_diff')
@pytest.mark.parametrize("symbolic", [True, False])
def test04_switch_autodiff_backward(t, symbolic):
    UInt32 = dr.uint32_array_t(t)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        def f(a, b):
            return a * 4, b

        def g(a, b):
            return a * 8, -b

        idx = UInt32(0, 0, 1, 1)
        a = t(1.0, 2.0, 3.0, 4.0)
        b = t(1.0, 1.0, 1.0, 1.0)

        dr.enable_grad(a, b)

        result = dr.switch(idx, [f, g], a, b)
        assert dr.allclose(result, [[4, 8, 24, 32], [1, 1, -1, -1]])

        dr.backward(dr.sum(result, axis=None))
        assert dr.allclose(dr.grad(a), [4, 4, 8, 8])
        assert dr.allclose(dr.grad(b), [1, 1, -1, -1])


@pytest.test_arrays('float,shape=(*),jit,is_diff')
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test05_switch_autodiff_backward_implicit(t, symbolic):
    UInt32 = dr.uint32_array_t(t)

    idx = UInt32(0, 0, 1, 1)
    a = t(1.0, 2.0, 3.0, 4.0)
    i = UInt32(3, 2, 1, 0)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        if True:
            data = t(1.0, 2.0, 3.0, 4.0)
            dr.enable_grad(data)

            def f(a, i):
                return a + dr.gather(t, data, i)

            def g(a, i):
                return a + 4 * dr.gather(t, data, i)

            result = dr.switch(idx, [f, g], a, i)
            assert dr.allclose(result, [5, 5, 11, 8])

            dr.backward(result)
            assert dr.allclose(dr.grad(data), [4, 4, 1, 1])

        if True:
            data = t(3.0)
            dr.enable_grad(data)

            def f(a, i):
                return data + 0

            def g(a, i):
                return a + 4 * data

            result = dr.switch(idx, [f, g], a, i)
            assert dr.allclose(result, [3, 3, 15, 16])

            dr.backward(result)
            assert dr.allclose(dr.grad(data), 10)

        if True:
            data = t(3.0)
            dr.enable_grad(data)

            def f(a, i):
                return data

            def g(a, i):
                return data

            result = dr.switch(idx, [f, g], a, i)
            assert dr.allclose(result, [3, 3, 3, 3])

            dr.backward(result)
            assert dr.allclose(dr.grad(data), 4)

@pytest.test_arrays('float,shape=(*),jit,is_diff')
def test06_invalid_implicit_dependence(t):
    UInt32 = dr.uint32_array_t(t)

    data = t(3.0, 4.0)
    dr.enable_grad(data)

    def f(a, i):
        return data

    def g(a, i):
        return a + 4 * data

    idx = UInt32(0, 0, 1, 1)
    a = t(1.0, 2.0, 3.0, 4.0)
    i = UInt32(3, 2, 1, 0)

    with pytest.raises(RuntimeError) as e:
        dr.switch(idx, [f, g], a, i)

    assert "You performed a differentiable operation that mixes symbolic" in str(e.value.__cause__)


@pytest.test_arrays('float,shape=(*),jit')
def test07_uninitialized_array_in(t):
    idx = dr.uint32_array_t(t)(0, 0, 1, 1)
    with pytest.raises(RuntimeError) as e:
        dr.switch(idx, [lambda a: a, lambda a: a*2], t())
    assert "some of the call arguments have incompatible sizes (encountered arrays of length 4 and 0)" in str(e.value)


@pytest.test_arrays('float,shape=(*),jit')
def test08_uninitialized_array_out(t):
    idx = dr.uint32_array_t(t)(0, 0, 1, 1)
    with pytest.raises(RuntimeError) as e:
        dr.switch(idx, [lambda a: a, lambda a: t()], t(1, 2, 3, 4))
    assert "field 'result' is uninitialized" in str(e.value)


# Keyword calling, pytrees, differentiation
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float,shape=(*),jit')
def test09_complex(t, symbolic):
    UInt32 = dr.uint32_array_t(t)
    Bool = dr.mask_t(t)

    def f0(a: dict, b: tuple):
        return dict(
            rv0=a['key']*5,
            rv1=b
        )

    def f1(a: dict, b: tuple):
        return dict(
            rv0=a['key']*3,
            rv1=(b[1], b[0]))

    c = [ f0, f1 ]

    index = UInt32(0, 0, 1, 1)
    expected = {
        'rv0' : t(5, 10, 9, 12),
        'rv1' : (
            t(2, 2, 12, 13),
            t(10, 11, 2, 2)
        )
    }

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        a = dict(key=t(1, 2, 3, 4))
        b = (t(2), t(10, 11, 12, 13))
        dr.enable_grad(a, b)

        result = dr.switch(index, c, b=b, a=a)
        dr.detail.check_compatibility(result, expected, True, "result")
        assert dr.all(result['rv0'] == expected['rv0'])
        assert dr.all(result['rv1'][0] == expected['rv1'][0])
        assert dr.all(result['rv1'][1] == expected['rv1'][1])

        if dr.is_diff_v(t):
            dr.forward_from(a['key'])
            assert dr.all(dr.grad(result['rv0']) == t(5, 5, 3, 3))
            assert dr.all(dr.grad(result['rv1'][0]) == 0)
            assert dr.all(dr.grad(result['rv1'][1]) == 0)


# Devirtualization of literal constants
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test10_devirtualize(t, optimize):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
        c = [
            lambda a: (t(0), t(1)),
            lambda a: (t(0), t(2))
        ]

        x, y = dr.switch(t(0, 0, 1, 1), c, t(1,2,3, 4))
        assert x.state == (dr.VarState.Literal if optimize else dr.VarState.Unevaluated)
        assert y.state == dr.VarState.Unevaluated


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test11_no_mutate(t, optimize, symbolic):
    def f1(x):
        x += 10
        return x

    def f2(x):
        x += 100
        return x

    def f3(x):
        x += 1000
        return x

    targets = [f1, f2, f3]

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            assert dr.all(dr.switch(t(0, 1, 2), targets, t(1, 2, 3)) == [11, 102, 1003])


@pytest.test_arrays('uint32,shape=(*),jit')
def test12_out_of_bounds_symbolic(t, capsys):
    targets = [lambda x:x, lambda x: x+1]

    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        dr.eval(dr.switch(t(0, 1, 100), targets, t(1), mode='symbolic'))
    transcript = capsys.readouterr().err
    assert "Attempted to invoke callable with index 100, but this value must be strictly smaller than 2" in transcript


@pytest.test_arrays('uint32,shape=(*),jit')
def test13_out_of_bounds_evaluted(t):
    targets = [lambda x:x, lambda x: x+1]

    with pytest.raises(RuntimeError) as e:
        with dr.scoped_set_flag(dr.JitFlag.Debug, True):
            dr.eval(dr.switch(t(0, 1, 100), targets, t(1), mode='evaluated'))
    assert 'out-of-bounds callable ID 101 (must be < 3).' in str(e)

@pytest.mark.parametrize("opaque_mask", [True, False])
@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test14_switch_vec_fully_masked(t, symbolic, opaque_mask):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)
        Bool = dr.mask_t(Int)

        global_var = Int(0, 1, 2, 3)

        def assert_literal(active, x):
            assert active.state == dr.VarState.Literal and active[0] is True
            dr.scatter_reduce(dr.ReduceOp.Add, global_var, x[1], UInt32([0]))
            return x

        c = [
            lambda a, b, active: assert_literal(active, (a * 4, Int(2))),
            lambda a, b, active: assert_literal(active, (a * 8, -b))
        ]

        index = UInt32(0, 0, 1, 1)
        a = Int(1, 2, 3, 4)
        b = Int(1)
        active = Bool(False)
        if opaque_mask:
            dr.make_opaque(active)

        # Masked case
        result = dr.switch(index, c, a, b, active)
        assert dr.allclose(result, [[0, 0, 0, 0], [0, 0, 0, 0]])

        # Masked case, as keyword argument
        result = dr.switch(index, c, a, b, active=active)
        assert dr.allclose(result, [[0, 0, 0, 0], [0, 0, 0, 0]])

        # No side-effects were applied
        assert dr.allclose(global_var, [0, 1, 2, 3])

@pytest.mark.parametrize("mode", ["symbolic", "evaluated"])
@pytest.test_arrays('is_diff,shape=(*),float32')
@dr.syntax
def test15_ad_nested_implicit_dep_fwd(t, mode):
    UInt32 = dr.uint32_array_t(t)
    idx = UInt32(0, 0, 0, 1)
    a = t(0.1, 0.2, 0.3, 0.4)

    # Implicit dependence on a variable accessed via `dr.gather`
    data = t(1.0, 2.0, 3.0, 4.0)
    dr.enable_grad(data)
    data2 = dr.square(data)

    def f(a, i):
        def f1(a, i):
            return a + dr.gather(t, data2, i)

        def f2(a, i):
            return a + dr.gather(t, data2, 3-i)

        return dr.switch(i//2, [f1, f2], a, i, mode=mode, label='inner')

    def g(a, i):
        return t(123)

    i = UInt32(3, 2, 1, 0)

    r = dr.switch(idx, [f, g], a, i, mode=mode, label='outer')
    assert dr.allclose(r, [[1.1, 4.2, 4.3, 123]])

    dr.set_grad(data, [1, 2, 3, 4])
    rg = dr.forward_to(r)
    assert dr.allclose(rg, [2, 8, 8, 0])


@pytest.mark.parametrize('variant', [0, 1])
@pytest.test_arrays('is_jit,-is_diff,shape=(*),uint32')
def test16_optimize_away(t, variant):
    # Ensure that unreferenced outputs aren't included
    Int = t
    UInt32 = dr.uint32_array_t(Int)

    c = [
        lambda a, b: (a + 1, b ^ a),
        lambda a, b: (a + 2, b & a + 1)
    ]

    index = UInt32(0, 0, 1, 1)
    a = Int(1, 2, 3, 4)
    b = Int(1, 4, 1, 2)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        a, b = dr.switch(index, c, a, b^1)
        if variant == 0:
            b = None
        dr.eval(a, b)
        hist = dr.kernel_history((dr.KernelType.JIT,))

    ir = hist[0]['ir'].getvalue()

    if dr.backend_v(t) is dr.JitBackend.LLVM:
        assert ir.count(' = xor') == 2*variant
    elif dr.backend_v(t) is dr.JitBackend.CUDA:
        assert ir.count('xor.b32') == 2*variant


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test17_switch_uneven_buckets(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)
        Bool = dr.mask_t(Int)

        c = [
            lambda x, active: x,
            lambda x, active: x * 10
        ]

        # One thread to 0, two threads masked, three threads to 1
        index = UInt32(0, 1, 1, 1, 1, 1)
        m = Bool([True, False, False, True, True, True])

        x = Int(1, 2, 3, 4, 5, 6)

        # Masked case
        result = dr.switch(index, c, x, m)
        assert dr.allclose(result, [1, 0, 0, 40, 50, 60])

        # Masked case, as keyword argument
        result = dr.switch(index, c, x, active=m)
        assert dr.allclose(result, [1, 0, 0, 40, 50, 60])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test15_switch_scalar_mask(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)

        c = [
            lambda x, active: x,
            lambda x, active: x * 10
        ]

        index = UInt32(0, 0, 1, 1, 1)
        x = Int(1, 2, 3, 4, 5)

        m = True # Use a scalar mask

        # Masked case
        result = dr.switch(index, c, x, m)
        assert dr.allclose(result, [1, 2, 30, 40, 50])

        # Masked case, as keyword argument
        result = dr.switch(index, c, x, active=m)
        assert dr.allclose(result, [1, 2, 30, 40, 50])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test16_switch_loop_in_target(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Int = t
        UInt32 = dr.uint32_array_t(Int)

        impl_data = Int([3])
        dr.make_opaque(impl_data)

        @dr.syntax()
        def a(x, y=impl_data):
            z = y + x
            while z < 10:
                z += 2
            return 3 * z

        c = [
            lambda x: a(x),
            lambda x: x * 10
        ]

        index = UInt32(0, 0, 1, 1, 1)
        x = Int(1, 2, 3, 4, 5)

        result = dr.switch(index, c, x)
        assert dr.allclose(result, [30, 33, 30, 40, 50])


@pytest.mark.parametrize("symbolic", [True, False])
@pytest.test_arrays('float32,diff,shape=(*)')
def test17_bwd_in_switch(t, symbolic):
    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        Float = t
        UInt32 = dr.uint32_array_t(t)

        buf1 = Float(0, 0, 0, 0)
        dr.enable_grad(buf1)
        buf2 = dr.gather(Float, buf1, UInt32([0, 1, 2, 3]))  # Postponed AD edge

        def f1(a, b):
            c = 3 * a
            d = dr.gather(Float, buf2, b)
            dr.backward(2 * d)

            return d + c

        def f2(a, _):
            return a

        funcs = [ f1, f2 ]
        index = UInt32(0, 0, 1, 1)

        a = Float(1, 2, 3, 4)
        b = UInt32([0, 1, 2, 3])

        result = dr.switch(index, funcs, a, b)
        assert dr.allclose(result, [3, 6, 3, 4])
        assert dr.allclose(buf1.grad, [2, 2, 0, 0])


# Loop mask should implicitly be applied to calls (with AD too)
@pytest.mark.parametrize("symbolic_config", [(True, True), (False, True), (False, False)])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax()
def test18_apply_loop_mask(t, capsys, symbolic_config):
    Float = t
    UInt32 = dr.uint32_array_t(t)
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic_config[0]):
            with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic_config[1]):
                def f(value):
                    return value * 2
                def g(value):
                    return Float(0)

                switch_idx = UInt32([0, 1, 123]) # idx=2 is invalid

                val = Float(2)
                dr.enable_grad(val)
                dr.set_grad(val, 1)

                # Every idx does one iteration of the loop, except idx=2
                count = UInt32(0, 0, 1)
                out = Float(1)
                while count < 1:
                    out = dr.switch(switch_idx, [f, g], val)
                    count = count + 1

                dr.forward_to(out)

                # idx=2 output values should be ignored
                assert dr.allclose(out[0], 4)
                assert dr.allclose(out[1], 0)
                assert dr.allclose(out.grad[0], 2)
                assert dr.allclose(out.grad[1], 0)
                transcript = capsys.readouterr().err
                assert "Attempted to invoke callable with index 123, but this value must be strictly smaller than 2" not in transcript
