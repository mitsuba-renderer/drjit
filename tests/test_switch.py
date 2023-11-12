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
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test02_switch_vec_simple(t, recorded, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
        Int = t
        UInt32 = dr.uint32_array_t(Int)
        Bool = dr.mask_t(Int)

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
        if recorded:
            assert out.count("jit_var_vcall(") == 1
        else:
            # two kernel launches (not merged!) with 2 inputs/outputs and 2 side effects
            assert out.count("(n=2, in=4, out=0, se=2") == 2

# + Masking for some elements
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test03_switch_vec_masked(t, recorded):
    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
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
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test04_failure_incompatible_dict(t, recorded):
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            funcs=(
                lambda a: dict(a=a),
                lambda a: dict(b=a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "error encountered while processing arguments of type 'dict' and 'dict': dictionaries have incompatible keys (['b'] vs ['a'])" in str(e.value.__cause__)


# Let's test a few failures -- dynamic arrays with mismatched sizes
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test05_failure_incompatible_shape(t, recorded):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            funcs=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXu(a, a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "incompatible input lengths (3 and 2)" in str(e.value.__cause__)

    r = dr.switch(
        index=t(0, 0, 1, 1),
        funcs=(
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
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test06_failure_incompatible_type(t, recorded):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            funcs=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXf(a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "incompatible input types" in str(e.value.__cause__)

# Let's test a few failures -- raising an exception in a callable
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test07_failure_incompatible_type(t, recorded):
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
@pytest.mark.parametrize("recorded", [True, False])
def test02_switch_autodiff_forward(t, recorded):
    UInt32 = dr.uint32_array_t(t)

    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
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
@pytest.test_arrays('float,shape=(*),jit,is_diff')
@pytest.mark.parametrize("recorded", [True, False])
def test03_switch_autodiff_forward_implicit(t, recorded):
    UInt32 = dr.uint32_array_t(t)
    idx = UInt32(0, 0, 1, 1)
    a = t(1.0, 2.0, 3.0, 4.0)

    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
        # Implicit dependence on a variable accessed via `dr.gather`
        if True:
            data = t(1.0, 2.0, 3.0, 4.0)
            dr.enable_grad(data)
            data2 = dr.sqr(data)

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
        if False:
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
@pytest.mark.parametrize("recorded", [True, False])
def test04_switch_autodiff_backward(t, recorded):
    UInt32 = dr.uint32_array_t(t)

    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
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
@pytest.mark.parametrize("recorded", [True, False])
def test05_switch_autodiff_backward_implicit(t, recorded):
    UInt32 = dr.uint32_array_t(t)

    idx = UInt32(0, 0, 1, 1)
    a = t(1.0, 2.0, 3.0, 4.0)
    i = UInt32(3, 2, 1, 0)

    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
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

    assert "the symbolic computation being recorded" in str(e.value.__cause__)

@pytest.test_arrays('float,shape=(*),jit')
def test07_invalid_empty_array_in(t):
    idx = dr.uint32_array_t(t)(0, 0, 1, 1)
    with pytest.raises(RuntimeError) as e:
        dr.switch(idx, [lambda a: a, lambda a: a*2], t())
    assert "mismatched argument sizes (4 and 0)" in str(e.value)

@pytest.test_arrays('float,shape=(*),jit')
def test08_invalid_empty_array_out(t):
    idx = dr.uint32_array_t(t)(0, 0, 1, 1)
    with pytest.raises(RuntimeError) as e:
        dr.switch(idx, [lambda a: a, lambda a: t()], t(1, 2, 3, 4))
    assert "empty/uninitialized" in str(e.value)


# Keyword calling, pytrees, differentiation
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('float,shape=(*),jit')
def test09_complex(t, recorded):
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

    with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded):
        a = dict(key=t(1, 2, 3, 4))
        b = (t(2), t(10, 11, 12, 13))
        dr.enable_grad(a, b)

        result = dr.switch(index, c, b=b, a=a)
        dr.detail.check_compatibility(result, expected)
        assert dr.all(result['rv0'] == expected['rv0'])
        assert dr.all(result['rv1'][0] == expected['rv1'][0])
        assert dr.all(result['rv1'][1] == expected['rv1'][1])

        if dr.is_diff_v(t):
            dr.forward_from(a['key'])
            assert dr.all(dr.grad(result['rv0']) == t(5, 5, 3, 3))
            assert dr.all(dr.grad(result['rv1'][0]) == 0)
            assert dr.all(dr.grad(result['rv1'][1]) == 0)
