import drjit as dr
import pytest
import sys

@pytest.mark.parametrize('cond', [True, False])
def test01_scalar(cond):
    # Test a simple 'if' statement in scalar mode
    r = dr.if_stmt(
        args = (4,),
        cond = cond,
        true_fn = lambda x: x + 1,
        false_fn = lambda x: x + 2
    )
    assert r == (5 if cond else 6)


@pytest.test_arrays('uint32,is_jit,shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('cond', [True, False])
def test02_jit_uniform(t, cond, mode):
    # Test a simple 'if' statement with a uniform condition. Can be optimized
    r = dr.if_stmt(
        args = (t(4),),
        cond = dr.mask_t(t)(cond),
        true_fn = lambda x: x + 1,
        false_fn = lambda x: x + dr.opaque(t, 2),
        mode=mode
    )

    assert r.state == (dr.VarState.Literal if cond else dr.VarState.Unevaluated)
    assert r == (5 if cond else 6)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test03_simple(t, mode):
    # Test a simple 'if' statement, general case
    r, = dr.if_stmt(
        args = (t(4),),
        cond = dr.mask_t(t)(True, False),
        true_fn = lambda x: (x + 1,),
        false_fn = lambda x: (x + 2,),
        mode=mode
    )
    assert dr.all(r == t(5, 6))


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test04_test_inconsistencies(t, mode):
    # Test a few problem cases -- mismatched types/sizes (2 flavors)
    with pytest.raises(RuntimeError, match="the type of state variable 'x' changed from"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x: (x + 1,),
            false_fn = lambda x: (x > 2,),
            mode=mode,
            rv_labels=('x',)
        )

    with pytest.raises(RuntimeError, match="the size of state variable 'x' of type"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda _: (t(1, 2),),
            false_fn = lambda _: (t(1, 2, 3),),
            mode=mode,
            rv_labels=('x',)
        )

    with pytest.raises(RuntimeError, match="incompatible sizes"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda _: t(1, 2, 3),
            false_fn = lambda _: t(1, 2, 3),
            mode=mode
        )

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test05_uninitialized_input(t, mode):
    with pytest.raises(RuntimeError, match="state variable 'y' of type .* is uninitialized"):
        dr.if_stmt(
            args = (t(4),t()),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, y),
            false_fn = lambda x, y: (x > 2, y),
            arg_labels = ('x', 'y'),
            mode=mode
        )

    with pytest.raises(RuntimeError, match="'cond' cannot be empty"):
        dr.if_stmt(
            args = (t(4),t(5)),
            cond = dr.mask_t(t)(),
            true_fn = lambda x, y: (x > 1, y),
            false_fn = lambda x, y: (x > 2, y),
            rv_labels = ('x', 'y'),
            mode=mode
        )


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test06_uninitialized_output(t, mode):
    with pytest.raises(RuntimeError, match="state variable 'y' of type .* is uninitialized"):
        dr.if_stmt(
            args = (t(4),t(3)),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, y),
            false_fn = lambda x, y: (x > 2, t()),
            arg_labels= ('a', 'b'),
            rv_labels = ('x', 'y'),
            mode=mode
        )

    with pytest.raises(RuntimeError, match="state variable 'y' of type .* is uninitialized"):
        dr.if_stmt(
            args = (t(4),t(3)),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, t()),
            false_fn = lambda x, y: (x > 2, y),
            arg_labels= ('a', 'b'),
            rv_labels = ('x', 'y'),
            mode=mode
        )

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test07_test_exceptions(t, mode):
    # Exceptions raise in 'true_fn' and 'false_fn' should be correctly propagated
    def my_fn(_):
        raise RuntimeError("foo")

    with pytest.raises(RuntimeError) as e:
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = my_fn,
            false_fn = lambda x: x > 2,
            mode=mode
        )
    assert "foo" in str(e.value.__cause__)

    with pytest.raises(RuntimeError) as e:
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda _: t(1, 2),
            false_fn = my_fn,
            mode=mode
        )
    assert "foo" in str(e.value.__cause__)

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test08_test_syntax_simple(t, mode):
    # Test the @dr.syntax approach to defining 'if' statements
    @dr.syntax
    def f(x, t, mode):
        if dr.hint(x > 2, mode=mode):
            y = t(5)
        else:
            y = t(10)
        return y

    @dr.syntax
    def g(x, t, mode):
        y = t(10)
        if dr.hint(x > 2, mode=mode):
            y = t(5)
        return y

    x = t(1,2,3,4)
    assert dr.all(f(x, t, mode) == (10, 10, 5, 5))
    assert dr.all(g(x, t, mode) == (10, 10, 5, 5))


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test09_test_mutate(t, mode):
    # Everything should work as expected when 'if_fn' and 'true_fn' mutate their inputs
    @dr.syntax
    def f(x, y, t, mode):
        if dr.hint(x > 2, mode=mode):
            y += 10
        else:
            y += 100
        return y

    x = t(1,2,3,4)
    y = t(x)
    assert dr.all(f(x, y, t, mode) == (101, 102, 13, 14))

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test09_incompatible_scalar(t, mode):
    # It's easy to accidentally define a scalar in an incompatible way. Catch this.
    @dr.syntax
    def f(x, t, mode):
        if dr.hint(x > 2, mode=mode):
            y = 5
        else:
            y = 10

    with pytest.raises(RuntimeError, match="the non-array state variable 'y' of type 'int' changed from '5' to '10'"):
        f(t(1,2,3,4), t, mode)

    # but also let users turn off this check
    @dr.syntax
    def f2(x, t, mode):
        if dr.hint(x > 2, mode=mode, strict=False):
            y = 5
        else:
            y = 10

    f2(t(1,2,3,4), t, mode)


@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('variant_2', [0, 1, 2, 3])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test10_side_effect(t, mode, variant, variant_2, capsys, drjit_verbose):
    # Ensure that side effects are correctly tracked in evaluated and symbolic modes
    if variant == 0:
        buf = t(0, 0)
    else:
        buf = dr.zeros(t, 2)

    def true_fn():
        if variant_2 & 1:
            dr.scatter_add(buf, index=0, value=1, mode=dr.ReduceMode.Local)

    def false_fn():
        if variant_2 & 2:
            dr.scatter_add(buf, index=1, value=1, mode=dr.ReduceMode.Local)

    dr.if_stmt(
        args = (),
        cond = dr.mask_t(t)(True, True, True, False, False),
        true_fn = true_fn,
        false_fn = false_fn,
        mode=mode
    )

    assert dr.all(buf == [3*(variant_2 & 1), 2 * ((variant_2 & 2) >> 1)])

    # Check that the scatter operation did not make unnecessary copies
    if mode == 'symbolic' and variant_2 == 3:
        transcript = capsys.readouterr().out
        assert transcript.count('[direct]') == 2-variant


@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('variant_2', [0, 1, 2, 3])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test11_side_effect_v2(t, mode, variant, variant_2, capsys, drjit_verbose):
    # The same test should work even when 'buf' is passed as part of 'args'
    if variant == 0:
        buf = t(0, 0)
    else:
        buf = dr.zeros(t, 2)

    def true_fn(buf):
        if variant_2 & 1:
            dr.scatter_add(buf, index=0, value=1, mode=dr.ReduceMode.Local)
        return (buf,)

    def false_fn(buf):
        if variant_2 & 2:
            dr.scatter_add(buf, index=1, value=1, mode=dr.ReduceMode.Local)
        return (buf,)

    buf, = dr.if_stmt(
        args = (buf,),
        cond = dr.mask_t(t)(True, True, True, False, False),
        true_fn = true_fn,
        false_fn = false_fn,
        arg_labels=('buf',),
        rv_labels=('buf',),
        mode=mode
    )

    assert dr.all(buf == [3*(variant_2 & 1), 2 * ((variant_2 & 2) >> 1)])

    # Check that the scatter operation did not make unnecessary copies
    if mode == 'symbolic' and variant_2 == 3:
        transcript = capsys.readouterr().out
        assert transcript.count('[direct]') == 2-variant


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test12_mutation_rules(t, mode):
    # Mutation rules for arguments and return values:
    #
    # Both 'true_fn' and 'false_fn' should receive the original function
    # argument values. (Even if they are called successively, and 'false_fn'
    # performed an in-place update)
    #
    # Following the operation, 'if_stmt' returns a modified version
    # of the original argument if it underwent an in-place update on
    # either branch. Otherwise, it returns a copy containing the result.

    def true_fn(x, y):
        x[0] += 1
        y[0] = y[0] + 1
        return x, y

    def false_fn(x, y):
        x[1] += 1
        y[1] = y[1] + 1
        return x, y

    xi0, xi1 = t(1, 2), t(3, 4)
    yi0, yi1 = t(10, 20), t(30, 40)
    xi, yi = [xi0, xi1], [yi0, yi1]

    xo, yo = dr.if_stmt(
        args=(xi, yi),
        cond=dr.mask_t(t)(True, False),
        true_fn=true_fn,
        false_fn=false_fn,
        arg_labels=('x', 'y'),
        rv_labels=('x', 'y'),
        mode=mode
    )

    xo0, xo1, yo0, yo1 = xo[0], xo[1], yo[0], yo[1]

    assert dr.all(xo0 == t(2, 2)) and dr.all(xo1 == t(3, 5))
    assert dr.all(yo0 == t(11, 20)) and dr.all(yo1 == t(30, 41))
    assert dr.all(yi0 == t(10, 20)) and dr.all(yi1 == t(30, 40))

    assert xo0 is xi0
    assert xo1 is xi1
    assert xo is xi
    assert yo0 is not yi0
    assert yo1 is not yi1
    assert yo is yi


def test13_limitations():
    # Test that issues related to current limitations of the AST processing
    # are correctly reported
    with pytest.raises(SyntaxError, match="use of 'return' inside a transformed 'while' loop or 'if' statement is currently not supported."):
        @dr.syntax
        def foo(x):
            if x < 0:
                return -x
    @dr.syntax
    def foo(x):
        if dr.hint(x < 0, mode='scalar'):
            return -x


@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax(recursive=True)
def test14_nested_ast_trafo(t):
    # Test that @dr.syntax works for local function declarations if called with recursive=True
    y = t(1, 2, 3)

    def g():
        if y < 2:
            z = t(-1)
        else:
            z = t(1)
        return z

    assert dr.all(g() == t(-1, 1, 1))


@pytest.test_arrays('uint32,jit,shape=(*)')
@dr.syntax
def test20_nested_if_stmt(t):
    # Test that a 3-level nested 'if' construction compiles
    # (yes, this was broken at some point!)
    x = dr.arange(t, 10)
    if x < 5:
        if x < 2:
            y = x + 1
        else:
            y = x - 1
    else:
        if x < 7:
            if x < 6:
                y = x + 1
            else:
                y = x - 1
        else:
            y = x - 1
    assert dr.all(y == [1, 2, 1, 2, 3, 6, 5, 6, 7, 8])


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test15_preserve_unchanged(t):
    # Check that unchanged variables (including differentiable ones) are simply
    # passed through
    a = t(1, 1)
    b = t(2, 2)
    dr.enable_grad(b)
    assert dr.grad_enabled(b)
    ai = (a.index, a.index_ad)
    bi = (b.index, b.index_ad)

    ao, bo = dr.if_stmt(
        args=(a, b),
        cond=dr.mask_t(t)(True, False),
        true_fn=lambda x, y: (x, y),
        false_fn=lambda x, y: (x, y))
    ai2 = (ao.index, ao.index_ad)
    bi2 = (bo.index, bo.index_ad)

    assert not dr.grad_enabled(ao)
    assert dr.grad_enabled(bo)
    assert ai == ai2
    assert bi == bi2


@pytest.test_arrays('uint32,is_diff,shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
def test16_mutate_dr_syntax(t, mode):
    # Simple test for array mutation/non-mutation combined with @dr.syntax

    @dr.syntax
    def maybe_dec(x, mode):
        if dr.hint(x > 1, mode=mode):
            x -= 1
        else:
            pass

    @dr.syntax
    def ret_dec(x, mode):
        if dr.hint(x > 1, mode=mode):
            x = x - 1
        else:
            pass
        return x

    x = dr.arange(t, 3)
    xo = x
    maybe_dec(x, mode)
    assert xo is x
    assert dr.all(xo == (0, 1, 1))

    x = dr.arange(t, 3)
    xo = x
    y = ret_dec(x, mode)
    assert xo is x
    assert dr.all(xo == (0, 1, 2))
    assert dr.all(y == (0, 1, 1))


@pytest.test_arrays('uint32,is_diff,shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
def test23_mutate_dr_syntax_v2(t, mode):
    # Simple test for array mutation/non-mutation combined with @dr.syntax

    @dr.syntax
    def maybe_dec(x, mode):
        if dr.hint(x <= 1, mode=mode):
            pass
        else:
            x -= 1

    @dr.syntax
    def ret_dec(x, mode):
        if dr.hint(x <= 1, mode=mode):
            pass
        else:
            x = x - 1
        return x

    x = dr.arange(t, 3)
    xo = x
    maybe_dec(x, mode)
    assert xo is x
    assert dr.all(xo == (0, 1, 1))

    x = dr.arange(t, 3)
    xo = x
    y = ret_dec(x, mode)
    assert xo is x
    assert dr.all(xo == (0, 1, 2))
    assert dr.all(y == (0, 1, 1))


@pytest.test_arrays('uint32,is_diff,shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('tt', ['tuple', 'list', 'dict', 'dataclass', 'nested'])
@pytest.mark.parametrize('mutate', [True, False])
def test17_mutate_other_containers(t, tt, mutate, mode):
    # One last test about mutation/non-mutation involving lots of PyTree types

    if tt == 'tuple' and not mutate:
        return

    x = t(10, 20)
    y = t(30, 40)

    def true_fn(z):
        if mutate:
            if tt == 'dict':
                z['x'] += 1
            elif tt == 'dataclass':
                z.x += 1
            elif tt == 'tuple':
                z0 = z[0]
                z0 += 1
            else:
                z[0] += 1
        else:
            if tt == 'dict':
                z['x'] = z['x'] + 1
            elif tt == 'dataclass':
                z.x = z.x + 1
            else:
                z[0] = z[0] + 1
        return (z,)

    def false_fn(z):
        return (z,)

    if tt == 'tuple':
        z = (x, y)
    elif tt == 'list':
        z = [x, y]
    elif tt == 'dict':
        z = {'x': x, 'y': y}

    elif tt == 'dataclass':
        from dataclasses import dataclass
        @dataclass
        class Z:
            x: t = t(0)
            y: t = t(0)
        z = Z(x, y)
    elif tt == 'nested':
        Z = sys.modules[t.__module__].Array2u
        z = Z(x, y)
    else:
        raise Exception("Unknown case")

    zo, = dr.if_stmt(
        args = (z,),
        cond = dr.mask_t(t)(True, False),
        true_fn = true_fn,
        false_fn = false_fn,
        mode=mode,
        arg_labels=('z',),
        rv_labels=('z',)
    )

    if tt == 'dataclass':
        assert dr.all(zo.x == (11, 20)) and dr.all(zo.y == (30, 40))
        assert zo.y is y
        if mutate:
            assert zo.x is x
        else:
            assert zo.x is not x
    elif tt == 'dict':
        assert dr.all(zo['x'] == (11, 20)) and dr.all(zo['y']== (30, 40))
        assert zo['y'] is y
        if mutate:
            assert zo['x'] is x
        else:
            assert zo['x'] is not x
    else:
        assert dr.all(zo[0] == (11, 20)) and dr.all(zo[1] == (30, 40))
        if tt == 'nested':
            assert zo[1].index is y.index and zo[1] is not y
            assert dr.all(z[0] == (11, 20))
        else:
            assert zo[1] is y
            if mutate:
                assert zo[0] is x
            else:
                assert zo[0] is not x

    assert dr.all(x == (10 + (mutate and tt != 'nested'), 20))
    assert dr.all(y == (30, 40))
