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
    r = dr.if_stmt(
        args = (t(4),),
        cond = dr.mask_t(t)(True, False),
        true_fn = lambda x: x + 1,
        false_fn = lambda x: x + 2,
        mode=mode
    )
    assert dr.all(r == t(5, 6))


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test04_test_inconsistencies(t, mode):
    # Test a few problem cases -- mismatched types/sizes (2 flavors)
    with pytest.raises(RuntimeError, match="inconsistent types"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x: x + 1,
            false_fn = lambda x: x > 2,
            mode=mode
        )

    with pytest.raises(RuntimeError, match="incompatible sizes"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x: t(1, 2),
            false_fn = lambda x: t(1, 2, 3),
            mode=mode
        )

    with pytest.raises(RuntimeError, match="incompatible sizes"):
        dr.if_stmt(
            args = (t(4),),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x: t(1, 2, 3),
            false_fn = lambda x: t(1, 2, 3),
            mode=mode
        )

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test05_uninitialized_input(t, mode):
    with pytest.raises(RuntimeError, match="field 'y' is uninitialized"):
        dr.if_stmt(
            args = (t(4),t()),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, y),
            false_fn = lambda x, y: (x > 2, y),
            rv_labels = ('x', 'y'),
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
    with pytest.raises(RuntimeError, match="field 'y' is uninitialized"):
        dr.if_stmt(
            args = (t(4),t(3)),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, y),
            false_fn = lambda x, y: (x > 2, t()),
            rv_labels = ('x', 'y'),
            mode=mode
        )

    with pytest.raises(RuntimeError, match="field 'y' is uninitialized"):
        dr.if_stmt(
            args = (t(4),t(3)),
            cond = dr.mask_t(t)(True, False),
            true_fn = lambda x, y: (x > 1, t()),
            false_fn = lambda x, y: (x > 2, y),
            rv_labels = ('x', 'y'),
            mode=mode
        )

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test06_test_exceptions(t, mode):
    # Exceptions raise in 'true_fn' and 'false_fn' should be correctly propagated
    def my_fn(x):
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
            true_fn = lambda x: t(1, 2),
            false_fn = my_fn,
            mode=mode
        )
    assert "foo" in str(e.value.__cause__)

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test07_test_syntax_simple(t, mode):
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
def test08_test_mutate(t, mode):
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

    with pytest.raises(RuntimeError, match="inconsistent scalar Python object of type 'int' for field 'y'"):
        f(t(1,2,3,4), t, mode)


@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test10_side_effect(t, mode, variant, capsys, drjit_verbose):
    # Ensure that side effects are correctly tracked in evaluated and symbolic modes
    if variant == 0:
        buf = t(0, 0)
    else:
        buf = dr.zeros(t, 2)

    def true_fn():
        dr.scatter_add(buf, index=0, value=1)

    def false_fn():
        dr.scatter_add(buf, index=1, value=1)

    dr.if_stmt(
        args = (),
        cond = dr.mask_t(t)(True, True, True, False, False),
        true_fn = true_fn,
        false_fn = false_fn,
        mode=mode
    )

    assert dr.all(buf == [3, 2])

    # Check that the scatter operation did not make unnecessary copies
    if mode == 'symbolic':
        transcript = capsys.readouterr().out
        assert transcript.count('[direct]') == 2-variant


@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test11_side_effect_v2(t, mode, variant, capsys, drjit_verbose):
    # The same test should work even when 'buf' is passed as part of 'args'
    if variant == 0:
        buf = t(0, 0)
    else:
        buf = dr.zeros(t, 2)

    def true_fn(buf):
        dr.scatter_add(buf, index=0, value=1)
        return buf

    def false_fn(buf):
        dr.scatter_add(buf, index=1, value=1)
        return buf

    buf = dr.if_stmt(
        args = (buf,),
        cond = dr.mask_t(t)(True, True, True, False, False),
        true_fn = true_fn,
        false_fn = false_fn,
        mode=mode
    )

    assert dr.all(buf == [3, 2])

    # Check that the scatter operation did not make unnecessary copies
    if mode == 'symbolic':
        transcript = capsys.readouterr().out
        assert transcript.count('[direct]') == 2-variant

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test11_do_not_copy_unchanged(t, mode):
    # When parts of PyTrees aren't modified by 'true_fn' / 'false_fn', then these
    # should come out unchanged (even the Python object should be the same one)

    m = sys.modules[t.__module__]

    def true_fn(x, y, z):
        return (x + 1, y, z)

    def false_fn(x, y, z):
        z['b'] += 1
        return (x + 2, y, z)

    xi = t(1, 2)
    yi = t(3, 4)
    zi = {
        'a' : (
            [
                m.Array3f(3, 4, 5)
            ],
        ),
        'b' : t(3)
    }

    zi_1 = id(zi)
    zi_2 = id(zi['a'])
    zi_3 = id(zi['a'][0])
    zi_4 = id(zi['a'][0][0])
    zi_5 = id(zi['b'])

    xo, yo, zo = dr.if_stmt(
        args = (xi, yi, zi),
        cond = dr.mask_t(t)(True, False),
        true_fn = true_fn,
        false_fn = false_fn,
        mode=mode
    )

    zo_1 = id(zo)
    zo_2 = id(zo['a'])
    zo_3 = id(zo['a'][0])
    zo_4 = id(zo['a'][0][0])
    zo_5 = id(zo['b'])

    assert xo is not xi
    assert yo is yi
    assert zi_4 == zo_4
    assert zi_3 == zo_3
    assert zi_2 == zo_2
    assert zi_1 != zo_1
    assert zi_5 != zo_5
    assert dr.all(xo == [2, 4])


def test12_limitations():
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


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
def test13_ad_fwd(t, mode):
    @dr.syntax
    def f(x, mode):
        if dr.hint(x < 5, mode=mode):
            y = 10*x
        else:
            y = 100*x
        return x, y

    x = dr.arange(t, 10)
    dr.enable_grad(x)

    xo, yo = f(x, mode)
    assert dr.all(xo == dr.arange(t, 10))
    assert dr.all(yo == [0, 10, 20, 30, 40, 500, 600, 700, 800, 900])
    dr.forward_from(x, flags=0)
    assert dr.all(dr.grad(xo) == dr.full(t, 1, 10))
    assert dr.all(dr.grad(yo) == [10, 10, 10, 10, 10, 100, 100, 100, 100, 100])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
def test14_ad_bwd(t, mode):
    @dr.syntax
    def f(x, mode):
        if dr.hint(x < 5, mode=mode):
            y = 10*x
        else:
            y = 100*x
        return y

    x = dr.arange(t, 10)
    dr.enable_grad(x)

    y = f(x, mode)
    dr.backward_from(y)
    assert dr.all(dr.grad(x) == [10, 10, 10, 10, 10, 100, 100, 100, 100, 100])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test15_ad_fwd_implicit_dep(t, mode):
    y = t(1)
    dr.enable_grad(y)
    dr.set_grad(y, 1)

    x = dr.arange(t, 10)

    if dr.hint(x < 5, exclude=[y]):
        z = x*y
    else:
        z = x-y

    dr.forward_to(z)
    assert dr.all(dr.grad(z) == [0, 1, 2, 3, 4, -1, -1, -1, -1, -1])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test16_ad_bwd_implicit_dep(t, mode):
    y = t(1)
    dr.enable_grad(y)
    dr.set_grad(y, 1)

    x = dr.arange(t, 10)

    if dr.hint(x < 5, exclude=[y]):
        z = x*y
    else:
        z = x-y

    dr.backward_from(z)
    assert dr.all(dr.grad(y) == 6)
