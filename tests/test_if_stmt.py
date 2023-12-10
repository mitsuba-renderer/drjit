import drjit as dr
import pytest

@pytest.mark.parametrize('cond', [True, False])
def test01_scalar(cond):
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
def test05_test_exceptions(t, mode):
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
def test06_test_syntax_simple(t, mode):
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
def test07_incompatible_scalar(t, mode):
    @dr.syntax
    def f(x, t, mode):
        if dr.hint(x > 2, mode=mode):
            y = 5
        else:
            y = 10

    with pytest.raises(RuntimeError, match="inconsistent scalar Python object of type 'int' for field 'y'"):
        f(t(1,2,3,4), t, mode)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test08_side_effect(t, mode):
    buf = t(0, 0)

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
