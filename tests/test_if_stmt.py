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
