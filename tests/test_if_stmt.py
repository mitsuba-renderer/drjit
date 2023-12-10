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
@pytest.mark.parametrize('mode', ['evaluate', 'symbolic'])
@pytest.mark.parametrize('cond', [True, False])
def test02_scalar_jit(cond):
    r = dr.if_stmt(
        args = (4,),
        cond = dr.mask_t(t)(True),
        true_fn = lambda x: x + 1,
        false_fn = lambda x: x + dr.opaque(t, 2),
        mode=mode
    )

    assert r.state == dr.VarState.Literal
    assert r == (5 if cond else 6)
