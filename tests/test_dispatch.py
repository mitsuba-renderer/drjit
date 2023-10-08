import drjit as dr
import pytest
import sys

def test01_switch_scalar():
    c = [
        lambda x: x+1,
        lambda x: x*10
    ]

    assert dr.switch(0, c, 5) == 6
    assert dr.switch(1, c, 5) == 50
    assert dr.switch(0, c, x=5) == 6
    assert dr.switch(1, c, x=5) == 50
    assert dr.switch(1, c, True, x=5) == 50
    assert dr.switch(1, c, False, x=5) is None
    assert dr.switch(1, c, x=5, active=False) is None

@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test02_switch_vec_simple(t, recorded, drjit_verbose, capsys):
    #with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded)
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

    # two kernel launches (not merged!) with 2 inputs/outputs and 2 side effects
    out = capsys.readouterr().out
    assert out.count("(n=2, in=4, out=0, se=2") == 2

@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('int32,-uint32,shape=(*),jit')
def test03_switch_vec_masked(t, recorded):
    #with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded)
    Int = t
    UInt32 = dr.uint32_array_t(Int)
    Bool = dr.mask_t(Int)

    c = [
        lambda a, b: (a * 4, Int(2)),
        lambda a, b: (a * 8, -b)
    ]

    index = UInt32(0, 0, 1, 1)
    a = Int(1, 2, 3, 4)
    b = Int(1)
    active = Bool(True, False, True, False)

    # Masked case
    result = dr.switch(index, c, a, b, active)
    assert dr.allclose(result, [[4, 0, 24, 0], [2, 0, -1, 0]])

@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test04_failure_incompatible_dict(t, recorded):
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            callables=(
                lambda a: dict(a=a),
                lambda a: dict(b=a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "error encountered while processing arguments of type 'dict' and 'dict': dictionaries have incompatible keys (['b'] vs ['a'])" in str(e.value.__cause__)


@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test05_failure_incompatible_shape(t, recorded):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            callables=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXu(a, a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "incompatible input lengths (3 and 2)" in str(e.value.__cause__)

    r = dr.switch(
        index=t(0, 0, 1, 1),
        callables=(
            lambda a: m.ArrayXu(a*1, a*2),
            lambda a: m.ArrayXu(a*3, a*4)
        ),
        a=t(1, 2, 3, 4)
    )
    assert dr.all(r == m.ArrayXu(
        t(1, 2, 9, 12),
        t(2, 4, 12, 16)
    ), axis=None)


@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('uint32,shape=(*),jit')
def test06_failure_incompatible_type(t, recorded):
    m = sys.modules[t.__module__]
    with pytest.raises(RuntimeError) as e:
        dr.switch(
            index=t(0, 0, 1, 1),
            callables=(
                lambda a: m.ArrayXu(a, a),
                lambda a: m.ArrayXf(a, a)
            ),
            a=t(1, 2, 3, 4)
        )
    assert "incompatible input types" in str(e.value.__cause__)


# Keyword calling, pytrees, differentiation
@pytest.mark.parametrize("recorded", [True, False])
@pytest.test_arrays('float,shape=(*),jit')
def test04_complex(t, recorded):
    #with dr.scoped_set_flag(dr.JitFlag.VCallRecord, recorded)
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

    c = [
        f0, f1
    ]

    index = UInt32(0, 0, 1, 1)
    a = {
        'key': t(1, 2, 3, 4)
    }
    b = (
        t(2),
        t(10, 11, 12, 13)
    )
    expected = {
        'rv0' : a['key'] *t(5, 5, 3, 3),
        'rv1' : (
            t(2, 2, 12, 13),
            t(10, 11, 2, 2)
        )
    }

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
