import drjit as dr
import pytest

def test01_all_any_scalar():
    from drjit.scalar import Array2b, ArrayXb

    assert dr.all(True) == True
    assert dr.all(False) == False
    assert dr.any(True) == True
    assert dr.any(False) == False
    assert dr.any(()) == False
    assert dr.all(()) == True
    assert dr.all((True,)) == True
    assert dr.all((False,)) == False
    assert dr.any((True,)) == True
    assert dr.any((False,)) == False
    assert dr.all([True, True]) == True
    assert dr.all([True, False]) == False
    assert dr.all([False, False]) == False
    assert dr.any([True, True]) == True
    assert dr.any([True, False]) == True
    assert dr.any([False, False]) == False
    assert type(dr.all(Array2b(True, True))) is bool
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, True)) == True
    assert dr.all(Array2b(True, False)) == False
    assert dr.all(Array2b(False, False)) == False
    assert dr.any(Array2b(True, True)) == True
    assert dr.any(Array2b(True, False)) == True
    assert dr.any(Array2b(False, False)) == False
    assert type(dr.all(ArrayXb(True, True))) is ArrayXb
    assert len(dr.all(ArrayXb(True, True))) == 1
    assert dr.all(ArrayXb(True, True))[0] == True
    assert dr.all(ArrayXb(True, False))[0] == False
    assert dr.all(ArrayXb(False, False))[0] == False
    assert dr.any(ArrayXb(True, True))[0] == True
    assert dr.any(ArrayXb(True, False))[0] == True
    assert dr.any(ArrayXb(False, False))[0] == False


# Tests dr.{any/all}[_nested] and implicit conversion to 'bool'
@pytest.test_arrays('shape=(1, *), bool')
def test02_any_all_nested(t):
    t0 = dr.value_t(t)
    v0 = t0([True, False, False])

    v = t(v0)
    assert len(v) == 1 and len(v0) == 3

    v0_all = dr.all(v0)
    v0_any = dr.any(v0)
    assert type(v0_all) is t0 and len(v0_all) == 1 and \
           type(v0_any) is t0 and len(v0_any) == 1

    assert bool(v0_all) == False
    assert bool(v0_any) == True

    va = dr.all(v)
    assert type(va) is t0 and len(va) == 3

    van = dr.all_nested(v)
    assert type(van) is t0 and len(van) == 1

    with pytest.raises(TypeError) as ei:
        dr.all((True, "hello"))

    assert "unsupported operand type(s)" in str(ei.value)


@pytest.test_arrays('shape=(1, *)')
def test03_implicit_bool_conversion_failures(t):
    if not dr.is_mask_v(t):
        with pytest.raises(TypeError, match=r'implicit conversion to \'bool\' is only supported for scalar mask arrays.'):
            bool(t()[0])
    else:
        with pytest.raises(RuntimeError, match=r'implicit conversion to \'bool\' requires an array with at most 1 dimension \(this one has 2 dimensions\).'):
            bool(t())
        with pytest.raises(RuntimeError, match=r'implicit conversion to \'bool\' requires an array with at most 1 element \(this one has 2 elements\).'):
            bool(dr.value_t(t)(True, False))
