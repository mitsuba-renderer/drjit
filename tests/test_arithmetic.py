import drjit as dr
import pytest
import sys

@pytest.test_arrays('-bool,shape=(3, *)', '-bool,shape=(*)')
def test01_comparison(t):
    m = dr.mask_t(t)
    assert dr.all(t(1, 2, 3) == t(1, 2, 3))
    assert not dr.all(t(1, 2, 3) == t(1, 3, 3))
    assert dr.all(t(1, 2, 3) != t(4, 5, 6))
    assert not dr.all(t(1, 2, 3) != t(4, 2, 6))
    assert dr.any(t(1, 2, 3) == t(1, 2, 3))
    assert not dr.any(t(1, 2, 3) == t(4, 5, 6))
    assert dr.any(t(1, 2, 3) != t(1, 3, 3))
    assert not dr.any(t(1, 2, 3) != t(1, 2, 3))

    assert dr.all((t(1, 2, 3) < t(0, 2, 4)) == m(False, False, True))
    assert dr.all((t(1, 2, 3) <= t(0, 2, 4)) == m(False, True, True))
    assert dr.all((t(1, 2, 3) > t(0, 2, 4)) == m(True, False, False))
    assert dr.all((t(1, 2, 3) >= t(0, 2, 4)) == m(True, True, False))
    assert dr.all((t(1, 2, 3) == t(0, 2, 4)) == m(False, True, False))
    assert dr.all((t(1, 2, 3) != t(0, 2, 4)) == m(True, False, True))

    with pytest.raises(RuntimeError, match='Incompatible arguments'):
        mod = sys.modules[t.__module__]
        mod.Array3f(1, 2, 3) + mod.Complex2f(1, 2)

    with pytest.raises(RuntimeError, match='Incompatible arguments'):
        mod = sys.modules[t.__module__]
        mod.Array3f(1, 2, 3) + mod.Array4f(1, 2, 3, 4)

def test02_allclose():
    assert dr.allclose(2, 2)
    assert dr.allclose([1, 2, 3], [1, 2, 3])
    assert dr.allclose([1, 1, 1], 1)
    assert dr.allclose([[1, 1], [1, 1], [1, 1]], 1)
    assert not dr.allclose(2, 3)
    assert not dr.allclose([1, 2, 3], [1, 4, 3])
    assert not dr.allclose([1, 1, 1], 2)
    assert not dr.allclose(float('nan'), float('nan'))
    assert dr.allclose(float('nan'), float('nan'), equal_nan=True)

    with pytest.raises(RuntimeError) as ei:
        assert not dr.allclose([1,2,3], [1,4])
    assert 'incompatible sizes' in str(ei.value)

    np = pytest.importorskip("numpy")
    assert dr.allclose(np.array([1, 2, 3]), [1, 2, 3])
    assert dr.allclose(np.array([1, 2, 3]), dr.scalar.Array3f(1, 2, 3))
    assert dr.allclose(np.array([1, float('nan'), 3.0]), [1, float('nan'), 3], equal_nan=True)

@pytest.test_arrays('-bool,shape=(3)', '-bool,shape=(3, *)', '-bool,shape=(*, *)')
def test03_binop_simple(t):
    a = t(1, 2, 3)
    assert dr.all(a + a == t(2, 4, 6))
    assert dr.all(a + (1, 2, 3) == t(2, 4, 6))
    assert dr.all(a + [1, 2, 3] == t(2, 4, 6))
    assert dr.all((1, 2, 3) + a == t(2, 4, 6))
    assert dr.all([1, 2, 3] + a == t(2, 4, 6))
    assert dr.all(a - a == t(0, 0, 0))
    assert dr.all(a * a == t(1, 4, 9))

    if dr.is_float_v(a):
        assert dr.all(a / a == t(1, 1, 1))
        with pytest.raises(TypeError, match=r'unsupported operand type\(s\)'):
            a // a
    else:
        assert dr.all(a // a == t(1, 1, 1))
        with pytest.raises(TypeError, match=r'unsupported operand type\(s\)'):
            a / a

    if dr.is_integral_v(a):
        assert dr.all(a << 1 == t(2, 4, 6))
        assert dr.all(a >> 1 == t(0, 1, 1))

    m = dr.mask_t(t)
    assert dr.all(m(True, False, True) | True == m(True, True, True))
    assert dr.all(m(True, False, True) & False == m(False, False, False))
    assert dr.all(m(True, False, True) ^ True == m(False, True, False))

@pytest.test_arrays('bool,shape=(*)', 'bool,shape=(4)', 'bool,shape=(4, *)', 'bool,shape=(*, *)')
def test04_binop_bool(t):
    assert dr.all(t([True, True, False, False]) & t([True, False, True, False]) == t(True, False, False, False))
    assert dr.all(t([True, True, False, False]) | t([True, False, True, False]) == t(True, True, True, False))
    assert dr.all(t([True, True, False, False]) ^ t([True, False, True, False]) == t(False, True, True, False))

@pytest.test_arrays('-bool,shape=(3)', '-bool,shape=(3, *)', '-bool, shape=(*, *)')
def test05_binop_inplace(t):
    a = t(1, 2, 3)
    b = t(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == t(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == t(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == t(3, 4, 2))

    if dr.is_float_v(t):
        a = dr.int_array_t(t)(1)
        c = a
        a += b
        assert a is not c and type(a) is t and dr.all(a == t(3, 4, 2))

    if dr.size_v(t) == dr.Dynamic:
        a = t(1)
        b = t([1, 2, 3], 2, 3)
        c = a
        a += b
        assert a is c
        assert len(a) == 3
        assert len(a[0]) == 3 and len(a[1]) == 1 and len(a[2]) == 1
        assert dr.all(a[0] == dr.value_t(t)(2, 3, 4))
        assert a[1] == 3
        assert a[2] == 4

@pytest.test_arrays('type=int32,shape=(3)', 'type=int32,shape=(3, *)', 'type=int32, shape=(*, *)')
def test05_unop(t):
    assert dr.all(-t(1, 2, 3) == t(-1, -2, -3))
    assert dr.all(+t(1, 2, 3) == t(1, 2, 3))
    assert dr.all(abs(t(1, -2, 3)) == t(1, 2, 3))
    m = dr.mask_t(t)
    assert dr.all(~m(True, False, True) == m(False, True, False))

def test06_select():
    import drjit.scalar as s
    assert dr.select(True, "hello", "world") == "hello"
    result = dr.select(s.Array2b(True, False), 1, 2)
    assert isinstance(result, s.Array2i) and dr.all(result == s.Array2i(1, 2))
    result = dr.select(s.Array2b(True, False), 1, 2.0)
    assert isinstance(result, s.Array2f) and dr.all(result == s.Array2f(1, 2))
    result = dr.select(s.ArrayXb(True, False), 1, 2)
    assert isinstance(result, s.ArrayXi) and dr.all(result == s.ArrayXi(1, 2))
    result = dr.select(s.ArrayXb(True, False), 1, 2.0)
    assert isinstance(result, s.ArrayXf) and dr.all(result == s.ArrayXf(1, 2))

    result = dr.select(s.Array2b(True, False), s.Array2i(3, 4), s.Array2i(5, 6))
    assert isinstance(result, s.Array2i) and dr.all(result == s.Array2i(3, 6))
    result = dr.select(s.Array2b(True, False), s.Array2i(3, 4), s.Array2f(5, 6))
    assert isinstance(result, s.Array2f) and dr.all(result == s.Array2f(3, 6))
    result = dr.select(s.ArrayXb(True, False), s.ArrayXi(3, 4), s.ArrayXi(5, 6))
    assert isinstance(result, s.ArrayXi) and dr.all(result == s.ArrayXi(3, 6))
    result = dr.select(s.ArrayXb(True, False), s.ArrayXi(3, 4), s.ArrayXf(5, 6))
    assert isinstance(result, s.ArrayXf) and dr.all(result == s.ArrayXf(3, 6))

    result = dr.select(True, 1, s.ArrayXf(5, 6))
    assert isinstance(result, s.ArrayXf) and dr.all(result == s.ArrayXf(1, 1))

    try:
        import drjit.llvm as l
        success = True
    except:
        success = False

    if success:
        result = dr.select(l.Array2b(True, False), 1, 2)
        assert isinstance(result, l.Array2i) and dr.all(result == l.Array2i(1, 2))
        result = dr.select(l.Array2b(True, False), 1, 2.0)
        assert isinstance(result, l.Array2f) and dr.all(result == l.Array2f(1, 2))
        result = dr.select(l.ArrayXb(True, False), 1, 2)
        assert isinstance(result, l.ArrayXi) and dr.all(result == l.ArrayXi(1, 2))
        result = dr.select(l.ArrayXb(True, False), 1, 2.0)
        assert isinstance(result, l.ArrayXf) and dr.all(result == l.ArrayXf(1, 2))

        result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2i(5, 6))
        assert isinstance(result, l.Array2i) and dr.all(result == l.Array2i(3, 6))
        result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2f(5, 6))
        assert isinstance(result, l.Array2f) and dr.all(result == l.Array2f(3, 6))
        result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXi(5, 6))
        assert isinstance(result, l.ArrayXi) and dr.all(result == l.ArrayXi(3, 6))
        result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXf(5, 6))
        assert isinstance(result, l.ArrayXf) and dr.all(result == l.ArrayXf(3, 6))

