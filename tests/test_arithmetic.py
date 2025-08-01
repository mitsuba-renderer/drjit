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

    if not dr.is_half_v(a):
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
    m = dr.mask_t(t)([False, True, False])
    a[m] += b
    assert a is c and dr.all(a == t(4, 9, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == t(3, 4, 2))

    if dr.is_float_v(t):
        a = dr.int32_array_t(t)(1) if dr.is_half_v(t) else dr.int_array_t(t)(1)
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

    if dr.is_jit_v(t) and dr.size_v(t) == 3:
        vt = dr.value_t(t)
        v = t(1, 2, 3)
        v.x += 5
        vy = v.y
        vy += 3
        vz = v[2]
        vz += 1
        assert dr.all(v == t(6, 5, 4))
        assert v.y is vy
        assert v.z is vz
        del vz
        del vy


@pytest.test_arrays('type=int32,shape=(3)', 'type=int32,shape=(3, *)', 'type=int32, shape=(*, *)')
def test05_unop(t):
    assert dr.all(~t(0) == t(-1))
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
    result = dr.select(s.ArrayXb(True, False, True), True, False)
    assert isinstance(result, s.ArrayXb) and dr.all(result == s.ArrayXb(True, False, True))

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
        result = dr.select(l.ArrayXb(True, False, True), True, False)
        assert isinstance(result, l.ArrayXb) and dr.all(result == l.ArrayXb(True, False, True))

        result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2i(5, 6))
        assert isinstance(result, l.Array2i) and dr.all(result == l.Array2i(3, 6))
        result = dr.select(l.Array2b(True, False), l.Array2i(3, 4), l.Array2f(5, 6))
        assert isinstance(result, l.Array2f) and dr.all(result == l.Array2f(3, 6))
        result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXi(5, 6))
        assert isinstance(result, l.ArrayXi) and dr.all(result == l.ArrayXi(3, 6))
        result = dr.select(l.ArrayXb(True, False), l.ArrayXi(3, 4), l.ArrayXf(5, 6))
        assert isinstance(result, l.ArrayXf) and dr.all(result == l.ArrayXf(3, 6))

        # Test select on PyTree types (DRJIT_STRUCT, dataclasses, sequences, dicts)
        class MyPoint2f:
            DRJIT_STRUCT = { 'x' : l.Float, 'y': l.Float }

            def __init__(self, x: l.Float = l.Float(), y: l.Float = l.Float()):
                self.x = x
                self.y = y

        from dataclasses import dataclass
        @dataclass
        class Foo:
            x: l.Float = l.Float(0)
            y: l.Float = l.Float(0)

        m = l.Bool(False, False, True, True)
        result = dr.select(m, dr.zeros(MyPoint2f, 4), dr.ones(MyPoint2f, 4))
        assert isinstance(result, MyPoint2f)
        result = dr.select(m, dr.zeros(Foo, 4), dr.ones(Foo, 4))
        assert isinstance(result, Foo)
        result = dr.select(m, [l.Float(1), l.Float(2)], [l.Float(-1), l.Float(-2)])
        assert str(result) == '[[-1, -1, 1, 1], [-2, -2, 2, 2]]'
        result = dr.select(m, (l.Float(1), l.Float(2)), (l.Float(-1), l.Float(-2)))
        assert str(result) == '([-1, -1, 1, 1], [-2, -2, 2, 2])'
        result = dr.select(m, { 'a' : l.Float(1), 'b' : l.Float(2) },
                         { 'a' : l.Float(-1), 'b' : l.Float(-2) })
        assert str(result) == "{'a': [-1, -1, 1, 1], 'b': [-2, -2, 2, 2]}"

        a = l.Quaternion4f(1.0, 2.0, 3.0, 4.0)
        b = l.Quaternion4f(4.0, 3.0, 2.0, 1.0)
        result = dr.select(dr.isnan(a), b, a)
        assert isinstance(result, l.Quaternion4f)
        result = dr.select(dr.isnan(a), a, 0)
        assert isinstance(result, l.Quaternion4f)
        result = dr.select(dr.isnan(a), 0, a)
        assert isinstance(result, l.Quaternion4f)
        result = dr.select(True, a, 0)
        assert isinstance(result, l.Quaternion4f)
        result = dr.select(l.Array4b([True, False], [False, True], [True, False], [True, True]), a, 0)
        assert isinstance(result, l.Quaternion4f)
        assert str(result) == '[1i+3k+4,\n 2j+4]'
        with pytest.raises(RuntimeError) as e:
            result = dr.select(l.Array2b(True, False), a, 0)

        a = l.Complex2f(1.0, 2.0)
        b = l.Complex2f(4.0, 3.0)
        result = dr.select(dr.isnan(a), b, a)
        assert isinstance(result, l.Complex2f)
        result = dr.select(dr.isnan(a), a, 0)
        assert isinstance(result, l.Complex2f)
        result = dr.select(dr.isnan(a), 0, a)
        assert isinstance(result, l.Complex2f)
        result = dr.select(True, a, 0)
        assert isinstance(result, l.Complex2f)
        result = dr.select(l.Array2b([True, False], [False, True]), a, 0)
        assert isinstance(result, l.Complex2f)
        assert(str(result) == '[1,\n 2j]')
        with pytest.raises(RuntimeError) as e:
            result = dr.select(l.Array4b(True, False, True, True), a, 0)

        assert dr.select(True, None, None) is None
        assert dr.select(l.Array1b(True), None, None) is None
        class Dummy(int):
            pass
        with pytest.raises(RuntimeError, match=r"encountered incompatible objects with an unknown type \(not a Dr.Jit array, not a PyTree\)."):
            dr.select(l.Array2b(True, False), Dummy(1), Dummy(2))

@pytest.test_arrays('type=float32,shape=(*)')
def test07_power(t):
    assert dr.allclose(t(2)**0, t(1))
    assert dr.allclose(t(2)**1, t(2))
    assert dr.allclose(t(2)**-1, t(1/2))
    assert dr.allclose(t(2)**-13, t(2**-13))
    assert dr.allclose(t(2)**13, t(2**13))
    assert dr.allclose(t(2)**2.5, t(2**2.5))
    assert dr.allclose(t(2)**-2.5, t(2**-2.5))


@pytest.test_arrays('type=int32,jit,shape=(*)')
@pytest.mark.parametrize("value", [True, False])
def test08_scoped_set_flag_const_prop(t, value):
    with dr.scoped_set_flag(dr.JitFlag.ConstantPropagation, value):
        # Create two literal constant arrays
        a, b = t(4), t(5)

        # This addition operation can be immediately performed and does not need to be recorded
        c1 = a + b

        # Double-check that c1 and c2 refer to the same Dr.Jit variable
        c2 = t(9)
        assert (c1.index == c2.index) == value


@pytest.test_arrays('type=int32,jit,shape=(*)')
@pytest.mark.parametrize("value", [True, False])
def test09_scoped_set_flag_lvn(t, value):
    with dr.scoped_set_flag(dr.JitFlag.ValueNumbering, value):
        # Create two nonliteral arrays stored in device memory
        a, b = t(1, 2, 3), t(4, 5, 6)

        # Perform the same arithmetic operation twice
        c1 = a + b
        c2 = a + b

        # Verify that c1 and c2 reference the same Dr.Jit variable
        assert (c1.index == c2.index) == value


@pytest.test_arrays('type=int32,jit,shape=(3, *)')
def test10_state(t):
    a = t(1, 2, 3)
    assert a.state == dr.VarState.Literal
    a = a + 1
    assert a.state == dr.VarState.Literal

    a.x = type(a.x)(1,2,3)
    assert a.x.state == dr.VarState.Evaluated
    assert a.state == dr.VarState.Mixed
    a.y = a.x + 1
    assert a.y.state == dr.VarState.Unevaluated
    assert a.state == dr.VarState.Mixed
    a.z = type(a.x)()
    assert a.z.state == dr.VarState.Invalid
    assert a.state == dr.VarState.Mixed


@pytest.test_arrays('type=float32,jit,shape=(*)', 'type=float32,jit,shape=(3, *)')
def test11_reinterpret(t):
    U = dr.uint32_array_t(t)
    U64 = dr.uint64_array_t(t)
    v = dr.reinterpret_array(U, t(1.0))
    assert v[0] == 0x3f800000
    v2 = dr.reinterpret_array(t, v)
    assert v2[0] == 1.0
    with pytest.raises(RuntimeError) as e:
        dr.reinterpret_array(U64, t(1.0))
    e = e.value
    if hasattr(e, '__cause__') and e.__cause__ is not None:
        e = e.__cause__
    assert 'cannot reinterpret-cast between types of different size' in str(e)


@pytest.test_arrays('float,jit,shape=(3, *)')
def test12_div_via_rcp(t, drjit_verbose, capsys):
    x = dr.opaque(t, 3)
    x / dr.opaque(dr.value_t(t), 4)
    transcript = capsys.readouterr().out
    assert transcript.count('= mul(') == 3
    assert transcript.count('= div(') + transcript.count('= rcp(') + transcript.count('= div.approx(') + transcript.count('= rcp.approx(') == 1


@pytest.test_arrays('float, jit, shape=(*)', 'float, shape=(3)')
def test13_hypot(t):
    assert dr.allclose(dr.hypot(t(3), t(4)), t(5))


@pytest.test_arrays('uint32, jit, shape=(*)', 'uint32, jit, shape=(*)')
def test14_lzcnt(t):
    bits = 64 if '64' in t.__name__ else 32
    out = (bits, bits-1, bits-7)
    assert dr.all(dr.lzcnt(t(0, 1, 100)) == t(out))
    assert tuple(dr.lzcnt(t(i))[0] for i in (0, 1, 100)) == out
    if bits == 32:
        assert tuple(dr.lzcnt(i) for i in (0, 1, 100)) == out


@pytest.test_arrays('uint32, jit, shape=(*)', 'uint64, jit, shape=(*)')
def test15_tzcnt(t):
    bits = 64 if '64' in t.__name__ else 32
    assert dr.all(dr.tzcnt(t(0, 1, 100)) == t(bits, 0, 2))
    assert tuple(dr.tzcnt(t(i))[0] for i in (0, 1, 100)) == (bits, 0, 2)
    if bits == 32:
        assert tuple(dr.tzcnt(i) for i in (0, 1, 100)) == (bits, 0, 2)

@pytest.test_arrays('uint64, jit, shape=(*)', 'uint32, jit, shape=(*)')
def test16_popcnt(t):
    assert dr.all(dr.popcnt(t(0, 1, 100)) == t(0, 1, 3))
    assert tuple(dr.popcnt(i) for i in (0, 1, 100)) == (0, 1, 3)
    assert tuple(dr.popcnt(t(i))[0] for i in (0, 1, 100)) == (0, 1, 3)


@pytest.test_arrays('uint32, jit, shape=(*)')
def test17_brev(t):
    inp = (0xcafe, 0x1, 0x12345678)
    out = (0x7f530000, 0x80000000, 0x1e6a2c48)
    assert dr.all(dr.brev(t(inp)) == t(out))
    assert tuple(dr.brev(i) for i in inp) == out
    assert tuple(dr.brev(t(i))[0] for i in inp) == out

@pytest.test_arrays('uint32, jit, shape=(*)')
def test17_log2i(t):
    assert dr.all(dr.log2i(t(1, 2, 100)) == t(0, 1, 6))
    assert tuple(dr.log2i(i) for i in (1, 2, 100)) == (0, 1, 6)

@pytest.test_arrays('float, jit, shape=(*)')
def test18_lgamma_erfinv(t):
    # Spot-check the bindings of erfinv and tgamma
    assert dr.allclose(dr.lgamma(1.2), -0.0853741)
    assert dr.allclose(dr.lgamma(-1.2), 1.57918)
    assert dr.allclose(dr.erfinv(.3), 0.272463)
    assert dr.allclose(dr.erfinv(.8), 0.906194)

@pytest.test_arrays('float, shape=(3, *)')
def test19_incompatible(t):
    v = dr.value_t(t)
    with pytest.raises(RuntimeError) as e:
        v(1, 2, 3) + v()
    msg = 'operands have incompatible sizes'
    assert msg in str(e.value)
    with pytest.raises(RuntimeError) as e:
        t(1, 2, 3) + t()
    assert msg in str(e.value.__cause__)

@pytest.test_arrays('float, shape=(*)')
def test20_rcp_inf(t):
    y = dr.rcp(t(0.0, -0.0))
    assert dr.all(~dr.isnan(y))
    assert y[0] > 0
    assert y[1] < 0

@pytest.test_arrays('bool, shape=(2, *)', 'bool, shape=(2)')
def test21_int_promote(t):
    from drjit.scalar import Array2b
    Array2i = dr.int32_array_t(Array2b)
    assert type(dr.select(Array2b(True, False), 1, 2)) is Array2i
    assert type(dr.select(Array2b(True, False), -1, 1)) is Array2i

@pytest.test_arrays('-bool, shape=(*)')
def test22_and_mask(t):
    arr = t(1, 2, 3, 4)
    m = arr < 3

    out = arr & m
    assert dr.all(out == t(1, 2, 0, 0))

    out = arr & [True, True, False, False]
    assert dr.all(out == t(1, 2, 0, 0))

    arr &= m
    assert dr.all(arr == t(1, 2, 0, 0))

@pytest.test_arrays('float32, shape=(*), is_diff')
def test23_rsqrt(t):
    x = t(0.001, 0.1, 1, 2, 3, 4, 100)
    y = 1 / dr.sqrt(x)
    z = dr.rsqrt(x)
    assert dr.all((y-z) / y < 1e-7)

    x = t(0, float('inf'))
    y = 1 / dr.sqrt(x)
    z = dr.rsqrt(x)

    assert dr.isinf(y[0]) == dr.isinf(z[0])
    assert y[1] == z[1]


@pytest.mark.parametrize("opaque", [True, False])
@pytest.test_arrays('int32, shape=(*)','int32, shape=(2, *)')
def test24_mul_hi_wide(t, opaque):
    np = pytest.importorskip("numpy")
    if dr.is_unsigned_v(t):
        a_s = 0xcafe9876
        b_s = 0xfafecafe
        with pytest.warns(RuntimeWarning):
            c_s = np.uint32(a_s)*np.uint32(b_s)
    else:
        # Identical bit representation
        a_s = -889284490
        b_s = -83965186
        with pytest.warns(RuntimeWarning):
            c_s = np.int32(a_s)*np.int32(b_s)

    a, b = t(a_s), t(b_s)

    if opaque:
        dr.make_opaque(a)
        dr.make_opaque(b)

    c = a*b
    d   = dr.mul_hi(a, b)
    d_s = dr.mul_hi(a_s, b_s)

    assert dr.all(c == c_s)
    assert dr.all(d == d_s)

    e   = dr.mul_wide(a, b)
    e_s = dr.mul_wide(a_s, b_s)

    assert dr.all(e == e_s)
