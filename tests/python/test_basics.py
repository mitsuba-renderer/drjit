import enoki as ek
import pytest
import gc


def get_class(name):
    """Resolve a package+class name into the corresponding type"""
    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    return value


@pytest.fixture(autouse=True)
def run_gc():
    """
    Run the garbage collector after each test to avoid LLVM <-> CUDA
    allocate/free sequences, which cause test failures.
    """
    yield
    gc.collect()
    gc.collect()


def test01_construct():
    a = ek.scalar.Array3f(1, 2, 3)
    assert(repr(a) == "[1.0, 2.0, 3.0]")
    a = ek.scalar.Array3f([1, 2, 3])
    assert(repr(a) == "[1.0, 2.0, 3.0]")
    a = ek.scalar.Array3f((1, 2, 3))
    assert(repr(a) == "[1.0, 2.0, 3.0]")
    a = ek.scalar.Array3f(4)
    assert(repr(a) == "[4.0, 4.0, 4.0]")

    with pytest.raises(TypeError) as ei:
        a = ek.scalar.Array3f("asdf")

    assert "Array3f constructor expects: 1 or 3 values of type 'float', a " \
        "matching list/tuple, or a NumPy/PyTorch array." in str(ei.value)


@pytest.mark.parametrize("cname", ["enoki.scalar.Array1f",
                                   "enoki.scalar.Array1i",
                                   "enoki.packet.Float",
                                   "enoki.packet.Int",
                                   "enoki.cuda.Float",
                                   "enoki.cuda.Int",
                                   "enoki.cuda.ad.Float",
                                   "enoki.cuda.ad.Int",
                                   "enoki.llvm.Float",
                                   "enoki.llvm.Int",
                                   "enoki.llvm.ad.Float",
                                   "enoki.llvm.ad.Int"
                                   ])
def test02_basic_ops(cname):
    t = get_class(cname)
    if 'cuda' in cname:
        ek.set_device(0)
    elif 'llvm' in cname:
        ek.set_device(-1)

    # Test basic arithmetic operations
    import enoki.router as r
    nc = r._entry_evals
    v1 = t(4)
    v2 = t(2)
    assert v1.x == 4
    assert v2.x == 2
    v3 = v1 + v2
    assert v3.x == 6
    v3 = v1 - v2
    assert v3.x == 2
    v3 = v1 * v2
    assert v3.x == 8

    if ek.scalar_t(t) is int:
        v3 = v1 // v2
        assert v3.x == 2
        with pytest.raises(ek.Exception):
            v3 = v1 / v2
    else:
        v3 = v1 / v2
        assert v3.x == 2.0
        with pytest.raises(ek.Exception):
            v3 = v1 // v2

    # Make sure that in-place ops are truly in-place
    v = t(1)
    v_id = id(v)

    if ek.scalar_t(t) is int:
        v //= v
        v >>= 0
        v <<= 0
        v %= v
    else:
        v /= v

    v += v
    v -= v
    v *= v

    assert v_id == id(v)

    # We now exactly how many times Array.entry()/set_entry()
    # should have been called during all of the above
    entry_evals = 6
    if 'scalar' in cname:
        entry_evals = 39 if ek.scalar_t(t) is int else 30

    assert r._entry_evals == nc + entry_evals


def test03_type_promotion():
    import enoki.packet as s

    assert type(s.Array1b() | s.Array1b()) is s.Array1b
    assert type(s.Array1b() ^ s.Array1b()) is s.Array1b
    assert type(s.Array1b() & s.Array1b()) is s.Array1b

    with pytest.raises(ek.Exception) as ei:
        _ = s.Array1b() + s.Array1b()
    assert "add(): requires arithmetic operands!" in str(ei)

    assert type(s.Array1i() + s.Array1i()) is s.Array1i
    assert type(s.Array1i() + s.Array1u()) is s.Array1u
    assert type(s.Array1i() + s.Array1i64()) is s.Array1i64
    assert type(s.Array1i() + s.Array1u64()) is s.Array1u64
    assert type(s.Array1i() + s.Array1f()) is s.Array1f
    assert type(s.Array1i() + s.Array1f64()) is s.Array1f64

    with pytest.raises(ek.Exception) as ei:
        _ = ek.sqrt(s.Array1i())
    assert "sqrt(): requires floating point operands!" in str(ei)

    assert type(s.Array1f() + s.Array1f()) is s.Array1f
    assert type(s.Array1f() + s.Array1u()) is s.Array1f
    assert type(s.Array1f() + s.Array1i64()) is s.Array1f
    assert type(s.Array1f() + s.Array1u64()) is s.Array1f
    assert type(s.Array1f() + s.Array1f()) is s.Array1f
    assert type(s.Array1f() + s.Array1f64()) is s.Array1f64

    assert type(s.Array1f() | s.Array1b()) is s.Array1f
    assert type(s.Array1f() ^ s.Array1b()) is s.Array1f
    assert type(s.Array1f() & s.Array1b()) is s.Array1f


# Run various standard operations on a 3D array
def test04_operators():
    I3 = ek.scalar.Array3i
    F3 = ek.scalar.Array3f

    assert(I3(1, 2, 3) + I3(0, 1, 0) == I3(1, 3, 3))
    assert(I3(1, 2, 3) - I3(0, 1, 0) == I3(1, 1, 3))
    assert(I3(1, 2, 3) * I3(0, 1, 0) == I3(0, 2, 0))
    assert(I3(1, 2, 3) // I3(2, 2, 2) == I3(0, 1, 1))
    assert(I3(1, 2, 3) % I3(2, 2, 2) == I3(1, 0, 1))
    assert(I3(1, 2, 3) << 1 == I3(2, 4, 6))
    assert(I3(1, 2, 3) >> 1 == I3(0, 1, 1))
    assert(I3(1, 2, 3) & 1 == I3(1, 0, 1))
    assert(I3(1, 2, 3) | 1 == I3(1, 3, 3))
    assert(I3(1, 2, 3) ^ 1 == I3(0, 3, 2))
    assert(-I3(1, 2, 3) == I3(-1, -2, -3))
    assert(~I3(1, 2, 3) == I3(-2, -3, -4))
    assert(ek.abs(I3(1, -2, 3)) == I3(1, 2, 3))
    assert(abs(I3(1, -2, 3)) == I3(1, 2, 3))
    assert(ek.abs(I3(1, -2, 3)) == I3(1, 2, 3))
    assert(ek.fmadd(F3(1, 2, 3), F3(2, 3, 1), F3(1, 1, 1)) == F3(3, 7, 4))
    assert(ek.fmsub(F3(1, 2, 3), F3(2, 3, 1), F3(1, 1, 1)) == F3(1, 5, 2))
    assert(ek.fnmadd(F3(1, 2, 3), F3(2, 3, 1), F3(1, 1, 1)) == F3(-1, -5, -2))
    assert(ek.fnmsub(F3(1, 2, 3), F3(2, 3, 1), F3(1, 1, 1)) == F3(-3, -7, -4))


def all_arrays(cond=lambda x: True):
    a = list(ek.scalar.__dict__.items())
    a += ek.packet.__dict__.items()
    a += ek.cuda.__dict__.items()
    a += ek.llvm.__dict__.items()
    a += ek.llvm.__dict__.items()
    return [v for k, v in a if isinstance(v, type) and cond(v)
            and not ek.is_special_v(v)
            and not ek.array_depth_v(v) >= 3
            and not (ek.array_depth_v(v) >= 2 and 'scalar' in v.__module__)]


# Run various standard operations on *every available* type
@pytest.mark.parametrize("t", all_arrays())
def test05_scalar(t):
    if not ek.is_array_v(t) or ek.array_size_v(t) == 0:
        return
    if t.IsCUDA:
        ek.set_device(0)
    elif t.IsLLVM:
        ek.set_device(-1)

    if ek.is_mask_v(t):
        assert ek.all_nested(t(True))
        assert ek.any_nested(t(True))
        assert ek.none_nested(t(False))
        assert ek.all_nested(t(False) ^ t(True))
        assert ek.all_nested(ek.eq(t(False), t(False)))
        assert ek.none_nested(ek.eq(t(True), t(False)))

    if ek.is_arithmetic_v(t):
        assert t(1) + t(1) == t(2)
        assert t(3) - t(1) == t(2)
        assert t(2) * t(2) == t(4)
        assert ek.min(t(2), t(3)) == t(2)
        assert ek.max(t(2), t(3)) == t(3)

        if ek.is_signed_v(t):
            assert t(2) * t(-2) == t(-4)
            assert ek.abs(t(-2)) == t(2)

        if ek.is_integral_v(t):
            assert t(6) // t(2) == t(3)
            assert t(7) % t(2) == t(1)
            assert t(7) >> 1 == t(3)
            assert t(7) << 1 == t(14)
            assert t(1) | t(2) == t(3)
            assert t(1) ^ t(3) == t(2)
            assert t(1) & t(3) == t(1)
        else:
            assert t(6) / t(2) == t(3)
            assert ek.sqrt(t(4)) == t(2)
            assert ek.fmadd(t(1), t(2), t(3)) == t(5)
            assert ek.fmsub(t(1), t(2), t(3)) == t(-1)
            assert ek.fnmadd(t(1), t(2), t(3)) == t(1)
            assert ek.fnmsub(t(1), t(2), t(3)) == t(-5)
            assert (t(1) & True) == t(1)
            assert (t(1) & False) == t(0)
            assert (t(1) | False) == t(1)

        assert ek.all_nested(t(3) > t(2))
        assert ek.all_nested(ek.eq(t(2), t(2)))
        assert ek.all_nested(ek.neq(t(3), t(2)))
        assert ek.all_nested(t(1) >= t(1))
        assert ek.all_nested(t(2) < t(3))
        assert ek.all_nested(t(1) <= t(1))
        assert ek.select(ek.eq(t(2), t(2)), t(4), t(5)) == t(4)
        assert ek.select(ek.eq(t(3), t(2)), t(4), t(5)) == t(5)
        t2 = t(2)
        assert ek.hsum(t2) == t.Value(2 * len(t2))
        assert ek.dot(t2, t2) == t.Value(4 * len(t2))
        assert ek.dot_async(t2, t2) == t(4 * len(t2))

        value = t(1)
        value[ek.eq(value, t(1))] = t(2)
        value[ek.eq(value, t(3))] = t(5)
        assert value == t(2)


def test06_reinterpret_cast():
    I3 = ek.scalar.Array3i
    F3 = ek.scalar.Array3f
    B3 = ek.scalar.Array3b
    LI3 = ek.scalar.Array3i64
    LF3 = ek.scalar.Array3f64

    assert ek.mask_t(I3) is B3
    assert ek.scalar_t(F3) is float
    assert ek.scalar_t(I3) is int
    assert ek.float_array_t(I3) is F3
    assert ek.int_array_t(F3) is I3
    assert ek.float_array_t(LI3) is LF3
    assert ek.int_array_t(LF3) is LI3

    assert ek.reinterpret_array(I3, F3(1)).x == 0x3f800000
    assert ek.reinterpret_array(F3, I3(0x3f800000)).x == 1.0


@pytest.mark.parametrize("pkg", [ek.cuda, ek.llvm])
def test07_gather_ravel_unravel(pkg):
    ek.set_device(0 if 'cuda' in pkg.__name__ else -1)
    str_1 = '[[0.0, 1.0, 2.0],\n [3.0, 4.0, 5.0],\n [6.0, 7.0, 8.0],\n' \
        ' [9.0, 10.0, 11.0]]'
    str_2 = '[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]'
    a = ek.arange(pkg.Float, 100)
    b = ek.gather(pkg.Array3f, a, pkg.UInt(0, 1, 2, 3))
    assert repr(b) == str_1
    c = ek.ravel(b)
    assert repr(c) == str_2
    d = ek.unravel(pkg.Array3f, c)
    assert repr(d) == str_1


@pytest.mark.parametrize("t", all_arrays(lambda a:
                                         getattr(a, 'IsFloat', False)))
def test07_sincos(t):
    def poly2(x, c0, c1, c2):
        x2 = ek.sqr(x)
        return ek.fmadd(x2, c2, ek.fmadd(x, c1, c0))

    def sincos(x):
        Float = type(x)
        Int = ek.int_array_t(Float)

        xa = ek.abs(x)

        j = Int(xa * 1.2732395447351626862)

        j = (j + Int(1)) & ~Int(1)

        y = Float(j)

        Shift = Float.Type.Size * 8 - 3

        sign_sin = ek.reinterpret_array(Float, j << Shift) ^ x
        sign_cos = ek.reinterpret_array(Float, (~(j - Int(2)) << Shift))

        y = xa - y * 0.78515625 \
               - y * 2.4187564849853515625e-4 \
               - y * 3.77489497744594108e-8

        z = y * y
        z |= ek.eq(xa, ek.Infinity)

        s = poly2(z, -1.6666654611e-1,
                  8.3321608736e-3,
                  -1.9515295891e-4) * z

        c = poly2(z, 4.166664568298827e-2,
                  -1.388731625493765e-3,
                  2.443315711809948e-5) * z

        s = ek.fmadd(s, y, y)
        c = ek.fmadd(c, z, ek.fmadd(z, -0.5, 1))

        polymask = ek.eq(j & Int(2), ek.zero(Int))

        return (
            ek.mulsign(ek.select(polymask, s, c), sign_sin),
            ek.mulsign(ek.select(polymask, c, s), sign_cos)
        )

    if t.IsCUDA:
        ek.set_device(0)
    elif t.IsLLVM:
        ek.set_device(-1)
    s, c = sincos(t(1))
    if t.Size != 0:
        assert ek.allclose(s**2 + c**2, 1)


@pytest.mark.parametrize("cname", ["enoki.packet.Int",
                                   "enoki.packet.Int64",
                                   "enoki.packet.UInt",
                                   "enoki.packet.UInt64",
                                   "enoki.cuda.Int",
                                   "enoki.cuda.Int64",
                                   "enoki.cuda.UInt",
                                   "enoki.cuda.UInt64",
                                   "enoki.llvm.Int",
                                   "enoki.llvm.Int64",
                                   "enoki.llvm.UInt",
                                   "enoki.llvm.UInt64"])
def test08_divmod(cname):
    t = get_class(cname)
    if 'cuda' in cname:
        ek.set_device(0)
    elif 'llvm' in cname:
        ek.set_device(-1)

    index = ek.arange(t, 10000000)
    index[index < len(index) // 2] = -index
    index *= 256203161

    for i in range(1, 100):
        assert index // i == index // ek.full(t, i, 1, eval=True)
        assert index % i == index % ek.full(t, i, 1, eval=True)

    if t.IsSigned:
        for i in range(1, 100):
            assert index // -i == index // ek.full(t, -i, 1, eval=True)
            assert index % -i == index % ek.full(t, -i, 1, eval=True)


@pytest.mark.parametrize("cname", ["enoki.cuda.Float", "enoki.llvm.Float"])
def test09_repeat_tile(cname):
    t = get_class(cname)
    a3 = get_class(cname.replace('Float', 'Array3f'))
    if 'cuda' in cname:
        ek.set_device(0)
    elif 'llvm' in cname:
        ek.set_device(-1)
    vec = t([1, 2, 3])
    tiled = t([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    reptd = t([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3])

    assert ek.tile(vec, 5) == tiled
    assert ek.tile(a3(vec, vec + 1, vec + 2), 5) == \
        a3(tiled, tiled + 1, tiled + 2)

    assert ek.repeat(vec, 5) == reptd
    assert ek.repeat(a3(vec, vec + 1, vec + 2), 5) == \
        a3(reptd, reptd + 1, reptd + 2)


@pytest.mark.parametrize("cname", ["enoki.cuda.Float", "enoki.llvm.Float"])
def test10_meshgrid(cname):
    t = get_class(cname)
    if 'cuda' in cname:
        ek.set_device(0)
    elif 'llvm' in cname:
        ek.set_device(-1)
    import numpy as np
    a = ek.linspace(t, 0, 1, 3)
    b = ek.linspace(t, 0, 1, 4)
    c, d = ek.meshgrid(a, b)
    ek.schedule(c, d)
    cn, dn = np.meshgrid(a.numpy(), b.numpy())
    assert ek.allclose(c.numpy(), cn.ravel())
    assert ek.allclose(d.numpy(), dn.ravel())


@pytest.mark.parametrize("cname", ["enoki.cuda.Float", "enoki.llvm.Float"])
def test11_binary_search(cname):
    t = get_class(cname)
    if 'cuda' in cname:
        ek.set_device(0)
    elif 'llvm' in cname:
        ek.set_device(-1)
    import numpy as np

    data_np = np.float32(np.sort(np.random.normal(size=10000)))
    search_np = np.float32(np.random.normal(size=10000))
    data = t(data_np)
    search = t(search_np)

    index = ek.binary_search(
        0, len(data) - 1,
        lambda index: ek.gather(t, data, index) < search
    )

    value = ek.gather(t, data, index)
    cond = ek.eq(index, len(data)-1) | (value >= search)
    assert ek.all(cond)


def test12_slice_setitem():
    a = ek.zero(ek.scalar.ArrayXf, 5)
    a[2] = 1.0
    assert ek.allclose(a, [0, 0, 1, 0, 0])
    a[2:] = [2.0, 1.0, 1.0]
    assert ek.allclose(a, [0, 0, 2, 1, 1])
    a[:] = 0.0
    assert ek.allclose(a, [0, 0, 0, 0, 0])

    v = ek.scalar.Array3f(0)
    v[2] = 1.0
    assert ek.allclose(v, [0, 0, 1])
    v[1:] = 2.0
    assert ek.allclose(v, [0, 2, 2])

    m = ek.scalar.Matrix3f(0)
    m[1:, 1] = 1.0
    assert ek.allclose(m, [[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    m[0, 1:] = 2.0
    assert ek.allclose(m, [[0, 2, 2], [0, 1, 0], [0, 1, 0]])
    m[1:, 1:] = 3.0
    assert ek.allclose(m, [[0, 2, 2], [0, 3, 3], [0, 3, 3]])
