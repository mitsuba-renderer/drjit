import drjit as dr
import pytest

@pytest.test_arrays('float32,shape=(3)', 'float32,shape=(*)',
                    'float32,shape=(3, *)', 'float32,shape=(*, *)')
def test1_binop_promote_broadcast(t):
    v = dr.value_t(t)

    x = t(10, 100, 1000) + 1
    assert type(x) is t and dr.all(x == t(11, 101, 1001))
    x = t(10, 100, 1000) + v(1)
    assert type(x) is t and dr.all(x == t(11, 101, 1001))

    x = 1 + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 101, 1001))
    x = v(1) + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 101, 1001))

    x = t(10, 100, 1000) + (1, 2, 3)
    assert type(x) is t and dr.all(x == t(11, 102, 1003))

    x = (1, 2, 3) + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 102, 1003))

    x = [1, 2, 3] + t(10, 100, 1000)
    assert type(x) is t and dr.all(x == t(11, 102, 1003))

    if dr.size_v(t) != dr.Dynamic:
        x = t(10, 100, 1000) + (v(1), v(2), v(3))
        assert type(x) is t and dr.all(x == t(11, 102, 1003))

        x = t(10, 100, 1000) + [v(1), v(2), v(3)]
        assert type(x) is t and dr.all(x == t(11, 102, 1003))

        x = (v(1), v(2), v(3)) + t(10, 100, 1000)
        assert type(x) is t and dr.all(x == t(11, 102, 1003))

        x = [v(1), v(2), v(3)] + t(10, 100, 1000)
        assert type(x) is t and dr.all(x == t(11, 102, 1003))

def test2_binop_promote_pairs():
    a = [
        dr.scalar.ArrayXi,
        dr.scalar.ArrayXu,
        dr.scalar.ArrayXi64,
        dr.scalar.ArrayXu64,
        dr.scalar.ArrayXf,
        dr.scalar.ArrayXf64
    ]

    for i0, t0 in enumerate(a):
        for i1, t1 in enumerate(a):
            t2 = type(dr.zeros(t0) + dr.zeros(t1))
            t3 = t0 if i0 > i1 else t1
            assert t2 is t3

def test3_binop_promote_misc():
    try:
        dr.llvm.Float(0)
    except:
        pytest.skip()

    x = dr.zeros(dr.llvm.Array3i) + dr.zeros(dr.llvm.Float)
    assert type(x) is dr.llvm.Array3f
    x = dr.zeros(dr.llvm.ArrayXi) + dr.zeros(dr.llvm.Float)
    assert type(x) is dr.llvm.ArrayXf
    x = dr.zeros(dr.scalar.Array3i) + dr.zeros(dr.llvm.Float)
    assert type(x) is dr.llvm.Array3f
    x = dr.zeros(dr.scalar.Complex2f) + dr.zeros(dr.llvm.Float)
    assert type(x) is dr.llvm.Complex2f
    x = dr.zeros(dr.llvm.Float) + dr.zeros(dr.scalar.Complex2f)
    assert type(x) is dr.llvm.Complex2f
    x = dr.zeros(dr.scalar.Complex2f64) + dr.zeros(dr.llvm.Complex2f)
    assert type(x) is dr.llvm.Complex2f64
    x = dr.zeros(dr.scalar.Array3i) + 1.0
    assert type(x) is dr.scalar.Array3f
    x = dr.zeros(dr.scalar.Array3i) + (1, 2, 3)
    assert type(x) is dr.scalar.Array3i
    x = dr.zeros(dr.scalar.Array3i) + (1, 2.0, 3)
    assert type(x) is dr.scalar.Array3f
    x = dr.zeros(dr.scalar.ArrayXf) + dr.zeros(dr.llvm.Array3f)
    assert type(x) is dr.llvm.ArrayXf
    x = dr.zeros(dr.scalar.ArrayXf) + dr.zeros(dr.llvm.ArrayXf)
    assert type(x) is dr.llvm.ArrayXf

    with pytest.raises(RuntimeError, match="Incompatible arguments."):
        x = dr.zeros(dr.scalar.Complex2f64) + dr.zeros(dr.llvm.Array3f)
    with pytest.raises(RuntimeError, match="Incompatible arguments."):
        x = dr.zeros(dr.scalar.Complex2f64) + dr.zeros(dr.llvm.Array2f)

    a = dr.scalar.Array3i(1)
    with pytest.raises(RuntimeError) as ei:
        a + "asdf"
    assert "drjit.scalar.Array3i.__add__(): Encountered an unsupported argument of type 'str' (must be a Dr.Jit array or a type that can be converted into one)" in str(ei.value)

    a + 2**10
    with pytest.raises(RuntimeError) as ei:
        a + 2**100
    assert "drjit.scalar.Array3i.__add__(): Could not promote type 'int' to 'drjit.scalar.Array3u64'." in str(ei.value)

def test4_binop_promote_ad():
    try:
        dr.llvm.Float(0)
    except:
        pytest.skip()

    x = dr.zeros(dr.llvm.Float) + dr.zeros(dr.llvm.ad.Float)
    assert type(x) is dr.llvm.ad.Float
    x = dr.zeros(dr.llvm.ad.Float) + dr.zeros(dr.llvm.Float)
    assert type(x) is dr.llvm.ad.Float
    x = dr.zeros(dr.llvm.Float) * dr.zeros(dr.llvm.ad.Array3f)
    assert type(x) is dr.llvm.ad.Array3f
    x = dr.zeros(dr.llvm.ad.Float) * dr.zeros(dr.llvm.Array3f)
    assert type(x) is dr.llvm.ad.Array3f
    x = dr.zeros(dr.llvm.Float) + dr.zeros(dr.llvm.ad.Complex2f)
    assert type(x) is dr.llvm.ad.Complex2f
    x = dr.zeros(dr.llvm.ad.Float) + dr.zeros(dr.llvm.Complex2f)
    assert type(x) is dr.llvm.ad.Complex2f