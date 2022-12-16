import drjit as dr
import pytest
import importlib


def test01_init_zero():
    from drjit.scalar import Array0f, Array3f, ArrayXf
    a = Array0f()
    assert len(a) == 0

    a = Array3f()
    assert len(a) == 3
    for i in range(3):
        if dr.DEBUG:
            assert a[i] != a[i]
        else:
            assert a[i] == 0

    a = ArrayXf()
    assert len(a) == 0
    with pytest.raises(IndexError) as ei:
        a[0]
    assert "entry 0 is out of bounds (the array is of size 0)." in str(ei.value)


def test02_init_sequence_static():
    from drjit.scalar import Array0f, Array1f, Array3f

    a = Array3f((0, 1, 2))
    for i in range(3):
        assert a[i] == i

    a = Array3f([0, 1, 2])
    for i in range(3):
        assert a[i] == i

    class my_list(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    a = Array3f(my_list([0, 1, 2]))
    for i in range(3):
        assert a[i] == i

    a = Array3f(0, 1, 2)
    for i in range(3):
        assert a[i] == i

    assert Array1f(1)[100] == 1

    pytest.skip("nanobind layer")

    with pytest.raises(TypeError) as ei:
        Array3f("test")
    assert "input sequence has wrong size (expected 3, got 4)" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array0f(1)
    assert "too many arguments provided" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        a = Array3f((0, 1))
    assert "input sequence has wrong size" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        a = Array3f((0, 1, 2, 3))
    assert "input sequence has wrong size" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f("tst")
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f((0, "foo", 2))
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        Array3f(0, "foo", 2)
    assert "could not initialize element with a value of type 'str'" in str(ei.value)


def test03_init_sequence_dynamic():
    from drjit.scalar import ArrayXf

    a = ArrayXf((0, 1, 2, 3, 4))
    assert len(a) == 5
    for i in range(5):
        assert a[i] == i

    a = ArrayXf([0, 1, 2])
    assert len(a) == 3
    for i in range(3):
        assert a[i] == i

    class my_list(list):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    a = ArrayXf(my_list([0, 1, 2]))
    for i in range(3):
        assert a[i] == i

    a = ArrayXf(0, 1, 2)
    for i in range(3):
        assert a[i] == i

    for i in range(3):
        a[i] += 1

    for i in range(3):
        assert a[i] == i + 1

    assert ArrayXf(1)[100] == 1

    pytest.skip("nanobind layer")

    with pytest.raises(TypeError) as ei:
        ArrayXf("test")
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        ArrayXf((0, "foo", 2))
    assert "could not initialize element with a value of type 'str'" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        ArrayXf(0, "foo", 2)
    assert "could not initialize element with a value of type 'str'" in str(ei.value)



def test04_indexing():
    from drjit.scalar import ArrayXf, Array3f
    a = ArrayXf([0, 1, 2])
    b = Array3f([0, 1, 2])
    assert a[-1] == 2 and b[-1] == 2

    with pytest.raises(IndexError) as ei:
        a[3]

    with pytest.raises(IndexError) as ei:
        a[-4]

    with pytest.raises(IndexError) as ei:
        b[3]

    with pytest.raises(IndexError) as ei:
        b[-4]


def test05_indexed_assignment():
    from drjit.scalar import ArrayXf, Array3f

    a = Array3f([0]*3)
    b = ArrayXf([0]*5)
    for i in range(3):
        a[i] = i
    for i in range(len(a)):
        a[i] += 1
    for i in range(len(b)):
        b[i] = i
    for i in range(len(b)):
        b[i] += 1
    for i in range(len(a)):
        assert a[i] == i + 1
    for i in range(len(b)):
        assert b[i] == i + 1


def test06_constructor_copy():
    from drjit.scalar import Array3f, ArrayXf
    a = Array3f(1, 2, 3)
    b = Array3f(a)
    c = ArrayXf(1, 2, 3, 4)
    d = ArrayXf(c)
    assert len(a) == len(b)
    assert len(c) == len(d)
    for i in range(len(a)):
        assert a[i] == b[i]
    for i in range(len(c)):
        assert c[i] == d[i]


def test07_constructor_broadcast():
    from drjit.scalar import Array3f, ArrayXf, ArrayXb
    a = Array3f(3)
    assert len(a) == 3 and a[0] == 3 and a[1] == 3 and a[2] == 3
    a = ArrayXf(3)
    assert len(a) == 1 and a[0] == 3
    a = ArrayXb(True)
    assert len(a) == 1 and a[0] == True


def test08_all_any():
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

    pytest.skip("nanobind layer")

    assert type(dr.all(ArrayXb(True, True))) is ArrayXb
    assert len(dr.all(ArrayXb(True, True))) == 1
    assert dr.all(ArrayXb(True, True))[0] == True
    assert dr.all(ArrayXb(True, False))[0] == False
    assert dr.all(ArrayXb(False, False))[0] == False
    assert dr.any(ArrayXb(True, True))[0] == True
    assert dr.any(ArrayXb(True, False))[0] == True
    assert dr.any(ArrayXb(False, False))[0] == False

    assert type(dr.all(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) is dr.llvm.Bool
    assert len(dr.all(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) == 3
    assert type(dr.all_nested(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) is dr.llvm.Bool
    assert len(dr.all_nested(dr.llvm.Array1b(dr.llvm.Bool([True, False, False])))) == 1

    with pytest.raises(TypeError) as ei:
        dr.all((True, "hello"))
    assert "unsupported operand type(s)" in str(ei.value)

@pytest.mark.skip("nanobind layer")
def test09_implicit_to_bool():
    from drjit.scalar import Array3f, ArrayXf, Array3b, ArrayXb
    with pytest.raises(TypeError) as ei:
        bool(ArrayXf(1))
    assert "ArrayXf.__bool__(): implicit conversion to 'bool' is only supported for scalar mask arrays!" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        bool(Array3f(1))
    assert "Array3f.__bool__(): implicit conversion to 'bool' is only supported for scalar mask arrays!" in str(ei.value)

    with pytest.raises(RuntimeError) as ei:
        bool(ArrayXb(True, False))
    assert "ArrayXb.__bool__(): implicit conversion to 'bool' requires a scalar mask array (array size was 2)." in str(ei.value)

    assert bool(ArrayXb(True))
    assert not bool(ArrayXb(False))


@pytest.mark.skip("nanobind layer")
@pytest.mark.parametrize('value', [(dr.scalar.Array3f, dr.scalar.Array3b), (dr.scalar.ArrayXf, dr.scalar.ArrayXb)])
def test10_comparison(value):
    c, m = value
    assert dr.all(c(1, 2, 3) == c(1, 2, 3))
    assert not dr.all(c(1, 2, 3) == c(1, 3, 3))
    assert dr.all(c(1, 2, 3) != c(4, 5, 6))
    assert not dr.all(c(1, 2, 3) != c(4, 2, 6))
    assert dr.any(c(1, 2, 3) == c(1, 2, 3))
    assert not dr.any(c(1, 2, 3) == c(4, 5, 6))
    assert dr.any(c(1, 2, 3) != c(1, 3, 3))
    assert not dr.any(c(1, 2, 3) != c(1, 2, 3))

    assert dr.all((c(1, 2, 3) < c(0, 2, 4)) == m(False, False, True))
    assert dr.all((c(1, 2, 3) <= c(0, 2, 4)) == m(False, True, True))
    assert dr.all((c(1, 2, 3) > c(0, 2, 4)) == m(True, False, False))
    assert dr.all((c(1, 2, 3) >= c(0, 2, 4)) == m(True, True, False))
    assert dr.all((c(1, 2, 3) == c(0, 2, 4)) == m(False, True, False))
    assert dr.all((c(1, 2, 3) != c(0, 2, 4)) == m(True, False, True))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test11_shape():
    import drjit.scalar as s
    import drjit.llvm as l

    assert dr.shape(s.Array0f()) == (0,) and s.Array0f().shape == (0,)
    assert dr.shape(s.Array2f()) == (2,) and s.Array2f().shape == (2,)
    assert dr.shape(l.Float()) == (0,) and l.Float().shape == (0,)
    assert dr.shape(l.Float(1, 2, 3)) == (3,) and l.Float(1, 2, 3).shape == (3,)
    assert dr.shape(l.Array2f()) == (2, 0) and l.Array2f().shape == (2, 0)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3))) == (2,3) and \
           l.Array2f(l.Float(1, 2, 3)).shape == (2,3)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3),
                              l.Float(2, 3, 4))) == (2,3) and \
            l.Array2f(l.Float(1, 2, 3), l.Float(2, 3, 4)).shape == (2,3)
    assert dr.shape(l.Array2f(l.Float(1, 2, 3),
                              l.Float(2, 3))) is None and \
           l.Array2f(l.Float(1, 2, 3), l.Float(2, 3)).shape is None

@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test11_repr():
  import drjit.scalar as s
  import drjit.llvm as l

  assert repr(s.Array0f()) == '[]'
  assert repr(s.ArrayXf()) == '[]'
  assert repr(s.Array1f(1)) == '[1]'
  assert repr(s.Array1f(1.5)) == '[1.5]'
  assert repr(s.Array2f(1, 2)) == '[1, 2]'
  assert repr(s.ArrayXf(1, 2)) == '[1, 2]'

  assert repr(l.Array0f()) == '[]'
  assert repr(l.ArrayXf()) == '[]'
  assert repr(l.Array1f(1)) == '[[1]]'
  assert repr(l.Array1f(1.5)) == '[[1.5]]'
  assert repr(l.Array2f(1, 2)) == '[[1, 2]]'
  assert repr(l.Array2f(1, [2, 3])) == '[[1, 2],\n' \
                                       ' [1, 3]]'
  assert repr(l.ArrayXf(1, 2)) == '[[1, 2]]'
  assert repr(l.Float(range(1000))) == '[0, 1, 2, 3, 4, .. 990 skipped .., 995, 996, 997, 998, 999]'


def test12_binop_simple():
    from drjit.scalar import Array3f, ArrayXf, Array3u, ArrayXb
    a = Array3f(1, 2, 3)
    assert dr.all(a + a == Array3f(2, 4, 6))
    assert dr.all(a + (1, 2, 3) == Array3f(2, 4, 6))
    assert dr.all(a + [1, 2, 3] == Array3f(2, 4, 6))
    assert dr.all((1, 2, 3) + a == Array3f(2, 4, 6))
    assert dr.all([1, 2, 3] + a == Array3f(2, 4, 6))
    assert dr.all(a - a == Array3f(0, 0, 0))
    assert dr.all(a * a == Array3f(1, 4, 9))
    assert dr.all(a / a == Array3f(1, 1, 1))

    with pytest.raises(TypeError) as ei:
        a // a
    # assert "unsupported operand type(s)" in str(ei.value) # TODO

    a = ArrayXf(1, 2, 3)
    assert dr.all(a + a == ArrayXf(2, 4, 6))
    assert dr.all(a - a == ArrayXf(0, 0, 0))
    assert dr.all(a * a == ArrayXf(1, 4, 9))
    assert dr.all(a / a == ArrayXf(1, 1, 1))

    a = Array3u(1, 2, 3)
    assert dr.all(a + a == Array3u(2, 4, 6))
    assert dr.all(a - a == Array3u(0, 0, 0))
    assert dr.all(a * a == Array3u(1, 4, 9))
    assert dr.all(a // a == Array3u(1, 1, 1))
    assert dr.all(a << 1 == Array3u(2, 4, 6))
    assert dr.all(a >> 1 == Array3u(0, 1, 1))

    with pytest.raises(TypeError) as ei:
        a / a
    # assert "unsupported operand type(s)" in str(ei.value) # TODO

    assert dr.all(ArrayXb([True, True, False, False]) & ArrayXb([True, False, True, False]) == ArrayXb(True, False, False, False))
    assert dr.all(ArrayXb([True, True, False, False]) | ArrayXb([True, False, True, False]) == ArrayXb(True, True, True, False))
    assert dr.all(ArrayXb([True, True, False, False]) ^ ArrayXb([True, False, True, False]) == ArrayXb(False, True, True, False))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test13_binop_promote_broadcast():
    import drjit.llvm as l
    import drjit.scalar as s

    x = s.ArrayXf(10, 100, 1000) + 1
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 101, 1001))
    x = 1 + s.ArrayXf(10, 100, 1000)
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 101, 1001))
    x = s.ArrayXf(10, 100, 1000) + (1, 2, 3)
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 102, 1003))
    x = (1, 2, 3) + s.ArrayXf(10, 100, 1000)
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 102, 1003))
    x = [1, 2, 3] + s.ArrayXf(10, 100, 1000)
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 102, 1003))
    x = s.ArrayXf(10, 100, 1000) + [1, 2, 3]
    assert type(x) is s.ArrayXf and dr.all(x == s.ArrayXf(11, 102, 1003))

    x = s.Array3f(10, 100, 1000) + 1
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 101, 1001))
    x = 1 + s.Array3f(10, 100, 1000)
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 101, 1001))
    x = s.Array3f(10, 100, 1000) + (1, 2, 3)
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
    x = (1, 2, 3) + s.Array3f(10, 100, 1000)
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
    x = [1, 2, 3] + s.Array3f(10, 100, 1000)
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))
    x = s.Array3f(10, 100, 1000) + [1, 2, 3]
    assert type(x) is s.Array3f and dr.all(x == s.Array3f(11, 102, 1003))

    x = l.Float(10, 100, 1000) + 1
    assert type(x) is l.Float and dr.all(x == l.Float(11, 101, 1001))
    x = 1 + l.Float(10, 100, 1000)
    assert type(x) is l.Float and dr.all(x == l.Float(11, 101, 1001))
    x = l.Float(10, 100, 1000) + (1, 2, 3)
    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
    x = (1, 2, 3) + l.Float(10, 100, 1000)
    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
    x = [1, 2, 3] + l.Float(10, 100, 1000)
    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))
    x = l.Float(10, 100, 1000) + [1, 2, 3]
    assert type(x) is l.Float and dr.all(x == l.Float(11, 102, 1003))

    x = l.Array3f(10, 100, 1000) + 1
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
    x = 1 + l.Array3f(10, 100, 1000)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
    x = l.Array3f(10, 100, 1000) + (1, 2, 3)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
    x = (1, 2, 3) + l.Array3f(10, 100, 1000)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
    x = [1, 2, 3] + l.Array3f(10, 100, 1000)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))
    x = l.Array3f(10, 100, 1000) + [1, 2, 3]
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 102, 1003))

    x = s.Array3i(10, 100, 1000) + l.Float(1)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))
    x = s.Array3f(10, 100, 1000) + l.Float(1)
    assert type(x) is l.Array3f and dr.all_nested(x == l.Array3f(11, 101, 1001))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test14_binop_inplace():
    import drjit.scalar as s
    import drjit.llvm as l

    a = s.Array3f(1, 2, 3)
    b = s.Array3f(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == s.Array3f(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == s.Array3f(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == s.Array3f(3, 4, 2))

    a = s.ArrayXf(1, 2, 3)
    b = s.ArrayXf(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == s.ArrayXf(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == s.ArrayXf(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == s.ArrayXf(3, 4, 2))

    a = l.Float(1, 2, 3)
    b = l.Float(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == l.Float(3, 5, 4))
    a += 1
    assert a is c and dr.all(a == l.Float(4, 6, 5))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == l.Float(3, 4, 2))

    a = l.ArrayXf(1, 2, 3)
    b = l.ArrayXf(2, 3, 1)
    c = a
    a += b
    assert a is c and dr.all(a == l.ArrayXf(3, 5, 4))
    a = 1
    c = a
    a += b
    assert a is not c and dr.all(a == l.ArrayXf(3, 4, 2))


@pytest.mark.skip("nanobind layer")
@pytest.mark.parametrize('m', [dr.scalar, dr.llvm])
def test15_unop(m):
    assert dr.all(-m.ArrayXf(1, 2, 3) == m.ArrayXf(-1, -2, -3))
    assert dr.all(+m.ArrayXf(1, 2, 3) == m.ArrayXf(1, 2, 3))
    assert dr.all(abs(m.ArrayXf(1, -2, 3)) == m.ArrayXf(1, 2, 3))
    assert dr.all(-m.Array3f(1, 2, 3) == m.Array3f(-1, -2, -3))
    assert dr.all(+m.Array3f(1, 2, 3) == m.Array3f(1, 2, 3))
    assert dr.all(abs(m.Array3f(1, -2, 3)) == m.Array3f(1, 2, 3))
    assert dr.all(-m.Array3i(1, 2, 3) == m.Array3i(-1, -2, -3))
    assert dr.all(+m.Array3i(1, 2, 3) == m.Array3i(1, 2, 3))
    assert dr.all(~m.Array3i(1, 2, 3) == m.Array3i(-2, -3, -4))
    assert dr.all(abs(m.Array3i(1, -2, 3)) == m.Array3i(1, 2, 3))
    assert dr.all(~m.Array3b(True, False, True) == m.Array3b(False, True, False))
    assert dr.all(~m.ArrayXb(True, False, True) == m.ArrayXb(False, True, False))


@pytest.mark.skip("nanobind layer")
def test16_type_promotion_errors():
    from drjit.scalar import Array3f
    a = Array3f()
    with pytest.raises(TypeError) as ei:
        a + "asdf"
    assert "Array3f.__add__(): encountered an unsupported argument of type 'str' (must be a Dr.Jit array or a type that can be converted into one)" in str(ei.value)

    a + 2**10
    with pytest.raises(TypeError):
        a + 2**100


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test18_traits():
    import drjit.scalar as s
    import drjit.llvm as l
    from drjit import Dynamic

    assert not dr.is_array_v(()) and not dr.is_array_v(1.0)
    assert dr.is_array_v(s.Array3f) and dr.is_array_v(s.Array3f())
    assert dr.is_array_v(s.ArrayXf) and dr.is_array_v(s.ArrayXf())
    assert dr.is_array_v(l.Array3f) and dr.is_array_v(l.Array3f())
    assert dr.is_array_v(l.ArrayXf) and dr.is_array_v(l.ArrayXf())
    assert dr.size_v(1) == 1
    assert dr.size_v("test") == 1
    assert dr.size_v(s.Array3f) == 3 and dr.size_v(s.Array3f()) == 3
    assert dr.size_v(l.Array3f) == 3 and dr.size_v(l.Array3f()) == 3
    assert dr.size_v(s.ArrayXf) == Dynamic and dr.size_v(s.ArrayXf()) == Dynamic
    assert dr.size_v(l.ArrayXf) == Dynamic and dr.size_v(l.ArrayXf()) == Dynamic

    assert dr.depth_v(1) == 0
    assert dr.depth_v("test") == 0
    assert dr.depth_v(s.Array3f) == 1 and dr.depth_v(s.Array3f()) == 1
    assert dr.depth_v(s.ArrayXf) == 1 and dr.depth_v(s.ArrayXf()) == 1
    assert dr.depth_v(l.Float) == 1 and dr.depth_v(l.Float()) == 1
    assert dr.depth_v(l.Array3f) == 2 and dr.depth_v(l.Array3f()) == 2
    assert dr.depth_v(l.ArrayXf) == 2 and dr.depth_v(l.ArrayXf()) == 2

    assert dr.scalar_t(1) is int
    assert dr.scalar_t("test") is str
    assert dr.scalar_t(s.Array3f) is float and dr.scalar_t(s.Array3f()) is float
    assert dr.scalar_t(s.Array3b) is bool and dr.scalar_t(s.Array3b()) is bool
    assert dr.scalar_t(s.ArrayXf) is float and dr.scalar_t(s.ArrayXf()) is float
    assert dr.scalar_t(s.ArrayXb) is bool and dr.scalar_t(s.ArrayXb()) is bool
    assert dr.scalar_t(l.Float) is float and dr.scalar_t(l.Float()) is float
    assert dr.scalar_t(l.Bool) is bool and dr.scalar_t(l.Bool()) is bool
    assert dr.scalar_t(l.Array3f) is float and dr.scalar_t(l.Array3f()) is float
    assert dr.scalar_t(l.Array3b) is bool and dr.scalar_t(l.Array3b()) is bool
    assert dr.scalar_t(l.ArrayXf) is float and dr.scalar_t(l.ArrayXf()) is float
    assert dr.scalar_t(l.ArrayXb) is bool and dr.scalar_t(l.ArrayXb()) is bool

    assert dr.value_t(1) is int
    assert dr.value_t("test") is str
    assert dr.value_t(s.Array3f) is float and dr.value_t(s.Array3f()) is float
    assert dr.value_t(s.Array3b) is bool and dr.value_t(s.Array3b()) is bool
    assert dr.value_t(s.ArrayXf) is float and dr.value_t(s.ArrayXf()) is float
    assert dr.value_t(s.ArrayXb) is bool and dr.value_t(s.ArrayXb()) is bool
    assert dr.value_t(l.Float) is float and dr.value_t(l.Float()) is float
    assert dr.value_t(l.Bool) is bool and dr.value_t(l.Bool()) is bool
    assert dr.value_t(l.Array3f) is l.Float and dr.value_t(l.Array3f()) is l.Float
    assert dr.value_t(l.Array3b) is l.Bool and dr.value_t(l.Array3b()) is l.Bool
    assert dr.value_t(l.ArrayXf) is l.Float and dr.value_t(l.ArrayXf()) is l.Float
    assert dr.value_t(l.ArrayXb) is l.Bool and dr.value_t(l.ArrayXb()) is l.Bool

    assert dr.mask_t(1) is bool
    assert dr.mask_t("test") is bool
    assert dr.mask_t(s.Array3f) is s.Array3b and dr.mask_t(s.Array3f()) is s.Array3b
    assert dr.mask_t(s.Array3b) is s.Array3b and dr.mask_t(s.Array3b()) is s.Array3b
    assert dr.mask_t(s.ArrayXf) is s.ArrayXb and dr.mask_t(s.ArrayXf()) is s.ArrayXb
    assert dr.mask_t(s.ArrayXb) is s.ArrayXb and dr.mask_t(s.ArrayXb()) is s.ArrayXb
    assert dr.mask_t(l.Float) is l.Bool and dr.mask_t(l.Float()) is l.Bool
    assert dr.mask_t(l.Bool) is l.Bool and dr.mask_t(l.Bool()) is l.Bool
    assert dr.mask_t(l.Array3f) is l.Array3b and dr.mask_t(l.Array3f()) is l.Array3b
    assert dr.mask_t(l.Array3b) is l.Array3b and dr.mask_t(l.Array3b()) is l.Array3b
    assert dr.mask_t(l.ArrayXf) is l.ArrayXb and dr.mask_t(l.ArrayXf()) is l.ArrayXb
    assert dr.mask_t(l.ArrayXb) is l.ArrayXb and dr.mask_t(l.ArrayXb()) is l.ArrayXb

    assert dr.is_integral_v(1) and dr.is_integral_v(int)
    assert dr.is_integral_v(s.Array3i()) and dr.is_integral_v(s.Array3i)
    assert not dr.is_integral_v(1.0) and not dr.is_integral_v(float)
    assert not dr.is_integral_v(s.Array3f()) and not dr.is_integral_v(s.Array3f)
    assert not dr.is_integral_v("str") and not dr.is_integral_v(str)
    assert not dr.is_integral_v(False) and not dr.is_integral_v(bool)
    assert not dr.is_integral_v(s.Array3b()) and not dr.is_integral_v(s.Array3b)

    assert not dr.is_float_v(1) and not dr.is_float_v(int)
    assert not dr.is_float_v(s.Array3i()) and not dr.is_float_v(s.Array3i)
    assert dr.is_float_v(1.0) and dr.is_float_v(float)
    assert dr.is_float_v(s.Array3f()) and dr.is_float_v(s.Array3f)
    assert not dr.is_float_v("str") and not dr.is_float_v(str)
    assert not dr.is_float_v(False) and not dr.is_float_v(bool)
    assert not dr.is_float_v(s.Array3b()) and not dr.is_float_v(s.Array3b)

    assert dr.is_arithmetic_v(1) and dr.is_arithmetic_v(int)
    assert dr.is_arithmetic_v(s.Array3i()) and dr.is_arithmetic_v(s.Array3i)
    assert dr.is_arithmetic_v(1.0) and dr.is_arithmetic_v(float)
    assert dr.is_arithmetic_v(s.Array3f()) and dr.is_arithmetic_v(s.Array3f)
    assert not dr.is_arithmetic_v("str") and not dr.is_arithmetic_v(str)
    assert not dr.is_arithmetic_v(False) and not dr.is_arithmetic_v(bool)
    assert not dr.is_arithmetic_v(s.Array3b()) and not dr.is_arithmetic_v(s.Array3b)

    assert not dr.is_mask_v(1) and not dr.is_mask_v(int)
    assert not dr.is_mask_v(s.Array3i()) and not dr.is_mask_v(s.Array3i)
    assert not dr.is_mask_v(1.0) and not dr.is_mask_v(float)
    assert not dr.is_mask_v(s.Array3f()) and not dr.is_mask_v(s.Array3f)
    assert not dr.is_mask_v("str") and not dr.is_mask_v(str)
    assert dr.is_mask_v(False) and dr.is_mask_v(bool)
    assert dr.is_mask_v(s.Array3b()) and dr.is_mask_v(s.Array3b)

    assert dr.uint32_array_t(float) is int
    assert dr.bool_array_t(float) is bool
    assert dr.float32_array_t(int) is float

    assert dr.bool_array_t(dr.scalar.Array3f) is dr.scalar.Array3b
    assert dr.int32_array_t(dr.scalar.Array3f) is dr.scalar.Array3i
    assert dr.uint32_array_t(dr.scalar.Array3f64) is dr.scalar.Array3u
    assert dr.int64_array_t(dr.scalar.Array3f) is dr.scalar.Array3i64
    assert dr.uint64_array_t(dr.scalar.Array3f) is dr.scalar.Array3u64
    assert dr.uint_array_t(dr.scalar.Array3f) is dr.scalar.Array3u
    assert dr.int_array_t(dr.scalar.Array3f) is dr.scalar.Array3i
    assert dr.uint_array_t(dr.scalar.Array3f64) is dr.scalar.Array3u64
    assert dr.int_array_t(dr.scalar.Array3f64) is dr.scalar.Array3i64
    assert dr.float_array_t(dr.scalar.Array3u) is dr.scalar.Array3f
    assert dr.float32_array_t(dr.scalar.Array3u) is dr.scalar.Array3f
    assert dr.float_array_t(dr.scalar.Array3u64) is dr.scalar.Array3f64
    assert dr.float32_array_t(dr.scalar.Array3u64) is dr.scalar.Array3f
    assert dr.float_array_t(dr.scalar.TensorXu64) is dr.scalar.TensorXf64


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test19_select():
    import drjit.scalar as s
    import drjit.llvm as l
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


@pytest.mark.skip("nanobind layer")
def test20_component_access():
    from drjit.scalar import Array3f, Array4f
    from drjit.llvm import Array3f as Array3fL

    a = Array4f(4, 5, 6, 7)
    assert a.x == 4 and a.y == 5 and a.z == 6 and a.w == 7
    a.x, a.y, a.z, a.w = 1, 2, 3, 4
    assert a.x == 1 and a.y == 2 and a.z == 3 and a.w == 4
    a = Array3f(1, 2, 3)
    assert a.x == 1 and a.y == 2 and a.z == 3

    assert a.index == 0 and a.index_ad == 0

    with pytest.raises(TypeError) as ei:
        a.w == 4
    assert "Array3f: does not have a 'w' component!" in str(ei.value)

    a = Array3fL(1, 2, 3)
    assert a.index == 0 and a.index_ad == 0
    assert a.x.index != 0 and a.y.index != 0 and a.z.index != 0


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test21_zero_or_full():
    import drjit.scalar as s
    import drjit.llvm as l
    assert type(dr.zeros(dtype=int)) is int and dr.zeros(dtype=int) == 0
    assert type(dr.zeros(dtype=int, shape=(1,))) is int and dr.zeros(dtype=int, shape=(1,)) == 0
    assert type(dr.zeros(float)) is float and dr.zeros(float) == 0.0
    assert type(dr.zeros(float, shape=(1,))) is float and dr.zeros(float, shape=(1,)) == 0.0
    assert type(dr.zeros(float, shape=(100,))) is float and dr.zeros(float, shape=(100,)) == 0.0
    with pytest.raises(TypeError) as ei:
        dr.zeros(str)
    assert "Unsupported dtype!" in str(ei.value)

    with pytest.raises(TypeError) as ei:
        dr.zeros(5)
    assert "incompatible function arguments" in str(ei.value)

    assert type(dr.zeros(s.Array3f)) is s.Array3f and dr.all(dr.zeros(s.Array3f) == s.Array3f(0))
    assert type(dr.zeros(s.Array3f, shape=(3,))) is s.Array3f and dr.all(dr.zeros(s.Array3f, shape=(3,)) == s.Array3f(0))
    assert type(dr.zeros(l.Array3f)) is l.Array3f and dr.all(dr.zeros(l.Array3f) == l.Array3f(0))
    assert type(dr.zeros(l.Array3f, shape=(3, 5))) is l.Array3f and dr.shape(dr.zeros(l.Array3f, shape=(3, 5))) == (3, 5)
    assert type(dr.zeros(l.Array3f, shape=10)) is l.Array3f and dr.shape(dr.zeros(l.Array3f, shape=10)) == (3, 10)
    assert type(dr.zeros(l.ArrayXf, shape=(8, 5))) is l.ArrayXf and dr.shape(dr.zeros(l.ArrayXf, shape=(8, 5))) == (8, 5)
    assert type(dr.zeros(l.Array3b, shape=10)) is l.Array3b and dr.shape(dr.zeros(l.Array3b, shape=10)) == (3, 10)
    assert type(dr.zeros(l.ArrayXb, shape=(8, 5))) is l.ArrayXb and dr.shape(dr.zeros(l.ArrayXb, shape=(8, 5))) == (8, 5)
    assert type(dr.full(l.ArrayXf, value=123, shape=(8, 5))) is l.ArrayXf and dr.all_nested(dr.full(l.ArrayXf, value=123, shape=(8, 5)) == 123)

    class Ray:
        DRJIT_STRUCT = { 'o': l.Array3f, 'd': l.Array3f, 'maxt' : l.Float, 'hit' : l.Bool }

    ray = dr.zeros(Ray, 10)
    assert isinstance(ray, Ray)
    assert isinstance(ray.o, l.Array3f) and isinstance(ray.d, l.Array3f) \
        and isinstance(ray.maxt, l.Float) and isinstance(ray.hit , l.Bool)

    assert ray.o.shape == (3, 10) and ray.d.shape == (3, 10) and ray.maxt.shape == (10,)
    assert dr.all_nested(ray.o == 0) and dr.all_nested(ray.d == 0) and dr.all(ray.maxt == 0) \
        and dr.all(ray.hit == False)

    ray = dr.ones(Ray, 10)
    assert dr.all_nested(ray.o == 1) and dr.all_nested(ray.d == 1) and dr.all(ray.maxt == 1) \
        and dr.all(ray.hit == True)


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test22_masked_assignment():
    import drjit.scalar as s
    import drjit.llvm as l

    a = s.Array3f(1,2,3)
    a[a <= 2] = 10
    assert dr.all(a == s.Array3f(10, 10, 3))

    a = l.Float(1,2,3)
    a[a <= 2] = 10
    assert dr.all(a == l.Float(10, 10, 3))

    a = l.Array3f(1,2,3)
    a[a <= 2] = 10
    assert dr.all(a == l.Array3f(10, 10, 3))


def test23_linspace():
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


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test24_arange():
    import drjit.scalar as s
    import drjit.llvm as l

    assert dr.all(dr.arange(s.ArrayXu, 5) == s.ArrayXu(0, 1, 2, 3, 4))
    assert dr.all(dr.arange(s.ArrayXf, 5) == s.ArrayXf(0, 1, 2, 3, 4))
    assert dr.all(dr.arange(s.ArrayXi, -2, 5, 2) == s.ArrayXi(-2, 0, 2, 4))
    assert dr.all(dr.arange(dtype=s.ArrayXf, start=-2, stop=5, step=2) == s.ArrayXf(-2, 0, 2, 4))

    assert dr.all(dr.arange(l.UInt, 5) == l.UInt(0, 1, 2, 3, 4))
    assert dr.all(dr.arange(l.Float, 5) == l.Float(0, 1, 2, 3, 4))
    assert dr.all(dr.arange(l.Int, -2, 5, 2) == l.Int(-2, 0, 2, 4))
    assert dr.all(dr.arange(dtype=l.Float, start=-2, stop=5, step=2) == l.Float(-2, 0, 2, 4))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test25_linspace():
    import drjit.scalar as s
    import drjit.llvm as l

    assert dr.allclose(dr.linspace(s.ArrayXf, -2, 4, 4), s.ArrayXf(-2, 0, 2, 4))
    assert dr.allclose(dr.linspace(s.ArrayXf, start=-2, stop=5, num=4), s.ArrayXf(-2, 1/3, 8/3, 5))
    assert dr.allclose(dr.linspace(s.ArrayXf, start=-2, stop=4, num=4, endpoint=False), s.ArrayXf(-2, -0.5, 1, 2.5))
    assert dr.allclose(dr.linspace(l.Float, -2, 4, 4), l.Float(-2, 0, 2, 4))
    assert dr.allclose(dr.linspace(l.Float, start=-2, stop=5, num=4), l.Float(-2, 1/3, 8/3, 5))
    assert dr.allclose(dr.linspace(l.Float, start=-2, stop=4, num=4, endpoint=False), l.Float(-2, -0.5, 1, 2.5))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test26_fast_cast(capsys):
    import drjit.llvm as l
    try:
        dr.set_log_level(5)
        x = l.Int(0, 1, 2)
        y = l.Float(x)
        z = x + 0.0

        a = l.Array3f(y, y, y)
        b = l.Array3u(a)

        out, err = capsys.readouterr()
        assert out.count('jit_var_cast') == 5
        assert out.count('jit_poke') == 3
    finally:
        dr.set_log_level(0)


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test27_gather_simple():
    import drjit.llvm as l
    import drjit.scalar as s

    assert dr.all(dr.gather(
        dtype=s.ArrayXf,
        source=dr.arange(s.ArrayXf, 10),
        index=s.ArrayXu(0, 5, 3)
    ) == s.ArrayXf(0, 5, 3))

    assert dr.all(dr.gather(
        dtype=s.ArrayXf,
        source=dr.arange(s.ArrayXf, 10),
        index=s.ArrayXu(0, 5, 3),
        active=s.ArrayXb(True, False, True)
    ) == s.ArrayXf(0, 0, 3))

    assert dr.all(dr.gather(
        dtype=l.Float,
        source=dr.arange(l.Float, 10),
        index=l.UInt(0, 5, 3)
    ) == l.Float(0, 5, 3))

    assert dr.all(dr.gather(
        dtype=l.Float,
        source=dr.arange(l.Float, 10),
        index=l.UInt(0, 5, 3),
        active=l.Bool(True, False, True)
    ) == l.Float(0, 0, 3))



@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test28_scatter_simple():
    import drjit.llvm as l
    import drjit.scalar as s

    target = dr.empty(s.ArrayXf, 3)
    dr.scatter(target, 1, [0, 1])
    dr.scatter(target, 2, 2)
    assert dr.all(target == s.ArrayXf([1, 1, 2]))

    target = dr.full(s.ArrayXf, shape=3, value=5)
    dr.scatter(target, [10, 12], [0, 1], [True, False])
    dr.scatter(target, 2, 2)
    assert dr.all(target == s.ArrayXf([10, 5, 2]))

    target = dr.empty(l.Float, 3)
    dr.scatter(target, 1, [0, 1])
    dr.scatter(target, 2, 2)
    assert dr.all(target == l.Float([1, 1, 2]))

    target = dr.full(l.Float, shape=3, value=5)
    dr.scatter(target, [10, 12], [0, 1], [True, False])
    dr.scatter(target, 2, 2)
    assert dr.all(target == l.Float([10, 5, 2]))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test29_gather_complex():
    import drjit.scalar as s
    import drjit.llvm as l

    assert dr.all_nested(dr.gather(
        dtype=l.Array3f,
        source=dr.arange(l.Float, 12),
        index=l.UInt(0, 2, 3)
    ) == l.Array3f(
        l.Float(0, 6, 9),
        l.Float(1, 7, 10),
        l.Float(2, 8, 11),
    ))

    assert dr.all_nested(dr.gather(
        dtype=l.Array3f,
        source=dr.arange(l.Float, 12),
        index=l.UInt(0, 2, 3),
        active=l.Bool(True, False, True)
    ) == l.Array3f(
        l.Float(0, 0, 9),
        l.Float(1, 0, 10),
        l.Float(2, 0, 11),
    ))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test30_gather_complex_2():
    x = dr.scalar.ArrayXf([1, 2, 3, 4])
    y = dr.scalar.ArrayXf([5, 6, 7, 8])
    i = dr.scalar.ArrayXu([1, 0])

    class MyStruct:
        DRJIT_STRUCT = { 'a' : dr.scalar.ArrayXf }
        def __init__(self, a: dr.scalar.ArrayXf=1):
            self.a = a

    x = MyStruct(x)
    r = dr.gather(MyStruct, x, i)
    assert type(r) is MyStruct
    assert dr.all(r.a == dr.scalar.ArrayXf([2, 1]))

    pytest.skip("nanobind layer")

    r = dr.gather(tuple, (x, y), i)
    assert type(r) is tuple and len(r) == 2
    assert dr.all(r[0] == dr.scalar.ArrayXf([2, 1]))
    assert dr.all(r[1] == dr.scalar.ArrayXf([6, 5]))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test31_scatter_complex():
    import drjit.scalar as s
    import drjit.llvm as l

    target = dr.empty(l.Float, 6)
    # dr.scatter(target, l.Array3f([[1, 4], [2, 5], [3, 6]]), [0, 1]) # TODO
    dr.scatter(target, l.Array3f([[1, 4], [2, 5], [3, 6]]), l.UInt([0, 1]))

    assert dr.all(target == l.Float([1, 2, 3, 4, 5, 6]))


@pytest.mark.skip("nanobind layer")
def test31_scatter_complex_2():
    x = dr.scalar.ArrayXf([1, 2, 3, 4])
    y = dr.scalar.ArrayXf([5, 6, 7, 8])
    a = dr.scalar.ArrayXf([10, 11])
    b = dr.scalar.ArrayXf([20, 21])
    i = dr.scalar.ArrayXu([1, 0])
    dr.scatter((x, y), (a, b), i)
    assert dr.all(x == dr.scalar.ArrayXf([11, 10, 3, 4]))
    assert dr.all(y == dr.scalar.ArrayXf([21, 20, 7, 8]))

    class MyStruct:
        DRJIT_STRUCT = { 'a' : dr.scalar.ArrayXf }
        def __init__(self, a: dr.scalar.ArrayXf):
            self.a = a

    x = MyStruct(x)
    y = MyStruct(b)
    dr.scatter(x, y, i)
    assert dr.all(x.a == dr.scalar.ArrayXf(21, 20, 3, 4))


@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test32_ravel(capsys):
    import drjit.scalar as s
    import drjit.llvm as l

    try:
        dr.set_log_level(5)

        assert dr.all(dr.ravel(s.Array3f(1, 2, 3), order='C') == s.ArrayXf([1, 2, 3]))
        assert dr.all(dr.ravel(s.Array3f(1, 2, 3), order='F') == s.ArrayXf([1, 2, 3]))

        x = l.Array3f([1, 2], [3, 4], [5, 6])
        assert dr.all(dr.ravel(x.x) is x.x)
        assert dr.all(dr.ravel(x, order='C') == l.Float([1, 2, 3, 4, 5, 6]))
        assert dr.all(dr.ravel(x) == l.Float([1, 3, 5, 2, 4, 6]))

        out, err = capsys.readouterr()
        assert out.count('jit_var_scatter') == 6
        assert out.count('jit_poke') == 18
    finally:
        dr.set_log_level(0)


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test33_unravel(capsys):
    import drjit.scalar as s
    import drjit.llvm as l

    try:
        dr.set_log_level(5)

        assert dr.all(dr.unravel(s.Array3f, s.ArrayXf([1, 2, 3]), order='C') == s.Array3f(1, 2, 3))
        assert dr.all(dr.unravel(s.Array3f, s.ArrayXf([1, 2, 3]), order='F') == s.Array3f(1, 2, 3))

        x = l.Float([1, 2, 3, 4, 5, 6])
        assert dr.all(dr.unravel(l.Float, x) is x)
        assert dr.all_nested(dr.unravel(l.Array3f, x, order='C') == l.Array3f([1, 2], [3, 4], [5, 6]))
        assert dr.all_nested(dr.unravel(l.Array3f, x) == l.Array3f([1, 4], [2, 5], [3, 6]))

        out, err = capsys.readouterr()
        assert out.count('jit_var_new_gather') == 6
        assert out.count('jit_poke') == 18
    finally:
        dr.set_log_level(0)


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test34_schedule(capsys):
    import drjit.llvm as l
    try:
        class MyStruct:
            DRJIT_STRUCT = { 'a' : l.Float }
            def __init__(self, a: l.Float):
                self.a = a

        dr.set_log_level(5)
        assert dr.schedule() is False
        assert dr.schedule(123, [], [123], (), (123,), {123: 456}) is False
        x = l.Float([1, 2]) + 3
        assert dr.schedule(x) is True

        y = l.Array3f(x + 1, x + 2, x + 3)
        assert dr.schedule({'hello': [(y, 4)]}) is True
        z = MyStruct(x + 4)
        dr.schedule(z)
        out, err = capsys.readouterr()
        assert out.count('jit_var_schedule') == 5
    finally:
        dr.set_log_level(0)


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test35_to_dlpack_numpy_cpu():
    import drjit.scalar as s
    import drjit.llvm as l

    from_dlpack = None
    try:
        import numpy as np
        from_dlpack = getattr(np, '_from_dlpack', None)
        from_dlpack = getattr(np, 'from_dlpack', from_dlpack)
    except:
        pass

    if from_dlpack is None:
        pytest.skip('NumPy is missing/too old')

    x = l.Array3f([1, 2], [3, 4], [5, 6])
    assert x.__dlpack_device__() == (1, 0)
    assert np.all(from_dlpack(x) == np.array([[1, 2], [3, 4], [5, 6]]))
    assert np.all(x.__array__() == np.array([[1, 2], [3, 4], [5, 6]]))

    x = s.ArrayXf(1, 2, 3, 4)
    assert np.all(x.__array__() == np.array([1, 2, 3, 4]))
    x = s.Array3f(1, 2, 3)
    assert np.all(x.__array__() == np.array([1, 2, 3]))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason='CUDA mode is unsupported')
def test36_to_dlpack_numpy_gpu():
    np = pytest.importorskip("numpy")

    import drjit.cuda as c

    x = c.Array3f([1, 2], [3, 4], [5, 6])
    assert np.all(x.__array__() == np.array([[1, 2], [3, 4], [5, 6]]))


def test38_construct_from_numpy_1():
    # Simple scalar conversions, different types, static sizes
    np = pytest.importorskip("numpy")

    import drjit.scalar as s

    assert dr.all(s.Array3f(np.array([1, 2, 3], dtype=np.float32)) == s.Array3f(1, 2, 3))
    assert dr.all(s.Array3f(np.array([1, 2, 3], dtype=np.float64)) == s.Array3f(1, 2, 3))
    assert dr.all(s.Array3f(np.array([1, 2, 3], dtype=np.int32)) == s.Array3f(1, 2, 3))

    with pytest.raises(TypeError):
        # Size mismatch
        s.Array3f(np.array([1, 2, 3, 4], dtype=np.float32))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test39_construct_from_numpy_2():
    # Dynamically allocated arrays + implicit casts
    np = pytest.importorskip("numpy")

    import drjit.scalar as s
    import drjit.llvm as l

    p = np.array([1, 2, 3, 4], dtype=np.float32)
    r = s.ArrayXf(p)
    assert dr.all(r == s.ArrayXf(1, 2, 3, 4))

    p = np.array([1, 2, 3, 4], dtype=np.float32)
    r = l.Float(p)
    assert dr.all(r == l.Float(1, 2, 3, 4))

    # Check if zero-copy constructor works
    p[0] = 5
    assert dr.all(r == l.Float(5, 2, 3, 4))

    r = l.Float64(p)
    assert dr.all(r == l.Float64(5, 2, 3, 4))
    p[0] = 1
    assert dr.all(r == l.Float64(5, 2, 3, 4))

    with pytest.raises(TypeError) as ei:
        l.Array4f(p)
    assert "unable to initialize from tensor of type 'numpy.ndarray'. The input must have the following configuration for this to succeed: shape=(4, *), dtype=float32, order='C'" in str(ei.value)


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test40_construct_from_numpy_3():
    # Nested arrays, CPU-only
    np = pytest.importorskip("numpy")

    import drjit.llvm as l

    p = np.array([[1, 2], [4, 5], [6, 7]], dtype=np.float32)

    with pytest.raises(TypeError) as ei:
        r = l.Float(p)
    assert "The input must have the following configuration for this to succeed: shape=(*), dtype=float32, order='C'" in str(ei.value)

    r = l.Array3f(p)
    assert dr.all_nested(r == l.Array3f([1, 2], [4, 5], [6, 7]))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.CUDA), reason='CUDA mode is unsupported')
def test41_construct_from_numpy_4():
    # Nested arrays, copy to CUDA Dr.Jit array
    np = pytest.importorskip("numpy")

    import drjit.cuda as c

    p = np.array([[1, 2], [4, 5], [6, 7]], dtype=np.float32)

    r = c.Array3f(p)
    assert dr.all_nested(r == c.Array3f([1, 2], [4, 5], [6, 7]))
    r = c.Array3i64(p)
    assert dr.all_nested(r == c.Array3i64([1, 2], [4, 5], [6, 7]))


@pytest.mark.skip("nanobind layer")
@pytest.mark.skipif(not dr.has_backend(dr.JitBackend.LLVM), reason='LLVM mode is unsupported')
def test42_prevent_inefficient_cast(capsys):
    import drjit.scalar as s
    import drjit.llvm as l
    import drjit.llvm.ad as la

    with pytest.raises(TypeError) as ei:
        s.ArrayXf(la.Float([1, 2, 3, 4]))

    with pytest.raises(TypeError) as ei:
        l.Float(s.ArrayXf([1, 2, 3, 4]))

    with pytest.raises(TypeError) as ei:
        l.Float(la.Float([1, 2, 3, 4]))

    with pytest.warns(RuntimeWarning, match=r"implicit conversion"):
        with pytest.raises(TypeError) as ei:
            l.Array3f(la.Array3f(1))

    #  la.Float(l.Float([1, 2, 3, 4]))


@pytest.fixture(scope="module", params=['drjit.cuda', 'drjit.llvm'])
def m(request):
    if 'cuda' in request.param:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
    yield importlib.import_module(request.param)


def test43_minimum(m):
    assert dr.allclose(dr.minimum(6.0, 4.0), 4.0)

    a = dr.minimum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Float

    a = dr.minimum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Float

    a = dr.minimum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Array3f

    a = dr.minimum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [1, 2, 2])
    assert type(a) is m.Array3f

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 3], [1, 1, 3], [1, 2, 3]])
    assert type(a) is m.ArrayXf

    a = dr.minimum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 1, 1], [2, 1, 2], [3, 3, 3]])
    assert type(a) is m.ArrayXf


def test44_maximum(m):
    assert dr.allclose(dr.maximum(6.0, 4.0), 6.0)

    a = dr.maximum(m.Float([1, 2, 3]), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Float

    a = dr.maximum(m.Float([1, 2, 3]), [2.0, 2.0, 2.0])
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Float

    a = dr.maximum(m.Array3f(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Array3f

    a = dr.maximum(m.Array3i(1, 2, 3), m.Float(2))
    assert dr.allclose(a, [2, 2, 3])
    assert type(a) is m.Array3f

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Float(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) is m.ArrayXf

    a = dr.maximum(m.ArrayXf([1, 2, 5], [2, 1, 4], [3, 4, 3]), m.Array3f(1, 2, 3))
    assert dr.allclose(a, [[1, 2, 5], [2, 2, 4], [3, 4, 3]])
    assert type(a) is m.ArrayXf


def test45_iter(m):
    a = m.Float([0, 1, 2, 3, 4 ,5])
    for i, v in enumerate(a):
        assert v == i

    a = m.Array0f()
    count = 0
    for i, v in enumerate(a):
        count += 1
    assert count == 0

    a = m.Array1f([0, 1, 2, 3, 4 ,5])
    count = 0
    for i, v in enumerate(a):
        assert dr.allclose(v, [0, 1, 2, 3, 4 ,5])
        count += 1
    assert count == 1

    a = m.Array3f([[0, 0], [1, 1], [2, 2]])
    count = 0
    for i, v in enumerate(a):
        assert dr.allclose(v, i)
        count += 1
    assert count == 3


#@pytest.mark.parametrize('name', ['sqrt', 'cbrt', 'sin', 'cos', 'tan', 'asin',
#                                  'acos', 'atan', 'sinh', 'cosh', 'tanh',
#                                  'asinh', 'acosh', 'atanh', 'exp', 'exp2',
#                                  'log', 'log2', 'floor', 'ceil', 'trunc',
#                                  'round', 'rcp', 'rsqrt'])
#def test17_spotcheck_unary_math(name):
#    from drjit.scalar import ArrayXf, PCG32
#    import math
#    func_ref = getattr(math, name, None)
#    if name == 'cbrt':
#        func_ref = lambda x: x**(1/3)
#    elif name == 'exp2':
#        func_ref = lambda x: 2**x
#    elif name == 'log2':
#        log2 = lambda x: math.log(x) / math.log(2)
#    elif name == 'round':
#        func_ref = round
#    elif name == 'rcp':
#        func_ref = lambda x : 1/x
#    elif name == 'rsqrt':
#        func_ref = lambda x : math.sqrt(1/x)
#
#    rng = PCG32()
#    x = ArrayXf((rng.next_float32() for i in range(10)))
#    if name == 'acosh':
#        x += 1
#    ref = ArrayXf([func_ref(y) for y in x])
#    func = getattr(dr, name)
#    value = func(x)
#    value_2 = ArrayXf(func(y) for y in x)
#    assert dr.allclose(value, func)
#

