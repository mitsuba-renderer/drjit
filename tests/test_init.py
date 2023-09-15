import drjit as dr
import pytest

# Test the default ``Array()`` initialization for every Dr.Jit type
@pytest.test_arrays()
def test01_init_default(t):
    if dr.is_dynamic_v(t):
        s = '[]'
    else:
        size = dr.size_v(t)
        if dr.is_complex_v(t) or dr.is_quaternion_v(t):
            s = '0'
        else:
            s = 'False' if dr.is_mask_v(t) else '0'
            s = '[' + ', '.join([s] * size) + ']'
        if dr.depth_v(t) > 1:
            s = '[' + ',\n '.join([s]*size) + ']'
    assert str(t()) == s


# Test broadcasting from a constant, i.e., ``Array(1)``
@pytest.test_arrays('-tensor, -bool, -shape=()')
def test02_init_broadcast(t):
    size = dr.size_v(t)
    if size < 0:
        size = 1

    if dr.is_matrix_v(t):
        s = '[[' + '],\n ['.join([', '.join(['1' if i == j else '0' for i in range(size)]) for j in range(size)]) + ']]'
    elif dr.is_vector_v(t):
        s = '[' + ', '.join(['1'] * size) + ']'
        if dr.depth_v(t) - dr.is_jit_v(t) > 1:
            s = '[' + ',\n '.join([s]*size) + ']'
    else:
        s = '1'

    if dr.is_jit_v(t):
        s = '[' + '\n '.join(s.split('\n')) + ']'

    assert str(t(1)) == s

    # Test copy constructor
    assert str(t(t(1))) == s


# Test array initialization from a list of arguments (explicit, list, tuple, iterable)
@pytest.test_arrays('-tensor')
def test03_init_list(t):
    with pytest.raises(TypeError, match='Constructor does not take keyword arguments.'):
        t(hello='world')

    value = False if dr.is_mask_v(t) else 0

    size = dr.size_v(t)
    if size >= 0 or size == dr.Dynamic:
        if size == dr.Dynamic:
            size = 5
        else:
            msg = rf'Input has the wrong size \(expected {size} elements, got {size+1}\).'
            with pytest.raises(TypeError, match=msg):
                t(*([value] * (size + 1)))
            with pytest.raises(TypeError, match=msg):
                t([value] * (size + 1))
            with pytest.raises(TypeError, match=msg):
                t(tuple([value] * (size + 1)))
            with pytest.raises(TypeError, match=msg):
                t((value for i in range(size + 1)))

        class my_list(list):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

        # Explicit
        v1 = t(*([value] * size))

        # List
        v2 = t([value] * size)

        # Tuple
        v3 = t(tuple([value] * size))

        # Sequence
        v4 = t(my_list(value for i in range(size)))

        # Iterator
        v5 = t((value for i in range(size)))

        # Explicit, from elements of another array
        v6 = t(*[v1[i] for i in range(len(v1))])

        s = str(value)

        if dr.is_matrix_v(t):
            s = '[[' + '],\n ['.join([', '.join([s for i in range(size)]) for j in range(size)]) + ']]'
        elif dr.is_vector_v(t) or dr.size_v(t) == dr.Dynamic:
            s = '[' + ', '.join([s] * size) + ']'
            if dr.depth_v(t) - dr.is_jit_v(t) > 1:
                s = '[' + ',\n '.join([s]*size) + ']'

        if dr.is_jit_v(t) and dr.depth_v(t) > 1:
            s = '[' + '\n '.join(s.split('\n')) + ']' if size > 0 else '[]'

        assert str(v1) == s
        assert str(v2) == s
        assert str(v3) == s
        assert str(v4) == s
        assert str(v5) == s
        assert str(v6) == s

        if size > 1:
            msg_1 = r'Could not construct from sequence \(invalid type in input\).'
            msg_2 = 'Item assignment failed.'
            msg = '(' + msg_1 + '|' + msg_2 + ')'
            with pytest.raises(TypeError, match=msg):
                t(*([None] * size))
            with pytest.raises(TypeError, match=msg):
                t([None] * size)
            with pytest.raises(TypeError, match=msg):
                t(tuple([None] * size))
            with pytest.raises(TypeError, match=msg):
                t((None for i in range(size)))


# Test initialization+stringification of arrays with diff. # of components
@pytest.test_arrays('vector, int, shape=(3, *)')
def test04_init_ragged(t):
    assert str(t(1, [2, 3], 4)) == '[[1, 2, 4],\n [1, 3, 4]]'
    assert str(t([1, 5, 6], [2, 3], 4)) == '[ragged array]'


# Test literal initialization
@pytest.test_arrays('float32, shape=(*), jit')
def test05_literal(t, drjit_verbose, capsys):
    v = t(123)
    assert "literal" in capsys.readouterr().out
    v[0] = 124
    assert "jit_poke" in capsys.readouterr().out


# Test efficient initialization from lists/tuples/sequences
@pytest.test_arrays('float32, shape=(*), jit')
def test06_literal(t, drjit_verbose, capsys):
    t(123, 234)
    assert "jit_var_mem_copy" in capsys.readouterr().out
    t([123, 234])
    assert "jit_var_mem_copy" in capsys.readouterr().out
    t(range(10))
    assert "jit_var_mem_copy" in capsys.readouterr().out


# Test dr.zeros (1)
@pytest.test_arrays('float32, shape=(*)')
def test07_zeros(t, drjit_verbose, capsys):
    is_jit = "jit" in t.__meta__
    v = dr.zeros(t)
    assert len(v) == 1 and v[0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    v = dr.zeros(t, 100)
    assert len(v) == 100 and v[0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    v = dr.zeros(t, [100])
    assert len(v) == 100 and v[0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="The provided 'shape' and 'dtype' parameters are incompatible."):
        v = dr.zeros(t, (100, 200))

# Test dr.zeros (2)
@pytest.test_arrays('float32, shape=(3, *)')
def test08_zeros_3d(t, drjit_verbose, capsys):
    is_jit = "jit" in t.__meta__
    v = dr.zeros(t)
    assert len(v) == 3 and len(v[1]) == 1 and v[0][0] == 0

    v = dr.zeros(t, 100)
    assert len(v) == 3 and len(v[1]) == 100 and v[0][0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    v = dr.zeros(t, (3, 100))
    assert len(v) == 3 and len(v[1]) == 100 and v[0][0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="The provided 'shape' and 'dtype' parameters are incompatible."):
        v = dr.zeros(t, (100, 3))


# Test dr.zeros (3)
@pytest.test_arrays('float32, shape=(*, *)')
def test09_zeros_nd(t, drjit_verbose, capsys):
    is_jit = "jit" in t.__meta__
    v = dr.zeros(t)
    assert len(v) == 1 and len(v[1]) == 1 and v[0][0] == 0

    v = dr.zeros(t, 100)
    assert len(v) == 1 and len(v[1]) == 100 and v[0][0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out

    v = dr.zeros(t, (3, 100))
    assert len(v) == 3 and len(v[1]) == 100 and v[0][0] == 0
    assert not is_jit or "literal" in capsys.readouterr().out


# Test dr.zeros (4)
@pytest.test_arrays('float32, matrix, shape=(3, 3, *)')
def test10_zeros_matrix(t):
    v = dr.zeros(t)
    assert len(v) == 3 and len(v[1]) == 3 and len(v[1][1]) == 1 and v[1][1][0] == 0
    v = dr.zeros(t, 100)
    assert len(v) == 3 and len(v[1]) == 3 and len(v[1][1]) == 100 and v[1][1][1] == 0
    v = dr.zeros(t, (3, 3, 100))
    assert len(v) == 3 and len(v[1]) == 3 and len(v[1][1]) == 100 and v[1][1][1] == 0
    with pytest.raises(RuntimeError, match="The provided 'shape' and 'dtype' parameters are incompatible."):
        v = dr.zeros(t, (3, 4, 100))


# Test dr.zeros (5)
@pytest.test_arrays('float32, shape=(*)')
def test11_zeros_struct(t):
    class Q:
        DRJIT_STRUCT = { 'a': t }

    is_jit = "jit" in t.__meta__
    v = dr.zeros(Q)
    assert type(v) is Q and type(v.a) is t and len(v.a) == 1 and v.a[0] == 0
    v = dr.zeros(Q, 100)
    assert type(v) is Q and type(v.a) is t and len(v.a) == 100 and v.a[0] == 0


# Test dr.zeros (t)
def test12_zeros_simple():
    v = dr.zeros(float)
    assert type(v) is float and v == 0
    v = dr.zeros(int)
    assert type(v) is int and v == 0


# Test dr.empty
@pytest.test_arrays('float32, shape=(*)')
def test13_empty(t, drjit_verbose, capsys):
    is_jit = "jit" in t.__meta__
    v = dr.empty(t)
    assert len(v) == 1
    assert not is_jit or "jit_var_new" in capsys.readouterr().out

    v = dr.empty(t, 100)
    assert len(v) == 100
    assert not is_jit or "jit_var_new" in capsys.readouterr().out

    v = dr.empty(t, [100])
    assert len(v) == 100
    assert not is_jit or "jit_var_new" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="The provided 'shape' and 'dtype' parameters are incompatible."):
        v = dr.empty(t, [100, 200])


# Test dr.full (2)
@pytest.test_arrays('float32, shape=(3, *)')
def test14_zeros_3d(t, drjit_verbose, capsys):
    is_jit = "jit" in t.__meta__
    v = dr.full(t, 5)
    assert len(v) == 3 and len(v[1]) == 1 and v[0][0] == 5

    v = dr.full(t, 5, 100)
    assert len(v) == 3 and len(v[1]) == 100 and v[0][0] == 5
    assert not is_jit or "literal" in capsys.readouterr().out

    v = dr.full(dtype=t, value=5, shape=(3, 100))
    assert len(v) == 3 and len(v[1]) == 100 and v[0][0] == 5
    assert not is_jit or "literal" in capsys.readouterr().out

    with pytest.raises(RuntimeError, match="The provided 'shape' and 'dtype' parameters are incompatible."):
        v = dr.full(t, 5, (100, 3))


@pytest.test_arrays('shape=(*), -bool')
def test15_arange(t):
    assert dr.all(dr.arange(t, 5) == t(0, 1, 2, 3, 4))

    if dr.is_signed_v(t):
        assert dr.all(dr.arange(t, -2, 5, 2) == t(-2, 0, 2, 4))
        assert dr.all(dr.arange(t, start=-2, stop=5, step=2) == t(-2, 0, 2, 4))


@pytest.test_arrays('shape=(*), -bool, float')
def test16_linspace(t):
    assert dr.allclose(dr.linspace(t, -2, 4, 4), t(-2, 0, 2, 4))
    assert dr.allclose(dr.linspace(t, start=-2, stop=5, num=4), t(-2, 1/3, 8/3, 5))
    assert dr.allclose(dr.linspace(t, start=-2, stop=4, num=4, endpoint=False), t(-2, -0.5, 1, 2.5))


@pytest.test_arrays('shape=(*), uint32')
def test17_repr_long_1(t):
  assert repr(t(range(1000))) == '[0, 1, 2, 3, 4, .. 990 skipped .., 995, 996, 997, 998, 999]'


@pytest.test_arrays('shape=(1, *), uint32')
def test18_repr_long_2(t):
  assert repr(t([range(1000)])) == '[[0],\n [1],\n [2],\n [3],\n [4],\n .. 990 skipped ..,\n [995],\n [996],\n [997],\n [998],\n [999]]'


@pytest.test_packages()
def test19_shape_vectorized(p):
    if 'scalar' in p.__name__:
        assert dr.shape(p.Array0f()) == (0,) and p.Array0f().shape == (0,)
        assert dr.shape(p.Array2f()) == (2,) and p.Array2f().shape == (2,)
        assert dr.shape(p.Matrix3f()) == (3, 3) and p.Matrix3f().shape == (3, 3)
        assert dr.shape(p.ArrayXf(1, 2, 3)) == (3,) and p.ArrayXf(1, 2, 3).shape == (3,)
        assert p.ArrayXf().ndim == 1
    else:
        v = p.Float([1,2,3,4,5])
        assert dr.shape(p.Array0f()) == (0, 0) and p.Array0f().shape == (0, 0)
        assert dr.shape(p.Array2f()) == (2, 0) and p.Array2f().shape == (2, 0)
        assert dr.shape(p.Matrix3f()) == (3, 3, 0) and p.Matrix3f().shape == (3, 3, 0)
        assert dr.shape(p.Matrix3f(v)) == (3, 3, 5) and p.Matrix3f(v).shape == (3, 3, 5)
        assert dr.shape(p.ArrayXf(1, 2, 3)) == (3, 1) and p.ArrayXf(1, 2, 3).shape == (3, 1)
        assert dr.shape(p.ArrayXf(1, v, 3)) == (3, 5) and p.ArrayXf(1, v, 3).shape == (3, 5)
        assert dr.shape(p.ArrayXf([1, 2], v, 3)) is None and p.ArrayXf([1, 2], v, 3).shape is None
        assert dr.shape(p.Array2f([1, 2], [])) is None and p.Array2f([1, 2], []).shape is None
        assert p.ArrayXf().ndim == 2

def test20_shape_other():
    assert dr.shape(0) == ()
    assert dr.shape([]) == (0,)
    assert dr.shape([1]) == (1,)
    assert dr.shape([1, 2]) == (2,)
    assert dr.shape([[1, 2], [3, 4]]) == (2, 2)
    assert dr.shape([[1, 2, 5], [3, 4, 5]]) == (2, 3)
    assert dr.shape([[[1], [2], [5]], [[3], [4], [5]]]) == (2, 3, 1)
    assert dr.shape((dr.scalar.Array3f(), dr.scalar.Array3f())) == (2, 3)

    # Ragged
    assert dr.shape([[1, 2, 5], [3, 4]]) is None
    assert dr.shape([[1, 2], [3, 4, 5]]) is None
    assert dr.shape([[1, 2], 1]) is None
    assert dr.shape([1, [1, 2]]) is None
    assert dr.shape((dr.scalar.Array2f(), dr.scalar.Array3f())) is None


@pytest.test_arrays('matrix, -jit, float32')
def test21_stringify_matrix_scalar(t):
    if dr.depth_v(t) > 2:
        return
    if dr.size_v(t) == 2:
        m = t([[1,2],[3,4]])
        ref = "[[1, 2],\n [3, 4]]"
    elif dr.size_v(t) == 3:
        m = t([[1,2,3],[4,5,6],[7,8,9]])
        ref = "[[1, 2, 3],\n [4, 5, 6],\n [7, 8, 9]]"
    elif dr.size_v(t) == 4:
        m = t([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
        ref = "[[1, 2, 3, 4],\n [5, 6, 7, 8],\n " \
            "[9, 10, 11, 12],\n [13, 14, 15, 16]]"

    assert str(m) == ref

    try:
        import numpy as np
    except:
        return

    def simplify(s):
        s = s.replace(",", '')
        s = s.replace(' ', '')
        s = s.replace('\n', '')
        s = s.replace('.', '')
        return s

    assert simplify(str(np.array(m))) == simplify(ref)


@pytest.test_arrays('matrix, jit, float32')
def test22_stringify_matrix_vectorized(t):
    if dr.depth_v(t) > 3:
        return
    if dr.size_v(t) == 2:
        m = t([[1,2],[3,[4,-1]]])
        ref = "[[[1, 2],\n  [3, 4]],\n [[1, 2],\n  [3, -1]]]"
    elif dr.size_v(t) == 3:
        m = t([[1,2,3],[4,5,6],[7,8,[9, -1]]])
        ref = "[[[1, 2, 3],\n  [4, 5, 6],\n  [7, 8, 9]],\n" \
            " [[1, 2, 3],\n  [4, 5, 6],\n  [7, 8, -1]]]"
    elif dr.size_v(t) == 4:
        m = t([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,[16, -1]]])
        ref = "[[[1, 2, 3, 4],\n  [5, 6, 7, 8],\n  [9, 10, 11, 12],\n" \
            "  [13, 14, 15, 16]],\n [[1, 2, 3, 4],\n  [5, 6, 7, 8],\n" \
            "  [9, 10, 11, 12],\n  [13, 14, 15, -1]]]"

    assert str(m) == ref

    try:
        import numpy as np
        m = np.array(m)
        m = np.moveaxis(m, -1, 0)
    except:
        return

    def simplify(s):
        s = s.replace(",", '')
        s = s.replace(' ', '')
        s = s.replace('\n', '')
        s = s.replace('.', '')
        return s

    assert simplify(str(m)) == simplify(ref)


@pytest.test_arrays('float32, shape=(3, *)')
def test23_init_from_ndarray_various_cases(t):
    np = pytest.importorskip("numpy")

    v = t(np.array([1, 2, 3], dtype=np.float32))
    assert dr.all(v == t(1, 2, 3))

    v = t(np.array([[1], [2], [3]], dtype=np.float32))
    assert dr.all(v == t(1, 2, 3))

    a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32, order='F')
    v = t(a)
    assert dr.all(v == t([1, 2], [3, 4], [5, 6]), axis=None)
    a[0] = 5 # Gather made a copy, comparison still holds
    assert dr.all(v == t([1, 2], [3, 4], [5, 6]), axis=None)

    # Implicit conversion
    v = t(np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32, order='C'))
    assert dr.all(v == t([1, 2], [3, 4], [5, 6]), axis=None)

    a = np.array([1, 2, 3], dtype=np.float32)
    tv = dr.value_t(t)
    v = tv(a)
    assert dr.all(v == [1, 2, 3])
    a[1] = 5 # Should affect 'v' as well
    assert dr.any(v != [1, 2, 3])
    assert dr.all(v == [1, 5, 3])

    # Once more, with an implicit conversion
    a = np.array([1, 2, 3], dtype=np.float64)
    tv = dr.value_t(t)
    v = tv(a)
    assert dr.all(v == [1, 2, 3])
    a[1] = 5 # This time, 'v' should be unaffected
    assert dr.all(v == [1, 2, 3])

    msg = r"Unable to initialize from an array of type 'ndarray'. The input " \
        r"should have the following configuration for this to succeed: " \
        r"ndim=2, shape=\(3, \*\), dtype=float32, order='C'."

    with pytest.raises(TypeError, match=msg):
        v = t(np.array([[1, 2], [3, 4], [5, 6], [7, 8]]))

    with pytest.raises(TypeError, match=msg):
        v = t(np.array(1))


@pytest.test_arrays('tensor, float32')
def test24_init_tensor_from_ndarray(t):
    np = pytest.importorskip("numpy")

    v = t(np.array(1, dtype=np.float32))
    assert v.shape == () and v.array[0] == 1

    v = t(np.array([1], dtype=np.float32))
    assert v.shape == (1,) and v.array[0] == 1

    a = np.array([1, 2], dtype=np.float32)
    v = t(a)
    assert v.shape == (2,) and dr.all(v.array == (1, 2))
    if dr.is_jit_v(t):
        a[0] = 5
        assert dr.any(v.array != (1, 2)) and dr.all(v.array == (5, 2))

    a = np.array([1, 2], dtype=np.int32)
    v = t(a)
    assert v.shape == (2,) and dr.all(v.array == (1, 2))
    if dr.is_jit_v(t):
        a[0] = 5
        assert dr.any(v.array == (1, 2))

    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert dr.all(t(a) == t([[1, 2, 3], [4, 5, 6]]), axis=None)
