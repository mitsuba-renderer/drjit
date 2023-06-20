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
    if size >= 0:
        msg = rf'Input has the wrong size \(expected {size} elements, got {size+1}\).'
        with pytest.raises(TypeError, match=msg):
            t(*([value] * (size + 1)))
        with pytest.raises(TypeError, match=msg):
            t([value] * (size + 1))
        with pytest.raises(TypeError, match=msg):
            t(tuple([value] * (size + 1)))
        with pytest.raises(TypeError, match=msg):
            t((value for i in range(size + 1)))

        # Explicit
        v1 = t(*([value] * size))

        # List
        v2 = t([value] * size)

        # Tuple
        v3 = t(tuple([value] * size))

        # Iterator
        v4 = t((value for i in range(size)))

        # Explicit, from elements of another array
        v5 = t(*[v1[i] for i in range(len(v1))])

        s = str(value)

        if dr.is_matrix_v(t):
            s = '[[' + '],\n ['.join([', '.join([s for i in range(size)]) for j in range(size)]) + ']]'
        elif dr.is_vector_v(t):
            s = '[' + ', '.join([s] * size) + ']'
            if dr.depth_v(t) - dr.is_jit_v(t) > 1:
                s = '[' + ',\n '.join([s]*size) + ']'

        if dr.is_jit_v(t):
            s = '[' + '\n '.join(s.split('\n')) + ']' if size > 0 else '[]'

        assert str(v1) == s
        assert str(v2) == s
        assert str(v3) == s
        assert str(v4) == s
        assert str(v5) == s

        if size > 1:
            msg = 'Item assignment failed.'
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
def test03_init_ragged(t):
    assert str(t(1, [2, 3], 4)) == '[[1, 2, 4],\n [1, 3, 4]]'
    assert str(t([1, 5, 6], [2, 3], 4)) == '[ragged array]'
