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
@pytest.test_arrays('-tensor', '-bool', '-shape=()')
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

@pytest.test_arrays('-tensor')
def test03_init_explicit_list(t):
    with pytest.raises(TypeError, match='Constructor does not take keyword arguments.'):
        t(hello='world')

    value = False if dr.is_mask_v(t) else 0

    size = dr.size_v(t)
    if size >= 0:
        with pytest.raises(TypeError, match=rf'Input has the wrong size \(expected {size} elements, got {size+1}\).'):
            t(*([value] * (size + 1)))

        v = t(*([value] * size))
        s = str(value)

        if dr.is_matrix_v(t):
            s = '[[' + '],\n ['.join([', '.join([s for i in range(size)]) for j in range(size)]) + ']]'
        elif dr.is_vector_v(t):
            s = '[' + ', '.join([s] * size) + ']'
            if dr.depth_v(t) - dr.is_jit_v(t) > 1:
                s = '[' + ',\n '.join([s]*size) + ']'

        if dr.is_jit_v(t):
            s = '[' + '\n '.join(s.split('\n')) + ']' if size > 0 else '[]'

        assert str(v) == s

        if size > 1:
            with pytest.raises(TypeError, match='Item assignment failed.'):
                t(*([None] * size))
