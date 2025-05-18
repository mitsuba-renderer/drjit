import drjit as dr
import pytest

# Just an existence test
@pytest.test_arrays('float32, jit, shape=(*)')
def test01_reorder_switch(t):
    UInt32 = dr.uint32_array_t(t)
    N = 4

    idx = dr.arange(UInt32, N) % 2
    arg = dr.arange(t, N)
    dr.make_opaque(arg)

    def cheap_func(arg):
        return arg

    def expensive_func(arg):
        return arg * 2

    result = dr.switch(idx, [cheap_func, expensive_func], arg)
    dr.allclose(result, [0, 2, 2, 6])


# Test that reorder is valid inside loops
@pytest.test_arrays('float32, jit, shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@dr.syntax(recursive=True)
def test01_reorder_loop(t, mode):
    UInt32 = dr.uint32_array_t(t)
    N = 4

    idx = dr.arange(UInt32, N)
    arg = dr.arange(t, N)
    dr.make_opaque(arg)

    def f(arg):
        i = UInt32(0)
        while dr.hint(i < 32, mode=mode):
            j = UInt32(0)

            # Arbitrary aritmetic
            while dr.hint(j < 10, mode=mode):
                arg = arg + j
                j += 1

            i = dr.reorder_threads(idx % 32, 2, i)
            i += 1

            # Early exit one thread per iteraion
            i = dr.select(idx % 32 < i, 100, i)

        return arg

    result = f(arg)
    dr.allclose(result, [45, 91, 137, 183])
