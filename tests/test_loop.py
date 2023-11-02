import drjit as dr
import pytest


@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.function
def test01_simple(t):
    i = t([0, 5])
    z = t(0)
    while i < 10:
        i += 1
        o = 5
        z = o + i
    assert dr.all(z == t(15, 15))
