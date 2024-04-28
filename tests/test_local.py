import drjit as dr
from dataclasses import dataclass
import pytest

@pytest.test_arrays('jit,-tensor')
@pytest.mark.parametrize('eval', [True, False])
def test01_simple(t, eval):
    if dr.size_v(t) == 0:
        with pytest.raises(TypeError, match="type does not contain any Jit-tracked arrays"):
            dr.alloc_local(t, 10)
        return

    if dr.size_v(t) == dr.Dynamic:
        s = dr.alloc_local(t, 10, dr.zeros(t, 3))
    else:
        s = dr.alloc_local(t, 10, dr.zeros(t))
    v = s[0]
    assert len(s) == 10
    assert v.state == dr.VarState.Literal
    assert dr.all(v == t(0), axis=None)

    s[0] = t(1)
    if eval:
        dr.eval(s)

    assert dr.all((s[0] == t(1)) & (s[1] == t(0)), axis=None)


@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test02_fill_in_loop_then_read(t):
    s = dr.alloc_local(t, 10)
    i = t(0)

    while i < 10:
        i += 1
        s[i] = t(i)

    assert s[3] == 3


@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test03_bubble_sort(t):
    import sys
    n = 32
    s = dr.alloc_local(t, n)
    rng = sys.modules[t.__module__].PCG32(10000)

    i = t(0)
    while i < n:
        s[i] = rng.next_uint32()
        i += 1


    i = t(0)
    while i < n-1:
        j = t(0)
        while j < n-i-1:
            s0, s1 = s[j], s[j+1]
            if s0 > s1:
                s0, s1 = s1, s0
            s[j], s[j+1] = s0, s1
            j+= 1
        i += 1

    result = [s[j] for j in range(n)]
    dr.eval(result)
    for i in range(len(result)-1):
        assert dr.all(result[i] <= result[i + 1])


@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test04_conditional(t):
    s = dr.alloc_local(t, 1, value = dr.zeros(t, 2))
    i = t(0, 1)

    if i > 0:
        s[0] = 10
    else:
        s[0] = 11

    for i in range(2): # evaluate twice (intentional)
        assert dr.all(s[0] == [11, 10])


# No derivatives
# Test copy logic
# PyTree allocation
# Use in reverse-mode while loop
