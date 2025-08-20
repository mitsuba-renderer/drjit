import drjit as dr
from dataclasses import dataclass
import pytest
import re

@pytest.test_arrays('jit,-tensor')
@pytest.mark.parametrize('eval', [True, False])
def test01_simple(t, eval):
    if dr.size_v(t) == 0:
        with pytest.raises(TypeError, match="type does not contain any Jit-tracked arrays"):
            dr.alloc_local(t, 2)
        return

    if dr.size_v(t) == dr.Dynamic:
        s = dr.alloc_local(t, 2, dr.zeros(t, 3))
    else:
        s = dr.alloc_local(t, 2, dr.zeros(t))
    v = s[0]
    assert len(s) == 2
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
        s[i] = t(i)
        i += 1

    assert s[3] == 3


@pytest.test_arrays('jit,uint32,shape=(*)')
@pytest.mark.parametrize('variant', [0,1])
@pytest.mark.parametrize('symbolic_loop', [True, False])
@pytest.mark.parametrize('symbolic_cond', [True, False])
@dr.syntax
def test03_bubble_sort(t, variant, symbolic_loop, symbolic_cond):
    import sys
    n = 32
    s = dr.alloc_local(t, n)
    Bool = dr.mask_t(t)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic_loop):
        with dr.scoped_set_flag(dr.JitFlag.SymbolicConditionals, symbolic_cond):
            rng = sys.modules[t.__module__].PCG32()
            rng.state = 1234 + dr.arange(t, 10000)
            rng.inc = t(1)
            rng.next_uint32()
            i = t(0)
            while i < n:
                s[i] = rng.next_uint32()
                i += 1


            i = t(0)
            cont=Bool(True)
            while (i < n-1) & cont:
                j = t(0)
                cont=Bool(variant==0)
                while j < n-i-1:
                    if dr.hint(variant == 0, mode='scalar'):
                        s0, s1 = s[j], s[j+1]
                        if s0 > s1:
                            s0, s1 = s1, s0
                        s[j], s[j+1] = s0, s1
                    else:
                        if s[j] > s[j+1]:
                            s[j], s[j+1] = s[j+1], s[j]
                            cont = Bool(True)
                    j+= 1
                i += 1

    result = [s[j] for j in range(n)]
    dr.eval(result)
    for i in range(len(result)-1):
        assert dr.all(result[i] <= result[i + 1])
    q = [a[0] for a in result]
    assert q == [25948406, 86800510, 163991264, 724914361, 798920662, 848337899, 1331190098, 1441102920, 1445257284, 1461834408, 1497151495, 1547771419, 1603554384, 1691880696, 1797163244, 1936973067, 2311034952, 2444167623, 2607360471, 2819391842, 2948902546, 2967546311, 3059896137, 3153572993, 3235986370, 3262142078, 3348457850, 3408296642, 3467369332, 3614138351, 3631189001, 3663081236]


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


@pytest.test_arrays('diff,float32,shape=(*)')
@dr.syntax
def test05_nodiff(t):
    s = dr.alloc_local(t, 1)
    x = t(0)
    dr.enable_grad(x)
    with pytest.raises(RuntimeError, match=re.escape(r"Local memory writes are not differentiable. You must use 'drjit.detach()' to disable gradient tracking of the written value.")):
        s[0] = x


@pytest.test_arrays('jit,-diff,float32,shape=(*)')
@dr.syntax
def test06_copy(t):
    s0 = dr.alloc_local(t, 2)
    s0[0] = 123
    s0[1] = 456

    s1 = dr.Local(s0)
    s1[0] += 100
    s0[0] += 1000


    assert s0[0] == 1123
    assert s0[1] == 456
    assert s1[0] == 223
    assert s1[1] == 456


@pytest.test_arrays('jit,-diff,float32,shape=(*)')
@dr.syntax
def test07_pytree(t):
    from dataclasses import dataclass

    @dataclass
    class XY:
        x: t
        y: t

    result = dr.alloc_local(XY, size=2, value=dr.zeros(XY))
    result[0] = XY(t(3),t(4))
    result[1] = XY(t(5),t(6))
    assert "XY(x=[3], y=[4])" in str(result[0])
    assert "XY(x=[5], y=[6])" in str(result[1])


@pytest.test_arrays('jit,-diff,uint32,shape=(*)')
@dr.syntax
def test08_oob_read(t, capsys):
    i, r = t(0), t(0)
    v = dr.alloc_local(t, size=10, value=t(0))
    with pytest.raises(RuntimeError, match=r"out of bounds read \(source size=10, offset=100\)"):
        v[100]
    with pytest.raises(RuntimeError, match=r"out of bounds write \(target size=10, offset=100\)"):
        v[100] = 0
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        while i < 100:
            r += v[i]
            i += 1
        assert r == 0

    transcript = capsys.readouterr().err
    assert 'drjit.Local.read(): out-of-bounds read from position 99 in an array of size 10' in transcript


@pytest.test_arrays('jit,-diff,uint32,shape=(*)')
@dr.syntax
def test09_oob_write(t, capsys):
    i, r = t(0), t(0)
    v = dr.alloc_local(t, size=10, value=t(0))
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        while i < 100:
            v[i] = i
            i += 1
    print(v[0])

    transcript = capsys.readouterr().err
    assert 'drjit.Local.write(): out-of-bounds write to position 99 in an array of size 10' in transcript


@pytest.test_arrays('jit,-diff,uint32,shape=(*)')
@dr.syntax
def test10_loop_read_then_write(t, capsys):
    local = dr.alloc_local(t, size=8, value=t(0))
    size = t(0)
    val = t(0)

    while size < 5:
        val += local[size]
        size += 1
        local[size] = t(4)

    assert val[0] == 16

@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test11_write_mask_simple(t):
    Bool = dr.mask_t(t)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    mask = True
    value = t(1, 1, 1)

    s.write(value, t(0), active=mask) 

    assert dr.all(s[0] == [1, 1, 1])

    mask = Bool(True, False, True)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    s.write(value, t(0), active=mask) 

    assert dr.all(s[0] == [1, 0, 1])

@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test12_write_mask_conditional(t):
    Bool = dr.mask_t(t)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    mask = Bool(True, False, True)

    i = t(0, 1, 1)

    if i > 0:
        s.write(t(1), t(0), active=mask) 

    assert dr.all(s[0] == [0, 0, 1])

@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test13_read_mask_simple(t):
    Bool = dr.mask_t(t)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    s[0] = 1
    mask = True

    x = dr.zeros(t, 3)
    x += s.read(t(0), active=mask)

    assert dr.all(x == [1, 1, 1])

    mask = Bool(True, False, True)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    s[0] = 1
    x = dr.zeros(t, 3)
    x += s.read(t(0), mask)

    assert dr.all(x == [1, 0, 1])

@pytest.test_arrays('jit,uint32,shape=(*)')
@dr.syntax
def test14_read_mask_conditional(t):
    Bool = dr.mask_t(t)
    s = dr.alloc_local(t, 1, value = dr.zeros(t))
    s[0] = 1
    mask = Bool(True, False, True)

    i = t(0, 1, 1)

    x = dr.zeros(t, 3)

    if i > 0:
        x += s.read(t(0), active=mask) 

    assert dr.all(x == [0, 0, 1])
