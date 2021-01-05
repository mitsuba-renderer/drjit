import enoki as ek
import pytest

def get_class(name):
    """Resolve a package+class name into the corresponding type"""
    if 'cuda' in name:
        if not ek.has_backend(ek.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in name:
        if not ek.has_backend(ek.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')

    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    ek.jit_set_flag(ek.JitFlag.RecordLoops, True)

    return value

def setup_function(function):
    ek.jit_set_flag(ek.JitFlag.RecordLoops, True)

def teardown_function(function):
    ek.disable_flag(ek.JitFlag.RecordLoops)

pkgs = ["enoki.cuda", "enoki.cuda.ad",
        "enoki.llvm", "enoki.llvm.ad"]

@pytest.mark.parametrize("pkg", pkgs)
def test01_ctr(pkg):
    p = get_class(pkg)

    i = ek.arange(p.Int, 0, 10)

    loop = p.Loop(i)
    while loop.cond(i < 5):
        i += 1

    assert i == p.Int(5, 5, 5, 5, 5, 5, 6, 7, 8, 9)


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", pkgs)
def test02_multiple_values(pkg, variant):
    p = get_class(pkg)

    i = ek.arange(p.Int, 0, 10)
    v = ek.zero(p.Array3f, 10)

    if variant == 1:
        v.y = p.Float(0)

    loop = p.Loop(i, v)
    while loop.cond(i < 5):
        i.assign(i + 1)
        f = p.Float(i)
        v.x += f
        v.y += 2*f
        v.z += 4*f

    if variant == 0:
        ek.eval(i, v)
    else:
        ek.eval(i)
        ek.eval(v.x)
        ek.eval(v.y)
        ek.eval(v.z)

    assert i == p.Int(5, 5, 5, 5, 5, 5, 6, 7, 8, 9)
    assert v.y == p.Int(30, 28, 24, 18, 10, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("pkg", pkgs)
def test03_failures(pkg):
    p = get_class(pkg)

    i = p.Int()
    v = ek.zero(p.Array3f, 10)

    with pytest.raises(ek.Exception) as e:
        p.Loop(i, v)

    assert 'Variables provided to enoki::Loop() must be fully initialized' in str(e.value)

    if 'ad' in pkg:
        i = p.Int(0)
        ek.enable_grad(v)
        with pytest.raises(ek.Exception) as e:
            p.Loop(i, v)
        assert 'Symbolic loop encountered a differentiable array with enabled gradients! This is not supported.' in str(e.value)


@pytest.mark.parametrize("pkg", pkgs)
def test04_side_effect(pkg):
    p = get_class(pkg)

    i = ek.zero(p.Int, 10)
    j = ek.zero(p.Int, 10)
    buf = ek.zero(p.Float, 10)

    loop = p.Loop(i, j)
    while loop.cond(i < 10):
        j += i
        i += 1
        ek.scatter_add(target=buf, value=p.Float(i), index=0)

    ek.eval(i, j)
    assert i == p.Int([10]*10)
    assert buf == p.Float(550, *([0]*9))
    assert j == p.Int([45]*10)


@pytest.mark.parametrize("pkg", pkgs)
def test05_side_effect_noloop(pkg):
    p = get_class(pkg)

    i = ek.zero(p.Int, 10)
    j = ek.zero(p.Int, 10)
    buf = ek.zero(p.Float, 10)
    ek.disable_flag(ek.JitFlag.RecordLoops)

    loop = p.Loop(i, j)
    while loop.cond(i < 10):
        j += i
        i += 1
        ek.scatter_add(target=buf, value=p.Float(i), index=0, mask=loop.mask())

    assert i == p.Int([10]*10)
    assert buf == p.Float(550, *([0]*9))
    assert j == p.Int([45]*10)


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("pkg", pkgs)
def test06_test_collatz(pkg, variant):
    p = get_class(pkg)

    def collatz(value: p.Int):
        counter = p.Int(0)
        loop = p.Loop(value, counter)
        while (loop.cond(ek.neq(value, 1))):
            is_even = ek.eq(value & 1, 0)
            value.assign(ek.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return value, counter

    value, ctr = collatz(ek.arange(p.Int, 1, 11))
    if variant == 0:
        ek.eval(value, ctr)
    elif variant == 1:
        ek.eval(value)
        ek.eval(ctr)
    elif variant == 2:
        ek.eval(ctr)
        ek.eval(value)

    assert value == p.Int([1]*10)
    assert ctr == p.Int([0,1,7,2,5,8,16,3,19,6])

@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", ["enoki.cuda",
                                 "enoki.cuda.ad",
                                 "enoki.llvm",
                                 "enoki.llvm.ad"])
def test07_loop_nest(pkg, variant):
    p = get_class(pkg)

    def collatz(value: p.Int):
        counter = p.Int(0)
        loop = p.Loop(value, counter)
        while (loop.cond(ek.neq(value, 1))):
            is_even = ek.eq(value & 1, 0)
            value.assign(ek.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return counter

    i = p.Int(1)
    buf = ek.full(p.Int, 1000, 16)
    ek.eval(buf)

    if variant == 0:
        loop_1 = p.Loop(i)
        while loop_1.cond(i <= 10):
            ek.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1
    else:
        for i in range(1, 11):
            ek.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1

    assert buf == p.Int(0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000, 1000, 1000, 1000, 1000, 1000)
