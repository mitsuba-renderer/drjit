import drjit as dr
import pytest
import gc

def get_class(name):
    """Resolve a package+class name into the corresponding type"""
    if 'cuda' in name:
        if not dr.has_backend(dr.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    elif 'llvm' in name:
        if not dr.has_backend(dr.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')

    name = name.split('.')
    value = __import__(".".join(name[:-1]))
    for item in name[1:]:
        value = getattr(value, item)
    dr.set_flag(dr.JitFlag.LoopRecord, True)

    return value

def setup_function(function):
    dr.set_flag(dr.JitFlag.LoopRecord, True)

def teardown_function(function):
    dr.set_flag(dr.JitFlag.LoopRecord, False)

pkgs = ["drjit.cuda", "drjit.cuda.ad",
        "drjit.llvm", "drjit.llvm.ad"]

pkgs_ad = ["drjit.cuda.ad",
           "drjit.llvm.ad"]

@pytest.mark.parametrize("pkg", pkgs)
def test01_ctr(pkg):
    p = get_class(pkg)

    i = dr.arange(p.Int, 0, 10)

    loop = p.Loop("MyLoop", lambda: i)
    while loop(i < 5):
        i = i + 1

    assert i == p.Int(5, 5, 5, 5, 5, 5, 6, 7, 8, 9)


@pytest.mark.parametrize("pkg", pkgs)
def test02_record_loop(pkg):
    p = get_class(pkg)

    for i in range(3):
        dr.set_flag(dr.JitFlag.LoopRecord, not i == 0)
        dr.set_flag(dr.JitFlag.LoopOptimize, i == 2)

        for j in range(2):
            x = dr.arange(p.Int, 0, 10)
            y = dr.zeros(p.Float, 1)
            z = p.Float(1)

            loop = p.Loop("MyLoop", lambda: (x, y, z))
            while loop(x < 5):
                y += p.Float(x)
                x += 1
                z = z + 1

            if j == 0:
                dr.schedule(x, y, z)

            assert z == p.Int(6, 5, 4, 3, 2, 1, 1, 1, 1, 1)
            assert y == p.Int(10, 10, 9, 7, 4, 0, 0, 0, 0, 0)
            assert x == p.Int(5, 5, 5, 5, 5, 5, 6, 7, 8, 9)


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", pkgs)
def test03_multiple_values(pkg, variant):
    p = get_class(pkg)

    i = dr.arange(p.Int, 0, 10)
    v = dr.zeros(p.Array3f, 10)

    if variant == 1:
        v.y = p.Float(0)

    loop = p.Loop("MyLoop", lambda: (i, v))
    while loop(i < 5):
        i.assign(i + 1)
        f = p.Float(i)
        v.x += f
        v.y += 2*f
        v.z += 4*f

    if variant == 0:
        dr.eval(i, v)
    else:
        dr.eval(i)
        dr.eval(v.x)
        dr.eval(v.y)
        dr.eval(v.z)

    assert i == p.Int(5, 5, 5, 5, 5, 5, 6, 7, 8, 9)
    assert v.y == p.Int(30, 28, 24, 18, 10, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("pkg", pkgs)
def test04_side_effect(pkg):
    p = get_class(pkg)

    i = dr.zeros(p.Int, 10)
    j = dr.zeros(p.Int, 10)
    buf = dr.zeros(p.Float, 10)

    loop = p.Loop("MyLoop", lambda: (i, j))
    while loop(i < 10):
        j += i
        i += 1
        dr.scatter_reduce(op=dr.ReduceOp.Add, target=buf, value=p.Float(i), index=0)

    dr.eval(i, j)
    assert i == p.Int([10]*10)
    assert buf == p.Float(550, *([0]*9))
    assert j == p.Int([45]*10)


@pytest.mark.parametrize("pkg", pkgs)
def test05_side_effect_noloop(pkg):
    p = get_class(pkg)

    i = dr.zeros(p.Int, 10)
    j = dr.zeros(p.Int, 10)
    buf = dr.zeros(p.Float, 10)
    dr.set_flag(dr.JitFlag.LoopRecord, False)

    loop = p.Loop("MyLoop", lambda: (i, j))
    while loop(i < 10):
        j += i
        i += 1
        dr.scatter_reduce(op=dr.ReduceOp.Add, target=buf, value=p.Float(i), index=0)

    assert i == p.Int([10]*10)
    assert buf == p.Float(550, *([0]*9))
    assert j == p.Int([45]*10)


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("pkg", pkgs)
def test06_test_collatz(pkg, variant):
    p = get_class(pkg)

    def collatz(value: p.Int):
        counter = p.Int(0)
        loop = p.Loop("collatz", lambda: (value, counter))
        while (loop(dr.neq(value, 1))):
            is_even = dr.eq(value & 1, 0)
            value.assign(dr.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return value, counter

    value, ctr = collatz(dr.arange(p.Int, 1, 11))
    if variant == 0:
        dr.eval(value, ctr)
    elif variant == 1:
        dr.eval(value)
        dr.eval(ctr)
    elif variant == 2:
        dr.eval(ctr)
        dr.eval(value)

    assert value == p.Int([1]*10)
    assert ctr == p.Int([0,1,7,2,5,8,16,3,19,6])

@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", ["drjit.cuda",
                                 "drjit.cuda.ad",
                                 "drjit.llvm",
                                 "drjit.llvm.ad"])
def test07_loop_nest(pkg, variant):
    p = get_class(pkg)

    def collatz(value: p.Int):
        counter = p.Int(0)
        loop = p.Loop("Nested", lambda: (value, counter))
        while (loop(dr.neq(value, 1))):
            is_even = dr.eq(value & 1, 0)
            value.assign(dr.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return counter

    i = p.Int(1)
    buf = dr.full(p.Int, 1000, 16)
    dr.eval(buf)

    if variant == 0:
        loop_1 = p.Loop("MyLoop", lambda: i)
        while loop_1(i <= 10):
            dr.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1
    else:
        for i in range(1, 11):
            dr.scatter(buf, collatz(p.Int(i)), i - 1)
            i += 1

    assert buf == p.Int(0, 1, 7, 2, 5, 8, 16, 3, 19, 6, 1000, 1000, 1000, 1000, 1000, 1000)



@pytest.mark.parametrize("pkg", pkgs)
def test08_loop_simplification(pkg):
    # Test a regression where most loop variables are optimized away
    p = get_class(pkg)
    res = dr.zeros(p.Float, 10)
    active = p.Bool(True)
    depth = p.UInt32(1000)

    loop = p.Loop("TestLoop")
    loop.put(lambda: (active, depth, res))
    loop.init()
    while loop(active):
        res += dr.arange(p.Float, 10)
        depth -= 1
        active &= (depth > 0)
    del loop
    del active
    del depth
    assert res == dr.arange(p.Float, 10) * 1000


@pytest.mark.parametrize("pkg", pkgs)
def test09_failure_invalid_loop_arg(pkg):
    p = get_class(pkg)
    i = p.Int(1)
    with pytest.raises(RuntimeError) as e:
        p.Loop("MyLoop", i)
    assert 'expected a lambda function' in str(e.value)

    i = p.Int(1)
    with pytest.raises(TypeError) as e:
        l = p.Loop("MyLoop")
        l.put(i)
    assert 'incompatible function arguments' in str(e.value)
    del l


@pytest.mark.parametrize("pkg", pkgs_ad)
def test10_failure_invalid_type(pkg):
    p = get_class(pkg)
    i = 1
    with pytest.raises(dr.Exception) as e:
        p.Loop("MyLoop", lambda: i)
    assert 'you must use Dr.Jit arrays/structs' in str(e.value)


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", pkgs)
def test11_failure_unitialized(pkg, variant):
    p = get_class(pkg)
    i = p.Int(0)
    j = p.Int() if variant == 0 else p.Int(1)
    l = None
    with pytest.raises(dr.Exception) as e:
        l = p.Loop("MyLoop", lambda: (i, j))
        while l(i < 10):
            j = p.Int() if variant == 1 else i + 1
            i += 1

    assert 'is uninitialized!' in str(e.value)
    del l


@pytest.mark.parametrize("pkg", pkgs)
def test12_failure_changing_type(pkg):
    p = get_class(pkg)
    i = p.Int(0)
    with pytest.raises(dr.Exception) as e:
        l = p.Loop("MyLoop", lambda: i)
        while l(i < 10):
            i = p.Float(0)
    assert 'the type of loop state variables must remain the same throughout the loop' in str(e.value)
    del l


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("pkg", pkgs_ad)
def test13_failures_ad(pkg, variant):
    p = get_class(pkg)
    i = p.Int(0)
    v = p.Array3f(1,2,2)
    l = None
    with pytest.raises(dr.Exception) as e:
        if variant == 0:
            dr.enable_grad(v)
        l = p.Loop("MyLoop", lambda: (i, v))
        while l(i < 10):
            i = i + 1
            if variant == 1:
                dr.enable_grad(v)
    print(str(e.value))
    assert 'one of the supplied loop state variables of type Float is attached to the AD graph' in str(e.value)
    del l


@pytest.mark.parametrize("pkg", pkgs)
def test14_failure_state_leak(pkg):
    p = get_class(pkg)
    i, j = p.Int(0), p.Int(0)

    l = p.Loop("MyLoop", lambda: i)
    while l(i < 10):
        i = i + 1
        j += i

    with pytest.raises(RuntimeError) as e:
        j[0]
    assert 'placeholder variables are used to record computation symbolically' in str(e.value)
    del l


@pytest.mark.parametrize("pkg", pkgs)
def test15_scalar_side_effect(pkg):
    # Ensure that a scalar side effect takes place multiple times if the loop processes larger arrays
    p = get_class(pkg)

    for i in range(1):
        dr.set_flag(dr.JitFlag.LoopRecord, i == 1)

        active = p.Bool(True)
        unrelated = dr.arange(p.UInt32, 123)

        loop = p.Loop("Test", lambda: (active, unrelated))
        target = dr.zeros(p.UInt32)

        while loop(active):
            dr.scatter_reduce(dr.ReduceOp.Add, target, 1, 0)
            active &= False

        assert target[0] == 123
