import drjit as dr
import pytest
from dataclasses import dataclass
import sys

dr.set_log_level(dr.LogLevel.Trace)
# dr.set_flag(dr.JitFlag.KernelFreezing, True)
dr.set_flag(dr.JitFlag.ReuseIndices, False)


def get_single_entry(x):
    tp = type(x)
    result = x
    shape = dr.shape(x)
    if len(shape) == 2:
        result = result[shape[0] - 1]
    if len(shape) == 3:
        result = result[shape[0] - 1][shape[1] - 1]
    return result


@pytest.test_arrays("uint32, jit, shape=(*)")
def test01_basic(t):
    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o0 = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    o0 = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)
    
@pytest.test_arrays("uint32, jit, shape=(*)")
def test02_flush_kernel_cache(t):
    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o0 = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)

    dr.flush_kernel_cache()

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    # Flushing the kernel cache should force a re-trace
    o0 = func(i0, i1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test02_output_tuple(t):
    @dr.freeze
    def func(x, y):
        return (x + y, x * y)

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    (o0, o1) = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    (o0, o1) = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test03_output_list(t):
    @dr.freeze
    def func(x, y):
        return [x + y, x * y]

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    [o0, o1] = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    [o0, o1] = func(i0, i1)
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test04_output_dict(t):
    @dr.freeze
    def func(x, y):
        return {"add": x + y, "mul": x * y}

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o = func(i0, i1)
    o0 = o["add"]
    o1 = o["mul"]
    assert dr.all(t(2, 2, 2) == o0)
    assert dr.all(t(0, 1, 0) == o1)

    i0 = t(1, 2, 3)
    i1 = t(3, 2, 1)

    o = func(i0, i1)
    o0 = o["add"]
    o1 = o["mul"]
    assert dr.all(t(4, 4, 4) == o0)
    assert dr.all(t(3, 4, 3) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test05_nested_tuple(t):
    @dr.freeze
    def func(x):
        return (x + 1, x + 2, (x + 3, x + 4))

    i0 = t(0, 1, 2)

    (o0, o1, (o2, o3)) = func(i0)
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)
    assert dr.all(t(3, 4, 5) == o2)
    assert dr.all(t(4, 5, 6) == o3)

    i0 = t(1, 2, 3)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test06_drjit_struct(t):
    class Point:
        x: t
        y: t
        DRJIT_STRUCT = {"x": t, "y": t}

    @dr.freeze
    def func(x):
        p = Point()
        p.x = x + 1
        p.y = x + 2
        return p

    i0 = t(0, 1, 2)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)

    i0 = t(1, 2, 3)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(2, 3, 4) == o0)
    assert dr.all(t(3, 4, 5) == o1)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test06_dataclass(t):
    @dataclass
    class Point:
        x: t
        y: t

    @dr.freeze
    def func(x):
        p = Point(x + 1, x + 2)
        return p

    i0 = t(0, 1, 2)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(1, 2, 3) == o0)
    assert dr.all(t(2, 3, 4) == o1)

    i0 = t(1, 2, 3)

    o = func(i0)
    o0 = o.x
    o1 = o.y
    assert dr.all(t(2, 3, 4) == o0)
    assert dr.all(t(3, 4, 5) == o1)


@pytest.test_arrays("float32, jit, shape=(*)")
def test07_traverse_cb(t):
    pkg = get_pkg(t)
    Sampler = pkg.Sampler

    def func(sampler):
        return sampler.next()

    frozen = dr.freeze(func)

    sampler_frozen = Sampler(10)
    sampler_func = Sampler(10)

    result1_frozen = frozen(sampler_frozen)
    result1_func = func(sampler_func)
    assert dr.allclose(result1_frozen, result1_func)

    sampler_frozen = Sampler(10)
    sampler_func = Sampler(10)

    result2_frozen = frozen(sampler_frozen)
    result2_func = func(sampler_func)
    assert dr.allclose(result2_frozen, result2_func)

    assert frozen.n_recordings == 1

    result3_frozen = frozen(sampler_frozen)
    result3_func = func(sampler_func)
    assert dr.allclose(result3_func, result3_frozen)

    assert frozen.n_recordings == 1


@pytest.test_arrays("uint32, jit, shape=(*)")
def test07_scatter(t):
    @dr.freeze
    def func(x):
        dr.scatter(x, 0, dr.arange(t, 3))

    x = t(0, 1, 2)
    func(x)

    x = t(0, 1, 2)
    y = x + 1
    z = x
    w = t(x)

    func(x)

    assert dr.all(t(0, 0, 0) == x)
    assert dr.all(t(1, 2, 3) == y)
    assert dr.all(t(0, 0, 0) == z)
    assert dr.all(t(0, 1, 2) == w)


@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
def test08_optimization(t):
    @dr.freeze
    def func(state, ref):
        for k, x in state.items():
            dr.enable_grad(x)
            loss = dr.mean(dr.square(x - ref))

            dr.backward(loss)

            grad = dr.grad(x)
            dr.disable_grad(x)
            state[k] = x - grad

    state = {"x": t(0, 0, 0, 0)}

    ref = t(1, 1, 1, 1)
    func(state, ref)
    assert dr.allclose(t(0.5, 0.5, 0.5, 0.5), state["x"])

    state = {"x": t(0, 0, 0, 0)}
    ref = t(1, 1, 1, 1)
    func(state, ref)

    assert dr.allclose(t(0.5, 0.5, 0.5, 0.5), state["x"])


@pytest.test_arrays("uint32, jit, shape=(*)")
def test09_resized(t):
    @dr.freeze
    def func(x, y):
        return x + y

    i0 = t(0, 1, 2)
    i1 = t(2, 1, 0)

    o0 = func(i0, i1)
    assert dr.all(t(2, 2, 2) == o0)

    i0 = dr.arange(t, 64) + dr.opaque(t, 0)
    i1 = dr.arange(t, 64) + dr.opaque(t, 0)
    r0 = i0 + i1
    dr.eval(i0, i1, r0)

    o0 = func(i0, i1)
    assert dr.all(r0 == o0)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test10_changed_input_dict(t):
    @dr.freeze
    def func(d: dict):
        d["y"] = d["x"] + 1

    x = t(0, 1, 2)
    d = {"x": x}

    func(d)
    assert dr.all(t(1, 2, 3) == d["y"])

    x = t(1, 2, 3)
    d = {"x": x}

    func(d)
    assert dr.all(t(2, 3, 4) == d["y"])


@pytest.test_arrays("uint32, jit, shape=(*)")
def test11_changed_input_dataclass(t):
    @dataclass
    class Point:
        x: t

    @dr.freeze
    def func(p: Point):
        p.x = p.x + 1

    p = Point(x=t(0, 1, 2))

    func(p)
    assert dr.all(t(1, 2, 3) == p.x)

    p = Point(x=t(1, 2, 3))

    func(p)
    assert dr.all(t(2, 3, 4) == p.x)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test12_kwargs(t):
    @dr.freeze
    def func(x=t(0, 1, 2)):
        return x + 1

    y = func(x=t(0, 1, 2))
    assert dr.all(t(1, 2, 3) == y)

    y = func(x=t(1, 2, 3))
    assert dr.all(t(2, 3, 4) == y)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test13_opaque(t):
    @dr.freeze
    def func(x, y):
        return x + y

    x = t(0, 1, 2)
    dr.set_label(x, "x")
    y = dr.opaque(t, 1)
    dr.set_label(y, "y")
    z = func(x, y)
    assert dr.all(t(1, 2, 3) == z)

    x = t(1, 2, 3)
    y = t(1, 2, 3)
    z = func(x, y)
    assert dr.all(t(2, 4, 6) == z)

    assert func.n_recordings == 2


@pytest.test_arrays("float32, jit, -is_diff, shape=(*)")
def test14_performance(t):
    import time

    n = 1024
    n_iter = 1_000
    n_iter_warmeup = 10

    def func(x, y):
        z = 0.5
        result = dr.fma(dr.square(x), y, z)
        result = dr.sqrt(dr.abs(result) + dr.power(result, 10))
        result = dr.log(1 + result)
        return result

    frozen = dr.freeze(func)

    for name, fn in [("normal", func), ("frozen", frozen)]:
        x = dr.arange(t, n)  # + dr.opaque(t, i)
        y = dr.arange(t, n)  # + dr.opaque(t, i)
        dr.eval(x, y)
        for i in range(n_iter + n_iter_warmeup):
            if i == n_iter_warmeup:
                t0 = time.time()

            result = fn(x, y)

            dr.eval(result)

        dr.sync_thread()
        elapsed = time.time() - t0
        print(f"{name}: average {1000 * elapsed / n_iter:.3f} ms / iteration")


@pytest.test_arrays("uint32, jit, shape=(*)")
def test15_aliasing(t):
    @dr.freeze
    def func(x, y):
        return x + y

    print("aliased:")
    x = t(0, 1, 2)
    y = x
    z = func(x, y)
    assert dr.all(t(0, 2, 4) == z)

    print("aliased:")
    x = t(1, 2, 3)
    y = x
    z = func(x, y)
    assert dr.all(t(2, 4, 6) == z)

    print("non-aliased:")
    x = t(1, 2, 3)
    y = t(2, 3, 4)
    z = func(x, y)
    assert dr.all(t(3, 5, 7) == z)
    assert func.n_recordings == 2


@pytest.test_arrays("uint32, jit, shape=(*)")
def test16_non_jit_types(t):
    @dr.freeze
    def func(x, y):
        return x + y

    x = t(1, 2, 3)
    y = 1

    # with pytest.raises(RuntimeError):
    z = func(x, y)


@pytest.test_arrays("uint32, jit, cuda, -is_diff, shape=(*)")
def test17_literal(t):
    # dr.set_log_level(dr.LogLevel.Trace)
    # dr.set_flag(dr.JitFlag.KernelHistory, True)

    @dr.freeze
    def func(x, y):
        z = x + y
        w = t(1)
        return z, w

    # Literals
    x = t(0, 1, 2)
    dr.set_label(x, "x")
    y = t(1)
    dr.set_label(y, "y")
    z, w = func(x, y)
    assert dr.all(z == t(1, 2, 3))
    assert dr.all(w == t(1))

    x = t(0, 1, 2)
    y = t(1)
    z, w = func(x, y)
    assert dr.all(z == t(1, 2, 3))
    assert dr.all(w == t(1))

    assert func.n_recordings == 1

    x = t(0, 1, 2)
    y = t(2)
    z = func(x, y)
    print(f"{y.index=}")
    assert func.n_recordings == 1


@pytest.test_arrays("uint32, jit, shape=(*)")
def test18_pointers(t):
    UInt32 = dr.uint32_array_t(t)

    @dr.freeze
    def func(x):
        idx = dr.arange(UInt32, 0, dr.width(x), 3)

        return dr.gather(t, x, idx)

    y = func(t(0, 1, 2, 3, 4, 5, 6))

    print(y)

@pytest.test_arrays("uint32, jit, shape=(*)")
def test19_gather_memcpy(t):
    """
    The gather operation might be elided in favor of a memcpy 
    if the index is a literal of size 1.
    The source of the memcpy is however not known to the recording
    mechansim as it might index into the source array.
    """

    def func(x, idx: int):
        idx = t(idx)
        return dr.gather(t, x, idx)

    frozen = dr.freeze(func)

    for i in range(3):
        x = dr.arange(t, i, 3 + i)
        dr.make_opaque(x)
        ref = func(x, 2)
        result = frozen(x, 2)

        print(f"{result=}")

        assert dr.all(ref == result)

    assert frozen.n_recordings == 1


def get_pkg(t):
    with dr.detail.scoped_rtld_deepbind():
        m = pytest.importorskip("call_ext")
    backend = dr.backend_v(t)
    if backend == dr.JitBackend.LLVM:
        return m.llvm
    elif backend == dr.JitBackend.CUDA:
        return m.cuda


@pytest.mark.parametrize("symbolic", [True])
@pytest.test_arrays("float32, jit, -is_diff, shape=(*)")
def test19_vcall(t, symbolic):
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = A(), B()

    c = BasePtr(a, a, None, a, a)

    xi = t(1, 2, 8, 3, 4)
    yi = t(5, 6, 8, 7, 8)

    @dr.freeze
    def func(c, xi, yi):
        return c.f(xi, yi)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = func(c, xi, yi)

    assert dr.all(xo == t(10, 12, 0, 14, 16))
    assert dr.all(yo == t(-1, -2, 0, -3, -4))

    c = BasePtr(a, a, None, b, b)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        xo, yo = func(c, xi, yi)

    assert func.n_recordings == 1

    assert dr.all(xo == t(10, 12, 0, 21, 24))
    assert dr.all(yo == t(-1, -2, 0, 3, 4))


@pytest.mark.parametrize("symbolic", [True])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("opaque", [True, False])
@pytest.test_arrays("float32, -is_diff, jit, shape=(*)")
def test20_vcall_optimize(t, symbolic, optimize, opaque):
    print(f"{symbolic=}")
    print(f"{optimize=}")
    print(f"{opaque=}")
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = B(), B()

    dr.set_label(A.opaque, "A.opaque")
    dr.set_label(B.opaque, "B.opaque")

    a.value = t(2)
    b.value = t(3)

    print(f"{a.value.index=}")

    if opaque:
        dr.make_opaque(a.value, b.value)

    c = BasePtr(a, a, None, a, a)
    dr.set_label(c, "c")

    x = t(1, 2, 8, 3, 4)
    dr.set_label(x, "x")

    def func(c, xi):
        return c.g(xi)

    frozen = dr.freeze(func)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = frozen(c, x)

    assert dr.all(xo == func(c, x))

    a.value = t(3)
    b.value = t(2)

    if opaque:
        dr.make_opaque(a.value, b.value)

    c = BasePtr(a, a, None, b, b)
    dr.set_label(c, "c")

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = frozen(c, x)

    assert frozen.n_recordings == 1
    # if opaque:
    #     assert frozen.n_recordings == 1
    # else:
    #     assert frozen.n_recordings == 2

    assert dr.all(xo == func(c, x))


@pytest.mark.parametrize("symbolic", [True])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("opaque", [True, False])
@pytest.test_arrays("float32, -is_diff, jit, shape=(*)")
def test21_multiple_vcalls(t, symbolic, optimize, opaque):
    print(f"{symbolic=}")
    print(f"{optimize=}")
    print(f"{opaque=}")
    pkg = get_pkg(t)

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    Mask = dr.mask_t(t)
    a, b = B(), B()

    dr.set_label(A.opaque, "A.opaque")
    dr.set_label(B.opaque, "B.opaque")

    a.value = t(2)
    b.value = t(3)

    print(f"{a.value.index=}")

    if opaque:
        dr.make_opaque(a.value, b.value)

    c = BasePtr(a, a, None, a, a)
    dr.set_label(c, "c")

    x = t(1, 2, 8, 3, 4)
    dr.set_label(x, "x")

    def func(c, xi):
        x = c.h(xi)
        dr.make_opaque(x)
        return c.g(x)

    frozen = dr.freeze(func)

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = frozen(c, x)

    assert dr.all(xo == func(c, x))

    a.value = t(3)
    b.value = t(2)

    if opaque:
        dr.make_opaque(a.value, b.value)

    c = BasePtr(a, a, None, b, b)
    dr.set_label(c, "c")

    with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
        with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
            xo = frozen(c, x)

    assert frozen.n_recordings == 1
    # if opaque:
    #     assert frozen.n_recordings == 1
    # else:
    #     assert frozen.n_recordings == 2

    assert dr.all(xo == func(c, x))


@pytest.test_arrays("float32, jit, shape=(*)")
def test01_freeze(t):
    UInt32 = dr.uint32_array_t(t)
    Float = dr.float32_array_t(t)

    @dr.freeze
    def my_kernel(x):
        x_int = UInt32(x)
        result = x * x
        result_int = UInt32(result)

        return result, x_int, result_int

    for i in range(3):
        print(f"------------------------------ {i}")
        x = Float([1.0, 2.0, 3.0]) + dr.opaque(Float, i)

        y, x_int, y_int = my_kernel(x)
        dr.schedule(y, x_int, y_int)
        print("Input was:", x)
        print("Outputs were:", y, x_int, y_int)
        assert dr.allclose(y, dr.square(x))
        assert dr.allclose(x_int, UInt32(x))
        assert dr.allclose(y_int, UInt32(y))
        print("------------------------------")


@pytest.mark.parametrize("freeze_first", (True, False))
@pytest.test_arrays("float32, jit, shape=(*)")
def test12_calling_frozen_from_frozen(t, freeze_first):
    mod = sys.modules[t.__module__]
    Float = mod.Float32
    Array3f = mod.Array3f
    n = 37
    x = dr.full(Float, 1.5, n) + dr.opaque(Float, 2)
    y = dr.full(Float, 0.5, n) + dr.opaque(Float, 10)
    dr.eval(x, y)

    @dr.freeze
    def fun1(x):
        return dr.square(x)

    @dr.freeze
    def fun2(x, y):
        return fun1(x) + fun1(y)

    # Calling a frozen function from a frozen function.
    if freeze_first:
        dr.eval(fun1(x))

    result1 = fun2(x, y)
    assert dr.allclose(result1, dr.square(x) + dr.square(y))

    if not freeze_first:
        # If the nested function hasn't been recorded yet, calling it
        # while freezing the outer function shouldn't freeze it with
        # those arguments.
        # In other words, any freezing mechanism should be completely
        # disabled while recording a frozen function.
        # assert fun1.frozen.kernels is None

        # We should therefore be able to freeze `fun1` with a different
        # type of argument, and both `fun1` and `fun2` should work fine.
        result2 = fun1(Array3f(0.5, x, y))
        assert dr.allclose(result2, Array3f(0.5 * 0.5, dr.square(x), dr.square(y)))

        result3 = fun2(2 * x, 0.5 * y)
        assert dr.allclose(result3, dr.square(2 * x) + dr.square(0.5 * y))


@pytest.test_arrays("float32, jit, shape=(*)")
def test17_no_inputs(t):
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    Float = mod.Float

    @dr.freeze
    def fun(a):
        x = t(dr.linspace(Float, -1, 1, 10)) + a
        source = x + 2 * x
        # source = get_single_entry(x + 2 * x)
        index = dr.arange(UInt32, dr.width(source))
        active = index % UInt32(2) != 0

        return dr.gather(Float, source, index, active)

    a = t(0.1)
    res1 = fun(a)
    res2 = fun(a)
    res3 = fun(a)

    assert dr.allclose(res1, res2)
    assert dr.allclose(res1, res3)


@pytest.test_arrays("float32, jit, shape=(*)")
def test18_with_gathers(t):
    import numpy as np

    n = 20
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(t, n))))

    def fun(x, idx):
        active = idx % 2 != 0
        source = get_single_entry(x)
        return dr.gather(type(source), source, idx, active=active)

    fun_frozen = dr.freeze(fun)

    # 1. Recording call
    x1 = t(rng.uniform(low=-1, high=1, size=shape))
    idx1 = dr.arange(UInt32, n)
    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1, idx1))

    # 2. Different source as during recording
    x2 = t(rng.uniform(low=-2, high=-1, size=shape))
    idx2 = idx1

    result2 = fun_frozen(x2, idx2)
    assert dr.allclose(result2, fun(x2, idx2))

    x3 = x2
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3, idx3))

    # 3. Same source as during recording
    result4 = fun_frozen(x1, idx1)
    assert dr.allclose(result4, result1)


@pytest.test_arrays("float32, cuda, jit, shape=(*)")
def test20_scatter_with_op(t):
    import numpy as np

    n = 16
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32

    rng = np.random.default_rng(seed=1234)

    def func(x, idx):
        active = idx % 2 != 0

        result = x - 0.5
        dr.scatter(x, result, idx, active=active)
        return result

    func_frozen = dr.freeze(func)

    # 1. Recording call
    print("-------------------- start result1")
    x1 = t(rng.uniform(low=-1, high=1, size=[n]))
    x1_copy = t(x1)
    x1_copy_copy = t(x1)
    idx1 = dr.arange(UInt32, n)

    result1 = func_frozen(x1, idx1)

    # assert dr.allclose(x1, x1_copy)
    assert dr.allclose(result1, func(x1_copy, idx1))

    # 2. Different source as during recording
    print("-------------------- start result2")
    # TODO: problem: during trace, the actual x1 Python variable changes
    #       from index r2 to index r12 as a result of the `scatter`.
    #       But in subsequent launches, even if we successfully create a new
    #       output buffer equivalent to r12, it doesn't get assigned to `x2`.
    x2 = t(rng.uniform(low=-2, high=-1, size=[n]))
    x2_copy = t(x2)
    idx2 = idx1
    # print(f'Before: {x2.index=}, {idx2.index=}')

    result2 = func_frozen(x2, idx2)
    # print(f'After : {x2.index=}, {idx2.index=}')
    print("-------------------- done with result2")
    assert dr.allclose(result2, func(x2_copy, idx2))
    # assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = t(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = func_frozen(x3, idx3)
    assert dr.allclose(result3, func(x3_copy, idx3))
    # assert dr.allclose(x3, x3_copy)

    print("=====================================")
    # # 3. Same source as during recording
    result4 = func_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    # # assert dr.allclose(x1_copy_copy, x1)


@pytest.test_arrays("float32, llvm, jit, shape=(*)")
def test_segv(t):
    import numpy as np

    n = 16
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32

    rng = np.random.default_rng(seed=1234)

    def func(x):
        idx = dr.arange(UInt32, dr.width(x))
        dr.set_label(x, "x1")
        dr.set_label(idx, "idx")
        active = idx % 2 != 0
        dr.set_label(active, "active")

        result = x - 0.5
        dr.set_label(result, "result")
        dr.scatter(x, result, idx, active=active)
        dr.set_label(x, "x2")

    func_frozen = dr.freeze(func)

    print("-------------------- start result1")
    x0 = t(rng.uniform(low=-1, high=1, size=[n]))
    # x1 = t(x0)
    x1 = x0
    x2 = t(x1)

    result1 = func_frozen(x1)

    result2 = func_frozen(x2)


# @pytest.test_arrays("float32, llvm, jit, shape=(*)")
# def test_segv(t):
#     import numpy as np
#
#     n = 16
#     mod = sys.modules[t.__module__]
#     UInt32 = mod.UInt32
#
#     rng = np.random.default_rng(seed=1234)
#
#     def func(x):
#         return x + 1
#
#     func_frozen = dr.freeze(func)
#
#     for i in range(3):
#         x = t(rng.uniform(low=-1, high=1, size=[n]))
#         result = func_frozen(x)
#         dr.eval(result)


@pytest.test_arrays("float32, jit, shape=(*)")
def test21_with_gather_and_scatter(t):
    # TODO: this function seems to be causing some problems with pytest,
    # something about `repr()` being called on a weird / uninitialized JIT variable.
    # This crash is triggered even when the test should otherwise pass.

    import numpy as np

    n = 20
    mod = sys.modules[t.__module__]
    UInt32 = mod.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(t, n))))

    def fun(x, idx):
        active = idx % 2 != 0
        dest = get_single_entry(x)

        values = dr.gather(UInt32, idx, idx, active=active)
        values = type(dest)(values)
        dr.scatter(dest, values, idx, active=active)
        return dest, values

    fun_frozen = dr.freeze(fun)

    # 1. Recording call
    x1 = t(rng.uniform(low=-1, high=1, size=shape))
    x1_copy = t(x1)
    x1_copy_copy = t(x1)
    idx1 = dr.arange(UInt32, n)

    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1_copy, idx1))
    assert dr.allclose(x1, x1_copy)

    # 2. Different source as during recording
    x2 = t(rng.uniform(low=-2, high=-1, size=shape))
    x2_copy = t(x2)
    idx2 = idx1

    result2 = fun_frozen(x2, idx2)
    assert dr.allclose(result2, fun(x2_copy, idx2))
    assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = t(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3_copy, idx3))
    assert dr.allclose(x3, x3_copy)

    # 3. Same source as during recording
    result4 = fun_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    assert dr.allclose(x1_copy_copy, x1)


@pytest.mark.parametrize("relative_size", ["<", "=", ">"])
@pytest.test_arrays("float32, jit, shape=(*)")
def test22_gather_only_pointer_as_input(t, relative_size):
    mod = sys.modules[t.__module__]
    Array3f = mod.Array3f
    Float = mod.Float32
    UInt32 = mod.UInt32

    import numpy as np

    rng = np.random.default_rng(seed=1234)

    if relative_size == "<":

        def fun(v):
            idx = dr.arange(UInt32, 0, dr.width(v), 3)
            return Array3f(
                dr.gather(Float, v, idx),
                dr.gather(Float, v, idx + 1),
                dr.gather(Float, v, idx + 2),
            )

    elif relative_size == "=":

        def fun(v):
            idx = dr.arange(UInt32, 0, dr.width(v)) // 2
            return Array3f(
                dr.gather(Float, v, idx),
                dr.gather(Float, v, idx + 1),
                dr.gather(Float, v, idx + 2),
            )

    elif relative_size == ">":

        def fun(v):
            max_width = dr.width(v)
            idx = dr.arange(UInt32, 0, 5 * max_width)
            # TODO(!): what can we do against this literal being baked into the kernel?
            active = (idx + 2) < max_width
            return Array3f(
                dr.gather(Float, v, idx, active=active),
                dr.gather(Float, v, idx + 1, active=active),
                dr.gather(Float, v, idx + 2, active=active),
            )

    fun_freeze = dr.freeze(fun)

    def check_results(v, result):
        size = v.size
        if relative_size == "<":
            expected = v.T
        if relative_size == "=":
            idx = np.arange(0, size) // 2
            expected = v.ravel()
            expected = np.stack(
                [
                    expected[idx],
                    expected[idx + 1],
                    expected[idx + 2],
                ],
                axis=0,
            )
        elif relative_size == ">":
            idx = np.arange(0, 5 * size)
            mask = (idx + 2) < size
            expected = v.ravel()
            expected = np.stack(
                [
                    np.where(mask, expected[(idx) % size], 0),
                    np.where(mask, expected[(idx + 1) % size], 0),
                    np.where(mask, expected[(idx + 2) % size], 0),
                ],
                axis=0,
            )

        assert np.allclose(result.numpy(), expected)

    # Note: Does not fail for n=1
    n = 7
    # dr.set_log_level(dr.LogLevel.Debug)

    for i in range(3):
        v = rng.uniform(size=[n, 3])
        result = fun(Float(v.ravel()))
        check_results(v, result)

    for i in range(10):
        if i <= 5:
            n_lanes = n
        else:
            n_lanes = n + i

        v = rng.uniform(size=[n_lanes, 3])
        result = fun_freeze(Float(v.ravel()))
        # print(f'{i=}, {n_lanes=}, {v.shape=}, {result.numpy().shape=}')

        expected_width = {
            "<": n_lanes,
            "=": n_lanes * 3,
            ">": n_lanes * 3 * 5,
        }[relative_size]

        # if i == 0:
        # assert len(fun_freeze.frozen.kernels)
        # for kernel in fun_freeze.frozen.kernels.values():
        #     assert kernel.original_input_size == n * 3
        #     if relative_size == "<":
        #         assert kernel.original_launch_size == expected_width
        #         assert kernel.original_launch_size_ratio == (False, 3, True)
        #     elif relative_size == "=":
        #         assert kernel.original_launch_size == expected_width
        #         assert kernel.original_launch_size_ratio == (False, 1, True)
        #     else:
        #         assert kernel.original_launch_size == expected_width
        #         assert kernel.original_launch_size_ratio == (True, 5, True)

        assert dr.width(result) == expected_width
        if relative_size == ">" and n_lanes != n:
            pytest.xfail(
                reason="The width() of the original input is baked into the kernel to compute the `active` mask during the first launch, so results are incorrect once the width changes."
            )

        check_results(v, result)


@pytest.test_arrays("float32, jit, shape=(*)")
def test24_multiple_kernels(t):
    def fn(x: dr.ArrayBase, y: dr.ArrayBase, flag: bool):
        # TODO: test with gathers and scatters, which is a really important use-case.
        # TODO: test with launches of different sizes (including the auto-sizing logic)
        # TODO: test with an intermediate output of literal type
        # TODO: test multiple kernels that scatter_add to a newly allocated kernel in sequence.

        # First kernel uses only `x`
        quantity = 0.5 if flag else -0.5
        intermediate1 = x + quantity
        intermediate2 = x * quantity
        dr.eval(intermediate1, intermediate2)

        # Second kernel uses `x`, `y` and one of the intermediate result
        result = intermediate2 + y

        # The function returns some mix of outputs
        return intermediate1, None, y, result

    n = 15
    x = dr.full(t, 1.5, n) + dr.opaque(t, 0.2)
    y = dr.full(t, 0.5, n) + dr.opaque(t, 0.1)
    dr.eval(x, y)

    ref_results = fn(x, y, flag=True)
    dr.eval(ref_results)

    fn_frozen = dr.freeze(fn)
    for _ in range(2):
        results = fn_frozen(x, y, flag=True)
        assert dr.allclose(results[0], ref_results[0])
        assert results[1] is None
        assert dr.allclose(results[2], y)
        assert dr.allclose(results[3], ref_results[3])

    # TODO:
    # We don't yet make a difference between check and no-check

    # for i in range(4):
    #     new_y = y + float(i)
    #     # Note: we did not enabled `check` mode, so changing this Python
    #     # value will not throw an exception. The new value has no influence
    #     # on the result even though without freezing, it would.
    #     # TODO: support "signature" detection and create separate frozen
    #     #       function instances.
    #     new_flag = (i % 2) == 0
    #     results = fn_frozen(x, new_y, flag=new_flag)
    #     assert dr.allclose(results[0], ref_results[0])
    #     assert results[1] is None
    #     assert dr.allclose(results[2], new_y)
    #     assert dr.allclose(results[3], x * 0.5 + new_y)


@pytest.test_arrays("float32, jit, shape=(*)")
def test27_global_flag(t):
    Float = t

    @dr.freeze
    def my_fn(a, b, c=0.5):
        return a + b + c

    # Recording
    one = Float([1.0] * 9)
    result1 = my_fn(one, one, c=0.1)
    assert dr.allclose(result1, 2.1)

    # Can change the type of an input
    result2 = my_fn(one, one, c=Float(0.6))
    assert dr.allclose(result2, 2.6)

    assert my_fn.n_recordings == 2

    # Disable frozen kernels globally, now the freezing
    # logic should be completely bypassed
    with dr.scoped_set_flag(dr.JitFlag.KernelFreezing, False):
        result3 = my_fn(one, one, c=0.9)
        assert dr.allclose(result3, 2.9)


# @pytest.mark.parametrize("struct_style", ["drjit", "dataclass"])
@pytest.mark.parametrize("struct_style", ["drjit", "dataclass"])
# @pytest.test_arrays("float32, llvm, jit, -is_diff, shape=(*)")
@pytest.test_arrays("float32, jit, shape=(*)")
def test28_return_types(t, struct_style):
    # WARN: only working on CUDA!
    mod = sys.modules[t.__module__]
    Float = t
    Array3f = mod.Array3f
    UInt32 = mod.UInt32

    import numpy as np

    if struct_style == "drjit":

        class ToyDataclass:
            DRJIT_STRUCT: dict = {"a": Float, "b": Float}
            a: Float
            b: Float

            def __init__(self, a=None, b=None):
                self.a = a
                self.b = b

    else:
        assert struct_style == "dataclass"

        @dataclass(kw_only=True, frozen=True)
        class ToyDataclass:
            a: Float
            b: Float

    print("T1")

    # 1. Many different types
    @dr.freeze
    def toy1(x: Float) -> Float:
        y = x**2 + dr.sin(x)
        z = x**2 + dr.cos(x)
        return (x, y, z, ToyDataclass(a=x, b=y), {"x": x, "yi": UInt32(y)}, [[[[x]]]])

    for i in range(2):
        input = Float(np.full(17, i))
        # input = dr.full(Float, i, 17)
        result = toy1(input)
        # print(f"{input=}")
        assert isinstance(result[0], Float)
        assert isinstance(result[1], Float)
        assert isinstance(result[2], Float)
        assert isinstance(result[3], ToyDataclass)
        assert isinstance(result[4], dict)
        assert result[4].keys() == set(("x", "yi"))
        assert isinstance(result[4]["x"], Float)
        assert isinstance(result[4]["yi"], UInt32)
        assert isinstance(result[5], list)
        assert isinstance(result[5][0], list)
        assert isinstance(result[5][0][0], list)
        assert isinstance(result[5][0][0][0], list)

    # 2. Many different types
    @dr.freeze
    def toy2(x: Float, target: Float) -> Float:
        dr.scatter(target, 0.5 + x, dr.arange(UInt32, dr.width(x)))
        return None

    for i in range(3):
        input = Float([i] * 17)
        target = dr.opaque(Float, 0, dr.width(input))
        # target = dr.full(Float, 0, dr.width(input))
        # target = dr.empty(Float, dr.width(input))

        result = toy2(input, target)
        assert dr.allclose(target, 0.5 + input)
        assert result is None

    # 3. DRJIT_STRUCT as input and returning nested dictionaries
    @dr.freeze
    def toy3(x: Float, y: ToyDataclass) -> Float:
        x_d = dr.detach(x, preserve_type=False)
        return {
            "a": x,
            "b": (x, UInt32(2 * y.a + y.b)),
            "c": None,
            "d": {
                "d1": x + x,
                "d2": Array3f(x_d, -x_d, 2 * x_d),
                "d3": None,
                "d4": {},
                "d5": tuple(),
                "d6": list(),
                "d7": ToyDataclass(a=x, b=2 * x),
            },
            "e": [x, {"e1": None}],
        }

    for i in range(3):
        input = Float([i] * 5)
        input2 = ToyDataclass(a=input, b=Float(4.0))
        result = toy3(input, input2)
        assert isinstance(result, dict)
        assert isinstance(result["a"], Float)
        assert isinstance(result["b"], tuple)
        assert isinstance(result["b"][0], Float)
        assert isinstance(result["b"][1], UInt32)
        assert result["c"] is None
        assert isinstance(result["d"], dict)
        assert isinstance(result["d"]["d1"], Float)
        assert isinstance(result["d"]["d2"], Array3f)
        assert result["d"]["d3"] is None
        assert isinstance(result["d"]["d4"], dict) and len(result["d"]["d4"]) == 0
        assert isinstance(result["d"]["d5"], tuple) and len(result["d"]["d5"]) == 0
        assert isinstance(result["d"]["d6"], list) and len(result["d"]["d6"]) == 0
        assert isinstance(result["d"]["d7"], ToyDataclass)
        assert dr.allclose(result["d"]["d7"].a, input)
        assert dr.allclose(result["d"]["d7"].b, 2 * input)
        assert isinstance(result["e"], list)
        assert isinstance(result["e"][0], Float)
        assert isinstance(result["e"][1], dict)
        assert result["e"][1]["e1"] is None


@pytest.test_arrays("float32, jit, shape=(*)")
def test29_drjit_struct_and_matrix(t):
    package = sys.modules[t.__module__]
    Float = package.Float
    Array4f = package.Array4f
    Matrix4f = package.Matrix4f

    class MyTransform4f:
        DRJIT_STRUCT = {
            "matrix": Matrix4f,
            "inverse": Matrix4f,
        }

        def __init__(self, matrix: Matrix4f = None, inverse: Matrix4f = None):
            self.matrix = matrix
            self.inverse = inverse

    @dataclass(kw_only=False, frozen=False)
    class Camera:
        to_world: MyTransform4f

    @dataclass(kw_only=False, frozen=False)
    class Batch:
        camera: Camera
        value: float = 0.5
        offset: float = 0.5

    @dataclass(kw_only=False, frozen=False)
    class Result:
        value: Float
        constant: int = 5

    def fun(batch: Batch, x: Array4f):
        res1 = batch.camera.to_world.matrix @ x
        res2 = batch.camera.to_world.matrix @ x + batch.offset
        res3 = batch.value + x
        res4 = Result(value=batch.value)
        return res1, res2, res3, res4

    fun_frozen = dr.freeze(fun)

    n = 7
    for i in range(4):
        x = Array4f(
            *(dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i) + k for k in range(4))
        )
        mat = Matrix4f(
            *(
                dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i) + ii + jj
                for jj in range(4)
                for ii in range(4)
            )
        )
        trafo = MyTransform4f()
        trafo.matrix = mat
        trafo.inverse = dr.rcp(mat)

        batch = Batch(
            camera=Camera(to_world=trafo),
            value=dr.linspace(Float, -1, 0, n) - dr.opaque(Float, i),
        )
        # dr.eval(x, trafo, batch.value)

        results = fun_frozen(batch, x)
        expected = fun(batch, x)

        assert len(results) == len(expected)
        for result_i, (value, expected) in enumerate(zip(results, expected)):
            print(f"{result_i}: {value=}")
            print(f"{result_i}: {expected=}")

            assert type(value) == type(expected)
            if isinstance(value, Result):
                value = value.value
                expected = expected.value
            assert dr.allclose(value, expected), str(result_i)


@pytest.test_arrays("float32, jit, shape=(*)")
def test30_with_dataclass_in_out(t):
    mod = sys.modules[t.__module__]
    Int32 = mod.Int32
    UInt32 = mod.UInt32
    Bool = mod.Bool

    @dataclass(kw_only=True, frozen=False)
    class MyRecord:
        step_in_segment: Int32
        total_steps: UInt32
        short_segment: Bool

    def acc_fn(record: MyRecord):
        record.step_in_segment += Int32(2)
        return Int32(record.total_steps + record.step_in_segment)

    # Initialize MyRecord
    n_rays = 100
    record = MyRecord(
        step_in_segment=UInt32([1] * n_rays),
        total_steps=UInt32([0] * n_rays),
        short_segment=dr.zeros(Bool, n_rays),
    )

    # Create frozen kernel that contains another function
    frozen_acc_fn = dr.freeze(acc_fn)

    accumulation = dr.zeros(UInt32, n_rays)
    n_iter = 12
    for _ in range(n_iter):
        accumulation += frozen_acc_fn(record)

    expected = 0
    for i in range(n_iter):
        expected += 0 + 2 * (i + 1) + 1
    assert dr.all(accumulation == expected)


@pytest.test_arrays("float32, jit, shape=(*)")
def test32_allocated_scratch_buffer(t):
    # FIX: leaks types, memory and variables
    """
    Frozen functions may want to allocate some scratch space, scatter to it
    in a first kernel, and read / use the values later on. As long as the
    size of the scratch space can be guessed (e.g. a multiple of the launch width,
    or matching the width of an existing input), we should be able to support it.

    On the other hand, the "scattering to an unknown buffer" pattern may actually
    be scattering to an actual pre-existing buffer, which the user simply forgot
    to include in the `state` lambda. In order to catch that case, we at least
    check that the "scratch buffer" was read from in one of the kernels.
    Otherwise, we assume it was meant as a side-effect into a pre-existing buffer.
    """
    mod = sys.modules[t.__module__]
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)
    UInt32 = mod.UInt32

    # Note: we are going through an object / method, otherwise the closure
    # checker would already catch the `forgotten_target_buffer` usage.
    class Model:
        DRJIT_STRUCT = {
            "some_state": UInt32,
            # "forgotten_target_buffer": UInt32,
        }

        def __init__(self):
            self.some_state = UInt32([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.forgotten_target_buffer = self.some_state + 1
            dr.eval(self.some_state, self.forgotten_target_buffer)

        @dr.freeze
        def fn1(self, x):
            # Note: assuming here that the width of `forgotten_target_buffer` doesn't change
            index = dr.arange(UInt32, dr.width(x)) % dr.width(
                self.forgotten_target_buffer
            )
            dr.scatter(self.forgotten_target_buffer, x, index)

            return 2 * x

        @dr.freeze
        def fn2(self, x):
            # Scratch buffer with width equal to a state variable
            scratch = dr.zeros(UInt32, dr.width(self.some_state))
            # Kernel 1: write to `scratch`
            index = dr.arange(UInt32, dr.width(x)) % dr.width(self.some_state)
            dr.scatter(scratch, x, index)
            # Kernel 2: use values from `scratch` directly
            result = dr.square(scratch)
            # We don't actually return `scratch`, its lifetime is limited to the frozen function.
            return result

        @dr.freeze
        def fn3(self, x):
            # Scratch buffer with width equal to a state variable
            scratch = dr.zeros(UInt32, dr.width(self.some_state))
            # Kernel 1: write to `scratch`
            index = dr.arange(UInt32, dr.width(x)) % dr.width(self.some_state)
            dr.scatter(scratch, x, index)
            # Kernel 2: use values from `scratch` via a gather
            result = x + dr.gather(UInt32, scratch, index)
            # We don't actually return `scratch`, its lifetime is limited to the frozen function.
            return result

    model = Model()

    # Suspicious usage, should not allow it to avoid silent surprising behavior
    for i in range(4):
        x = UInt32(list(range(i + 2)))
        assert dr.width(x) < dr.width(model.forgotten_target_buffer)

        if dr.flag(dr.JitFlag.KernelFreezing):
            with pytest.raises(
                RuntimeError,
                match="was created before recording was started, but it was not specified as an input variable!",
            ):
                result = model.fn1(x)
            break

        else:
            result = model.fn1(x)
            assert dr.allclose(result, 2 * x)

            expected = UInt32(model.some_state + 1)
            dr.scatter(expected, x, dr.arange(UInt32, dr.width(x)))
            assert dr.allclose(model.forgotten_target_buffer, expected)

    # Expected usage, we should allocate the buffer and allow the launch
    for i in range(4):
        x = UInt32(list(range(i + 2)))  # i+1
        assert dr.width(x) < dr.width(model.some_state)
        result = model.fn2(x)
        expected = dr.zeros(UInt32, dr.width(model.some_state))
        dr.scatter(expected, x, dr.arange(UInt32, dr.width(x)))
        assert dr.allclose(result, dr.square(expected))

    # Expected usage, we should allocate the buffer and allow the launch
    for i in range(4):
        x = UInt32(list(range(i + 2)))  # i+1
        assert dr.width(x) < dr.width(model.some_state)
        result = model.fn3(x)
        assert dr.allclose(result, 2 * x)


@pytest.test_arrays("float32, jit, shape=(*)")
def test33_simple_reductions(t):
    import numpy as np

    mod = sys.modules[t.__module__]
    Float = mod.Float32
    n = 37

    @dr.freeze
    def simple_sum(x):
        return dr.sum(x)

    @dr.freeze
    def simple_product(x):
        return dr.prod(x)

    @dr.freeze
    def simple_min(x):
        return dr.min(x)

    @dr.freeze
    def simple_max(x):
        return dr.max(x)

    @dr.freeze
    def sum_not_returned_wide(x):
        return dr.sum(x) + x

    @dr.freeze
    def sum_not_returned_single(x):
        return dr.sum(x) + 4

    def check_expected(fn, expected):
        result = fn(x)

        assert dr.width(result) == dr.width(expected)
        assert isinstance(result, Float)
        assert dr.allclose(result, expected)

    for i in range(3):
        x = dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i)

        x_np = x.numpy()
        check_expected(simple_sum, np.sum(x_np).item())
        check_expected(simple_product, np.prod(x_np).item())
        check_expected(simple_min, np.min(x_np).item())
        check_expected(simple_max, np.max(x_np).item())

        check_expected(sum_not_returned_wide, np.sum(x_np).item() + x)
        check_expected(sum_not_returned_single, np.sum(x_np).item() + 4)

@pytest.test_arrays("float32, jit, shape=(*)")
def test34_prefix_reductions(t):
    import numpy as np
    
    mod = sys.modules[t.__module__]
    Float = mod.Float32
    n = 37

    @dr.freeze
    def prefix_sum(x):
        return dr.prefix_reduce(dr.ReduceOp.Add, x, exclusive = False)
    
    def check_expected(fn, expected):
        result = fn(x)

        assert dr.width(result) == dr.width(expected)
        assert isinstance(result, Float)
        assert dr.allclose(result, expected)

    for i in range(3):
        x = dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i)

        x_np = x.numpy()
        check_expected(prefix_sum, Float(np.cumsum(x_np)))


@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
def test34_reductions_with_ad(t):
    Float = t
    n = 37

    @dr.freeze
    def sum_with_ad(x, width_opaque):
        intermediate = 2 * x + 1
        dr.enable_grad(intermediate)

        result = dr.square(intermediate)

        # Unfortunately, as long as we don't support creating opaque values
        # within a frozen kernel, we can't use `dr.mean()` directly.
        loss = dr.sum(result) / width_opaque
        # loss = dr.mean(result)
        dr.backward(loss)

        print(f"{dr.grad(intermediate).index=}")

        return result, intermediate

    @dr.freeze
    def product_with_ad(x):
        dr.enable_grad(x)
        loss = dr.prod(x)
        dr.backward_from(loss)

    for i in range(3):
        x = dr.linspace(Float, 0, 1, n + i) + dr.opaque(Float, i)
        result, intermediate = sum_with_ad(x, dr.opaque(Float, dr.width(x)))
        assert dr.width(result) == n + i

        assert dr.grad_enabled(result)
        assert dr.grad_enabled(intermediate)
        assert not dr.grad_enabled(x)
        intermediate_expected = 2 * x + 1
        assert dr.allclose(intermediate, intermediate_expected)
        assert dr.allclose(result, dr.square(intermediate_expected))
        assert sum_with_ad.n_recordings == 1
        assert dr.allclose(dr.grad(result), 0)
        assert dr.allclose(
            dr.grad(intermediate), 2 * intermediate_expected / dr.width(x)
        )

    for i in range(3):
        x = dr.linspace(Float, 0.1, 1, n + i) + dr.opaque(Float, i)
        result = product_with_ad(x)

        assert result is None
        assert dr.grad_enabled(x)
        with dr.suspend_grad():
            expected_grad = dr.prod(x) / x
        assert dr.allclose(dr.grad(x), expected_grad)


# @pytest.test_arrays("float32, jit, shape=(*)")
# def test35_mean(t):
#     def func(x):
#         return dr.mean(x)
#
#     frozen_func = dr.freeze(func)
#
#     n = 10
#
#     for i in range(3):
#         x = dr.linspace(t, 0, 1, n + i) + dr.opaque(t, i)
#
#         result = frozen_func(x)
#         assert dr.allclose(result, func(x))


@pytest.test_arrays("float32, jit, shape=(*)")
def test36_size_aliasing(t):
    def func(x, y):
        return x + 1, y + 2

    frozen_func = dr.freeze(func)

    n = 3

    for i in range(3):
        x = dr.linspace(t, 0, 1, n) + dr.opaque(t, i)
        y = dr.linspace(t, 0, 1, n + i) + dr.opaque(t, i)

        result = frozen_func(x, y)

        assert dr.allclose(result, func(x, y))

    """
    We should have two recordings, one for which the experssions x+1 and y+1 are compiled into the same kernel,
    because x and y have the same size and one where they are compiled seperately because their sizes are different.
    """
    assert frozen_func.n_recordings == 2


@pytest.test_arrays("float32, jit, -is_diff, shape=(*)")
def test37_pointer_aliasing(t):
    """
    Dr.Jit employs a memory cache, which means that two variables
    get allocated the same memory region, if one is destroyed
    before the other is created.
    Since we track variables using their pointers in the `RecordThreadState`,
    we have to update the `ptr_to_slot` mapping for new variables.
    """

    n = 4

    def func(x):
        y = x + 1
        dr.make_opaque(y)
        for i in range(3):
            y = y + 1
            dr.make_opaque(y)
        return y

    for i in range(10):
        frozen_func = dr.freeze(func)

        x = dr.linspace(t, 0, 1, n) + dr.opaque(t, i)
        assert dr.allclose(frozen_func(x), func(x))

        x = dr.linspace(t, 0, 1, n) + dr.opaque(t, i)
        assert dr.allclose(frozen_func(x), func(x))


@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
def test38_simple_ad_fully_inside(t):
    mod = sys.modules[t.__module__]
    Float = mod.Float

    def my_kernel(x):
        dr.enable_grad(x)

        result = x * x
        dr.backward(result)

        return result

    for start_enabled in (True, False):
        # Re-freeze
        my_kernel_frozen = dr.freeze(my_kernel)

        for i in range(3):
            print(f"------------------------------ {i=}, {start_enabled=}")
            x = Float([1.0, 2.0, 3.0]) + dr.opaque(Float, i)
            if start_enabled:
                dr.enable_grad(x)

            y = my_kernel_frozen(x)
            grad_x = dr.grad(x)
            grad_y = dr.grad(y)
            dr.schedule(y, grad_x, grad_y)
            print(f"Input was: {x=}")
            print(f"Outputs were: {y=}, {grad_y=}, {grad_x=}")
            assert dr.allclose(y, dr.square(x))
            assert dr.allclose(grad_y, 0)
            assert dr.allclose(grad_x, 2 * x)

            # Status of grad_enabled should be restored (side-effect of the function),
            #  even if it wasn't enabled at first
            assert dr.grad_enabled(x)
            print("------------------------------")


@pytest.mark.parametrize("set_some_literal_grad", (False,))
@pytest.mark.parametrize("inputs_end_enabled", (True, False))
@pytest.mark.parametrize("inputs_start_enabled", (True,))
@pytest.mark.parametrize("params_end_enabled", (False, False))
@pytest.mark.parametrize("params_start_enabled", (True,))
@pytest.mark.parametrize("freeze", (True,))
@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
def test39_suspend_resume(
    t,
    params_start_enabled,
    params_end_enabled,
    inputs_start_enabled,
    inputs_end_enabled,
    set_some_literal_grad,
    freeze,
):

    mod = sys.modules[t.__module__]
    Float = mod.Float32
    UInt32 = mod.UInt32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)

    # TODO: remove this
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)

    class MyModel:
        DRJIT_STRUCT = {"params": Float}

        def __init__(self, params):
            self.params = params
            self.frozen_eval = dr.freeze(type(self).eval) if freeze else type(self).eval

        def eval(
            self,
            x: Float,
            params_end_enabled: bool,
            inputs_end_enabled: bool,
            set_some_literal_grad: bool,
        ):
            idx = dr.arange(UInt32, dr.width(x)) % dr.width(self.params)
            latents = dr.gather(Float, self.params, idx)
            result = x * latents

            with dr.resume_grad():
                dr.set_grad_enabled(self.params, params_end_enabled)
                dr.set_grad_enabled(x, inputs_end_enabled)
                if set_some_literal_grad:
                    # If grads are not enabled, this will get ignored, which is fine
                    dr.set_grad(x, Float(6.66))
                print(f"resumed: {dr.grad_enabled(x)=}")

            print(f"suspended: {dr.grad_enabled(x)=}")
            return result

    model = MyModel(params=Float([1, 2, 3, 4, 5]))

    for i in range(3):
        print(f"------------------------------ {i=}")
        # Inputs of different widths
        x = Float([0.1, 0.2, 0.3, 0.4, 0.5, 0.6] * (i + 1)) + dr.opaque(Float, i)

        dr.set_grad_enabled(model.params, params_start_enabled)
        dr.set_grad_enabled(x, inputs_start_enabled)

        # print(f'Before: {model.params.index=}, {dr.grad(model.params).index=}, {dr.grad(model.params).is_literal_()}')

        dr.eval(x, dr.grad(x))
        print(f"{dr.grad_enabled(x)=}, {x.index=}, {x.index_ad=}")

        with dr.suspend_grad():
            with dr.resume_grad():
                print(f"{dr.grad_enabled(x)=}, {x.index=}, {x.index_ad=}")
            result = model.frozen_eval(
                model, x, params_end_enabled, inputs_end_enabled, set_some_literal_grad
            )

        # dr.eval(result, model.params, dr.grad(model.params))
        print(f"After: {model.params.index=}, {dr.grad(model.params).index=} ")
        assert not dr.grad_enabled(result)
        assert dr.grad_enabled(model.params) == params_end_enabled
        assert dr.grad_enabled(x) == inputs_end_enabled

        # The frozen function should restore the right width, even for a zero-valued literal.
        # The default gradients are a zero-valued literal array
        # with a width equal to the array's width
        grads = dr.grad(model.params)
        assert dr.width(grads) == dr.width(model.params)
        assert dr.all(grads == 0)

        grads = dr.grad(x)
        assert dr.width(grads) == dr.width(x)
        if inputs_end_enabled and set_some_literal_grad:
            assert dr.all(grads == 6.66)
        else:
            assert dr.all(grads == 0)

        assert model.frozen_eval.n_recordings == 1

    # dr.set_log_level(log_level)


@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
@pytest.mark.parametrize("freeze", (True,))
@pytest.mark.parametrize("change_params_width", (False,))
def test40_with_grad_scatter(t, freeze: bool, change_params_width):
    mod = sys.modules[t.__module__]
    Float = mod.Float32
    UInt32 = mod.UInt32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)

    class Model:
        DRJIT_STRUCT = {"params": Float}

        def __init__(self, n):
            self.params = Float(list(range(1, n + 1)))
            assert dr.width(self.params) == n
            dr.enable_grad(self.params)

        def __call__(self):
            # Cheeky workaround for the frozen kernel signature checking
            pass

    def my_kernel(model, x, opaque_params_width):
        idx = dr.arange(UInt32, dr.width(x)) % opaque_params_width

        with dr.resume_grad():
            latents = dr.gather(Float, model.params, idx)
            contrib = x * latents
            dr.backward_from(contrib)

            return dr.detach(contrib)

    model = Model(5)
    my_kernel_frozen = dr.freeze(my_kernel) if freeze else my_kernel

    for i in range(6):
        print(f"------------------------------ {i=}")
        # Different width at each iteration
        x = Float([1.0, 2.0, 3.0] * (i + 1)) + dr.opaque(Float, i)

        # The frozen kernel should also support the params (and therefore its gradient buffer)
        # changing width without issues.
        if change_params_width and (i == 3):
            model = Model(10)
        # Reset gradients
        dr.set_grad(model.params, 0)
        assert dr.grad_enabled(model.params)

        with dr.suspend_grad():
            y = my_kernel_frozen(model, x, dr.opaque(UInt32, dr.width(model.params)))
        assert not dr.grad_enabled(x)
        assert not dr.grad_enabled(y)
        assert dr.grad_enabled(model.params)

        grad_x = dr.grad(x)
        grad_y = dr.grad(y)
        grad_p = dr.grad(model.params)
        print(f"Input was: {x=}")
        print(f"Outputs were: {y=}, {grad_y=}, {grad_x=}, {grad_p=}")
        # assert dr.allclose(y, dr.sqr(x))

        # Expected grads
        assert dr.allclose(grad_y, 0)
        assert dr.allclose(grad_x, 0)
        grad_p_expected = dr.zeros(Float, dr.width(model.params))
        idx = dr.arange(UInt32, dr.width(x)) % dr.width(model.params)
        dr.scatter_reduce(dr.ReduceOp.Add, grad_p_expected, x, idx)
        assert dr.allclose(grad_p, grad_p_expected)
        print(f"------------------------------")

    # dr.set_log_level(log_level)


@pytest.test_arrays("float32, jit, is_diff, shape=(*)")
def test41_tutorial_example(t):
    mod = sys.modules[t.__module__]
    Float = mod.Float32
    UInt32 = mod.UInt32

    @dr.freeze
    def frozen_eval(inputs, idx, params, target_value, grad_factor):
        intermediate = dr.gather(Float, params, idx)
        result = 0.5 * dr.square(intermediate) * inputs

        # Since reductions are not supported yet, we cannot compute a single
        # loss value here. It's not really a problem though, since DrJit can
        # backpropagate starting from arrays of any widths.
        loss_per_entry = dr.square(result - target_value) * grad_factor

        # The gradients resulting from backpropagation will be directly accumulated
        # (via dr.scatter_add()) into the gradient buffer of `params` (= `dr.grad(params)`).
        dr.backward_from(loss_per_entry)

        # It's fine to return the primal values of `result`, but keep in mind that they will
        # not be differentiable w.r.t. `params`.
        return dr.detach(result)

    params = Float([1, 2, 3, 4, 5])

    for _ in range(3):
        dr.disable_grad(params)
        dr.enable_grad(params)
        assert dr.all(dr.grad(params) == 0)

        inputs = Float([0.1, 0.2, 0.3])
        idx = UInt32([1, 2, 3])
        # Represents the optimizer's loss scale
        grad_factor = 4096 / dr.opaque(Float, dr.width(inputs))

        result = frozen_eval(
            inputs, idx, params, target_value=0.5, grad_factor=grad_factor
        )
        assert not dr.grad_enabled(result)
        # Gradients were correctly accumulated to `params`'s gradients.
        assert not dr.all(dr.grad(params) == 0)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test42_compress(t):

    mod = sys.modules[t.__module__]
    Float = mod.Float32
    UInt32 = mod.UInt32

    pkg = get_pkg(t)
    Sampler = pkg.Sampler

    def func(sampler: Sampler) -> UInt32:
        indices = dr.compress(sampler.next() < 0.5)
        return indices

    frozen = dr.freeze(func)

    sampler_func = Sampler(10)
    sampler_frozen = Sampler(10)
    for i in range(3):
        dr.all(frozen(sampler_frozen) == func(sampler_func))

    sampler_func = Sampler(11)
    sampler_frozen = Sampler(11)
    for i in range(3):
        dr.all(frozen(sampler_frozen) == func(sampler_func))

    assert frozen.n_recordings == 1


@pytest.test_arrays("uint32, llvm, -is_diff, jit, shape=(*)")
def test43_scatter_reduce_expanded(t):

    def func(target: t, src: t):
        dr.scatter_reduce(dr.ReduceOp.Add, target, src, dr.arange(t, dr.width(src)) % 2)

    frozen = dr.freeze(func)

    for i in range(4):
        src = dr.full(t, 1, 10 + i)
        dr.make_opaque(src)

        result = t([0] * (i + 2))
        dr.make_opaque(result)
        frozen(result, src)
        print(f"{result=}")

        reference = t([0] * (i + 2))
        dr.make_opaque(reference)
        func(reference, src)

        assert dr.all(result == reference)

    assert frozen.n_cached_recordings == 1
    assert frozen.n_recordings == 4
    
@pytest.test_arrays("uint32, llvm, -is_diff, jit, shape=(*)")
def test44_scatter_reduce_expanded_identity(t):

    def func(src: t):
        target = dr.zeros(t, 5)
        dr.scatter_reduce(dr.ReduceOp.Add, target, src, dr.arange(t, dr.width(src)) % 2)
        return target

    frozen = dr.freeze(func)

    for i in range(4):
        src = dr.full(t, 1, 10 + i)
        dr.make_opaque(src)

        result =  frozen(src)
        print(f"{result=}")

        reference = func(src)

        assert dr.all(result == reference)

    assert frozen.n_recordings == 1

@pytest.test_arrays("uint32, llvm, -is_diff, jit, shape=(*)")
def test45_scatter_reduce_expanded_no_memset(t):

    def func(src: t):
        target = dr.full(t, 5)
        dr.scatter_reduce(dr.ReduceOp.Add, target, src, dr.arange(t, dr.width(src)) % 2)
        return target

    frozen = dr.freeze(func)

    for i in range(4):
        src = dr.full(t, 1, 10 + i)
        dr.make_opaque(src)

        result =  frozen(src)
        print(f"{result=}")

        reference = func(src)

        assert dr.all(result == reference)

    assert frozen.n_recordings == 4
    assert frozen.n_cached_recordings == 1

@pytest.test_arrays("uint32, jit, shape=(*)")
def test44_python_inputs(t):

    def func(x: t, neg: bool):
        if neg:
            return -x + 1
        else:
            return x + 1

    frozen = dr.freeze(func)

    for i in range(3):
        for neg in [False, True]:
            x = t(1, 2, 3) + dr.opaque(t, i)

            res = frozen(x, neg)
            ref = func(x, neg)
            assert dr.all(res == ref)

    assert frozen.n_recordings == 2


@pytest.test_arrays("uint32, jit, shape=(*)")
def test35_scatter_inc(t):

    mod = sys.modules[t.__module__]
    Float = mod.Float32
    UInt32 = mod.UInt32

    n = 37

    def acc_with_scatter_inc(x, counter, out_buffer, max_points):
        active = x > 0.5
        out_idx = dr.scatter_inc(counter, UInt32(0), active=active)
        active &= out_idx < max_points

        # TODO: also test within a loop
        dr.scatter(out_buffer, x, out_idx, active=active)

    def test(i, func):
        x = dr.linspace(Float, 0.1, 1, n + i) + dr.opaque(Float, i) / 100
        counter = UInt32(dr.opaque(UInt32, 0))
        out_buffer = dr.zeros(Float, 10)
        max_points = UInt32(dr.opaque(UInt32, dr.width(out_buffer)))

        dr.set_label(x, "x")
        dr.set_label(counter, "counter")
        dr.set_label(out_buffer, "out_buffer")
        dr.set_label(max_points, "max_points")
        dr.eval(x, counter, out_buffer, max_points)

        func(x, counter, out_buffer, max_points)

        return out_buffer, counter

    def func(i):
        return test(i, acc_with_scatter_inc)

    acc_with_scatter_inc = dr.freeze(acc_with_scatter_inc)

    def frozen(i):
        return test(i, acc_with_scatter_inc)

    for i in range(3):

        res, _ = frozen(i)
        ref, _ = func(i)

        # assert dr.all(res == ref)

        # Should have filled all of the entries
        assert dr.all(res > 0.5)


@pytest.test_arrays("uint32, jit, shape=(*)")
def test36_read_while_frozen(t):
    # dr.set_flag(dr.JitFlag.KernelFreezing, True)
    assert dr.flag(dr.JitFlag.KernelFreezing)

    def func(x):
        return x[1]

    frozen = dr.freeze(func)

    x = t(1, 2, 3)
    with pytest.raises(RuntimeError):
        frozen(x)

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test37_var_upload(t):
    # FIX: leaks variables

    def func(x):

        arrays = []

        for i in range(3):
            y = dr.arange(t, 3)
            dr.make_opaque(y)
            arrays.append(y)

        del arrays
        del y
        
        return x / t(10, 10, 10)


    frozen = dr.freeze(func)

    for i in range(3):
        x = dr.arange(t, 3)
        dr.make_opaque(x)

        # with pytest.raises(RuntimeError, match = "created while recording"):
        with pytest.raises(RuntimeError):
            z = frozen(x)

        # assert dr.allclose(z, func(x))

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test38_grad_isolate(t):
    dr.set_flag(dr.JitFlag.ReuseIndices, False)

    def f(x):
        return x * 2
    def g(y):
        return y * 3

    def func(y):
        z = g(y)
        dr.backward(z)

    frozen = dr.freeze(func)

    for i in range(3):
        
        print("Reference:")
        x = dr.arange(t, 3)
        dr.make_opaque(x)
        dr.enable_grad(x)

        y = f(x)
        with dr.isolate_grad():
            func(y)

        ref = dr.grad(x)
        
        print("Frozen:")
        x = dr.arange(t, 3)
        dr.make_opaque(x)
        dr.enable_grad(x)

        y = f(x)
        dr.make_opaque(y)
        frozen(y)

        res = dr.grad(x)

        assert dr.allclose(ref, res)

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test39_isolate_grad_fwd(t):

    def f(x):
        return x*x

    def g(y):
          return y*2

    def func(x):
        with dr.isolate_grad():
            y = f(x)
            dr.forward(x)
        return y

    def frozen(x):
        y = f(x)
        dr.forward(x)
        return y
    frozen = dr.freeze(frozen)

    for i in range(3):
        x = t(i)
        dr.make_opaque(x)
        dr.enable_grad(x)

        y = func(x)
        # z = g(y)


        ref = dr.grad(y)

        
        x = t(i)
        dr.make_opaque(x)
        dr.enable_grad(x)

        y = frozen(x)
        # z = g(y)


        res = dr.grad(y)

        print(f"{res=}")
        
        assert dr.allclose(ref, res)
        
        
@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test40_grad_postponed_part(t):
    dr.set_flag(dr.JitFlag.ReuseIndices, False)

    def f(x):
        return x * x * 2
    def g(y):
        return y * y * 3

    def func(y1, y2):
        z1 = g(y1)
        z2 = g(y2)
        dr.backward(z1)

    frozen = dr.freeze(func)

    def run(i, name, func):
        print(f"{name}:")
        x1 = dr.arange(t, 3) + i
        dr.make_opaque(x1)
        dr.enable_grad(x1)
        y1 = f(x1)
        
        x2 = dr.arange(t, 3) + i
        dr.make_opaque(x2)
        dr.enable_grad(x2)
        dr.set_grad(x2, 2)
        y2 = f(x2)
        
        func(y1, y2)

        dx1 = dr.grad(x1)
        dx2_1 = dr.grad(x2)
            
        dr.set_grad(x2, 1)
        dr.backward(x2)
        dx1_2 = dr.grad(x2)

        return [dx1, dx1_2, dx2_1]

    for i in range(3):

        def isolated(y1, y2):
            with dr.isolate_grad():
                func(y1, y2)

        ref = run(i, "reference", isolated)
        res = run(i, "frozen", frozen)
        
        for ref, res in zip(ref, res):
            assert dr.allclose(ref, res)

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test40_nested(t):
    
    pkg = get_pkg(t)
    mod = sys.modules[t.__module__]

    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()
    a.value = dr.ones(t, 16)
    dr.enable_grad(a.value)

    U = mod.UInt32
    xi = t(1, 2, 8, 3, 4)
    yi = dr.reinterpret_array(U, BasePtr(a, a, a, a, a))

    def nested(self, xi, yi):
        return self.nested(xi, yi)

    def func(c, xi, yi):
        return dr.dispatch(c, nested, xi, yi)

    frozen = dr.freeze(func)

    for i in range(3):
        c = BasePtr(a, a, a, b, b)
        xi = t(1, 2, 8, 3, 4)
        yi = dr.reinterpret_array(U, BasePtr(a, a, a, a, a))
        
        with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, True):
            xref = func(c, xi, yi)
            
        assert dr.all(xref == xi + 1)
        
        c = BasePtr(a, a, a, b, b)
        xi = t(1, 2, 8, 3, 4)
        yi = dr.reinterpret_array(U, BasePtr(a, a, a, a, a))
        
        with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, True):
            xfrozen = frozen(c, xi, yi)
            
        assert dr.all(xfrozen == xref)

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test41_call_raise(t):
    
    mod = sys.modules[t.__module__]
    pkg = get_pkg(t)

    UInt = mod.UInt
    
    A, B, Base, BasePtr = pkg.A, pkg.B, pkg.Base, pkg.BasePtr
    a, b = A(), B()

    def f(x: t):
        raise RuntimeError("test")
    def g(self, x: t):
        if isinstance(self, B):
            raise RuntimeError
        return x + 1

    
    c = BasePtr(a, a, a, b, b)
    
    with pytest.raises(RuntimeError):
        dr.dispatch(c, g, t(1, 1, 2, 2, 2))

@pytest.test_arrays("float32, jit, diff, shape=(*)")
def test42_reduce_dot(t):
    def func(x, y):
        return dr.dot(x, y)

    frozen = dr.freeze(func)

    for i in range(3):
        x = dr.arange(t, 10 + i)
        y = dr.arange(t, 10 + i)

        result = frozen(x, y)
        reference = func(x, y)

        assert dr.allclose(result, reference)

    assert frozen.n_recordings == 1
        
# @pytest.test_arrays("float32, jit, diff, shape=(*)")
# def test42_raise(t):
#     # FIX: leaks variables
#
#     def func(x):
#         # return t(1, 2, 3) + x
#         return x + 1
#         
#
#
#     frozen = dr.freeze(func)
#
#     for i in range(3):
#         x = dr.arange(t, 3)
#         dr.make_opaque(x)
#
#         with pytest.raises(RuntimeError):
#             z = frozen(x)
