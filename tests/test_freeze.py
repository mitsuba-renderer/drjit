# Copyright (c) 2024 NVIDIA CORPORATION.
#
# All rights reserved. Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from dataclasses import dataclass
import time

import drjit as dr
import nvtx
import pytest


FLOAT_TYPES = [dr.cuda.Float32, dr.cuda.ad.Float32]
ARRAY_TYPES = [
    dr.cuda.Float32,
    dr.cuda.Array3f,
    dr.cuda.Matrix4f,
]
ARRAY_TYPES_AD = ARRAY_TYPES + [
    dr.cuda.ad.Float32,
    dr.cuda.ad.Array3f,
    dr.cuda.ad.Matrix4f,
]


def get_single_entry(x):
    tp = type(x)
    result = x
    if tp.Depth == 2:
        result = result[tp.Size - 1]
    if tp.Depth == 3:
        result = result[tp.Size - 1][tp.Size - 1]
    return result


def test01_freeze():
    Float = dr.cuda.Float32
    UInt32 = dr.cuda.UInt32
    log_level = dr.log_level()
    # dr.set_log_level(dr.LogLevel.Debug)

    @dr.kernel()
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
        assert dr.allclose(y, dr.sqr(x))
        assert dr.allclose(x_int, UInt32(x))
        assert dr.allclose(y_int, UInt32(y))
        print("------------------------------")

    dr.set_log_level(log_level)


def test02_args_to_kwargs():
    from drjit.freeze import FrozenFunction

    def my_fn(a, b, c=0.5, d=None, e="hey"):
        return a + b + c

    frozen = FrozenFunction(my_fn)

    def fill_kwargs(*args, **kwargs):
        frozen.args_to_kwargs(args, kwargs)
        return kwargs

    kwargs = fill_kwargs(0.1, 0.2, 0.3, d="d")
    expected = {
        "a": 0.1,
        "b": 0.2,
        "c": 0.3,
        "d": "d",
        "e": "hey",
    }
    assert set(kwargs.keys()) == set(expected.keys())
    for k, v in expected.items():
        assert kwargs[k] == v

    def my_fn_with_varargs(a, b, *args, c=0.5):
        return a + b + c

    def my_fn_with_kwargs(a, b, c=0.5, **kwargs):
        return a + b + c

    for fn in (my_fn_with_varargs, my_fn_with_kwargs):
        with pytest.raises(ValueError, match="Not supported.+"):
            frozen = FrozenFunction(fn)


def test03_trace():
    from drjit.freeze import FrozenFunction

    Float = dr.cuda.Float32
    UInt32 = dr.cuda.UInt32
    Array3f = dr.cuda.Array3f

    def trace(fn, *args, **kwargs):
        frozen = FrozenFunction(fn)
        # Call once to trigger tracing with the given arguments
        _ = frozen(*args, **kwargs)
        return frozen

    n = 5
    one = Float([1.0] * n)  # Not a literal
    two = dr.full(UInt32, 2, n)  # Literal
    point = Array3f(one, 2 * one, 3 * one)

    def my_fn1_raw(a, b, c=0.5, d=None, e="hey"):
        d = dr.full(Float, 5.0, dr.width(a))
        return dr.sqr(a) + b + c + d

    my_fn1 = dr.kernel(my_fn1_raw)
    for _ in range(4):
        result = my_fn1(one, 0.4)
        assert isinstance(result, Float)
        assert dr.allclose(result, dr.sqr(one) + 0.4 + 0.5 + 5.0)

    my_fn1 = dr.kernel(my_fn1_raw)
    for _ in range(4):
        result = my_fn1(one, two, c=two, d=one)
        assert isinstance(result, Float)
        assert dr.allclose(result, dr.sqr(one) + two + two + 5.0)

    # Returning Python values is somewhat supported
    @dr.kernel()
    def my_fn2(a, b):
        return a + b, "sum"

    result1, py_val = my_fn2(one, two)
    assert dr.allclose(result1, 3)
    assert py_val == "sum"

    # Since it was recorded with the first argument not being a literal,
    # we cannot switch to a literal in subsequent calls.
    with pytest.raises(ValueError, match='.+parameter "__fn_inputs.a" was a non literal-valued.+' "and is a literal.+"):
        _ = my_fn2(two, two)
    with pytest.raises(ValueError, match='.+parameter "__fn_inputs.b" was a literal-valued.+' "and is not a literal.+"):
        _ = my_fn2(one, one)

    result2, py_val = my_fn2(one + one, two)
    assert dr.allclose(result2, 4)
    assert py_val == "sum"

    def my_fn3(a: Float, b: Array3f):
        r1 = a + b
        r2 = b
        r3 = a + 1.0
        r4 = Float(4.3)
        return r1, r2, r3, r4

    frozen_fn3 = trace(my_fn3, one, point)
    # One entry of `b` is in fact `a`, so it doesn't count as a separate input
    assert len(frozen_fn3.kernels) == 1
    for kernel in frozen_fn3.kernels.values():
        assert kernel.n_inputs() == 1 + (3 - 1)
        # Three of the outputs are `b`, which is already evaluated, so it doesn't
        # count as a separate output. The literal value also doesn't count as
        # a kernel output.
        assert kernel.n_outputs() == 3 + (3 - 3) + 1 + 0


@pytest.mark.parametrize("check_results", (False, True))
def test04_runtime(check_results):
    """
    Measure total runtime with:
    1. Tracing and launching each time.
    2. Tracing one, launching the frozen kernel each time.

    We use a simple function, where the Python and tracing
    overheads are expected to dominate.
    """
    Float = dr.cuda.Float32
    n = 1024

    if check_results:
        n_iter = 1000
        n_iter_warmup = 0
    else:
        n_iter = 10000
        n_iter_warmup = 10

    def my_normal_kernel(x, y):
        z = dr.full(Float, 0.5, dr.width(x))
        result = dr.fma(dr.sqr(x), y, z)
        result = dr.sqrt(dr.abs(result) + dr.power(result, 10))
        result = dr.log(1 + result)
        return result

    my_frozen_kernel = dr.kernel(my_normal_kernel)

    for name, fn in [("normal", my_normal_kernel), ("frozen", my_frozen_kernel)]:

        for i in range(n_iter + n_iter_warmup):
            if i == n_iter_warmup:
                t0 = time.time()

            with nvtx.annotate(f"{name} {i}"):
                x = dr.arange(Float, n) + dr.opaque(Float, i)
                y = dr.arange(Float, n) - dr.opaque(Float, i)

                result = fn(x, y)

                if check_results:
                    assert isinstance(result, Float), type(result)
                    expected = x * x * y + 0.5
                    expected = dr.log(1 + dr.sqrt(dr.abs(expected) + dr.power(expected, 10)))
                    assert dr.allclose(result, expected)
                else:
                    dr.eval(result)

        if not check_results:
            with nvtx.annotate(f"{name} sync"):
                dr.sync_thread()
                elapsed = time.time() - t0
                print(f"{name}: average {1000 * elapsed / n:.2f} ms / iteration")


@pytest.mark.parametrize("tp", ARRAY_TYPES)
def test05_composite_types(tp):
    Float = dr.cuda.Float32
    UInt32 = dr.cuda.UInt32
    Array3f = dr.cuda.Array3f
    n = 37

    @dr.kernel()
    def single_input_output(x):
        # Note: for matrices, +0.5 only adds to the diagonal
        return x + 0.5

    @dr.kernel()
    def multi_inputs_single_output(x, y, dummy):
        # Note that we have an unused input.
        return x + 0.5 + y

    @dr.kernel()
    def single_input_multi_outputs(x):
        return x + 0.5, Float(0.1), x - 0.5

    @dr.kernel()
    def single_input_multi_outputs_list(x):
        return list((x + 0.5, Float(0.1), x - 0.5))

    def multi_inputs_multi_outputs_raw(x, y, z, other):
        a = x + 3.5
        b = y + z

        entry = get_single_entry(x)
        c = Array3f(entry)

        if tp.Depth == 3:
            d = x @ y + 0.5
        else:
            d = x * y + 0.5

        # Note that `other` is returned as-is and does
        # not participate in the kernels.
        return a, b, c, d, other

    multi_inputs_multi_outputs = dr.kernel(multi_inputs_multi_outputs_raw)

    @dr.kernel()
    def multi_inputs_dict_outputs(x, y, z, other):
        # TODO: check that ordering of the keys is respected
        a, b, c, d, e = multi_inputs_multi_outputs_raw(x, y, z, other)
        return {"a": a, "d": d, "c": c, "b": b, "e": e}

    # TODO: test for nested lists / dicts in outputs, even if it's just
    #       to check that they are disabled.

    for i in range(4):
        x = dr.full(tp, 1.5, n) + dr.opaque(Float, i)
        y = dr.full(tp, 0.5, n) + dr.opaque(Float, i)
        other = dr.full(tp, -0.1, n) + dr.opaque(UInt32, i)

        result = single_input_output(x)
        assert isinstance(result, tp)
        assert dr.allclose(result, x + 0.5)

        result = multi_inputs_single_output(x, y, other)
        assert isinstance(result, tp)
        assert dr.allclose(result, x + 0.5 + y), result

        for fn, ret_tp in [(single_input_multi_outputs, tuple), (single_input_multi_outputs_list, list)]:
            result = fn(x)
            assert isinstance(result, ret_tp)
            result1, result2, result3 = result
            assert isinstance(result1, tp)
            assert isinstance(result2, Float)
            assert isinstance(result3, tp)
            assert dr.allclose(result1, x + 0.5)
            assert dr.allclose(result2, 0.1)
            assert dr.allclose(result3, x - 0.5)

        for fn, ret_tp in [(multi_inputs_multi_outputs, tuple), (multi_inputs_dict_outputs, dict)]:
            # Repeated inputs, that will also appear in the output
            result = fn(x, y, x, other)
            assert isinstance(result, ret_tp)
            if ret_tp is tuple:
                result1, result2, result3, result4, result5 = result
            else:
                assert tuple(result.keys()) == ("a", "d", "c", "b", "e")
                result1 = result["a"]
                result2 = result["b"]
                result3 = result["c"]
                result4 = result["d"]
                result5 = result["e"]

            assert isinstance(result1, tp)
            assert isinstance(result2, tp)
            assert isinstance(result3, Array3f)
            assert isinstance(result4, tp)
            assert isinstance(result5, tp)
            assert dr.allclose(result1, x + 3.5), f"{result1=}, {x+3.5=}"
            assert dr.allclose(result2, y + x)

            entry = get_single_entry(x)
            for k in range(3):
                assert dr.allclose(result3[k], entry)

            if tp.Depth == 3:
                assert dr.allclose(result4, x @ y + 0.5)
            else:
                assert dr.allclose(result4, x * y + 0.5)
            assert dr.allclose(result5, other)


def test06_history_entries():
    # Frozen kernel launches should add entries in the history like all
    # other launches.
    Float = dr.cuda.Float

    def fn(x):
        # Note: for matrices, +0.5 only adds to the diagonal
        return x + 0.5

    fn_frozen = dr.kernel(fn)

    x = dr.full(Float, 1.5, 10) + dr.opaque(Float, 0.1)
    dr.eval(x)
    # Warmup launch to ensure the kernel is in cache
    dr.eval(fn(x))

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        result = fn(x)
        dr.eval(result)
        history = dr.kernel_history()
        assert len(history) == 1
        ref_entry = history[0]

    # Warmup launch to ensure the frozen function has been recorded
    dr.eval(fn_frozen(x))

    assert len(fn_frozen.frozen.kernels) == 1
    for kernel in fn_frozen.frozen.kernels.values():
        assert kernel.hash == (0, ref_entry["hash_low"], ref_entry["hash_high"])

    y = x + 0.3
    dr.eval(y)
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        result = fn_frozen(y)
        dr.eval(result)
        history = dr.kernel_history()
        assert len(history) == 1
        frozen_entry = history[0]
    assert dr.allclose(result, 1.5 + 0.1 + 0.3 + 0.5)

    assert ref_entry.keys() == frozen_entry.keys()
    for k, ref in ref_entry.items():
        if k in (
            "codegen_time",
            "backend_time",
            "execution_time",
            "input_variables",
            "output_variables",
            "all_variables",
            "pointer_variables",
        ):
            continue
        ours = frozen_entry[k]
        if k == "ir":
            ref = ref.read()
            ours = ours.read()
        if k == "output_count":
            # The frozen kernel counts as one output operation
            ours -= 1
        if k == "is_frozen":
            assert not ref
            assert ours
            continue

        assert ref == ours, (k, ref)


@pytest.mark.skip
def test07_with_reduction():
    # TODO: input is rng state (?)
    # TODO: input is wide
    assert False


@pytest.mark.skip
def test08_with_only_literal_inputs():
    # TODO: also test literals computed from input literals, which do not participate
    # in the kernel? (essentially `constexpr`).
    assert False


@pytest.mark.skip
def test09_test_inputs_no_longer_aliasing():
    """
    Input variables that alias in the same call (e.g. multiple entries of an `Array3f`
    that refer to the same variable) must keep aliasing in the same way in all
    subsequent calls.
    """
    assert False


def test10_varargs_not_supported():
    expected = r".+variable positional or keyword arguments in frozen function signature.+"
    with pytest.raises(ValueError, match=expected):

        @dr.kernel()
        def fail1(*args, a=None):
            return 0

    with pytest.raises(ValueError, match=expected):

        @dr.kernel()
        def fail2(a, b, **kwargs):
            return 0

    with pytest.raises(ValueError, match=expected):

        @dr.kernel()
        def fail3(a, b, *args, c=None, d=None, **kargs):
            return 0


TEST11_GLOBAL_VAR = 9.9


def test11_closures_not_supported():
    Float = dr.cuda.Float32
    n = 37
    x = dr.full(Float, 1.5, n) + dr.opaque(Float, 2)
    y = dr.full(Float, 0.5, n) + dr.opaque(Float, 10)

    # Supported: using functions, types / classes and builtins
    def fun1(a):
        return dr.sqr(Float(a))

    @dr.kernel()
    def fun2(a, b):
        return fun1(a) + fun1(b) + dr.abs(min(0.9, 0.1))

    for _ in range(4):
        res = fun2(x, y)
        assert dr.allclose(res, x * x + y * y + 0.1)

    # Not supported: captured variable from local scope
    my_local = dr.full(Float, 0.5, n) + dr.opaque(Float, 66)
    with pytest.raises(RuntimeError, match=r'.+nonlocal variable "my_local".+'):

        @dr.kernel()
        def fun3(a):
            return a + my_local

    # Not supported: captured variable from global scope
    with pytest.raises(RuntimeError, match=r'.+global variable "TEST11_GLOBAL_VAR".+'):

        @dr.kernel()
        def fun4(a):
            return a + TEST11_GLOBAL_VAR

    # More difficult to detect calling another function that has closures.
    my_other_local = dr.full(Float, 0.5, n) + dr.opaque(Float, 99)

    def fun5(a):
        return dr.sqr(a) + my_other_local

    with pytest.raises(RuntimeError, match=r'.+nonlocal variable "my_other_local".+'):

        @dr.kernel()
        def fun6(a, b):
            return fun5(a) + fun5(b) + min()

    # Not supported, but difficult to detect: variables nested in another module.
    # We assume that they are used responsibly as constants, e.g. `dr.inf`.


@pytest.mark.parametrize("freeze_first", (True, False))
def test12_calling_frozen_from_frozen(freeze_first):
    Float = dr.cuda.Float32
    Array3f = dr.cuda.Array3f
    n = 37
    x = dr.full(Float, 1.5, n) + dr.opaque(Float, 2)
    y = dr.full(Float, 0.5, n) + dr.opaque(Float, 10)
    dr.eval(x, y)

    @dr.kernel()
    def fun1(x):
        return dr.sqr(x)

    @dr.kernel()
    def fun2(x, y):
        return fun1(x) + fun1(y)

    # Calling a frozen function from a frozen function.
    if freeze_first:
        dr.eval(fun1(x))

    result1 = fun2(x, y)
    assert dr.allclose(result1, dr.sqr(x) + dr.sqr(y))

    if not freeze_first:
        # If the nested function hasn't been recorded yet, calling it
        # while freezing the outer function shouldn't freeze it with
        # those arguments.
        # In other words, any freezing mechanism should be completely
        # disabled while recording a frozen function.
        assert fun1.frozen.kernels is None

        # We should therefore be able to freeze `fun1` with a different
        # type of argument, and both `fun1` and `fun2` should work fine.
        result2 = fun1(Array3f(0.5, x, y))
        assert dr.allclose(result2, Array3f(0.5 * 0.5, dr.sqr(x), dr.sqr(y)))

        result3 = fun2(2 * x, 0.5 * y)
        assert dr.allclose(result3, dr.sqr(2 * x) + dr.sqr(0.5 * y))


@pytest.mark.parametrize("tp", ARRAY_TYPES)
def test13_literals_must_stay_constant(tp):
    Float = dr.cuda.Float32
    UInt32 = dr.cuda.UInt32
    Array3f = dr.cuda.Array3f
    n = 49

    def fun_raw(x, y, lit1, z):
        """
        Checks that all of the following are supported:
        - Literal in the input
        - Different literal in the output
        - Literal from the input that makes it to the output
        """
        a = x + 0.5
        b = lit1 + x
        c = y + lit1
        d = Float(6.0)  # This will be a new (constant) literal
        e = Array3f(7.0)
        return a, b, c, d, e

    fun_checked = dr.kernel(fun_raw, check=True)
    fun_unchecked = dr.kernel(fun_raw)

    x = dr.full(tp, 1.5, n) + dr.opaque(tp, 3)
    y = dr.full(tp, 0.5, n) + dr.opaque(tp, 4)
    lit1_u32 = UInt32(8)
    lit1_f32 = Float(8)
    lit1_f32_bis = Float(8)
    lit2_u32 = UInt32(3)
    lit2_f32 = Float(3.0)

    # Supported: passing the same literal over and over
    for my_lit in [lit1_f32, lit1_f32_bis]:
        for i in range(4):
            result = fun_checked(x, y, my_lit, y)
            assert isinstance(result, tuple)
            result1, result2, result3, result4, result5 = result
            assert isinstance(result1, tp)
            assert isinstance(result2, tp)
            assert isinstance(result3, tp)
            assert isinstance(result4, Float)
            assert isinstance(result5, Array3f)

            assert dr.allclose(result1, x + 0.5)
            assert result2 == lit1_f32 + x
            assert dr.allclose(result3, y + lit1_f32)
            assert result4.is_literal_()
            assert dr.all(dr.eq(result4, 6.0))
            for i in range(3):
                assert result5[i].is_literal_()
            assert dr.all_nested(dr.eq(result5, 7.0))

    # Not supported: passing a different literal value or type.
    # 1. Checked mode: will throw an exception if the wrong value or type
    #    is passed for a literal.
    expected = (
        r'.+parameter "__fn_inputs.lit1" was a DrJit array with type'
        r" <class \'drjit.cuda.Float\'>.+it has type <class \'drjit.cuda.UInt\'>.+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, y, lit1_u32, y)

    expected = (
        r'.+argument "__fn_inputs.lit1" had literal value \[8.0\] \(type:'
        r" <class \'drjit.cuda.Float\'>\).+different value \(\[3.0\],"
        r" type <class \'drjit.cuda.Float\'>\).+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, y, lit2_f32, y)

    expected = (
        r'.+parameter "__fn_inputs.lit1" was a DrJit array with type'
        r" <class \'drjit.cuda.Float\'>.+it has type <class \'drjit.cuda.UInt\'>.+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, y, lit2_u32, y)

    # 2. Unchecked mode (default): faster, but the results will be
    #    silently incorrect if another literal value is passed.
    baked_lit = lit1_u32
    for i in range(4):
        result = fun_unchecked(x, y, baked_lit, y)
        assert isinstance(result, tuple)
        result1, result2, result3, result4, result5 = result
        assert isinstance(result1, tp)
        assert isinstance(result2, tp)
        assert isinstance(result3, tp)
        assert isinstance(result4, Float)
        assert isinstance(result5, Array3f)

        assert dr.allclose(result1, x + 0.5)
        assert result2 == baked_lit + x
        assert dr.allclose(result3, y + baked_lit)
        assert result4.is_literal_()
        assert dr.all(dr.eq(result4, 6.0))
        for i in range(3):
            assert result5[i].is_literal_()
        assert dr.all_nested(dr.eq(result5, 7.0))

    new_lit = lit2_f32
    result1, result2, result3, result4, result5 = fun_unchecked(x, y, new_lit, y)
    assert dr.allclose(result1, x + 0.5)
    # We _expect_ results to be incorrect here (results use `baked_lit` instead of `new_lit`)
    assert result2 == baked_lit + x
    assert dr.allclose(result3, y + baked_lit)
    assert result4.is_literal_()
    assert dr.all(dr.eq(result4, 6.0))
    for i in range(3):
        assert result5[i].is_literal_()
    assert dr.all_nested(dr.eq(result5, 7.0))


@pytest.mark.parametrize("tp", ARRAY_TYPES)
def test14_python_values_must_stay_constant(tp):
    Float = dr.cuda.Float32
    Mask = dr.mask_t(Float)
    n = 49

    # TODO: allow returning Python values
    def fun_raw(x, p1, y, p2):
        if p2:
            return x + y + p1
        else:
            return x + y - p1

    fun_checked = dr.kernel(fun_raw, check=True)
    fun_unchecked = dr.kernel(fun_raw)

    x = dr.full(tp, 1.5, n) + dr.opaque(tp, 3)
    y = dr.full(tp, 0.5, n) + dr.opaque(tp, 4)

    # Supported: passing the same Python values over and over
    for i in range(4):
        result = fun_checked(x, 4.5, y, p2=True)
        assert isinstance(result, tp)
        assert dr.allclose(result, x + y + 4.5)

    # Not supported: passing the same literal over and over
    # 1. Checked mode: will throw an exception if the wrong value or type
    #    is passed for a literal.
    expected = (
        r".+parameter \"__fn_inputs.p1\" had literal value 4\.5.+a different" r" value \(hi, type <class 'str'>.+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, "hi", y, True)

    expected = (
        r".+parameter \"__fn_inputs.p1\" had literal value 4\.5.+a different"
        r" value \(4.50001, type <class 'float'>.+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, 4.50001, y, p2=True)

    expected = (
        r".+parameter \"__fn_inputs.p2\" had literal value True.+a different" r" value \(False, type <class 'bool'>.+"
    )
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, 4.5, y, p2=False)

    expected = r".+parameter \"__fn_inputs.p1\" was not a DrJit array.+" r"it has type {}.+".format(str(tp))
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, y, y, p2=True)

    expected = r".+parameter \"__fn_inputs.p2\" was not a DrJit array.+" r"it has type <class 'drjit.cuda.Bool'>.+"
    with pytest.raises(ValueError, match=expected):
        fun_checked(x, 4.5, y, p2=Mask(False))

    # 2. Unchecked mode (default): faster, but the results will be
    #    silently incorrect if another literal value is passed.
    expected = x + y + 4.5
    for _ in range(4):
        result = fun_unchecked(x, 4.5, y, p2=True)
        assert isinstance(result, tp)
        assert dr.allclose(result, expected)

    result = fun_unchecked(x, -99.0, y, p2=False)
    assert dr.allclose(result, expected)
    result = fun_unchecked(x, 4.5, y, p2=True)
    assert dr.allclose(result, expected)


def test15_input_sizes():
    Float = dr.cuda.Float32

    @dr.kernel()
    def fun1(x, y):
        return dr.sqr(x + y)

    @dr.kernel()
    def fun2(x, y):
        z = dr.arange(Float, dr.width(x))
        return dr.sqr(x + y) - z

    @dr.kernel()
    def fun3(x, y):
        z = dr.full(Float, 9.9, dr.width(x))
        z += x
        return dr.sqr(x + y) - z

    for n in (4, 1, 1045):
        x = dr.full(Float, 1.5, n) + dr.opaque(Float, 3)
        y = Float(0.5)

        result = fun1(x, y)
        assert isinstance(result, Float)
        assert dr.width(result == n)
        assert dr.allclose(result, dr.sqr(x + y))

        result = fun2(x, y)
        assert isinstance(result, Float)
        assert dr.width(result == n)
        assert dr.allclose(result, dr.sqr(x + y) - dr.arange(Float, n))

        result = fun3(x, y)
        assert isinstance(result, Float)
        assert dr.width(result == n)
        assert dr.allclose(result, dr.sqr(x + y) - (9.9 + x))


@pytest.mark.parametrize("tp", [dr.cuda.ad.Float32, dr.cuda.ad.Array3f, dr.cuda.ad.Matrix4f])
def test16_with_ad(tp):
    Float = dr.cuda.ad.Float32
    UInt32 = dr.cuda.ad.UInt32
    n = 37

    def fun1_raw(x, y):
        return (1 * x + 2 * y), x, UInt32(x)

    fun1_frozen = dr.kernel(fun1_raw)

    def fun2(i: int, use_frozen: bool):
        # Use the frozen kernel as part of a broader differentiable computation
        a = dr.full(tp, 1.5, n) + dr.opaque(tp, i)  # Not a literal
        b = Float(0.5)  # Literal
        dr.enable_grad(a, b)

        fn = fun1_frozen if use_frozen else fun1_raw
        # TODO: when tracing the function for the first time, we need
        # to force the evaluation of any possible future AD operation?
        # I.e. enqueue variables with grad enabled, trigger the traversal, etc.
        # But only up to the inputs, not further!
        # Maybe take a flag enable_ad: ADMode so that we don't do this wastefully
        # if it's not needed, and so that we can know whether the user will want
        # forward or backward propagation.
        result1, result2 = fn(2 * a, 3 * b)
        result = 4 * result1 + 5 * result2
        loss = dr.mean(result)
        dr.backward(loss)

        return dr.grad(a), dr.grad(b)

    if tp != Float:
        pytest.xfail("Test not implemented yet")

    # 1. Gradients disabled, but still using AD types
    for i in range(4):
        a = dr.full(tp, 1.5, n) + dr.opaque(tp, i)  # Not a literal
        b = UInt32(5)  # Literal
        result1, result2, result3 = fun1_frozen(a, b)
        assert isinstance(result1, type(a))
        assert isinstance(result2, type(a))
        assert isinstance(result3, UInt32), type(result3)
        assert dr.allclose(result1, a + 2 * b)
        assert dr.allclose(result2, a)
        assert dr.allclose(result3, UInt32(a))
        assert dr.allclose(dr.grad(result1), 0)
        assert dr.allclose(dr.grad(result2), 0)
        assert dr.allclose(dr.grad(result3), 0)

    # 2. Gradients enabled
    pytest.xfail("Not supported yet")
    for i in range(4):
        grad_a, grad_b = fun2(i, use_frozen=False)
        assert dr.allclose(grad_a, 18 / n)
        assert dr.allclose(grad_b, 24.0)

    for i in range(4):
        grad_a, grad_b = fun2(i, use_frozen=True)

        if i == 0:
            fr = fun1_frozen.frozen
            assert len(fr.kernels) == 1
            for kernel in fr.kernels.values():
                # TODO: it should maybe take the gradients as inputs too?
                assert kernel.n_inputs() == 1 + 0
                # TODO: it should probably output the new gradients?
                assert kernel.n_outputs() == 1

        assert dr.allclose(grad_a, 18 / n)
        assert dr.allclose(grad_b, 24.0)

    """
    loss = 1/n * sum(
        4 * (
            1 * 2 * a + 2 * 3 * b
        )
        + 5 * (
            2 * a
        )
    )

    loss = 1/n * n * (
          4 * 2 * a + 4 * 2 * 3 * b
        + 5 * 2 * a
    )
    loss = 18 * a[0] + 24 * b
    """


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test17_no_inputs(tp):
    UInt32 = dr.cuda.UInt32

    @dr.kernel(check=True)
    def fun(a):
        leaf_t = dr.leaf_array_t(tp)

        x = tp(dr.linspace(leaf_t, -1, 1, 10)) + a
        source = get_single_entry(x + 2 * x)
        index = dr.arange(UInt32, dr.width(source))
        active = dr.neq(index % UInt32(2), 0)

        return dr.gather(leaf_t, source, index, active)

    a = tp(0.1)
    res1 = fun(a)
    res2 = fun(a)
    res3 = fun(a)

    assert dr.allclose(res1, res2)
    assert dr.allclose(res1, res3)


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test18_with_gathers(tp):
    import numpy as np

    n = 20
    UInt32 = dr.cuda.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(tp, n))))

    def fun(x, idx):
        active = dr.neq(idx % 2, 0)
        source = get_single_entry(x)
        return dr.gather(type(source), source, idx, active=active)

    fun_frozen = dr.kernel(fun)

    # 1. Recording call
    x1 = tp(rng.uniform(low=-1, high=1, size=shape))
    idx1 = dr.arange(UInt32, n)
    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1, idx1))

    # 2. Different source as during recording
    x2 = tp(rng.uniform(low=-2, high=-1, size=shape))
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


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test19_with_scatters_pure_side_effect(tp):
    import numpy as np

    n = 20
    UInt32 = dr.cuda.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    def get_index(x):
        x = get_single_entry(x)
        if x.IsDiff:
            return x.detach_().index
        else:
            return x.index

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(tp, n))))

    def fun(x, y, idx):
        active = dr.neq(idx % 2, 0)
        dest = get_single_entry(x)
        assert dest.Depth == 1
        assert y.Depth == 1

        dr.scatter(dest, y, idx, active=active)
        # No return value, but the side-effect should still be evaluated.
        return

    fun_frozen = dr.kernel(fun, enabled=True)

    # 1. Recording call
    x1 = tp(rng.uniform(low=-1, high=1, size=shape))
    idx_x1 = get_index(x1)
    x1_copy = tp(x1)
    idx_x1_copy = get_index(x1_copy)
    x1_copy_copy = tp(x1)
    idx_x1_copy_copy = get_index(x1_copy_copy)
    y = dr.leaf_array_t(tp)(rng.uniform(low=-1, high=1, size=(n,)))
    idx1 = dr.arange(UInt32, n)

    # The `scatter` operation will write to a fresh copy of the variable in
    # case it detects there are external references to it. The new variable
    # index is replaced in-place.
    fun_frozen(x1, y, idx1)
    assert get_index(x1) > idx_x1

    fun(x1_copy, y, idx1)
    assert get_index(x1_copy) > idx_x1_copy

    assert dr.allclose(x1, x1_copy)

    # 2. Different source as during recording
    x2 = tp(rng.uniform(low=-2, high=-1, size=shape))
    idx_x2 = get_index(x2)
    x2_copy = tp(x2)
    idx_x2_copy = get_index(x2_copy)
    idx2 = idx1

    # The copy / no-copy behavior should match the non-frozen call.
    fun_frozen(x2, y, idx2)
    assert get_index(x2) > idx_x2

    fun(x2_copy, y, idx2)
    assert get_index(x2_copy) == idx_x2_copy and idx_x2_copy == idx_x2

    assert dr.allclose(x2, x2_copy)

    # 3. Different source as during recording, with a different scattering pattern
    x3 = x2
    idx_x3 = get_index(x3)
    x3_copy = tp(x3) + 0.0

    idx_x3_copy = get_index(x3_copy)
    idx3 = UInt32([i for i in reversed(range(n))])

    fun_frozen(x3, y, idx3)
    assert get_index(x3) > idx_x3

    fun(x3_copy, y, idx3)
    assert get_index(x3_copy) == idx_x3_copy and idx_x3_copy == idx_x3

    assert dr.allclose(x3, x3_copy)

    # 4. Same source as during recording
    fun_frozen(x1_copy_copy, y, idx1)
    assert dr.allclose(x1_copy_copy, x1)
    assert get_index(x1_copy_copy) > idx_x1_copy_copy

    # 5. Simple check where the scatter target is directly an input
    if dr.depth_v(tp) == 1:

        @dr.kernel()
        def fun_direct(dest):
            assert dest.Depth == 1
            dr.scatter(dest, 0.5, dr.arange(UInt32, dr.width(dest)))
            # No return value, but the side-effect should still be evaluated.
            return

        for _ in range(3):
            dest = tp(rng.uniform(low=-2, high=-1, size=shape))
            result = fun_direct(dest)
            assert result is None
            assert dr.allclose(dest, 0.5)


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test20_with_scatters_and_op(tp):
    import numpy as np

    n = 20
    UInt32 = dr.cuda.UInt32
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(tp, n))))

    def fun(x, idx):
        active = dr.neq(idx % 2, 0)
        dest = get_single_entry(x)

        # print('------- start scatter')
        # `dest` (== `x` == r2) gets copied to a new variable (r12) before creating
        # the `scatter` operation, which is not recorded in the trace (why?),
        # and then destination pointer r13 is extracted from there.
        # The destination variable `r12` doesn't show up in the kernel record.
        result = dest + 0.5
        dr.scatter(dest, result, idx, active=active)
        # print('------- done with scatter')
        return result

    fun_frozen = dr.kernel(fun, enabled=True)

    # 1. Recording call
    # print('-------------------- start result1')
    x1 = tp(rng.uniform(low=-1, high=1, size=shape))
    x1_copy = tp(x1)
    x1_copy_copy = tp(x1)
    idx1 = dr.arange(UInt32, n)
    # print(f'Before: {x1.index=}, {idx1.index=}')

    result1 = fun_frozen(x1, idx1)
    # print(f'After : {x1.index=}, {idx1.index=}')
    # print('-------------------- done with result1')
    assert dr.allclose(result1, fun(x1_copy, idx1))
    assert dr.allclose(x1, x1_copy)

    # 2. Different source as during recording
    # print('-------------------- start result2')
    # TODO: problem: during trace, the actual x1 Python variable changes
    #       from index r2 to index r12 as a result of the `scatter`.
    #       But in subsequent launches, even if we successfully create a new
    #       output buffer equivalent to r12, it doesn't get assigned to `x2`.
    x2 = tp(rng.uniform(low=-2, high=-1, size=shape))
    x2_copy = tp(x2)
    idx2 = idx1
    # print(f'Before: {x2.index=}, {idx2.index=}')

    result2 = fun_frozen(x2, idx2)
    # print(f'After : {x2.index=}, {idx2.index=}')
    # print('-------------------- done with result2')
    assert dr.allclose(result2, fun(x2_copy, idx2))
    assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = tp(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3_copy, idx3))
    assert dr.allclose(x3, x3_copy)

    # 3. Same source as during recording
    result4 = fun_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    assert dr.allclose(x1_copy_copy, x1)


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test21_with_gather_and_scatter(tp):
    # TODO: this function seems to be causing some problems with pytest,
    # something about `repr()` being called on a weird / uninitialized JIT variable.
    # This crash is triggered even when the test should otherwise pass.

    import numpy as np

    n = 20
    UInt32 = dr.uint32_array_t(dr.leaf_array_t(tp))
    # dr.set_log_level(dr.LogLevel.Debug)

    rng = np.random.default_rng(seed=1234)
    shape = tuple(reversed(dr.shape(dr.zeros(tp, n))))

    def fun(x, idx):
        active = dr.neq(idx % 2, 0)
        dest = get_single_entry(x)

        values = dr.gather(UInt32, idx, idx, active=active)
        values = type(dest)(values)
        dr.scatter(dest, values, idx, active=active)
        return dest, values

    fun_frozen = dr.kernel(fun, enabled=True)

    # 1. Recording call
    x1 = tp(rng.uniform(low=-1, high=1, size=shape))
    x1_copy = tp(x1)
    x1_copy_copy = tp(x1)
    idx1 = dr.arange(UInt32, n)

    result1 = fun_frozen(x1, idx1)
    assert dr.allclose(result1, fun(x1_copy, idx1))
    assert dr.allclose(x1, x1_copy)

    # 2. Different source as during recording
    x2 = tp(rng.uniform(low=-2, high=-1, size=shape))
    x2_copy = tp(x2)
    idx2 = idx1

    result2 = fun_frozen(x2, idx2)
    assert dr.allclose(result2, fun(x2_copy, idx2))
    assert dr.allclose(x2, x2_copy)

    x3 = x2
    x3_copy = tp(x3)
    idx3 = UInt32([i for i in reversed(range(n))])
    result3 = fun_frozen(x3, idx3)
    assert dr.allclose(result3, fun(x3_copy, idx3))
    assert dr.allclose(x3, x3_copy)

    # 3. Same source as during recording
    result4 = fun_frozen(x1_copy_copy, idx1)
    assert dr.allclose(result4, result1)
    assert dr.allclose(x1_copy_copy, x1)


@pytest.mark.parametrize(
    "Array3f",
    [
        dr.cuda.Array3f,
        dr.cuda.ad.Array3f,
    ],
)
@pytest.mark.parametrize("relative_size", ["<", "=", ">"])
def test22_gather_only_pointer_as_input(Array3f, relative_size):
    import numpy as np

    assert Array3f.Depth == 2 and Array3f.Size == 3
    Float = dr.leaf_array_t(Array3f)
    UInt32 = dr.uint32_array_t(Float)

    rng = np.random.default_rng(seed=1234)

    if relative_size == "<":

        def fun(v):
            idx = dr.arange(UInt32, 0, dr.width(v), 3)
            return Array3f(dr.gather(Float, v, idx), dr.gather(Float, v, idx + 1), dr.gather(Float, v, idx + 2))

    elif relative_size == "=":

        def fun(v):
            idx = dr.arange(UInt32, 0, dr.width(v)) // 2
            return Array3f(dr.gather(Float, v, idx), dr.gather(Float, v, idx + 1), dr.gather(Float, v, idx + 2))

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

    fun_freeze = dr.kernel(fun, check=True)

    def check_results(v, result):
        size = v.size
        if relative_size == "<":
            expected = v
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
            ).T
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
            ).T

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

        if i == 0:
            assert len(fun_freeze.frozen.kernels)
            for kernel in fun_freeze.frozen.kernels.values():
                assert kernel.original_input_size == n * 3
                if relative_size == "<":
                    assert kernel.original_launch_size == expected_width
                    assert kernel.original_launch_size_ratio == (False, 3, True)
                elif relative_size == "=":
                    assert kernel.original_launch_size == expected_width
                    assert kernel.original_launch_size_ratio == (False, 1, True)
                else:
                    assert kernel.original_launch_size == expected_width
                    assert kernel.original_launch_size_ratio == (True, 5, True)

        assert dr.width(result) == expected_width
        if relative_size == ">" and n_lanes != n:
            pytest.xfail(
                reason="The width() of the original input is baked into the kernel to compute the `active` mask during the first launch, so results are incorrect once the width changes."
            )

        check_results(v, result)


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test23_baked_width_or_literal_width(tp):
    # TODO: test cases where `dr.width(input)` or `dr.width(intermediate)`
    #       plays a role.
    # TODO: test also when the width is explicitly made opaque.
    # max_width = dr.width(v)
    # idx = dr.arange(UInt32, 0, 5 * max_width)
    # active = idx < 2 * max_width
    # return Array3f(
    #     dr.gather(Float, v, (idx) % max_width, active=active),
    #     dr.gather(Float, v, (idx + 1) % max_width, active=active),
    #     dr.gather(Float, v, (idx + 2) % max_width, active=active)
    pytest.xfail("Test not implemented yet")


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test24_multiple_kernels(tp):

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
    x = dr.full(tp, 1.5, n) + dr.opaque(tp, 0.2)
    y = dr.full(tp, 0.5, n) + dr.opaque(tp, 0.1)
    dr.eval(x, y)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        ref_results = fn(x, y, flag=True)
        dr.eval(ref_results)
        assert len(dr.kernel_history()) == 2

    fn_frozen = dr.kernel(fn)
    for _ in range(2):
        results = fn_frozen(x, y, flag=True)
        assert dr.allclose(results[0], ref_results[0])
        assert results[1] is None
        assert dr.allclose(results[2], y)
        assert dr.allclose(results[3], ref_results[3])

    for i in range(4):
        new_y = y + float(i)
        # Note: we did not enabled `check` mode, so changing this Python
        # value will not throw an exception. The new value has no influence
        # on the result even though without freezing, it would.
        # TODO: support "signature" detection and create separate frozen
        #       function instances.
        new_flag = (i % 2) == 0
        results = fn_frozen(x, new_y, flag=new_flag)
        assert dr.allclose(results[0], ref_results[0])
        assert results[1] is None
        assert dr.allclose(results[2], new_y)
        assert dr.allclose(results[3], x * 0.5 + new_y)


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test25_multiple_times_same_kernel(tp):
    # Test launching the same kernel (i.e. same hash) multiple times
    # in the same function.
    # Make sure that pre-allocated buffers are used, too.
    pytest.xfail("Test not implemented yet")


@pytest.mark.parametrize("tp", ARRAY_TYPES_AD)
def test26_modified_kwargs(tp):
    # Test a function that modifies some variables that are part of the input,
    # e.g. a nested property in a dict. Could be modified:
    # - At the Python level (e.g. entry replaced)
    # - At the DrJit level (e.g. scattered to the variable)
    pytest.xfail("Test not implemented yet")


def test27_global_flag():
    Float = dr.cuda.Float32

    @dr.kernel()
    def my_fn(a, b, c=0.5):
        return a + b + c

    # Recording
    one = Float([1.0] * 9)
    result1 = my_fn(one, one, c=0.1)
    assert dr.allclose(result1, 2.1)

    # Can't change the type of an input
    with pytest.raises(ValueError, match=".+was not a DrJit array.+"):
        _ = my_fn(one, one, c=Float(0.6))

    # Disable frozen kernels globally, now the freezing
    # logic should be completely bypassed
    with dr.scoped_set_flag(dr.JitFlag.KernelFreezing, False):
        result3 = my_fn(one, one, c=0.9)
        assert dr.allclose(result3, 2.9)


@pytest.mark.parametrize("tp", [dr.cuda.Float, dr.cuda.ad.Float])
@pytest.mark.parametrize("struct_style", ["drjit", "dataclass"])
def test28_return_types(tp, struct_style):
    Float = tp
    Array3f = dr.cuda.Array3f
    UInt32 = dr.uint32_array_t(dr.leaf_array_t(tp))

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

    # 1. Many different types
    @dr.kernel()
    def toy1(x: Float) -> Float:
        y = x**2 + dr.sin(x)
        z = x**2 + dr.cos(x)
        return (x, y, z, ToyDataclass(a=x, b=y), {"x": x, "yi": UInt32(y)}, [[[[x]]]])

    for i in range(3):
        input = Float(np.full(17, i))
        result = toy1(input)
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
    @dr.kernel(enabled=True)
    def toy2(x: Float, target: Float) -> Float:
        dr.scatter(target, 0.5 + x, dr.arange(UInt32, dr.width(x)))
        return None

    for i in range(3):
        input = Float([i] * 17)
        target = dr.empty(Float, dr.width(input))
        result = toy2(input, target)
        assert dr.allclose(target, 0.5 + input)
        assert result is None

    # 3. DRJIT_STRUCT as input and returning nested dictionaries
    @dr.kernel(enabled=True)
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


@pytest.mark.parametrize("package", [dr.cuda, dr.cuda.ad])
def test29_drjit_struct_and_matrix(package):
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

    fun_frozen = dr.kernel(fun)

    n = 7
    for i in range(4):
        x = Array4f(*(dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i) + k for k in range(4)))
        mat = Matrix4f(
            *(dr.linspace(Float, 0, 1, n) + dr.opaque(Float, i) + ii + jj for jj in range(4) for ii in range(4))
        )
        trafo = MyTransform4f()
        trafo.matrix = mat
        trafo.inverse = dr.inverse(mat)

        batch = Batch(camera=Camera(to_world=trafo), value=dr.linspace(Float, -1, 0, n) - dr.opaque(Float, i))
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


def test30_with_dataclass_in_out():
    Int32 = dr.cuda.Int32
    UInt32 = dr.cuda.UInt32
    Bool = dr.cuda.Bool

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
    frozen_acc_fn = dr.kernel(acc_fn)

    accumulation = dr.zeros(UInt32, n_rays)
    n_iter = 12
    for _ in range(n_iter):
        accumulation += frozen_acc_fn(record)

    expected = 0
    for i in range(n_iter):
        expected += 0 + 2 * (i + 1) + 1
    assert dr.all(dr.eq(accumulation, expected))


def test31_state_lambda_return_type():
    UInt32 = dr.cuda.UInt32
    some_state = UInt32([1, 2, 3])

    def fn(x):
        return 2 * x

    frozen1 = dr.kernel(fn, (lambda **_: some_state))
    with pytest.raises(TypeError, match="`state` lambda.*UInt"):
        frozen1(UInt32([0, 1, 2]))

    frozen2 = dr.kernel(fn, (lambda **_: {"my_state": some_state}))
    with pytest.raises(TypeError, match="`state` lambda.*dict"):
        frozen2(UInt32([0, 1, 2]))

    frozen3 = dr.kernel(fn, (lambda **_: (some_state,)))
    frozen3(UInt32([0, 1, 2]))


def test32_allocated_scratch_buffer():
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
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)
    UInt32 = dr.cuda.UInt32

    # Note: we are going through an object / method, otherwise the closure
    # checker would already catch the `forgotten_target_buffer` usage.
    class Model:
        def __init__(self):
            self.some_state = UInt32([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.forgotten_target_buffer = self.some_state + 1
            dr.eval(self.some_state, self.forgotten_target_buffer)

        @dr.kernel(state=(lambda self, **_: (self.some_state,)))
        def fn1(self, x):
            # Note: assuming here that the width of `forgotten_target_buffer` doesn't change
            index = dr.arange(UInt32, dr.width(x)) % dr.width(self.forgotten_target_buffer)
            dr.scatter(self.forgotten_target_buffer, x, index)

            return 2 * x

        @dr.kernel(state=(lambda self, **_: (self.some_state,)))
        def fn2(self, x):
            # Scratch buffer with width equal to a state variable
            scratch = dr.zeros(UInt32, dr.width(self.some_state))
            # Kernel 1: write to `scratch`
            index = dr.arange(UInt32, dr.width(x)) % dr.width(self.some_state)
            dr.scatter(scratch, x, index)
            # Kernel 2: use values from `scratch` directly
            result = dr.sqr(scratch)
            # We don't actually return `scratch`, its lifetime is limited to the frozen function.
            return result

        @dr.kernel(state=(lambda self, **_: (self.some_state,)))
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
        x = UInt32(list(range(i + 1)))
        assert dr.width(x) < dr.width(model.forgotten_target_buffer)

        if dr.flag(dr.JitFlag.KernelFreezing):
            expected_error = (
                f"found write-enabled pointer variable.*could not find"
                f" variable {model.forgotten_target_buffer.index}.*"
            )
            with pytest.raises(ValueError, match=expected_error):
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
        assert dr.allclose(result, dr.sqr(expected))

    # Expected usage, we should allocate the buffer and allow the launch
    for i in range(4):
        x = UInt32(list(range(i + 2)))  # i+1
        assert dr.width(x) < dr.width(model.some_state)
        result = model.fn3(x)
        assert dr.allclose(result, 2 * x)


def test33_simple_reductions():
    import numpy as np

    # dr.set_flag(dr.JitFlag.KernelFreezing, False)
    Float = dr.cuda.Float32
    n = 37

    @dr.kernel()
    def simple_sum(x):
        return dr.sum(x)

    @dr.kernel()
    def simple_product(x):
        return dr.prod(x)

    @dr.kernel()
    def simple_min(x):
        return dr.min(x)

    @dr.kernel()
    def simple_max(x):
        return dr.max(x)

    @dr.kernel()
    def sum_not_returned_wide(x):
        return dr.sum(x) + x

    @dr.kernel()
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
        check_expected(simple_product, np.product(x_np).item())
        check_expected(simple_min, np.min(x_np).item())
        check_expected(simple_max, np.max(x_np).item())

        check_expected(sum_not_returned_wide, np.sum(x_np).item() + x)
        check_expected(sum_not_returned_single, np.sum(x_np).item() + 4)


def test34_reductions_with_ad():
    # dr.set_flag(dr.JitFlag.KernelFreezing, False)
    Float = dr.cuda.ad.Float32
    n = 37

    @dr.kernel()
    def sum_with_ad(x, width_opaque):
        intermediate = 2 * x + 1
        dr.enable_grad(intermediate)

        result = dr.sqr(intermediate)

        # Unfortunately, as long as we don't support creating opaque values
        # within a frozen kernel, we can't use `dr.mean()` directly.
        loss = dr.sum(result) / width_opaque
        dr.backward(loss)
        return result, intermediate

    @dr.kernel()
    def product_with_ad(x):
        dr.enable_grad(x)
        loss = dr.prod(x)
        dr.backward_from(loss)

    for i in range(3):
        x = dr.linspace(Float, 0, 1, n + i) + dr.opaque(Float, i)
        result, intermediate = sum_with_ad(x, dr.opaque(Float, dr.width(x)))

        assert dr.grad_enabled(result)
        assert dr.grad_enabled(intermediate)
        assert not dr.grad_enabled(x)
        intermediate_expected = 2 * x + 1
        assert dr.allclose(intermediate, intermediate_expected)
        assert dr.allclose(result, dr.sqr(intermediate_expected))
        assert dr.allclose(dr.grad(result), 0)
        assert dr.allclose(dr.grad(intermediate), 2 * intermediate_expected / dr.width(x))

    for i in range(3):
        x = dr.linspace(Float, 0.1, 1, n + i) + dr.opaque(Float, i)
        result = product_with_ad(x)

        assert result is None
        assert dr.grad_enabled(x)
        with dr.suspend_grad():
            expected_grad = dr.prod(x) / x
        assert dr.allclose(dr.grad(x), expected_grad)


    # TODO: test reductions on literals (e.g. sum ones())
    # TODO: test reductions with AD enabled


# TODO: check that buffers pre-allocated for one kernel and used multiple times are properly eval-ed, and never re-allocated multiple times.
# TODO: make sure to test function inputs that alias to each other, and then no longer alias
# TODO: test creating an array in the function, scattering to it and returning (this should be the main use case of pre-allocated arrays)

if __name__ == "__main__":
    # test01_freeze()
    # test03_trace()
    # test04_runtime(False)
    # test05_composite_types(dr.cuda.Float32)
    # test10_varargs_not_supported()
    # test16_with_ad(dr.cuda.ad.Float32)
    # test18_with_gathers(dr.cuda.Float32)
    # test19_with_scatters_pure_side_effect(dr.cuda.Matrix4f)
    # test20_with_scatters_and_op(dr.cuda.Float32)
    # test21_with_gather_and_scatter(dr.cuda.Float32)
    # test21_with_gather_and_scatter(dr.cuda.ad.Float32)
    # test22_gather_only_pointer_as_input(dr.cuda.Array3f)
    # test24_multiple_kernels(dr.cuda.Float)
    # test32_allocated_scratch_buffer()
    # test33_simple_reductions()
    test34_reductions_with_ad()
