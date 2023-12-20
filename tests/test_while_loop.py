import drjit as dr
import pytest


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test01_simple(t, mode, optimize):
    # Test a very basic loop in a few different modes
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = dr.arange(t, 7)
        z = t(0)

        while dr.hint(i < 5, mode=mode):
            i += 1
            z = i + 4

        assert dr.all(i == t(5, 5, 5, 5, 5, 5, 6))
        assert dr.all(z == t(9, 9, 9, 9, 9, 0, 0))


@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test02_nested_loop_disallowed_config(t):
    # Can't record an evaluated loop within a symbolic recording session
    with pytest.raises(RuntimeError) as e:
        i, j = t(5), t(5)
        while i < 10:
            i += 1
            with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
                while j < 10:
                    j += 1

    err_msg='Dr.Jit is currently recording symbolic computation and cannot execute'
    assert err_msg in str(e.value.__cause__)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("version", [0, 1])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test03_change_type(t, mode, version):
    # Can't change the type of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, mode=mode):
            if dr.hint(version == 0, mode='scalar'):
                i = i + 1.1
            else:
                i = None

    err_msg = "the body of this loop changed the type of loop state variable 'i'"
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test04_change_size(t, mode):
    # Can't change the size of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5, 10)
        while dr.hint(i < 10, mode=mode):
            i = t(10, 11, 12)

    err_msg = "the body of this loop changed the size of loop state variable 'i'"
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test05_incompatible_size(t, mode):
    # Can't mix variables with incompatible sizes
    with pytest.raises(RuntimeError) as e:
        i = t(5, 6, 7)
        j = t(2, 7)
        while i < 10:
            i += 1
            j += 1

    err_msg="The body of this loop operates on arrays of size 3. Loop state variable 'j' has an incompatible size 2."
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test06_cond_err(t, mode):
    # The loop condition might raise an exception, which should be propagated without problems
    def mycond(v):
        raise Exception("oh no")
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(mycond(i), mode=mode):
            i += 1
    assert "oh no" in str(e.value.__cause__)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test07_body_err(t, mode):
    # The body might raise an exception, which should be propagated without problems
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, mode=mode):
            raise Exception("oh no")
    assert "oh no" in str(e.value.__cause__)

@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(2, *)')
@dr.syntax
def test08_dependency_structure(t, optimize, drjit_verbose, capsys):
    # Test that only requested variables are being evaluated
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        a = t(1234, 5678)
        b = dr.value_t(t)(0)
        one = dr.opaque(t, 1)

        while b == 0:
            b = dr.value_t(t)(100)
            a += one

        for i in range(2):
            assert a.x + 1 == 1236
            s = capsys.readouterr().out
            assert ('i32 1234' in s or '0x4d2' in s)
            assert not ('i32 5678' in s or '0x162e' in s)

            assert a.y + 1 == 5680
            s = capsys.readouterr().out
            assert not ('i32 1234' in s or '0x4d2' in s)
            assert ('i32 5678' in s or '0x162e' in s)


@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test09_loop_optimizations(t, optimize):
    # Test that the loop optimizes away constant loop state
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        a = t(1234)
        b = t(1234)
        index = t(0)

        a_const = 0
        b_const = 0
        it_count = 0

        while index == 0:
            a_const += int(a.state == dr.VarState.Literal)
            b_const += int(b.state == dr.VarState.Literal)
            it_count += 1

            tmp = t(1234)
            assert a.index != tmp.index
            a = tmp
            b += 0
            index += 1

        if dr.hint(not optimize, mode='scalar'):
            assert a_const == 0
            assert b_const == 0
            assert it_count == 1
            assert a.state == dr.VarState.Unevaluated
            assert b.state == dr.VarState.Unevaluated
        else:
            assert a_const == 1
            assert b_const == 1
            assert it_count == 2
            assert a.state == dr.VarState.Literal
            assert b.state == dr.VarState.Literal


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test10_scatter_v1(t, mode, optimize, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = t(0, 0)
        j = t(0) # Dummy variable to cause 2xrecording
        dr.set_label(v, 'v')

        while dr.hint(i < 10, mode=mode):
            i += 1
            j = j
            dr.scatter_add(target=v, index=0, value=1)

        assert v[0] == 19
        assert dr.all(i == 10)
        assert v[0] == 19

        # Check that the scatter operation did not make unnecessary copies
        if mode == 'symbolic':
            transcript = capsys.readouterr().out
            assert transcript.count('[direct]') == 2

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test10_scatter_v2(t, mode, optimize, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = t(0, 0)
        dr.set_label(v, 'v')

        while dr.hint(i < 10, mode=mode, exclude=[v]):
            i += 1
            dr.scatter_add(target=v, index=0, value=1)

        assert v[0] == 19
        assert dr.all(i == 10)
        assert v[0] == 19

        # Check that the scatter operation did not make unnecessary copies
        if mode == 'symbolic':
            transcript = capsys.readouterr().out
            assert transcript.count('[direct]') == 1

@pytest.mark.parametrize('mode1', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('mode2', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test11_nested_loop(mode1, mode2, t):
    n = dr.arange(t, 17)
    i, accum = t(0), t(0)

    try:
        while dr.hint(i < n, mode=mode1, label='outer'):
            j = t(0)
            while dr.hint(j < i, mode=mode2, label='inner'):
                j += 1
                accum += 1
            i += 1
        err = None
    except Exception as e:
        err = e

    if dr.hint(mode1 == 'symbolic' and mode2 == 'evaluated', mode='scalar'):
        # That combination is not allowed
        assert 'cannot execute a loop in *evaluated mode*' in str(err.__cause__)
    else:
        assert err is None
        assert dr.all(accum == (n*(n-1)) // 2)


@pytest.mark.parametrize('mode1', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('mode2', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test11_nested_loop_with_side_effect(mode1, mode2, t):
    n = dr.arange(t, 17)
    i, accum = t(0), dr.zeros(t, 17)

    try:
        while dr.hint(i < n, mode=mode1, label='outer'):
            j = t(0)
            while dr.hint(j < i, mode=mode2, label='inner'):
                j += 1
                dr.scatter_add(target=accum, index=n, value=1)
            i += 1
        err = None
    except Exception as e:
        err = e

    if dr.hint(mode1 == 'symbolic' and mode2 == 'evaluated', mode='scalar'):
        # That combination is not allowed
        assert 'cannot execute a loop in *evaluated mode*' in str(err.__cause__)
    else:
        assert err is None
        assert dr.all(accum == (n*(n-1)) // 2)


@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test12_optimize_away(t, optimize):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        # Test that the loop completely optimizes away with a 'false' loop condition
        a = t(1, 2, 3)
        ai = a.index
        j = t(1)

        while (a < 10) & (j == 0):
            a += 1

        assert (a.index == ai) == optimize

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test13_simple_diff_loop(t, optimize, mode):
    i, j = dr.int32_array_t(t)(0), t(1)
    dr.enable_grad(j)
    dr.set_grad(j, 1.1)

    while dr.hint(i < 5, mode=mode):
        j = j * 2
        i += 1

    assert dr.allclose(dr.forward_to(j), 32*1.1)

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('float,is_diff,shape=(*)')
def test13_complex_diff_loop(t, optimize, mode):
    i = dr.int32_array_t(t)(0)
    lvars = [t(0) for i in range(10)]
    dr.enable_grad(lvars[5])
    dr.set_grad(lvars[5], 1)

    while dr.hint(i < 3, mode=mode):
        lvars = [lvars[k] + lvars[k-1] for k in range(10)]
        i += 1

    dr.forward_to(lvars)
    lvars = [dr.grad(lvars[i])[0] for i in range(10)]
    assert lvars == [ 0, 0, 0, 0, 0, 1, 3, 3, 1, 0 ]


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test14_no_mutate_inputs(t, optimize, mode):
    x = t(1)
    z = t(2)

    xo, zo = dr.while_loop(
        state=(x,z),
        cond=lambda x, z: x < 10,
        body=lambda x, z: (x + 1, z)
    )

    assert xo == 10 and x == 1
    assert zo is z

    q1 = (t(3), t(4, 5))
    q2 = (t(6), t(7, 8))

    def mut(q1, q2):
        item = q2[1]
        item += 4
        return q1, q2

    q1o, q2o = dr.while_loop(
        state=(q1,q2),
        cond=lambda q1, q2: q2[1]<10,
        body=mut
    )

    assert q1o is q1
    assert q2o is not q2
    assert q2o[0] is q2[0]
    assert q2o[1] is not q2[1]

@pytest.mark.parametrize('symbolic', [True, False])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test15_loop_in_vcall(t, optimize, symbolic):
    @dr.syntax
    def f(t, x):
        count = t(0)
        while count < 10:
            x *= 2
            count += x

        return count

        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, symbolic):
            with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, symbolic):
                with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
                    with dr.scoped_set_flag(dr.JitFlag.OptimizeCalls, optimize):
                        x = dr.arange(t, 3)+2
                        indices = dr.full(t, 0, 3)

                        out = dr.switch(indices, [f], t, x)
                        assert dr.all(out == [1,2,3])

def test16_limitations():
    # Test that issues related to current limitations of the AST processing
    # are correctly reported
    with pytest.raises(SyntaxError, match="use of 'break' inside a transformed 'while' loop or 'if' statement is currently not supported."):
        @dr.syntax
        def foo(x):
            while x > 0:
                x += 1
                if x == 5:
                    break
    @dr.syntax
    def foo(x):
        while dr.hint(x > 0, mode='scalar'):
            x += 1
            if dr.hint(x == 5, mode='scalar'):
                break
            else:
                continue
