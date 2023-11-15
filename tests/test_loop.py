import drjit as dr
import pytest


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test01_simple(t, method, optimize):
    # Test a very basic loop in a few different modes
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = dr.arange(t, 7)
        z = t(0)

        while dr.hint(i < 5, method=method):
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


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("version", [0, 1])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test03_change_type(t, method, version):
    # Can't change the type of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, method=method):
            if dr.hint(version == 0, method='scalar'):
                i = i + 1.1
            else:
                i = None

    err_msg = "the body of this loop changed the type of loop state variable 'i'"
    assert err_msg in str(e.value)


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test04_change_size(t, method):
    # Can't change the size of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5, 10)
        while dr.hint(i < 10, method=method):
            i = t(10, 11, 12)

    err_msg = "the body of this loop changed the size of loop state variable 'i'"
    assert err_msg in str(e.value)


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test05_incompatible_size(t, method):
    # Can't mix variables with incompatible sizes
    with pytest.raises(RuntimeError) as e:
        i = t(5, 6, 7)
        j = t(2, 7)
        while i < 10:
            i += 1
            j += 1

    err_msg="The body of this loop operates on arrays of size 3. Loop state variable 'j' has an incompatible size 2."
    assert err_msg in str(e.value)


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test06_cond_err(t, method):
    # The loop condition might raise an exception, which should be propagated without problems
    def mycond(v):
        raise Exception("oh no")
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(mycond(i), method=method):
            i += 1
    assert "oh no" in str(e.value.__cause__)


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test07_body_err(t, method):
    # The body might raise an exception, which should be propagated without problems
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, method=method):
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

        if dr.hint(not optimize, method='scalar'):
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


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test10_scatter_v1(t, method, optimize):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = dr.zeros(t, 1)
        dr.set_label(v, 'v')

        while dr.hint(i < 10, method=method):
            i += 1
            dr.scatter_add(target=v, index=0, value=1)

        assert v[0] == 19
        assert dr.all(i == 10)
        assert v[0] == 19


@pytest.mark.parametrize('method', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test10_scatter_v2(t, method, optimize):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = dr.zeros(t, 1)
        dr.set_label(v, 'v')

        while dr.hint(i < 10, method=method, exclude=[v]):
            i += 1
            dr.scatter_add(target=v, index=0, value=1)

        assert v[0] == 19
        assert dr.all(i == 10)
        assert v[0] == 19

@pytest.mark.parametrize('method1', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('method2', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test11_nested_loop(method1, method2, t):
    n = dr.arange(t, 17)
    i, accum = t(0), t(0)

    try:
        while dr.hint(i < n, method=method1, label='outer'):
            j = t(0)
            while dr.hint(j < i, method=method2, label='inner'):
                j += 1
                accum += 1
            i += 1
        err = None
    except Exception as e:
        err = e

    if dr.hint(method1 == 'symbolic' and method2 == 'evaluated', method='scalar'):
        # That combination is not allowed
        assert 'cannot execute a loop in *evaluated mode*' in str(err.__cause__)
    else:
        assert err is None
        assert dr.all(accum == (n*(n-1)) // 2)


@pytest.mark.parametrize('method1', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('method2', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test11_nested_loop_with_side_effect(method1, method2, t):
    n = dr.arange(t, 17)
    i, accum = t(0), dr.zeros(t, 17)

    try:
        while dr.hint(i < n, method=method1, label='outer'):
            j = t(0)
            while dr.hint(j < i, method=method2, label='inner'):
                j += 1
                dr.scatter_add(target=accum, index=n, value=1)
            i += 1
        err = None
    except Exception as e:
        err = e

    if dr.hint(method1 == 'symbolic' and method2 == 'evaluated', method='scalar'):
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