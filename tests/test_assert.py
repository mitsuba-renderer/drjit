import os

import drjit as dr
import pytest

def test01_assert_bool():
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        dr.assert_true(True)
        dr.assert_false(False)

        with pytest.raises(AssertionError, match='Assertion failure!'):
            dr.assert_true(False)
        with pytest.raises(AssertionError, match='Assertion failure!'):
            dr.assert_false(True)


@pytest.test_arrays('shape=(*), uint32, jit')
def test02_assert_vec(t):
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        i = t(1, 2, 3, 4, 5, 6)

        # Unanimous
        dr.assert_true(i > 0)
        dr.assert_false(i > 10)

        # Partial
        with pytest.raises(AssertionError, match='Assertion failure!'):
            dr.assert_true(i >= 4)
        with pytest.raises(AssertionError, match='Assertion failure!'):
            dr.assert_false(i >= 4)


@pytest.test_arrays('shape=(*), uint32, jit')
@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
def test03_assert_true_loop(t, mode, capsys):
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        i = t(2, 3, 4, 5)

        def body_unanimous(i):
            dr.print("{}", i, active=i > 10)
            dr.assert_true(i > 0)
            return (i + 1,)
        i = dr.while_loop(
            (i,),
            lambda i: i<8,
            body_unanimous,
            mode=mode
        )
        dr.eval(i)

        i = t(2, 3, 4, 5)
        def body_partial_true(i):
            dr.assert_true(i >= 4)
            return (i + 1,)

        if mode == 'evaluated':
            with pytest.raises(RuntimeError) as e:
                i = dr.while_loop(
                    (i,),
                    lambda i: i<8,
                    body_partial_true,
                    mode=mode
                )
                dr.eval(i)
                assert "Assertion failure!" in str(e.value.__cause__)
        else:
            i = dr.while_loop(
                (i,),
                lambda i: i<8,
                body_partial_true,
                mode=mode
            )
            dr.eval(i)
            captured = capsys.readouterr()
            assert "Assertion failure!" in str(captured.err)


@pytest.test_arrays('shape=(*), uint32, jit')
@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
def test04_assert_false_loop(t, mode, capsys):
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        i = t(2, 3, 4, 5)

        def body_unanimous(i):
            dr.print("{}", i, active=i > 10)
            dr.assert_false(i > 10)
            return (i + 1,)
        i = dr.while_loop(
            (i,),
            lambda i: i<8,
            body_unanimous,
            mode=mode
        )
        dr.eval(i)

        i = t(2, 3, 4, 5)
        def body_partial_false(i):
            dr.assert_false(i >= 4)
            return (i + 1,)

        if mode == 'evaluated':
            with pytest.raises(RuntimeError) as e:
                i = dr.while_loop(
                    (i,),
                    lambda i: i<8,
                    body_partial_false,
                    mode=mode
                )
                dr.eval(i)
                assert "Assertion failure!" in str(e.value.__cause__)
        else:
            i = dr.while_loop(
                (i,),
                lambda i: i<8,
                body_partial_false,
                mode=mode
            )
            dr.eval(i)
            captured = capsys.readouterr()
            assert "Assertion failure!" in str(captured.err)


def test05_debug_trace_func():
    """
    This could lead to `std::bad_cast` or a PyTest internal error
    when DrJit's `trace_func()` did not correctly handle `frame.f_lineno == None`.
    """
    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        for _ in os.walk("."):
            pass
