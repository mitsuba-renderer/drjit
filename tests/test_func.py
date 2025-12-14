import drjit as dr
import pytest


@pytest.test_arrays('float32,shape=(*),jit')
def test01_basic(t):
    """dr.func emits the function body as a separate callable in the IR."""
    def f(x):
        return x * 3

    f_wrapped = dr.func(f)

    x = t(1, 2, 3)

    # Without dr.func
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        result_plain = f(x)
        dr.eval(result_plain)
        hist_plain = dr.kernel_history((dr.KernelType.JIT,))

    assert dr.allclose(result_plain, t(3, 6, 9))
    ir_plain = hist_plain[0]['ir'].getvalue()
    if dr.backend_v(t) is dr.JitBackend.LLVM:
        assert 'call fastcc void' not in ir_plain
        assert 'define fastcc void' not in ir_plain
    elif dr.backend_v(t) is dr.JitBackend.CUDA:
        assert 'call.uni' not in ir_plain

    # With dr.func
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        result_wrapped = f_wrapped(x)
        dr.eval(result_wrapped)
        hist_wrapped = dr.kernel_history((dr.KernelType.JIT,))

    assert dr.allclose(result_wrapped, t(3, 6, 9))
    ir_wrapped = hist_wrapped[0]['ir'].getvalue()
    if dr.backend_v(t) is dr.JitBackend.LLVM:
        assert 'call fastcc void' in ir_wrapped
        assert ir_wrapped.count('define fastcc void') == 1
    elif dr.backend_v(t) is dr.JitBackend.CUDA:
        assert 'call.uni' in ir_wrapped


@pytest.test_arrays('float32,shape=(*),jit')
def test02_explicit_backend(t):
    """dr.func with an explicit backend parameter."""
    b = dr.backend_v(t)

    @dr.func(backend=b)
    def f(x):
        return x * 3

    x = t(1, 2, 3)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        result = f(x)
        dr.eval(result)
        hist = dr.kernel_history((dr.KernelType.JIT,))

    assert dr.allclose(result, t(3, 6, 9))
    assert len(hist) == 1

    ir = hist[0]['ir'].getvalue()
    if dr.backend_v(t) is dr.JitBackend.LLVM:
        assert 'call fastcc void' in ir
    elif dr.backend_v(t) is dr.JitBackend.CUDA:
        assert 'call.uni' in ir


@pytest.test_arrays('float32,shape=(*),jit')
def test03_pytree_input(t):
    """Backend detection works when the Dr.Jit array is nested in a PyTree."""
    @dr.func
    def f(data):
        return data['x'] + data['y']

    x = t(1, 2, 3)
    y = t(4, 5, 6)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        result = f({'x': x, 'y': y})
        dr.eval(result)
        hist = dr.kernel_history((dr.KernelType.JIT,))

    assert dr.allclose(result, t(5, 7, 9))
    assert len(hist) == 1

    ir = hist[0]['ir'].getvalue()
    if dr.backend_v(t) is dr.JitBackend.LLVM:
        assert 'call fastcc void' in ir
    elif dr.backend_v(t) is dr.JitBackend.CUDA:
        assert 'call.uni' in ir
