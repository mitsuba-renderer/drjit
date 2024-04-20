import drjit as dr
import pytest

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test01_simple_diff_loop_fwd(t, optimize, mode):
    # Forward-mode derivative of a simple loop with 1 differentiable state variable
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
def test02_complex_diff_loop_fwd(t, optimize, mode):
    # Forward-mode derivative of a loop with more complex variable dependences
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
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test03_sum_loop_fwd(t, mode):
    # Tests a simple sum loop that adds up a differentiable quantity
    # Forward-mode is the easy case, and the test just exists here
    # as cross-check for the reverse-mode version below
    UInt32 = dr.uint32_array_t(t)
    Float = t

    y, i = Float(0), UInt32(0)
    x = dr.linspace(Float, .25, 1, 4)
    dr.enable_grad(x)
    xo = x

    while dr.hint(i < 10, mode=mode):
        y += x**i
        i += 1

    dr.forward_from(xo)

    assert dr.allclose(y, [1.33333, 1.99805, 3.77475, 10])
    assert dr.allclose(y.grad, [1.77773, 3.95703, 12.0956, 45])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test04_sum_loop_rev(t, mode):
    # Test the "sum loop" optimization (max_iterations=-1) for
    # consistency against test03
    UInt32 = dr.uint32_array_t(t)
    Float = t

    y, i = Float(0), UInt32(0)
    x = dr.linspace(Float, .25, 1, 4)
    xo = x
    dr.enable_grad(x)

    while dr.hint(i < 10, max_iterations=-1, mode=mode):
        y += x**i
        i += 1

    assert dr.grad_enabled(y)
    dr.backward_from(y)

    assert dr.allclose(y, [1.33333, 1.99805, 3.77475, 10])
    assert dr.allclose(xo.grad, [1.77773, 3.95703, 12.0956, 45])

@pytest.mark.parametrize('variant', ['fwd', 'bwd'])
@pytest.test_arrays('float32,is_diff,shape=(*)')
def test05_evaluated_ad_kernel_launch_count(t, variant):
    # Check that the forward/reverse-mode derivative of an
    # evaluated loop launches a similar number of kernels
    UInt = dr.uint32_array_t(t)

    x = t(2,3,4,5)
    dr.enable_grad(x)
    iterations = 50

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        dr.kernel_history_clear()
        _, y, i = dr.while_loop(
            state=(x, t(1, 1, 1, 1), dr.zeros(UInt, 4)),
            cond=lambda x, y, i: i<iterations,
            body=lambda x, y, i: (x, .5*(y + x/y), i + 1),
            labels=('x', 'y', 'i'),
            mode='evaluated'
        )
        h = dr.kernel_history((dr.KernelType.JIT,))

    from math import sqrt
    assert len(h) >= iterations and len(h) < iterations + 3
    assert dr.allclose(y, (sqrt(2), sqrt(3), sqrt(4), sqrt(5)))

    if variant == 'fwd':
        x.grad = dr.opaque(t, 1)
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
            g = dr.forward_to(y)
            dr.eval(g)
            h = dr.kernel_history((dr.KernelType.JIT,))
    elif variant == 'bwd':
        y.grad = dr.opaque(t, 1)
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
            g = dr.backward_to(x)
            dr.eval(g)
            h = dr.kernel_history((dr.KernelType.JIT,))
    else:
        raise Exception('internal error')
    assert dr.allclose(g, (1/(2*sqrt(2)), 1/(2*sqrt(3)), 1/(2*sqrt(4)), 1/(2*sqrt(5))))
    assert len(h) >= iterations and len(h) < iterations + 3
    for k in h:
        assert k['operation_count'] < iterations

@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test06_gather_in_loop_fwd(t, mode, variant):
    x = dr.opaque(t, 0, 3)
    xo = x
    dr.enable_grad(x)
    i = dr.uint32_array_t(x)(0)
    y = dr.zeros(t)
    if dr.hint(variant == 0, mode='scalar'):
        while dr.hint(i < 3, mode=mode):
            y += dr.gather(t, x, i)*i
            i += 1
    else:
        while dr.hint(i < 3, mode=mode, exclude=[x]):
            y += dr.gather(t, x, i)*i
            i += 1
    xo.grad = [1, 2, 3]
    dr.forward_to(y)
    assert y.grad == 8


@pytest.mark.parametrize('variant', [0, 1])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test07_gather_in_loop_fwd_nested(t, mode, variant):
    UInt = dr.uint32_array_t(t)
    x = dr.opaque(t, 0, 3)
    x.label="x"
    xo = x
    dr.enable_grad(x)
    y = dr.zeros(t)
    j = dr.zeros(UInt, 3)
    if dr.hint(variant == 0, mode='scalar'):
        while dr.hint(j < 2, mode=mode):
            i = dr.zeros(UInt, 3)
            while dr.hint(i < 3, mode=mode):
                y += dr.gather(t, x, i)*i
                i += 1
            j += 1
    else:
        while dr.hint(j < 2, mode=mode, exclude=[x]):
            i = dr.zeros(UInt, 3)
            while dr.hint(i < 3, mode=mode, exclude=[x]):
                y += dr.gather(t, x, i)*i
                i += 1
            j += 1
    xo.grad = [1, 2, 3]
    dr.forward_to(y)
    assert dr.all(y.grad == 16)

