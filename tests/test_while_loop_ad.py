import drjit as dr
import pytest
import sys

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
@pytest.mark.parametrize('make_copy', [True, False])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test04_sum_loop_rev(t, mode, make_copy):
    # Test the "sum loop" optimization (max_iterations=-1) for
    # consistency against test03
    UInt32 = dr.uint32_array_t(t)
    Float = t

    y, i = Float(0), UInt32(0)
    x = dr.linspace(Float, .25, 1, 4)
    dr.enable_grad(x)
    if make_copy:
        xo = Float(x)
    else:
        xo = x

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


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,diff,shape=(*)')
@dr.syntax
def test08_bwd_in_loop(t, mode):
    UInt = dr.uint32_array_t(t)
    x = dr.opaque(t, 0, 3)
    dr.enable_grad(x)

    y = dr.gather(t, x, UInt([2, 1, 0])) # Edge that will be postponed

    i = dr.zeros(UInt, 3)
    while dr.hint(i < 2, mode=mode, exclude=[y]):
        with dr.resume_grad():
            dr.backward(2 * dr.gather(t, y, UInt([0, 1, 2])))
            i += 1

    assert dr.all(x.grad == 4)

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,diff,shape=(*)')
def test09_sum_loop_extra(t, mode):
    # Test the case, where extra loop state variables are added
    # after the differentiable ones.

    @dr.syntax
    def loop(l: list, t, mode):
        mod = sys.modules[t.__module__]
        Float = mod.Float
        UInt = mod.UInt

        y = Float(0)
        i = dr.arange(UInt, 10)
        dr.make_opaque(i)

        while dr.hint(i < 10, mode = mode, max_iterations=-1):
            y += l[1] + Float(i)
            i += 1

        return y

    # Construct an array so that the m_inputs field in LoopOp looks like this:
    # [{has_grad_in = false, ...}, {has_grad_in = true, ...}, {has_grad_in = false, ...}]
    l = [t(1), t(2)]
    dr.make_opaque(l[0])

    dr.enable_grad(l[1])

    for _ in range(10):
        y = loop(l, t, mode)

        loss = dr.mean(dr.square(y))

        dr.backward(loss)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test32_simple_loop(t, mode):
    # Testcase for simple backwards derivatives with gathers
    i = dr.uint32_array_t(t)(0)
    x = dr.ones(t, 10)
    q = dr.zeros(t)
    dr.enable_grad(x, 10)

    while dr.hint(i < 10, max_iterations=-1, mode=mode):
        q += dr.gather(t, x, i)
        i += 1

    dr.backward(q)
    assert dr.all(x.grad == [1]*10)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test33_general_bwd_scaling(t):
    # y_out = y_in * 2^N. Straight reverse-mode case that backward_simple
    # cannot handle (iterations compose multiplicatively, not additively).
    UInt = dr.uint32_array_t(t)

    x = t(1.0, 2.0, 3.0, 4.0)
    dr.enable_grad(x)
    y = t(x)
    i = UInt(0)

    while dr.hint(i < 3, mode='symbolic', max_iterations=3):
        y = y * 2
        i += 1

    dr.backward_from(y)
    assert dr.allclose(x.grad, [8, 8, 8, 8])
    assert dr.allclose(y, [8, 16, 24, 32])


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test34_general_bwd_fwd_cross_check(t):
    # Non-sum loop: each step is y_new = (y + 1) * x, so after N iterations
    # starting from y=0, y = x + x^2 + ... + x^N. Compare reverse-mode
    # through backward_general against forward-mode AD (which is independent
    # of the new code path).
    UInt = dr.uint32_array_t(t)

    def run(x_in):
        y = t(0.0)
        i = UInt(0)
        while dr.hint(i < 4, mode='symbolic', max_iterations=4):
            y = (y + 1) * x_in
            i += 1
        return y

    x_fwd = t(1.1, 1.3, 1.7, 2.1)
    dr.enable_grad(x_fwd)
    dr.set_grad(x_fwd, 1.0)
    y_fwd = run(x_fwd)
    dr.forward_to(y_fwd)
    fwd_grad = t(y_fwd.grad)

    x_bwd = t(1.1, 1.3, 1.7, 2.1)
    dr.enable_grad(x_bwd)
    y_bwd = run(x_bwd)
    dr.backward_from(y_bwd)
    assert dr.allclose(x_bwd.grad, fwd_grad)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test35_general_bwd_per_thread_iter_count(t):
    # Per-lane iteration count: lane k runs k+1 iterations of y = y * 2,
    # driven by an int state var. backward_general must back off per lane.
    UInt = dr.uint32_array_t(t)

    x = t(1.0, 2.0, 3.0, 4.0)
    dr.enable_grad(x)
    y = t(x)
    i = UInt(0)
    n = UInt(1, 2, 3, 4)

    while dr.hint(i < n, mode='symbolic', max_iterations=4):
        y = y * 2
        i += 1

    dr.backward_from(y)
    # dy/dx per lane = 2^n[k]
    assert dr.allclose(x.grad, [2.0, 4.0, 8.0, 16.0])


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test36_general_bwd_fibonacci(t):
    # Non-additive coupling between state variables (Fibonacci-style).
    # a_{k+1} = b_k,  b_{k+1} = a_k + b_k. After 3 iterations:
    #   a_3 = 1*a_0 + 2*b_0,  b_3 = 2*a_0 + 3*b_0.
    # Therefore ∂(a_3 + b_3)/∂a_0 = 3, ∂(a_3 + b_3)/∂b_0 = 5.
    UInt = dr.uint32_array_t(t)

    a = t(1.0, 2.0)
    b = t(3.0, 5.0)
    dr.enable_grad(a, b)
    a0, b0 = t(a), t(b)
    i = UInt(0)

    while dr.hint(i < 3, mode='symbolic', max_iterations=3):
        a, b = b, a + b
        i += 1

    dr.backward_from(a + b)
    assert dr.allclose(a0.grad, [3.0, 3.0])
    assert dr.allclose(b0.grad, [5.0, 5.0])


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test37_general_bwd_matches_simple(t):
    # Sum-loop pattern: both backward_simple (max_iterations=-1) and
    # backward_general (max_iterations=10) should produce the same gradient.
    UInt = dr.uint32_array_t(t)

    def run(max_it):
        x = dr.linspace(t, 0.25, 1, 4)
        dr.enable_grad(x)
        y = t(0.0)
        i = UInt(0)
        while dr.hint(i < 10, mode='symbolic', max_iterations=max_it):
            y += x ** i
            i += 1
        dr.backward_from(y)
        return x.grad

    g_simple = run(-1)
    g_general = run(10)
    assert dr.allclose(g_simple, g_general)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test38_general_bwd_compress_rejected(t):
    # compress=True + max_iterations>0 combination is rejected by ad_loop.
    UInt = dr.uint32_array_t(t)

    x = t(1.0, 2.0)
    dr.enable_grad(x)
    y = t(x)
    i = UInt(0)

    with pytest.raises(RuntimeError, match='compress'):
        while dr.hint(i < 2, mode='symbolic', max_iterations=2, compress=True):
            y = y * 2
            i += 1
        dr.backward_from(y)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test39_general_bwd_nested(t):
    # Nested differentiable loops, both with positive max_iterations. Verifies
    # that an inner symbolic loop within a trajectory-recording outer backward
    # pass composes cleanly.  a_final = x^2 * (x+1)^2 in closed form, so
    # dx/da_final = 2*x*(x+1)*(2*x+1).
    UInt = dr.uint32_array_t(t)

    x = t(2.0, 1.5)
    dr.enable_grad(x)

    a = t(1.0, 1.0)
    b = t(x)
    j = UInt(0)
    while dr.hint(j < 2, mode='symbolic', max_iterations=2):
        i = UInt(0)
        while dr.hint(i < 2, mode='symbolic', max_iterations=2):
            a = a * b
            i += 1
        b = b + 1.0
        j += 1

    dr.backward_from(a)
    xv = t(2.0, 1.5)
    expected = 2 * xv * (xv + 1) * (2 * xv + 1)
    assert dr.allclose(x.grad, expected)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@dr.syntax
def test40_general_bwd_missing_hint_rejected(t):
    # A loop with non-sum-pattern reverse-mode AD and no max_iterations hint
    # must raise a clear error rather than silently producing wrong gradients.
    UInt = dr.uint32_array_t(t)

    x = t(1.0, 2.0)
    dr.enable_grad(x)
    y = t(x)
    i = UInt(0)

    with pytest.raises(RuntimeError, match='max_iterations'):
        while dr.hint(i < 2, mode='symbolic'):
            y = y * 2
            i += 1
        dr.backward_from(y)


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test41_general_bwd_gather(t):
    # gather() from an external grad-enabled buffer inside a non-sum loop
    # (max_iterations>0). Gradients to the external buffer flow via the AD
    # graph walked by pass 2's ad_traverse(Backward); no special handling
    # in backward_general is required because gradient reaches implicit
    # inputs naturally through graph connectivity.
    #
    # Body: y = y * table[i]; i += 1
    # After 3 iters: y_3 = table[0] * table[1] * table[2]
    # Loss = sum(y_3) over 4 lanes (each lane produces the same y_3 = 24)
    # dL/dtable[k] = (lanes) * product of other table entries
    #
    # Note: the loop is written without @dr.syntax so that `table` stays a
    # closure capture (implicit input) rather than being promoted into the
    # loop state, which is what we want to exercise here.
    UInt = dr.uint32_array_t(t)

    table = t(2.0, 3.0, 4.0, 5.0)
    dr.enable_grad(table)

    y = t(1.0, 1.0, 1.0, 1.0)
    i = UInt(0)

    def cond(i, y):
        return i < 3
    def body(i, y):
        return i + 1, y * dr.gather(t, table, i)

    i, y = dr.while_loop((i, y), cond, body,
                         mode='symbolic', max_iterations=3)

    dr.backward_from(dr.sum(y))
    # dL/dtable[0] = 4 * 3*4 = 48
    # dL/dtable[1] = 4 * 2*4 = 32
    # dL/dtable[2] = 4 * 2*3 = 24
    # dL/dtable[3] = 0
    assert dr.allclose(table.grad, [48, 32, 24, 0])


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test42_general_bwd_overflow_debug_warns(t, capsys):
    # When the user underestimates max_iterations, backward_general silently
    # truncates in release mode (accepted behavior). In debug mode the
    # trajectory-array-write bounds check fires and logs a warning, so the
    # user can spot the undersized hint.
    UInt = dr.uint32_array_t(t)

    # The body must produce a gradient that actually references the trajectory
    # values (here: y_new = y*z), otherwise the trajectory write chain isn't
    # reached during AD evaluation and no warning is needed.
    y = t(1.0, 2.0)
    z = t(3.0, 4.0)
    dr.enable_grad(y, z)
    i = UInt(0)

    with dr.scoped_set_flag(dr.JitFlag.Debug, True):
        def cond(i, y, z):
            return i < 5
        def body(i, y, z):
            return i + 1, y * z, z
        # The user's condition wants 5 iterations but max_iterations=3.
        i, y, z = dr.while_loop((i, y, z), cond, body,
                                 mode='symbolic', max_iterations=3)
        dr.backward_from(y)
        dr.eval(z.grad)  # force eval so the bounds-check callback fires

    transcript = capsys.readouterr().err
    assert 'out-of-bounds write' in transcript
    assert 'array of size 3' in transcript


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test43_general_bwd_invariant_passengers(t, drjit_verbose, capsys):
    # Many differentiable "passenger" state variables that never change in the
    # body. They should not trigger max_iterations-sized trajectory storage,
    # yet gradients through them must still be correct. The verbose log is
    # used to count per-iteration trajectory writes: only the two non-invariant
    # state variables (i, y) must appear in `array_write` operations.
    UInt = dr.uint32_array_t(t)

    N_PASSENGERS = 32
    MAX_ITER = 4

    size = 3
    x = dr.linspace(t, 1.5, 2.5, size)
    dr.enable_grad(x)

    passengers = tuple(dr.full(t, float(k + 1), size) for k in range(N_PASSENGERS))
    for p in passengers:
        dr.enable_grad(p)
    passengers_in = tuple(t(p) for p in passengers)

    y = dr.ones(t, size)
    i = dr.zeros(UInt, size)

    def cond(i, y, x, *ps):
        return i < MAX_ITER

    def body(i, y, x, *ps):
        # Body uses only x and ps[0]; ps[1:] are pure ballast invariants.
        return (i + 1, y * x * ps[0], x) + ps

    capsys.readouterr()  # drop setup log
    state_out = dr.while_loop((i, y, x) + passengers, cond, body,
                              mode='symbolic', max_iterations=MAX_ITER)
    y_out = state_out[1]
    dr.backward_from(dr.sum(y_out))
    dr.eval(x.grad)

    # Only (i, y) change during the loop body. The other 1+N_PASSENGERS state
    # variables are JIT-invariant and must be excised from trajectory storage.
    # The pass-1 body is recorded twice (the symbolic loop re-records once),
    # so each non-invariant contributes 2 array_writes.
    log = capsys.readouterr().out
    assert log.count('array_write') == 2 * 2  # (i, y) * 2 passes

    # dL/dx and dL/dpassengers[0] should be nonzero; ps[1:] receive zero grad.
    assert dr.all(x.grad != 0)
    assert dr.all(passengers_in[0].grad != 0)
    for k in range(1, N_PASSENGERS):
        assert dr.all(passengers_in[k].grad == 0)


@pytest.test_arrays('float32,is_diff,shape=(*)')
def test44_general_bwd_matrix_sqrt_newton(t):
    # Newton iteration for the matrix square root:
    #   Y_{k+1} = 0.5 * (Y_k + A @ Y_k^-1),  Y_0 = A.
    # Converges quadratically to sqrt(A) for positive-definite A. A matrix
    # inverse in every step makes this a strongly nonlinear loop body. We
    # check that the loop converges to sqrt(A), and that reverse-mode AD
    # agrees with forward-mode AD along a random symmetric probe direction.
    mod = sys.modules[t.__module__]
    M4 = mod.Matrix4f
    UInt = dr.uint32_array_t(t)

    NUM_ITER = 6

    def sqrt_loop(A):
        Y = M4(A)
        i = UInt(0)
        def cond(Y, i, A):
            return i < NUM_ITER
        def body(Y, i, A):
            return (0.5 * (Y + A @ dr.rcp(Y)), i + 1, A)
        Y, _, _ = dr.while_loop((Y, i, A), cond, body,
                                mode='symbolic', max_iterations=NUM_ITER)
        return Y

    # Positive-definite A = M M^T for a well-conditioned M.
    M_base = M4([[1.0, 0.1, 0.2, 0.0],
                 [0.1, 1.2, 0.1, 0.1],
                 [0.2, 0.1, 1.3, 0.1],
                 [0.0, 0.1, 0.1, 1.1]])
    A_base = M_base @ M_base.T

    # Primal: Y @ Y should recover A.
    Y = sqrt_loop(A_base)
    dr.assert_allclose(Y @ Y, A_base, atol=1e-5)

    # Symmetric probe direction for the Jacobian check.
    dA = M4([[ 0.30, -0.10,  0.20,  0.05],
             [-0.10,  0.20,  0.00,  0.15],
             [ 0.20,  0.00,  0.10, -0.10],
             [ 0.05,  0.15, -0.10,  0.25]])
    dA = 0.5 * (dA + dA.T)

    # Reverse mode: gradient of L = trace(Y) with respect to every entry of A.
    A_bwd = M4(A_base)
    dr.enable_grad(A_bwd)
    L_bwd = dr.trace(sqrt_loop(A_bwd))
    dr.backward_from(L_bwd)
    bwd_probe = dr.trace(dr.grad(A_bwd).T @ dA)

    # Forward mode: seed A.grad = dA, compute dL/d(epsilon).
    A_fwd = M4(A_base)
    dr.enable_grad(A_fwd)
    dr.set_grad(A_fwd, dA)
    L_fwd = dr.trace(sqrt_loop(A_fwd))
    dr.forward_to(L_fwd)
    fwd_probe = dr.grad(L_fwd)

    dr.assert_allclose(bwd_probe, fwd_probe, rtol=1e-4)
