import drjit as dr
import pytest
import sys


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


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test02_mutate(t, mode):
    # Test that in-place mutation of loop state variables is correctly
    # tracked and propagated to the inputs of dr.while_loop
    @dr.syntax
    def test_nomut(x, mode):
        while dr.hint(x < 10, mode=mode):
            x = x + 1
        return x

    @dr.syntax
    def test_mut(x, mode):
        while dr.hint(x < 10, mode=mode):
            x += 1
        return x

    x = t(1)
    y = test_nomut(x, mode)
    dr.eval(x, y)
    assert x == 1 and y == 10

    x = t(1)
    y = test_mut(x, mode)
    dr.eval(x, y)
    assert x == 10 and y == 10


@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test03_nested_loop_warn_config(t, capsys):
    # Can't record an evaluated loop within a symbolic recording session
    i, j = t(5), t(5)
    while i < 10:
        i += 1
        with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
            while j < 10:
                j += 1

    transcript = capsys.readouterr().err
    assert transcript.count('forcing conditional statement to symbolic mode') == 1


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("version", [0, 1])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test04_change_type(t, mode, version):
    # Can't change the type of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, mode=mode):
            if dr.hint(version == 0, mode='scalar'):
                i = i + 1.1
            else:
                i = None

    err_msg = "the type of state variable 'i' changed from"
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test05_change_size(t, mode):
    # Can't change the size of a variable
    with pytest.raises(RuntimeError) as e:
        i = t(5, 10)
        while dr.hint(i < 10, mode=mode):
            i = t(10, 11, 12)

    err_msg = "the size of state variable 'i' of type "
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test06_incompatible_size(t, mode):
    # Can't mix variables with incompatible sizes
    with pytest.raises(RuntimeError) as e:
        i = t(5, 6, 7)
        j = t(2, 7)
        while dr.hint(i < 10, mode=mode):
            i += 1
            j += 1

    err_msg = "this operation processes arrays of size 3, while state variable 'j' has an incompatible size 2."
    assert err_msg in str(e.value)


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test07_uninitialized_before(t, mode):
    # Loop state must be fully defined before loop
    with pytest.raises(RuntimeError, match=r"state variable 'j' of type .* is uninitialized"):
        i = t(5, 6, 7)
        j = t()
        while dr.hint(i < 10, mode=mode):
            i += 1
            j += 1


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test08_uninitialized_after(t, mode):
    # Loop state must be fully defined after each loop iteration
    with pytest.raises(RuntimeError, match=r"state variable 'j' of type .* is uninitialized"):
        i = t(5, 6)
        j = t(7, 8)
        while dr.hint(i < 10, mode=mode):
            i += 1
            j = t()


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test09_cond_err(t, mode):
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
def test10_body_err(t, mode):
    # The body might raise an exception, which should be propagated without problems
    with pytest.raises(RuntimeError) as e:
        i = t(5)
        while dr.hint(i < 10, mode=mode):
            raise Exception("oh no")
    assert "oh no" in str(e.value.__cause__)


@pytest.mark.parametrize('optimize', [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(2, *)')
@dr.syntax
def test11_dependency_structure(t, optimize, drjit_verbose, capsys):
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
def test12_loop_optimizations(t, optimize):
    # Test that the loop optimizes away constant loop state
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        a = t(1234)
        b = t(1234)
        index = t(0)

        a_const = 0
        b_const = 0
        it_count = 0

        while dr.hint(index == 0, strict=False):
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
            assert b.state == dr.VarState.Literal
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
def test13_scatter_v1(t, mode, optimize, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = t(0, 0)
        j = t(0) # Dummy variable to cause 2xrecording
        dr.set_label(v, 'v')

        while dr.hint(i < 10, mode=mode):
            i += 1
            j = j
            dr.scatter_add(target=v, index=0, value=1, mode=dr.ReduceMode.Local)

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
def test14_scatter_v2(t, mode, optimize, drjit_verbose, capsys):
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = t(0, 1)
        v = t(0, 0)
        dr.set_label(v, 'v')

        while dr.hint(i < 10, mode=mode, exclude=[v]):
            i += 1
            dr.scatter_add(target=v, index=0, value=1, mode=dr.ReduceMode.Local)

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
def test15_nested_loop(mode1, mode2, t):
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
def test16_nested_loop_with_side_effect(mode1, mode2, t):
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
def test17_optimize_away(t, optimize):
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
def test18_no_mutate_inputs(t, optimize, mode):
    x = dr.opaque(t, 1)
    z = dr.opaque(t, 2)

    xo, zo = dr.while_loop(
        state=(x,z),
        cond=lambda x, z: x < 10,
        body=lambda x, z: (x + 1, z),
        mode=mode,
        labels=("x", "z")
    )

    assert xo is not x
    assert xo == 10
    assert x == 1
    assert zo is z

    q1 = (dr.opaque(t, 3), t(4, 5))
    q2 = (dr.opaque(t, 6), t(7, 8))

    def mut(q1, q2):
        item = q2[1]
        item += 4
        return q1, q2

    q1o, q2o = dr.while_loop(
        state=(q1,q2),
        cond=lambda q1, q2: q2[1]<10,
        body=mut,
        mode=mode
    )

    assert q1o is q1
    assert q2o is q2
    assert q2o[0] is q2[0]
    assert q2o[1] is q2[1]
    assert dr.allclose(q2, ([6], [11, 12]))


@pytest.mark.parametrize('symbolic', [True, False])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test19_loop_in_vcall(t, optimize, symbolic):
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
                    indices = dr.opaque(t, 0, 3)

                    out = dr.switch(indices, [f, f], t, x)
                    assert dr.all(out == [12,18,24])


def test20_limitations():
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


@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
@pytest.mark.parametrize("compress", [True, False])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
@dr.syntax
def test21_compress(t, mode, compress):
    # Test loop compression on Collatz sequence
    state = dr.arange(t, 10000) + 1
    it_count = dr.zeros(t, 10000)

    while dr.hint(state != 1, mode=mode, compress=compress):
        state = dr.select(
            state & 1 == 0,
            state // 2,
            3*state + 1
        )
        it_count += 1

    assert dr.sum(it_count) == 849666


@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test22_loop_with_fork(t):
    # Test a more complex example of a random walk involving particles that can
    # split. These splits are handled by scattering values to a separate buffer
    # and processing this buffer in turn until no particles are left.

    @dr.syntax
    def f(t):
        m = sys.modules[t.__module__]
        UInt32 = t
        Bool = m.Bool
        Float = m.Float
        Array2f = m.Array2f
        PCG32 = m.PCG32
        size = 1000
        sizes = []

        class Particle:
            DRJIT_STRUCT = {
                'index' : UInt32,
                'weight': Float
            }

            def __init__(self, index=None, weight=None):
                self.index = index
                self.weight = weight


        # Create an initial set of particles with weight 1
        state = Particle(
            index=dr.arange(UInt32, size),
            weight=dr.full(Float, 1, size)
        )

        while size > 0:
            # Initialize the random number generator for each particle
            rng = PCG32()
            rng.seed(initstate=state.index)

            # Create an 1-element array that will be used as an atomic index into a queue
            queue_index = UInt32(0)

            # Preallocate memory for the queue. The necessary amount of memory is
            # task-dependent (how many splits there are)
            queue_size = int(1.1*size)
            queue = dr.empty(dtype=Particle, shape=queue_size)

            # Create an opaque variable representing the number 'loop_state'.
            # This keeps this changing value from being baked into the program,
            # which is needed for proper kernel caching
            queue_size_o = dr.opaque(UInt32, queue_size)

            # Initially, all particles are active
            active = dr.full(Bool, True, size)

            while active:
                # Some update process that modifies the particle weight
                weight_factor = dr.fma(rng.next_float32(), 0.125, 1-0.125/2)
                state.weight *= weight_factor

                # Russian roulette
                if state.weight < .5:
                    if rng.next_float32() > state.weight:
                        active = Bool(False)

                # Random split if the energy exceeds a given threshold
                if state.weight > 1.5:
                    # Spawn a new particle to be handled in a future iteration of
                    # the parent loop, so that current and new particle each have
                    # 50% of the original weight. For this process to be consistent
                    # across runs, set an ID based on this particle's random number
                    # generator state

                    new_state = Particle(
                        index=rng.next_uint32(),
                        weight=state.weight / 2
                    )

                    state.weight /= 2

                    # Atomically reserve a slot in 'queue'
                    slot = dr.scatter_inc(queue_index, index=0)

                    # Be careful not to write beyond the end of the queue
                    valid = slot < queue_size_o

                    # Write 'new_state' into the reserved slot
                    dr.scatter(target=queue, value=new_state, index=slot, active=valid)

            next_size = queue_index[0]

            if next_size > queue_size:
                print('Warning: Preallocated queue was too small: tried to store '
                      f'{next_size} elements in a queue of size {queue_size}')
                next_size = queue_size

            state = dr.reshape(type(state), value=queue, shape=next_size, shrink=True)
            size = next_size
            sizes.append(size)
        return sizes

    sizes = f(t)
    assert sizes == [706, 269, 100, 28, 9, 1, 0]


@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test23_dr_syntax_default_args(t):
    # This test isn't really specific to while loops and just checks
    # that @dr.syntax preserves the contents of function default arguments
    @dr.syntax
    def f(t, limit=10):
        i = t(0, 0)
        while i < limit:
            i += 1
        return i
    assert dr.all(f(t) == [10, 10])


def test24_dr_syntax_confusion():
    with pytest.raises(RuntimeError, match='wrong order'):
        @dr.syntax
        @dr.wrap(source='drjit', target='torch')
        def f2(x):
            return x


@pytest.test_arrays('float32,is_diff,shape=(*)')
@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
@pytest.mark.parametrize('variant', [0, 1])
def test25_preserve_unchanged(t, mode, variant):
    # Check that unchanged variables (including
    # differentiable ones) are simply passed through
    a = t(1, 1)
    b = t(2, 2)
    dr.enable_grad(b)
    assert dr.grad_enabled(b)
    ai = (a.index, a.index_ad)
    bi = (b.index, b.index_ad)

    if variant == 1:
        c = 1
    else:
        c = t(3, 3)

    ao, bo, co = dr.while_loop(
        state=(a, b, c),
        cond=lambda x, y, z: x == 0,
        body=lambda x, y, z: (x, y, z+1),
        strict=False)
    ai2 = (ao.index, ao.index_ad)
    bi2 = (bo.index, bo.index_ad)

    assert not dr.grad_enabled(ao)
    assert dr.grad_enabled(bo)
    assert ai == ai2
    assert bi == bi2


@pytest.test_arrays('uint32,jit,shape=(*)')
@dr.syntax
def test26_gather(t):
    # Test that gather operations within loops work
    source = t(1, 2, 3)
    i = t(0, 1)
    x = t(0)

    while i < 3:
        x += dr.gather(t, source=source, index=i)
        i += 1

    assert dr.all(x == [6, 5])


@pytest.test_arrays('uint32,jit,shape=(*)')
@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
@dr.syntax
def test27_partial_eval(t, mode):
    # Test that we can use loop outputs that have already been evaluated
    idx = dr.zeros(t)
    val = dr.zeros(t)

    while dr.hint(idx < 3, mode=mode):
        idx += 1
        val += 2 + idx # `val` is dependent on `idx` at each iteration

    assert idx == 3 # Only evaluate loop state partially
    assert val + idx == 15 # Re-use partially evaluated state

@pytest.test_arrays('uint32,jit,shape=(*)')
def test28_loop_state_aliasing(t):
    # Test that we can add a variable to a loop twice

    @dr.syntax
    def loop(t, x, y: t, n = 10):

        i = t(0)
        while dr.hint(i < n):
            # Somewhat complicated gather that cannot be elided
            y += dr.gather(t, x[0], y)
            i += 1

        return y

    x = dr.arange(t, 100)
    y = dr.arange(t, 5)

    dr.make_opaque(x, y)

    y = loop(t, [x, x], y)


@pytest.test_arrays('float32,diff,shape=(*)')
@pytest.mark.parametrize('mode', ['symbolic', 'evaluated'])
def test29_preserve_differentiability_suspend(t, mode):
    x = t(0, 0)
    y = t(1, 2)
    dr.enable_grad(x, y)
    y_id = y.index_ad

    with dr.suspend_grad():
        def cond_fn(x, _):
            return x < 10

        def body_fn(x, y):
            return x + y, y

        x, y = dr.while_loop(
            state=(x, y),
            cond=cond_fn,
            labels=('x', 'y'),
            body=body_fn,
            mode=mode
        )

    assert not dr.grad_enabled(x)
    assert dr.grad_enabled(y)
    assert y.index_ad == y_id


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.mark.parametrize("variant", [0, 1])
@pytest.test_arrays('uint32,is_jit,is_tensor')
@dr.syntax
def test30_tensor_loop(t, mode, optimize, variant):
    # Let's use tensors as loop condition and variables
    with dr.scoped_set_flag(dr.JitFlag.OptimizeLoops, optimize):
        i = dr.arange(t, 7)
        z = t(0)

        while dr.hint(i < 5, mode=mode):
            i += 1
            if variant == 0:
                z = i + 4
            else:
                z += 1

        assert dr.all(i == t([5, 5, 5, 5, 5, 5, 6]))

    if variant == 0:
        assert dr.all(z == t([9, 9, 9, 9, 9, 0, 0]))
    else:
        assert dr.all(z == t([5, 4, 3, 2, 1, 0, 0]))
