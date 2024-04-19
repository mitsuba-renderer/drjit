import drjit as dr
import pytest

@pytest.mark.parametrize('variant', [1, 2, 3])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float, is_diff, shape=(*)')
@dr.syntax
def test01_backward_inside(t, variant, mode):
    # Test that we can run dr.backward from *inside* an 'if' statement, and
    # that this propagates correctly to a prior computation graph

    i = dr.arange(dr.uint32_array_t(t), 5)
    x = dr.arange(t, 5)
    dr.enable_grad(x)
    y = x * 2
    yo = y

    if dr.hint(i < 2, mode=mode):
        if dr.hint(variant & 1, mode='scalar'):
            z = y * 2
            dr.backward_from(z)
    else:
        if dr.hint(variant & 2, mode='scalar'):
            z = y * 3
            dr.backward_from(z)

    b1 = variant & 1
    b2 = (variant & 2) >> 1

    assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2])
    assert yo is y


@pytest.mark.parametrize('variant', [1, 2, 3])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float, is_diff, shape=(*)')
@dr.syntax
def test02_forward_inside(t,variant, mode):
    # Test that we can run dr.forward from *inside* an 'if' statement, and
    # that this propagates correctly from a prior computation graph

    i = dr.arange(dr.uint32_array_t(t), 5)
    x = dr.arange(t, 5)
    dr.enable_grad(x)
    y = x * 2
    x.grad = 1
    dr.forward_from(x, dr.ADFlag.ClearEdges)
    z = t(0)
    if dr.hint(i < 2, mode=mode):
        if dr.hint(variant & 1, mode='scalar'):
            z = dr.forward_to(y*2)
    else:
        if dr.hint(variant & 2, mode='scalar'):
            z = dr.forward_to(y*3)

    b1 = variant & 1
    b2 = (variant & 2) >> 1

    assert dr.all(z == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2])


@pytest.mark.parametrize('variant', [1, 2, 3])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float, is_diff, shape=(*)')
@dr.syntax
def test03_backward_outside(t, variant, mode):
    # Test that we can run dr.backward from *outside* an 'if'
    # statement, and that the resulting derivatives propagate
    # correctly through a prior computation graph

    i = dr.arange(dr.uint32_array_t(t), 5)
    x = dr.arange(t, 5)
    dr.enable_grad(x)
    y = x * 2
    yo = y
    z = t(0)

    if dr.hint(i < 2, mode=mode):
        if dr.hint(variant & 1, mode='scalar'):
            z = y * 2
    else:
        if dr.hint(variant & 2, mode='scalar'):
            z = y * 3

    b1 = variant & 1
    b2 = (variant & 2) >> 1
    dr.backward_from(z)

    assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2])
    assert yo is y


@pytest.mark.parametrize('variant', [1, 2, 3])
@pytest.mark.parametrize('same_size', [True, False])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('source_evaluated', [True, False])
@pytest.test_arrays('float, is_diff, shape=(*)')
@dr.syntax
def test04_backward_gather_inside(t, variant, mode, same_size, source_evaluated):
    # Variant of test01 where the differentiable read is replaced by a
    # differentiable gather

    i = dr.arange(dr.uint32_array_t(t), 5)
    x = dr.arange(t, 5 if same_size else 6)
    dr.enable_grad(x)
    y = x * 2
    if source_evaluated:
        dr.eval(y)
    z = t(0)

    if dr.hint(i < 2, mode=mode):
        if dr.hint(variant & 1, mode='scalar'):
            z = dr.gather(t, y, i) * 2
            dr.backward_from(z)
    else:
        if dr.hint(variant & 2, mode='scalar'):
            z = dr.gather(t, y, i) * 3
            dr.backward_from(z)

    b1 = variant & 1
    b2 = (variant & 2) >> 1

    if same_size:
        assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2])
    else:
        assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2, 0])


@pytest.mark.parametrize('variant', [1, 2, 3])
@pytest.mark.parametrize('same_size', [True, False])
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('source_evaluated', [True, False])
@pytest.test_arrays('float, is_diff, shape=(*)')
@dr.syntax
def test05_backward_gather_outside(t, variant, mode, same_size, source_evaluated):
    # Variant of test02 where the differentiable read is replaced by a
    # differentiable gather

    i = dr.arange(dr.uint32_array_t(t), 5)
    x = dr.arange(t, 5 if same_size else 6)
    dr.enable_grad(x)
    y = x * 2
    if source_evaluated:
        dr.eval(y)
    z = t(0)

    if dr.hint(i < 2, mode=mode):
        if dr.hint(variant & 1, mode='scalar'):
            z = dr.gather(t, y, i) * 2
    else:
        if dr.hint(variant & 2, mode='scalar'):
            z = dr.gather(t, y, i) * 3

    dr.backward_from(z)

    b1 = variant & 1
    b2 = (variant & 2) >> 1

    if same_size:
        assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2])
    else:
        assert dr.all(x.grad == [4*b1, 4*b1, 6*b2, 6*b2, 6*b2, 0])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float, is_diff, shape=(*)')
def test06_ad_bwd(t, mode):
    # Test that we can backpropagate through a sequence of nested 'if' statements
    @dr.syntax
    def f(x, mode):
        if dr.hint(x < 5, mode=mode):
            y = 10*x
        else:
            if dr.hint(x < 7, mode=mode):
                y = 100*x
            else:
                y = 1000*x
        return y

    x = dr.arange(t, 10)
    dr.enable_grad(x)

    y = f(x, mode)
    dr.backward_from(y)
    assert dr.all(dr.grad(x) == [10, 10, 10, 10, 10, 100, 100, 1000, 1000, 1000])


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test07_ad_bwd_implicit_dep(t, mode):
    # Identical to the above, but for reverse mode
    y = t(1)
    dr.enable_grad(y)
    dr.set_grad(y, 1)

    x = dr.arange(t, 10)

    if dr.hint(x < 5, exclude=[y]):
        z = x*y
    else:
        z = x-y

    dr.backward_from(z)
    assert dr.all(dr.grad(y) == 6)


@pytest.test_arrays('float32,is_diff,shape=(*)')
@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize('variant', [0, 1])
@dr.syntax
def test07_diff_gather_nest_bwd(t, mode, variant):
    # Test that we can backpropagate gathers a sequence of nested 'if' statements
    idx = dr.arange(dr.uint32_array_t(t), 10)
    y = dr.zeros(t, 11)
    z = dr.zeros(t, 10)
    dr.enable_grad(y)

    if dr.hint(variant == 0, mode='scalar'):
        if dr.hint(idx > 3, label='outer', mode=mode, exclude=[y]):
            if dr.hint(idx < 8, label='inner', mode=mode, exclude=[y]):
                z = dr.gather(t, y, idx)
    else:
        if dr.hint(idx > 3, label='outer', mode=mode):
            if dr.hint(idx < 8, label='inner', mode=mode):
                z = dr.gather(t, y, idx)

    dr.backward_from(z)
    assert dr.sum(y.grad) == 4


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
def test08_ad_fwd(t, mode):
    # Test that we can forward-propagate through a series of 'if' statements
    @dr.syntax
    def f(x, mode):
        if dr.hint(x < 5, mode=mode):
            y = 10*x
        else:
            if dr.hint(x < 7, mode=mode):
                y = 100*x
            else:
                y = 1000*x
        return x, y

    x = dr.arange(t, 10)
    dr.enable_grad(x)
    xi = x

    xo, yo = f(x, mode)
    assert dr.all(xo == dr.arange(t, 10))
    assert dr.all(yo == [0, 10, 20, 30, 40, 500, 600, 7000, 8000, 9000])
    dr.forward_from(x, flags=0)
    assert dr.all(dr.grad(xo) == dr.full(t, 1, 10))
    assert dr.all(dr.grad(yo) == [10, 10, 10, 10, 10, 100, 100, 1000, 1000, 1000])
    assert xi is xo


@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test09_ad_fwd_implicit_dep(t, mode):
    # Ensure that implicit dependencies ('y') are correctly tracked
    y = t(1)
    dr.enable_grad(y)
    dr.set_grad(y, 1)

    x = dr.arange(t, 10)

    if dr.hint(x < 5, exclude=[y]):
        z = x*y
    else:
        z = x-y

    dr.forward_to(z)
    assert dr.all(dr.grad(z) == [0, 1, 2, 3, 4, -1, -1, -1, -1, -1])


