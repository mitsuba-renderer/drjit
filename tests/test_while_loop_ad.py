import drjit as dr
import pytest

@pytest.mark.parametrize('mode', ['evaluated', 'symbolic'])
@pytest.mark.parametrize("optimize", [True, False])
@pytest.test_arrays('float,is_diff,shape=(*)')
@dr.syntax
def test01_simple_diff_loop(t, optimize, mode):
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
def test02_complex_diff_loop(t, optimize, mode):
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
