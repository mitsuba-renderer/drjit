import drjit as dr
import pytest
import warnings

configs_jax = []
configs_torch = []

try:
    # Ignore deprecation warnings generated by the PyTorch package
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=DeprecationWarning)
        with dr.detail.scoped_rtld_deepbind():
            import torch
            torch.tensor([1])

        # The following operation can trigger a deprecation warning, so let's trigger
        # it here once to silence the warning instead of putting the condition into
        # every test.
        import torch.autograd.forward_ad as fwd_ad
        with fwd_ad.dual_level():
            fwd_ad.make_dual(torch.tensor(1.0), torch.tensor(1.0))

        supports_bool = True
        if torch.__version__ < torch.torch_version.TorchVersion('2.1.3'):
            supports_bool = False
    configs_torch.append(('torch', supports_bool, False))
except ImportError:
    pass

try:
    import jax
    from jax import config
    config.update("jax_enable_x64", True) # Enable double precision
    # JAX DLPack conversion does not support boolean-valued arrays
    # (see https://github.com/google/jax/issues/19352)
    supports_bool = False

    configs_jax.append(('jax', supports_bool, False))
    configs_jax.append(('jax', supports_bool, True))
    jit = jax.jit
except ImportError:
    pass

configs = configs_torch + configs_jax

def wrap(config):
    def wrapper(func):
        if config[2]:
            func = jax.jit(func)
        return dr.wrap(source='drjit', target=config[0])(func)
    return wrapper

def wrap_flipped(config):
    def wrapper(func):
        return dr.wrap(target='drjit', source=config[0])(func)
    return wrapper

def torch_dtype(t):
    import torch
    vt = dr.type_v(t)
    if vt == dr.VarType.Float16:
        return torch.float16
    elif vt == dr.VarType.Float32:
        return torch.float32
    elif vt == dr.VarType.Float64:
        return torch.float64
    else:
        raise Exception("Unsupported variable type")


@pytest.mark.parametrize('is_diff', [True, False])
@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test01_simple_bwd(t, config, is_diff):
    @wrap(config)
    def test_fn(x):
        return x * 2

    x = dr.arange(t, 3)
    if is_diff:
        dr.enable_grad(x)
    y = test_fn(x)
    assert dr.all(y == [0, 2, 4])

    if is_diff:
        y.grad = [10, 20, 30]
        dr.backward_to(x)
        assert dr.all(x.grad == [20, 40, 60])
    else:
        assert not dr.grad_enabled(y)

@pytest.mark.parametrize('scalar_deriv', [True, False])
@pytest.mark.parametrize('is_diff', [True, False])
@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test02_flipped_simple_bwd(t, config, is_diff, scalar_deriv):
    @wrap_flipped(config)
    def test_fn(x):
        assert dr.is_array_v(x)
        return x * 2

    import torch
    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt)
    x.requires_grad = is_diff

    y = test_fn(x)
    assert torch.all(y == torch.arange(3, dtype=dt) * 2)

    if is_diff:
        if scalar_deriv:
            y.sum().backward()
            assert torch.all(x.grad == torch.tensor([2, 2, 2], dtype=dt))
        else:
            torch.autograd.backward(y, torch.tensor([10, 20, 30], dtype=dt))
            assert torch.all(x.grad == torch.tensor([20, 40, 60], dtype=dt))
    else:
        assert y.grad is None

@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test03_simple_fwd(t, config):
    @wrap(config)
    def test_fn(x):
        return x * 2

    x = dr.arange(t, 3)
    dr.enable_grad(x)
    y = test_fn(x)

    x.grad = [10, 20, 30]
    dr.forward_to(y)
    assert dr.all(y == [0, 2, 4])
    assert dr.all(y.grad == [20, 40, 60])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test04_flipped_simple_fwd(t, config):
    import torch.autograd.forward_ad as fwd_ad

    @wrap_flipped(config)
    def test_fn(x):
        assert dr.is_array_v(x)
        return x * 2

    dt = torch_dtype(t)
    x = torch.tensor(
        [1, 2, 3], dtype=dt
    )
    x.requires_grad = True

    xd = torch.tensor(
        [10, 20, 30], dtype=dt
    )

    with fwd_ad.dual_level():
        z = fwd_ad.make_dual(x, xd)
        w = test_fn(z)
        w, wd = fwd_ad.unpack_dual(w)

        assert torch.all(w == x*2)
        assert torch.all(wd == xd*2)


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test05_simple_multiarg_bwd(t, config):
    @wrap(config)
    def test_fn(x, y):
        return x + y, y, x

    x = dr.arange(t, 3)
    y = t(4)
    dr.enable_grad(x, y)
    a, b, c = test_fn(x, y)

    a.grad = [10, 20, 30]
    b.grad = [40]
    c.grad = [50, 60, 70]
    dr.backward_to(x, y)

    assert dr.all(a == [4, 5, 6])
    assert dr.all(b == [4])
    assert dr.all(c == [0, 1, 2])

    assert dr.all(x.grad == [60, 80, 100])
    assert dr.all(y.grad == [100])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test06_flipped_simple_multiarg_bwd(t, config):
    @wrap_flipped(config)
    def test_fn(x, y):
        return x + y, y, x

    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt, requires_grad=True)
    y = torch.tensor([4], dtype=dt, requires_grad=True)
    a, b, c = test_fn(x, y)

    a.grad = torch.tensor([10, 20, 30], dtype=dt)
    b.grad = torch.tensor([40], dtype=dt)
    c.grad = torch.tensor([50, 60, 70], dtype=dt)

    assert torch.all(a == torch.tensor([4, 5, 6], dtype=dt))
    assert torch.all(b == torch.tensor([4], dtype=dt))
    assert torch.all(c == torch.tensor([0, 1, 2], dtype=dt))

    torch.autograd.backward(
        (a, b, c),
        (
            torch.tensor([10, 20, 30], dtype=dt),
            torch.tensor([40], dtype=dt),
            torch.tensor([50, 60, 70], dtype=dt),
        ))

    assert torch.all(x.grad == torch.tensor([60, 80, 100], dtype=dt))
    assert torch.all(y.grad == torch.tensor([100], dtype=dt))

@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test07_simple_multiarg_fwd(t, config):
    @wrap(config)
    def test_fn(x, y):
        return x + y, y, x

    x = dr.arange(t, 3)
    y = t(4)
    dr.enable_grad(x, y)
    a, b, c = test_fn(x, y)

    assert dr.all(a == [4, 5, 6])
    assert dr.all(b == [4])
    assert dr.all(c == [0, 1, 2])

    x.grad = [10, 20, 30]
    y.grad = [40]
    dr.forward_to(a, b, c)
    assert dr.all(a.grad == [50, 60, 70])
    assert dr.all(b.grad == [40])
    assert dr.all(c.grad == [10, 20, 30])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "not implemented for 'Half'")
def test08_filled_simple_multiarg_fwd(t, config):
    @wrap_flipped(config)
    def test_fn(x, y):
        return x + y, y + 1, x + 1

    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt, requires_grad=True)
    y = torch.tensor([4], dtype=dt, requires_grad=True)
    xd = torch.tensor([10, 20, 30], dtype=dt)
    yd = torch.tensor([40], dtype=dt)

    with fwd_ad.dual_level():
        x = fwd_ad.make_dual(x, xd)
        y = fwd_ad.make_dual(y, yd)
        a, b, c = test_fn(x, y)

        a, ad = fwd_ad.unpack_dual(a)
        b, bd = fwd_ad.unpack_dual(b)
        c, cd = fwd_ad.unpack_dual(c)

        assert dr.all(ad == torch.tensor([50, 60, 70], dtype=dt))
        assert dr.all(bd == torch.tensor([40], dtype=dt))
        assert dr.all(cd == torch.tensor([10, 20, 30], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test09_nondiff_bwd(t, config):
    @wrap(config)
    def test_fn(x, y, z):
        return x, y, z

    x = dr.arange(t, 3)
    dr.enable_grad(x)
    y = dr.int32_array_t(t)(x)

    if config[1]:
        z = y > 0
    else:
        z = y

    a, b, c = test_fn(x, y, z)
    assert dr.all(a == x) and dr.all(b == y) and dr.all(c == z)

    a.grad = [10, 20, 30]
    dr.backward_to(x)

    assert dr.all(x.grad == [10, 20, 30])

@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test10_flipped_nondiff_bwd(t, config):
    with dr.detail.scoped_rtld_deepbind():
        @wrap_flipped(config)
        def test_fn(x, y, z):
            return x*2, y+1, ~z

        dt = torch_dtype(t)
        x = torch.arange(3, dtype=dt, requires_grad=True)
        y = x.type(torch.int32)
        if config[1]:
            z = y > 0
        else:
            z = y

        a, b, c = test_fn(x, y, z)
        assert torch.all(a == x*2) and torch.all(b == y + 1) and torch.all(c == ~z)

        torch.autograd.backward(a, torch.tensor([10, 20, 30], dtype=dt))
        assert dr.all(x.grad == torch.tensor([20, 40, 60], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test11_nondiff_fwd(t, config):
    @wrap(config)
    def test_fn(x, y, z):
        return x, y, z

    x = dr.arange(t, 3)
    dr.enable_grad(x)
    y = dr.int32_array_t(t)(x)
    if config[1]:
        z = y > 0
    else:
        z = y

    a, b, c = test_fn(x, y, z)
    assert dr.all(a == x) and dr.all(b == y) and dr.all(c == z)

    x.grad = [10, 20, 30]
    dr.forward_to(a)

    assert dr.all(a.grad == [10, 20, 30])

@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "not implemented for 'Half'")
def test12_flipped_nondiff_fwd(t, config):
    @wrap_flipped(config)
    def test_fn(x, y, z):
        return x*2, y+1, ~z

    dt = torch_dtype(t)


    with fwd_ad.dual_level():
        x = fwd_ad.make_dual(
            torch.arange(3, dtype=dt, requires_grad=True),
            torch.tensor([10, 20, 30], dtype=dt)
        )
        y = x.type(torch.int32)
        if config[1]:
            z = y > 0
        else:
            z = y

        a, b, c = test_fn(x, y, z)


        a, ad = fwd_ad.unpack_dual(a)

        assert torch.all(a == x*2) and torch.all(b == y + 1) and torch.all(c == ~z)
        assert dr.all(ad == torch.tensor([20, 40, 60], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test13_scalar_bwd(t, config):
    @wrap(config)
    def test_fn(x, y, z):
        return x*2, y, z

    x = dr.arange(t, 3)
    dr.enable_grad(x)

    a, b, c = test_fn(x, 4, 5.0)
    assert dr.all(a == x*2) and dr.all(b == 4) and dr.all(c == 5)

    a.grad = [10, 20, 30]
    dr.backward_to(x)

    assert dr.all(x.grad == [20, 40, 60])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "not implemented for 'Half'")
def test14_flipped_scalar_bwd(t, config):
    @wrap_flipped(config)
    def test_fn(x, y, z):
        return x*2, y+1, z+1

    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt, requires_grad=True)

    a, b, c = test_fn(x, 4, 5.0)
    assert torch.all(a == x*2) and (b == 5) and (c == 6)

    torch.autograd.backward(a, torch.tensor([10, 20, 30], dtype=dt))

    assert torch.all(x.grad == torch.tensor([20, 40, 60], dtype=dt))

@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test15_scalar_fwd(t, config):
    @wrap(config)
    def test_fn(x, y, z):
        return x, y, z

    x = dr.arange(t, 3)
    dr.enable_grad(x)

    a, b, c = test_fn(x, 4, 5.0)
    assert dr.all(a == x) and dr.all(b == 4) and dr.all(c == 5)

    x.grad = [10, 20, 30]
    dr.forward_to(a)

    assert dr.all(a.grad == [10, 20, 30])

@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.mark.skip(reason='Skipped until issue https://github.com/pytorch/pytorch/issues/117491 is fixed.')
def test14_flipped_scalar_fwd(t, config):
    @wrap_flipped(config)
    def test_fn(x, y, z):
        return x*2, y+1, z+1

    dt = torch_dtype(t)
    with fwd_ad.dual_level():
        x = fwd_ad.make_dual(
            torch.arange(3, dtype=dt, requires_grad=True),
            torch.tensor([10, 20, 30], dtype=dt)
        )

        a, b, c = test_fn(x, 4, 5.0)
        assert torch.all(a == x*2) and torch.all(b == 5) and torch.all(c == 6)
        a, ad = fwd_ad.unpack_dual(a)
        assert torch.all(xd == torch.tensor([20, 40, 60], dtype=dt))


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test15_custom_class_bwd(t, config):
    class MyClass:
        pass

    @wrap(config)
    def test_fn(x, y):
        return x, y

    x = dr.arange(t, 3)
    dr.enable_grad(x)

    y = MyClass()

    a, b = test_fn(x, y)
    assert dr.all(a == x) and b is y

    a.grad = [10, 20, 30]
    dr.backward_to(x)

    assert dr.all(x.grad == [10, 20, 30])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "not implemented for 'Half'")
def test16_flipped_custom_class_bwd(t, config):
    class MyClass:
        pass

    @wrap_flipped(config)
    def test_fn(x, y):
        return x, y

    dt = torch_dtype(t)
    x = torch.arange(3, requires_grad=True, dtype=dt)

    y = MyClass()

    a, b = test_fn(x, y)
    assert torch.all(a == x) and b is y

    torch.autograd.backward(a, torch.tensor([10, 20, 30], dtype=dt))

    assert torch.all(x.grad == torch.tensor([10, 20, 30], dtype=dt))


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test17_custom_class_fwd(t, config):
    class MyClass:
        pass

    @wrap(config)
    def test_fn(x, y):
        return x, y

    x = dr.arange(t, 3)
    dr.enable_grad(x)
    y = MyClass()

    a, b = test_fn(x, y)
    assert dr.all(a == x) and b is y

    x.grad = [10, 20, 30]
    dr.forward_to(a)

    assert dr.all(a.grad == [10, 20, 30])


@pytest.mark.skip(reason='Skipped until issue https://github.com/pytorch/pytorch/issues/117491 is fixed.')
@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test18_flipped_custom_class_fwd(t, config):
    class MyClass:
        pass

    @wrap_flipped(config)
    def test_fn(x, y):
        return x, y

    with fwd_ad.dual_level():
        dt = torch_dtype(t)
        x = fwd_ad.make_dual(
            torch.arange(3, dtype=dt, requires_grad=True),
            torch.tensor([10, 20, 30], dtype=dt)
        )

        y = MyClass()

        a, b = test_fn(x, y)
        assert torch.all(a == x) and b is y

        a, ad = fwd_ad.unpack_dual(a)

        assert torch.all(ad == torch.tensor([10, 20, 30], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test19_args_kwargs_bwd(t, config):
    @wrap(config)
    def test_fn(*args, **kwargs):
        return args[0] * kwargs["y"]

    x = dr.arange(t, 3)
    y = t(4)
    dr.enable_grad(x, y)
    r = test_fn(x, y=y)

    r.grad = [10, 20, 30]
    dr.backward_to(x, y)

    assert dr.all(x.grad == [40, 80, 120])
    assert dr.all(y.grad == [80])

@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test20_flipped_args_kwargs_bwd(t, config):
    @wrap_flipped(config)
    def test_fn(*args, **kwargs):
        return args[0] * kwargs["y"]

    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt, requires_grad=True)
    y = torch.tensor([4], dtype=dt, requires_grad=True)
    r = test_fn(x, y=y)

    torch.autograd.backward(
        r,
        torch.tensor([10, 20, 30], dtype=dt)
    )

    assert torch.all(x.grad == torch.tensor([40, 80, 120], dtype=dt))
    assert torch.all(y.grad == torch.tensor([80], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test21_args_kwargs_fwd(t, config):
    @wrap(config)
    def test_fn(*args, **kwargs):
        return args[0] * kwargs["y"]

    x = dr.arange(t, 3)
    y = t(4)
    dr.enable_grad(x, y)
    r = test_fn(x, y=y)

    x.grad = [10, 20, 30]
    y.grad = [40]
    g = dr.forward_to(r)

    assert dr.all(g == [40, 120, 200])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
@pytest.skip_on(RuntimeError, "not implemented for 'Half'")
def test22_flipped_args_kwargs_fwd(t, config):
    @wrap_flipped(config)
    def test_fn(*args, **kwargs):
        return args[0] * kwargs["y"]

    with fwd_ad.dual_level():
        dt = torch_dtype(t)
        x = fwd_ad.make_dual(
            torch.arange(3, dtype=dt, requires_grad=True),
            torch.tensor([10, 20, 30], dtype=dt)
        )
        y = fwd_ad.make_dual(
            torch.tensor([4], dtype=dt, requires_grad=True),
            torch.tensor([40], dtype=dt)
        )

        r = test_fn(x, y=y)

        r, rd = fwd_ad.unpack_dual(r)

        assert dr.all(rd == torch.tensor([40, 120, 200], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(3, *)')
def test23_nested_arrays_bwd(t, config):
    @wrap(config)
    def test_fn(x, y):
        return (x*y).sum()

    x = t([1, 2], [3, 4], [5, 6])
    y = t(10, 20, 30)
    dr.enable_grad(x, y)
    r = test_fn(x, y)

    r.grad = [100, 200, 300]
    dr.backward_to(x, y)
    assert dr.all(x.grad == [6000, 12000, 18000], axis=None)
    assert dr.all(y.grad == [1800, 4200, 6600])


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(3, *)')
def test24_nested_arrays_fwd(t, config):
    @wrap(config)
    def test_fn(x, y):
        return (x*y).sum()

    x = t([1, 2], [3, 4], [5, 6])
    y = t(10, 20, 30)
    dr.enable_grad(x, y)
    r = test_fn(x, y)
    x.grad = [[10, 20], [30, 40], [50, 60]]
    y.grad = [100, 200, 300]

    g = dr.forward_to(r)
    assert dr.is_tensor_v(g)
    assert g.array[0] == 10000


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test25_pytree_bwd(t, config):
    @wrap(config)
    def test_fn(x):
        return {
            123:(x[0]["hello"] + 2*x[1]["world"][0])
        }
    x = t(1, 2, 3)
    y = t(4, 5, 6)
    dr.enable_grad(x, y)
    xt = [
        { 'hello' : x },
        { 'world' : (y,) }
    ]
    rt = test_fn(xt)
    r = rt[123]

    r.grad = [100, 200, 300]
    dr.backward_to(x, y)
    assert dr.all(x.grad == [100, 200, 300])
    assert dr.all(y.grad == [200, 400, 600])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test26_flipped_pytree_bwd(t, config):
    @wrap_flipped(config)
    def test_fn(x):
        return {
            123:(x[0]["hello"] + 2*x[1]["world"][0])
        }

    dt = torch_dtype(t)
    x = torch.tensor([1, 2, 3], dtype=dt, requires_grad=True)
    y = torch.tensor([4, 5, 6], dtype=dt, requires_grad=True)
    xt = [
        { 'hello' : x },
        { 'world' : (y,) }
    ]
    rt = test_fn(xt)
    r = rt[123]

    torch.autograd.backward(r, torch.tensor([100, 200, 300], dtype=dt))
    assert torch.all(x.grad == torch.tensor([100, 200, 300], dtype=dt))
    assert torch.all(y.grad == torch.tensor([200, 400, 600], dtype=dt))

@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test25_pytree_fwd(t, config):
    @wrap(config)
    def test_fn(x):
        return {
            123:(x[0]["hello"] + 2*x[1]["world"][0])
        }
    x = t(1, 2, 3)
    y = t(4, 5, 6)
    dr.enable_grad(x, y)
    xt = [
        { 'hello' : x },
        { 'world' : (y,) }
    ]
    rt = test_fn(xt)
    r = rt[123]
    x.grad=[10,20,30]
    y.grad=[40,50,60]
    dr.forward_to(r)

    assert dr.all(r.grad == [90, 120, 150])


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test26_flipped_pytree_fwd(t, config):
    @wrap_flipped(config)
    def test_fn(x):
        return {
            123:(x[0]["hello"] + 2*x[1]["world"][0])
        }

    with fwd_ad.dual_level():
        dt = torch_dtype(t)
        x = fwd_ad.make_dual(
            torch.tensor([1, 2, 3], dtype=dt, requires_grad=True),
            torch.tensor([10, 20, 30], dtype=dt)
        )
        y = fwd_ad.make_dual(
            torch.tensor([4, 5, 6], dtype=dt, requires_grad=True),
            torch.tensor([40, 50, 60], dtype=dt)
        )

        xt = [
            { 'hello' : x },
            { 'world' : (y,) }
        ]
        rt = test_fn(xt)
        r = rt[123]

        r, rd = fwd_ad.unpack_dual(r)
        assert torch.all(rd == torch.tensor([90, 120, 150], dtype=dt))


@pytest.mark.parametrize('config', configs)
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test27_exception(t, config):
    @wrap(config)
    def test_fn(x):
        raise RuntimeError('foo')

    with pytest.raises(RuntimeError) as err:
        test_fn(t(1, 2, 3))
    assert 'foo' in str(err.value.__cause__)


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float32,shape=(*)')
def test28_flipped_exception(t, config):
    @wrap_flipped(config)
    def test_fn(x):
        raise RuntimeError('foo')
    with pytest.raises(RuntimeError) as err:
        test_fn(torch.tensor([1, 2, 3]))
    assert 'foo' in str(err.value)


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,llvm,float,shape=(*)')
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test29_flipped_non_tensor_output_bwd(t, config):
    @wrap_flipped(config)
    def test_fn(x):
        a = dr.gather(t, x.array, 0)
        b = dr.gather(t, x.array, 1)
        c = dr.gather(t, x.array, 2)
        return a, b * 2, c * 3

    import torch
    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt)
    x.requires_grad = True

    out1, out2, out3 = test_fn(x)
    assert out1 == 0
    assert out2 == 2
    assert out3 == 6

    (out1 + out2 + out3).backward()
    assert torch.all(x.grad == torch.tensor([1, 2, 3], dtype=dt))


@pytest.mark.parametrize('config', configs_torch)
@pytest.test_arrays('is_diff,float,shape=(*)')
def test30_nested(t, config):
    @wrap(config)
    def add(x, y):
        return x + y

    @wrap_flipped(config)
    def test_fn(x, y):
        return x * add(x, y)

    import torch
    dt = torch_dtype(t)
    x = torch.arange(3, dtype=dt)
    y = x * 4
    x.requires_grad = True
    y.requires_grad = True

    out = test_fn(x, y)
    assert torch.all(out == torch.tensor([0, 5, 20], dtype=dt))

    torch.autograd.backward(out, torch.tensor([1, 2, 3], dtype=dt))
    assert torch.all(x.grad == torch.tensor([0, 12, 36], dtype=dt))
    assert torch.all(y.grad == torch.tensor([0, 2, 6], dtype=dt))
