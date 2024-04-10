import drjit as dr
from drjit.dda import dda, integrate, integrate_ref
from dataclasses import dataclass
import pytest
import sys

@dataclass
class Voxel:
    t: float
    idx: tuple[int, ...]
    p0: tuple[float, ...]
    p1: tuple[float, ...]


def dda_bruteforce(
    ray_o: tuple[float, ...],
    ray_d: tuple[float, ...],
    ray_max: float = float("inf"),
    grid_res: tuple[int, ...] = (1, 1, 1),
    grid_min: tuple[float, ...] = (0, 0, 0),
    grid_max: tuple[float, ...] = (1, 1, 1),
) -> list[Voxel]:
    """
    Brute-force DDA routine that enumerates all grid cells and computes
    intersections with each one. Used in dda_check()
    """
    import itertools

    n = len(grid_res)
    grid_res = tuple(reversed(grid_res))
    grid_scale = tuple(grid_res[i] / (grid_max[i] - grid_min[i]) for i in range(n))
    ray_o = tuple((ray_o[i] - grid_min[i]) * grid_scale[i] for i in range(n))
    ray_d = tuple(ray_d[i] * grid_scale[i] for i in range(n))
    result: list[Voxel] = []

    for idx in itertools.product(*tuple(range(res) for res in grid_res)):
        t_min, t_max = float("-inf"), float("inf")
        valid = False

        for i in range(n):
            if ray_d[i] == 0:
                continue

            t_min_i = (idx[i] - ray_o[i]) / ray_d[i]
            t_max_i = (idx[i] + 1 - ray_o[i]) / ray_d[i]
            t_min = max(t_min, min(t_min_i, t_max_i))
            t_max = min(t_max, max(t_min_i, t_max_i))
            valid = True

        t_min = max(t_min, 0)
        t_max = min(t_max, ray_max)

        if t_min < t_max and valid:
            result.append(
                Voxel(
                    t_min,
                    tuple(reversed(idx)),
                    tuple(ray_o[i] + ray_d[i] * t_min - idx[i] for i in range(n)),
                    tuple(ray_o[i] + ray_d[i] * t_max - idx[i] for i in range(n)),
                )
            )

    result.sort(key=lambda x: x.t)
    return result


def dda_check(
    ray_o: tuple[float, ...],
    ray_d: tuple[float, ...],
    ray_max: float = float("inf"),
    grid_res: tuple[int, ...] = (1, 1, 1),
    grid_min: tuple[float, ...] = (0, 0, 0),
    grid_max: tuple[float, ...] = (1, 1, 1),
) -> None:
    """
    Compare the proper and brute force versions of DDA against each other.
    """
    from drjit.scalar import Array3f, Array3u

    ref = dda_bruteforce(
        grid_res=grid_res,
        grid_min=grid_min,
        grid_max=grid_max,
        ray_o=ray_o,
        ray_d=ray_d,
        ray_max=ray_max,
    )

    def dda_cb(state, idx, p0, p1, active):
        if active:
            state.append(Voxel(0, tuple(idx), tuple(p0), tuple(p1)))
        return state, True

    out = dda(
        func=dda_cb,
        state=[],
        ray_o=Array3f(ray_o),
        ray_d=Array3f(ray_d),
        ray_max=ray_max,
        grid_min=Array3f(grid_min),
        grid_max=Array3f(grid_max),
        grid_res=Array3u(grid_res),
        active=True
    )

    assert len(ref) == len(out)
    for i in range(len(ref)):
        # print(f"ref[{i}]={ref[i]}")
        # print(f"out[{i}]={out[i]}")
        assert ref[i].idx == out[i].idx
        assert ref[i].p0 == pytest.approx(out[i].p0, rel=1e-5, abs=1e-5)
        assert ref[i].p1 == pytest.approx(out[i].p1, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("s", (-1, 1))
def test01_single_voxel_inside(s):
    """A ray starting and ending inside a single cell"""
    dda_check(ray_o=(0.5, 0.5, 0.5), ray_d=(0.1 * s, 0.2 * s, 0.3 * s), ray_max=1.0)


@pytest.mark.parametrize("s", (-1, 1))
def test02_single_voxel_outside(s):
    """A ray starting and ending outside a single cell"""
    dda_check(ray_o=(0, 0, -s * 1.1), ray_d=(0.1 * s, 0.2 * s, 0.3 * s))


@pytest.mark.parametrize("s", (-1, 1))
def test03_several_outside_1(s):
    """A ray starting outside of the bounding box traversing 6 cells."""
    dda_check(
        ray_o=(0, 0, -s * 1.1),
        ray_d=(0.1 * s, 0.2 * s, 0.3 * s),
        grid_res=(3, 4, 5),
        grid_min=(-1, -1, -1),
        grid_max=(1, 1, 1),
    )

@pytest.mark.parametrize("s", (-1, 1))
@pytest.mark.parametrize("z", (0, 0.001))
def test04_several_outside_2(s, z):
    """
    Variation of test03 with different scales, optionally
    with a zero-valued direction component
    """
    dda_check(
        ray_o=(0, 0, -s * 1.1),
        ray_d=(z, 0.2 * s, 0.3 * s),
        grid_res=(9, 8, 1),
        grid_min=(-1, -1, -1),
        grid_max=(1, 1, 1),
    )


@pytest.mark.parametrize("o", ((1e-10, 3.4, 4.5), (1e-10, 3, 4)))
@pytest.mark.parametrize("s", (-1, 1))
@pytest.mark.parametrize("maxt", (float('inf'), 3.5))
def test05_several_inout_2(s, o, maxt):
    """
    Ray starting within the grid, optionally with a starting position
    that is exactly on a grid cell boundary
    """
    dda_check(
        ray_o=o,
        ray_d=(0, 0.25 * s, 0.3 * s),
        grid_res=(9, 8, 1),
        grid_min=(0, 0, 0),
        grid_max=(1, 8, 9),
        ray_max=maxt
    )


def test06_invalid():
    """Invalid ray with a zero-valued ray direction"""
    dda_check(
        ray_o=(.5, .5, .5),
        ray_d=(0, 0, 0),
        grid_res=(3, 3, 3)
    )


@pytest.mark.parametrize("s", (-1, 1))
def test07_diagonal(s):
    """Test a ray that goes through corners"""
    dda_check(
        ray_o=(-s, -s, -s),
        ray_d=(s, s, s),
        grid_res=(10, 10, 10),
        grid_min=(-1, -1, -1),
        grid_max=(1, 1, 1)
    )

def check(t, rng, vol, n_samples=4, grid_min=None, grid_max=None, mode=None):
    """
    Helper function to check the correctness of the analytic integration routines
    """
    if grid_min is None:
        grid_min = t(-1)

    if grid_max is None:
        grid_max = t(1)

    ndim = vol.ndim
    tv = dr.value_t(t)

    p0 = t([tv(rng.next_float32()) for _ in range(ndim)])
    p1 = t([tv(rng.next_float32()) for _ in range(ndim)])

    grid_scale = grid_max - grid_min
    p0 = p0 *grid_scale + grid_min
    p1 = p1 *grid_scale + grid_min

    val_ref = integrate_ref(
        ray_o=p0,
        ray_d=p1 - p0,
        ray_max=tv(1),
        grid_min=grid_min,
        grid_max=grid_max,
        vol=vol,
        n_samples=n_samples
    )

    val_dda = integrate(
        ray_o=p0,
        ray_d=p1 - p0,
        ray_max=tv(1),
        grid_min=grid_min,
        grid_max=grid_max,
        vol=vol,
        mode=mode
    )

    # print("----")
    # print("Result: (ref/dda)")
    # print(val_ref)
    # print(val_dda)

    assert dr.allclose(val_ref, val_dda)

configs = ('float, shape=(2, *), -complex, -float16', 'float, shape=(3, *), -float16')
configs_ad = ('float, shape=(2, *), diff, -complex, -float16', 'float, diff, shape=(3, *), -float16')

@pytest.test_arrays(*configs)
def test08_integrate_constant(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    vol = dr.tensor_t(t)(dr.value_t(t)(1 for _ in range(2**ndim)),
                         shape=(2,)*ndim)
    rng = m.PCG32(16)
    check(t, rng, vol)


@pytest.test_arrays(*configs)
def test09_integrate_constant_nonuniform(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    shape=(3, 2, 7)
    grid_max=[2, 4, 3]
    data = dr.value_t(t)([1]*dr.prod(shape[:ndim]))
    vol = dr.tensor_t(t)(data, shape=shape[:ndim])
    rng = m.PCG32(16)
    check(t, rng, vol, mode='evaluated', grid_min=t(0), grid_max=t(grid_max[:ndim]))


@pytest.test_arrays(*configs)
def test10_integrate_gradient(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    tt = dr.tensor_t(t)
    vol = dr.tensor_t(t)((0, 1)*(2**(ndim-1)), shape=(2,)*ndim)
    rng = m.PCG32(16)
    check(t, rng, vol)


@pytest.test_arrays(*configs)
def test11_integrate_random(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    data = m.PCG32(res**ndim).next_float32()*2-1
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    check(t, rng, vol, n_samples=512, grid_min=t(0),
          grid_max=t(range(1, ndim+1)))


@pytest.test_arrays(*configs)
def test12_integrate_random_nonuniform(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    shape=(3, 2, 5)[:ndim]
    data = m.PCG32(dr.prod(shape)).next_float32()*2-1
    vol = dr.tensor_t(t)(data, shape=shape)
    rng = m.PCG32(16)
    check(t, rng, vol, n_samples=2048, grid_min=t(0),
          grid_max=t(range(1, ndim+1)))


@pytest.test_arrays(*configs)
def test13_integrate_random(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    data = m.PCG32(res**ndim).next_float32()*2-1
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    check(t, rng, vol, n_samples=512, grid_min=t(0),
          grid_max=t(range(1, ndim+1)))

def check_grad(t, rng, vol, diff, grad_val, n_samples=4, grid_min=None, grid_max=None, mode=None, rtol=None):
    """
    Helper function to check the correctness of the analytic integration routines
    """
    if grid_min is None:
        grid_min = t(-1)

    if grid_max is None:
        grid_max = t(1)

    ndim = vol.ndim
    tv = dr.value_t(t)

    p0 = t([tv(rng.next_float32()) for _ in range(ndim)])
    p1 = t([tv(rng.next_float32()) for _ in range(ndim)])

    grid_scale = grid_max - grid_min
    p0 = p0 *grid_scale + grid_min
    p1 = p1 *grid_scale + grid_min

    if True:
        dr.enable_grad(vol)

        val_ref = integrate_ref(
            ray_o=p0,
            ray_d=p1 - p0,
            ray_max=tv(1),
            grid_min=grid_min,
            grid_max=grid_max,
            vol=vol,
            n_samples=n_samples
        )

        if diff == 'fwd':
            vol.grad = grad_val
            grad_ref = dr.forward_to(val_ref)
            dr.disable_grad(vol)
        else:
            val_ref.grad = grad_val
            grad_ref = dr.backward_to(vol)
            dr.disable_grad(vol)

    if True:
        dr.enable_grad(vol)

        val_dda = integrate(
            ray_o=p0,
            ray_d=p1 - p0,
            ray_max=tv(1),
            grid_min=grid_min,
            grid_max=grid_max,
            vol=vol,
            mode=mode
        )
        assert dr.allclose(val_ref, val_dda, rtol=rtol)

        if diff == 'fwd':
            vol.grad = grad_val
            grad_dda = dr.forward_to(val_ref)
        else:
            val_dda.grad = grad_val
            grad_dda = dr.backward_to(vol)

    print("----")
    print("Result: (ref/dda)")
    print(val_ref)
    print(val_dda)
    print("Grad: (ref/dda)")
    print(grad_ref)
    print(grad_dda)

    assert dr.allclose(grad_ref, grad_dda, rtol=rtol)

@pytest.test_arrays(*configs_ad)
def test14_fwd_ad_constant(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    data = m.PCG32(res**ndim).next_float32()*2-1
    dr.eval(data)
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    check_grad(t, rng, vol, n_samples=512, diff='fwd', grad_val=1)


@pytest.test_arrays(*configs_ad)
def test15_fwd_ad_random(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    vol_rng = m.PCG32(res**ndim)
    data = vol_rng.next_float32()*2-1
    grad_val = vol_rng.next_float32()*2-1
    dr.eval(data)
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    check_grad(t, rng, vol, n_samples=512, diff='fwd', grad_val=grad_val)


@pytest.test_arrays(*configs_ad)
def test16_rev_ad_constant(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    data = m.PCG32(res**ndim).next_float32()*2-1
    dr.eval(data)
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    check_grad(t, rng, vol, n_samples=1024, diff='rev', grad_val=1)


@pytest.test_arrays(*configs_ad)
def test17_rev_ad_random(t):
    m = sys.modules[t.__module__]
    ndim = dr.size_v(t)
    res = 3
    vol_rng = m.PCG32(res**ndim)
    data = vol_rng.next_float32()*2-1
    dr.eval(data)
    vol = dr.tensor_t(t)(data, shape=(res,)*ndim)
    rng = m.PCG32(16)
    grad_val = rng.next_float32()*2-1
    check_grad(t, rng, vol, n_samples=1024, diff='rev', rtol=1e-4, grad_val=grad_val)
