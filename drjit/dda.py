import drjit as dr
from typing import Callable, TypeVar, Literal, Union, Any, Tuple, Optional

ArrayNfT  = TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNuT  = TypeVar("ArrayNuT", bound=dr.AnyArray)
ArrayNiT  = TypeVar("ArrayNiT", bound=dr.AnyArray)
BoolT     = TypeVar("BoolT", bound=Union[dr.ArrayBase, bool])
FloatT    = TypeVar("FloatT", bound=dr.AnyArray)
BoolT     = TypeVar("BoolT", bound=dr.AnyArray)
TensorXfT = TypeVar("TensorXfT", bound=dr.AnyArray)
StateT    = TypeVar("StateT")

def dda(
    ray_o: ArrayNfT,
    ray_d: ArrayNfT,
    ray_max: object,
    grid_res: ArrayNuT,
    grid_min: ArrayNfT,
    grid_max: ArrayNfT,
    func: Callable[[StateT, ArrayNuT, ArrayNfT, ArrayNfT, BoolT], Tuple[StateT, BoolT]],
    state: StateT,
    active: BoolT,
    mode: Literal["scalar", "symbolic", "evaluated", None] = None,
    max_iterations: Optional[int] = None
) -> StateT:
    r"""
    N-dimensional digital differential analyzer (DDA).

    This function traverses the intersection of a Cartesian coordinate grid and
    a specified ray or ray segment. The following snippet shows how to use it
    to enumerate the intersection of a grid with a single ray.

    .. code-block:: python

       from drjit.scalar import Array3f, Array3i, Float, Bool

       def dda_fun(state: list, index: Array3i,
                   pt_in: Array3f, pt_out: Array3f) -> tuple[list, bool]:
           # Entered a grid cell, stash it in the 'state' variable
           state.append(Array3f(index))
           return state, Bool(True)

       result = dda(
            ray_o = Array3f(-.1),
            ray_d = Array3f(.1, .2, .3),
            ray_max = Float(float('inf')),
            grid_res = Array3i(10),
            grid_min = Array3f(0),
            grid_max = Array3f(1),
            func = dda_fun,
            state = [],
            active = Bool(True)
       )

       print(result)

    Since all input elements are Dr.Jit arrays, everything works analogously
    when processing ``N`` rays and ``N`` potentially different grid
    configurations. The entire process can be captured symbolically.

    The function takes the following arguments. Note that many of them are
    generic `type variables
    <https://mypy.readthedocs.io/en/stable/generics.html>`__ (signaled by
    ending with a capital ``T``). To support different dimensions and
    precisions, the implementation must be able to deal with various input
    types, which is communicated by these type variables.

    Args:
        ray_o (ArrayNfT): the ray origin, where the ``ArrayNfT`` type variable
          refers to an n-dimensional scalar or Jit-compiled floating point array.

        ray_d (ArrayNfT): the ray direction. Does not need to be normalized.

        ray_max (object): the maximum extent along the ray, which is permitted
          to be infinite. The value is specfied as a multiple of the norm of
          ``ray_d``, which is not necessarily unit-length. Must be of type
          :py:func:`dr.value_t(ArrayNfT) <drjit.value_t>`.

        grid_res (ArrayNuT): the grid resolution, where the ``ArrayNuT`` type
          variable refers to a matched 32-bit integer array (i.e.,
          :py:func:`ArrayNuT = dr.int32_array_t(ArrayNfT) <drjit.int32_array_t>`).

        grid_min (ArrayNfT): the minimum position of the grid bounds.

        grid_max (ArrayNfT): the maximum position of the grid bounds.

        func (Callable[[StateT, ArrayNuT, ArrayNfT, ArrayNfT, BoolT], tuple[StateT, BoolT]]):
          a callback that will be invoked when the DDA traverses a grid cell. It must
          take the following five positional arguments:

          1. ``arg0: StateT``: An arbitrary state value.

          2. ``arg1: ArrayNuT``: An integer array specifying the cell index
             along each dimension.

          3. ``arg2: ArrayNfT``: The fractional position (:math:`\in [0, 1]^n`)
             where the ray *enters* the current cell.

          4. ``arg3: ArrayNfT``: The fractional position (:math:`\in [0, 1]^n`)
             where the ray *leaves* the current cell.

          5. ``arg4: BoolT``: A boolean array specifying which elements are
             active.

          The callback should then return a tuple of type ``tuple[StateT,
          BoolT]`` containing

          1. An updated state value.

          2. A boolean array that can be used to exit the loop prematurely for
             some or all rays. The iteration stops if the associated entry of
             the return value equals ``False``.

        state (StateT): an arbitrary *initial* state that will be passed
          to the callback.

        active (BoolT): an array specifying which elements of the input are
          active, where the ``BoolT`` type variable refers to a matched
          boolean array (i.e., :py:func:`BoolT = dr.mask_t(ray_o.x) <drjit.mask_t>`).

        mode: (str | None): The operation can operate in scalar, symbolic, or
          evaluated modes---see the ``mode`` argument and the documentation of
          :py:func:`drjit.while_loop` for details.

        max_iterations: int | None: Bound on the iteration count that is needed
          for reverse-mode differentiation. Forwarded to the ``max_iterations``
          parameter of :py:func:`drjit.while_loop`.

    Returns:
        StateT: The function returns the final state value of the callback upon
        termination.

    .. note::

       Just like the Dr.Jit texture interface, the implementation uses the
       convention that voxel sizes and positions are specified from last to
       first component (e.g. ``(Z, Y, X)``), while regular 3D positions use the
       opposite ``(X, Y, Z)`` order.

       In particular, all ``ArrayNuT``-typed parameters of the function and the
       callback use the ZYX convention, while ``ArrayNfT``-typed parameters use
       the ``XYZ`` convention.
    """

    ArrayNf = type(ray_o)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    assert type(ray_d) is ArrayNf
    assert type(ray_max) is Float
    assert type(grid_res) is ArrayNu
    assert type(grid_min) is ArrayNf
    assert type(grid_max) is ArrayNf
    assert type(active) is Bool

    # Switch axis convention
    grid_res = ArrayNu(reversed(grid_res))

    # Linear map to grid coordinates (likely optimized away if
    # 'grid_*' are literal constants)
    grid_res_f = ArrayNf(grid_res)
    grid_scale = grid_res_f / (grid_max - grid_min)
    grid_offset = -grid_min * grid_scale

    # Transform the ray into the grid coordinate system
    ray_o = dr.fma(ray_o, grid_scale, grid_offset)
    ray_d = ray_d * grid_scale
    rcp_d = dr.rcp(ray_d)
    inf_t = ray_d == 0

    # Per-axis intersection of the ray with the grid bounds
    t_min_v = -ray_o * rcp_d
    t_max_v = (grid_res_f - ray_o) * rcp_d
    t_min_v2 = dr.minimum(t_min_v, t_max_v)
    t_max_v2 = dr.maximum(t_min_v, t_max_v)

    # Disable extent computation for dims where the ray direction is zero
    t_min_v[inf_t] = -dr.inf
    t_max_v[inf_t] =  dr.inf

    # Reduce constraints to a single ray interval
    t_min = dr.maximum(dr.max(t_min_v2), 0)
    t_max = dr.minimum(dr.min(t_max_v2), ray_max)

    # Only run the DDA algorithm if the interval is nonempty
    active = active & (t_max > t_min) & dr.isfinite(t_max) # type: ignore

    # Advance the ray to the start of the interval
    ray_o = dr.fma(ray_d, t_min, ray_o)
    t_min, t_max = 0, t_max - t_min # type: ignore

    # Compute the integer step direction
    step = ArrayNi(dr.select(ray_d >= 0, 1, -1))
    abs_rcp_d = abs(rcp_d)

    # Integer grid coordinates
    pi = dr.clip(ArrayNi(ray_o), 0, ArrayNi(grid_res - 1))

    # Fractional entry position
    p0 = ray_o - ArrayNf(pi)

    # Step size to next interaction
    dt_v = dr.select(ray_d >= 0, dr.fma(-p0, rcp_d, rcp_d), -p0 * rcp_d)
    dt_v[inf_t] = rcp_d

    def body_fn(
        active: BoolT, state: StateT, dt_v: ArrayNfT, p0: ArrayNfT, pi: ArrayNiT, t_rem: Any,
    ) -> Tuple[BoolT, StateT, ArrayNfT, ArrayNfT, ArrayNiT, Any]:
        # Select the smallest step. It's possible that dt == 0 when starting
        # directly on a grid line.
        dt = dr.minimum(dr.min(dt_v), t_rem)
        mask = dt_v == dt

        # Compute an updated position
        p1 = dr.fma(ray_d, dt, p0)

        # Invoke the user-provided callback
        state, cont = func(state, ArrayNu(reversed(pi)),
                           p0, p1, active & (dt > 0)) # type: ignore

        # Advance
        dt_v = dr.select(mask, abs_rcp_d, dt_v - dt)
        p1[mask] = dr.select(ray_d >= 0, Float(0), Float(1))
        pi[mask] += step
        t_rem -= dt

        active &= dr.all((pi >= 0) & (pi < ArrayNi(grid_res))) & (t_rem > 0) & cont

        return active, state, dt_v, p1, pi, t_rem

    return dr.while_loop(
        state=(active, state, dt_v, p0, pi, t_max),
        body=body_fn,
        cond=lambda *args: args[0],
        mode=mode,
        labels=("active", "state", "dt_v", "p1", "pi", "t_rem"),
        max_iterations=max_iterations
    )[1]


def _int_cell_2d(
    state: Tuple[FloatT, ArrayNuT, FloatT],
    index: ArrayNuT,
    p_a: ArrayNfT,
    p_b: ArrayNfT,
    active: BoolT,
) -> Tuple[Tuple[FloatT, ArrayNuT, FloatT], BoolT]:
    """
    Compute the analytic integral of a bilinear interpolant within a 2D grid
    cell. This is an implementation detail of ``integrate()`` defined below.
    """
    Float = type(p_a.x)
    Bool = dr.mask_t(Float)

    source, stride, accum = state
    gather, lerp = dr.gather, dr.lerp

    offset = index @ stride
    v00 = gather(Float, source, offset, active)
    v01 = gather(Float, source, offset + stride[1], active)
    v10 = gather(Float, source, offset + stride[0], active)
    v11 = gather(Float, source, offset + stride[1] + stride[0], active)

    def bilerp(p: ArrayNfT) -> FloatT:
        v0 = lerp(v00, v01, p.x)
        v1 = lerp(v10, v11, p.x)
        return lerp(v0, v1, p.y)

    # Interpolated volume lookup at interval endpoints
    v_a, v_b = bilerp(p_a), bilerp(p_b)

    # Analytic def. integral of bilinear interpolant
    r = 1 / 2 * (v_a + v_b) - 1 / 6 * dr.prod(p_a - p_b) * (v00 - v01 - v10 + v11)
    r *= dr.norm(p_b - p_a)

    return ((source, stride, accum + r), Bool(True))


def _int_cell_3d(
    state: Tuple[FloatT, ArrayNuT, FloatT],
    index: ArrayNuT,
    p_a: ArrayNfT,
    p_d: ArrayNfT,
    active: BoolT,
) -> Tuple[Tuple[FloatT, ArrayNuT, FloatT], BoolT]:
    """
    Compute the analytic integral of a trilinear interpolant within a 3D grid
    cell. This is an implementation detail of ``integrate()`` defined below.
    """
    Float = type(p_a.x)
    Bool = dr.mask_t(Float)

    source, stride, accum = state
    gather, lerp = dr.gather, dr.lerp

    offset = index @ stride
    v000 = gather(Float, source, offset, active)
    v001 = gather(Float, source, offset + stride[2], active)
    v010 = gather(Float, source, offset + stride[1], active)
    v011 = gather(Float, source, offset + stride[1] + stride[2], active)
    v100 = gather(Float, source, offset + stride[0], active)
    v101 = gather(Float, source, offset + stride[2] + stride[0], active)
    v110 = gather(Float, source, offset + stride[1] + stride[0], active)
    v111 = gather(Float, source, offset + stride[2] + stride[1] + stride[0], active)

    def trilerp(p: ArrayNfT) -> FloatT:
        v00 = lerp(v000, v001, p.x)
        v01 = lerp(v010, v011, p.x)
        v10 = lerp(v100, v101, p.x)
        v11 = lerp(v110, v111, p.x)
        v0 = lerp(v00, v01, p.y)
        v1 = lerp(v10, v11, p.y)
        return lerp(v0, v1, p.z)

    # Interpolated volume lookup at interval endpoints and two intermediate ones
    v_a = trilerp(p_a)
    v_b = trilerp(lerp(p_a, p_d, 1/3))
    v_c = trilerp(lerp(p_a, p_d, 2/3))
    v_d = trilerp(p_d)

    # Analytic def. integral of trilinear interpolant (Simpson 3/8)
    r = 1 / 8 * (v_a + 3 * (v_b + v_c) + v_d)
    r *= dr.norm(p_d - p_a)

    return ((source, stride, accum + r), Bool(True))


def integrate(
    ray_o: ArrayNfT,
    ray_d: ArrayNfT,
    ray_max: FloatT,
    grid_min: ArrayNfT,
    grid_max: ArrayNfT,
    vol: dr.AnyArray,
    active: object = None,
    mode: Literal["scalar", "symbolic", "evaluated", None] = None,
) -> FloatT:
    """
    Compute an analytic definite integral of a bi- or trilinear interpolant.

    This function uses DDA (:py:func:`drjit.dda.dda()`) to step along the
    voxels of a 2D/3D volume traversed by a finite segment or a infinite-length
    ray. It analytically computes and accumulates the definite integral of the
    interpolant in each voxel.

    The input 2D/3D volume is provided using a tensor ``vol`` (e.g., of type
    :py:class:`drjit.cuda.ad.TensorXf`) with an implicitly specified grid
    resolution ``vol.shape``. This data volume is placed into an axis-aligned
    region with bounds (``grid_min``, ``grid_max``).

    The operation provides an efficient forward and backward derivative.

    .. note::

       Just like the Dr.Jit texture interface, the implementation uses the
       convention that voxel sizes and positions are specified from last to
       first component (e.g. ``(Z, Y, X)``), while regular 3D positions use the
       opposite ``(X, Y, Z)`` order.

       In particular, ``vol.shape`` uses the ZYX convention, while the
       ``ArrayNfT``-typed parameters use the ``XYZ`` convention.

       One important difference to the texture classes is that the interpolant
       is sampled at integer grid positions, whereas the Dr.Jit :ref:`texture
       classes <textures>` places values at cell centers, i.e. with a ``.5``
       fractional offset.
    """

    # Some type consistency checks
    ArrayNf = type(ray_o)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    if active is None:
        active = Bool(True)

    assert type(ray_d) is ArrayNf
    assert type(grid_min) is ArrayNf
    assert type(grid_max) is ArrayNf
    assert type(ray_max) is Float
    assert type(active) is Bool
    assert dr.is_tensor_v(vol) and dr.array_t(vol) is Float

    if dr.grad_enabled(ray_o, ray_d, grid_min, grid_max):
        raise Exception(
            'integrate(): the implementation has only been tested with '
            'differentiable volume data and will likely need changes to '
            'enable gradient tracking for ray and/or grid parameters')

    ndim = len(ray_o)
    res = vol.shape
    grid_scale = (ArrayNf(reversed(res)) - 1) / (grid_max - grid_min)
    ray_scale = dr.sqrt(dr.squared_norm(ray_d) / dr.squared_norm(ray_d * grid_scale))

    if ndim == 2:
        # (source, stride, accum)
        state = (vol.array, ArrayNu(res[1], 1), Float(0))
        int_func = _int_cell_2d
    elif ndim == 3:
        state = (vol.array, ArrayNu(res[1] * res[2], res[2], 1), Float(0))
        int_func = _int_cell_3d
    else:
        raise Exception("Unsupported number of dimensions!")

    state = dda(
        ray_o=ray_o,
        ray_d=ray_d,
        ray_max=ray_max,
        grid_min=grid_min,
        grid_max=grid_max,
        grid_res=ArrayNu(res) - 1,
        func=int_func,
        active=active,
        state=state,
        mode=mode,
        # This loop admits a simple reverse-mode derivative since it only adds
        # up values without having complex differentiable interdependences.
        max_iterations=-1
    )

    return state[2] * ray_scale

def integrate_ref(
    ray_o: ArrayNfT,
    ray_d: ArrayNfT,
    ray_max: FloatT,
    grid_min: ArrayNfT,
    grid_max: ArrayNfT,
    vol: dr.AnyArray,
    n_samples: int,
) -> FloatT:
    """
    Numerical reference for checking the correctness of :py:func:`integrate()`.
    This function densely samples a finite ray segment and computes an integral
    by applying the trapezoid rule to the interpolated texture lookups.
    """
    ArrayNf = type(ray_o)
    Float = dr.value_t(ArrayNf)
    UInt32 = dr.uint32_array_t(Float)

    assert type(ray_d) is ArrayNf
    assert type(grid_min) is ArrayNf
    assert type(grid_max) is ArrayNf
    assert type(ray_max) is Float
    assert dr.is_tensor_v(vol) and dr.array_t(vol) is Float

    ray_d = ray_d * ray_max
    int_scale = dr.norm(ray_d) / (2*(n_samples - 1))

    # Affine transformation from world space to texture space to map the
    # interval [grid_min, grid_max] with values stored at grid points onto the
    # [0, 1] domain with values stored at grid centers (i.e., how interpolation
    # works in Texture2f/Texture3f)
    grid_width = grid_max - grid_min
    grid_res_f = ArrayNf(reversed(vol.shape))
    scale = (grid_res_f - 1) / (grid_res_f * grid_width)
    offset = (grid_max + grid_min - 2*grid_min*grid_res_f) \
        / (2 * (grid_res_f * grid_width))

    # Transform the ray into this coordinate system
    ray_o = dr.fma(ray_o, scale, offset)
    ray_d = ray_d * scale

    n_elems = dr.width(ray_o, ray_d)
    index = dr.tile(dr.arange(UInt32, n_samples), n_elems)

    pos = dr.fma(
        dr.repeat(ray_d, n_samples),
        Float(index) / (n_samples - 1),
        dr.repeat(ray_o, n_samples),
    )

    active = dr.all((pos >= 0) & (pos <= 1))
    weight = dr.select((index == 0) | (index == n_samples - 1), 1, 2)

    import sys
    texture_name = ArrayNf.__name__.replace('Array', 'Texture')
    texture_cls = getattr(sys.modules[ArrayNf.__module__], texture_name)
    lookup = texture_cls(vol[..., None], use_accel=False).eval(pos, active=active)[0]

    return dr.block_sum(lookup * weight, n_samples) * int_scale
