from __future__ import annotations
import drjit
import sys

if sys.version_info < (3, 11):
    from typing_extensions import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any
else:
    from typing import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any

# Import classes/functions from C++ extension
MatrixView = drjit.detail.nn.MatrixView
CoopVec = drjit.detail.nn.CoopVec
pack = drjit.detail.nn.pack
unpack = drjit.detail.nn.unpack
matvec = drjit.detail.nn.matvec
view = drjit.detail.nn.view
cast = drjit.detail.nn.cast
T = drjit.detail.nn.T

TensorOrViewOrNone: TypeAlias = Union[
    drjit.ArrayBase,
    MatrixView,
    None
]

class Module:
    """
    This is the base class of a modular set of operations that make
    the specification of neural network architectures more convenient.

    Module subclasses are :ref:`PyTrees <pytrees>`, which means that various
    Dr.Jit operations can automatically traverse them.

    Constructing a neural network generally involves the following pattern:

    .. code-block::

       # 1. Establish the network structure
       net = nn.Sequential(
           nn.Linear(-1, 32, bias=False),
           nn.ReLU(),
           nn.Linear(-1, 3)
       )

       # 2. Instantiate the network for a specific backend + input size
       net = net.alloc(TensorXf16, 2)

       # 3. Pack coefficients into a training-optimal layout
       coeffs, net = nn.pack(net, layout='training')

    Network evaluation expects a :ref:`cooperative vector <coop_vec>` as input
    (i.e., ``net(nn.CoopVec(...))``) and returns another cooperative vector.
    The ``coeffs`` buffer contains all weight/bias data in training-optimal
    format and can be optimized, which will directly impact the packed network
    ``net`` that references this buffer.
    """
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        """
        Evaluate the model with an input cooperative vector and return the result.
        """
        raise NotImplementedError(f"{type(self).__name__}.__call__() implementation is missing.")

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        """
        Internal method used to propagate argument sizes and allocate weight
        storage of all NN modules.

        The method takes to parameters as input: a weight storage type
        ``dtype`` (e.g., :py:class:`drjit.cuda.ad.TensorXf16`) and ``size``,
        the number of input arguments of the module. The function returns a
        potentially new module instance with allocated weights, plus the number
        of outputs.
        """
        return self, size

    def alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1, rng: Optional[drjit.random.Generator] = None) -> Module:
        """
        Returns a new instance of the model with allocated weights.

        This function expects a suitable tensor ``dtype`` (e.g.
        :py:class:`drjit.cuda.ad.TensorXf16` or
        :py:class:`drjit.llvm.ad.TensorXf`) that will be used to store the
        weights on the device.

        If the model or one of its sub-models is automatically sized (e.g.,
        ``input_features=-1`` in :py:class:`drjit.nn.Linear`), the final
        network configuration may ambiguous and an exception will be raised.
        Specify the optional ``size`` parameter in such cases to inform the
        allocation about the size of the input cooperative vector.

        Layer weights are initialized using pseudorandom values obtained from
        the specified generator object ``rng``.

        Specifying a newly seeded random number generator with the same seed
        ensures that weights will be consistent across runs (i.e., calling
        ``alloc()`` twice will produce the same initialization).

        If ``rng=None`` (the default), a generator is constructed on the fly
        via ``dr.rng(seed=0x100000000)``. This particular seed value is used to
        de-correlate the network weights with respect to any potential future
        network evaluations that might be produced by a random number generator
        with the default seed (``0``). (Please ignore this paragraph if it
        is unclear, it explains a protection against a subtle/niche issue.)
        """

        if rng is None:
            rng = drjit.rng(seed=0x100000000)

        return self._alloc(dtype, size, rng)[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

class Sequential(Module, Sequence[Module]):
    """
    This model evaluates provided arguments ``arg[0]``, ``arg[1]``, ..., in sequence.
    """
    DRJIT_STRUCT = { 'layers' : tuple }

    layers: tuple[Module, ...]

    def __init__(self, *args: Module):
        self.layers = args

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        for l in self.layers:
            arg = l(arg)
        return arg

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        result = []
        for l in self.layers:
            l_new, size = l._alloc(dtype, size, rng)
            result.append(l_new)
        return Sequential(*result), size

    def __len__(self):
        """Return the number of contained models"""
        return len(self.layers)

    def __getitem__(self, index: int, /) -> Module: # type: ignore
        """Return the model at position ``index``"""
        return self.layers[index]

    def __repr__(self) -> str:
        s = 'Sequential(\n'
        n = len(self.layers)
        for i in range(n):
            s += '  ' + repr(self.layers[i]).replace('\n', '\n  ')
            if i + 1 < n:
                s += ','
            s += '\n'
        s += ')'
        return s

class ReLU(Module):
    r"""
    ReLU (rectified linear unit) activation function.

    This model evaluates the following expression:

    .. math::

       \mathrm{ReLU}(x) = \mathrm{max}\{x, 0\}.

    """

    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return drjit.maximum(arg, 0)

class LeakyReLU(Module):
    r"""
    "Leaky" ReLU (rectified linear unit) activation function.

    This model evaluates the following expression:

    .. math::

       \mathrm{LeakyReLU}(x) = \begin{cases}
          x,&\mathrm{if}\ x\ge 0,\\
          \texttt{negative\_slope}\cdot x,&\mathrm{otherwise}.
       \end{cases}
    """

    DRJIT_STRUCT = { 'negative_slope': Union[float, drjit.ArrayBase] }
    def __init__(self, negative_slope: Union[float, drjit.ArrayBase] = 1e-2):
        self.negative_slope = negative_slope

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return drjit.maximum(arg, 0) + drjit.minimum(arg, 0.0) * self.negative_slope


class Exp2(Module):
    r"""
    Applies the base-2 exponential function to each component.

    .. math::

       \mathrm{Exp2}(x) = 2^x

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return drjit.exp2(arg)

class Exp(Module):
    r"""
    Applies the exponential function to each component.

    .. math::

       \mathrm{Exp}(x) = e^x
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return drjit.exp2(arg * (1 / drjit.log(2)))

class Tanh(Module):
    r"""
    Applies the hyperbolic tangent function to each component.

    .. math::

       \mathrm{Tanh}(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return drjit.tanh(arg)

class ScaleAdd(Module):
    r"""
    Scale the input by a fixed scale and apply an offset.

    Note that ``scale`` and ``offset`` are assumed to be constant (i.e., not trainable).

    .. math::

       \mathrm{ScaleAdd}(x) = x\cdot\texttt{scale} + \texttt{offset}
    """
    DRJIT_STRUCT = {'scale': Union[None, float, int, drjit.ArrayBase],
                    'offset': Union[None, float, int, drjit.ArrayBase]}
    def __init__(self, scale: Union[float, int, drjit.ArrayBase, None] = None,
                 offset: Union[float, int, drjit.ArrayBase, None] = None):
        self.scale = scale
        self.offset = offset
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        if not self.scale or not self.offset:
            raise Exception("drjit.nn.ScaleAdd(): you must set a scale and offset!")
        return drjit.fma(arg, self.scale, self.offset)

class Cast(Module):
    """
    Cast the input cooperative vector to a different precision. Should be
    instantiated with the desired element type, e.g. ``Cast(drjit.cuda.ad.Float32)``
    """
    DRJIT_STRUCT = { 'dtype': Optional[Type[drjit.ArrayBase]] }
    def __init__(self, dtype: Optional[Type[drjit.ArrayBase]] = None):
        self.dtype = dtype
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return cast(arg, self.dtype)
    def __repr__(self):
        return f'Cast(dtype={self.dtype.__name__})'

class Linear(Module):
    r"""
    This layer represents a learnable affine linear transformation of the input
    data following the expression :math:`\mathbf{y} = \mathbf{A}\mathbf{x} +
    \mathbf{b}`.

    It takes ``in_features`` inputs and returns a cooperative vector with
    ``out_features`` dimensions. The following parameter values have a special
    a meaning:

    - ``in_features=-1``: set the input size to match the previous model's
      output (or the input of the network, if there is no previous model).

    - ``out_features=-1``: set the output size to match the input size.

    The bias (:math:`\textbf{b}`) term is optional and can be disabled by
    specifying ``bias=False``.

    The method :py:func:`Module.alloc` initializes the underlying coefficient
    storage with random weights following a uniform Xavier initialization,
    i.e., uniform variates on the interval :math:`[-k,k]` where
    :math:`k=1/\sqrt{\texttt{out\_features}}`.
    """
    config: Tuple[int, int, bool]
    weights: TensorOrViewOrNone
    bias: TensorOrViewOrNone

    DRJIT_STRUCT = {
        'config': Tuple[int, int, bool],
        'weights': TensorOrViewOrNone,
        'bias': TensorOrViewOrNone
    }

    def __init__(self, in_features: int = -1, out_features: int = -1, bias = True) -> None:
        self.config = (in_features, out_features, bias)
        self.weights = self.bias = None

    def __repr__(self) -> str:
        s = f'Linear(in_features={self.config[0]}, out_features={self.config[1]}'
        if not self.config[2]:
            s += ', bias=False'
        s += ')'
        return s

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        if self.weights is None:
            raise RuntimeError(
                "Uninitialized network. Call 'net = net.alloc(""<Tensor type>"
                ")' to initialize the weight storage first. Following this, "
                "use 'drjit.nn.pack()' to transform the network into an "
                "optimal layout for evaluation."
            )
        elif not isinstance(self.weights, MatrixView) or \
           (self.bias is not None and not isinstance(self.bias, MatrixView)):
            raise RuntimeError(
                "Uninitialized network. Use 'drjit.nn.pack()' to transform"
                "the network into an optimal layout for evaluation."
            )
        return matvec(self.weights, arg, self.bias)

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        in_features, out_features, bias = self.config
        if in_features < 0:
            in_features = size
        if out_features < 0:
            out_features = in_features
        if in_features == -1 or out_features == -1:
            raise RuntimeError("The network contains layers with an unspecified "
                               "size. You must specify the input size to drjit.nn.Module.alloc().")

        result = Linear(in_features, out_features, bias)
        # Xavier (uniform) initialization, matches PyTorch
        scale = drjit.sqrt(1 / out_features)
        Float32 = drjit.float32_array_t(dtype)
        samples = rng.random(Float32, (out_features, in_features))
        result.weights = dtype(drjit.fma(samples, 2, -1) * scale)
        if bias:
            result.bias = drjit.zeros(dtype, out_features)
        return result, out_features

def _sincos_tri(t: T) -> tuple[T, T]:
    """Implementation detail of the TriEncode class"""
    s = t - .25
    st = s - drjit.round(s)
    ct = t - drjit.round(t)
    return (
        drjit.fma(drjit.abs(st), -4, 1),
        drjit.fma(drjit.abs(ct), -4, 1)
    )

class TriEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using
    triangular sine and cosine approximations of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin_\triangle(2^0\,x)\\
           \cos_\triangle(2^0\,x)\\
           \vdots\\
           \cos_\triangle(2^{n-1}\, x)\\
           \sin_\triangle(2^{n-1}\, x)
       \end{bmatrix}

    where

    .. math::

       \cos_\triangle(x) = 1-4\left|x-\mathrm{round}(x)\right|

    and

    .. math::

       \sin_\triangle(x) = \cos_\triangle(x-1/4)

    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components conincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional parameter ``shift`` to phase-shift the :math:`i`-th
    frequency by :math:`2\,\pi\,\mathrm{shift}` to avoid this behavior.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/tri_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/tri_encode_dark.svg
      :class: only-dark
      :width: 600px
      :align: center
    """

    DRJIT_STRUCT = { 'octaves' : int, 'shift': float, 'channels': int }

    def __init__(self, octaves: int = 0, shift: float = 0) -> None:
        self.octaves = octaves
        self.shift = shift
        self.channels = -1

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        r = TriEncode(self.octaves, self.shift)
        r.channels = size
        return r, size * self.octaves * 2

    def __repr__(self) -> str:
        return f'TriEncode(octaves={self.octaves}, shift={self.shift}, in_channels={self.channels}, out_features={self.channels*self.octaves*2})'

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        args, r = list(arg), list()
        for arg in args:
            for i in range(self.octaves):
                s, c = _sincos_tri(drjit.fma(arg, 2**i, self.shift*i))
                r.append(s)
                r.append(c)
        return CoopVec(r)


class SinEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using sines
    and cosines of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin(2^0\, 2\pi x)\\
           \cos(2^0\, 2\pi x)\\
           \vdots\\
           \sin(2^{n-1}\, 2\pi x)\\
           \cos(2^{n-1}\, 2\pi x)\\
       \end{bmatrix}


    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components conincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional parameter ``shift`` to phase-shift the :math:`i`-th
    frequency by :math:`\mathrm{shift}` radians to avoid this behavior.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/sin_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/sin_encode_dark.svg
      :class: only-dark
      :width: 600px
      :align: center
    """

    DRJIT_STRUCT = { 'octaves' : int, 'shift': Union[tuple, None], 'channels': int }

    def __init__(self, octaves: int = 0, shift: float = 0) -> None:
        self.octaves = octaves
        self.channels = -1

        if shift == 0:
            self.shift = None
        else:
            self.shift = (drjit.sin(shift * 2 * drjit.pi),
                          drjit.cos(shift * 2 * drjit.pi))

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        r = SinEncode(self.octaves)
        r.channels = size
        r.shift = self.shift
        return r, size * self.octaves * 2

    def __repr__(self) -> str:
        return f'SinEncode(octaves={self.octaves}, shift={self.shift}, in_channels={self.channels}, out_features={self.channels*self.octaves*2})'

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        args, r = list(arg), list()
        for arg in args:
            s, c = drjit.sincos(arg * 2 * drjit.pi)
            r.append(s)
            r.append(c)
            for _ in range(1, self.octaves):
                # Recurrence for double angle sine/cosine
                s2 = 2 * s
                s, c = s2 * c, drjit.fma(-s2, s, 1)
                r.append(s)
                r.append(c)

                if self.shift:
                    # Recurrence for sine/cosine angle addition
                    ss, cs = self.shift
                    s, c = drjit.fma(s, cs,  c*ss), \
                           drjit.fma(c, cs, -s*ss)

        return CoopVec(r)


class PyHashGrid3D:
    def __init__(
        self,
        array_type,
        hashmap_size: int,
        num_levels: int,
        base_res: int,
        per_level_scale: float,
        num_features: int = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        grid_lod: GridLod = None,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ):
        """Python-based implementation of the INGP hashgrid.

        :parameter smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        :parameter smooth_weight_lambda: the value of lambda used for the straight-through estimator.
        """
        # Note: we don't want to lose precision during e.g. interpolation
        # weight computations, so we will still use f32 in many computation
        # even when the `array_type` is `mi.Float16`.
        self.StorageFloat = array_type
        self.Float = dr.float32_array_t(array_type) if dr.is_half_v(array_type) else array_type
        self.VectorDf = {
            (mi.Float16, 2): mi.Vector2f,  # Still f32
            (mi.Float32, 2): mi.Vector2f,
            (mi.Float64, 2): mi.Vector2d,
            (mi.Float16, 3): mi.Vector3f,  # Still f32
            (mi.Float32, 3): mi.Vector3f,
            (mi.Float64, 3): mi.Vector3d,
        }[(array_type, self.dimensionality())]
        self.VectorDu = dr.uint32_array_t(self.VectorDf)

        self._hashmap_size = hashmap_size
        self._num_levels = num_levels
        self._base_res = base_res
        self._per_level_scale = per_level_scale
        self.log2_per_level_scale = dr.log2(self._per_level_scale)
        self._num_features = num_features
        self._align_corners = align_corners

        # TODO: add support for Torch NGP indexing compatibility
        self._torchngp_compat = torchngp_compat
        if self._torchngp_compat:
            raise NotImplementedError("Not supported yet: `torchngp_compat = True` in the Python-based hashgrid.")

        self.grid_lod = grid_lod

        assert (self._hashmap_size % 8) == 0

        self.level_offsets = [None] * (self._num_levels + 1)
        self.cell_sizes = [0.0] * self._num_levels
        max_params = dr.largest(dr.scalar.UInt32) // 2
        dimensionality = self.dimensionality()
        offset = 0
        for level_i in range(self._num_levels):
            res = self._grid_resolution(self._grid_scale(level_i))
            stride = dr.power(float(res), dimensionality)
            params_in_level = int(max_params) if (stride > max_params) else (res**dimensionality)
            params_in_level = next_multiple(params_in_level, 8)
            params_in_level = min(params_in_level, self._hashmap_size)

            self.level_offsets[level_i] = offset
            self.cell_sizes[level_i] = 1.0 / (self._base_res * self._per_level_scale**level_i)
            offset += params_in_level

        self.level_offsets[-1] = offset

        self.current_step = 0
        self.smooth_weight_gradients = smooth_weight_gradients
        self.smooth_weight_lambda = smooth_weight_lambda

        params_size = self.level_offsets[-1] * self._num_features
        self.data = dr.zeros(array_type, params_size)

    @annotate("PyHashGrid3D.__call__")
    def __call__(
        self, p: mi.Array3f | mi.Array3h | list[mi.Float32] | list[mi.Float16], active: dr.ArrayBase
    ) -> list[mi.Float32] | list[mi.Float16]:
        # TODO: check if faster as a drjit loop (although then a custom backward is mandatory)
        assert len(p) == 3

        p = self.VectorDf(*p)
        invalid = active & dr.any((p < 0) | (p >= 1))

        result = []
        grid = self.data

        for level_i in range(self._num_levels):
            if not self.grid_lod.is_level_used(level_i, self.current_step):
                # Unused level, output some zeros
                zeros = dr.full(self.StorageFloat, 0, dr.width(p))
                for _ in range(self._num_features):
                    result.append(zeros)
                continue

            scale = self._grid_scale(level_i)
            res = self._grid_resolution(scale)
            level_offset = self.level_offsets[level_i]
            this_level_size = self.level_offsets[level_i + 1] - level_offset
            num_features_fourth = self._num_features - (self._num_features % 4)
            num_features_even = self._num_features - (self._num_features % 2)

            omega_l = self.grid_lod.interpolation_weight(level_i, self.current_step)

            def indexing_function(x, y, z):
                index = x  # Warning: not a copy
                stride = res
                if stride <= this_level_size:
                    index = index + stride * y
                    stride *= res
                if stride <= this_level_size:
                    index = index + stride * z
                    stride *= res

                if this_level_size < stride:
                    index = x ^ (y * mi.UInt32(2654435761)) ^ (z * mi.UInt32(805459861))

                sub_grid_index = level_offset + (index % mi.UInt32(this_level_size))
                return self._num_features * sub_grid_index

            # Unrolls and use packed gathers
            def acc_features(weight: self.StorageFloat, index: mi.UInt32):
                if self.smooth_weight_gradients:
                    weight_smooth = cosine_ramp(weight)
                    weight = weight + self.smooth_weight_lambda * (weight_smooth - dr.detach(weight_smooth))
                for k in range(0, num_features_fourth, 4):
                    v = dr.gather4(self.StorageFloat, grid, index + k, active)
                    values[k] = dr.fma(v[0], weight, values[k])
                    values[k + 1] = dr.fma(v[1], weight, values[k + 1])
                    values[k + 2] = dr.fma(v[2], weight, values[k + 2])
                    values[k + 3] = dr.fma(v[3], weight, values[k + 3])
                for k in range(num_features_fourth, num_features_even, 2):
                    v = dr.gather2(self.StorageFloat, grid, index + k, active)
                    values[k] = dr.fma(v[0], weight, values[k])
                    values[k + 1] = dr.fma(v[1], weight, values[k + 1])
                for k in range(num_features_even, self._num_features):
                    v = dr.gather(self.StorageFloat, grid, index + k, active)
                    values[k] = dr.fma(v, weight, values[k])

            # --- Trilinear interpolation at this level
            # Note: f32 precision, even when StorageFloat is f16
            p_offset: float = 0.0 if self._align_corners else 0.5
            pos = dr.fma(p, scale, p_offset)
            pos0 = dr.floor2int(self.VectorDu, pos)
            pos1 = pos0 + 1

            w1 = pos - pos0
            w0 = self.Float(1.0) - w1

            values = [self.StorageFloat(0.0)] * self._num_features

            weight = self.StorageFloat(w0.x * w0.y * w0.z)
            index = indexing_function(pos0.x, pos0.y, pos0.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w0.x * w0.y * w1.z)
            index = indexing_function(pos0.x, pos0.y, pos1.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w0.x * w1.y * w0.z)
            index = indexing_function(pos0.x, pos1.y, pos0.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w0.x * w1.y * w1.z)
            index = indexing_function(pos0.x, pos1.y, pos1.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w1.x * w0.y * w0.z)
            index = indexing_function(pos1.x, pos0.y, pos0.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w1.x * w0.y * w1.z)
            index = indexing_function(pos1.x, pos0.y, pos1.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w1.x * w1.y * w0.z)
            index = indexing_function(pos1.x, pos1.y, pos0.z)
            acc_features(weight, index)
            weight = self.StorageFloat(w1.x * w1.y * w1.z)
            index = indexing_function(pos1.x, pos1.y, pos1.z)
            acc_features(weight, index)

            for v in values:
                # Since we're not really validating the inputs, let's at least
                # consistently output NaNs for out-of-range inputs.
                # Without this `select`, NaNs would often show up only in the
                # backward pass, which is really difficult to debug.
                v = omega_l * dr.select(invalid, dr.nan, v & active)

                # TODO: reset optimizer state (e.g. Adam) when the level gets enabled?
                if not self.grid_lod.is_level_optimized(level_i, self.current_step):
                    v = dr.detach(v)
                result.append(v)

        return result

    # --------------------

    def step(self, step_i: int, optimizer, rng):
        self.current_step = step_i

    # --------------------

    def dimensionality(self) -> int:
        return 3

    def default_padding_value(self) -> float:
        return 0.0

    def n_dims_encoded(self, n_dims_in: int) -> int:
        return self._num_features * self._num_levels

    def n_params(self) -> int:
        return dr.width(self.data)

    def params(self) -> mi.Float16 | mi.Float32:
        return self.data

    def set_params(self, values: dr.ArrayBase, copy=False) -> None:
        """
        Should pass `values` as a DrJit array (float32 only for now).
        """
        assert dr.width(values) == dr.width(self.data)
        assert type(values) == type(self.data)
        if copy:
            self.data = type(self.data)(values)
        else:
            assert isinstance(values, type(self.data))
            self.data = values

    def hashmap_size(self) -> int:
        return self._hashmap_size

    def num_levels(self) -> int:
        return self._num_levels

    def base_res(self) -> int:
        return self._base_res

    def per_level_scale(self) -> float:
        return self._per_level_scale

    def num_features(self) -> int:
        return self._num_features

    def align_corners(self) -> bool:
        return self._align_corners

    def torchngp_compat(self) -> int:
        return self._torchngp_compat

    def level_offset(self, level_i: int) -> int:
        """
        Helpful to build e.g. debug visualizations by splitting params per level
        Must be called with an index up to and including `num_levels()`.

        Warning: the level offset returned does *not* account for the feature count.
        Each level contains `num_features()` times the difference between the next
        offset entries.
        """
        return self.level_offsets[level_i]

    # --------------------

    def _grid_scale(self, level) -> float:
        """
        The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
        than the number of cells. This is slightly different from the notation in the paper,
        but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
        """
        return dr.exp2(level * self.log2_per_level_scale) * self._base_res - 1.0

    def _grid_resolution(self, scale):
        return int(dr.ceil(scale)) + (0 if self._align_corners else 1)
