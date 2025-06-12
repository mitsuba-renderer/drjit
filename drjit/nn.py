from __future__ import annotations
import drjit
import sys
from dataclasses import dataclass, field

if sys.version_info < (3, 11):
    from typing_extensions import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any, overload
else:
    from typing import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any, overload

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

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, /) -> Tuple[Module, int]:
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

    def alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1) -> Module:
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
        """
        return self._alloc(dtype, size)[0]

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

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1, /) -> Tuple[Module, int]:
        result = []
        for l in self.layers:
            l_new, size = l._alloc(dtype, size)
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
    :math:`k=1/\sqrt{\texttt{out\_features}}`. Call :py:func:`drjit.seed()` prior
    to this step to ensure that weights are always initialized with the same
    values, which can be helpful for hyperpararameter tuning and
    reproducibility.
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

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int = -1, /) -> Tuple[Module, int]:
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
        samples = drjit.rand(Float32, (out_features, in_features))
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

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int = -1, /) -> Tuple[Module, int]:
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

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int = -1, /) -> Tuple[Module, int]:
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


def cosine_ramp(x):
    """ "Smoothed" ramp to help features blend-in without instabilities"""
    return 0.5 * (1.0 - drjit.cos(drjit.pi * x))

def div_round_up(num: int, divisor: int) -> int:
    return (num + divisor - 1) // divisor


def next_multiple(num: int, multiple: int) -> int:
    return multiple * div_round_up(num, multiple)

@dataclass
class HashEncodingConfig:
    """
    This class holds information about a hash based encoding, such as hash grids
    or permutohedral encodings.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """
    dimension: int
    n_levels: int
    n_features_per_level: int
    hashmap_size: int
    base_resolution: int
    per_level_scale: float
    align_corners: bool
    torchngp_compat: bool
    smooth_weight_gradients: bool
    smooth_weight_lambda: float

    def __init__(
        self,
        dimension: int = -1,
        n_levels: int = -1,
        n_features_per_level: int = -1,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None:
        self.dimension = dimension
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.hashmap_size = hashmap_size
        self.base_resolution = base_resolution
        self.per_level_scale = per_level_scale
        self.log2_per_level_scale = drjit.log2(per_level_scale)
        self.align_corners = align_corners
        self.torchngp_compat = torchngp_compat
        self.smooth_weight_gradients = smooth_weight_gradients
        self.smooth_weight_lambda = smooth_weight_lambda


class HashEncoding(Module):
    """
    This class serves as the interface for Hash based encodings.
    """
    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
        "_cell_sizes": list,
    }

    def _alloc(
        self, dtype: Type[drjit.ArrayBase], size: int = -1, /
    ) -> Tuple[Module, int]:

        dimension = self.dimension
        if dimension < 0:
            dimension = size

        n_levels = self.n_levels
        n_features_per_level = self.n_features_per_level

        if dimension < 0 or n_levels < 0 or n_features_per_level < 0:
            raise RuntimeError(
                "The network contains layers with an unspecified "
                "size. You must specify the input size to drjit.nn.Module.alloc()."
            )


        result = type(self)()
        result._config = HashEncodingConfig(
            dimension=dimension,
            hashmap_size=self.hashmap_size,
            n_levels=n_levels,
            base_resolution=self.base_resolution,
            per_level_scale=self.per_level_scale,
            n_features_per_level=n_features_per_level,
            align_corners=self.align_corners,
            torchngp_compat=self.torchngp_compat,
            smooth_weight_gradients=self.smooth_weight_gradients,
            smooth_weight_lambda=self.smooth_weight_lambda,
        )
        result._alloc_internal(dtype)

        return result, result.out_features

    def n_params(self) -> int:
        return drjit.width(self.data)

    def set_params(self, values: drjit.ArrayBase, copy=False) -> None:
        """
        Should pass `values` as a DrJit array (float32 only for now).
        """
        assert drjit.width(values) == drjit.width(self.data)
        assert type(values) is type(self.data)
        if copy:
            self.data = type(self.data)(values)
        else:
            assert isinstance(values, type(self.data))
            self.data = values

    @property
    def dimension(self) -> int:
        """
        The dimensionality of this hash encoding.
        """
        return self._config.dimension

    @property
    def hashmap_size(self) -> int:
        """
        The hashmap size provided when constructing this encoding.
        """
        return self._config.hashmap_size

    @property
    def n_levels(self) -> int:
        """
        The number of levels in this hash encoding.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._config.n_levels

    @property
    def base_resolution(self) -> int:
        """
        The resolution of the 0th level.
        """
        return self._config.base_resolution

    @property
    def per_level_scale(self) -> float:
        """
        The per level scale factor, with which the scale of each level grows.
        """
        return self._config.per_level_scale

    @property
    def n_features_per_level(self) -> int:
        """
        The number of features per level.
        The actual number of output features of this encoding is determined by
        ``n_level * n_features_per_level``.
        """
        return self._config.n_features_per_level

    @property
    def align_corners(self) -> bool:
        """
        If the corners of the hashgrid should be aligned to the edges of its domain.
        """
        return self._config.align_corners

    @property
    def torchngp_compat(self) -> int:
        return self._config.torchngp_compat

    @property
    def smooth_weight_gradients(self) -> bool:
        return self._config.smooth_weight_gradients

    @property
    def smooth_weight_lambda(self) -> float:
        return self._config.smooth_weight_lambda

    @property
    def out_features(self) -> int:
        return self._config.n_features_per_level *  self.n_levels

    def _init_types(self, p):
        """
        Gets the type aliases used for this hash encoding computation, given a
        position value ``p``.
        """
        dtype = self.dtype
        mod = sys.modules[dtype.__module__]
        self.PCG32 = mod.PCG32
        self.StorageFloat = dtype
        self.UInt32 = drjit.uint32_array_t(dtype)
        self.Int32 = drjit.int32_array_t(dtype)
        self.ArrayXu = mod.ArrayXu
        self.ArrayXi = mod.ArrayXi
        self.StorageFloatXf = mod.ArrayXf16 if drjit.is_half_v(dtype) else mod.ArrayXf

        if isinstance(p, CoopVec):
            p = list(p)

        if isinstance(p, list):
            self.PositionFloat = drjit.leaf_t(p[0])
        else:
            self.PositionFloat = drjit.leaf_t(p)

        self.PositionFloatXf = mod.ArrayXf16 if drjit.is_half_v(self.PositionFloat) else mod.ArrayXf

    def _acc_features(self, level_i, weight, index, values: list, active):
        """
        Accumulates the ``self.num_features`` features at the given index.
        This function tries to use packet gather operations if possible, to improve
        the performance of the backward pass, as atomic packet scatters
        perform better than their non packeted counterparts.
        """

        # We define a gather function to be able to handle MatrixViews
        if isinstance(self.data, MatrixView):
            assert self.data.shape[1] == 1
            assert self.data.stride == 1
            def gather(type, source, index, mask, **kwargs):
                return drjit.gather(type, source.buffer, self.data.offset + index, mask, **kwargs)
        else:
            def gather(type, source, index, mask, **kwargs):
                return drjit.gather(type, source, index, mask, **kwargs)

        if self.smooth_weight_gradients:
            weight_smooth = cosine_ramp(weight)
            weight = weight + self.smooth_weight_lambda * (weight_smooth - drjit.detach(weight_smooth))

        v = gather(
            self.StorageFloatXf,
            self.data,
            index // self.n_features_per_level,
            active,
            shape=(self.n_features_per_level, drjit.width(index)),
        )

        for k in range(0, self.n_features_per_level):
            values[level_i * self.n_features_per_level + k] = drjit.fma(
                v[k], weight, values[level_i * self.n_features_per_level + k]
            )

    def indexing_function(self, key, level_i):
        raise NotImplementedError()


class HashGridEncoding(HashEncoding):
    """

    This encoding implements a Multiresolution Hash Grid. For every resolution level,
    this encoding looks up the :math:`2^D` vertices of the cell in which the input point is
    located, performs multilinear interpolation, and concatenates the features accross
    all resolution levels.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """
    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
        "_cell_sizes": list,
        "_grid_offsets": list,
    }

    @overload
    def __init__(
        self,
        dimension: int = -1,
        n_levels: int = -1,
        n_features_per_level: int = -1,
        *,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        align_corners: bool = False,
        torchngp_compat: bool = False,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(self) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        """
        This encoding is based on the Multiresolution Hash Grid encoding introduced
        in :cite:`mueller2022instant`.

        Args:
            dimension: The dimensionality of the hash encoding. This corresponds to
                the number of input features the encoding can take.
            n_levels: Hash encodings generally make use of multiple levels of the same
                encoding with different scales. This parameter specifies the number of
                levels used by this encoding.
            n_features_per_level: More than one feature can be stored in a vertex per
                level. This value specifies how many, and the number of output features
                of the hash encoding layer is given by ``n_levels * n_features_per_level``.
                This value should always be a multiple of two, in order to ensure efficient
                gradient backpropagation.
            hashmap_size: Specifies the maximal number of parameters per level of the
                hash encoding. HashGrids will use a dense grid lookup for layers with
                a low enough scale, and use less than ``hashmap_size`` number of parameters
                per level.
            base_resolution: The scale factor of the 0th layer in the hash encoding.
            per_level_scale: To calculate the scale of a layer, the scale of the previous
                layer is multiplied by this value.
            align_corners: If this value is ``True``, the simplex vertices are aligned
                with the domain of the encoding [0, 1].
            smooth_weight_gradients: whether to smooth the gradients of the weights
                by using a straight-through estimator.
            smooth_weight_lambda: the value of lambda used for the straight-through estimator.
        """
        if len(args) == 0:
            self._config = HashEncodingConfig()
        else:
            self._config = HashEncodingConfig(*args, **kwargs)

    def __call__(
        self, p: drjit.ArrayBase | list[drjit.ArrayBase] | CoopVec, active=True
    ) -> CoopVec:
        self._init_types(p)

        if isinstance(p, list) or isinstance(p, CoopVec):
            p = self.PositionFloatXf(p)

        values = [self.StorageFloat(0.0)] * self.n_features_per_level * self.n_levels

        for level_i in range(self.n_levels):
            scale = self._grid_scale(level_i)

            p_offset: float = 0.0 if self.align_corners else 0.5
            pos = drjit.fma(p, scale, p_offset)

            pos0 = self.ArrayXu(drjit.floor(pos))

            w1 = pos - pos0
            w0 = 1.0 - w1

            def acc(offset: self.ArrayXu):
                """
                Given one of the ``2**self.dimensionality()`` vertices, this function
                calculates the grid position and interpolation weight and accumulates
                the features into the ``values`` field.
                """
                pos_grid = pos0 + self.ArrayXu(offset)
                weight = drjit.select(self.ArrayXu(offset) == 0, w0, w1)
                weight = drjit.prod(weight, axis=0)

                index = self.indexing_function(pos_grid, level_i)
                self._acc_features(level_i, weight, index, values, active)

            for offset in self._grid_offsets:
                acc(self.ArrayXu(offset))

        values = [v & active for v in values]

        return CoopVec(*values)

    def indexing_function(self, key, level_i):
        """
        This function is used to index the underlying data array of the
        hash grid given a grid position.
        """

        scale = self._grid_scale(level_i)
        res = self._grid_resolution(scale)
        level_offset = self._level_offsets[level_i]
        this_level_size = self._level_offsets[level_i + 1] - level_offset

        indexing_primes = [1, self.UInt32(2654435761), self.UInt32(805459861)]

        stride = 1
        index = self.UInt32(0)
        for d in range(self.dimension):
            index += key[d] * stride
            stride *= res

            if stride > this_level_size:
                break

        if this_level_size < stride:
            index = self.UInt32(0)
            for d in range(0, self.dimension):
                index ^= key[d] * indexing_primes[d]

        sub_grid_index = level_offset + (index % self.UInt32(this_level_size))
        return self.n_features_per_level * sub_grid_index

    def _alloc_internal(self, dtype: Type[drjit.ArrayBase]):
        self.dtype = drjit.leaf_t(dtype)

        assert (self.hashmap_size % 8) == 0, (
            f"Invalid hashmap size {self.hashmap_size}, must be a multiple of 8."
        )

        self._level_offsets = [None] * (self.n_levels + 1)
        self._cell_sizes = [0.0] * self.n_levels
        max_params = drjit.scalar.UInt32(0xFFFFFFFF) // 2
        offset = 0

        for level_i in range(self.n_levels):
            res = self._grid_resolution(self._grid_scale(level_i))
            stride = drjit.power(float(res), self.dimension)

            params_in_level = (
                int(max_params) if (stride > max_params) else (res**self.dimension)
            )
            params_in_level = next_multiple(params_in_level, 8)
            params_in_level = min(params_in_level, self.hashmap_size)

            self._level_offsets[level_i] = offset
            self._cell_sizes[level_i] = 1.0 / (
                self.base_resolution * self.per_level_scale**level_i
            )
            offset += params_in_level

        self._level_offsets[-1] = offset

        params_size = self._level_offsets[-1] * self.n_features_per_level
        self.data = drjit.zeros(dtype, params_size)

        # Stores the pattern of offsets used to index the 2** n corners of a voxel
        self._grid_offsets = [
            [(i >> j) & 1 for j in range(self.dimension)]
            for i in range(2**self.dimension)
        ]

    def _grid_scale(self, level) -> float:
        """
        The -1 means that `base_resolution` refers to the number of grid _vertices_ rather
        than the number of cells. This is slightly different from the notation in the paper,
        but results in nice, power-of-2-scaled parameter grids that fit better into cache lines.
        """
        return (
            drjit.exp2(level * self._config.log2_per_level_scale) * self.base_resolution
            - 1.0
        )

    def _grid_resolution(self, scale) -> int:
        return (
            int(drjit.ceil(scale))
            + (0 if self.align_corners else 1)
            + (1 if self.torchngp_compat else 0)
        )

    def __repr__(self) -> str:
        return (
            f"HashGrid(\n"
            f"    dtype={self.dtype},\n"
            f"    dimension={self.dimension},\n"
            f"    hashmap_size={self.hashmap_size},\n"
            f"    n_levels={self.n_levels},\n"
            f"    n_features_per_level={self.n_features_per_level},\n"
            f"    base_resolution={self.base_resolution},\n"
            f"    per_level_scale={self.per_level_scale},\n"
            f"    align_corners={self.align_corners},\n"
            f"    torchngp_compat={self.torchngp_compat},\n"
            f"    smooth_weight_gradients={self.smooth_weight_gradients},\n"
            f"    smooth_weight_lambda={self.smooth_weight_lambda}\n"
            ")"
        )

    # --------------------

    def default_padding_value(self) -> float:
        return 0.0

    def n_dims_encoded(self, n_dims_in: int) -> int:
        return self.n_features_per_level * self.n_levels

    # --------------------

    def level_offset(self, level_i: int) -> int:
        """
        Helpful to build e.g. debug visualizations by splitting params per level
        Must be called with an index up to and including `n_levels`.

        Warning: the level offset returned does *not* account for the feature count.
        Each level contains `num_features()` times the difference between the next
        offset entries.
        """
        return self._level_offsets[level_i]


class SimplifiedPermutohedralEncoding(HashEncoding):
    """
    This encoding is based on the encoding presented in :cite:`rosu2023permutosdf`.
    Whereas hash grid encodings use a grid lattice, this uses a permutohedral lattice,
    where the simplexes are consist of triangles, tetrahedrons etc. The main advantage
    is that the number of vertices per simplex and therefore the number of memory
    lookups per sample per layer grow linear with the number of dimensions.
    In this implementation we make a few simplification, and skip the elevation to a
    hyperplane in d + 1 dimensional space. It instead performs the sorting and interpolation
    steps in d-dimensional space.

    Args:
        dimension: The dimensionality of the hash encoding. This corresponds to
            the number of input features the encoding can take.
        n_levels: Hash encodings generally make use of multiple levels of the same
            encoding with different scales. This parameter specifies the number of
            levels used by this encoding.
        n_features_per_level: More than one feature can be stored in a vertex per
            level. This value specifies how many, and the number of output features
            of the hash encoding layer is given by ``n_levels * n_features_per_level``.
            This value should always be a multiple of two, in order to ensure efficient
            gradient backpropagation.
        hashmap_size: Specifies the maximal number of parameters per level of the
            hash encoding. HashGrids will use a dense grid lookup for layers with
            a low enough scale, and use less than ``hashmap_size`` number of parameters
            per level.
        base_resolution: The scale factor of the 0th layer in the hash encoding.
        per_level_scale: To calculate the scale of a layer, the scale of the previous
            layer is multiplied by this value.
        align_corners: If this value is ``True``, the simplex vertices are aligned
            with the domain of the encoding [0, 1].
        smooth_weight_gradients: whether to smooth the gradients of the weights
            by using a straight-through estimator.
        smooth_weight_lambda: the value of lambda used for the straight-through estimator.
    """

    DRJIT_STRUCT = {
        "data": drjit.ArrayBase,
        "dtype": type,
        "_config": HashEncodingConfig,
        "_level_offsets": list,
    }

    @overload
    def __init__(
        self,
        dimension: int = -1,
        n_levels: int = -1,
        n_features_per_level: int = -1,
        *,
        hashmap_size: int = 2**19,
        base_resolution: int = 16,
        per_level_scale: float = 2,
        smooth_weight_gradients: bool = False,
        smooth_weight_lambda: float = 1.0,
    ) -> None: ...

    @overload
    def __init__(self) -> None: ...

    def __init__(self, *args, **kwargs) -> None:
        if len(args) == 0:
            self._config = HashEncodingConfig()
        else:
            self._config = HashEncodingConfig(*args, **kwargs)

    def __call__(
        self, p: drjit.ArrayBase | list[drjit.ArrayBase] | CoopVec, active=True
    ) -> Any:
        self._init_types(p)

        if isinstance(p, list) or isinstance(p, CoopVec):
            p = self.PositionFloatXf(p)

        values = [self.StorageFloat(0.0)] * self.n_features_per_level * self.n_levels

        for level_i in range(self.n_levels):
            scale = self._level_scale(level_i)

            # ---- Apply scaling factor

            x = p * scale

            base = self.ArrayXi(drjit.floor(x))
            fract = x - base

            # ---- Insertion sort fract into xk and indices xki
            xk = self.PositionFloatXf(fract)
            xki = self.ArrayXi([i for i in range(self.dimension)])

            for i in range(self.dimension):
                for j in range(i, self.dimension):
                    xk_i = self.PositionFloat(xk[i])
                    xk_j = self.PositionFloat(xk[j])
                    swap = xk_i > xk_j
                    xk[i] = drjit.select(swap, xk_j, xk_i)
                    xk[j] = drjit.select(swap, xk_i, xk_j)

                    xki_i = self.Int32(xki[i])
                    xki_j = self.Int32(xki[j])
                    xki[i] = drjit.select(swap, xki_j, xki_i)
                    xki[j] = drjit.select(swap, xki_i, xki_j)

            # ---- Calculate Barycentric coordinates and grid offset vectors
            grid_offsets = [None] * (self.dimension + 1)
            weights = drjit.zeros(
                self.PositionFloatXf, shape=(self.dimension + 1, drjit.width(p))
            )

            for rank in range(self.dimension):
                weights[rank] = xk[rank] - (xk[rank - 1] if rank > 0 else 0)
                grid_offset = drjit.select(
                    (fract > xk[rank])
                    | (fract == xk[rank])
                    & (self.ArrayXi([i for i in range(self.dimension)]) >= xki[rank]),
                    1,
                    0,
                )
                grid_offsets[rank] = grid_offset

            weights[self.dimension] = 1 - xk[self.dimension - 1]
            grid_offsets[self.dimension] = drjit.zeros(
                self.ArrayXi, shape=(self.dimension, drjit.width(p))
            )

            # ---- Accumulate features for each offset vector
            for rank in range(self.dimension + 1):
                offset = grid_offsets[rank]

                pos_grid = base + self.ArrayXu(offset)
                weight = weights[rank]

                index = self.indexing_function(pos_grid, level_i)
                self._acc_features(level_i, weight, index, values, active)

        values = [v & active for v in values]

        return CoopVec(*values)

    def _alloc_internal(self, dtype: Type[drjit.ArrayBase]):
        self.dtype = drjit.leaf_t(dtype)

        assert (self.hashmap_size % 8) == 0, (
            f"Invalid hashmap size {self.hashmap_size}, must be a multiple of 8."
        )

        self._level_offsets = [None] * (self.n_levels + 1)

        offset = 0
        for i in range(self.n_levels):
            params_in_level = self.hashmap_size
            self._level_offsets[i] = offset
            offset += params_in_level

        self._level_offsets[-1] = offset

        params_size = self._level_offsets[-1] * self.n_features_per_level
        self.data = drjit.zeros(dtype, params_size)

    def _level_scale(self, level) -> float:
        return (
            drjit.exp2(level * self._config.log2_per_level_scale) * self.base_resolution
        )

    def _resolution(self, scale) -> int:
        return int(drjit.ceil(scale))

    def indexing_function(self, key, level_i):
        """
        This function is used to index the underlying data array of the
        hash grid given a grid position.
        """

        def base_convert_hash(key: self.ArrayXu) -> self.UInt32:
            k = self.UInt32(0)
            for dim in range(self.dimension):
                k += key[dim]
                k *= 2531011
            return k

        level_offset = self._level_offsets[level_i]
        this_level_size = self._level_offsets[level_i + 1] - level_offset

        index = base_convert_hash(key)
        sub_grid_index = level_offset + (index % self.UInt32(this_level_size))
        return self.n_features_per_level * sub_grid_index

    def __repr__(self):
        return (
            f"SimplifiedPermutohedral(\n"
            f"    dtype={self.dtype},\n"
            f"    dimension={self.dimension},\n"
            f"    hashmap_size={self.hashmap_size},\n"
            f"    n_levels={self.n_levels},\n"
            f"    n_features_per_level={self.n_features_per_level},\n"
            f"    base_resolution={self.base_resolution},\n"
            f"    per_level_scale={self.per_level_scale},\n"
            f"    smooth_weight_gradients={self.smooth_weight_gradients},\n"
            f"    smooth_weight_lambda={self.smooth_weight_lambda}\n"
            ")"
        )

