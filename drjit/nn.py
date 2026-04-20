from __future__ import annotations
import drjit
import sys
from collections.abc import MutableMapping
from .hashgrid import HashGridEncoding, PermutoEncoding
from . import hashgrid

if sys.version_info < (3, 11):
    from typing_extensions import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any, Dict, Iterator, TypeVar
else:
    from typing import Tuple, Sequence, Union, Type, TypeAlias, Optional, Any, Dict, Iterator, TypeVar

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

# Type variable for activation-like modules whose ``__call__`` preserves the
# input type (either :class:`CoopVec` or a Dr.Jit tensor).
_InputT = TypeVar('_InputT', CoopVec, drjit.ArrayBase)


def _walk_params(root: 'Module', prefix: str):
    """Walk a Module's ``DRJIT_STRUCT`` tree and collect parameter leaves.

    Returns ``(params, packed_key)`` where ``params`` maps dotted paths to
    ``(owner, attr)`` pairs that locate each leaf on its parent module, and
    ``packed_key`` is the key whose entry backs a :func:`pack`-produced
    buffer (or ``None`` in tensor mode).
    """
    params: Dict[str, Tuple[Any, str]] = {}
    packed_key: Optional[str] = None

    def visit_module(mod: 'Module', sub_prefix: str) -> None:
        nonlocal packed_key
        if getattr(mod, '_packed_buffer', None) is not None:
            key = f'{sub_prefix}.weights' if sub_prefix else 'weights'
            params[key] = (mod, '_packed_buffer')
            packed_key = key
            return

        struct = getattr(type(mod), 'DRJIT_STRUCT', None)
        if struct is None:
            return

        for name in struct:
            sub_key = f'{sub_prefix}.{name}' if sub_prefix else name
            visit(getattr(mod, name, None), sub_key, mod, name)

    def visit(value: Any, key: str, owner: Any, attr: Optional[str]) -> None:
        if value is None:
            return
        if isinstance(value, Module):
            visit_module(value, key)
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                visit(item, f'{key}.{i}', owner=None, attr=None)
        elif isinstance(value, MatrixView):
            return
        elif isinstance(value, drjit.ArrayBase) and attr is not None:
            params[key] = (owner, attr)

    visit_module(root, prefix)
    return params, packed_key

class Module(MutableMapping):
    """
    This is the base class of a modular set of operations that make
    the specification of neural network architectures more convenient.

    Module subclasses are :ref:`PyTrees <pytrees>`, which means that various
    Dr.Jit operations can automatically traverse them.

    Every allocated :class:`Module` additionally behaves as a
    :class:`collections.abc.MutableMapping` keyed by dotted parameter paths
    (e.g., ``'layers.0.weights'``). This mirrors the :class:`drjit.opt.Optimizer`
    interface and enables symmetric parameter transfer:

    .. code-block:: python

       opt = Adam(lr=1e-3)
       opt.update(net)       # pull every parameter into the optimizer (once)

       for i in range(n):
           net.update(opt)   # push optimizer state back into the net
           ...

    After attach, the optimizer is the source of truth: ``opt.update(net)``
    must not be called again. On a packed module the mapping exposes a single
    ``'weights'`` entry whose underlying buffer is referenced by the per-layer
    :class:`MatrixView` instances; writes to that entry are in-place so the
    views remain valid.

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
       net = nn.pack(net, layout='training')

    Network evaluation expects a :ref:`cooperative vector <coop_vec>` as input
    (i.e., ``net(nn.CoopVec(...))``) and returns another cooperative vector.
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

    def alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1,
              rng: Optional[drjit.random.Generator] = None) -> Module:
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

    # ---- MutableMapping interface ---------------------------------------

    def _params_cache(self) -> Dict[str, Tuple[Any, str]]:
        d = getattr(self, '_params', None)
        if d is None:
            params, packed_key = _walk_params(self, getattr(self, 'prefix', ''))
            self._params = params
            self._packed_key = packed_key
            d = params
        return d

    def __getitem__(self, key: str) -> drjit.ArrayBase:
        slot = self._params_cache().get(key)
        if slot is None:
            raise KeyError(key)
        owner, attr = slot
        return getattr(owner, attr)

    def __setitem__(self, key: str, value: drjit.ArrayBase) -> None:
        slot = self._params_cache().get(key)
        if slot is None:
            raise KeyError(key)
        owner, attr = slot

        if key == self._packed_key:
            # MatrixViews on the sub-layers point into this buffer's memory;
            # a reference swap would leave them dangling.
            getattr(owner, attr)[:] = value
            return

        existing = getattr(owner, attr)
        if type(value) is not type(existing):
            value = type(existing)(value)
        setattr(owner, attr, value)

    def __delitem__(self, key: str) -> None:
        raise TypeError(
            "drjit.nn.Module: parameter keys cannot be deleted via the "
            "mapping interface."
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._params_cache())

    def __len__(self) -> int:
        return len(self._params_cache())

    def __contains__(self, key: object) -> bool:
        return key in self._params_cache()

class Sequential(Module):
    """
    This model evaluates provided arguments ``arg[0]``, ``arg[1]``, ..., in sequence.

    The optional ``prefix`` keyword is prepended to every key exposed through
    the :class:`MutableMapping` interface (e.g., ``'mlp.layers.0.weights'``
    instead of ``'layers.0.weights'``). This is useful when sharing a single
    optimizer across multiple networks. The prefix is retained through
    :func:`pack`.
    """
    DRJIT_STRUCT = { 'layers' : tuple, 'prefix': str }

    layers: tuple[Module, ...]
    prefix: str

    def __init__(self, *args: Module, prefix: str = ''):
        self.layers = args
        self.prefix = prefix

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        for l in self.layers:
            arg = l(arg)
        return arg

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        result = []
        for l in self.layers:
            l_new, size = l._alloc(dtype, size, rng)
            result.append(l_new)
        return Sequential(*result, prefix=self.prefix), size

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

    Accepts both :class:`CoopVec` and tensor inputs.
    """

    DRJIT_STRUCT = { }
    def __call__(self, arg: _InputT, /) -> _InputT:
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

    Accepts both :class:`CoopVec` and tensor inputs.
    """

    DRJIT_STRUCT = { 'negative_slope': Union[float, drjit.ArrayBase] }
    def __init__(self, negative_slope: Union[float, drjit.ArrayBase] = 1e-2):
        self.negative_slope = negative_slope

    def __call__(self, arg: _InputT, /) -> _InputT:
        return drjit.maximum(arg, 0) + drjit.minimum(arg, 0.0) * self.negative_slope


class Exp2(Module):
    r"""
    Applies the base-2 exponential function to each component.

    .. math::

       \mathrm{Exp2}(x) = 2^x

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    Accepts both :class:`CoopVec` and tensor inputs.
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: _InputT, /) -> _InputT:
        return drjit.exp2(arg)

class Exp(Module):
    r"""
    Applies the exponential function to each component.

    .. math::

       \mathrm{Exp}(x) = e^x

    Accepts both :class:`CoopVec` and tensor inputs.
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: _InputT, /) -> _InputT:
        return drjit.exp2(arg * (1 / drjit.log(2)))

class Tanh(Module):
    r"""
    Applies the hyperbolic tangent function to each component.

    .. math::

       \mathrm{Tanh}(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}

    On the CUDA backend, this function directly maps to an efficient native GPU instruction.
    Accepts both :class:`CoopVec` and tensor inputs.
    """
    DRJIT_STRUCT = { }
    def __call__(self, arg: _InputT, /) -> _InputT:
        return drjit.tanh(arg)

class ScaleAdd(Module):
    r"""
    Scale the input by a fixed scale and apply an offset.

    Note that ``scale`` and ``offset`` are assumed to be constant (i.e., not trainable).

    .. math::

       \mathrm{ScaleAdd}(x) = x\cdot\texttt{scale} + \texttt{offset}

    Accepts both :class:`CoopVec` and tensor inputs.
    """
    DRJIT_STRUCT = {'scale': Union[None, float, int, drjit.ArrayBase],
                    'offset': Union[None, float, int, drjit.ArrayBase]}
    def __init__(self, scale: Union[float, int, drjit.ArrayBase, None] = None,
                 offset: Union[float, int, drjit.ArrayBase, None] = None):
        self.scale = scale
        self.offset = offset
    def __call__(self, arg: _InputT, /) -> _InputT:
        if not self.scale or not self.offset:
            raise Exception("drjit.nn.ScaleAdd(): you must set a scale and offset!")
        return drjit.fma(arg, self.scale, self.offset)

class Cast(Module):
    """
    Cast the input to a different precision. Should be instantiated with the
    desired element type, e.g. ``Cast(drjit.cuda.ad.Float32)``. Accepts both
    :class:`CoopVec` and tensor inputs.
    """
    DRJIT_STRUCT = { 'dtype': Optional[Type[drjit.ArrayBase]] }
    def __init__(self, dtype: Optional[Type[drjit.ArrayBase]] = None):
        self.dtype = dtype
    def __call__(self, arg: _InputT, /) -> _InputT:
        if isinstance(arg, CoopVec):
            return cast(arg, self.dtype)
        return drjit.tensor_t(self.dtype)(arg)
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

    def __call__(self, arg: _InputT, /) -> _InputT:
        w = self.weights
        if w is None:
            raise RuntimeError(
                "drjit.nn.Linear: uninitialized network. Call "
                "'net = net.alloc(<Tensor type>)' to initialize the weight "
                "storage first."
            )

        if isinstance(arg, CoopVec):
            if not isinstance(w, MatrixView):
                raise RuntimeError(
                    "drjit.nn.Linear: cooperative-vector evaluation requires "
                    "packed weights. Call 'drjit.nn.pack(net)' to transform "
                    "the network into a cooperative-vector layout."
                )
            return matvec(w, arg, self.bias)

        if isinstance(w, MatrixView):
            raise RuntimeError(
                "drjit.nn.Linear: tensor-mode evaluation requires unpacked 2D "
                "weights. Either wrap the input as 'nn.CoopVec(...)' or "
                "construct a fresh unpacked module (without calling "
                "'drjit.nn.pack()')."
            )

        y = w @ arg
        if self.bias is not None:
            bias = self.bias
            y = y + drjit.reshape(bias, (bias.shape[0], 1))
        return y

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        if not drjit.is_tensor_v(dtype):
            raise TypeError(f"Linear layer requires a Tensor type, but got {dtype}")

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

def _sincos_tri(t):
    """Triangular sine/cosine approximations with period 1."""
    s = t - .25
    return (drjit.fma(drjit.abs(s - drjit.round(s)), -4, 1),
            drjit.fma(drjit.abs(t - drjit.round(t)), -4, 1))


def _tri_encode_coopvec(arg, O, shift):
    r = []
    for a in list(arg):
        for i in range(O):
            r += _sincos_tri(drjit.fma(a, 2 ** i, shift * i))
    return CoopVec(r)


def _tri_encode_tensor(arg, O, shift):
    C, N = arg.shape
    tensor_tp = type(arg)
    out = drjit.empty(tensor_tp, (C * 2 * O, N))
    for i in range(O):
        s, c = _sincos_tri(drjit.fma(arg, 2.0 ** i, shift * i))
        out[2 * i::2 * O, :] = s.array
        out[2 * i + 1::2 * O, :] = c.array
    return out


def _sin_encode_rotations(shift, O):
    """Precompute the per-octave rotation following each double-angle step.

    Going from octave i-1 to i, the target angle is
    ``θ_i = 2 θ_{i-1} + β_i`` with ``β_i = 2π · s · (2 - i)``. Returns the
    list ``[(sin β_1, cos β_1), ..., (sin β_{O-1}, cos β_{O-1})]``, or
    ``None`` when the rotation is identically zero (``shift == 0`` collapses
    to the pure double-angle recurrence).
    """
    if O <= 1 or shift == 0:
        return None
    return [drjit.sincos(2 * drjit.pi * shift * (2 - i)) for i in range(1, O)]


def _sin_encode_step(s, c, rot):
    # Double-angle on (s, c), then an optional fixed rotation by the next β_i.
    s2 = 2 * s
    s, c = s2 * c, drjit.fma(-s2, s, 1)
    if rot is not None:
        sb, cb = rot
        s, c = drjit.fma(s, cb, c * sb), drjit.fma(c, cb, -s * sb)
    return s, c


def _sin_encode_coopvec(arg, O, shift):
    if O == 0:
        return CoopVec([])
    rots = _sin_encode_rotations(shift, O)
    r = []
    for a in list(arg):
        s, c = drjit.sincos(a * (2 * drjit.pi))
        r += [s, c]
        for i in range(1, O):
            s, c = _sin_encode_step(s, c, rots[i - 1] if rots else None)
            r += [s, c]
    return CoopVec(r)


def _sin_encode_tensor(arg, O, shift):
    C, N = arg.shape
    tensor_tp = type(arg)
    if O == 0:
        return drjit.zeros(tensor_tp, (0, N))
    out = drjit.empty(tensor_tp, (C * 2 * O, N))
    s, c = drjit.sincos(arg * (2 * drjit.pi))
    out[0::2 * O, :] = s.array
    out[1::2 * O, :] = c.array
    rots = _sin_encode_rotations(shift, O)
    for i in range(1, O):
        s, c = _sin_encode_step(s, c, rots[i - 1] if rots else None)
        out[2 * i::2 * O, :] = s.array
        out[2 * i + 1::2 * O, :] = c.array
    return out


class TriEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using
    triangular sine and cosine approximations of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin_\triangle(2^0\,x + 0\cdot s)\\
           \cos_\triangle(2^0\,x + 0\cdot s)\\
           \vdots\\
           \sin_\triangle(2^{n-1}\, x + (n-1)\cdot s)\\
           \cos_\triangle(2^{n-1}\, x + (n-1)\cdot s)
       \end{bmatrix}

    where

    .. math::

       \cos_\triangle(x) = 1-4\left|x-\mathrm{round}(x)\right|

    and

    .. math::

       \sin_\triangle(x) = \cos_\triangle(x-1/4).

    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components coincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional ``shift`` parameter :math:`s` (in *fractional
    periods*, so that ``shift=0.25`` is a quarter period) to phase-shift the
    :math:`i`-th octave by :math:`i\cdot s` and avoid this.

    Accepts both :class:`CoopVec` and 2D tensor inputs (batched evaluation).
    For a tensor of shape ``(C, N)`` with ``N`` independent samples, the
    output has shape ``(2\,n\,C, N)``.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/tri_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/tri_encode_dark.svg
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

    def __call__(self, arg: _InputT, /) -> _InputT:
        if isinstance(arg, CoopVec):
            return _tri_encode_coopvec(arg, self.octaves, self.shift)
        return _tri_encode_tensor(arg, self.octaves, self.shift)


class SinEncode(Module):
    r"""
    Map an input onto a higher-dimensional space by transforming it using sines
    and cosines of an increasing frequency.

    .. math::

       x\mapsto \begin{bmatrix}
           \sin\bigl(2\pi(2^0\,x + 0\cdot s)\bigr)\\
           \cos\bigl(2\pi(2^0\,x + 0\cdot s)\bigr)\\
           \vdots\\
           \sin\bigl(2\pi(2^{n-1}\,x + (n-1)\cdot s)\bigr)\\
           \cos\bigl(2\pi(2^{n-1}\,x + (n-1)\cdot s)\bigr)
       \end{bmatrix}

    The value :math:`n` refers to the number of *octaves*. This layer increases
    the dimension by a factor of :math:`2n`.

    Note that this encoding has period 1. If your input exceeds the interval
    :math:`[0, 1]`, it is advisable that you reduce it to this range to avoid
    losing information.

    Minima/maxima of higher frequency components coincide on a regular
    lattice, which can lead to reduced fitting performance at those locations.
    Specify the optional ``shift`` parameter :math:`s` (in *fractional
    periods*, so that ``shift=0.25`` is a quarter period) to phase-shift the
    :math:`i`-th octave by :math:`i\cdot s` and avoid this.

    Accepts both :class:`CoopVec` and 2D tensor inputs (batched evaluation).
    For a tensor of shape ``(C, N)`` with ``N`` independent samples, the
    output has shape ``(2\,n\,C, N)``.

    The following plot shows the first two octaves applied to the linear
    function on :math:`[0, 1]` (without shift).

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/sin_encode_light.svg
      :class: only-light
      :width: 600px
      :align: center

    .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/sin_encode_dark.svg
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
        r = SinEncode(self.octaves, self.shift)
        r.channels = size
        return r, size * self.octaves * 2

    def __repr__(self) -> str:
        return f'SinEncode(octaves={self.octaves}, shift={self.shift}, in_channels={self.channels}, out_features={self.channels*self.octaves*2})'

    def __call__(self, arg: _InputT, /) -> _InputT:
        if isinstance(arg, CoopVec):
            return _sin_encode_coopvec(arg, self.octaves, self.shift)
        return _sin_encode_tensor(arg, self.octaves, self.shift)

class HashEncodingLayer(Module):
    """
    Simple layer wrapping a hash encoding like :py:class:`drjit.nn.HashGridEncoding`
    or :py:class:`drjit.nn.PermutoEncoding`.

    Note that the parameters of the encoding will not be included when packing the
    network, as the data representations are generally incompatible. You must initialize
    the encoding parameters separately.
    """

    def __init__(
        self,
        encoding: hashgrid.HashEncoding,
    ) -> None:
        self.encoding = encoding

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, rng: drjit.random.Generator, /) -> Tuple[Module, int]:
        layer = HashEncodingLayer(self.encoding)
        size = self.encoding.n_features_per_level  * self.encoding.n_levels
        return layer, size

    def __call__(self, arg: CoopVec, /) -> CoopVec:
        arg = list(arg)
        return CoopVec(self.encoding(arg))

    @property
    def data(self):
        self.encoding.data

    def __repr__(self) -> str:
        return self.encoding.__repr__()
