from __future__ import annotations
import drjit
from typing import Tuple, Sequence, Union, Type, TypeAlias, Optional

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

class Model:
    """
    Base class
    """
    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        tp = type(self)
        raise NotImplementedError(f"{tp.__module__}.{tp.__name__}.__call__() implementation is missing.")

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int, /) -> Tuple[Model, int]:
        return self, size

    def alloc(self, dtype: Type[drjit.ArrayBase], size: int, /) -> Model:
        return self._alloc(dtype, size)[0]

    def __repr__(self) -> str:
        tp = type(self)
        return f"{tp.__module__}.{tp.__name__}()"

class Sequential(Model, Sequence[Model]):
    DRJIT_STRUCT = { 'layers' : tuple }

    layers: tuple[Model, ...]

    def __init__(self, *args: Model):
        self.layers = args

    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        for l in self.layers:
            arg = l(arg)
        return arg

    def _alloc(self, dtype: Type[drjit.ArrayBase], size: int = -1, /) -> Tuple[Model, int]:
        result = []
        for l in self.layers:
            l_new, size = l._alloc(dtype, size)
            result.append(l_new)
        return Sequential(*result), size

    def __len__(self):
        return len(self.layers)

    def __getitem__(self, index: Union[int], /) -> Model: # type: ignore
        return self.layers[index]

    def __repr__(self) -> str:
        s = 'drjit.Sequential(\n'
        n = len(self.layers)
        for i in range(n):
            s += '  ' + repr(self.layers[i]).replace('\n', '\n  ')
            if i + 1 < n:
                s += ','
            s += '\n'
        s += ')'
        return s

class ReLU(Model):
    r"""
    Rectified linear unit activation function.

    This function computes

    .. math::

       \mathrm{relu}(x) = \mathrm{max}\{x, 0\}

    """

    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        return drjit.maximum(arg, 0)


class Exp2(Model):
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        return drjit.exp2(arg)

class Exp(Model):
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        return drjit.exp2(arg * (1 / drjit.log(2)))

class Tanh(Model):
    DRJIT_STRUCT = { }
    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        return drjit.tanh(arg)

class Cast(Model):
    DRJIT_STRUCT = { 'dtype': Optional[Type[drjit.ArrayBase]] }
    def __init__(self, dtype: Optional[Type[drjit.ArrayBase]] = None):
        self.dtype = dtype
    def __call__(self, arg: CoopVec, /) -> CoopVec:
        return cast(arg, self.dtype)

class Linear(Model):
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
        s = f'drjit.Linear({self.config[0]}, {self.config[1]}'
        if not self.config[2]:
            s += ', bias=False'
        s += ')'
        return s

    def __call__(self, arg: CoopVec[T], /) -> CoopVec[T]:
        if not isinstance(self.weights, MatrixView) or \
           (self.bias is not None and not isinstance(self.bias, MatrixView)):
            raise RuntimeError("You must call <model>.alloc() before using this function")
        return matvec(self.weights, arg, self.bias)

    def _alloc(self, dtype: Type[drjit.ArrayBase], size : int = -1, /) -> Tuple[Model, int]:
        in_features, out_features, bias = self.config
        if in_features < 0:
            in_features = size
        if out_features < 0:
            out_features = in_features
        if in_features == -1 or out_features == -1:
            raise RuntimeError("The network contains layers with an unspecified "
                               "size. You must specify the input size to drjit.Module.alloc().")

        result = Linear(in_features, out_features, bias)
        scale = drjit.sqrt(1 / out_features) # Xavier (uniform)
        Float32 = drjit.float32_array_t(dtype)
        samples = drjit.rand(Float32, (out_features, in_features))
        result.weights = dtype(drjit.fma(samples, 2, -1) * scale)
        if bias:
            result.bias = drjit.zeros(dtype, out_features)
        return result, out_features
