# This file contains declarations to patch Dr.Jit's stub files. nanobind's
# stubgen automatically applies them during the build process

# --------------------------- General --------------------------

# Let's remove this everywhere
__meta__:

# -------------------------- ArrayBase -------------------------

# A few more removals
drjit.ArrayBase.__getattribute__:
drjit.ArrayBase.__delattr__:
drjit.ArrayBase.__delitem__:

# ArrayBase: Typed item access and assignment
drjit.ArrayBase.__getitem__:
    def __getitem__(self, key, /) -> _ItemT: ...

drjit.ArrayBase.__setitem__:
    def __setitem__(self, key, value: _UnionT, /): ...

# Typed unary operations
drjit.ArrayBase.__(neg|abs|pos|invert)__:
    def __\1__(self: _SelfT, /) -> _SelfT: ...

# The 'type:ignore' annotations in the following are needed because the second
# overload overlaps with the first one while returning a different type, which
# PyRight does not like.
#
# And MyPy does not like that the semantics of __eq__/__ne__ differ from that
# of a base class ('object')

# Typed binary comparison operations that produce a mask
drjit.ArrayBase.__(eq|ne|lt|le|gt|ge)__:
    @overload # type: ignore
    def __\1__(self, arg: _SelfT | _UnionT, /) -> _MaskT: ...
    @overload # type: ignore
    def __\1__(self: _UnionT2, arg: ArrayBase[_SelfT2, _ItemT2, _UnionT2, _MaskT2, _RedT2], /) -> _MaskT: ... # type: ignore

# Typed binary arithmetic operations
drjit.ArrayBase.__((?:r|i|)(?:add|sub|mul|truediv|floordiv|mod|lshift|rshift|and|or|xor))__:
    @overload  # type: ignore
    def __\1__(self, arg: _SelfT | _UnionT, /) -> _SelfT: ...
    @overload
    def __\1__(self: _UnionT2, arg: ArrayBase[_SelfT2, _ItemT2, _UnionT2, _MaskT2, _RedT2], /) -> _SelfT2: ...

# Power has a different signature, so the pattern must be adapted
drjit.ArrayBase.__((?:r|i|)pow)__:
    @overload # type: ignore
    def __\1__(self, arg: _SelfT | _UnionT, mod=None, /) -> _SelfT: ...
    @overload
    def __\1__(self: _UnionT2, arg: ArrayBase[_SelfT2, _ItemT2, _UnionT2, _MaskT2, _RedT2], mod=None, /) -> _SelfT2: ...

# ----------------------- dr.* functions -----------------------
#
# Typed unary operations
drjit.(abs|square|sqr)$:
    @overload
    def \1(arg: _ArrayT, /) -> _ArrayT:
        \doc
    @overload
    def \1(arg: int, /) -> int: ...
    @overload
    def \1(arg: float, /) -> float: ...

drjit.(sqrt|rsqrt|exp|exp2|log|log2|cos|sin|tan|acos|asin|atan|sinh|cosh|tanh|acosh|asinh|atanh|erf|erfinv|rad2deg|deg2rad)$:
    @overload
    def \1(arg: _ArrayT, /) -> _ArrayT:
        \doc
    @overload
    def \1(arg: float, /) -> float: ...

# Improve the types of reduction operations
drjit.(sum|prod|min|max|norm)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], axis: Literal[0] = 0) -> _RedT:
        \doc
    @overload
    def \1(value: Sequence[_T], axis: Literal[0] = 0) -> _T: ...
    @overload
    def \1(value: int, axis: int | None = 0) -> int: ...
    @overload
    def \1(value: float, axis: int | None = 0) -> float: ...

drjit.(all|any|none)$:
    @overload
    def \1(value: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], axis: Literal[0] = 0) -> _RedT:
        \doc
    @overload
    def \1(value: Sequence[_T], axis: Literal[0] = 0) -> _T: ...
    @overload
    def \1(value: bool, axis: int | None = 0) -> bool: ...

drjit.select$:
    @overload
    def select(arg0: bool | ArrayBase, arg1: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], arg2: _UnionT, /) -> _SelfT:
        \doc
    @overload
    def select(arg0: bool | ArrayBase, arg1: _UnionT, arg2: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], /) -> _SelfT: ...
    @overload
    def select(arg0: bool | ArrayBase, arg1: _ItemT, arg2: _ItemT) -> _ItemT: ...

drjit.(atan2|minimum|maximum)$:
    @overload
    def \1(arg0: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], arg1: _UnionT, /) -> _SelfT:
        \doc
    @overload
    def \1(arg0: _UnionT, arg1: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], /) -> _SelfT: ...
    @overload
    def \1(arg0: _T, arg1: _T, /) -> _T: ...

drjit.(empty|zeros)$:
    def \1(dtype: type[_T], shape: int | Sequence[int]) -> _T:
        \doc

drjit.fma$:
    @overload
    def fma(arg0: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], arg1: _UnionT | _SelfT, arg2: _UnionT | _SelfT) -> _SelfT: ...
    @overload
    def fma(arg0: _UnionT | _SelfT, arg1: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT], arg2: _UnionT | _SelfT) -> _SelfT: ...
    @overload
    def fma(arg0: _UnionT | _SelfT, arg1: _UnionT | _SelfT, arg2: ArrayBase[_SelfT, _ItemT, _UnionT, _MaskT, _RedT]) -> _SelfT: ...
    @overload
    def fma(arg0: _T, arg1: _T, arg2: _T) -> _T:
        \doc

# -------------- drjit.syntax, interop, detail ----------------

drjit.interop.[^w].*:
drjit.interop.wrap:
    \from typing import Callable, TypeVar, Union
    \from types import ModuleType
    T = TypeVar("T")

    def wrap(source: Union[str, ModuleType],
             target: Union[str, ModuleType]) -> Callable[[T], T]:
        \doc

# Type checkers don't know what a capsule object is, remove this for now
drjit.detail.bind:

# Preserve type of functions mapped through @dr.syntax
drjit.ast.syntax:
    \from typing import Callable, TypeVar, overload
    T = TypeVar("T")

    @overload
    def syntax(*, recursive: bool = False, print_ast: bool = False, print_code: bool = False) -> Callable[[T], T]:
        \doc

    @overload
    def syntax(f: T, /) -> T: ...

# ------------------- Backend-specific part -------------------

# Typed versions of these are already provided by drjit.ArrayBase
\.(Tensor|Array|Float|Int|Bool|Matrix|Quaternion).*__(set|del)item__:
\.(Tensor|Array|Float|Int|Bool|Matrix|Quaternion).*__getitem__:
    pass

PCG32.__isub__:
    def __isub__(self, arg: Int64, /) -> PCG32: # type: ignore
        \doc
