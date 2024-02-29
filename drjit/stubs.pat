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
    def __getitem__(self, key, /) -> ValT: ...

drjit.ArrayBase.__setitem__:
    def __setitem__(self, key, value: ElemT, /): ...

# Typed unary operations
drjit.ArrayBase.__(neg|abs|pos|invert)__:
    def __\1__(self: SelfT, /) -> SelfT: ...

# The 'type:ignore' annotations in the following are needed because the second
# overload overlaps with the first one while returning a different type, which
# PyRight does not like.
#
# And MyPy does not like that the semantics of __eq__/__ne__ differ from that
# of a base class ('object')

# Typed binary comparison operations that produce a mask
drjit.ArrayBase.__(eq|ne|lt|le|gt|ge)__:
    @overload # type: ignore
    def __\1__(self, arg: SelfT | ElemT, /) -> MaskT: ...
    @overload # type: ignore
    def __\1__(self: ElemT2, arg: ArrayBase[SelfT2, ValT2, ElemT2, MaskT2, RedT2], /) -> MaskT: ... # type: ignore

# Typed binary arithmetic operations
drjit.ArrayBase.__((?:r|i|)(?:add|sub|mul|truediv|floordiv|mod|lshift|rshift|and|or|xor))__:
    @overload  # type: ignore
    def __\1__(self, arg: SelfT | ElemT, /) -> SelfT: ...
    @overload
    def __\1__(self: ElemT2, arg: ArrayBase[SelfT2, ValT2, ElemT2, MaskT2, RedT2], /) -> SelfT2: ...

# Power has a different signature, so the pattern must be adapted
drjit.ArrayBase.__((?:r|i|)pow)__:
    @overload # type: ignore
    def __\1__(self, arg: SelfT | ElemT, mod=None, /) -> SelfT: ...
    @overload
    def __\1__(self: ElemT2, arg: ArrayBase[SelfT2, ValT2, ElemT2, MaskT2, RedT2], mod=None, /) -> SelfT2: ...

# ----------------------- dr.* functions -----------------------

drjit.(imag|real)$:
    @overload
    def \1(arg: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], /) -> ValT:
        \doc
    @overload
    def \1(arg: complex, /) -> float: ...

# Improve the types of reduction operations
drjit.(sum|prod|min|max|norm|squared_norm)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: int, axis: int | None = 0) -> int: ...
    @overload
    def \1(value: float, axis: int | None = 0) -> float: ...

drjit.(all|any|none)$:
    @overload
    def \1(value: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: bool, axis: int | None = 0) -> bool: ...

drjit.select$:
    @overload
    def select(arg0: bool | ArrayBase, arg1: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], arg2: ElemT, /) -> SelfT:
        \doc
    @overload
    def select(arg0: bool | ArrayBase, arg1: ElemT, arg2: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], /) -> SelfT: ...
    @overload
    def select(arg0: bool | ArrayBase, arg1: ValT, arg2: ValT) -> ValT: ...

drjit.(atan2|minimum|maximum)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], arg1: ElemT, /) -> SelfT:
        \doc
    @overload
    def \1(arg0: ElemT, arg1: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], /) -> SelfT: ...
    @overload
    def \1(arg0: T, arg1: T, /) -> T: ...

drjit.(empty|zeros)$:
    def \1(dtype: type[T], shape: int | Sequence[int] = 1) -> T:
        \doc

drjit.(full|opaque)$:
    @overload
    def \1(dtype: type[ArrayBase[SelfT, ValT, ElemT, MaskT, RedT]], value: ElemT, shape: int | Sequence[int] = 1) -> SelfT:
        \doc
    @overload
    def \1(dtype: type[T], value: T, shape: int | Sequence[int]) -> T: ...

drjit.fma$:
    @overload
    def fma(arg0: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], arg1: ElemT | SelfT, arg2: ElemT | SelfT, /) -> SelfT: ...
    @overload
    def fma(arg0: ElemT | SelfT, arg1: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], arg2: ElemT | SelfT, /) -> SelfT: ...
    @overload
    def fma(arg0: ElemT | SelfT, arg1: ElemT | SelfT, arg2: ArrayBase[SelfT, ValT, ElemT, MaskT, RedT], /) -> SelfT: ...
    @overload
    def fma(arg0: T, arg1: T, arg2: T) -> T:
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
