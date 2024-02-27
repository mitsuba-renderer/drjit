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
    def __getitem__(self, key, /) -> ValueT: ...

drjit.ArrayBase.__setitem__:
    def __setitem__(self, key, value: ValueT | ScalarT, /): ...

# Typed unary operations
drjit.ArrayBase.__(neg|abs|pos|invert)__:
    def __\1__(self: SelfT, /) -> SelfT: ...

# ArrayBase: Typed binary comparison operations that produce a mask
drjit.ArrayBase.__(eq|ne|lt|le|gt|ge)__:
    @overload # type: ignore
    def __\1__(self: SelfT, arg: SelfT | ValueT | ScalarT, /) -> MaskT: ...
    @overload
    def __\1__(self, arg, /) -> ArrayBase: ...

# ArrayBase: Typed binary arithmetic operations
drjit.ArrayBase.__((?:r|i|)(?:add|sub|mul|truediv|floordiv|mod|lshift|rshift|and|or|xor))__:
    @overload # type: ignore
    def __\1__(self: SelfT, arg: SelfT | ValueT | ScalarT, /) -> SelfT: ...
    @overload
    def __\1__(self, arg, /) -> ArrayBase: ...

# ArrayBase: power has a different signature, so the pattern must be adapted
drjit.ArrayBase.__((?:r|i|)pow)__:
    @overload # type: ignore
    def __\1__(self: SelfT, arg: SelfT | ValueT | ScalarT, mod=None, /) -> SelfT: ...
    @overload
    def __\1__(self, arg, mod=None, /) -> ArrayBase: ...

# ----------------------- dr.* functions -----------------------

# Improve the types of reduction operations
drjit.(sum|prod|min|max)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[ValueT, ScalarT, MaskT, ReduceT], axis: Literal[0] = 0) -> ReduceT:
        \doc
    @overload
    def \1(value: Sequence[ValueT], axis: Literal[0] = 0) -> ValueT: ...
    @overload
    def \1(value: int, axis: object = 0) -> int: ...
    @overload
    def \1(value: float, axis: object = 0) -> float: ...
    @overload
    def \1(value: object, axis: int | None = 0) -> object: ...

drjit.(all|any|none)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[ValueT, ScalarT, MaskT, ReduceT], axis: Literal[0] = 0) -> ReduceT:
        \doc
    @overload
    def \1(value: Sequence[ValueT], axis: Literal[0] = 0) -> ValueT: ...
    @overload
    def \1(value: bool, axis: object = 0) -> bool: ...
    @overload
    def \1(value: object, axis: int | None = 0) -> object: ...

# -------------- drjit.syntax, interop, detail ----------------

drjit.(atan2|minimum|maximum)$:
    @overload
    def \1(arg0: ArrayT, arg1: ArrayT, /) -> ArrayT:
        \doc
    @overload
    def \1(arg0: int, arg1: int, /) -> int: ...
    @overload
    def \1(arg0: float, arg1: float, /) -> float: ...
    @overload
    def \1(arg0: object, arg1: object, /) -> object: ...

# -------------- drjit.syntax, interop, detail ----------------

drjit.interop.wrap:
    def wrap(source: typing.Union[str, types.ModuleType],
             target: typing.Union[str, types.ModuleType]):
        \doc

# Type checkers don't know what a capsule object is, remove this for now
drjit.detail.bind:

# Preserve type of functions mapped through @dr.syntax
drjit.ast.syntax:
    \from typing import Callable, TypeVar, overload
    T = TypeVar("T")

    @overload
    def syntax( *, recursive: bool = False, print_ast: bool = False, print_code: bool = False) -> Callable[[T], T]:
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

