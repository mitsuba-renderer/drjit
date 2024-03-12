# This file contains declarations to patch Dr.Jit's stub files. nanobind's
# stubgen automatically applies them during the build process
#
# The syntax of this file is described here:
#
# https://nanobind.readthedocs.io/en/latest/typing.html#pattern-files
#
# The design of type signatures and the use of generics and type variables is
# explained in the Dr.Jit documentation section entitled "Type Signatures".
#
# Whenever possible, it's preferable to specify signatures to C++ bindings
# using the nb::sig() override. The rules below are used in cases where that is
# not possible, or when the typing-specific overloads of a funciton deviate
# significantly from the overload chain implemented using nanobind.

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
    def __setitem__(self, key, value: ValCpT, /): ...

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
    def __\1__(self, arg: SelfCpT, /) -> MaskT: ... # type: ignore[override]

# Typed binary arithmetic operations
drjit.ArrayBase.__((?:r|i|)(?:add|sub|mul|truediv|floordiv|mod|lshift|rshift|and|or|xor))__:
    def __\1__(self, arg: SelfCpT, /) -> SelfT: ...

# Power has a different signature, so the pattern must be adapted
drjit.ArrayBase.__((?:r|i|)pow)__:
    def __\1__(self, arg: SelfCpT, mod=None, /) -> SelfT: ...

# ----------------------- dr.* functions -----------------------

drjit.(imag|real)$:
    @overload
    def \1(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> ValT:
        \doc
    @overload
    def \1(arg: complex, /) -> float: ...

# Improve the types of reduction operations
drjit.(sum|prod|min|max|norm|squared_norm)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: int, axis: int | None = 0) -> int: ...
    @overload
    def \1(value: float, axis: int | None = 0) -> float: ...

drjit.(dot|abs_dot)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], arg1: SelfCpT, /) -> RedT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> RedT: ...
    @overload
    def \1(arg0: Sequence[T], arg1: Sequence[T], /) -> T: ...

drjit.(all|any|none)$:
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: bool, axis: int | None = 0) -> bool: ...

drjit.select$:
    @overload
    def select(arg0: bool | AnyArray, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], arg2: ValCpT, /) -> SelfT:
        \doc
    @overload
    def select(arg0: bool | AnyArray, arg1: ValCpT, arg2: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> SelfT: ...
    @overload
    def select(arg0: bool | AnyArray, arg1: T, arg2: T) -> T: ...

drjit.(atan2|minimum|maximum)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], arg1: SelfCpT, /) -> SelfT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> SelfT: ...
    @overload
    def \1(arg0: T, arg1: T, /) -> T: ...

drjit.(empty|zeros)$:
    def \1(dtype: type[T], shape: int | Sequence[int] = 1) -> T:
        \doc

drjit.(full|opaque)$:
    @overload
    def \1(dtype: type[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT]], value: ValCpT, shape: int | Sequence[int] = 1) -> SelfT:
        \doc
    @overload
    def \1(dtype: type[T], value: T, shape: int | Sequence[int]) -> T: ...

drjit.(fma|lerp)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], arg1: SelfCpT, arg2: SelfCpT, /) -> SelfT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], arg2: SelfCpT, /) -> SelfT: ...
    @overload
    def \1(arg0: SelfCpT, arg1: SelfCpT, arg2: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> SelfT: ...
    @overload
    def \1(arg0: T, arg1: T, arg2: T) -> T: ...

drjit.reshape$:
    def reshape(dtype: type[T], value: object, shape: int | Sequence[int], order: str = 'A', shrink: bool = False) -> T:
        \doc


drjit.(isnan|isinf|isfinite)$:
    @overload
    def \1(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> MaskT:
        \doc
    @overload
    def \1(arg: float, /) -> bool: ...

drjit.meshgrid$:
    def meshgrid(*args: *Ts, indexing: Literal['xy', 'ij'] = 'xy') -> tuple[*Ts]:
        \doc

# -------------- drjit.syntax, interop, detail ----------------

# Clean the drjit.interop stub
drjit.interop.[^wT].*:

# Type checkers don't know what a capsule object is, remove this for now
drjit.detail.bind:

# ------------------- Backend-specific part -------------------

# Typed versions of these are already provided by drjit.ArrayBase
\.(Tensor|Array|Float|Int|Bool|Matrix|Complex|Quaternion|(Int|UInt|Float|)(16|32|64|)).*__(set|del)item__:
\.(Tensor|Array|Float|Int|Bool|Matrix|Complex|Quaternion|(Int|UInt|Float|)(16|32|64|)).*__getitem__:
    pass

PCG32.__isub__:
    def __isub__(self, arg: Int64, /) -> PCG32: # type: ignore
        \doc

drjit.scalar.__prefix__:
    from builtins import (
        bool as Bool,
        float as Float,
        float as Float16,
        float as Float32,
        float as Float64,
        int as Int,
        int as Int16,
        int as Int32,
        int as Int64,
        int as UInt,
        int as UInt16,
        int as UInt32,
        int as UInt64
    )
