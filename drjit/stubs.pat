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
    @overload
    def __setitem__(self, key: MaskT, value: SelfCpT, /): ...
    @overload
    def __setitem__(self, key: object, value: ValCpT, /): ...

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
    def \1(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> ValT:
        \doc
    @overload
    def \1(arg: complex, /) -> float: ...

# Improve the types of reduction operations
drjit.(sum|prod|min|max|norm|squared_norm)$:
    \from typing import Literal
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: int, axis: int | None = 0) -> int: ...
    @overload
    def \1(value: float, axis: int | None = 0) -> float: ...

drjit.(dot|abs_dot)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg1: SelfCpT, /) -> RedT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> RedT: ...
    @overload
    def \1(arg0: Sequence[T], arg1: Sequence[T], /) -> T: ...

drjit.(all|any|none)$:
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], axis: Literal[0] = 0) -> RedT:
        \doc
    @overload
    def \1(value: Sequence[T], axis: Literal[0] = 0) -> T: ...
    @overload
    def \1(value: bool, axis: int | None = 0) -> bool: ...

drjit.select$:
    @overload
    def select(arg0: bool | AnyArray, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg2: ValCpT, /) -> SelfT:
        \doc
    @overload
    def select(arg0: bool | AnyArray, arg1: ValCpT, arg2: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> SelfT: ...
    @overload
    def select(arg0: bool | AnyArray, arg1: T, arg2: T) -> T: ...

drjit.(atan2|minimum|maximum)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg1: SelfCpT, /) -> SelfT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> SelfT: ...
    @overload
    def \1(arg0: T, arg1: T, /) -> T: ...

drjit.(empty|zeros|ones)$:
    def \1(dtype: type[T], shape: int | Sequence[int] = 1) -> T:
        \doc

drjit.(full|opaque)$:
    @overload
    def \1(dtype: type[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT]], value: ValCpT, shape: int | Sequence[int] = 1) -> SelfT:
        \doc
    @overload
    def \1(dtype: type[T], value: T, shape: int | Sequence[int]) -> T: ...

drjit.(fma|lerp)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg1: SelfCpT, arg2: SelfCpT, /) -> SelfT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg2: SelfCpT, /) -> SelfT: ...
    @overload
    def \1(arg0: SelfCpT, arg1: SelfCpT, arg2: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> SelfT: ...
    @overload
    def \1(arg0: T, arg1: T, arg2: T) -> T: ...

drjit.reshape$:
    def reshape(dtype: type[T], value: object, shape: int | Sequence[int], order: Literal['A', 'C', 'F'] = 'A', shrink: bool = False) -> T:
        \doc

drjit.(isnan|isinf|isfinite)$:
    @overload
    def \1(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> MaskT:
        \doc
    @overload
    def \1(arg: float, /) -> bool: ...

drjit.meshgrid$:
    def meshgrid(*args: *Ts, indexing: Literal['xy', 'ij'] = 'xy') -> tuple[*Ts]:
        \doc

# Typing information for the clip function below
drjit.clip$:
    @overload
    def clip(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], min: SelfCpT, max: SelfCpT) -> SelfT:
        \doc
    @overload
    def clip(value: SelfCpT, min: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], max: SelfCpT) -> SelfT: ...
    @overload
    def clip(value: SelfCpT, min: SelfCpT, max: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT]) -> SelfT: ...
    @overload
    def clip(value: T, min: T, max: T) -> T: ...

drjit.mask_t$:
    @overload
    def mask_t(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> type[MaskT]:
        \doc
    @overload
    def mask_t(arg: type[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT]], /) -> type[MaskT]: ...
    @overload
    def mask_t(arg: object, /) -> bool: ...

drjit.value_t$:
    @overload
    def value_t(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> type[ValT]:
        \doc
    @overload
    def value_t(arg: type[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT]], /) -> type[ValT]: ...
    @overload
    def value_t(arg: object, /) -> type: ...

drjit.array_t$:
    @overload
    def array_t(arg: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> type[PlainT]:
        \doc
    @overload
    def array_t(arg: type[ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT]], /) -> type[PlainT]: ...
    @overload
    def array_t(arg: object, /) -> type: ...

drjit.uint32_array_t$:
    @overload
    def uint32_array_t(arg: type[ArrayBase] | ArrayBase) -> type[AnyArray]:
        \doc
    @overload
    def uint32_array_t(arg: object) -> int: ...

drjit.int32_array_t$:
    @overload
    def int32_array_t(arg: type[ArrayBase] | ArrayBase) -> type[AnyArray]:
        \doc
    @overload
    def int32_array_t(arg: object) -> int: ...


drjit.custom$:
    \from typing import Protocol, ParamSpec

    Ps = ParamSpec("Ps")
    Tc = TypeVar("Tc", covariant=True)

    class CustomOpT(Protocol[Ps, Tc]):
        def eval(self, *args: Ps.args, **kwargs: Ps.kwargs) -> Tc:...

    def custom(arg0: type[CustomOpT[Ps, T]], /, *args: Ps.args, **kwargs: Ps.kwargs) -> T:
        \doc

drjit.switch$:
    \from typing import Sequence

    # Helper type variable and protocol to type-check ``dr.switch()``
    class CallablePT(Protocol[Ps, Tc]):
        def __call__(self, *args: Ps.args, **kwargs: Ps.kwargs) -> Tc:...

    def switch(index: int | AnyArray,
               targets: Sequence[CallablePT[Ps, T]],
               *args: Ps.args,
               mode: Literal['symbolic', 'evaluated', None] = None,
               label: str | None = None,
               **kwargs: Ps.kwargs) -> T:
        \doc

drjit.dispatch$:
    # Helper type variable and protocol to type-check ``dr.dispatch()``
    InstT = TypeVar("InstT", contravariant=True)
    class CallableSelfPT(Protocol[InstT, Ps, Tc]):
        def __call__(self, arg0: InstT, /, *args: Ps.args, **kwargs: Ps.kwargs) -> Tc:...

    def dispatch(inst: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT],
                 target: CallableSelfPT[ValT, Ps, T],
                 *args: Ps.args,
                 mode: Literal['symbolic', 'evaluated', None] = None,
                 label: str | None = None,
                 **kwargs: Ps.kwargs) -> T:
        \doc

drjit.sh_eval$:
    def sh_eval(d: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], order: int) -> list[ValT]:
        \doc

drjit.sh_eval$:
    def sh_eval(d: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], order: int) -> list[ValT]:
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
