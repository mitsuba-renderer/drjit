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
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], axis: Axis = 0,
           mode: Literal['symbolic', 'evaluated', None] = None) -> RedT:
        \doc
    @overload
    def \1(value: Iterable[T], axis: Axis = 0, mode: Literal['symbolic', 'evaluated', None] = None) -> T: ...
    @overload
    def \1(value: int, axis: Axis = 0, mode: Literal['symbolic', 'evaluated', None] = None) -> int: ...
    @overload
    def \1(value: float, axis: Axis = 0, mode: Literal['symbolic', 'evaluated', None] = None) -> float: ...

drjit.(all|any|none)$:
    @overload
    def \1(value: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], axis: Axis = 0,
           mode: Literal['symbolic', 'evaluated', None] = None) -> RedT:
        \doc
    @overload
    def \1(value: Iterable[T], axis: Axis = 0, mode: Literal['symbolic', 'evaluated', None] = None) -> T: ...
    @overload
    def \1(value: bool, axis: Axis = 0, mode: Literal['symbolic', 'evaluated', None] = None) -> bool: ...

drjit.(dot|abs_dot)$:
    @overload
    def \1(arg0: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], arg1: SelfCpT, /) -> RedT:
        \doc
    @overload
    def \1(arg0: SelfCpT, arg1: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> RedT: ...
    @overload
    def \1(arg0: Sequence[T], arg1: Sequence[T], /) -> T: ...

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
    \from typing import Literal
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
#
drjit\.(cuda|llvm|auto)(.ad|).Array[0-4]([^0-4].*)\.__(set|del)item__:
drjit\.(cuda|llvm|auto)(.ad|).Array[0-4]([^0-4].*)\.__getitem__:
    xx: Array2\3
    xy: Array2\3
    xz: Array2\3
    xw: Array2\3
    yx: Array2\3
    yy: Array2\3
    yz: Array2\3
    yw: Array2\3
    zx: Array2\3
    zy: Array2\3
    zz: Array2\3
    zw: Array2\3
    wx: Array2\3
    wy: Array2\3
    wz: Array2\3
    ww: Array2\3
    xxx: Array3\3
    xxy: Array3\3
    xxz: Array3\3
    xxw: Array3\3
    xyx: Array3\3
    xyy: Array3\3
    xyz: Array3\3
    xyw: Array3\3
    xzx: Array3\3
    xzy: Array3\3
    xzz: Array3\3
    xzw: Array3\3
    xwx: Array3\3
    xwy: Array3\3
    xwz: Array3\3
    xww: Array3\3
    yxx: Array3\3
    yxy: Array3\3
    yxz: Array3\3
    yxw: Array3\3
    yyx: Array3\3
    yyy: Array3\3
    yyz: Array3\3
    yyw: Array3\3
    yzx: Array3\3
    yzy: Array3\3
    yzz: Array3\3
    yzw: Array3\3
    ywx: Array3\3
    ywy: Array3\3
    ywz: Array3\3
    yww: Array3\3
    zxx: Array3\3
    zxy: Array3\3
    zxz: Array3\3
    zxw: Array3\3
    zyx: Array3\3
    zyy: Array3\3
    zyz: Array3\3
    zyw: Array3\3
    zzx: Array3\3
    zzy: Array3\3
    zzz: Array3\3
    zzw: Array3\3
    zwx: Array3\3
    zwy: Array3\3
    zwz: Array3\3
    zww: Array3\3
    wxx: Array3\3
    wxy: Array3\3
    wxz: Array3\3
    wxw: Array3\3
    wyx: Array3\3
    wyy: Array3\3
    wyz: Array3\3
    wyw: Array3\3
    wzx: Array3\3
    wzy: Array3\3
    wzz: Array3\3
    wzw: Array3\3
    wwx: Array3\3
    wwy: Array3\3
    wwz: Array3\3
    www: Array3\3
    xxxx: Array4\3
    xxxy: Array4\3
    xxxz: Array4\3
    xxxw: Array4\3
    xxyx: Array4\3
    xxyy: Array4\3
    xxyz: Array4\3
    xxyw: Array4\3
    xxzx: Array4\3
    xxzy: Array4\3
    xxzz: Array4\3
    xxzw: Array4\3
    xxwx: Array4\3
    xxwy: Array4\3
    xxwz: Array4\3
    xxww: Array4\3
    xyxx: Array4\3
    xyxy: Array4\3
    xyxz: Array4\3
    xyxw: Array4\3
    xyyx: Array4\3
    xyyy: Array4\3
    xyyz: Array4\3
    xyyw: Array4\3
    xyzx: Array4\3
    xyzy: Array4\3
    xyzz: Array4\3
    xyzw: Array4\3
    xywx: Array4\3
    xywy: Array4\3
    xywz: Array4\3
    xyww: Array4\3
    xzxx: Array4\3
    xzxy: Array4\3
    xzxz: Array4\3
    xzxw: Array4\3
    xzyx: Array4\3
    xzyy: Array4\3
    xzyz: Array4\3
    xzyw: Array4\3
    xzzx: Array4\3
    xzzy: Array4\3
    xzzz: Array4\3
    xzzw: Array4\3
    xzwx: Array4\3
    xzwy: Array4\3
    xzwz: Array4\3
    xzww: Array4\3
    xwxx: Array4\3
    xwxy: Array4\3
    xwxz: Array4\3
    xwxw: Array4\3
    xwyx: Array4\3
    xwyy: Array4\3
    xwyz: Array4\3
    xwyw: Array4\3
    xwzx: Array4\3
    xwzy: Array4\3
    xwzz: Array4\3
    xwzw: Array4\3
    xwwx: Array4\3
    xwwy: Array4\3
    xwwz: Array4\3
    xwww: Array4\3
    yxxx: Array4\3
    yxxy: Array4\3
    yxxz: Array4\3
    yxxw: Array4\3
    yxyx: Array4\3
    yxyy: Array4\3
    yxyz: Array4\3
    yxyw: Array4\3
    yxzx: Array4\3
    yxzy: Array4\3
    yxzz: Array4\3
    yxzw: Array4\3
    yxwx: Array4\3
    yxwy: Array4\3
    yxwz: Array4\3
    yxww: Array4\3
    yyxx: Array4\3
    yyxy: Array4\3
    yyxz: Array4\3
    yyxw: Array4\3
    yyyx: Array4\3
    yyyy: Array4\3
    yyyz: Array4\3
    yyyw: Array4\3
    yyzx: Array4\3
    yyzy: Array4\3
    yyzz: Array4\3
    yyzw: Array4\3
    yywx: Array4\3
    yywy: Array4\3
    yywz: Array4\3
    yyww: Array4\3
    yzxx: Array4\3
    yzxy: Array4\3
    yzxz: Array4\3
    yzxw: Array4\3
    yzyx: Array4\3
    yzyy: Array4\3
    yzyz: Array4\3
    yzyw: Array4\3
    yzzx: Array4\3
    yzzy: Array4\3
    yzzz: Array4\3
    yzzw: Array4\3
    yzwx: Array4\3
    yzwy: Array4\3
    yzwz: Array4\3
    yzww: Array4\3
    ywxx: Array4\3
    ywxy: Array4\3
    ywxz: Array4\3
    ywxw: Array4\3
    ywyx: Array4\3
    ywyy: Array4\3
    ywyz: Array4\3
    ywyw: Array4\3
    ywzx: Array4\3
    ywzy: Array4\3
    ywzz: Array4\3
    ywzw: Array4\3
    ywwx: Array4\3
    ywwy: Array4\3
    ywwz: Array4\3
    ywww: Array4\3
    zxxx: Array4\3
    zxxy: Array4\3
    zxxz: Array4\3
    zxxw: Array4\3
    zxyx: Array4\3
    zxyy: Array4\3
    zxyz: Array4\3
    zxyw: Array4\3
    zxzx: Array4\3
    zxzy: Array4\3
    zxzz: Array4\3
    zxzw: Array4\3
    zxwx: Array4\3
    zxwy: Array4\3
    zxwz: Array4\3
    zxww: Array4\3
    zyxx: Array4\3
    zyxy: Array4\3
    zyxz: Array4\3
    zyxw: Array4\3
    zyyx: Array4\3
    zyyy: Array4\3
    zyyz: Array4\3
    zyyw: Array4\3
    zyzx: Array4\3
    zyzy: Array4\3
    zyzz: Array4\3
    zyzw: Array4\3
    zywx: Array4\3
    zywy: Array4\3
    zywz: Array4\3
    zyww: Array4\3
    zzxx: Array4\3
    zzxy: Array4\3
    zzxz: Array4\3
    zzxw: Array4\3
    zzyx: Array4\3
    zzyy: Array4\3
    zzyz: Array4\3
    zzyw: Array4\3
    zzzx: Array4\3
    zzzy: Array4\3
    zzzz: Array4\3
    zzzw: Array4\3
    zzwx: Array4\3
    zzwy: Array4\3
    zzwz: Array4\3
    zzww: Array4\3
    zwxx: Array4\3
    zwxy: Array4\3
    zwxz: Array4\3
    zwxw: Array4\3
    zwyx: Array4\3
    zwyy: Array4\3
    zwyz: Array4\3
    zwyw: Array4\3
    zwzx: Array4\3
    zwzy: Array4\3
    zwzz: Array4\3
    zwzw: Array4\3
    zwwx: Array4\3
    zwwy: Array4\3
    zwwz: Array4\3
    zwww: Array4\3
    wxxx: Array4\3
    wxxy: Array4\3
    wxxz: Array4\3
    wxxw: Array4\3
    wxyx: Array4\3
    wxyy: Array4\3
    wxyz: Array4\3
    wxyw: Array4\3
    wxzx: Array4\3
    wxzy: Array4\3
    wxzz: Array4\3
    wxzw: Array4\3
    wxwx: Array4\3
    wxwy: Array4\3
    wxwz: Array4\3
    wxww: Array4\3
    wyxx: Array4\3
    wyxy: Array4\3
    wyxz: Array4\3
    wyxw: Array4\3
    wyyx: Array4\3
    wyyy: Array4\3
    wyyz: Array4\3
    wyyw: Array4\3
    wyzx: Array4\3
    wyzy: Array4\3
    wyzz: Array4\3
    wyzw: Array4\3
    wywx: Array4\3
    wywy: Array4\3
    wywz: Array4\3
    wyww: Array4\3
    wzxx: Array4\3
    wzxy: Array4\3
    wzxz: Array4\3
    wzxw: Array4\3
    wzyx: Array4\3
    wzyy: Array4\3
    wzyz: Array4\3
    wzyw: Array4\3
    wzzx: Array4\3
    wzzy: Array4\3
    wzzz: Array4\3
    wzzw: Array4\3
    wzwx: Array4\3
    wzwy: Array4\3
    wzwz: Array4\3
    wzww: Array4\3
    wwxx: Array4\3
    wwxy: Array4\3
    wwxz: Array4\3
    wwxw: Array4\3
    wwyx: Array4\3
    wwyy: Array4\3
    wwyz: Array4\3
    wwyw: Array4\3
    wwzx: Array4\3
    wwzy: Array4\3
    wwzz: Array4\3
    wwzw: Array4\3
    wwwx: Array4\3
    wwwy: Array4\3
    wwwz: Array4\3
    wwww: Array4\3

# Typed versions of these are already provided by drjit.ArrayBase
drjit.(cuda|llvm|auto).*__(set|del)item__:
drjit.(cuda|llvm|auto).*__getitem__:
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

drjit.__prefix__:
    \from typing import TypeAlias
    \from collections.abc import Iterable, Sequence
    Axis: TypeAlias = int | tuple[int] | None
