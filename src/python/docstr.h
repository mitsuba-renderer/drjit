static const char *doc_is_array_v = R"(
is_array_v(arg, /)
Check if the input is a Dr.Jit array instance or type

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** or type(**arg**) is a Dr.Jit array type, and ``False`` otherwise)";

static const char *doc_array_size_v = R"(
array_size_v(arg, /)
Return the (static) size of the provided Dr.Jit array instance or type

Note that this function mainly exists to query type-level information. Use the
Python ``len()`` function to query the size in a way that does not distinguish
between static and dynamic arrays.

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns either the static size or :py:data:`drjit.Dynamic` when
    **arg** is a dynamic Dr.Jit array. Returns ``1`` for all other types.)";

static const char *doc_array_depth_v = R"(
array_depth_v(arg, /)
Return the depth of the provided Dr.Jit array instance or type

For example, an array consisting of floating point values (for example,
:py:class:`drjit.scalar.Array3f`) has depth ``1``, while an array consisting of
sub-arrays (e.g., :py:class:`drjit.cuda.Array3f`) has depth ``2``.

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the depth of the input, if it is a Dr.Jit array instance or
    type. Returns ``0`` for all other inputs.)";

static const char *doc_select = R"(
select(condition, x, y)
Select elements from inputs based on a condition

This function implements the component-wise operation

.. math::

   \mathrm{result}_i = \begin{cases}
       x_i,\quad&\text{if condition}_i,\\
       y_i,\quad&\text{otherwise.}
   \end{cases}

Args:
    condition (bool | drjit.ArrayBase): A Python or Dr.Jit mask/boolean type
    x (int | float | drjit.ArrayBase): A Python or Dr.Jit type
    y (int | float | drjit.ArrayBase): A Python or Dr.Jit type

Returns:
    float | int | drjit.ArrayBase: Component-wise result of the selection operation)";

static const char *doc_abs = R"(
abs(arg, /)
Compute the absolute value of the provided input. 

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    float | int | drjit.ArrayBase: Absolute value of the input)";

static const char *doc_max = R"(
max(arg0, /) -> float | int | drjit.ArrayBase
max(arg0, arg1, /) -> float | int | drjit.ArrayBase
Compute the maximum value of the provided inputs. 

This function can be used in two different ways: when invoked with two inputs,
it computes the componentwise maximum and returns a result of the type
``type(arg0 + arg1)`` (i.e., according to the usual implicit type conversion
rules).

When invoked with a single argument, it performs a horizontal reduction. Please
see the section on :ref:`horizontal reductions <horizontal-reductions>` for
details.

Args:
    arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type (optional)

Returns:
    Maximum of the input(s))";


static const char *doc_min = R"(
min(arg0, /) -> float | int | drjit.ArrayBase
min(arg0, arg1, /) -> float | int | drjit.ArrayBase
Compute the minimum value of the provided inputs. 

This function can be used in two different ways: when invoked with two inputs,
it computes the componentwise minimum and returns a result of the type
``type(arg0 + arg1)`` (i.e., according to the usual implicit type conversion
rules).

When invoked with a single argument, it performs a horizontal reduction. Please
see the section on :ref:`horizontal reductions <horizontal-reductions>` for
details.

Args:
    arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type (optional)

Returns:
    Minimum of the input(s))";

static const char *doc_sqrt = R"(
sqrt(arg, /)
Evaluate the square root of the provided input. 

Negative inputs produce a *NaN* output value.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Square root of the input)";


static const char *doc_cbrt = R"(
cbrt(arg, /)
Evaluate the cube root of the provided input. 

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Cube root of the input)";


static const char *doc_rcp = R"(
rcp(arg, /)
Evaluate the reciprocal (1 / arg) of the provided input. 

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU"). The result is slightly
approximate in this case (refer to the documentation of the instruction
`rcp.approx.ftz.f32` in the NVIDIA PTX manual for details).

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Reciprocal of the input)";

static const char *doc_rsqrt = R"(
rsqrt(arg, /)
Evaluate the reciprocal square root (1 / sqrt(arg)) of the provided input. 

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU"). The result is slightly
approximate in this case (refer to the documentation of the instruction
`rsqrt.approx.ftz.f32` in the NVIDIA PTX manual for details).

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Reciprocal square root of the input)";

static const char *doc_ceil= R"(
ceil(arg, /)
Evaluate the ceiling, i.e. the smallest integer >= arg.

The function does not convert the type of the input array. A separate
cast is necessary when integer output is desired.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Ceiling of the input)";


static const char *doc_floor = R"(
floor(arg, /)
Evaluate the floor, i.e. the largest integer <= arg.

The function does not convert the type of the input array. A separate
cast is necessary when integer output is desired.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Floor of the input)";


static const char *doc_trunc = R"(
trunc(arg, /)
Truncates arg to the nearest integer by towards zero.

The function does not convert the type of the input array. A separate
cast is necessary when integer output is desired.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Truncated result)";


static const char *doc_round = R"(
round(arg, /)

Rounds arg to the nearest integer using Banker's rounding for
half-way values.

This function is equivalent to ``std::rint`` in C++. It does not convert the
type of the input array. A separate cast is necessary when integer output is
desired.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Rounded result)";



static const char *doc_log = R"(
log(arg, /)
Natural exponential approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Natural logarithm of the input)";

static const char *doc_log2 = R"(
log2(arg, /)
Base-2 exponential approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Base-2 logarithm of the input)";

static const char *doc_exp = R"(
exp(arg, /)
Natural exponential approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Natural exponential of the input)";

static const char *doc_exp2 = R"(
exp2(arg, /)
Base-2 exponential approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Base-2 exponential of the input)";

static const char *doc_sin = R"(
sin(arg, /)
Sine approximation based on the CEPHES library.

The implementation of this function is designed to achieve low error on the domain
:math:`|x| < 8192` and will not perform as well beyond this range. See the
section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Sine of the input)";

static const char *doc_cos = R"(
cos(arg, /)
Sine approximation based on the CEPHES library.

The implementation of this function is designed to achieve low error on the
domain :math:`|x| < 8192` and will not perform as well beyond this range. See
the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Cosine of the input)";


static const char *doc_sincos = R"(
sincos(arg, /)
Sine/cosine approximation based on the CEPHES library.

The implementation of this function is designed to achieve low error on the
domain :math:`|x| < 8192` and will not perform as well beyond this range. See
the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using two operations involving the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Sine and cosine of the input)";

static const char *doc_tan = R"(
tan(arg, /)
Tangent approximation based on the CEPHES library.

The implementation of this function is designed to achieve low error on the
domain :math:`|x| < 8192` and will not perform as well beyond this range. See
the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When **arg** is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Tangent of the input)";

static const char *doc_asin = R"(
asin(arg, /)
Arcsine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Arcsine of the input)";


static const char *doc_acos = R"(
acos(arg, /)
Arcsine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Arccosine of the input)";


static const char *doc_atan = R"(
atan(arg, /)
Arctangent approximation

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Arctangent of the input)";

static const char *doc_atan2 = R"(
atan2(y, x, /)
Arctangent of two values

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    y (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
    x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Arctangent of **y**/**x**, using the argument signs to
    determine the quadrant of the return value)";

static const char *doc_ldexp = R"(
ldexp(x, n, /)
Multiply x by 2 taken to the power of n

Args:
    x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
    n (float | drjit.ArrayBase): A Python or Dr.Jit floating point type *without fractional component*

Returns:
    float | drjit.ArrayBase: The result of **x** multipled by 2 taken to the power **n**.)";

static const char *doc_sinh = R"(
sinh(arg, /)
Hyperbolic sine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic sine of the input)";


static const char *doc_cosh = R"(
cosh(arg, /)
Hyperbolic cosine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic cosine of the input)";


static const char *doc_sincosh = R"(
sincosh(arg, /)
Hyperbolic sine/cosine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Hyperbolic sine and cosine of the input)";

static const char *doc_tanh = R"(
tanh(arg, /)
Hyperbolic tangent approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic tangent of the input)";


static const char *doc_asinh = R"(
asinh(arg, /)
Hyperbolic arcsine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic arcsine of the input)";


static const char *doc_acosh = R"(
acosh(arg, /)
Hyperbolic arccosine approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic arccosine of the input)";


static const char *doc_atanh = R"(
atanh(arg, /)
Hyperbolic arctangent approximation based on the CEPHES library.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Hyperbolic arctangent of the input)";

static const char *doc_frexp = R"(
frexp(arg, /)
Break the given floating point number into normalized fraction and power of 2

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Normalized fraction
    ``frac`` on the interval :math:`[\frac{1}{2}, 1)` and an exponent ``exp``
    so that ``frac * 2**(exp + 1)`` equals **arg**.)";


static const char *doc_fma = R"(
fma(arg0, arg1, arg2, /)
Perform a *fused multiply-add* (FMA) operation.

Given arguments **arg0**, **arg1**, and **arg2**, this operation computes
**arg0** * **arg1** + **arg2** using only one final rounding step. The
operation is not only more accurate, but also more efficient, since FMA maps to
a native machine instruction on platforms targeted by Dr.Jit.

While FMA is traditionally a floating point operation, Dr.Jit also implements
FMA for integer arrays and maps it onto dedicated instructions provided by the
backend if possible (e.g. ``mad.lo.*`` for CUDA/PTX).

Args:
    arg0 (float | drjit.ArrayBase): First multiplication operand
    arg1 (float | drjit.ArrayBase): Second multiplication operand
    arg2 (float | drjit.ArrayBase): Additive operand

Returns:
    float | drjit.ArrayBase: Result of the FMA operation)";
