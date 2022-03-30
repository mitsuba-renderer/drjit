static const char *doc_is_array_v = R"(
is_array_v(arg, /)
Check if the input is a Dr.Jit array instance or type

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** or type(**arg**) is a Dr.Jit array type, and ``False`` otherwise)";

static const char *doc_is_struct_v = R"(
is_struct_v(arg, /)
Check if the input is a Dr.Jit-compatible data structure

Custom data structures can be made compatible with various Dr.Jit operations by
specifying a ``DRJIT_STRUCT`` member. See the section on :ref:`custom data
structure <custom-struct>` for details. This type trait can be used to check
for the existence of such a field.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** has a ``DRJIT_STRUCT`` member)";

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

static const char *doc_value_t = R"(
value_t(arg, /)
Return the *value type* underlying the provided Dr.Jit array or type (i.e., the
type of values obtained by accessing the contents using a 1D index).

When the input is not a Dr.Jit array or type, the function returns the input
unchanged. The following code fragment shows several example uses of
:py:func:`value_t`.

.. code-block::

    assert dr.value_t(dr.scalar.Array3f) is float
    assert dr.value_t(dr.cuda.Array3f) is dr.cuda.Float
    assert dr.value_t(dr.cuda.Matrix4f) is dr.cuda.Array4f
    assert dr.value_t("test") is str

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the value type of the provided Dr.Jit array, or the type of
    the input.
)";

static const char *doc_mask_t = R"(
mask_t(arg, /)
Return the *mask type* associated with the provided Dr.Jit array or type (i.e., the
type of comparisons involving the argument).

When the input is not a Dr.Jit array or type, the function returns the scalar
Python ``bool`` type. The following code fragment shows several example uses of
:py:func:`mask_t`.


.. code-block::

    assert dr.mask_t(dr.scalar.Array3f) is dr.scalar.Array3b
    assert dr.mask_t(dr.cuda.Array3f) is dr.cuda.Array3b
    assert dr.mask_t(dr.cuda.Matrix4f) is dr.cuda.Array44b
    assert dr.mask_t("test") is bool

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the mask type associated with the input or ``bool`` when the
    input is not a Dr.Jit array.
)";

static const char *doc_scalar_t = R"(
scalar_t(arg, /)
Return the *scalar type* associated with the provided Dr.Jit array or type (i.e., the
representation of elements at the lowest level)

When the input is not a Dr.Jit array or type, the function returns the input
unchanged. The following code fragment shows several example uses of
:py:func:`scalar_t`.

.. code-block::

    assert dr.scalar_t(dr.scalar.Array3f) is bool
    assert dr.scalar_t(dr.cuda.Array3f) is float
    assert dr.scalar_t(dr.cuda.Matrix4f) is float
    assert dr.scalar_t("test") is str

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the scalar type of the provided Dr.Jit array, or the type of
    the input.
)";

static const char *doc_is_mask_v = R"(
is_mask_v(arg, /)
Check whether the input array instance or type is a Dr.Jit mask array or a
Python ``bool`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents a Dr.Jit mask array or Python ``bool``
    instance or type.
)";

static const char *doc_is_integral_v = R"(
is_integral_v(arg, /)
Check whether the input array instance or type is an integral Dr.Jit array
or a Python ``int`` value/type.

Note that a mask array is not considered to be integral.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an integral Dr.Jit array or
    Python ``int`` instance or type.
)";

static const char *doc_is_float_v = R"(
is_float_v(arg, /)
Check whether the input array instance or type is a Dr.Jit floating point array
or a Python ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents a Dr.Jit floating point array or
    Python ``float`` instance or type.
)";


static const char *doc_is_arithmetic_v = R"(
is_arithmetic_v(arg, /)
Check whether the input array instance or type is an arithmetic Dr.Jit array
or a Python ``int`` or ``float`` value/type.

Note that a mask type (e.g. ``bool``, :py:class:`drjit.scalar.Array2b`, etc.)
is *not* considered to be arithmetic.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an arithmetic Dr.Jit array or
    Python ``int`` or ``float`` instance or type.
)";

static const char *doc_is_jit_v = R"(
is_jit_v(arg, /)
Check whether the input array instance or type represents a type that
undergoes just-in-time compilation.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an array type from the
    ``drjit.cuda.*`` or ``drjit.llvm.*`` namespaces, and ``False`` otherwise.
)";

static const char *doc_is_cuda_v = R"(
is_cuda_v(arg, /)
Check whether the input is a Dr.Jit CUDA array instance or type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an array type from the
    ``drjit.cuda.*`` namespace, and ``False`` otherwise.
)";

static const char *doc_is_llvm_v = R"(
is_llvm_v(arg, /)
Check whether the input is a Dr.Jit LLVM array instance or type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an array type from the
    ``drjit.llvm.*`` namespace, and ``False`` otherwise.
)";

static const char *doc_is_diff_v = R"(
is_diff_v(arg, /)
Check whether the input is a differentiable Dr.Jit array instance or type.

Note that this is a type-based statement that is unrelated to mathematical
differentiability. For example, the integral type :py:class:`drjit.cuda.ad.Int`
from the CUDA AD namespace satisfies ``is_diff_v(..) = 1``.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an array type from the
    ``drjit.[cuda/llvm].ad.*`` namespace, and ``False`` otherwise.
)";

static const char *doc_is_complex_v = R"(
is_complex_v(arg, /)
Check whether the input is a Dr.Jit array instance or type representing a complex number.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_quaternion_v = R"(
is_quaternion_v(arg, /)
Check whether the input is a Dr.Jit array instance or type representing a quaternion.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_matrix_v = R"(
is_matrix_v(arg, /)
Check whether the input is a Dr.Jit array instance or type representing a matrix.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_tensor_v = R"(
is_tensor_v(arg, /)
Check whether the input is a Dr.Jit array instance or type representing a tensor.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_special_v = R"(
is_special_v(arg, /)
Check whether the input is a _special_ Dr.Jit array instance or type.

A *special* array type requires precautions when performing arithmetic
operations like multiplications (complex numbers, quaternions, matrices).

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

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

static const char *doc_zero = R"(
Return a zero-valued Dr.Jit array of the desired shape. 

When ``dtype`` refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), ``shape`` must be a tuple that determines the
tensor rank and shape. 

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension. 

When a tuple is specified, its must be compatible with static dimensions of the
``dtype``. For example, ``dr.zero(dr.cuda.Array2f, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    shape (tuple | int): Shape of the desired array

Returns:
    object: A zero-initialized instance of type `dtype`
)";

static const char *doc_full = R"(
Return a constant-valued Dr.Jit array of the desired shape. 

When ``dtype`` refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), ``shape`` must be a tuple that determines the
tensor rank and shape. 

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension. 

When a tuple is specified, its must be compatible with static dimensions of the
``dtype``. For example, ``dr.full(dr.cuda.Array2f, value=123.0, shape=(3, 100))`` fails,
since the leading dimension is incompatible with 
:py:class:`drjit.cuda.Array2f`.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    value (object): An instance of the underlying scalar type (``float``/``int``/``bool``, etc.)
                    that will be used to initialize the array contents.
    shape (tuple | int): Shape of the desired array

Returns:
    object: A constant-initialized instance of type `dtype`
)";


static const char *doc_empty = R"(
Return a empty Dr.Jit array of the desired shape. 

When ``dtype`` refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), ``shape`` must be a tuple that determines the
tensor rank and shape. 

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension. 

When a tuple is specified, its must be compatible with static dimensions of the
``dtype``. For example, ``dr.empty(dr.cuda.Array2f, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    shape (tuple | int): Shape of the desired array

Returns:
    object: An empty instance of type `dtype`
)";


static const char *doc_shape = R"(
shape(arg, /)
Return a tuple describing dimension and shape of the provided Dr.Jit array or
tensor.

When the arrays is ragged, the implementation signals a failure by returning
``None``. A ragged array has entries of incompatible size, e.g. ``[[1, 2], [3,
4, 5]]``. Note that an scalar entries (e.g. ``[[1, 2], [3]]``) are acceptable,
since broadcasting can effectively convert them to any size.

The expressions ``drjit.shape(arg)`` and ``arg.shape`` are equivalent.

Args:
    arg (drjit.ArrayBase): an arbitrary Dr.Jit array or tensor

Returns:
    tuple | NoneType: A tuple describing the dimension and shape of the
    provided Dr.Jit input array or tensor. When the input array is *ragged*
    (i.e., when it contains components with mismatched sizes), the function
    returns ``None``.
)";


static const char *doc_ArrayBase_x = R"(
If ``value`` is a static Dr.Jit array of size 1 (or larger), the property
``value.x`` can be used synonymously with ``value[0]``. Otherwise, accessing
this field will generate a ``TypeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_y = R"(
If ``value`` is a static Dr.Jit array of size 2 (or larger), the property
``value.y`` can be used synonymously with ``value[1]``. Otherwise, accessing
this field will generate a ``TypeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_z = R"(
If ``value`` is a static Dr.Jit array of size 3 (or larger), the property
``value.z`` can be used synonymously with ``value[2]``. Otherwise, accessing
this field will generate a ``TypeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_w = R"(
If ``value`` is a static Dr.Jit array of size 4 (or larger), the property
``value.w`` can be used synonymously with ``value[3]``. Otherwise, accessing
this field will generate a ``TypeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_shape = R"(
This property contains a tuple describing dimension and shape of the
provided Dr.Jit array or tensor. When the input array is *ragged*
(i.e., when it contains components with mismatched sizes), the
property equals ``None``.

:type: tuple | NoneType)";

static const char *doc_ArrayBase_index = R"(
If ``self`` is a *leaf* Dr.Jit array managed by a just-in-time compiled backend
(i.e, CUDA or LLVM), this property contains the associated variable index in
the graph data structure storing the computation trace. This graph can be
visualized using :py:func:`drjit.graphviz`. Otherwise, the value of this
property equals zero. A *non-leaf* array (e.g. :py:class:`drjit.cuda.Array2i`)
consists of several JIT variables, whose indices must be queried separately.

Note that Dr.Jit maintains two computation traces at the same time: one
capturing the raw computation, and a higher-level graph for *automatic
differentiation* (AD). The index :py:attr:`index_ad` keeps track of the
variable index within the AD computation graph, if applicable.

:type: int)";

static const char *doc_ArrayBase_index_ad = R"(
If ``self`` is a *leaf* Dr.Jit array represented by an AD backend, this
property contains the variable index in the graph data structure storing the
computation trace for later differentiation (this graph can be visualized using
:py:func:`drjit.graphviz_ad`). A *non-leaf* array (e.g.
:py:class:`drjit.cuda.ad.Array2f`) consists of several AD variables, whose
indices must be queried separately.

Note that Dr.Jit maintains two computation traces at the same time: one
capturing the raw computation, and a higher-level graph for *automatic
differentiation* (AD). The index :py:attr:`index` keeps track of the
variable index within the raw computation graph, if applicable.

:type: int)";
