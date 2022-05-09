#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-variable"
#endif

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

static const char *doc_size_v = R"(
size_v(arg, /)
Return the (static) size of the outermost dimension of the provided Dr.Jit
array instance or type

Note that this function mainly exists to query type-level information. Use the
Python ``len()`` function to query the size in a way that does not distinguish
between static and dynamic arrays.

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns either the static size or :py:data:`drjit.Dynamic` when
    **arg** is a dynamic Dr.Jit array. Returns ``1`` for all other types.)";

static const char *doc_depth_v = R"(
depth_v(arg, /)
Return the depth of the provided Dr.Jit array instance or type

For example, an array consisting of floating point values (for example,
:py:class:`drjit.scalar.Array3f`) has depth ``1``, while an array consisting of
sub-arrays (e.g., :py:class:`drjit.cuda.Array3f`) has depth ``2``.

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the depth of the input, if it is a Dr.Jit array instance or
    type. Returns ``0`` for all other inputs.)";

static const char *doc_itemsize_v = R"(
itemsize_v(arg, /)
Return the per-item size (in bytes) of the scalar type underlying a Dr.Jit array

Args:
    arg (object): A Dr.Jit array instance or array type.

Returns:
    int: Returns the item size array elements in bytes.)";

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
    assert dr.value_t(dr.cuda.TensorXf) is float
    assert dr.value_t("test") is str

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Returns the value type of the provided Dr.Jit array, or the type of
    the input.
)";

static const char *doc_array_t = R"(
array_t(arg, /)
Return the *array form* of the provided Dr.Jit array or type.

There are several different cases:

- When `self` is a tensor, this property returns the storage representation
  of the tensor in the form of a linarized dynamic 1D array. For example,
  the following hold:

  .. code-block::

    assert dr.array_t(dr.scalar.TensorXf) is dr.scalar.ArrayXf
    assert dr.array_t(dr.cuda.TensorXf) is dr.cuda.Float

- When `arg` represents a special arithmetic object (matrix, quaternion, or
  complex number), `array_t` returns a similarly-shaped type with ordinary array
  semantics. For example, the following hold

  .. code-block::

    assert dr.array_t(dr.scalar.Complex2f) is dr.scalar.Array2f
    assert dr.array_t(dr.scalar.Matrix4f) is dr.scalar.Array44f

- In all other cases, the function returns the input type.

The property :py:func:`ArrayBase.array` returns a result with this the
type computed by this function.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Returns the array form as per the above description.
)";

static const char *doc_mask_t = R"(
mask_t(arg, /)
Return the *mask type* associated with the provided Dr.Jit array or type (i.e., the
type produced by comparisons involving the argument).

When the input is not a Dr.Jit array or type, the function returns the scalar
Python ``bool`` type. The following assertions illustrate the behavior of
:py:func:`mask_t`.


.. code-block::

    assert dr.mask_t(dr.scalar.Array3f) is dr.scalar.Array3b
    assert dr.mask_t(dr.cuda.Array3f) is dr.cuda.Array3b
    assert dr.mask_t(dr.cuda.Matrix4f) is dr.cuda.Array44b
    assert dr.mask_t("test") is bool

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Returns the mask type associated with the input or ``bool`` when the
    input is not a Dr.Jit array.
)";

static const char *doc_scalar_t = R"(
scalar_t(arg, /)
Return the *scalar type* associated with the provided Dr.Jit array or type (i.e., the
representation of elements at the lowest level)

When the input is not a Dr.Jit array or type, the function returns the input
unchanged. The following assertions illustrate the behavior of
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


static const char *doc_is_signed_v = R"(
is_signed_v(arg, /)
Check whether the input array instance or type is an signed Dr.Jit array
or a Python ``int`` or ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an signed Dr.Jit array or
    Python ``int`` or ``float`` instance or type.
)";


static const char *doc_is_unsigned_v = R"(
is_unsigned_v(arg, /)
Check whether the input array instance or type is an unsigned integer Dr.Jit
array or a Python ``bool`` value/type (masks and boolean values are also
considered to be unsigned).

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if **arg** represents an unsigned Dr.Jit array or
    Python ``bool`` instance or type.
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
Check whether the input is a *special* Dr.Jit array instance or type.

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

static const char *doc_zeros = R"(
Return a zero-valued Dr.Jit array of the desired shape.

When **dtype** refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), **shape** must be a tuple that determines the
tensor rank and shape.

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension.

When a tuple is specified, its must be compatible with static dimensions of the
**dtype**. For example, ``dr.zeros(dr.cuda.Array2f, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

When **dtype** refers to a mask array, the resulting array entries are
initialized to ``False``.


Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    shape (tuple | int): Shape of the desired array

Returns:
    object: A zero-initialized instance of type **dtype**.
)";

static const char *doc_ones = R"(
Return a Dr.Jit array of the desired shape initialized with the value 1.

When **dtype** refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), **shape** must be a tuple that determines the
tensor rank and shape.

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension.

When a tuple is specified, its must be compatible with static dimensions of the
**dtype**. For example, ``dr.ones(dr.cuda.Array2f, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

When **dtype** refers to a mask array, the resulting array entries are
initialized to ``True``.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    shape (tuple | int): Shape of the desired array

Returns:
    object: A zero-initialized instance of type **dtype**.
)";


static const char *doc_full = R"(
Return a constant-valued Dr.Jit array of the desired shape.

When **dtype** refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), **shape** must be a tuple that determines the
tensor rank and shape.

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension.

When a tuple is specified, its must be compatible with static dimensions of the
**dtype**. For example, ``dr.full(dr.cuda.Array2f, value=123.0, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    value (object): An instance of the underlying scalar type (``float``/``int``/``bool``, etc.)
                    that will be used to initialize the array contents.
    shape (tuple | int): Shape of the desired array

Returns:
    object: A constant-initialized instance of type **dtype**.
)";

static const char *doc_arange = R"(
This function generates an integer sequence on the interval [**start**,
**stop**) with step size **step**, where **start** = 0 and **step** = 1 if not
specified.

Args:
    dtype (type): Desired Dr.Jit array type. The **dtype** must refer to a
                  dynamically sized 1D Dr.Jit array such as
                  :py:class:`drjit.scalar.ArrayXu` or
                  :py:class:`drjit.cuda.Float`.
    start (int): Start of the interval. The default value is `0`.
    stop/size (int): End of the interval (not included). The name of this parameter
                     differs between the two provided overloads.
    step (int): Spacing between values. The default value is `1`.

Returns:
    object: The computed sequence of type **dtype**.
)";


static const char *doc_linspace = R"(
This function generates an evenly spaced floating point sequence of size
**num** covering the interval [**start**, **stop**].

Args:
    dtype (type): Desired Dr.Jit array type. The **dtype** must refer to a
                  dynamically sized 1D Dr.Jit floating point array, such as
                  :py:class:`drjit.scalar.ArrayXf` or
                  :py:class:`drjit.cuda.Float`.
    start (float): Start of the interval.
    stop (float): End of the interval.
    num (int): Number of samples to generate.
    endpoint (bool): Should the interval endpoint be included? The default is `True`.

Returns:
    object: The computed sequence of type **dtype**.
)";

static const char *doc_empty = R"(
Return a empty Dr.Jit array of the desired shape.

When **dtype** refers to a tensorial type (e.g.,
:py:class:`drjit.cuda.TensorXf`), **shape** must be a tuple that determines the
tensor rank and shape.

For simpler vectorized types (e.g., :py:class:`drjit.cuda.Array2f`), it can be
an integer that specifies the size along the last (dynamic) dimension.

When a tuple is specified, its must be compatible with static dimensions of the
**dtype**. For example, ``dr.empty(dr.cuda.Array2f, shape=(3, 100))`` fails,
since the leading dimension is incompatible with
:py:class:`drjit.cuda.Array2f`.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or :ref:`custom data structure <custom-struct>`.
    shape (tuple | int): Shape of the desired array

Returns:
    object: An empty instance of type **dtype**.
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

The expressions ``drjit.shape(arg)`` and ``arg.shape`` are equivalent.

:type: tuple | NoneType)";

static const char *doc_ArrayBase_array = R"(
This member plays multiple roles:

- When `self` is a tensor, this property returns the storage representation
  of the tensor in the form of a linarized dynamic 1D array.

- When `self` is a special arithmetic object (matrix, quaternion, or complex
  number), `array` provides an ordinary copy of the same data with ordinary
  array semantics.

- In all other cases, `array` is simply a reference to `self`.

:type: :py:func:`array_t(self) <array_t>`)";

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



static const char *doc_bool_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into a boolean version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3i`), it
   returns a *boolean* version (e.g. :py:class:`drjit.cuda.Array3b`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``bool_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``bool``
   (when called with a type) or it tries to convert the input into a ``bool``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_uint_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into a *unsigned integer*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
   returns an *unsigned integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``uint_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";


static const char *doc_int_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into a *signed integer*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
   returns an *signed integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``int_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";


static const char *doc_float_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into a *floating point*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u64`), it
   returns an *floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``float_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``float``
   (when called with a type) or it tries to convert the input into a ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_uint32_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an *unsigned 32 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *unsigned 32 bit* version (e.g. :py:class:`drjit.cuda.Array3u`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``uint32_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_int32_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an *signed 32 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *signed 32 bit* version (e.g. :py:class:`drjit.cuda.Array3i`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``int32_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_uint64_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an *unsigned 64 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *unsigned 64 bit* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``uint64_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_int64_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an *signed 64 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *signed 64 bit* version (e.g. :py:class:`drjit.cuda.Array3i64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``int64_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``int``
   (when called with a type) or it tries to convert the input into an ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_float32_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an 32 bit floating
point version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
   returns a *32 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``float32_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``float``
   (when called with a type) or it tries to convert the input into a ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";


static const char *doc_float64_array_t = R"(
Converts the provided Dr.Jit array/tensor, or type into an 64 bit floating
point version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
   returns a *64 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

2. When invoked with a Dr.Jit array *value*, it casts the input into the
   type ``float64_array_t(type(arg))``

3. When the input is not a Dr.Jit array or type, the function returns ``float``
   (when called with a type) or it tries to convert the input into a ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    object: Result of the conversion as described above.
)";

static const char *doc_slice_index = R"(
Computes an index array that can be used to slice a tensor. It is used
internally by Dr.Jit to implement complex cases of the ``__getitem__``
operation.

It must be called with the desired output **dtype**, which must be a dynamic
32-bit integer array. The **shape** parameter specifies the dimensions of the
input tensor, and **indices** contains the entries that would appear in a
complex slicing operation, but as a tuple. For example, ``[5:10:2, ..., None]``
would be specified as ``(slice(5, 10, 2), Ellipsis, None)``.

An example is shown below:

.. code-block:: pycon

    >>> dr.slice_index(dtype=dr.scalar.ArrayXu,
                       shape=(10, 1),
                       indices=(slice(0, 10, 2), 0))
    [0, 2, 4, 6, 8]

Args:
    dtype (type): A dynamic 32-bit unsigned integer Dr.Jit array type,
                  such as :py:class:`dr.scalar.ArrayXu` or
                  :py:class:`dr.cuda.UInt`.

    shape (tuple[int, ...]): The shape of the tensor to be sliced.

    indices (tuple[int|slice|ellipsis|NoneType|dr.ArrayBase, ...]):
        A set of indices used to slice the tensor. Its entries can be ``slice``
        instances, integers, integer arrays, ``...`` (ellipsis) or ``None``.

Returns:
    tuple[tuple[int, ...], drjit.ArrayBase]: Tuple consisting of the output array
    shape and a flattened unsigned integer array of type **dtype** containing
    element indices.
)";

static const char *doc_gather = R"(
Gather values from a flat array

This operation performs a *gather* (i.e., indirect memory read) from the
**source** array at position **index**. The optional **active** argument can be
used to disable some of the components, which is useful when not all indices
are valid; the corresponding output will be zero in this case.

The provided **dtype** is typically equal to ``type(source)``, in which case
this operation can be interpreted as a parallelized version of the Python array
indexing expression ``source[index]`` with optional masking (however, in
contrast to array indexing, negative indices are not handled).

This function can also be used to gather *nested* arrays like
:py:class:`drjit.cuda.Vector3f`, which represents a sequence of 3D vectors.
This is useful for populating populate vectors, matrices, etc., from a flat
input array. For example, the following operation loads 3D vectors

.. code-block::

    result = dr.gather(dr.cuda.Vector3f, source, index)

and is equivalent to

.. code-block::

    result = dr.Vector3f(
        dr.cuda.Float, source, index*3 + 0),
        dr.cuda.Float, source, index*3 + 1),
        dr.cuda.Float, source, index*3 + 2)
    )

.. danger::

    The indices provided to this operation are unchecked. Out-of-bounds reads
    are undefined behavior (if not disabled via the **active** parameter) and may
    crash the application. Negative indices are not permitted.

Args:
    dtype (type): The desired output type (typically equal to ``type(source)``,
      but other variations are possible as well, see the description above.)
    source (drjit.ArrayBase): a 1D dynamic Dr.Jit array from which data
      should be read.
    index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
      :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
      specifying gather indices. Dr.Jit will attempt an implicit conversion if
      another type is provided.
    active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
      :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
      specifying active components. Dr.Jit will attempt an implicit conversion
      if another type is provided. The default is `True`.

Returns:
    object: An instance of type **dtype** containing the result of the gather
    operation.
)";

static const char *doc_scatter = R"(
Scatter values into a flat array

This operation performs a *scatter* (i.e., indirect memory write) of the
**value** parameter to the **target** array at position **index**. The optional
**active** argument can be used to disable some of the individual write
operations, which is useful when not all provided values or indices are valid.

When **source** and **target** have the same types, this operation can be
interpreted as a parallelized version of the Python array indexing expression
``target[index] = value`` with optional masking. In contrast to array
indexing, negative indices are not handled, and conflicting writes to the
same location are considered undefined behavior.

This function can also be used to scatter *nested* arrays like
:py:class:`drjit.cuda.Vector3f`, which represents a sequence of 3D vectors.
This is useful for storing vectors, matrices, etc., into a flat
output array. For example, the following operation stores 3D vectors

.. code-block::

    target = dr.empty(dr.Float, 1024*3)
    dr.scatter(target, value, index)

and is equivalent to

.. code-block::

    dr.scatter(target, value[0], index*3 + 0)
    dr.scatter(target, value[1], index*3 + 1)
    dr.scatter(target, value[2], index*3 + 2)

.. danger::

    The indices provided to this operation are unchecked. Out-of-bounds writes
    are undefined behavior (if not disabled via the **active** parameter) and may
    crash the application. Negative indices are not permitted.

Args:
    target (drjit.ArrayBase): a 1D dynamic Dr.Jit array into which data
      should be written.
    value (object): The values to be written (typically of type ``type(target)``,
      but other variations are possible as well, see the description above.)
      Dr.Jit will attempt an implicit conversion if the the input is not an
      array type.
    index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
      :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
      specifying gather indices. Dr.Jit will attempt an implicit conversion if
      another type is provided.
    active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
      :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
      specifying active components. Dr.Jit will attempt an implicit conversion
      if another type is provided. The default is `True`.
)";

static const char *doc_ravel = R"(
Convert the input into a contiguous flat array

Args:
    arg (drjit.ArrayBase): An arbitrary Dr.Jit array or tensor

    order (str): A single character indicating the index order. ``'C'`` (the
        default) specifies row-major/C-style ordering, in which case the last
        index changes at the highest frequency. The other option ``'F'``
        indicates column-major/Fortran-style ordering, in which case the 
        first index changes at the highest frequency.


Returns:
    object: A dynamic 1D array containing the flattened representation of
    **arg** with the desired ordering. The type of the return value depends on
    the type of the input. When **arg** is already contiguous/flattened, this
    function returns it without making a copy.
)";


static const char *doc_schedule = R"(
Schedule the provided JIT variable(s) for later evaluation

This function causes **args** to be evaluated by the next kernel launch. In
other words, the effect of this operation is deferred: the next time that
Dr.Jit's LLVM or CUDA backends compile and execute code, they will include the
*trace* of the specified variables in the generated kernel and turn them into
an explicit memory-based representation.

Scheduling and evaluation of traced computation happens automatically, hence it
is rare that a user would need to call this function explicitly. Explicit
scheduling can improve performance in certain cases---for example, consider the
following code:

.. code-block::

    # Computation that produces Dr.Jit arrays
    a, b = ...

    # The following line launches a kernel that computes 'a'
    print(a)

    # The following line launches a kernel that computes 'b'
    print(b)

If the traces of ``a`` and ``b`` overlap (perhaps they reference computation
from an earlier step not shown here), then this is inefficient as these steps
will be executed twice. It is preferable to launch bigger kernels that leverage
common subexpressions, which is what :py:func:`drjit.schedule()` enables:

.. code-block::

    a, b = ... # Computation that produces Dr.Jit arrays

    # Schedule both arrays for deferred evaluation, but don't evaluate yet
    dr.schedule(a, b)

    # The following line launches a kernel that computes both 'a' and 'b'
    print(a)

    # References the stored array, no kernel launch
    print(b)

Note that :py:func:`drjit.eval()` would also have been a suitable alternative 
in the above example; the main difference to :py:func:`drjit.schedule()` is
that it does the evaluation immediately without deferring the kernel launch.

This function accepts a variable-length keyword argument and processes it
as follows:

- It recurses into sequences (``tuple``, ``list``, etc.)
- It recurses into the values of mappings (``dict``, etc.)
- It recurses into the fields of :ref:`custom data structures <custom-struct>`.

During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
arrays and incompatible types are ignored. Multiple variables can be
equivalently scheduled with a single :py:func:`drjit.schedule()` call or a
sequence of calls to :py:func:`drjit.schedule()`. Variables that are garbage
collected between the original :py:func:`drjit.schedule()` call and the next
kernel launch are ignored and will not be stored in memory.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
      :ref:`custom data structures <custom-struct>`, sequences, or mappings.
      The function will recursively traverse data structures to discover all
      Dr.Jit arrays.
)";

static const char *doc_eval = R"(
Immediately evaluate the provided JIT variable(s)

This function immediately invokes Dr.Jit's LLVM or CUDA backends to compile and
then execute a kernel containing the *trace* of the specified variables,
turning them into an explicit memory-based representation. The generated
kernel(s) will also include previously scheduled computation. The function
:py:func:`drjit.eval()` internally calls :py:func:`drjit.schedule()`---specifically,

.. code-block::

    dr.eval(arg_1, arg_2, ...)

is equivalent to

.. code-block::

    dr.schedule(arg_1, arg_2, ...)
    dr.eval()

Variable evaluation happens automatically as needed, hence it is rare that a
user would need to call this function explicitly. Explicit evaluation can
slightly improve performance in certain cases (the documentation of
:py:func:`drjit.schedule()` shows an example of such a use case.)

This function accepts a variable-length keyword argument and processes it
as follows:

- It recurses into sequences (``tuple``, ``list``, etc.)
- It recurses into the values of mappings (``dict``, etc.)
- It recurses into the fields of :ref:`custom data structures <custom-struct>`.

During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
arrays and incompatible types are ignored.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
      :ref:`custom data structures <custom-struct>`, sequences, or mappings.
      The function will recursively traverse data structures to discover all
      Dr.Jit arrays.
)";

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
