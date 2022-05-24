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
    bool: ``True`` if ``arg`` or type(``arg``) is a Dr.Jit array type, and ``False`` otherwise)";

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
    bool: ``True`` if ``arg`` has a ``DRJIT_STRUCT`` member)";

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
    ``arg`` is a dynamic Dr.Jit array. Returns ``1`` for all other types.)";

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
  of the tensor in the form of a linearized dynamic 1D array. For example,
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
    bool: ``True`` if ``arg`` represents a Dr.Jit mask array or Python ``bool``
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
    bool: ``True`` if ``arg`` represents an integral Dr.Jit array or
    Python ``int`` instance or type.
)";

static const char *doc_is_float_v = R"(
is_float_v(arg, /)
Check whether the input array instance or type is a Dr.Jit floating point array
or a Python ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents a Dr.Jit floating point array or
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
    bool: ``True`` if ``arg`` represents an arithmetic Dr.Jit array or
    Python ``int`` or ``float`` instance or type.
)";


static const char *doc_is_signed_v = R"(
is_signed_v(arg, /)
Check whether the input array instance or type is an signed Dr.Jit array
or a Python ``int`` or ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an signed Dr.Jit array or
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
    bool: ``True`` if ``arg`` represents an unsigned Dr.Jit array or
    Python ``bool`` instance or type.
)";

static const char *doc_is_jit_v = R"(
is_jit_v(arg, /)
Check whether the input array instance or type represents a type that
undergoes just-in-time compilation.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an array type from the
    ``drjit.cuda.*`` or ``drjit.llvm.*`` namespaces, and ``False`` otherwise.
)";

static const char *doc_is_cuda_v = R"(
is_cuda_v(arg, /)
Check whether the input is a Dr.Jit CUDA array instance or type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an array type from the
    ``drjit.cuda.*`` namespace, and ``False`` otherwise.
)";

static const char *doc_is_llvm_v = R"(
is_llvm_v(arg, /)
Check whether the input is a Dr.Jit LLVM array instance or type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an array type from the
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
    bool: ``True`` if ``arg`` represents an array type from the
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

static const char *doc_maximum = R"(
maximum(arg0, arg1, /) -> float | int | drjit.ArrayBase
Compute the element-wise maximum value of the provided inputs.

This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
according to the usual implicit type conversion rules).

Args:
    arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Maximum of the input(s))";


static const char *doc_minimum = R"(
minimum(arg0, arg1, /) -> float | int | drjit.ArrayBase
Compute the element-wise minimum value of the provided inputs.

This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
according to the usual implicit type conversion rules).

Args:
    arg0 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Minimum of the input(s))";


static const char *doc_max = R"(
max(arg, /) -> float | int | drjit.ArrayBase
Compute the maximum value in the provided input.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Maximum of the input)";


static const char *doc_min = R"(
min(arg, /) -> float | int | drjit.ArrayBase
Compute the minimum value in the provided input.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Minimum of the input)";


static const char *doc_sum = R"(
sum(arg, /) -> float | int | drjit.ArrayBase
Compute the sum of all array elements.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Sum of the input)";


static const char *doc_prod = R"(
prod(arg, /) -> float | int | drjit.ArrayBase
Compute the product of all array elements.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Product of the input)";


static const char *doc_all = R"(
all(arg, /) -> float | int | drjit.ArrayBase
Computes whether all input elements evaluate to ``True``.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Boolean array)";


static const char *doc_any = R"(
any(arg, /) -> float | int | drjit.ArrayBase
Computes whether any of the input elements evaluate to ``True``.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.


Args:
    arg (float | int | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Boolean array)";


static const char *doc_dot = R"(
dot(arg0, arg1, /) -> float | int | drjit.ArrayBase
Computes the dot product of two arrays.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg0 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    arg1 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Dot product of inputs)";


static const char *doc_norm = R"(
norm(arg, /) -> float | int | drjit.ArrayBase
Computes the norm of an array.

When the argument is a dynamic array, function performs a horizontal reduction.
Please see the section on :ref:`horizontal reductions <horizontal-reductions>`
for details.

Args:
    arg (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    Norm of the input)";


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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
using the native multi-function unit ("MUFU").

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Sine of the input)";

static const char *doc_cos = R"(
cos(arg, /)
Cosine approximation based on the CEPHES library.

The implementation of this function is designed to achieve low error on the
domain :math:`|x| < 8192` and will not perform as well beyond this range. See
the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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

When ``arg`` is a CUDA single precision array, the operation is implemented
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
Arccosine approximation based on the CEPHES library.

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
    float | drjit.ArrayBase: Arctangent of ``y``/``x``, using the argument signs to
    determine the quadrant of the return value)";

static const char *doc_ldexp = R"(
ldexp(x, n, /)
Multiply x by 2 taken to the power of n

Args:
    x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
    n (float | drjit.ArrayBase): A Python or Dr.Jit floating point type *without fractional component*

Returns:
    float | drjit.ArrayBase: The result of ``x`` multipled by 2 taken to the power ``n``.)";

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
    so that ``frac * 2**(exp + 1)`` equals ``arg``.)";


static const char *doc_fma = R"(
fma(arg0, arg1, arg2, /)
Perform a *fused multiply-add* (FMA) operation.

Given arguments ``arg0``, ``arg1``, and ``arg2``, this operation computes
``arg0`` * ``arg1`` + ``arg2`` using only one final rounding step. The
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
Return a zero-initialized instance of the desired type and shape

This function can create zero-initialized instances of various types. In
particular, ``dtype`` can be:

- A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
  specifies a sequence, it must be compatible with static dimensions of the
  ``dtype``. For example, ``dr.zeros(dr.cuda.Array2f, shape=(3, 100))`` fails,
  since the leading dimension is incompatible with
  :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
  the size of the last (dynamic) dimension, if available.

- A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
  specifies a sequence (list/tuple/..), it determines the tensor rank and
  shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
  the specified size.

- A :ref:`custom data structure <custom-struct>`. In this case,
  :py:func:`drjit.zero()` will invoke itself recursively to zero-initialize
  each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
initialized to ``False`` as opposed to zero.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`custom data structure <custom-struct>`.
    shape (Sequence[int] | int): Shape of the desired array

Returns:
    object: A zero-initialized instance of type ``dtype``.
)";

static const char *doc_ones = R"(
Return an instance of the desired type and shape filled with ones

This function can create one-initialized instances of various types. In
particular, ``dtype`` can be:

- A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
  specifies a sequence, it must be compatible with static dimensions of the
  ``dtype``. For example, ``dr.ones(dr.cuda.Array2f, shape=(3, 100))`` fails,
  since the leading dimension is incompatible with
  :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
  the size of the last (dynamic) dimension, if available.

- A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
  specifies a sequence (list/tuple/..), it determines the tensor rank and
  shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
  the specified size.

- A :ref:`custom data structure <custom-struct>`. In this case,
  :py:func:`drjit.ones()` will invoke itself recursively to initialize
  each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
initialized to ``True`` as opposed to one.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`custom data structure <custom-struct>`.
    shape (Sequence[int] | int): Shape of the desired array

Returns:
    object: A instance of type ``dtype`` filled with ones.
)";


static const char *doc_full = R"(
Return an constant-valued instance of the desired type and shape

This function can create constant-valued instances of various types. In
particular, ``dtype`` can be:

- A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
  specifies a sequence, it must be compatible with static dimensions of the
  ``dtype``. For example, ``dr.full(dr.cuda.Array2f, value=1.0, shape=(3,
  100))`` fails, since the leading dimension is incompatible with
  :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
  the size of the last (dynamic) dimension, if available.

- A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
  specifies a sequence (list/tuple/..), it determines the tensor rank and
  shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
  the specified size.

- A :ref:`custom data structure <custom-struct>`. In this case,
  :py:func:`drjit.full()` will invoke itself recursively to initialize
  each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`custom data structure <custom-struct>`.
    value (object): An instance of the underlying scalar type
      (``float``/``int``/``bool``, etc.) that will be used to initialize the
      array contents.
    shape (Sequence[int] | int): Shape of the desired array

Returns:
    object: A instance of type ``dtype`` filled with ``value``
)";

static const char *doc_empty = R"(
Return an uninitialized Dr.Jit array of the desired type and shape.

This function can create uninitialized buffers of various types. It is
essentially a wrapper around CPU/GPU variants of ``malloc()`` and produces
arrays filled with uninitialized/undefined data. It should only be used in
combination with a subsequent call to an operation like
:py:func:`drjit.scatter()` that overwrites the array contents with valid data.

The ``dtype`` parameter can be used to request:

- A Dr.Jit array type like :py:class:`drjit.cuda.Array2f`. When ``shape``
  specifies a sequence, it must be compatible with static dimensions of the
  ``dtype``. For example, ``dr.empty(dr.cuda.Array2f, shape=(3, 100))`` fails,
  since the leading dimension is incompatible with
  :py:class:`drjit.cuda.Array2f`. When ``shape`` is an integer, it specifies
  the size of the last (dynamic) dimension, if available.

- A tensorial type like :py:class:`drjit.scalar.TensorXf`. When ``shape``
  specifies a sequence (list/tuple/..), it determines the tensor rank and
  shape. When ``shape`` is an integer, the function creates a rank-1 tensor of
  the specified size.

- A :ref:`custom data structure <custom-struct>`. In this case,
  :py:func:`drjit.empty()` will invoke itself recursively to allocate memory
  for each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case, and the function returns a
  zero-initialized result (there is little point in instantiating uninitialized
  versions of scalar Python types).

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`custom data structure <custom-struct>`.
    shape (Sequence[int] | int): Shape of the desired array

Returns:
    object: An instance of type ``dtype`` with arbitrary/undefined contents.
)";


static const char *doc_arange = R"(
This function generates an integer sequence on the interval [``start``,
``stop``) with step size ``step``, where ``start`` = 0 and ``step`` = 1 if not
specified.

Args:
    dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
      dynamically sized 1D Dr.Jit array such as :py:class:`drjit.scalar.ArrayXu`
      or :py:class:`drjit.cuda.Float`.
    start (int): Start of the interval. The default value is `0`.
    stop/size (int): End of the interval (not included). The name of this
      parameter differs between the two provided overloads.
    step (int): Spacing between values. The default value is `1`.

Returns:
    object: The computed sequence of type ``dtype``.
)";


static const char *doc_linspace = R"(
This function generates an evenly spaced floating point sequence of size
``num`` covering the interval [``start``, ``stop``].

Args:
    dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
      dynamically sized 1D Dr.Jit floating point array, such as
      :py:class:`drjit.scalar.ArrayXf` or :py:class:`drjit.cuda.Float`.
    start (float): Start of the interval.
    stop (float): End of the interval.
    num (int): Number of samples to generate.
    endpoint (bool): Should the interval endpoint be included? The default is `True`.

Returns:
    object: The computed sequence of type ``dtype``.
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
Converts the provided Dr.Jit array/tensor type into a boolean version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3i`), it
   returns a *boolean* version (e.g. :py:class:`drjit.cuda.Array3b`).

2. When the input isn't a type, it returns ``bool_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``bool``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_uint_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *unsigned integer*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
   returns an *unsigned integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When the input isn't a type, it returns ``uint_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";


static const char *doc_int_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *signed integer*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
   returns an *signed integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When the input isn't a type, it returns ``int_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";


static const char *doc_float_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *floating point*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u64`), it
   returns an *floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

2. When the input isn't a type, it returns ``float_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_uint32_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *unsigned 32 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *unsigned 32 bit* version (e.g. :py:class:`drjit.cuda.Array3u`).

2. When the input isn't a type, it returns ``uint32_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_int32_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *signed 32 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *signed 32 bit* version (e.g. :py:class:`drjit.cuda.Array3i`).

2. When the input isn't a type, it returns ``int32_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_uint64_array_t = R"(
Converts the provided Dr.Jit array/tensor type into an *unsigned 64 bit*
version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *unsigned 64 bit* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When the input isn't a type, it returns ``uint64_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_int64_array_t = R"(
Converts the provided Dr.Jit array/tensor type into an *signed 64 bit* version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f`), it
   returns an *signed 64 bit* version (e.g. :py:class:`drjit.cuda.Array3i64`).

2. When the input isn't a type, it returns ``int64_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_float32_array_t = R"(
Converts the provided Dr.Jit array/tensor type into an 32 bit floating point version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
   returns a *32 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f`).

2. When the input isn't a type, it returns ``float32_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_float64_array_t = R"(
Converts the provided Dr.Jit array/tensor type into an 64 bit floating point version.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
   returns a *64 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

2. When the input isn't a type, it returns ``float64_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_detached_t =R"(
Converts the provided Dr.Jit array/tensor type into an non-differentiable version.

This function implements the following set of behaviors:

1. When invoked with a differentiable Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.ad.Array3f`), it
   returns a non-differentiable version (e.g. :py:class:`drjit.cuda.Array3f`).

2. When the input isn't a type, it returns ``detached_t(type(arg))``.

3. When the input type is non-differentiable or not a Dr.Jit array type, the function returns it unchanged.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_leaf_array_t =R"(
Extracts a leaf array type underlying a Python object tree, with a preference
for differentiable arrays.

This function implements the following set of behaviors:

1. When the input isn't a type, it returns ``leaf_array_t(type(arg))``.

2. When invoked with a Dr.Jit array type, returns the lowest-level array type
   underlying a potentially nested array.

3. When invoked with a sequence, mapping or custom data structure made of Dr.Jit arrays,
   examines underlying Dr.Jit array types and returns the lowest-level array type with
   a preference for differentiable arrays and floating points arrays.
   E.g. when passing a list containing arrays of type :py:class:`drjit.cuda.ad.Float` and :py:class:`drjit.cuda.UInt`,
   the function will return :py:class:`drjit.cuda.ad.Float`.

4. Otherwise returns ``None``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the extraction as described above.
)";

static const char *doc_expr_t =R"(
Computes the type of an arithmetic expression involving the provided Dr.Jit
arrays (instances or types), or builtin Python objects.

An exception will be raised when an invalid combination of types is provided.

For instance, this function can be used to compute the return type of the
addition of several Dr.Jit array:

.. code-block::

    a = drjit.llvm.Float(1.0)
    b = drjit.llvm.Array3f(1, 2, 3)
    c = drjit.llvm.ArrayXf(4, 5, 6)

    # type(a + b + c) == dr.expr_t(a, b, c) == drjit.llvm.ArrayXf

Args:
    *args (tuple): A variable-length list of Dr.Jit arrays, builtin Python
      objects, or types.

Returns:
    type: Result type of an arithmetic expression involving the provided variables.
)";

static const char *doc_slice_index = R"(
Computes an index array that can be used to slice a tensor. It is used
internally by Dr.Jit to implement complex cases of the ``__getitem__``
operation.

It must be called with the desired output ``dtype``, which must be a dynamic
32-bit integer array. The ``shape`` parameter specifies the dimensions of the
input tensor, and ``indices`` contains the entries that would appear in a
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
    shape and a flattened unsigned integer array of type ``dtype`` containing
    element indices.
)";

static const char *doc_gather = R"(
Gather values from a flat array or nested data structure

This function performs a *gather* (i.e., indirect memory read) from ``source``
at position ``index``. It expects a ``dtype`` argument and will return an
instance of this type. The optional ``active`` argument can be used to disable
some of the components, which is useful when not all indices are valid; the
corresponding output will be zero in this case.

This operation can be used in the following different ways:

1. When ``dtype`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
   this operation implements a parallelized version of the Python array
   indexing expression ``source[index]`` with optional masking. Example:

   .. code-block::

       source = dr.cuda.Float([...])
       index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
       result = dr.gather(dtype=type(source), source=source, index=index)

2. When ``dtype`` is a more complex type (e.g. a :ref:`custom source structure
   <custom-struct>`, nested Dr.Jit array, tuple, list, dictionary, etc.), the
   behavior depends:

   - When ``type(source)`` matches ``dtype``, the the gather operation threads
     through entries and invokes itself recursively. For example, the
     gather operation in

     .. code-block::

         result = dr.cuda.Array3f(...)
         index = dr.cuda.UInt([...])
         result = dr.gather(dr.cuda.Array3f, source, index)

     is equivalent to

     .. code-block::

         result = dr.cuda.Array3f(
             dr.gather(dr.cuda.Float, source.x, index),
             dr.gather(dr.cuda.Float, source.y, index),
             dr.gather(dr.cuda.Float, source.z, index))

   - Otherwise, the operation reconstructs the requested ``dtype`` from a flat
     ``source`` array, using C-style ordering with a suitably modified
     ``index``. For example, the gather below reads 3D vectors from a 1D array.


     .. code-block::

         source = dr.cuda.Float([...])
         index = dr.cuda.UInt([...])
         result = dr.gather(dr.cuda.Array3f, source, index)

     and is equivalent to

     .. code-block::

         result = dr.cuda.Vector3f(
             dr.gather(dr.cuda.Float, source, index*3 + 0),
             dr.gather(dr.cuda.Float, source, index*3 + 1),
             dr.gather(dr.cuda.Float, source, index*3 + 2))

.. danger::

    The indices provided to this operation are unchecked. Out-of-bounds reads
    are undefined behavior (if not disabled via the ``active`` parameter) and may
    crash the application. Negative indices are not permitted.

Args:
    dtype (type): The desired output type (typically equal to ``type(source)``,
      but other variations are possible as well, see the description above.)
    source (object): The object from which data should be read (typically a 1D
      Dr.Jit array, but other variations are possible as well, see the
      description above.)
    index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
      :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
      specifying gather indices. Dr.Jit will attempt an implicit conversion if
      another type is provided.
    active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
      :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
      specifying active components. Dr.Jit will attempt an implicit conversion
      if another type is provided. The default is `True`.

Returns:
    object: An instance of type ``dtype`` containing the result of the gather
    operation.
)";

static const char *doc_scatter = R"(
Scatter values into a flat array or nested data structure

This operation performs a *scatter* (i.e., indirect memory write) of the
``value`` parameter to the ``target`` array at position ``index``. The optional
``active`` argument can be used to disable some of the individual write
operations, which is useful when not all provided values or indices are valid.

This operation can be used in the following different ways:

1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
   this operation implements a parallelized version of the Python array
   indexing expression ``target[index] = value`` with optional masking. Example:

   .. code-block::

       target = dr.empty(dr.cuda.Float, 1024*1024)
       value = dr.cuda.Float([...])
       index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
       dr.scatter(target, value=value, index=index)

2. When ``target`` is a more complex type (e.g. a :ref:`custom source structure
   <custom-struct>`, nested Dr.Jit array, tuple, list, dictionary, etc.), the
   behavior depends:

   - When ``target`` and ``value`` are of the same type, the scatter operation
     threads through entries and invokes itself recursively. For example, the
     scatter operation in

     .. code-block::

         target = dr.cuda.Array3f(...)
         value = dr.cuda.Array3f(...)
         index = dr.cuda.UInt([...])
         dr.scatter(target, value, index)

     is equivalent to

     .. code-block::

         dr.scatter(target.x, value.x, index)
         dr.scatter(target.y, value.y, index)
         dr.scatter(target.z, value.z, index)

   - Otherwise, the operation flattens the ``value`` array and writes it using
     C-style ordering with a suitably modified ``index``. For example, the
     scatter below writes 3D vectors into a 1D array.

     .. code-block::

         target = dr.cuda.Float(...)
         value = dr.cuda.Array3f(...)
         index = dr.cuda.UInt([...])
         dr.scatter(target, value, index)

     and is equivalent to

     .. code-block::

         dr.scatter(target, value.x, index*3 + 0)
         dr.scatter(target, value.y, index*3 + 1)
         dr.scatter(target, value.z, index*3 + 2)

.. danger::

    The indices provided to this operation are unchecked. Out-of-bounds writes
    are undefined behavior (if not disabled via the ``active`` parameter) and may
    crash the application. Negative indices are not permitted.

    Dr.Jit makes no guarantees about the expected behavior when a scatter
    operation has *conflicts*, i.e., when a specific position is written
    multiple times by a single :py:func:`drjit.scatter()` operation.

Args:
    target (object): The object into which data should be written (typically a
      1D Dr.Jit array, but other variations are possible as well, see the
    description above.)
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

This operation takes a Dr.Jit array, typically with some static and some
dynamic dimensions (e.g., :py:class:`drjit.cuda.Array3f` with shape
`3xN`), and converts it into a flattened 1D dynamically sized array (e.g.,
:py:class:`drjit.cuda.Float`) using either a C or Fortran-style ordering
convention.

It can also convert Dr.Jit tensors into a flat representation, though only
C-style ordering is supported in this case.

For example,

.. code-block::

    x = dr.cuda.Array3f([1, 2], [3, 4], [5, 6])
    y = dr.ravel(x, order=...)

will produce

- ``[1, 3, 5, 2, 4, 6]`` with ``order='F'`` (the default for Dr.Jit arrays),
  which means that X/Y/Z components alternate.
- ``[1, 2, 3, 4, 5, 6]`` with ``order='C'``, in which case all X coordinates
  are written as a contiguous block followed by the Y- and then Z-coordinates.

Args:
    array (drjit.ArrayBase): An arbitrary Dr.Jit array or tensor

    order (str): A single character indicating the index order. ``'F'``
      indicates column-major/Fortran-style ordering, in which case the first
      index changes at the highest frequency. The alternative ``'C'`` specifies
      row-major/C-style ordering, in which case the last index changes at the
      highest frequency. The default value ``'A'`` (automatic) will use F-style
      ordering for arrays and C-style ordering for tensors.

Returns:
    object: A dynamic 1D array containing the flattened representation of
    ``array`` with the desired ordering. The type of the return value depends
    on the type of the input. When ``array`` is already contiguous/flattened,
    this function returns it without making a copy.
)";


static const char *doc_unravel = R"(
Load a sequence of Dr.Jit vectors/matrices/etc. from a contiguous flat array

This operation implements the inverse of :py:func:`drjit.ravel()`. In contrast
to :py:func:`drjit.ravel()`, it requires one additional parameter (``dtype``)
specifying type of the return value. For example,

.. code-block::

    x = dr.cuda.Float([1, 2, 3, 4, 5, 6])
    y = dr.unravel(dr.cuda.Array3f, x, order=...)

will produce an array of two 3D vectors with different contents depending
on the indexing convention:

- ``[1, 2, 3]`` and ``[4, 5, 6]`` when unraveled with ``order='F'`` (the
  default for Dr.Jit arrays), and
- ``[1, 3, 5]`` and ``[2, 4, 6]`` when unraveled with ``order='C'``

Args:
    dtype (type): An arbitrary Dr.Jit array type

    array (drjit.ArrayBase): A dynamically sized 1D Dr.Jit array instance
       that is compatible with ``dtype``. In other words, both must have the
       same underlying scalar type and be located imported in the same package
       (e.g., ``drjit.llvm.ad``).

    order (str): A single character indicating the index order. ``'F'`` (the
       default) indicates column-major/Fortran-style ordering, in which case
       the first index changes at the highest frequency. The alternative
       ``'C'`` specifies row-major/C-style ordering, in which case the last
       index changes at the highest frequency.


Returns:
    object: An instance of type ``dtype`` containing the result of the unravel
    operation.
)";


static const char *doc_schedule = R"(
Schedule the provided JIT variable(s) for later evaluation

This function causes ``args`` to be evaluated by the next kernel launch. In
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

Returns:
    bool: ``True`` if a variable was scheduled, ``False`` if the operation did
    not do anything.
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

Returns:
    bool: ``True`` if a variable was evaluated, ``False`` if the operation did
    not do anything.
)";

static const char *doc_dlpack_device = R"(
Returns a tuple containing the DLPack device type and device ID associated with
the given array)";

static const char *doc_dlpack = R"(
Returns a DLPack capsule representing the data in this array.

This operation may potentially perform a copy. For example, nested arrays like
:py:class:`drjit.llvm.Array3f` or :py:class:`drjit.cuda.Matrix4f` need to be
rearranged into a contiguous memory representation before they can be exposed.

In other case, e.g. for :py:class:`drjit.llvm.Float`,
:py:class:`drjit.scalar.Array3f`, or :py:class:`drjit.scalar.ArrayXf`, the data
is already contiguous and a zero-copy approach is used instead.)";

static const char *doc_array = R"(
Returns a NumPy array representing the data in this array.

This operation may potentially perform a copy. For example, nested arrays like
:py:class:`drjit.llvm.Array3f` or :py:class:`drjit.cuda.Matrix4f` need to be
rearranged into a contiguous memory representation before they can be wrapped.

In other case, e.g. for :py:class:`drjit.llvm.Float`,
:py:class:`drjit.scalar.Array3f`, or :py:class:`drjit.scalar.ArrayXf`, the data
is already contiguous and a zero-copy approach is used instead.)";

static const char *doc_detach =R"(
Transforms the input variable into its non-differentiable version (*detaches* it
from the AD computational graph).

This function is able to traverse data-structures such a sequences, mappings or
:ref:`custom data structure <custom-struct>` and applies the transformation to the
underlying variables.

When the input variable isn't a Dr.Jit differentiable array, it is returned as it is.

While the type of the returned array is preserved by default, it is possible to
set the ``preserve_type`` argument to false to force the returned type to be
non-differentiable.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequence, or mapping.

    preserve_type (bool): Defines whether the returned variable should preserve
        the type of the input variable.
Returns:
    object: The detached variable.
)";

static const char *doc_set_grad_enabled =R"(
Enable or disable gradient tracking on the provided variables.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
        :ref:`custom data structures <custom-struct>`, sequences, or mappings.

    value (bool): Defines whether gradient tracking should be enabled or
        disabled.
)";

static const char *doc_enable_grad = R"(
Enable gradient tracking for the provided variables.

This function accepts a variable-length list of arguments and processes it
as follows:

- It recurses into sequences (``tuple``, ``list``, etc.)
- It recurses into the values of mappings (``dict``, etc.)
- It recurses into the fields of :ref:`custom data structures <custom-struct>`.

During recursion, the function enables gradient tracking for all Dr.Jit arrays.
For every other types, this function won't do anything.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
        :ref:`custom data structures <custom-struct>`, sequences, or mappings.
)";

static const char *doc_disable_grad =R"(
Disable gradient tracking for the provided variables.

This function accepts a variable-length list of arguments and processes it
as follows:

- It recurses into sequences (``tuple``, ``list``, etc.)
- It recurses into the values of mappings (``dict``, etc.)
- It recurses into the fields of :ref:`custom data structures <custom-struct>`.

During recursion, the function disables gradient tracking for all Dr.Jit arrays.
For every other types, this function won't do anything.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
        :ref:`custom data structures <custom-struct>`, sequences, or mappings.
)";

static const char *doc_grad_enabled =R"(
Return whether gradient tracking is enabled on any of the given variables.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances,
      :ref:`custom data structures <custom-struct>`, sequences, or mappings.
      The function will recursively traverse data structures to discover all
      Dr.Jit arrays.

Returns:
    bool: ``True`` if any variable has gradient tracking enabled, ``False`` otherwise.
)";

static const char *doc_grad =R"(
Return the gradient value associated to a given variable.

When the variable doesn't have gradient tracking enabled, this function returns ``0``.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.

    preserve_type (bool): Defines whether the returned variable should preserve
        the type of the input variable.

Returns:
    object: the gradient value associated to the input variable.
)";

static const char *doc_set_grad =R"(
Set the gradient value to the provided variable.

Broadcasting is applied to the gradient value if necessary and possible to match
the type of the input variable.

Args:
    dst (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.

    src (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.
)";

static const char *doc_accum_grad =R"(
Accumulate into the gradient of a variable.

Broadcasting is applied to the gradient value if necessary and possible to match
the type of the input variable.

Args:
    dst (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.

    src (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.
)";

static const char *doc_replace_grad =R"(
Replace the gradient value of ``dst`` with the one of ``src``.

Broadcasting is applied to ``dst`` if necessary to match the type of ``src``.

Args:
    dst (object): An arbitrary Dr.Jit array, tensor, or scalar builtin instance.

    src (object): An differentiable Dr.Jit array or tensor.

Returns:
    object: the variable with the replaced gradients.
)";

static const char *doc_enqueue =R"(
Enqueues variable for the subsequent AD traversal.

In Dr.Jit, the process of automatic differentiation is split into two parts:

1. Discover and enqueue the variables to be considered as inputs during the
   subsequent AD traversal.
2. Traverse the AD graph starting from the enqueued variables to propagate the
   gradients towards the output variables (e.g. leaf in the AD graph).


This function handles the first part can operate in different modes depending on
the specified ``mode``:

- ``ADMode.Forward``: the provided ``value`` will be considered as input during
  the subsequent AD traversal.

- ``ADMode.Backward``: a traversal of the AD graph starting from the provided
  ``value`` will take place to find all potential source of gradients and
  enqueue them.

For example, a typical chain of operations to forward propagate the gradients
from ``a`` to ``b`` would look as follow:

.. code-block::

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)
    b = f(a) # some computation involving `a`
    dr.set_gradient(a, 1.0)
    dr.enqueue(dr.ADMode.Forward, a)
    dr.traverse(dr.llvm.ad.Float, dr.ADMode.Forward)
    grad = dr.grad(b)

It could be the case that ``f(a)`` involves other differentiable variables that
already contain some gradients. In this situation we can use ``ADMode.Backward``
to discover and enqueue them before the traversal.

.. code-block::

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)
    b = f(a, ...) # some computation involving `a` and some hidden variables
    dr.set_gradient(a, 1.0)
    dr.enqueue(dr.ADMode.Backward, b)
    dr.traverse(dr.llvm.ad.Float, dr.ADMode.Forward)
    grad = dr.grad(b)

Dr.Jit also provides a higher level API that encapsulate this logic in a few
different functions:

- :py:func:`drjit.forward_from`, :py:func:`drjit.forward`, :py:func:`drjit.forward_to`
- :py:func:`drjit.backward_from`, :py:func:`drjit.backward`, :py:func:`drjit.backward_to`

Args:
    mode (ADMode): defines the enqueuing mode (backward or forward)

    value (object): An arbitrary Dr.Jit array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.
)";

static const char *doc_traverse = R"(
Propagate derivatives through the enqueued set of edges in the AD computational
graph in the direction specified by ``mode``.

By default, Dr.Jit's AD system destructs the enqueued input graph during AD
traversal. This frees up resources, which is useful when working with large
wavefronts or very complex computation graphs. However, this also prevents
repeated propagation of gradients through a shared subgraph that is being
differentiated multiple times.

To support more fine-grained use cases that require this, the following flags
can be used to control what should and should not be destructed:

- ``ADFlag.ClearNone``: clear nothing
- ``ADFlag.ClearEdges``: delete all traversed edges from the computation graph
- ``ADFlag.ClearInput``: clear the gradients of processed input vertices (in-degree == 0)
- ``ADFlag.ClearInterior``: clear the gradients of processed interior vertices (out-degree != 0)
- ``ADFlag.ClearVertices``: clear gradients of processed vertices only, but leave edges intact
- ``ADFlag.Default``: clear everything (default behaviour)

Args:
    type (type): defines the Dr.JIT array type used to build the AD graph

    mode (ADMode): defines the mode traversal (backward or forward)

    flags (ADFlag): flags to control what should and should not be destructed
        during forward/backward mode traversal.
)";

static const char *doc_forward_from =R"(
Forward propagates gradients from a provided Dr.Jit differentiable array.

This function will first see the gradient value of the provided variable to ``1.0``
before executing the AD graph traversal.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

Args:
    arg (object): A Dr.Jit differentiable array instance.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.
)";

static const char *doc_forward =R"(
Forward propagates gradients from a provided Dr.Jit differentiable array.

This function will first see the gradient value of the provided variable to ``1.0``
before executing the AD graph traversal.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

This function is an alias of :py:func:`drjit.forward_from`.

Args:
    arg (object): A Dr.Jit differentiable array instance.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.
)";

static const char *doc_forward_to =R"(
Forward propagates gradients to a set of provided Dr.Jit differentiable arrays.

Internally, the AD computational graph will be first traversed backward to find
all potential source of gradient for the provided array. Then only the forward
gradient propagation traversal takes place.

The ``flags`` argument should be provided as a keyword argument for this function.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

Args:
    *args (tuple): A variable-length list of Dr.Jit differentiable array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.

Returns:
    object: the gradient value associated to the output variables.
)";

static const char *doc_backward_from =R"(
Backward propagates gradients from a provided Dr.Jit differentiable array.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

Args:
    arg (object): A Dr.Jit differentiable array instance.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.
)";

static const char *doc_backward =R"(
Backward propagate gradients from a provided Dr.Jit differentiable array.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

This function is an alias of :py:func:`drjit.backward_from`.

Args:
    arg (object): A Dr.Jit differentiable array instance.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.
)";

static const char *doc_backward_to =R"(
Backward propagate gradients to a set of provided Dr.Jit differentiable arrays.

Internally, the AD computational graph will be first traversed *forward* to find
all potential source of gradient for the provided array. Then only the backward
gradient propagation traversal takes place.

The ``flags`` argument should be provided as a keyword argument for this function.

An exception will be raised when the provided array doesn't have gradient tracking
enabled or if it isn't an instance of a Dr.Jit differentiable array type.

Args:
    *args (tuple): A variable-length list of Dr.Jit differentiable array, tensor,
        :ref:`custom data structure <custom-struct>`, sequences, or mapping.

    flags (ADFlag): flags to control what should and should not be destructed
        during the traversal.

Returns:
    object: the gradient value associated to the output variables.
)";

static const char *doc_graphviz =R"(
Assembles a graphviz diagram for the computational graph trace by the JIT.

Args:
    as_str (bool): whether the function should return the graphviz object as
        a string representation or not.

Returns:
    object: the graphviz obj (or its string representation).
)";

static const char *doc_graphviz_ad =R"(
Assembles a graphviz diagram for the computational graph trace by the AD system.

Args:
    as_str (bool): whether the function should return the graphviz object as
        a string representation or not.

Returns:
    object: the graphviz obj (or its string representation).
)";

static const char *doc_label =R"(
Returns the label of a given Dr.Jit array.

Args:
    arg (object): a Dr.Jit array instance.

Returns:
    str: the label of the given variable.
)";

static const char *doc_set_label =R"(
Sets the label of a provided Dr.Jit array, either in the JIT or the AD system.

When a :ref:`custom data structure <custom-struct>` is provided, the field names
will be used as suffix for the variables labels.

When a sequence or static array is provided, the item's indices will be appended
to the label.

When a mapping is provided, the item's key will be appended to the label.

Args:
    *arg (tuple): a Dr.Jit array instance and its corresponding label ``str`` value.

    **kwarg (dict): A set of (keyword, object) pairs.
)";

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif
