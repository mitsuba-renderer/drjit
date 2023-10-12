#if defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-variable"
#endif

static const char *doc_is_array_v = R"(
Check if the input is a Dr.Jit array instance or type

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` or type(``arg``) is a Dr.Jit array type, and
      ``False`` otherwise)";

static const char *doc_is_struct_v = R"(
Check if the input is a Dr.Jit-compatible data structure

Custom data structures can be made compatible with various Dr.Jit operations by
specifying a ``DRJIT_STRUCT`` member. See the section on :ref:`Pytrees
<pytrees>` for details. This type trait can be used to check
for the existence of such a field.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` has a ``DRJIT_STRUCT`` member)";

static const char *doc_size_v = R"(
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
Return the per-item size (in bytes) of the scalar type underlying a Dr.Jit array

Args:
    arg (object): A Dr.Jit array instance or array type.

Returns:
    int: Returns the item size array elements in bytes.)";

static const char *doc_value_t = R"(
Return the *value type* underlying the provided Dr.Jit array or type (i.e., the
type of values obtained by accessing the contents using a 1D index).

When the input is not a Dr.Jit array or type, the function returns the input
type without changes. The following code fragment shows several example uses of
:py:func:`value_t`.

.. code-block::

    assert dr.value_t(dr.scalar.Array3f) is float
    assert dr.value_t(dr.cuda.Array3f) is dr.cuda.Float
    assert dr.value_t(dr.cuda.Matrix4f) is dr.cuda.Array4f
    assert dr.value_t(dr.cuda.TensorXf) is float
    assert dr.value_t(str) is str
    assert dr.value_t("test") is str

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Returns the value type of the provided Dr.Jit array, or the type of
    the input.
)";

static const char *doc_array_t = R"(
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
Return the *mask type* associated with the provided Dr.Jit array or type (i.e., the
type produced by comparisons involving the argument).

When the input is not a Dr.Jit array or type, the function returns the scalar
Python ``bool`` type. The following assertions illustrate the behavior of
:py:func:`mask_t`.


.. code-block::

    assert dr.mask_t(dr.scalar.Array3f) is dr.scalar.Array3b
    assert dr.mask_t(dr.cuda.Array3f) is dr.cuda.Array3b
    assert dr.mask_t(dr.cuda.Matrix4f) is dr.cuda.Array44b
    assert dr.mask_t(bool) is bool
    assert dr.mask_t("test") is bool

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Returns the mask type associated with the input or ``bool`` when the
    input is not a Dr.Jit array.
)";

static const char *doc_scalar_t = R"(
Return the *scalar type* associated with the provided Dr.Jit array or type
(i.e., the representation of elements at the lowest level)

When the input is not a Dr.Jit array or type, the function returns its input
unchanged. The following assertions illustrate the behavior of
:py:func:`scalar_t`.


.. code-block::

    assert dr.scalar_t(dr.scalar.Array3f) is bool
    assert dr.scalar_t(dr.cuda.Array3f) is float
    assert dr.scalar_t(dr.cuda.Matrix4f) is float
    assert dr.scalar_t(str) is str
    assert dr.scalar_t("test") is str

Args:
    arg (object): An arbitrary Python object

Returns:
    int: Returns the scalar type of the provided Dr.Jit array, or the type of
    the input.
)";

static const char *doc_is_mask_v = R"(
Check whether the input array instance or type is a Dr.Jit mask array or a
Python ``bool`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents a Dr.Jit mask array or Python ``bool``
    instance or type.
)";

static const char *doc_is_integral_v = R"(
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
Check whether the input array instance or type is a Dr.Jit floating point array
or a Python ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents a Dr.Jit floating point array or
    Python ``float`` instance or type.
)";


static const char *doc_is_arithmetic_v = R"(
Check whether the input array instance or type is an arithmetic Dr.Jit array
or a Python ``int`` or ``float`` value/type.

Note that a mask type (e.g. ``bool``, :py:class:`drjit.scalar.Array2b`, etc.)
is *not* considered to be arithmetic.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an arithmetic Dr.Jit array or Python
    ``int`` or ``float`` instance or type.
)";


static const char *doc_is_signed_v = R"(
Check whether the input array instance or type is an signed Dr.Jit array
or a Python ``int`` or ``float`` value/type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an signed Dr.Jit array or Python
    ``int`` or ``float`` instance or type.
)";


static const char *doc_is_unsigned_v = R"(
Check whether the input array instance or type is an unsigned integer Dr.Jit
array or a Python ``bool`` value/type (masks and boolean values are also
considered to be unsigned).

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an unsigned Dr.Jit array or Python
    ``bool`` instance or type.
)";

static const char *doc_is_jit_v = R"(
Check whether the input array instance or type represents a type that
undergoes just-in-time compilation.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if ``arg`` represents an array type from the
    ``drjit.cuda.*`` or ``drjit.llvm.*`` namespaces, and ``False`` otherwise.
)";

static const char *doc_is_dynamic_v = R"(
Check whether the input instance or type represents a dynamically sized Dr.Jit
array type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
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

static const char *doc_backend_v = R"(
Returns the backend responsible for the given Dr.Jit array instance or type.

Args:
    arg (object): An arbitrary Python object

Returns:
    drjit.JitBackend: The associated Jit backend or ``drjit.JitBackend.None``.)";

static const char *doc_type_v = R"(
Returns the scalar type associated with the given Dr.Jit array instance or
type.

Args:
    arg (object): An arbitrary Python object

Returns:
    drjit.VarType: The associated type ``drjit.VarType.Void``.)";

static const char *doc_replace_type_t = R"(
Converts the provided Dr.Jit array/tensor type into an analogous version with
the specified scalar type.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* `arg0`, it returns an analogous
   version with a different scalar type, as specified via `arg1`. For example,
   when called with :py:class:`drjit.cuda.Array3u` and and
   :py:attr:`drjit.VarType.Float32`, it will return
   :py:class:`drjit.cuda.Array3f`.

2. When the input is not a type, it returns ``replace_type_t(type(arg0), arg1)``.

3. When the input is not a Dr.Jit type, the function returns `arg0`.

Args:
    arg0 (object): An arbitrary Python object

    arg1 (drjit.VarType): The desired variable type

Returns:
    type: Result of the conversion as described above.)";


static const char *doc_is_complex_v = R"(
Check whether the input is a Dr.Jit array instance or type representing a complex number.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_quaternion_v = R"(
Check whether the input is a Dr.Jit array instance or type representing a quaternion.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_vector_v = R"(
Check whether the input is a Dr.Jit array instance or type representing a vectorial array type.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";


static const char *doc_is_matrix_v = R"(
Check whether the input is a Dr.Jit array instance or type representing a matrix.

Args:
    arg (object): An arbitrary Python object

Returns:
    bool: ``True`` if the test was successful, and ``False`` otherwise.
)";

static const char *doc_is_tensor_v = R"(
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
select(condition, x, y, /)
Select elements from inputs based on a condition

This function implements the component-wise operation

.. math::

   \mathrm{result}_i = \begin{cases}
       x_i,\quad&\text{if condition}_i,\\
       y_i,\quad&\text{otherwise.}
   \end{cases}

Args:
    condition (bool | drjit.ArrayBase): A Python or Dr.Jit mask type
    x (int | float | drjit.ArrayBase): A Python or Dr.Jit type
    y (int | float | drjit.ArrayBase): A Python or Dr.Jit type

Returns:
    float | int | drjit.ArrayBase: Component-wise result of the selection operation)";

static const char *doc_abs = R"(
abs(arg, /)
Compute the absolute value of the provided input.

Args:
    arg (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    int | float | drjit.ArrayBase: Absolute value of the input)";

static const char *doc_maximum = R"(
maximum(arg0, arg1, /)
Compute the element-wise maximum value of the provided inputs.

This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
according to the usual implicit type conversion rules).

Args:
    arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    int | float | drjit.ArrayBase: Maximum of the input(s))";

static const char *doc_minimum = R"(
minimum(arg0, arg1, /)
Compute the element-wise minimum value of the provided inputs.

This function returns a result of the type ``type(arg0 + arg1)`` (i.e.,
according to the usual implicit type conversion rules).

Args:
    arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
    arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    int | float | drjit.ArrayBase: Minimum of the input(s))";


static const char *doc_min = R"(
Compute the minimum of the input array or tensor along one or multiple axes.

This function performs a horizontal minimum reduction of the input array,
tensor, or Python sequence along one or multiple axes. By default, it computes
the minimum along the outermost axis; specify ``axis=None`` to process all of
them at once. The minimum of an empty array is considered to be equal to
positive infinity.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (float | int | Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    float | int | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_sqr = R"(
Compute the square of the input array, tensor, or arithmetic type.

Args:
    arg (object): A Python or Dr.Jit arithmetic type

Returns:
    object: The result of the operation ``arg*arg``)";

static const char *doc_pow = R"(
Raise the first argument to a power specified via the second argument.

The function accepts Python arithmetic types, Dr.Jit arrays, and tensors. It
processes each input component separately. When `arg1` is a Python `int` or
integral `float` value, the function performs a sequence of multiplies. The
general case involves recursive use of the identity ``pow(x, y) = exp2(log2(x)
* y)``. There is no difference betweeen using :py:func:`drjit.power()` and the
* builtin ``**`` Python operator.

Args:
    arg (object): A Python or Dr.Jit arithmetic type

Returns:
    object: The result of the operation ``arg0**arg1``)";


static const char *doc_max = R"(
Compute the maximum of the input array or tensor along one or multiple axes.

This function performs a horizontal maximum reduction of the input array,
tensor, or Python sequence along one or multiple axes. By default, it computes
the maximum along the outermost axis; specify ``axis=None`` to process all of
them at once. The maximum of an empty array is considered to be equal to
positive infinity.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (float | int | Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    float | int | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_sum = R"(
Compute the sum of the input array or tensor along one or multiple axes.

This function performs a horizontal sum reduction by adding values of the input
array, tensor, or Python sequence along one or multiple axes. By default, it
sums along the outermost axis; specify ``axis=None`` to sum over all of them at
once. The horizontal sum of an empty array is considered to be zero.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (float | int | Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    float | int | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_prod = R"(
Compute the product of the input array or tensor along one or multiple axes.

This function performs horizontal product reduction by multiplying values of
the input array, tensor, or Python sequence along one or multiple axes. By
default, it multiplies along the outermost axis; specify ``axis=None`` to
process all of them at once. The horizontal product of an empty array is
considered to be equal to one.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (float | int | Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    float | int | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_all = R"(
Compute an ``&`` (AND) reduction of the input array or tensor along one
or multiple axes.

This type of reduction only applies to masks and is typically used to determine
whether all array elements evaluate to ``True``.

This function performs horizontal reduction by combining the input
array, tensor, or Python sequence entries using the ``&`` operator along one or
multiple axes. By default, it reduces along the outermost axis; specify
``axis=None`` to reduce over all of them at once. The reduced form of an empty
array is considered to be ``True``.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (bool | Sequence | drjit.ArrayBase): A Python or Dr.Jit mask type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    bool | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_any = R"(
Compute an ``|`` (OR) reduction of the input array or tensor along one
or multiple axes.

This type of reduction only applies to masks and is typically used to determine
whether at least one array element evaluates to ``True``.

This function performs a horizontal reduction by combining the input array,
tensor, or Python sequence entries using the ``|`` operator along one or
multiple axes. By default, it reduces along the outermost axis; specify
``axis=None`` to reduce over all of them at once. The reduced form of an empty
array is considered to be ``False``.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
important general information about their properties.

Args:
    value (bool | Sequence | drjit.ArrayBase): A Python or Dr.Jit mask type

    axis (int | None): The axis along which to reduce (Default: ``0``). A value
        of ``None`` causes a simultaneous reduction along all axes. Currently, only
        values of ``0`` and ``None`` are supported.

Returns:
    bool | drjit.ArrayBase: Result of the reduction operation)";

static const char *doc_dot = R"(
Compute the dot product of two arrays.

Whenever possible, the implementation uses a sequence of :py:func:`fma` (fused
multiply-add) operations to evaluate the dot product. When the input is a 1D
JIT array like :py:class:`drjit.cuda.Float`, the function evaluates the product
of the input arrays via :py:func:`drjit.eval` and then performs a sum reduction
via :py:func:`drjit.sum`.

See the section on :ref:`horizontal reductions <horizontal-reductions>` for
details on the properties of such horizontal reductions.

Args:
    arg0 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    arg1 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    float | int | drjit.ArrayBase: Dot product of inputs)";


static const char *doc_norm = R"(
Computes the 2-norm of a Dr.Jit array, tensor, or Python sequence.

The operation is equivalent to

.. code-block:: pycon

   dr.sqrt(dr.dot(arg, arg))

The :py:func:`dot` operation performs a horizontal reduction. Please see the
section on :ref:`horizontal reductions <horizontal-reductions>` for details on
their properties.

Args:
    arg (Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

Returns:
    float | int | drjit.ArrayBase: 2-norm of the input)";


static const char *doc_prefix_sum = R"(
Compute an exclusive or inclusive prefix sum of the input array.

By default, the function returns an output array :math:`\mathbf{y}` of the
same size as the input :math:`\mathbf{x}`, where

.. math::

   y_i = \sum_{j=0}^{i-1} x_j.

which is known as an *exclusive* prefix sum, as each element of the output
array excludes the corresponding input in its sum. When the ``exclusive``
argument is set to ``False``, the function instead returns an *inclusive*
prefix sum defined as

.. math::

   y_i = \sum_{j=0}^i x_j.

There is also a convenience alias :py:func:`drjit.cumsum` that computes an
inclusive sum analogous to various other nd-array frameworks.

Not all numeric data types are supported by :py:func:`prefix_sum`:
presently, the function accepts ``Int32``, ``UInt32``, ``UInt64``,
``Float32``, and ``Float64``-typed arrays.

The CUDA backend implementation for "large" numeric types (``Float64``,
``UInt64``) has the following technical limitation: when reducing 64-bit
integers, their values must be smaller than :math:`2^{62}`. When reducing
double precision arrays, the two least significant mantissa bits are clamped to
zero when forwarding the prefix from one 512-wide block to the next (at a *very
minor*, probably negligible loss in accuracy). See the implementation for
details on the rationale of this limitation.

Args:
    value (drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    exclusive (bool): Specifies whether or not the prefix sum should
      be exclusive (the default) or inclusive.

Returns:
    drjit.ArrayBase: An array of the same type containing the computed prefix sum.)";

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

static const char *doc_erf = R"(
erf(arg, /)
Error function approximation.

See the section on :ref:`transcendental function approximations
<transcendental-accuracy>` for details regarding accuracy.

Args:
    arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

Returns:
    float | drjit.ArrayBase: Sine of the input)";

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

- A :ref:`Pytree <pytrees>`. In this case, :py:func:`drjit.zeros()` will invoke
  itself recursively to zero-initialize each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
initialized to ``False`` as opposed to zero.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`Pytree <pytrees>`.
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

- A :ref:`Pytree <pytrees>`. In this case, :py:func:`drjit.ones()` will invoke
  itself recursively to initialize each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
initialized to ``True`` as opposed to one.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`Pytree <pytrees>`.
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

- A :ref:`Pytree <pytrees>`. In this case, :py:func:`drjit.full()` will invoke
  itself recursively to initialize each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case.

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`Pytree <pytrees>`.
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

- A :ref:`Pytree <pytrees>`. In this case, :py:func:`drjit.empty()` will invoke
  itself recursively to allocate memory for each field of the data structure.

- A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
  parameter is ignored in this case, and the function returns a
  zero-initialized result (there is little point in instantiating uninitialized
  versions of scalar Python types).

Args:
    dtype (type): Desired Dr.Jit array type, Python scalar type, or
      :ref:`Pytree <pytrees>`.
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
Return a tuple describing dimension and shape of the provided Dr.Jit array,
tensor, or standard sequence type.

When the arrays is ragged, the implementation signals a failure by returning
``None``. A ragged array has entries of incompatible size, e.g. ``[[1, 2], [3,
4, 5]]``. Note that scalar entries (e.g. ``[[1, 2], [3]]``) are acceptable,
since broadcasting can effectively convert them to any size.

The expressions ``drjit.shape(arg)`` and ``arg.shape`` are equivalent.

Args:
    arg (drjit.ArrayBase | Sequence): an arbitrary Dr.Jit array or tensor

Returns:
    tuple | NoneType: A tuple describing the dimension and shape of the
    provided Dr.Jit input array or tensor. When the input array is *ragged*
    (i.e., when it contains components with mismatched sizes), the function
    returns ``None``.
)";


static const char *doc_ArrayBase_x = R"(
If ``value`` is a static Dr.Jit array of size 1 (or larger), the property
``value.x`` can be used synonymously with ``value[0]``. Otherwise, accessing
this field will generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_y = R"(
If ``value`` is a static Dr.Jit array of size 2 (or larger), the property
``value.y`` can be used synonymously with ``value[1]``. Otherwise, accessing
this field will generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_z = R"(
If ``value`` is a static Dr.Jit array of size 3 (or larger), the property
``value.z`` can be used synonymously with ``value[2]``. Otherwise, accessing
this field will generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_w = R"(
If ``value`` is a static Dr.Jit array of size 4 (or larger), the property
``value.w`` can be used synonymously with ``value[3]``. Otherwise, accessing
this field will generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_real = R"(
If ``value`` is a complex Dr.Jit array, the property ``value.real`` returns the
real component (as does ``value[0]``). Otherwise, accessing this field will
generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_imag = R"(
If ``value`` is a complex Dr.Jit array, the property ``value.imag`` returns the
imaginary component (as does ``value[1]``). Otherwise, accessing this field will
generate a ``RuntimeError``.

:type: :py:func:`value_t(self) <value_t>`)";

static const char *doc_ArrayBase_shape = R"(
This property provides a tuple describing dimension and shape of the
provided Dr.Jit array or tensor. When the input array is *ragged*
(i.e., when it contains components with mismatched sizes), the
property equals ``None``.

The expressions ``drjit.shape(arg)`` and ``arg.shape`` are equivalent.

:type: tuple | NoneType)";

static const char *doc_ArrayBase_ndim = R"(
This property represents the dimension of the provided Dr.Jit array or tensor.

:type: int)";

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

static const char *doc_uint_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a *unsigned integer*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3f64`), it
   returns an *unsigned integer* version (e.g. :py:class:`drjit.cuda.Array3u64`).

2. When the input is not a type, it returns ``uint_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``int_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``int``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";


static const char *doc_float_array_t = R"(
float_array_t(arg, /)
Converts the provided Dr.Jit array/tensor type into a *floating point*
version with the same element size.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u64`), it
   returns an *floating point* version (e.g. :py:class:`drjit.cuda.Array3f64`).

2. When the input is not a type, it returns ``float_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``uint32_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``int32_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``uint64_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``int64_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``float32_array_t(type(arg))``.

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

2. When the input is not a type, it returns ``float64_array_t(type(arg))``.

3. When the input is not a Dr.Jit array or type, the function returns ``float``.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_reinterpret_array_t = R"(
Converts the provided Dr.Jit array/tensor type into a
version with the same element size and scalar type `type`.

This function implements the following set of behaviors:

1. When invoked with a Dr.Jit array *type* (e.g.
:py:class:`drjit.cuda.Array3f64`), it returns a matching array type with the
specified scalar type (e.g., :py:class:`drjit.cuda.Array3u64` when `arg1` is set
to `drjit.VarType.UInt64`).

2. When the input is not a type, it returns ``reinterpret_array_t(type(arg0), arg1)``.

3. When the input is not a Dr.Jit array or type, the function returns the
   associated Python scalar type.

Args:
    arg0 (object): An arbitrary Python object
    arg1 (drjit.VarType): The desired scalar type

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_detached_t = R"(
Converts the provided Dr.Jit array/tensor type into an non-differentiable
version.

This function implements the following set of behaviors:

1. When invoked with a differentiable Dr.Jit array *type* (e.g.
   :py:class:`drjit.cuda.ad.Array3f`), it returns a non-differentiable version
   (e.g. :py:class:`drjit.cuda.Array3f`).

2. When the input is not a type, it returns ``detached_t(type(arg))``.

3. When the input type is non-differentiable or not a Dr.Jit array type, the
   function returns it unchanged.

Args:
    arg (object): An arbitrary Python object

Returns:
    type: Result of the conversion as described above.
)";

static const char *doc_expr_t = R"(
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

It must be called with the desired output ``dtype``, which must be a
dynamically sized 1D array of 32-bit integers. The ``shape`` parameter
specifies the dimensions of a hypothetical input tensor, and ``indices``
contains the entries that would appear in a complex slicing operation, but as a
tuple. For example, ``[5:10:2, ..., None]`` would be specified as ``(slice(5,
10, 2), Ellipsis, None)``.

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

2. When ``dtype`` is a more complex type (e.g. a nested Dr.Jit array or :ref:`Pytree
   <pytrees>`), the behavior depends:

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
            dr.gather(dr.cuda.Float, source.z, index)
        )

     A similar recursive traversal is used for other kinds of
     sequences, mappings, and custom data structures.

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

    permute (bool): You can leave this flag at its default value (``False``).
      It exists to slightly improve the efficiency of a special case where an
      array is fully read by differentiable gathers without duplicate reads
      from any particular entry. (i.e., the gather indices are a permutation).
      This case arises in the implementation of wavefront-style virtual funtion
      dispatch.
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

2. When ``target`` is a more complex type (e.g. a nested Dr.Jit array or
   :ref:`Pytree <pytrees>`), the behavior depends:

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

     A similar recursive traversal is used for other kinds of
     sequences, mappings, and custom data structures.

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

    permute (bool): You can leave this flag at its default value (``False``).
      It exists to slightly improve the efficiency of a special case where a
      zero-initialized array is fully initialized by differentiable scatters
      without duplicate writes to an entry. (i.e., the scatter indices are a
      permutation). This case arises in the implementation of wavefront-style
      virtual funtion dispatch.
)";

static const char *doc_scatter_reduce = R"(
Atomically update values in a flat array or nested data structure.

This operation performs a *scatter-reduction* (i.e., an atomic
read-modify-write operation) using the ``value`` parameter to update the
``target`` array at position ``index``. The optional ``active`` argument can be
used to disable some of the individual RMW operations, which is useful when not
all provided values or indices are valid.

This operation can be used in the following different ways:

1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
   this operation implements a parallelized version of the Python array
   indexing expression ``target[index] = op(target[index], value)`` with
   optional masking. Example:

   .. code-block::

      target = dr.zeros(dr.cuda.Float, 1024*1024)
      value = dr.cuda.Float([...])
      index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
      dr.scatter_reduce(dr.ReduceOp.Add, target, value=value, index=index)

2. When ``target`` is a more complex type (e.g. a nested Dr.Jit array or
   :ref:`Pytree <pytrees>`), the behavior depends:

   - When ``target`` and ``value`` are of the same type, the scatter-reduction
     threads through entries and invokes itself recursively. For example, the
     scatter operation in

     .. code-block::

        op = dr.ReduceOp.Add
        target = dr.cuda.Array3f(...)
        value = dr.cuda.Array3f(...)
        index = dr.cuda.UInt([...])
        dr.scatter_reduce(op, target, value, index)

     is equivalent to

     .. code-block::

        dr.scatter_reduce(op, target.x, value.x, index)
        dr.scatter_reduce(op, target.y, value.y, index)
        dr.scatter_reduce(op, target.z, value.z, index)

     A similar recursive traversal is used for other kinds of
     sequences, mappings, and custom data structures.

   - Otherwise, the operation flattens the ``value`` array and writes it using
     C-style ordering with a suitably modified ``index``. For example, the
     scatter-reduction below writes 3D vectors into a 1D array.

     .. code-block::

        op = dr.ReduceOp.Add
        target = dr.cuda.Float(...)
        value = dr.cuda.Array3f(...)
        index = dr.cuda.UInt([...])
        dr.scatter_reduce(op, target, value, index)

     and is equivalent to

     .. code-block::

        dr.scatter_reduce(op, target, value.x, index*3 + 0)
        dr.scatter_reduce(op, target, value.y, index*3 + 1)
        dr.scatter_reduce(op, target, value.z, index*3 + 2)

.. warning::

   Various combinations of parameters are not supported or are
   backend-dependent:

   - Multiplicative reductions (:py:attr:`drjit.ReduceOp.Mul`) are not
     supported.

   - Mask/boolean array `target` values are not supported.

   - Bitwise reductions (:py:attr:`drjit.ReduceOp.And`,
     :py:attr:`drjit.ReduceOp.Or`) do not support floating point
     operands.

   - On the LLVM backend, some reductions may require newer versions of
     the LLVM library (v15 or newer).

   - On the CUDA backend, min/max reductions (:py:attr:`drjit.ReduceOp.Min`,
     :py:attr:`drjit.ReduceOp.Max`) do not support floating point
     operands.

.. danger::

    The indices provided to this operation are unchecked. Out-of-bounds writes
    are undefined behavior (if not disabled via the ``active`` parameter) and may
    crash the application. Negative indices are not permitted.

    Dr.Jit makes no guarantees about the relative ordering of atomic operations
    when a :py:func:`drjit.scatter_reduce()` writes to the same element
    multiple times. Combined with the non-associate nature of floating point
    operations, concurrent writes will generally introduce nondeterministic
    rounding error.

Args:
    reduce_op (drjit.ReduceOp): Specifies the type of update that should be performed.

    target (object): The object into which data should be written (typically a
      1D Dr.Jit array, but other variations are possible as well, see the
      description above.)

    value (object): The values to be used in the RMW operation (typically of
      type ``type(target)``, but other variations are possible as well, see
      the description above.) Dr.Jit will attempt an implicit conversion if the
      the input is not an array type.

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

Internally, `ravel` performs a series of calls to :py:func:`drjit.scatter()`
to suitably reorganize the array contents.

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

Internally, `unravel` performs a series of calls to :py:func:`drjit.gather()`
to suitably reorganize the array contents.

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

This function accepts a variable-length keyword argument and processes all
input arguments. It recursively traverses Pytrees :ref:`Pytrees <pytrees>`
(sequences, mappings, custom data structures, etc.).

During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
arrays and incompatible types are ignored. Multiple variables can be
equivalently scheduled with a single :py:func:`drjit.schedule()` call or a
sequence of calls to :py:func:`drjit.schedule()`. Variables that are garbage
collected between the original :py:func:`drjit.schedule()` call and the next
kernel launch are ignored and will not be stored in memory.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances or
         :ref:`Pytrees <pytrees>` (they will be recursively traversed to
         all differentiable variables.)

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

This function accepts a variable-length keyword argument and processes all
input arguments. It recursively traverses Pytrees :ref:`Pytrees <pytrees>`
(sequences, mappings, custom data structures, etc.).

During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
arrays and incompatible types are ignored.

Args:
    *args (tuple): A variable-length list of Dr.Jit array instances or
        :ref:`Pytrees <pytrees>` (they will be recursively traversed to discover
        all Dr.Jit arrays.)

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

static const char *doc_detach = R"(
Transforms the input variable into its non-differentiable version (*detaches* it
from the AD computational graph).

This function supports arbitrary Dr.Jit arrays/tensors and :ref:`Pytrees
<pytrees>` as input. In the latter case, it applies the transformation
recursively. When the input variable is not a Pytree or Dr.Jit array, it is
returned as it is.

While the type of the returned array is preserved by default, it is possible to
set the ``preserve_type`` argument to false to force the returned type to be
non-differentiable. For example, this will convert an array of type
:py:class:`drjit.llvm.ad.Float` into one of type :py:class:`drjit.llvm.Float`.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    preserve_type (bool): Defines whether the returned variable should preserve
        the type of the input variable.
Returns:
    object: The detached variable.
)";

static const char *doc_set_grad_enabled = R"(
Enable or disable gradient tracking on the provided variables.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor,
        :ref:`Pytree <pytrees>`, sequence, or mapping.

    value (bool): Defines whether gradient tracking should be enabled or
        disabled.
)";

static const char *doc_enable_grad = R"(
Enable gradient tracking for the provided variables.

This function accepts a variable-length keyword argument and processes all
input arguments. It recursively traverses Pytrees :ref:`Pytrees <pytrees>`
(sequences, mappings, custom data structures, etc.).

During this recursive traversal, the function enables gradient tracking for all
encountered Dr.Jit arrays. Variables of other types are ignored.

Args:
    *args (tuple): A variable-length list of Dr.Jit arrays/tensors or
        :ref:`Pytrees <pytrees>`.
)";

static const char *doc_disable_grad = R"(
Disable gradient tracking for the provided variables.

This function accepts a variable-length keyword argument and processes all
input arguments. It recursively traverses Pytrees :ref:`Pytrees <pytrees>`
(sequences, mappings, custom data structures, etc.).

During this recursive traversal, the function disables gradient tracking for all
encountered Dr.Jit arrays. Variables of other types are ignored.

Args:
    *args (tuple): A variable-length list of Dr.Jit arrays/tensors or
        :ref:`Pytrees <pytrees>`.
)";

static const char *doc_grad_enabled = R"(
Return whether gradient tracking is enabled on any of the given variables.

Args:
    *args (tuple): A variable-length list of Dr.Jit arrays/tensors instances or
      :ref:`Pytrees <pytrees>`. The function recursively traverses them to
      all differentiable variables.

Returns:
    bool: ``True`` if any of the input variables has gradient tracking enabled,
    ``False`` otherwise.
)";

static const char *doc_grad = R"(
Return the gradient value associated to a given variable.

When the variable doesn't have gradient tracking enabled, this function returns ``0``.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor or :ref:`Pytree <pytrees>`.

    preserve_type (bool): Should the operation preserve the input type in the
        return value? (This is the default). Otherwise, Dr.Jit will, e.g.,
        return a type of `drjit.cuda.Float` for an input of type
        `drjit.cuda.ad.Float`.

Returns:
    object: the gradient value associated to the input variable.
)";

static const char *doc_set_grad = R"(
Set the gradient associated with the provided variable.

This operation internally decomposes into two sub-steps:

.. code-block:: python

   dr.clear_grad(target)
   dr.accum_grad(target, source)

When `source` is not of the same type as `target`, Dr.Jit will try to broadcast
its contents into the right shape.

Args:
    target (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    source (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.
)";


static const char *doc_accum_grad = R"(
Accumulate the contents of one variable into the gradient of another variable.

When `source` is not of the same type as `target`, Dr.Jit will try to broadcast
its contents into the right shape.

Args:
    target (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    source (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.
)";

static const char *doc_clear_grad = R"(
Clear the gradient of the given variable.

Args:
    arg (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.
)";

static const char *doc_replace_grad = R"(
Replace the gradient value of ``arg0`` with the one of ``arg1``.

This is a relatively specialized operation to be used with care when
implementing advanced automatic differentiation-related features.

One example use would be to inform Dr.Jit that there is a better way to compute
the gradient of a particular expression than what the normal AD traversal of
the primal computation graph would yield.

The function promotes and broadcasts `arg0` and `arg1` if they are not of the
same type.

Args:
    arg0 (object): An arbitrary Dr.Jit array, tensor, Python arithmetic type, or :ref:`Pytree <pytrees>`.

    arg1 (object): An arbitrary Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

Returns:
    object: a new Dr.Jit array combining the *primal* value of `arg0` and the
    derivative of `arg1`.
)";

static const char *doc_enqueue = R"(
Enqueues the input variable(s) for subsequent gradient propagation

Dr.Jit splits the process of automatic differentiation into three parts:

1. Initializing the gradient of one or more input or output variables. The most
   common initialization entails setting the gradient of an output (e.g., an
   optimization loss) to ``1.0``.

2. Enqueuing nodes that should partake in the gradient propagation pass. Dr.Jit
   will follow variable dependences (edges in the AD graph) to find variables
   that are reachable from the enqueued variable.

3. Finally propagating gradients to all of the enqueued variables.

This function is responsible for step 2 of the above list and works differently
depending on the specified ``mode``:

-:py:attr:`drjit.ADMode.Forward`: Dr.Jit will recursively enqueue all variables that are
   reachable along forward edges. That is, given a differentiable operation ``a =
   b+c``, enqueuing ``c`` will also enqueue ``a`` for later traversal.

-:py:attr:`drjit.ADMode.Backward`: Dr.Jit will recursively enqueue all variables that are
  reachable along backward edges. That is, given a differentiable operation ``a =
  b+c``, enqueuing ``a`` will also enqueue ``b`` and ``c`` for later traversal.

For example, a typical chain of operations to forward propagate the gradients
from ``a`` to ``b`` might look as follow:

.. code-block:: python

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)
    b = f(a) # some computation involving `a`

    # The below three operations can also be written more compactly as dr.forward_from(a)
    dr.set_gradient(a, 1.0)
    dr.enqueue(dr.ADMode.Forward, a)
    dr.traverse(dr.ADMode.Forward)

    grad = dr.grad(b)

One interesting aspect of this design is that enqueuing and traversal don't
necessarily need to follow the same direction.

For example, we may only be interested in forward gradients reaching a specific
output node ``c``, which can be expressed as follows:

.. code-block:: python

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)

    b, c, d, e = f(a)

    dr.set_gradient(a, 1.0)
    dr.enqueue(dr.ADMode.Backward, b)
    dr.traverse(dr.ADMode.Forward)

    grad = dr.grad(b)

The same naturally also works in the reverse directiion. Dr.Jit provides a
higher level API that encapsulate such logic in a few different flavors:

- :py:func:`drjit.forward_from` (alias: :py:func:`drjit.forward`) and
  :py:func:`drjit.forward_to`.
- :py:func:`drjit.backward_from` (alias: :py:func:`drjit.backward`) and
  :py:func:`drjit.backward_to`.

Args:
    mode (drjit.ADMode): Specifies the set edges which Dr.Jit should follow to
       enqueue variables to be visited by a later gradient propagation phase.
      :py:attr:`drjit.ADMode.Forward` and:py:attr:`drjit.ADMode.Backward` refer to forward and
       backward edges, respectively.

    value (object): An arbitrary Dr.Jit array, tensor or
        :ref:`Pytree <pytrees>`.
)";

static const char *doc_traverse = R"(
Propagate gradients along the enqueued set of AD graph edges.

Given prior use of :py:func`drjit.enqueue()` to enqueue AD nodes for gradient
propagation, this functions now performs the actual gradient propagation into
either the forward or reverse direction (as specified by the ``mode``
parameter)

By default, the operation is destructive: it clears the gradients of visited
interior nodes and only retains gradients at leaf nodes. The term *leaf node*
is defined as follows:
refers to

- In forward AD, leaf nodes have no forward edges. They are outputs of a
  computation, and no other differentiable variable depends on them.

- In backward AD, leaf nodes have no backward edges. They are inputs to a
  computation.

By default, the traversal also removes the edges of visited nodes to isolate
them. These defaults are usually good ones: cleaning up the graph his frees up
resources, which is useful when working with large wavefronts or very complex
computation graphs. It also avoids potentially undesired derivative
contributions that can arise when the AD graphs of two unrelated computations
are connected by an edge and subsequently separately differentiated.

In advanced applications that require multiple AD traversals of the same graph,
specify specify different combinations of the enumeration
:py:class:`drjit.ADFlag` via the `flags` parameter.

Args:
    mode (drjit.ADMode): Specifies the direction in which gradients should be
        propgated. :py:attr:`drjit.ADMode.Forward`
        and:py:attr:`drjit.ADMode.Backward` refer to forward and backward
        traversal.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph are cleared
        during traversal. The default value is :py:attr:`drjit.ADFlag.Default`.
)";

static const char *doc_forward_from = R"(
forward_from(arg: drjit.ArrayBase, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Forward-propagate gradients from the provided Dr.Jit array or tensor.

This function sets the gradient of the provided Dr.Jit array or tensor ``arg``
to ``1.0`` and then forward-propagates derivatives through forward-connected
components of the computation graph (i.e., reaching all variables that directly
or indirectly depend on ``arg``).

The operation is equivalent to

.. code-block:: python

   dr.set_grad(arg, 1.0)
   dr.enqueue(dr.ADMode.Forward, h)
   dr.traverse(dr.ADMode.Forward, flags=flags)

Refer to the documentation functions :py:func:`drjit.set_grad()`,
:py:func:`drjit.enqueue()`, and :py:func:`drjit.traverse()` for further
details on the nuances of forward derivative propagation.

By default, the operation is destructive: it clears the gradients of visited
interior nodes and only retains gradients at leaf nodes. For details on this,
refer to the documentation of :py:func:`drjit.enqueue()` and the meaning of
the ``flags`` parameter.

The implementation raises an exception when the provided array does not support
gradient tracking, or when gradient tracking was not previously enabled via
:py:func:`drjit.enable_grad()`, as this generally indicates the presence of
a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoFlag` flag (e.g. by
passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

Args:
    args (object): A Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
        during traversal, and whether or not to fail when the input is not
        differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.
)";

static const char *doc_forward_to = R"(
forward_to(*args, *, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Forward-propagate gradients to the provided set of Dr.Jit arrays/tensors.

.. code-block:: python

   dr.enqueue(dr.ADMode.Backward, *args)
   dr.traverse(dr.ADMode.Forward, flags=flags)
   return dr.grad(*args)

Internally, the operation first traverses the computation graph *backwards*
from ``args`` to find potential paths along which gradients can flow to the
given set of arrays. Then, it performs a gradient propagation pass along the
detected variables.

For this to work, you must have previously enabled and specified input
gradients for inputs of the computation. (see :py:func:`drjit.enable_grad()`
and via :py:func:`drjit.set_grad()`).

Refer to the documentation functions :py:func:`drjit.enqueue()` and
:py:func:`drjit.traverse()` for further details on the nuances of forward
derivative propagation.

By default, the operation is destructive: it clears the gradients of visited
interior nodes and only retains gradients at leaf nodes. For details on this,
refer to the documentation of :py:func:`drjit.enqueue()` and the meaning of
the ``flags`` parameter.

The implementation raises an exception when the provided array does not support
gradient tracking, or when gradient tracking was not previously enabled via
:py:func:`drjit.enable_grad()`, as this generally indicates the presence of
a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoFlag` flag (e.g. by
passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

Args:
    *args (tuple): A variable-length list of Dr.Jit differentiable array, tensors,
        or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
        during traversal, and whether or not to fail when the input is not
        differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

Returns:
    object: the gradient value(s) associated with ``*args`` following the
    traversal.
)";

static const char *doc_forward = R"(
forward(arg: drjit.ArrayBase, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Forward-propagate gradients from the provided Dr.Jit array or tensor

This function is an alias of :py:func:`drjit.forward_from()`. Please refer to
the documentation of this function.

Args:
    args (object): A Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph are cleared
        during traversal. The default value is :py:attr:`drjit.ADFlag.Default`.
)";

static const char *doc_backward_from = R"(
backward_from(arg: drjit.ArrayBase, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Backpropagate gradients from the provided Dr.Jit array or tensor.

This function sets the gradient of the provided Dr.Jit array or tensor ``arg``
to ``1.0`` and then backpropagates derivatives through backward-connected
components of the computation graph (i.e., reaching differentiable variables
that potentially influence the value of ``arg``).

The operation is equivalent to

.. code-block:: python

   dr.set_grad(arg, 1.0)
   dr.enqueue(dr.ADMode.Backward, h)
   dr.traverse(dr.ADMode.Backward, flags=flags)

Refer to the documentation functions :py:func:`drjit.set_grad()`,
:py:func:`drjit.enqueue()`, and :py:func:`drjit.traverse()` for further
details on the nuances of derivative backpropagation.

By default, the operation is destructive: it clears the gradients of visited
interior nodes and only retains gradients at leaf nodes. For details on this,
refer to the documentation of :py:func:`drjit.enqueue()` and the meaning of
the ``flags`` parameter.

The implementation raises an exception when the provided array does not support
gradient tracking, or when gradient tracking was not previously enabled via
:py:func:`drjit.enable_grad()`, as this generally indicates the presence of
a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoFlag` flag (e.g. by
passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

Args:
    args (object): A Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
        during traversal, and whether or not to fail when the input is not
        differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.
)";

static const char *doc_backward_to = R"(
backward_to(*args, *, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Backpropagate gradients to the provided set of Dr.Jit arrays/tensors.

.. code-block:: python

   dr.enqueue(dr.ADMode.Forward, *args)
   dr.traverse(dr.ADMode.Backwards, flags=flags)
   return dr.grad(*args)

Internally, the operation first traverses the computation graph *forwards*
from ``args`` to find potential paths along which reverse-mode gradients can flow to the
given set of input variables. Then, it performs a backpropagation pass along the
detected variables.

For this to work, you must have previously enabled and specified input
gradients for outputs of the computation. (see :py:func:`drjit.enable_grad()`
and via :py:func:`drjit.set_grad()`).

Refer to the documentation functions :py:func:`drjit.enqueue()` and
:py:func:`drjit.traverse()` for further details on the nuances of
derivative backpropagation.

By default, the operation is destructive: it clears the gradients of visited
interior nodes and only retains gradients at leaf nodes. For details on this,
refer to the documentation of :py:func:`drjit.enqueue()` and the meaning of
the ``flags`` parameter.

The implementation raises an exception when the provided array does not support
gradient tracking, or when gradient tracking was not previously enabled via
:py:func:`drjit.enable_grad()`, as this generally indicates the presence of
a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoFlag` flag (e.g. by
passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

Args:
    *args (tuple): A variable-length list of Dr.Jit differentiable array, tensors,
        or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
        during traversal, and whether or not to fail when the input is not
        differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

Returns:
    object: the gradient value(s) associated with ``*args`` following the
    traversal.
)";

static const char *doc_backward = R"(
backward(arg: drjit.ArrayBase, flags: drjit.ADFlag | int = drjit.ADFlag.Default)

Backpropgate gradients from the provided Dr.Jit array or tensor.

This function is an alias of :py:func:`drjit.backward_from()`. Please refer to
the documentation of this function.

Args:
    args (object): A Dr.Jit array, tensor, or :ref:`Pytree <pytrees>`.

    flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
        during traversal, and whether or not to fail when the input is not
        differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.
)";


static const char *doc_graphviz = R"(
Return a GraphViz diagram describing registered JIT variables and their connectivity.

This function returns a representation of the computation graph underlying the
Dr.Jit just-in-time compiler, which is separate from the automatic
differentiation layer. See the :py:func:`graphviz_ad()` function to visualize
the computation graph of the latter.

The function depends on the ``graphviz`` Python package when
``as_string=False`` (the default).

Args:
    as_string (bool): if set to ``True``, the function will return raw GraphViz markup
        as a string. (Default: ``False``)

Returns:
    object: GraphViz object or raw markup.
)";

static const char *doc_graphviz_ad = R"(
Return a GraphViz diagram describing variables registered with the automatic
differentiation layer, as well as their connectivity.

This function returns a representation of the computation graph underlying the
Dr.Jit AD layer, which one architectural layer above the just-in-time compiler.
See the :py:func:`graphviz()` function to visualize the computation graph of
the latter.

The function depends on the ``graphviz`` Python package when
``as_string=False`` (the default).

Args:
    as_string (bool): if set to ``True``, the function will return raw GraphViz markup
        as a string. (Default: ``False``)

Returns:
    object: GraphViz object or raw markup.
)";

static const char *doc_whos = R"(
Return/print a list of live JIT variables.

This function provides information about the set of variables that are
currently registered with the Dr.Jit just-in-time compiler, which is separate
from the automatic differentiation layer. See the :py:func:`whos_ad()` function
for the latter.

Args:
    as_string (bool): if set to ``True``, the function will return the list in
        string form. Otherwise, it will print directly onto the console and return
        ``None``. (Default: ``False``)

Returns:
    NoneType | str: a human-readable list (if requested).
)";

static const char *doc_whos_ad = R"(
Return/print a list of live variables registered with the automatic differentiation layer.

This function provides information about the set of variables that are
currently registered with the Dr.Jit automatic differentiation layer,
which one architectural layer above the just-in-time compiler.
See the :py:func:`whos()` function to obtain informatoina about
the latter.

Args:
    as_string (bool): if set to ``True``, the function will return the list in
        string form. Otherwise, it will print directly onto the console and return
        ``None``. (Default: ``False``)

Returns:
    NoneType | str: a human-readable list (if requested).
)";

static const char *doc_ad_scope_enter = R"(
TODO
)";

static const char *doc_ad_scope_leave = R"(
TODO
)";

static const char *doc_suspend_grad = R"(
suspend_grad(*args, when = True)
Context manager for temporally suspending derivative tracking.

Dr.Jit's AD layer keeps track of a set of variables for which derivative
tracking is currently enabled. Using this context manager is it possible to
define a scope in which variables will be subtracted from that set, thereby
controlling what derivative terms shouldn't be generated in that scope.

The variables to be subtracted from the current set of enabled variables can be
provided as function arguments. If none are provided, the scope defined by this
context manager will temporally disable all derivative tracking.

.. code-block::

    a = dr.llvm.ad.Float(1.0)
    b = dr.llvm.ad.Float(2.0)
    dr.enable_grad(a, b)

    with suspend_grad(): # suspend all derivative tracking
        c = a + b

    assert not dr.grad_enabled(c)

    with suspend_grad(a): # only suspend derivative tracking on `a`
        d = 2.0 * a
        e = 4.0 * b

    assert not dr.grad_enabled(d)
    assert dr.grad_enabled(e)

In a scope where derivative tracking is completely suspended, the AD layer will
ignore any attempt to enable gradient tracking on a variable:

.. code-block:: python

    a = dr.llvm.ad.Float(1.0)

    with suspend_grad():
        dr.enable_grad(a) # <-- ignored
        assert not dr.grad_enabled(a)

    assert not dr.grad_enabled(a)

The optional ``when`` boolean keyword argument can be defined to specifed a
condition determining whether to suspend the tracking of derivatives or not.

.. code-block:: python

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)

    cond = condition()

    with suspend_grad(when=cond):
        b = 4.0 * a

    assert dr.grad_enabled(b) == not cond

Args:
    *args (tuple): A variable-length list of differentiable Dr.Jit array
        instances or :ref:`Pytrees <pytrees>`. The function will recursively
        traverse them to all differentiable variables.

    when (bool): An optional Python boolean determining whether to suspend
      derivative tracking.
)";

static const char *doc_resume_grad = R"(
resume_grad(*args, when = True)
Context manager for temporally resume derivative tracking.

Dr.Jit's AD layer keeps track of a set of variables for which derivative
tracking is currently enabled. Using this context manager is it possible to
define a scope in which variables will be added to that set, thereby controlling
what derivative terms should be generated in that scope.

The variables to be added to the current set of enabled variables can be
provided as function arguments. If none are provided, the scope defined by this
context manager will temporally resume derivative tracking for all variables.

.. code-block::

    a = dr.llvm.ad.Float(1.0)
    b = dr.llvm.ad.Float(2.0)
    dr.enable_grad(a, b)

    with suspend_grad():
        c = a + b

        with resume_grad():
            d = a + b

        with resume_grad(a):
            e = 2.0 * a
            f = 4.0 * b

    assert not dr.grad_enabled(c)
    assert dr.grad_enabled(d)
    assert dr.grad_enabled(e)
    assert not dr.grad_enabled(f)

The optional ``when`` boolean keyword argument can be defined to specifed a
condition determining whether to resume the tracking of derivatives or not.

.. code-block::

    a = dr.llvm.ad.Float(1.0)
    dr.enable_grad(a)

    cond = condition()

    with suspend_grad():
        with resume_grad(when=cond):
            b = 4.0 * a

    assert dr.grad_enabled(b) == cond

Args:
    *args (tuple): A variable-length list of differentiable Dr.Jit array
        instances or :ref:`Pytrees <pytrees>`. The function will recursively
        traverse them to all differentiable variables.

    when (bool): An optional Python boolean determining whether to resume
      derivative tracking.
)";

static const char *doc_isolate_grad = R"(
Context manager to temporarily isolate outside world from AD traversals.

Dr.Jit provides isolation boundaries to postpone AD traversals steps leaving a
specific scope. For instance this function is used internally to implement
differentiable loops and polymorphic calls.
)";
static const char *doc_label = R"(
Returns the label of a given Dr.Jit array.

Args:
    arg (object): a Dr.Jit array instance.

Returns:
    str: the label of the given variable.
)";

static const char *doc_has_backend = R"(
Check if the specified Dr.Jit backend was successfully initialized.)";

static const char *doc_set_label = R"(
Assign a label to the provided Dr.Jit array.

This can be helpful to identify computation in GraphViz output (see
:py:func:`drjit.graphviz`, :py:func:`graphviz_ad`).

The operations assumes that the array is tracked by the just-in-time compiler.
It has no effect on unsupported inputs (e.g., arrays from the ``drjit.scalar``
package). It recurses through :ref:`Pytrees <pytrees>` (tuples, lists,
dictionaries, custom data structures) and appends names (indices, dictionary
keys, field names) separated by underscores to uniquely identify each element.

The following ``**kwargs``-based shorthand notation can be used to assign
multiple labels at once:

.. code-block:: python

   set_label(x=x, y=y)

Args:
    *arg (tuple): a Dr.Jit array instance and its corresponding label ``str`` value.

    **kwarg (dict): A set of (keyword, object) pairs.
)";

static const char *doc_ADMode = R"(
Enumeration to distinguish different types of primal/derivative computation.

See also :py:func:`drjit.enqueue()`, :py:func:`drjit.traverse()`.
)";

static const char *doc_ADMode_Primal = R"(
Primal/original computation without derivative tracking. Note that this
is *not* a valid input to Dr.Jit AD routines, but it is sometimes useful
to have this entry when to indicate to a computation that derivative
propagation should not be performed.)";

static const char *doc_ADMode_Forward = R"(
Propagate derivatives in forward mode (from inputs to outputs))";

static const char *doc_ADMode_Backward = R"(
Propagate derivatives in backward/reverse mode (from outputs to inputs)";

static const char *doc_ADFlag = R"(
By default, Dr.Jit's AD system destructs the enqueued input graph during
forward/backward mode traversal. This frees up resources, which is useful
when working with large wavefronts or very complex computation graphs.
However, this also prevents repeated propagation of gradients through a
shared subgraph that is being differentiated multiple times.

To support more fine-grained use cases that require this, the following
flags can be used to control what should and should not be destructed.
)";

static const char *doc_ADFlag_ClearNone = "Clear nothing.";

static const char *doc_ADFlag_ClearEdges =
    "Delete all traversed edges from the computation graph";

static const char *doc_ADFlag_ClearInput =
    "Clear the gradients of processed input vertices (in-degree == 0)";

static const char *doc_ADFlag_ClearInterior =
    "Clear the gradients of processed interior vertices (out-degree != 0)";

static const char *doc_ADFlag_ClearVertices =
    "Clear gradients of processed vertices only, but leave edges intact. Equal "
    "to ``ClearInput | ClearInterior``.";

static const char *doc_ADFlag_Default =
    "Default: clear everything (edges, gradients of processed vertices). Equal "
    "to ``ClearEdges | ClearVertices``.";

static const char *doc_ADFlag_AllowNoGrad =
    "Don't fail when the input to a ``drjit.forward`` or ``backward`` "
    "operation is not a differentiable array.";

static const char *doc_JitBackend =
    "List of just-in-time compilation backends supported by Dr.Jit. See also "
    ":py:func:`drjit.backend_v()`.";

static const char *doc_JitBackend_None =
    "Indicates that a type is not handled by a Dr.Jit backend (e.g., a scalar type)";

static const char *doc_JitBackend_LLVM =
    "Dr.Jit backend targeting various processors via the LLVM compiler infractructure.";

static const char *doc_JitBackend_CUDA =
    "Dr.Jit backend targeting NVIDIA GPUs using PTX (\"Parallel Thread Excecution\") IR.";

static const char *doc_VarType =
    "List of possible scalar array types (not all of them are supported).";

static const char *doc_VarType_Void = "Unknown/unspecified type.";
static const char *doc_VarType_Bool = "Boolean/mask type.";
static const char *doc_VarType_Int8 = "Signed 8-bit integer.";
static const char *doc_VarType_UInt8 = "Unsigned 8-bit integer.";
static const char *doc_VarType_Int16 = "Signed 16-bit integer.";
static const char *doc_VarType_UInt16 = "Unsigned 16-bit integer.";
static const char *doc_VarType_Int32 = "Signed 32-bit integer.";
static const char *doc_VarType_UInt32 = "Unsigned 32-bit integer.";
static const char *doc_VarType_Int64 = "Signed 64-bit integer.";
static const char *doc_VarType_UInt64 = "Unsigned 64-bit integer.";
static const char *doc_VarType_Pointer = "Pointer to a memory address.";
static const char *doc_VarType_Float16 = "16-bit floating point format (IEEE 754).";
static const char *doc_VarType_Float32 = "32-bit floating point format (IEEE 754).";
static const char *doc_VarType_Float64 = "64-bit floating point format (IEEE 754).";

static const char *doc_ReduceOp =
    "Denotes the type of atomic read-modify-write (RMW) operation (for "
    "scatter-reductions) or aggregation to be performed by a horizontal "
    "reduction.";

static const char *doc_ReduceOp_None =
    "Perform an ordinary scatter operation that ignores the current entry "
    "(only applies to scatter-reductions).";

static const char *doc_ReduceOp_Add = "Addition.";
static const char *doc_ReduceOp_Mul = "Multiplication.";
static const char *doc_ReduceOp_Min = "Minimum.";
static const char *doc_ReduceOp_Max = "Maximum.";
static const char *doc_ReduceOp_And = "Binary AND operation.";
static const char *doc_ReduceOp_Or = "Binary OR operation.";

static const char *doc_CustomOp = R"(
Base class for implementing custom differentiable operations.

Dr.Jit can compute derivatives of builtin operations in both forward and
reverse mode. In some cases, it may be useful or even necessary to control how
a particular operation should be differentiated.

To do so, you may extend this class to provide *three* callback functions:

1. :py:func:`CustomOp.eval()`: Implements the *primal* evaluation of the
   function with detached inputs.

2. :py:func:`CustomOp.forward()`: Implements the *forward derivative* that
   propagates derivatives from input arguments to the return value

3. :py:func:`CustomOp.backward()`: Implements the *backward derivative* that
   propagates derivatives from the return value to the input arguments.

An example for a hypothetical custom addition operation is shown below

.. code-block:: python

    class Addition(dr.CustomOp):
        def eval(self, x, y):
            # Primal calculation without derivative tracking
            return x + y

        def forward(self):
            # Compute forward derivatives
            self.set_grad_out(self.grad_in('x') + self.grad_in('y'))

        def backward(self):
            # .. compute backward derivatives ..
            self.set_grad_in('x', self.grad_out())
            self.set_grad_in('y', self.grad_out())

        def name(self):
            # Optional: a descriptive name shown in GraphViz visualizations
            return "Addition"

You should never need to call these functions yourself---Dr.Jit will do so when
appropriate. To weave such a custom operation into the AD graph, use the
:py:func:`drjit.custom()` function, which expects a subclass of
:py:class:`drjit.CustomOp` as first argument, followed by arguments to the
actual operation that are directly forwarded to the ``.eval()`` callback.

.. code-block:: python

   # Add two numbers `x` and `y`. Calls our ``.eval()`` callback with detached arguments
   result = dr.custom(Addition, x, y)

Forward or backward derivatives are then automatically handled through the
standard operations. For example,

.. code-block:: python

   dr.backward(result)

will invoke the ``.backward()`` callback from above.

Many derivatives are more complex than the above examples and require access to
inputs or intermediate steps of the primal evaluation routine. You can simply
stash them in the instance (``self.field = ...``), which is shown below for a
differentiable multiplication operation that implements the product rule:

.. code-block:: python

    class Multiplication(dr.CustomOp):
        def eval(self, x, y):
            # Stash input arguments
            self.x = x
            self.y = y

            return x * y

        def forward(self):
            self.set_grad_out(self.y * self.grad_in('x') + self.x * self.grad_in('y'))

        def backward(self):
            self.set_grad_in('x', self.y * self.grad_out())
            self.set_grad_in('y', self.x * self.grad_out())

        def name(self):
            return "Multiplication"
)";

static const char *doc_CustomOp_eval = R"(
Evaluate the custom operation in primal mode.

You must implement this method when subclassing :py:class:`CustomOp`, since the
default implementation raises an exception. It should realize the original
(non-derivative-aware) form of a computation and may take an arbitrary sequence
of positional, keyword, and variable-length positional/keword arguments.

You should not need to call this function yourself---Dr.Jit will automatically do so
when performing custom operations through the :py:func:`drjit.custom()` interface.

Note that the input arguments passed to ``.eval()`` will be *detached* (i.e.
they don't have derivative tracking enabled). This is intentional, since
derivative tracking is handled by the custom operation along with the other
callbacks :py:func:`forward` and :py:func:`backward`.)";

static const char *doc_CustomOp_forward = R"(
Evaluate the forward derivative of the custom operation.

You must implement this method when subclassing :py:class:`CustomOp`, since the
default implementation raises an exception. It takes no arguments and has no
return value.

An implementation will generally perform repeated calls to :py:func:`grad_in`
to query the gradients of all function followed by a single call to
:py:func:`set_grad_out` to set the gradient of the return value.

For example, this is how one would implement the product rule of the primal
calculation ``x*y``, assuming that the ``.eval()`` routine stashed the inputs
in the custom operation object.

.. code-block:: python

   def forward(self):
       self.set_grad_out(self.y * self.grad_in('x') + self.x * self.grad_in('y'))

)";

static const char *doc_CustomOp_backward = R"(
Evaluate the backward derivative of the custom operation.

You must implement this method when subclassing :py:class:`CustomOp`, since the
default implementation raises an exception. It takes no arguments and has no
return value.

An implementation will generally perform a single call to :py:func:`grad_out`
to query the gradient of the function return value followed by a sequence of calls to
:py:func:`set_grad_in` to assign the gradients of the function inputs.

For example, this is how one would implement the product rule of the primal
calculation ``x*y``, assuming that the ``.eval()`` routine stashed the inputs
in the custom operation object.

.. code-block:: python

   def backward(self):
       self.set_grad_in('x', self.y * self.grad_out())
       self.set_grad_in('y', self.x * self.grad_out())
)";

static const char *doc_CustomOp_grad_out = R"(
Query the gradient of the return value.

Returns an object, whose type matches the original return value produced in
:py:func:`eval()`. This function should only be used within the
:py:func:`backward()` callback.)";

static const char *doc_CustomOp_set_grad_out = R"(
Accumulate a gradient into the return value.

This function should only be used within the :py:func:`forward()` callback.)";

static const char *doc_CustomOp_grad_in = R"(
Query the gradient of a specified input parameter.

The second argument specifies the parameter name as string. Gradients of
variable-length positional arguments (``*args``) can be queried by providing an
integer index instead.

This function should only be used within the :py:func:`forward()` callback.)";

static const char *doc_CustomOp_set_grad_in = R"(
Accumulate a gradient into the specified input parameter.

The second argument specifies the parameter name as string. Gradients of
variable-length positional arguments (``*args``) can be assigned by providing
an integer index instead.

This function should only be used within the :py:func:`backward()` callback.)";

static const char *doc_CustomOp_add_input = R"(
Register an implicit input dependency of the operation on an AD variable.

This function should be called by the :py:func:`eval()` implementation when an
operation has a differentiable dependence on an input that is not a ordinary
input argument of the function (e.g., a global program variable or a field of a
class).)";

static const char *doc_CustomOp_add_output = R"(
Register an implicit output dependency of the operation on an AD variable.

This function should be called by the :py:func:`eval()` implementation when an
operation has a differentiable dependence on an output that is not part of the
function return value (e.g., a global program variable or a field of a
class).")";

static const char *doc_CustomOp_name = R"(
Return a descriptive name of the ``CustomOp`` instance.

Amongst other things, this name is used to document the presence of the
custom operation in GraphViz debug output. (See :py:func:`graphviz_ad`.))";

static const char *doc_custom = R"(
Evaluate a custom differentiable operation.

It can be useful or even necessary to control how a particular operation should
be differentiated by Dr.Jit's automatic differentiation (AD) layer. The
:py:func:`drjit.custom` function  enables such use cases by stitching an opque
operation with user-defined primal and forward/backward derivative
implementations into the AD graph.

The function expects a subclass of the :py:class:`CustomOp` interface as first
argument. The remaining positional and keyword arguments are forwarded to the
:py:func:`CustomOp.eval` callback.

See the documentation of :py:class:`CustomOp` for examples on how to realize
such a custom operation.
)";

static const char *doc_switch = R"(
switch(index: int | drjit.ArrayBase, callables: Sequence[Callable], *args, **kwargs) -> object

Invoke one of multiple callables based on an index or an array of indices.

When provided with a *scalar* index (``type(index)`` is ``int``), this function
invokes one of the provided callables and is semantically equivalent to the
following Python code:

.. code-block:: python

   def switch(index: int, callables, *args, **kwargs):
       return callables[index](*args, **kwargs)

When provided with a Dr.Jit array of indices (32-bit unsigned integers), it
performs the vectorized equivalent of the above and assembles an array of
return values containing the result of all referenced callables.

.. code-block:: python

    import drjit as dr
    from drjit.llvm import UInt32

    def f1(x):
        return x

    def f2(x):
        return x*10

    index = UInt32(0, 1, 1, 0)
    res = dr.switch(index, [f1, f2], dr.arange(UInt32, 4))

    # res now contains [0, 10, 20, 3]

When a boolean Dr.Jit array (e.g., :py:class:`drjit.llvm.Bool`,
:py:class:`drjit.cuda.ad.Bool`, etc.) is specified as last positional argument
or as a keyword argument named ``active``, that argument is treated specially
and *not* forwarded to the provided callables. Entries of the input arrays
associated with a ``False`` mask entry are ignored and never passed to the
callables. The corresponding return value entries remain zero-initialized.

Dr.Jit will use one of two possible strategies to realize this operation
depending on the active compilation flags (see :py:func:`drjit.set_flag`,
:py:func:`drjit.scoped_set_flag`):

1. **Recorded mode**: When :py:attr:`drjit.JitFlag.VCallRecord` is set (the
   default), Dr.Jit transcribes every callable into an equivalent function in the
   generated low-level intermediate representation (LLVM IR or PTX) and targets
   them via an indirect jump instruction.

   One caveat with this approach is that Dr.Jit does not know the specific
   inputs reaching each callable at trace time. This knowledge will only become
   available later on when the generated code runs on the device (e.g., the
   GPU). Thus, callables receive *symbolic* input arrays that merely help to
   transcribe their implementation into low-level IR. Some operations involving
   such symbolic inputs are not valid and will fail:

   .. code-block:: python

      from drjit.llvm import Array3f, Float, UInt32

      # A callable 'f1' called by dr.switch()
      def f1(x: dr.llvm.Array3f):
          print(x)        # <-- fails
          y: Float = x[0] # <-- OK
          z: float = y[0] # <-- fails

   The common pattern of the failing operations is that they require variable
   evaluation (i.e., :py:func:`drjit.eval`). It's perfectly valid to index into
   nested Dr.Jit arrays like :py:class:`drjit.llvm.Array3f`---the end result
   should just not be a Python ``int`` or ``float`` since that would require
   knowing the actual array contents. Printing array contents is also possible,
   but it requires a special *unevaluated* operation named
   :py:func:`drjit.print_async`. If you wish to avoid such complications,
   consider wavefront-mode compilation discussed next.

2. **Wavefront mode**: When :py:attr:`drjit.JitFlag.VCallRecord` is *not* set,
   Dr.Jit *evaluates* the inputs  ``index``, ``args``, ``kwargs`` via
   :py:func:`drjit.eval`, groups them by the provided index, and invokes each
   callable with with the subset of inputs that reference it. Callables that
   are not referenced by any element of ``index`` are ignored.

   In this mode, a :py:func:`drjit.switch` statement will cause Dr.Jit to
   launch a series of kernels processing subsets of the input data (one per
   callable), which is referred to as *wavefronts* in Dr.Jit.

   This can negatively impact performance and memory usage as function
   arguments must be written to device memory. On the other hand,
   wavefront-mode execution is simpler to understand and debug. It is possible
   to single-step through programs, examine array contents, etc.

To switch the compilation mode locally, use :py:func:`drjit.scoped_set_flag` as
shown below:

.. code-block:: python

   with dr.scoped_set_flag(dr.JitFlag.VCallRecord, False):
       result = dr.switch(..)

The functions :py:func:`drjit.switch` and :py:func:`drjit.dispatch` may be
arbitrarily nested. However, a callable invoked by a symbolic-mode
:py:func:`drjit.switch` call may not perform a wavefront-style
:py:func:`dr.switch` or :py:func:`dr.dispatch` since this would require the
evaluation of symbolic variables.

Args:
    index (int|drjit.ArrayBase): a list of indices to choose the functions

    callables (Sequence[Callable]): a list of callables to which calls will be
      dispatched based on the `index` argument.

    *args (tuple): a variable-length list of positional arguments passed to the
      callables.

    **kwargs (dict): a variable-length list of keyword arguments passed to the
      callables.

Returns:
    object: A single function reutrn value, or an array of return values. The
    specifics depend on whether the `index` argument is scalar, and what the
    provided functions return.)";

static const char *doc_collect_indices = R"(
Return Dr.Jit variable indices associated with the provided data structure.

This function traverses Dr.Jit arrays, tensors, :ref:`Pytree <pytrees>` (lists,
tuples, dicts, custom data structures) and returns the indices of all detected
variables (in the order of traversal, may contain duplicates). The index
information is returned as a list of encoded 64 bit integers, where each
contains the AD variable index in the upper 32 bits and the JIT variable index
in the lower 32 bit.

Intended purely for internal Dr.Jit use, you probably should not call this in
your own application.)";

static const char *doc_update_indices = R"(
Create a copy of the provided input while replacing Dr.Jit variables with
new ones based on a provided set of indices.

This function works analogously to ``collect_indices``, except that it
consumes an index array and produces an updated output.

It recursively traverses and copies an input object that may be a Dr.Jit array,
tensor, or :ref:`Pytree <pytrees>` (list, tuple, dict, custom data structure)
while replacing any detected Dr.Jit variables with new ones based on the
provided index vector. The function returns the resulting object, while leaving
the input unchanged. The output array borrows the referenced array indices
as opposed to stealing them.

Intended purely for internal Dr.Jit use, you probably should not call this in
your own application.)";

static const char *doc_check_compatibility = R"(
Traverse two pytrees in parallel and ensure that they have an identical
structure.

Raises an exception is a mismatch is found (e.g., different types, arrays with
incompatible numbers of elements, dictionaries with different keys, etc.))";

static const char *doc_flag =
    "Query whether the given Dr.Jit compilation flag is active.";

static const char *doc_set_flag =
    "Set the value of the given Dr.Jit compilation flag.";

static const char *doc_scoped_set_flag = R"(
Context manager, which sets or unsets a Dr.Jit compilation flag in a local
execution scope.

For example, the following snippet shows how to temporarily disable a flag:

.. code-block:: python

   with dr.scoped_set_flag(dr.JitFlag.VCallRecord, False):
       # Code affected by the change should be placed here

   # Flag is returned to its original status
)";

static const char *doc_JitFlag = R"(
Flags that control how Dr.Jit compiles and optimizes programs.

This enumeration lists various flag that control how Dr.Jit compiles and
optimizes programs, most of which are enabled by default. The status of each
flag can be queried via :py:func:`drjit.flag` and enabled/disabled via the
:py:func:`drjit.scoped_flag` and :py:func:`drjit.scoped_set_flag` functions.

The most common reason to update the flags is to switch between *wavefront* and
*recorded* execution of loops and functions. The former eagerly executes
programs by breaking them into many smaller kernels, while the latter records
computation symbolically to assemble large *megakernels*. See the documentation
of :py:func:`drjit.switch` and :py:class:`drjit.Loop` for more details on these
two modes.

Dr.Jit flags are a thread-local property. This means that multiple independent
threads using Dr.Jit can set them independently without interfering with each
other.)";

static const char *doc_JitFlag_ConstantPropagation =R"(
**Constant propagation**: immediately evaluate arithmetic involving literal
constants on the host and don't generate any device-specific code for them.

For example, the following assertion holds when value numbering is enabled in
Dr.Jit.

.. code-block:: python

   from drjit.llvm import Int

   # Create two literal constant arrays
   a, b = Int(4), Int(5)

   # This addition operation can be immediately performed and does not need to be recorded
   c1 = a + b

   # Double-check that c1 and c2 refer to the same Dr.Jit variable
   c2 = Int(9)
   assert c1.index == c2.index

Enabled by default.)";

static const char *doc_JitFlag_ValueNumbering = R"(
**Local value numbering**: a simple variant of common subexpression elimination
that collapses identical expressions within basic blocks. For example, the
following assertion holds when value numbering is enabled in Dr.Jit.

.. code-block:: python

   from drjit.llvm import Int

   # Create two nonliteral arrays stored in device memory
   a, b = Int(1, 2, 3), Int(4, 5, 6)

   # Perform the same arithmetic operation twice
   c1 = a + b
   c2 = a + b

   # Verify that c1 and c2 reference the same Dr.Jit variable
   assert c1.index == c2.index

Enabled by default.)";

static const char *doc_JitFlag_VCallRecord = R"(
**Recorded function calls**: Dr.Jit provides two main ways of compiling
*indirect function calls* (aka. *virtual function calls* or *dynamic dispatch*).

1. **Recorded mode**: When this flag is set (the default), Dr.Jit captures
   callables by invoking them with symbolic/abstract arguments. These
   transcripts are then turned into function calls in the generated program.
   In a sense, recorded mode most closely preserves the original program
   semantics.

   The main advantage of recorded mode is:

   * It is very efficient in terms of device memory storage and bandwidth, since
     function call arguments and return values can be exchanged through fast
     CPU/GPU registers.

   Its main downsides are:

   * Symbolic arrays cannot be evaluated, printed, etc. Attempting to
     perform such operations will raise an exception.

     This limitation may be inconvenient especially when debugging code, in
     which case wavefront mode is preferable.

   * Thread divergence: neighboring SIMD lanes may target different callables,
     which can have a significant negative impact on refficiency.

   * A kernel with many callables can become quite large and costly to compile.

2. **Wavefront mode**: In this mode, Dr.Jit to launches a series of kernels
   processing subsets of the input data (one per callable).

   The main advantages of wavefront mode is:

   * Easy to debug / step through programs and examine intermediate results.

   * Kernels are smaller and avoid thread divergence, since Dr.Jit reorders
     computation by callable.

   The main downsides are:

   * Each callable essentially turns its own kernel that reads its input and
     writes outputs via device memory. The required memory bandwidth and
     storage are often so overwhelming that wavefront mode becomes impractical.

Recorded mode is enabled by default.)";

static const char *doc_JitFlag_IndexReuse = R"(
**Index reuse**: Dr.Jit consists of two main parts: the just-in-time compiler,
and the automatic differentiation layer. Both maintain an internal data
structure representing captured computation, in which each variable is
associated with an index (e.g., ``r1234`` in the JIT compiler,
and ``a1234`` in the AD graph).

The index of a Dr.Jit array in these graphs can be queried via the
:py:attr:`drjit.index` and :py:attr:`drjit.index_ad` variables, and they are
also visible in debug messages (if :py:func:`drjit.set_log_level` is set to a
more verbose debug level).

Dr.Jit aggressively reuses the indices of expired variables by default, but
this can make debug output difficult to interpret. When when debugging Dr.Jit
itself, it is often helpful to investigate the history of a particular
variable. In such cases, set this flag to ``False`` to disable variable reuse
both at the JIT and AD levels. This comes at a cost: the internal data
structures keep on growing, so it is not suitable for long-running
computations.

Index reuse is enabled by default.)";

static const char *doc_JitFlag_Default = "The default set of flags.";

#if defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

