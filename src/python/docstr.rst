.. ------------------------------------------------------------------------

.. topic:: introduction

    This file contains docstrings of various C++ bindings. The Dr.Jit build
    system loads this file and converts it into a C++ header file containing raw
    string versions of everything below. This generated header file is
    subsequently included by all bindings, which can reference the strings below
    with an added ``doc_`` prefix. For example, the docstring of ``is_array_v``
    is exposed as ``doc_is_array_v``, etc..

    The reason for this indirection is that Dr.Jit's RST (RestructuredText)
    documentation is most conveniently written and edited via proper ``.rst``
    file so that editors can provide syntax highlighting, spell-checking, etc.

    At the same time, nanobind tool expects these docstrings to be provided
    during the C++ compilation process. This file is therefore processed to
    generate a header file making the docstrings available at compile time.

.. ------------------------------------------------------------------------

.. topic:: is_array_v

    Check if the input is a Dr.Jit array instance or type

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` or type(``arg``) is a Dr.Jit array type, and
          ``False`` otherwise

.. topic:: is_struct_v

    Check if the input is a Dr.Jit-compatible data structure

    Custom data structures can be made compatible with various Dr.Jit operations by
    specifying a ``DRJIT_STRUCT`` member. See the section on :ref:`PyTrees
    <pytrees>` for details. This type trait can be used to check
    for the existence of such a field.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` has a ``DRJIT_STRUCT`` member

.. topic:: size_v

    Return the (static) size of the outermost dimension of the provided Dr.Jit
    array instance or type

    Note that this function mainly exists to query type-level information. Use the
    Python ``len()`` function to query the size in a way that does not distinguish
    between static and dynamic arrays.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        int: Returns either the static size or :py:data:`drjit.Dynamic` when
        ``arg`` is a dynamic Dr.Jit array. Returns ``1`` for all other types.

.. topic:: depth_v

    Return the depth of the provided Dr.Jit array instance or type

    For example, an array consisting of floating point values (for example,
    :py:class:`drjit.scalar.Array3f`) has depth ``1``, while an array consisting of
    sub-arrays (e.g., :py:class:`drjit.cuda.Array3f`) has depth ``2``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        int: Returns the depth of the input, if it is a Dr.Jit array instance or
        type. Returns ``0`` for all other inputs.

.. topic:: itemsize_v

    Return the per-item size (in bytes) of the scalar type underlying a Dr.Jit array

    Args:
        arg (object): A Dr.Jit array instance or array type.

    Returns:
        int: Returns the item size array elements in bytes.

.. topic:: value_t

    Return the *value type* underlying the provided Dr.Jit array or type (i.e., the
    type of values obtained by accessing the contents using a 1D index).

    When the input is not a Dr.Jit array or type, the function returns the input
    type without changes. The following code fragment shows several example uses of
    :py:func:`value_t`.

    .. code-block:: python

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

.. topic:: array_t

    Return the *plain array form* of the provided Dr.Jit array or type.

    There are several different cases:

    - When ``self`` is a tensor, this property returns the storage representation
      of the tensor in the form of a linearized dynamic 1D array. For example,
      the following hold:

      .. code-block:: python

        assert dr.array_t(dr.scalar.TensorXf) is dr.scalar.ArrayXf
        assert dr.array_t(dr.cuda.TensorXf) is dr.cuda.Float

    - When ``arg`` represents a special arithmetic object (matrix, quaternion, or
      complex number), ``array_t`` returns a similarly-shaped type with ordinary
      array semantics. For example, the following hold

      .. code-block:: python

        assert dr.array_t(dr.scalar.Complex2f) is dr.scalar.Array2f
        assert dr.array_t(dr.scalar.Matrix4f) is dr.scalar.Array44f

    - In all other cases, the function returns the input type.

    The property :py:func:`ArrayBase.array` returns a result with this the
    type computed by this function.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Returns the array form as per the above description.

.. topic:: tensor_t

    Return a tensor type that is compatible with the provided Dr.Jit array or type.

    This type trait is useful when a variable should be converted into a tensor,
    but it is not clear which tensor type is suitable (e.g., because the input
    has a dynamic type).

    Example usage:

    .. code-block:: python

       x = dr.llvm.Array3f(...)
       tp = dr.tensor_t(type(x)) # <-- returns dr.llvm.TensorXf
       x_t = tp(x)

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Returns a compatible tensor type or ``None``.

.. topic:: mask_t

    Return the *mask type* associated with the provided Dr.Jit array or type (i.e., the
    type produced by comparisons involving the argument).

    When the input is not a Dr.Jit array or type, the function returns the scalar
    Python ``bool`` type. The following assertions illustrate the behavior of
    :py:func:`mask_t`.


    .. code-block:: python

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

.. topic:: scalar_t

    Return the *scalar type* associated with the provided Dr.Jit array or type
    (i.e., the representation of elements at the lowest level)

    When the input is not a Dr.Jit array or type, the function returns its input
    unchanged. The following assertions illustrate the behavior of
    :py:func:`scalar_t`.


    .. code-block:: python

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

.. topic:: is_mask_v

    Check whether the input array instance or type is a Dr.Jit mask array or a
    Python ``bool`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents a Dr.Jit mask array or Python ``bool``
        instance or type.

.. topic:: is_integral_v

    Check whether the input array instance or type is an integral Dr.Jit array
    or a Python ``int`` value/type.

    Note that a mask array is not considered to be integral.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an integral Dr.Jit array or
        Python ``int`` instance or type.

.. topic:: is_float_v

    Check whether the input array instance or type is a Dr.Jit floating point array
    or a Python ``float`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents a Dr.Jit floating point array or
        Python ``float`` instance or type.

.. topic:: is_half_v

    Check whether the input array instance or type is a Dr.Jit half-precision floating
    point array or a Python ``half`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents a Dr.Jit half-precision
        floating point array or Python ``half`` instance or type.

.. topic:: is_arithmetic_v

    Check whether the input array instance or type is an arithmetic Dr.Jit array
    or a Python ``int`` or ``float`` value/type.

    Note that a mask type (e.g. ``bool``, :py:class:`drjit.scalar.Array2b`, etc.)
    is *not* considered to be arithmetic.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an arithmetic Dr.Jit array or Python
        ``int`` or ``float`` instance or type.

.. topic:: is_signed_v

    Check whether the input array instance or type is an signed Dr.Jit array
    or a Python ``int`` or ``float`` value/type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an signed Dr.Jit array or Python
        ``int`` or ``float`` instance or type.

.. topic:: is_unsigned_v

    Check whether the input array instance or type is an unsigned integer Dr.Jit
    array or a Python ``bool`` value/type (masks and boolean values are also
    considered to be unsigned).

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an unsigned Dr.Jit array or Python
        ``bool`` instance or type.

.. topic:: is_jit_v

    Check whether the input array instance or type represents a type that
    undergoes just-in-time compilation.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.cuda.*`` or ``drjit.llvm.*`` namespaces, and ``False`` otherwise.

.. topic:: is_dynamic_v

    Check whether the input instance or type represents a dynamically sized Dr.Jit
    array type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_diff_v

    Check whether the input is a differentiable Dr.Jit array instance or type.

    Note that this is a type-based statement that is unrelated to mathematical
    differentiability. For example, the integral type :py:class:`drjit.cuda.ad.Int`
    from the CUDA AD namespace satisfies ``is_diff_v(..) = 1``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if ``arg`` represents an array type from the
        ``drjit.[cuda/llvm].ad.*`` namespace, and ``False`` otherwise.

.. topic:: backend_v

    Returns the backend responsible for the given Dr.Jit array instance or type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        drjit.JitBackend: The associated Jit backend or ``drjit.JitBackend.None``.

.. topic:: type_v

    Returns the scalar type associated with the given Dr.Jit array instance or
    type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        drjit.VarType: The associated type ``drjit.VarType.Void``.

.. topic:: replace_type_t

    Converts the provided Dr.Jit array/tensor type into an analogous version with
    the specified scalar type.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* ``arg0``, it returns an analogous
       version with a different scalar type, as specified via ``arg1``. For example,
       when called with :py:class:`drjit.cuda.Array3u` and
       :py:attr:`drjit.VarType.Float32`, it will return
       :py:class:`drjit.cuda.Array3f`.

    2. When the input is not a type, it returns ``replace_type_t(type(arg0), arg1)``.

    3. When the input is not a Dr.Jit type, the function returns ``arg0``.

    Args:
        arg0 (object): An arbitrary Python object

        arg1 (drjit.VarType): The desired variable type

    Returns:
        type: Result of the conversion as described above.

.. topic:: is_complex_v

    Check whether the input is a Dr.Jit array instance or type representing a complex number.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_quaternion_v

    Check whether the input is a Dr.Jit array instance or type representing a quaternion.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_vector_v

    Check whether the input is a Dr.Jit array instance or type representing a vectorial array type.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_matrix_v

    Check whether the input is a Dr.Jit array instance or type representing a matrix.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_tensor_v

    Check whether the input is a Dr.Jit array instance or type representing a tensor.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: is_special_v

    Check whether the input is a *special* Dr.Jit array instance or type.

    A *special* array type requires precautions when performing arithmetic
    operations like multiplications (complex numbers, quaternions, matrices).

    Args:
        arg (object): An arbitrary Python object

    Returns:
        bool: ``True`` if the test was successful, and ``False`` otherwise.

.. topic:: select

    Select elements from inputs based on a condition

    This function uses a first mask argument to select between the subsequent
    two arguments. It implements the following component-wise operation:

    .. math::

       \mathrm{result}_i = \begin{cases}
           \texttt{arg1}_i,\quad&\text{if }\texttt{arg0}_i,\\
           \texttt{arg2}_i,\quad&\text{otherwise.}
       \end{cases}

    Args:
        arg0 (bool | drjit.ArrayBase): A Python or Dr.Jit mask type

        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit type, whose
          entries should be returned for ``True``-valued mask entries.

        arg2 (int | float | drjit.ArrayBase): A Python or Dr.Jit type, whose
          entries should be returned for ``False``-valued mask entries.

    Returns:
        float | int | drjit.ArrayBase: Component-wise result of the selection operation

.. topic:: abs

    Compute the absolute value of the provided input.

    This function evaluates the component-wise absolute value of the input
    scalar, array, or tensor. When called with a complex or quaternion-valued
    array, it uses a suitable generalization of the operation.

    Args:
        arg (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        int | float | drjit.ArrayBase: Absolute value of the input

.. topic:: maximum

    Compute the element-wise maximum value of the provided inputs.

    (Not to be confused with :py:func:`drjit.max`, which reduces the input
    along the specified axes to determine the maximum)

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        int | float | drjit.ArrayBase: Maximum of the input(s)

.. topic:: minimum

    Compute the element-wise minimum value of the provided inputs.

    (Not to be confused with :py:func:`drjit.min`, which reduces the input
    along the specified axes to determine the minimum)

    Args:
        arg0 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type
        arg1 (int | float | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        int | float | drjit.ArrayBase: Minimum of the input(s)

.. topic:: square

    Compute the square of the input array, tensor, or arithmetic type.

    Args:
        arg (object): A Python or Dr.Jit arithmetic type

    Returns:
        object: The result of the operation ``arg*arg``

.. topic:: pow

    Raise the first argument to a power specified via the second argument.

    This function evaluates the component-wise power of the input scalar, array, or
    tensor arguments. When called with a complex or quaternion-valued inputs, it
    uses a suitable generalization of the operation.

    When ``arg1`` is a Python ``int`` or integral ``float`` value, the function
    reduces operation to a sequence of multiplies and adds (potentially
    followed by a reciprocation operation when ``arg1`` is negative).

    The general case involves recursive use of the identity ``pow(arg0, arg1) =
    exp2(log2(arg0) * arg1)``.

    There is no difference between using :py:func:`drjit.power()` and the builtin
    Python ``**`` operator.

    Args:
        arg (object): A Python or Dr.Jit arithmetic type

    Returns:
        object: The result of the operation ``arg0**arg1``

.. topic:: matmul

    Compute a matrix-matrix, matrix-vector, vector-matrix, or inner product.

    This function implements the semantics of the ``@`` operator introduced in
    Python's `PEP 465 <https://peps.python.org/pep-0465/>`__. There is no practical
    difference between using :py:func:`drjit.matul()` or ``@`` in Dr.Jit-based
    code. Multiplication of matrix types (e.g., :py:class:`drjit.scalar.Matrix2f`)
    using the standard multiplication operator (``*``) is also based on on matrix
    multiplication.

    This function takes two Dr.Jit arrays and picks one of the following 5 cases
    based on their leading fixed-size dimensions.

    - **Matrix-matrix product**: If both arrays have leading static dimensions
      ``(n, n)``, they are multiplied like conventional matrices.

    - **Matrix-vector product**: If ``arg0`` has leading static dimensions ``(n,
      n)`` and ``arg1`` has leading static dimension ``(n,)``, the operation
      conceptually appends a trailing 1-sized dimension to ``arg1``, multiplies,
      and then removes the extra dimension from the result.

    - **Vector-matrix product**: If ``arg0`` has leading static dimensions ``(n,)``
      and ``arg1`` has leading static dimension ``(n, n)``, the operation
      conceptually prepends a leading 1-sized dimension to ``arg0``, multiplies,
      and then removes the extra dimension from the result.

    - **Inner product**: If ``arg0`` and ``arg1`` have leading static dimensions
      ``(n,)``, the operation returns the sum of the elements of ``arg0*arg1``.

    - **Scalar product**: If ``arg0`` or ``arg1`` is a scalar, the operation scales
      the elements of the other argument.

    It is legal to combine vectorized and non-vectorized types, e.g.

    .. code-block:: python

       dr.matmul(dr.scalar.Matrix4f(...), dr.cuda.Matrix4f(...))

    Also, note that doesn't matter whether an input is an instance of a matrix type
    or a similarly-shaped nested array---for example,
    :py:func:`drjit.scalar.Matrix3f` and :py:func:`drjit.scalar.Array33f` have the
    same shape and are treated identically.

    .. note::

       This operation only handles fixed-sizes arrays. A different approach is
       needed for multiplications involving potentially large dynamic
       arrays/tensors. Other other tools like PyTorch, JAX, or Tensorflow will be
       preferable in such situations (e.g., to train neural networks).

    Args:
        arg0 (dr.ArrayBase): Dr.Jit array type

        arg1 (dr.ArrayBase): Dr.Jit array type

    Returns:
        object: The result of the operation as defined above

.. topic:: reduce

    Reduce the input array, tensor, or iterable along the specified axis/axes.

    This function reduces arrays, tensors and other iterable Python types along
    one or multiple axes, where ``op`` selects the operation to be performed:

    - :py:attr:`drjit.ReduceOp.Add`: ``a[0] + a[1] + ...``.
    - :py:attr:`drjit.ReduceOp.Mul`: ``a[0] * a[1] * ...)``.
    - :py:attr:`drjit.ReduceOp.Min`: ``min(a[0], a[1], ...)``.
    - :py:attr:`drjit.ReduceOp.Max`: ``max(a[0], a[1], ...)``.
    - :py:attr:`drjit.ReduceOp.Or`: ``a[0] | a[1] | ...`` (integer arrays only).
    - :py:attr:`drjit.ReduceOp.And`: ``a[0] & a[1] & ...`` (integer arrays only).

    The functions :py:func:`drjit.sum()`, :py:func:`drjit.prod()`,
    :py:func:`drjit.min()`, and :py:func:`drjit.max()` are convenience aliases
    that call :py:func:`drjit.reduce()` with specific values of ``op``.

    By default, the reduction is along axis ``0`` (i.e., the outermost
    one), returning an instance of the array's element type. For instance,
    sum-reducing an array ``a`` of type :py:class:`drjit.cuda.Array3f` is
    equivalent to writing ``a[0] + a[1] + a[2]`` and produces a result of type
    :py:class:`drjit.cuda.Float`. Dr.Jit can trace this operation and include
    it in the generated kernel.

    Negative indices (e.g. ``axis=-1``) count backward from the innermost
    axis. Multiple axes can be specified as a tuple. The value ``axis=None``
    requests a simultaneous reduction over all axes.

    When reducing axes of a tensor, or when reducing the *trailing* dimension
    of a Jit-compiled array, some special precautions apply: these axes
    correspond to computational threads of a large parallel program that now
    have to coordinate to determine the reduced value. This can be done
    using the following strategies:

    - ``mode="evaluated"`` first evaluates the input array via
      :py:func:`drjit.eval()` and then launches a specialized reduction kernel.

      On the CUDA backend, this kernel makes efficient use of shared memory and
      cooperative warp instructions. The LLVM backend parallelizes the
      reduction via the built-in thread pool.

    - ``mode="symbolic"`` uses :py:func:`drjit.scatter_reduce()` to atomically
      scatter-reduce values into the output array. This strategy can be
      advantageous when the input is symbolic (making evaluation
      impossible) or both unevaluated and extremely large (making evaluation
      costly or impossible if there isn't enough memory).

      Disadvantages of this mode are that

      - Atomic scatters can suffer from memory contention (though the
        :py:func:`drjit.scatter_reduce()` function takes steps to reduce
        contention, see its documentation for details).

      - Atomic floating point scatter-addition is subject to non-deterministic
        rounding errors that arise from its non-commutative nature. Coupled
        with the scheduling-dependent execution order, this can lead to small
        variations across program runs. Integer reductions and floating point
        min/max reductions are unaffected by this.

    - ``mode=None`` (default) automatically picks a reasonable strategy
      according to the following logic:

      - Use evaluated mode when the input array is already evaluated, or when
        evaluating it would consume less than 1 GiB of memory.

      - Use evaluated mode when the necessary atomic reduction operation is
        :ref:`not supported <scatter_reduce_supported>` by the backend.

      - Otherwise, use symbolic mode.

    This function generally strips away reduced axes, but there is one notable
    exception: it will *never* remove a trailing dynamic dimension, if present
    in the input array.

    For example, reducing an instance of type :py:class:`drjit.cuda.Float`
    along axis ``0`` does not produce a scalar Python ``float``. Instead, the
    operation returns another array of the same type with a single element.
    This is intentional--unboxing the array into a Python scalar would require
    transferring the value from the GPU, which would incur costly
    synchronization overheads. You must explicitly index into the result
    (``result[0]``) to obtain a value with the underlying element type.

    Args:
        op (ReduceOp): The operation that should be applied along the
          reduced axis/axes.

        value (ArrayBase | Iterable | float | int): An input Dr.Jit array or tensor.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

        mode (str | None): optional parameter to force an evaluation strategy.
          Must equal ``"evaluated"``, ``"symbolic"``, or ``None``.

    Returns:
        The reduced array or tensor as specified above.

.. topic:: sum

    Sum-reduce the input array, tensor, or iterable along the specified axis/axes.

    This function sum-reduces arrays, tensors and other iterable Python types
    along one or multiple axes. It is equivalent to
    :py:func:`dr.reduce(dr.ReduceOp.Add, ...) <reduce>`. See the documentation of
    this function for further information.

    Args:
        value (ArrayBase | Iterable | float | int): An input Dr.Jit array,
          tensor, iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

        mode (str | None): optional parameter to force an evaluation strategy.
          Must equal ``"evaluated"``, ``"symbolic"``, or ``None``.

    Returns:
        object: The reduced array or tensor as specified above.

.. topic:: prod

    Multiplicatively reduce the input array, tensor, or iterable along the specified axis/axes.

    This function performs a multiplicative reduction along one or multiple axes of
    the provided Dr.Jit array, tensor, or iterable Python types. It is
    equivalent to :py:func:`dr.reduce(dr.ReduceOp.Mul, ...) <reduce>`. See
    the documentation of this function for further information.

    Args:
        value (ArrayBase | Iterable | float | int): An input Dr.Jit array,
          tensor, iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

        mode (str | None): optional parameter to force an evaluation strategy.
          Must equal ``"evaluated"``, ``"symbolic"``, or ``None``.

    Returns:
        object: The reduced array or tensor as specified above.

.. topic:: min

    Perform a minimum reduction of the input array, tensor, or iterable along
    the specified axis/axes.

    (Not to be confused with :py:func:`drjit.minimum`, which computes the
    smaller of two values).

    This function performs a minimum reduction along one or multiple axes of
    the provided Dr.Jit array, tensor, or iterable Python types. It is
    equivalent to :py:func:`dr.reduce(dr.ReduceOp.Min, ...) <reduce>`. See
    the documentation of this function for further information.

    Args:
        value (ArrayBase | Iterable | float | int): An input Dr.Jit array,
          tensor, iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

        mode (str | None): optional parameter to force an evaluation strategy.
          Must equal ``"evaluated"``, ``"symbolic"``, or ``None``.

    Returns:
        object: The reduced array or tensor as specified above.

.. topic:: max

    Perform a maximum reduction of the input array, tensor, or iterable along
    the specified axis/axes.

    (Not to be confused with :py:func:`drjit.maximum`, which computes the
    larger of two values).

    This function performs a maximum reduction along one or multiple axes of
    the provided Dr.Jit array, tensor, or iterable Python types. It is
    equivalent to :py:func:`dr.reduce(dr.ReduceOp.Max, ...) <reduce>`. See
    the documentation of this function for further information.

    Args:
        value (ArrayBase | Iterable | float | int): An input Dr.Jit array, tensor,
          iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

        mode (str | None): optional parameter to force an evaluation strategy.
          Must equal ``"evaluated"``, ``"symbolic"``, or ``None``.

    Returns:
        The reduced array or tensor as specified above.

.. topic:: all

    Check if all elements along the specified axis are active.

    Given a boolean-valued input array, tensor, or Python sequence, this function
    reduces elements using the ``&`` (AND) operator.

    By default, it reduces along index ``0``, which refers to the outermost axis.
    Negative indices (e.g. ``-1``) count backwards from the innermost axis. The
    special argument ``axis=None`` causes a simultaneous reduction over all axes.
    Note that the reduced form of an *empty* array is considered to be ``True``.

    The function is internally based on :py:func:`dr.reduce() <reduce>`. See
    the documentation of this function for further information.

    Like :py:func:`dr.reduce()`, this function does *not* strip away trailing
    dynamic dimensions if present in the input array. This means that reducing
    :py:class:`drjit.cuda.Bool` does not produce a scalar Python ``bool``.
    Instead, the operation returns another array of the same type with a single
    element. This is intentional--unboxing the array into a Python scalar would
    require transferring the value from the GPU, which would incur costly
    synchronization overheads. You must explicitly index into the result
    (``result[0]``) to obtain a value with the underlying element type.

    Boolean 1D arrays automatically convert to ``bool`` if they only contain a
    single element. This means that the aforementioned indexing operation
    happens implicitly in the following fragment:

    .. code-block:: python

       from drjit.cuda import Float

       x = Float(...)
       if dr.all(s < 0):
          # ...

    A last point to consider is that reductions along the last / trailing
    dynamic axis of an array are generally expensive. Its entries correspond to
    computational threads of a large parallel program that now have to
    coordinate to determine the reduced value. Normally, this involves
    :py:func:`drjit.eval` to evaluate and store the array in memory and then
    launch a device-specific reduction kernel. All of these steps interfere
    with Dr.Jit's regular mode of operation, which is to capture a maximally
    large program without intermediate evaluation.

    To avoid Boolean reductions, one can often use *symbolic operations* such
    as :py:func:`if_stmt`, :py:func:`while_loop`, etc. The :py:func:`@dr.syntax
    <syntax>` decorator can generate these automatically. For example, the
    following fragment predicates the execution of the body (``# ...``) based
    on the condition.

    .. code-block:: python

       @dr.syntax
       def f(x: Float):
           if a < 0:
              # ...

    Args:
        value (ArrayBase | Iterable | bool): An input Dr.Jit array, tensor,
          iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

    Returns:
        object: The reduced array or tensor as specified above.

.. topic:: any

    Check if any elements along the specified axis are active.

    Given a boolean-valued input array, tensor, or Python sequence, this function
    reduces elements using the ``|`` (OR) operator.

    By default, it reduces along index ``0``, which refers to the outermost axis.
    Negative indices (e.g. ``-1``) count backwards from the innermost axis. The
    special argument ``axis=None`` causes a simultaneous reduction over all axes.
    Note that the reduced form of an *empty* array is considered to be ``False``.

    The function is internally based on :py:func:`dr.reduce() <reduce>`. See
    the documentation of this function for further information.

    Like :py:func:`dr.reduce()`, this function does *not* strip away trailing
    dynamic dimensions if present in the input array. This means that reducing
    :py:class:`drjit.cuda.Bool` does not produce a scalar Python ``bool``.
    Instead, the operation returns another array of the same type with a single
    element. This is intentional--unboxing the array into a Python scalar would
    require transferring the value from the GPU, which would incur costly
    synchronization overheads. You must explicitly index into the result
    (``result[0]``) to obtain a value with the underlying element type.

    Boolean 1D arrays automatically convert to ``bool`` if they only contain a
    single element. This means that the aforementioned indexing operation
    happens implicitly in the following fragment:

    .. code-block:: python

       from drjit.cuda import Float

       x = Float(...)
       if dr.any(s < 0):
          # ...

    A last point to consider is that reductions along the last / trailing
    dynamic axis of an array are generally expensive. Its entries correspond to
    computational threads of a large parallel program that now have to
    coordinate to determine the reduced value. Normally, this involves
    :py:func:`drjit.eval` to evaluate and store the array in memory and then
    launch a device-specific reduction kernel. All of these steps interfere
    with Dr.Jit's regular mode of operation, which is to capture a maximally
    large program without intermediate evaluation.

    To avoid Boolean reductions, one can often use *symbolic operations* such
    as :py:func:`if_stmt`, :py:func:`while_loop`, etc. The :py:func:`@dr.syntax
    <syntax>` decorator can generate these automatically. For example, the
    following fragment predicates the execution of the body (``# ...``) based
    on the condition.

    .. code-block:: python

       @dr.syntax
       def f(x: Float):
           if a < 0:
              # ...

    Args:
        value (ArrayBase | Iterable | bool): An input Dr.Jit array, tensor,
          iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

    Returns:
        bool | drjit.ArrayBase: Result of the reduction operation

.. topic:: none

    Check if none elements along the specified axis are active.

    Given a boolean-valued input array, tensor, or Python sequence, this function
    reduces elements using the ``|`` (OR) operator and finally returns the bit-wise
    *inverse* of the result.

    The function is internally based on :py:func:`dr.reduce() <reduce>`. See
    the documentation of this function for further information.

    Like :py:func:`dr.reduce()`, this function does *not* strip away trailing
    dynamic dimensions if present in the input array. This means that reducing
    :py:class:`drjit.cuda.Bool` does not produce a scalar Python ``bool``.
    Instead, the operation returns another array of the same type with a single
    element. This is intentional--unboxing the array into a Python scalar would
    require transferring the value from the GPU, which would incur costly
    synchronization overheads. You must explicitly index into the result
    (``result[0]``) to obtain a value with the underlying element type.

    Boolean 1D arrays automatically convert to ``bool`` if they only contain a
    single element. This means that the aforementioned indexing operation
    happens implicitly in the following fragment:

    .. code-block:: python

       from drjit.cuda import Float

       x = Float(...)
       if dr.none(s < 0):
          # ...

    A last point to consider is that reductions along the last / trailing
    dynamic axis of an array are generally expensive. Its entries correspond to
    computational threads of a large parallel program that now have to
    coordinate to determine the reduced value. Normally, this involves
    :py:func:`drjit.eval` to evaluate and store the array in memory and then
    launch a device-specific reduction kernel. All of these steps interfere
    with Dr.Jit's regular mode of operation, which is to capture a maximally
    large program without intermediate evaluation.

    To avoid Boolean reductions, one can often use *symbolic operations* such
    as :py:func:`if_stmt`, :py:func:`while_loop`, etc. The :py:func:`@dr.syntax
    <syntax>` decorator can generate these automatically. For example, the
    following fragment predicates the execution of the body (``# ...``) based
    on the condition.

    .. code-block:: python

       @dr.syntax
       def f(x: Float):
           if a < 0:
              # ...

    Args:
        value (ArrayBase | Iterable | bool): An input Dr.Jit array, tensor,
          iterable, or scalar Python type.

        axes (int | tuple[int, ...] | None): The axis/axes along which
          to reduce. The default value is ``0``.

    Returns:
        bool | drjit.ArrayBase: Result of the reduction operation

.. topic:: dot

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
        float | int | drjit.ArrayBase: Dot product of inputs

.. topic:: abs_dot

    Compute the absolute value of the dot product of two arrays.

    This function implements a convenience short-hand for ``abs(dot(arg0, arg1))``.

    See the section on :ref:`horizontal reductions <horizontal-reductions>` for
    details on the properties of such horizontal reductions.

    Args:
        arg0 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

        arg1 (list | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        float | int | drjit.ArrayBase: Absolute value of the dot product of inputs

.. topic:: norm

    Computes the 2-norm of a Dr.Jit array, tensor, or Python sequence.

    The operation is equivalent to

    .. code-block:: python

       dr.sqrt(dr.dot(arg, arg))

    The :py:func:`norm` operation performs a horizontal reduction. Please see the
    section on :ref:`horizontal reductions <horizontal-reductions>` for details on
    their properties.

    Args:
        arg (Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        float | int | drjit.ArrayBase: 2-norm of the input

.. topic:: squared_norm

    Computes the squared 2-norm of a Dr.Jit array, tensor, or Python sequence.

    The operation is equivalent to

    .. code-block:: python

       dr.dot(arg, arg)

    The :py:func:`squared_norm` operation performs a horizontal reduction. Please see the
    section on :ref:`horizontal reductions <horizontal-reductions>` for details on
    their properties.

    Args:
        arg (Sequence | drjit.ArrayBase): A Python or Dr.Jit arithmetic type

    Returns:
        float | int | drjit.ArrayBase: squared 2-norm of the input

.. topic:: prefix_sum

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
        drjit.ArrayBase: An array of the same type containing the computed prefix sum.

.. topic:: sqrt

    Evaluate the square root of the provided input.

    This function evaluates the component-wise square root of the input
    scalar, array, or tensor. When called with a complex or quaternion-valued
    array, it uses a suitable generalization of the operation.

    Negative inputs produce a *NaN* output value. Consider using the
    :py:func:`safe_sqrt` function to work around issues where the input might
    occasionally be negative due to prior round-off errors.

    Another noteworthy behavior of the square root function is that it has an
    infinite derivative at ``arg=0``, which can cause infinities/NaNs in gradients
    computed via forward/reverse-mode AD. The :py:func:`safe_sqrt` function
    contains a workaround to ensure a finite derivative in this case.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Square root of the input

.. topic:: cbrt

    Evaluate the cube root of the provided input.

    This function is currently only implemented for real-valued inputs.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cube root of the input

.. topic:: rcp

    Evaluate the reciprocal (1 / arg) of the provided input.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    slightly approximately---see the documentation of the instruction
    ``rcp.approx.ftz.f32`` in the
    `NVIDIA PTX manual <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp>`__ for details.
    For full IEEE-754 compatibility, unset :py:attr:`drjit.JitFlag.FastMath`.

    When called with a matrix-, complex- or quaternion-valued array, this function
    uses the matrix, complex, or quaternion multiplicative inverse to evaluate the
    reciprocal.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Reciprocal of the input

.. topic:: rsqrt

    Evaluate the reciprocal square root (1 / sqrt(arg)) of the provided input.

    This function evaluates the component-wise reciprocal square root of the input
    scalar, array, or tensor. When called with a complex or quaternion-valued
    array, it uses a suitable generalization of the operation.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    slightly approximately---see the documentation of the instruction
    ``rsqrt.approx.ftz.f32`` in the
    `NVIDIA PTX manual <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp>`__ for details.
    For full IEEE-754 compatibility, unset :py:attr:`drjit.JitFlag.FastMath`.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Reciprocal square root of the input

.. topic:: ceil

    Evaluate the ceiling, i.e. the smallest integer >= arg.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Ceiling of the input

.. topic:: floor

    Evaluate the floor, i.e. the largest integer <= arg.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Floor of the input

.. topic:: trunc

    Truncates arg to the nearest integer by towards zero.

    The function does not convert the type of the input array. A separate
    cast is necessary when integer output is desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Truncated result

.. topic:: round

    Rounds the input to the nearest integer using Banker's rounding for half-way
    values.

    This function is equivalent to ``std::rint`` in C++. It does not convert the
    type of the input array. A separate cast is necessary when integer output is
    desired.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Rounded result

.. topic:: log

    Evaluate the natural logarithm.

    This function evaluates the component-wise natural logarithm of the input
    scalar, array, or tensor.
    It uses a suitable generalization of the operation when the input
    is complex- or quaternion-valued.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Natural logarithm of the input

.. topic:: log2

    Evaluate the base-2 logarithm.

    This function evaluates the component-wise base-2 logarithm of the input
    scalar, array, or tensor.
    It uses a suitable generalization of the operation when the input
    is complex- or quaternion-valued.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Base-2 logarithm of the input

.. topic:: exp

    Evaluate the natural exponential function.

    This function evaluates the component-wise natural exponential function of the
    input scalar, array, or tensor. It uses a suitable generalization of the
    operation when the input is complex- or quaternion-valued.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Natural exponential of the input

.. topic:: exp2

    Evaluate ``2`` raised to a given power.

    This function evaluates the component-wise base-2 exponential function of the
    input scalar, array, or tensor. It uses a suitable generalization of the
    operation when the input is complex- or quaternion-valued.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    When ``arg`` is a CUDA single precision array, the operation is implemented
    using the native multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Base-2 exponential of the input

.. topic:: erf

    Evaluate the error function.

    The `error function <https://en.wikipedia.org/wiki/Error_function>` is
    defined as

    .. math::

        \operatorname{erf}(z) = \frac{2}{\sqrt\pi}\int_0^z e^{-t^2}\,\mathrm{d}t.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    This function is currently only implemented for real-valued inputs.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: :math:`\mathrm{erf}(\textt{arg})`

.. topic:: erfinv

    Evaluate the inverse error function.

    This function evaluates the inverse of :py:func:`drjit.erf()`. Its
    implementation is based on the paper `Approximating the erfinv function
    <https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf>`__ by Mike Giles.

    This function is currently only implemented for real-valued inputs.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: :math:`\mathrm{erf}^{-1}(\textt{arg})`

.. topic:: lgamma

    Evaluate the natural logarithm of the absolute value the gamma function.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    This function is currently only implemented for real-valued inputs.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: :math:`\log|\Gamma(\texttt{arg})|`

.. topic:: sin

    Evaluate the sine function.

    This function evaluates the component-wise sine of the input scalar, array, or
    tensor. It uses a suitable generalization of the operation when the input is
    complex-valued.

    The default implementation of this function is based on the CEPHES library and
    is designed to achieve low error on the domain :math:`|x| < 8192` and will not
    perform as well beyond this range. See the section on :ref:`transcendental
    function approximations <transcendental-accuracy>` for details regarding
    accuracy.

    When ``arg`` is a CUDA single precision array, the operation instead uses the
    GPU's built-in multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Sine of the input

.. topic:: cos

    Evaluate the cosine function.

    This function evaluates the component-wise cosine of the input scalar, array,
    or tensor. It uses a suitable generalization of the operation when the input is
    complex-valued.

    The default implementation of this function is based on the CEPHES library. It
    is designed to achieve low error on the domain :math:`|x| < 8192` and will not
    perform as well beyond this range. See the section on :ref:`transcendental
    function approximations <transcendental-accuracy>` for details regarding
    accuracy.

    When ``arg`` is a CUDA single precision array, the operation instead uses
    the GPU's built-in multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Cosine of the input

.. topic:: sincos

    Evaluate both sine and cosine functions at the same time.

    This function simultaneously evaluates the component-wise sine and cosine of
    the input scalar, array, or tensor. This is more efficient than two separate
    calls to :py:func:`drjit.sin` and :py:func:`drjit.cos` when both are required.
    The function uses a suitable generalization of the operation when the input
    is complex-valued.

    The default implementation of this function is based on the CEPHES library. It
    is designed to achieve low error on the domain :math:`|x| < 8192` and will not
    perform as well beyond this range. See the section on :ref:`transcendental
    function approximations <transcendental-accuracy>` for details regarding
    accuracy.

    When ``arg`` is a CUDA single precision array, the operation instead uses
    the hardware's built-in multi-function ("MUFU") unit.


    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Sine and cosine of the input

.. topic:: tan

    Evaluate the tangent function.

    This function evaluates the component-wise tangent function associated with
    each entry of the input scalar, array, or tensor.
    The function uses a suitable generalization of the operation when the input
    is complex-valued.

    The default implementation of this function is based on the CEPHES library. It
    is designed to achieve low error on the domain :math:`|x| < 8192` and will not
    perform as well beyond this range. See the section on :ref:`transcendental
    function approximations <transcendental-accuracy>` for details regarding
    accuracy.

    When ``arg`` is a CUDA single precision array, the operation instead uses
    the GPU's built-in multi-function ("MUFU") unit.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Tangent of the input

.. topic:: asin

    Evaluate the arcsine function.

    This function evaluates the component-wise arcsine of the input scalar, array,
    or tensor. It uses a suitable generalization of the operation when called with
    a complex-valued input.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Real-valued inputs outside of the domain :math:`(-1, 1)` produce a *NaN* output
    value. Consider using the :py:func:`safe_asin` function to work around issues
    where the input might occasionally lie outside of this range due to prior
    round-off errors.

    Another noteworthy behavior of the arcsine function is that it has an infinite
    derivative at :math:`\texttt{arg}=\pm 1`, which can cause infinities/NaNs in
    gradients computed via forward/reverse-mode AD. The :py:func:`safe_asin`
    function contains a workaround to ensure a finite derivative in this case.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arcsine of the input

.. topic:: acos

    Evaluate the arccosine function.

    This function evaluates the component-wise arccosine of the input scalar, array,
    or tensor. It uses a suitable generalization of the operation when the input is
    complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Real-valued inputs outside of the domain :math:`(-1, 1)` produce a *NaN* output
    value. Consider using the :py:func:`safe_acos` function to work around issues
    where the input might occasionally lie outside of this range due to prior
    round-off errors.

    Another noteworthy behavior of the arcsine function is that it has an infinite
    derivative at :math:`\texttt{arg}=\pm 1`, which can cause infinities/NaNs in
    gradients computed via forward/reverse-mode AD. The :py:func:`safe_acos`
    function contains a workaround to ensure a finite derivative in this case.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arccosine of the input

.. topic:: atan

    Evaluate the arctangent function.

    This function evaluates the component-wise arctangent of the input scalar, array,
    or tensor. It uses a suitable generalization of the operation when the input is
    complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arctangent of the input

.. topic:: atan2

    Evaluate the four-quadrant arctangent function.

    This function is currently only implemented for real-valued inputs.

    See the section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        y (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
        x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Arctangent of ``y``/``x``, using the argument signs to
        determine the quadrant of the return value

.. topic:: ldexp

    Multiply x by 2 taken to the power of n

    Args:
        x (float | drjit.ArrayBase): A Python or Dr.Jit floating point type
        n (float | drjit.ArrayBase): A Python or Dr.Jit floating point type *without fractional component*

    Returns:
        float | drjit.ArrayBase: The result of ``x`` multiplied by 2 taken to the power ``n``.

.. topic:: sinh

    Evaluate the hyperbolic sine function.

    This function evaluates the component-wise hyperbolic sine of the input scalar,
    array, or tensor. The function uses a suitable generalization of the operation
    when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic sine of the input

.. topic:: cosh

    Evaluate the hyperbolic cosine function.

    This function evaluates the component-wise hyperbolic cosine of the input
    scalar, array, or tensor. The function uses a suitable generalization of the
    operation when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic cosine of the input

.. topic:: sincosh

    Evaluate both hyperbolic sine and cosine functions at the same time.

    This function simultaneously evaluates the component-wise hyperbolic sine and
    cosine of the input scalar, array, or tensor. This is more efficient than two
    separate calls to :py:func:`drjit.sinh` and :py:func:`drjit.cosh` when both are
    required. The function uses a suitable generalization of the operation when the
    input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Hyperbolic sine and cosine of the input

.. topic:: tanh

    Evaluate the hyperbolic tangent function.

    This function evaluates the component-wise hyperbolic tangent of the input
    scalar, array, or tensor. It uses a suitable generalization of the operation
    when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic tangent of the input

.. topic:: asinh

    Evaluate the hyperbolic arcsine function.

    This function evaluates the component-wise hyperbolic arcsine of the input
    scalar, array, or tensor. It uses a suitable generalization of the operation
    when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arcsine of the input

.. topic:: acosh

    Hyperbolic arccosine approximation.

    This function evaluates the component-wise hyperbolic arccosine of the input
    scalar, array, or tensor. It uses a suitable generalization of the operation
    when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arccosine of the input

.. topic:: atanh

    Evaluate the hyperbolic arctangent function.

    This function evaluates the component-wise hyperbolic arctangent of the input
    scalar, array, or tensor. It uses a suitable generalization of the operation
    when the input is complex-valued.

    The implementation of this function is based on the CEPHES library. See the
    section on :ref:`transcendental function approximations
    <transcendental-accuracy>` for details regarding accuracy.

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        float | drjit.ArrayBase: Hyperbolic arctangent of the input

.. topic:: frexp

    Break the given floating point number into normalized fraction and power of 2

    Args:
        arg (float | drjit.ArrayBase): A Python or Dr.Jit floating point type

    Returns:
        (float, float) | (drjit.ArrayBase, drjit.ArrayBase): Normalized fraction
        ``frac`` on the interval :math:`[\frac{1}{2}, 1)` and an exponent ``exp``
        so that ``frac * 2**(exp + 1)`` equals ``arg``.

.. topic:: fma

    Perform a *fused multiply-addition* (FMA) of the inputs.

    Given arguments ``arg0``, ``arg1``, and ``arg2``, this operation computes
    ``arg0`` * ``arg1`` + ``arg2`` using only one final rounding step. The
    operation is not only more accurate, but also more efficient, since FMA maps to
    a native machine instruction on all platforms targeted by Dr.Jit.

    When the input is complex- or quaternion-valued, the function internally uses
    a complex or quaternion product. In this case, it reduces the number of
    internal rounding steps instead of avoiding them altogether.

    While FMA is traditionally a floating point operation, Dr.Jit also implements
    FMA for integer arrays and maps it onto dedicated instructions provided by the
    backend if possible (e.g. ``mad.lo.*`` for CUDA/PTX).

    Args:
        arg0 (float | drjit.ArrayBase): First multiplication operand
        arg1 (float | drjit.ArrayBase): Second multiplication operand
        arg2 (float | drjit.ArrayBase): Additive operand

    Returns:
        float | drjit.ArrayBase: Result of the FMA operation

.. topic:: zeros

    Return a zero-initialized instance of the desired type and shape.

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

    - A :ref:`PyTree <pytrees>`. In this case, :py:func:`drjit.zeros()` will invoke
      itself recursively to zero-initialize each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case.

    Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
    initialized to ``False`` as opposed to zero.

    The function returns a *literal constant* array that consumes no device memory.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`PyTree <pytrees>`.
        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A zero-initialized instance of type ``dtype``.

.. topic:: ones

    Return an instance of the desired type and shape filled with ones.

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

    - A :ref:`PyTree <pytrees>`. In this case, :py:func:`drjit.ones()` will invoke
      itself recursively to initialize each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case.

    Note that when ``dtype`` refers to a scalar mask or a mask array, it will be
    initialized to ``True`` as opposed to one.

    The function returns a *literal constant* array that consumes no device memory.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`PyTree <pytrees>`.
        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A instance of type ``dtype`` filled with ones.

.. topic:: full

    Return an constant-valued instance of the desired type and shape.

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

    - A :ref:`PyTree <pytrees>`. In this case, :py:func:`drjit.full()` will invoke
      itself recursively to initialize each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case.

    The function returns a *literal constant* array that consumes no device memory.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`PyTree <pytrees>`.
        value (object): An instance of the underlying scalar type
          (``float``/``int``/``bool``, etc.) that will be used to initialize the
          array contents.
        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A instance of type ``dtype`` filled with ``value``

.. topic:: opaque

    Return an *opaque* constant-valued instance of the desired type and shape.

    This function is very similar to :py:func:`drjit.full` in that it creates
    constant-valued instances of various types including (potentially nested)
    Dr.Jit arrays, tensors, and :ref:`PyTrees <pytrees>`. Please refer to the
    documentation of :py:func:`drjit.full` for details on the function signature.
    However, :py:func:`drjit.full` creates *literal constant* arrays, which
    means that Dr.Jit is fully aware of the array contents.

    In contrast, :py:func:`drjit.opaque` produces an *opaque* array backed by a
    representation in device memory.

    .. rubric:: Why is this useful?

    Consider the following snippet, where a complex calculation is parameterized
    by the constant ``1``.

    .. code-block:: python

       from drjit.llvm import Float

       result = complex_function(Float(1), ...) # Float(1) is equivalent to dr.full(Float, 1)
       print(result)

    The ``print()`` statement will cause Dr.Jit to evaluate the queued computation,
    which likely also requires compilation of a new kernel (if that exact pattern
    of steps hasn't been observed before). Kernel compilation is costly and may be
    much slower than the actual computation that needs to be done.

    Suppose we later wish to evaluate the function with a different parameter:

    .. code-block:: python

       result = complex_function(Float(2), ...)
       print(result)

    The constant ``2`` is essentially copy-pasted into the generated program,
    causing a mismatch with the previously compiled kernel that therefore cannot be
    reused. This unfortunately means that we must once more wait a few tens or even
    hundreds of milliseconds until a new kernel has been compiled and uploaded to
    the device.

    This motivates the existence of :py:func:`drjit.opaque`. By making a variable
    opaque to Dr.Jit's tracing mechanism, we can keep constants out of the
    generated program and improve the effectiveness of the kernel cache:

    .. code-block:: python

       # The following lines reuse the compiled kernel regardless of the constant
       value = dr.opqaque(Float, 2)
       result = complex_function(value, ...)
       print(result)

    This function is related to :py:func:`drjit.make_opaque`, which can turn an
    already existing Dr.Jit array, tensor, or :ref:`PyTree <pytrees>` into an
    opaque representation.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`PyTree <pytrees>`.
        value (object): An instance of the underlying scalar type
          (``float``/``int``/``bool``, etc.) that will be used to initialize the
          array contents.
        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: A instance of type ``dtype`` filled with ``value``

.. topic:: empty

    Return an uninitialized Dr.Jit array of the desired type and shape.

    This function can create uninitialized buffers of various types. It should only
    be used in combination with a subsequent call to an operation like
    :py:func:`drjit.scatter()` that fills the array contents with valid data.

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

    - A :ref:`PyTree <pytrees>`. In this case, :py:func:`drjit.empty()` will invoke
      itself recursively to allocate memory for each field of the data structure.

    - A scalar Python type like ``int``, ``float``, or ``bool``. The ``shape``
      parameter is ignored in this case, and the function returns a
      zero-initialized result (there is little point in instantiating uninitialized
      versions of scalar Python types).

    :py:func:`drjit.empty` delays allocation of the underlying buffer until an
    operation tries to read/write the actual array contents.

    Args:
        dtype (type): Desired Dr.Jit array type, Python scalar type, or
          :ref:`PyTree <pytrees>`.
        shape (Sequence[int] | int): Shape of the desired array

    Returns:
        object: An instance of type ``dtype`` with arbitrary/undefined contents.

.. topic:: arange

    This function generates an integer sequence on the interval [``start``,
    ``stop``) with step size ``step``, where ``start`` = 0 and ``step`` = 1 if not
    specified.

    Args:
        dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
          dynamically sized 1D Dr.Jit array such as :py:class:`drjit.scalar.ArrayXu`
          or :py:class:`drjit.cuda.Float`.
        start (int): Start of the interval. The default value is ``0``.
        stop/size (int): End of the interval (not included). The name of this
          parameter differs between the two provided overloads.
        step (int): Spacing between values. The default value is ``1``.

    Returns:
        object: The computed sequence of type ``dtype``.

.. topic:: linspace

    This function generates an evenly spaced floating point sequence of size
    ``num`` covering the interval [``start``, ``stop``].

    Args:
        dtype (type): Desired Dr.Jit array type. The ``dtype`` must refer to a
          dynamically sized 1D Dr.Jit floating point array, such as
          :py:class:`drjit.scalar.ArrayXf` or :py:class:`drjit.cuda.Float`.

        start (float): Start of the interval.

        stop (float): End of the interval.

        num (int): Number of samples to generate.

        endpoint (bool): Should the interval endpoint be included?
          The default is ``True``.

    Returns:
        object: The computed sequence of type ``dtype``.

.. topic:: shape

    Return a tuple describing dimension and shape of the provided Dr.Jit array,
    tensor, or standard sequence type.

    When the input array is *ragged* the function raises a ``RuntimeError``.
    The term ragged refers to an array, whose components have mismatched sizes,
    such as ``[[1, 2], [3, 4, 5]]``. Note that scalar entries (e.g. ``[[1, 2],
    [3]]``) are acceptable, since broadcasting can effectively convert them to
    any size.

    The expressions :py:func:`drjit.shape(arg) <drjit.shape>` and
    :py:func:`arg.shape <drjit.ArrayBase.shape>` are equivalent.

    Args:
        arg (drjit.ArrayBase): an arbitrary Dr.Jit array or tensor

    Returns:
        tuple[int, ...]: A tuple describing the dimension and shape of the
        provided Dr.Jit input array or tensor.

.. topic:: ArrayBase_x

    If ``self`` is a static Dr.Jit array of size 1 (or larger), the property
    ``self.x`` can be used synonymously with ``self[0]``. Otherwise, accessing
    this field will generate a ``RuntimeError``.

    :type: :py:func:`value_t(self) <value_t>`

.. topic:: ArrayBase_y

    If ``self`` is a static Dr.Jit array of size 2 (or larger), the property
    ``self.y`` can be used synonymously with ``self[1]``. Otherwise, accessing
    this field will generate a ``RuntimeError``.

    :type: :py:func:`value_t(self) <value_t>`

.. topic:: ArrayBase_z

    If ``self`` is a static Dr.Jit array of size 3 (or larger), the property
    ``self.z`` can be used synonymously with ``self[2]``. Otherwise, accessing
    this field will generate a ``RuntimeError``.

    :type: :py:func:`value_t(self) <value_t>`

.. topic:: ArrayBase_w

    If ``self`` is a static Dr.Jit array of size 4 (or larger), the property
    ``self.w`` can be used synonymously with ``self[3]``. Otherwise, accessing
    this field will generate a ``RuntimeError``.

    :type: :py:func:`value_t(self) <value_t>`

.. topic:: ArrayBase_real

    If ``self`` is a complex Dr.Jit array, the property ``self.real`` returns the
    real component (as does ``self[0]``). Otherwise, the field returns ``self``.

.. topic:: ArrayBase_imag

    If ``self`` is a complex Dr.Jit array, the property ``self.imag`` returns the
    imaginary component (as does ``self[1]``). Otherwise, it returns a zero-valued
    array of the same type and shape as ``self``.

.. topic:: ArrayBase_T

    This property returns the transpose of ``self``. When the underlying
    array is not a matrix type, it raises a ``TypeError``.

.. topic:: ArrayBase_shape

    This property provides a tuple describing dimension and shape of the
    provided Dr.Jit array or tensor.

    When the input array is *ragged* the function raises a ``RuntimeError``.
    The term ragged refers to an array, whose components have mismatched sizes,
    such as ``[[1, 2], [3, 4, 5]]``. Note that scalar entries (e.g. ``[[1, 2],
    [3]]``) are acceptable, since broadcasting can effectively convert them to
    any size.

    The expressions :py:func:`drjit.shape(arg) <drjit.shape>` and
    :py:func:`arg.shape <drjit.ArrayBase.shape>` are equivalent.

    :type: tuple[int, ...]

.. topic:: ArrayBase_ndim

    This property represents the dimension of the provided Dr.Jit array or tensor.

    :type: int

.. topic:: ArrayBase_array

    This member plays multiple roles:

    - When ``self`` is a tensor, this property returns the storage representation
      of the tensor in the form of a linearized dynamic 1D array.

    - When ``self`` is a special arithmetic object (matrix, quaternion, or complex
      number), ``array`` provides an copy of the same data with ordinary array
      semantics.

    - In all other cases, ``array`` is simply a reference to ``self``.

    :type: :py:func:`array_t(self) <array_t>`

.. topic:: ArrayBase_index

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

    :type: int

.. topic:: ArrayBase_label

    If ``self`` is a *leaf* Dr.Jit array managed by a just-in-time compiled backend
    (i.e, CUDA or LLVM), this property contains a custom label that may be
    associated with the variable. This label is visible graph visualizations, such
    as :py:func:`drjit.graphviz` and :py:func:`drjit.graphviz_ad`. It is also added
    to the generated low-level IR (LLVM, PTX) to aid debugging.

    You may directly assign new labels to this variable or use the
    :py:func:`drjit.set_label` function to label entire data structures (e.g.,
    :ref:`PyTrees <pytrees>`).

    When :py:attr:`drjit.JitFlag.Debug` is set, this field will initially be
    set to the source code location (file + line number) that created variable.

    :type: str | None

.. topic:: ArrayBase_index_ad

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

    :type: int

.. topic:: ArrayBase_grad

    This property can be used to retrieve or set the gradient associated with the
    Dr.Jit array or tensor.

    The expressions ``drjit.grad(arg)`` and ``arg.grad`` are equivalent when
    ``arg`` is a Dr.Jit array/tensor.

    :type: drjit.ArrayBase

.. topic:: uint_array_t

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

.. topic:: int_array_t

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

.. topic:: float_array_t

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

.. topic:: uint32_array_t

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

.. topic:: int32_array_t

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

.. topic:: uint64_array_t

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

.. topic:: int64_array_t

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

.. topic:: float16_array_t

    Converts the provided Dr.Jit array/tensor type into a 16 bit floating point version.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g. :py:class:`drjit.cuda.Array3u`), it
       returns a *16 bit floating point* version (e.g. :py:class:`drjit.cuda.Array3f16`).

    2. When the input is not a type, it returns ``float16_array_t(type(arg))``.

    3. When the input is not a Dr.Jit array or type, the function returns ``half``.

    Args:
        arg (object): An arbitrary Python object

    Returns:
        type: Result of the conversion as described above.

.. topic:: float32_array_t

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

.. topic:: float64_array_t

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

.. topic:: reinterpret_array_t

    Converts the provided Dr.Jit array/tensor type into a
    version with the same element size and scalar type ``type``.

    This function implements the following set of behaviors:

    1. When invoked with a Dr.Jit array *type* (e.g.
    :py:class:`drjit.cuda.Array3f64`), it returns a matching array type with the
    specified scalar type (e.g., :py:class:`drjit.cuda.Array3u64` when ``arg1`` is set
    to `drjit.VarType.UInt64`).

    2. When the input is not a type, it returns ``reinterpret_array_t(type(arg0), arg1)``.

    3. When the input is not a Dr.Jit array or type, the function returns the
       associated Python scalar type.

    Args:
        arg0 (object): An arbitrary Python object
        arg1 (drjit.VarType): The desired scalar type

    Returns:
        type: Result of the conversion as described above.

.. topic:: detached_t

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

.. topic:: expr_t

    Computes the type of an arithmetic expression involving the provided Dr.Jit
    arrays (instances or types), or builtin Python objects.

    An exception will be raised when an invalid combination of types is provided.

    For instance, this function can be used to compute the return type of the
    addition of several Dr.Jit array:

    .. code-block:: python

        a = drjit.llvm.Float(1.0)
        b = drjit.llvm.Array3f(1, 2, 3)
        c = drjit.llvm.ArrayXf(4, 5, 6)

        # type(a + b + c) == dr.expr_t(a, b, c) == drjit.llvm.ArrayXf

    Args:
        *args (tuple): A variable-length list of Dr.Jit arrays, builtin Python
              objects, or types.

    Returns:
        type: Result type of an arithmetic expression involving the provided variables.

.. topic:: slice_index

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
                      such as :py:class:`drjit.scalar.ArrayXu` or
                      :py:class:`drjit.cuda.UInt`.

        shape (tuple[int, ...]): The shape of the tensor to be sliced.

        indices (tuple[int|slice|ellipsis|None|dr.ArrayBase, ...]):
            A set of indices used to slice the tensor. Its entries can be ``slice``
            instances, integers, integer arrays, ``...`` (ellipsis) or ``None``.

    Returns:
        tuple[tuple[int, ...], drjit.ArrayBase]: Tuple consisting of the output array
        shape and a flattened unsigned integer array of type ``dtype`` containing
        element indices.

.. topic:: gather

    Gather values from a flat array or nested data structure.

    This function performs a *gather* (i.e., indirect memory read) from ``source``
    at position ``index``. It expects a ``dtype`` argument and will return an
    instance of this type. The optional ``active`` argument can be used to disable
    some of the components, which is useful when not all indices are valid; the
    corresponding output will be zero in this case.

    This operation can be used in the following different ways:

    1. When ``dtype`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
       this operation implements a parallelized version of the Python array
       indexing expression ``source[index]`` with optional masking. Example:

       .. code-block:: python

           source = dr.cuda.Float([...])
           index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
           result = dr.gather(dtype=type(source), source=source, index=index)

    2. When ``dtype`` is a more complex type (e.g. a nested Dr.Jit array or :ref:`PyTree
       <pytrees>`), the behavior depends:

       - When ``type(source)`` matches ``dtype``, the gather operation threads
         through entries and invokes itself recursively. For example, the
         gather operation in

         .. code-block:: python

            result = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            result = dr.gather(dr.cuda.Array3f, source, index)

         is equivalent to

         .. code-block:: python

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


         .. code-block:: python

            source = dr.cuda.Float([...])
            index = dr.cuda.UInt([...])
            result = dr.gather(dr.cuda.Array3f, source, index)

         and is equivalent to

         .. code-block:: python

            result = dr.cuda.Vector3f(
                dr.gather(dr.cuda.Float, source, index*3 + 0),
                dr.gather(dr.cuda.Float, source, index*3 + 1),
                dr.gather(dr.cuda.Float, source, index*3 + 2))

    .. danger::

        The indices provided to this operation are unchecked by default. Attempting
        to read beyond the end of the ``source`` array is undefined behavior and
        may crash the application, unless such reads are explicitly disabled via the
        ``active`` parameter. Negative indices are not permitted.

        If *debug mode* is enabled via the :py:attr:`drjit.JitFlag.Debug` flag,
        Dr.Jit will insert range checks into the program. These checks disable
        out-of-bound reads and furthermore report warnings to identify problematic
        source locations:

        .. code-block:: pycon
           :emphasize-lines: 2-3

           >>> dr.gather(dtype=UInt, source=UInt(1, 2, 3), index=UInt(0, 1, 100))
           drjit.gather(): out-of-bounds read from position 100 in an array↵
           of size 3. (<stdin>:2)

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
          if another type is provided. The default is ``True``.

        mode (drjit.ReduceMode): The reverse-mode derivative of a gather is
          an atomic scatter-reduction. The execution of such atomics can be
          rather performance-sensitive (see the discussion of
          :py:class:`drjit.ReduceMode` for details), hence Dr.Jit offers a few
          different compilation strategies to realize them. Specifying this
          parameter selects a strategy for the derivative of a particular
          gather operation. The default is :py:attr:`drjit.ReduceMode.Auto`.

.. topic:: scatter

    Scatter values into a flat array or nested data structure.

    This operation performs a *scatter* (i.e., indirect memory write) of the
    ``value`` parameter to the ``target`` array at position ``index``. The optional
    ``active`` argument can be used to disable some of the individual write
    operations, which is useful when not all provided values or indices are valid.

    This operation can be used in the following different ways:

    1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
       this operation implements a parallelized version of the Python array
       indexing expression ``target[index] = value`` with optional masking. Example:

       .. code-block:: python

          target = dr.empty(dr.cuda.Float, 1024*1024)
          value = dr.cuda.Float([...])
          index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
          dr.scatter(target, value=value, index=index)

    2. When ``target`` is a more complex type (e.g. a nested Dr.Jit array or
       :ref:`PyTree <pytrees>`), the behavior depends:

       - When ``target`` and ``value`` are of the same type, the scatter operation
         threads through entries and invokes itself recursively. For example, the
         scatter operation in

         .. code-block:: python

            target = dr.cuda.Array3f(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter(target, value, index)

         is equivalent to

         .. code-block:: python

            dr.scatter(target.x, value.x, index)
            dr.scatter(target.y, value.y, index)
            dr.scatter(target.z, value.z, index)

         A similar recursive traversal is used for other kinds of
         sequences, mappings, and custom data structures.

       - Otherwise, the operation flattens the ``value`` array and writes it using
         C-style ordering with a suitably modified ``index``. For example, the
         scatter below writes 3D vectors into a 1D array.

         .. code-block:: python

            target = dr.cuda.Float(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter(target, value, index)

         and is equivalent to

         .. code-block:: python

            dr.scatter(target, value.x, index*3 + 0)
            dr.scatter(target, value.y, index*3 + 1)
            dr.scatter(target, value.z, index*3 + 2)

    .. danger::

        The indices provided to this operation are unchecked by default.
        Out-of-bound writes are considered undefined behavior and may crash the
        application (unless they are disabled via the ``active`` parameter).
        Negative indices are not permitted.

        If *debug mode* is enabled via the :py:attr:`drjit.JitFlag.Debug` flag,
        Dr.Jit will insert range checks into the program. These will catch
        out-of-bound writes and print an error message identifying the responsible
        line of code.

        Dr.Jit makes no guarantees about the expected behavior when a scatter
        operation has *conflicts*, i.e., when a specific position is written
        multiple times by a single :py:func:`drjit.scatter()` operation.

    Args:
        target (object): The object into which data should be written (typically a
          1D Dr.Jit array, but other variations are possible as well, see the
          description above.)

        value (object): The values to be written (typically of type ``type(target)``,
          but other variations are possible as well, see the description above.)
          Dr.Jit will attempt an implicit conversion if the input is not an
          array type.

        index (object): a 1D dynamic unsigned 32-bit Dr.Jit array (e.g.,
          :py:class:`drjit.scalar.ArrayXu` or :py:class:`drjit.cuda.UInt`)
          specifying gather indices. Dr.Jit will attempt an implicit conversion if
          another type is provided.

        active (object): an optional 1D dynamic Dr.Jit mask array (e.g.,
          :py:class:`drjit.scalar.ArrayXb` or :py:class:`drjit.cuda.Bool`)
          specifying active components. Dr.Jit will attempt an implicit conversion
          if another type is provided. The default is ``True``.

.. topic:: scatter_add

    Atomically add values to a flat array or nested data structure.

    This function is equivalent to
    :py:func:`drjit.scatter_reduce(drjit.ReduceOp.Add, ...) <scatter_reduce>` and
    exists for reasons of convenience. Please refer to
    :py:func:`drjit.scatter_reduce` for details on atomic scatter-reductions.

.. topic:: scatter_reduce

    Atomically update values in a flat array or nested data structure.

    This function performs an atomic *scatter-reduction*, which is a
    read-modify-write operation that applies one of several possible mathematical
    functions to selected entries of an array. The following are supported:

    - :py:attr:`drjit.ReduceOp.Add`: ``a=a+b``.
    - :py:attr:`drjit.ReduceOp.Max`: ``a=max(a, b)``.
    - :py:attr:`drjit.ReduceOp.Min`: ``a=min(a, b)``.
    - :py:attr:`drjit.ReduceOp.Or`: ``a=a | b`` (integer arrays only).
    - :py:attr:`drjit.ReduceOp.And`: ``a=a & b`` (integer arrays only).

    Here, ``a`` refers to an entry of ``target`` selected by ``index``, and ``b``
    denotes the associated element of ``value``. The operation resolves potential
    conflicts arising due to the parallel execution of this operation.

    The optional ``active`` argument can be used to disable some of the updates,
    e.g., when not all provided values or indices are valid.

    Atomic additions are subject to non-deterministic rounding errors.  The
    reason for this is that IEEE-754 addition are non-commutative. The execution
    order is scheduling-dependent, which can lead to small variations across
    program runs.

    Atomic scatter-reductions can have a *significant* detrimental impact on
    performance. When many threads in a parallel computation attempt to modify the
    same element, this can lead to *contention*---essentially a fight over which
    part of the processor **owns** the associated memory region, which can slow
    down a computation by many orders of magnitude. Dr.Jit provides several
    different compilation strategies to reduce these costs, which can be selected
    via the ``mode`` parameter. The documentation of :py:class:`drjit.ReduceMode`
    provides more detail and performance plots.

    .. _scatter_reduce_supported:

    .. rubric:: Backend support

    Many combinations of reductions and variable types are not supported. Some
    combinations depend on the *compute capability* (CC) of the underlying CUDA
    device or on the *LLVM version* (LV) and the host architecutre (AMD64,
    x86_64). The following matrices display the level of support.

    For CUDA:

    .. list-table::

       * - Reduction
         - ``Bool``
         - ``[U]Int{32,64}``
         - ``Float16``
         - ``Float32``
         - ``Float64``
       * - :py:attr:`ReduceOp.Identity`
         - ✅
         - ✅
         - ✅
         - ✅
         - ✅
       * - :py:attr:`ReduceOp.Add`
         - ❌
         - ✅
         - ⚠️  CC≥60
         - ✅
         - ⚠️  CC≥60
       * - :py:attr:`ReduceOp.Mul`
         - ❌
         - ❌
         - ❌
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.Min`
         - ❌
         - ❌
         - ⚠️  CC≥90
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.Max`
         - ❌
         - ❌
         - ⚠️  CC≥90
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.And`
         - ❌
         - ✅
         - ❌
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.Or`
         - ❌
         - ✅
         - ❌
         - ❌
         - ❌


    For LLVM:

    .. list-table::

       * - Reduction
         - ``Bool``
         - ``[U]Int{32,64}``
         - ``Float16``
         - ``Float32``
         - ``Float64``
       * - :py:attr:`ReduceOp.Identity`
         - ✅
         - ✅
         - ✅
         - ✅
         - ✅
       * - :py:attr:`ReduceOp.Add`
         - ❌
         - ✅
         - ⚠️  LV≥16
         - ✅
         - ✅
       * - :py:attr:`ReduceOp.Mul`
         - ❌
         - ❌
         - ❌
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.Min`
         - ❌
         - ⚠️  LV≥15
         - ⚠️  LV≥16, ARM64
         - ⚠️  LV≥15
         - ⚠️  LV≥15
       * - :py:attr:`ReduceOp.Max`
         - ❌
         - ⚠️  LV≥15
         - ⚠️  LV≥16, ARM64
         - ⚠️  LV≥15
         - ⚠️  LV≥15
       * - :py:attr:`ReduceOp.And`
         - ❌
         - ✅
         - ❌
         - ❌
         - ❌
       * - :py:attr:`ReduceOp.Or`
         - ❌
         - ✅
         - ❌
         - ❌
         - ❌

    The function raises an exception when the operation is not supported by the backend.

    .. rubric:: Scatter-reducing nested types

    This operation can be used in the following different ways:

    1. When ``target`` is a 1D Dr.Jit array like :py:class:`drjit.llvm.ad.Float`,
       this operation implements a parallelized version of the Python array
       indexing expression ``target[index] = op(target[index], value)`` with
       optional masking. Example:

       .. code-block:: python

          target = dr.zeros(dr.cuda.Float, 1024*1024)
          value = dr.cuda.Float([...])
          index = dr.cuda.UInt([...]) # Note: negative indices are not permitted
          dr.scatter_reduce(dr.ReduceOp.Add, target, value=value, index=index)

    2. When ``target`` is a more complex type (e.g. a nested Dr.Jit array or
       :ref:`PyTree <pytrees>`), the behavior depends:

       - When ``target`` and ``value`` are of the same type, the scatter-reduction
         threads through entries and invokes itself recursively. For example, the
         scatter operation in

         .. code-block:: python

            op = dr.ReduceOp.Add
            target = dr.cuda.Array3f(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter_reduce(op, target, value, index)

         is equivalent to

         .. code-block:: python

            dr.scatter_reduce(op, target.x, value.x, index)
            dr.scatter_reduce(op, target.y, value.y, index)
            dr.scatter_reduce(op, target.z, value.z, index)

         A similar recursive traversal is used for other kinds of
         sequences, mappings, and custom data structures.

       - Otherwise, the operation flattens the ``value`` array and writes it using
         C-style ordering with a suitably modified ``index``. For example, the
         scatter-reduction below writes 3D vectors into a 1D array.

         .. code-block:: python

            op = dr.ReduceOp.Add
            target = dr.cuda.Float(...)
            value = dr.cuda.Array3f(...)
            index = dr.cuda.UInt([...])
            dr.scatter_reduce(op, target, value, index)

         and is equivalent to

         .. code-block:: python

            dr.scatter_reduce(op, target, value.x, index*3 + 0)
            dr.scatter_reduce(op, target, value.y, index*3 + 1)
            dr.scatter_reduce(op, target, value.z, index*3 + 2)

    .. danger::

        The indices provided to this operation are unchecked by default.
        Out-of-bound writes are considered undefined behavior and may crash the
        application (unless they are disabled via the ``active`` parameter).
        Negative indices are not permitted.

        If *debug mode* is enabled via the :py:attr:`drjit.JitFlag.Debug` flag,
        Dr.Jit will insert range checks into the program. These will catch
        out-of-bound writes and print an error message identifying the responsible
        line of code.

        Dr.Jit makes no guarantees about the relative ordering of atomic operations
        when a :py:func:`drjit.scatter_reduce()` writes to the same element
        multiple times. Combined with the non-associate nature of floating point
        operations, concurrent writes will generally introduce non-deterministic
        rounding error.

    Args:
        op (drjit.ReduceOp): Specifies the type of update that should be performed.

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
          if another type is provided. The default is ``True``.

        mode (drjit.ReduceMode):  Dr.Jit offers several different strategies to
          implement atomic scatter-reductions that can be selected via this
          parameter. They achieve different best/worst case performance and, in
          the case of :py:attr:`drjit.ReduceMode.Expand`, involve additional
          memory storage overheads. The default is
          :py:attr:`drjit.ReduceMode.Auto`.

.. topic:: ravel

    Convert the input into a contiguous flat array.

    This operation takes a Dr.Jit array, typically with some static and some
    dynamic dimensions (e.g., :py:class:`drjit.cuda.Array3f` with shape
    `3xN`), and converts it into a flattened 1D dynamically sized array (e.g.,
    :py:class:`drjit.cuda.Float`) using either a C or Fortran-style ordering
    convention.

    It can also convert Dr.Jit tensors into a flat representation, though only
    C-style ordering is supported in this case.

    Internally, :py:func:`drjit.ravel()` performs a series of calls to
    :py:func:`drjit.scatter()` to suitably reorganize the array contents.

    For example,

    .. code-block:: python

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

.. topic:: unravel

    Load a sequence of Dr.Jit vectors/matrices/etc. from a contiguous flat array.

    This operation implements the inverse of :py:func:`drjit.ravel()`. In contrast
    to :py:func:`drjit.ravel()`, it requires one additional parameter (``dtype``)
    specifying type of the return value. For example,

    .. code-block:: python

        x = dr.cuda.Float([1, 2, 3, 4, 5, 6])
        y = dr.unravel(dr.cuda.Array3f, x, order=...)

    will produce an array of two 3D vectors with different contents depending
    on the indexing convention:

    - ``[1, 2, 3]`` and ``[4, 5, 6]`` when unraveled with ``order='F'`` (the
      default for Dr.Jit arrays), and
    - ``[1, 3, 5]`` and ``[2, 4, 6]`` when unraveled with ``order='C'``

    Internally, :py:func:`drjit.unravel()` performs a series of calls to
    :py:func:`drjit.gather()` to suitably reorganize the array contents.

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

.. topic:: schedule

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

    .. code-block:: python

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

    .. code-block:: python

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
    input arguments. It recursively traverses PyTrees :ref:`PyTrees <pytrees>`
    (sequences, mappings, custom data structures, etc.).

    During recursion, the function gathers all unevaluated Dr.Jit arrays. Evaluated
    arrays and incompatible types are ignored. Multiple variables can be
    equivalently scheduled with a single :py:func:`drjit.schedule()` call or a
    sequence of calls to :py:func:`drjit.schedule()`. Variables that are garbage
    collected between the original :py:func:`drjit.schedule()` call and the next
    kernel launch are ignored and will not be stored in memory.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances or
             :ref:`PyTrees <pytrees>` (they will be recursively traversed to
             all differentiable variables.)

    Returns:
        bool: ``True`` if a variable was scheduled, ``False`` if the operation did
        not do anything.

.. topic:: eval

    Evaluate the provided JIT variable(s)

    Dr.Jit automatically evaluates variables as needed, hence it is usually not
    necessary to call this function explicitly. That said, explicit evaluation may
    sometimes improve performance---refer to the documentation of
    :py:func:`drjit.schedule()` for an example of such a use case.

    :py:func:`drjit.eval()` invokes Dr.Jit's LLVM or CUDA backends to compile and
    then execute a kernel containing the all steps that are needed to evaluate the
    specified variables, which will turn them into a memory-based representation.
    The generated kernel(s) will also include computation that was previously
    scheduled via :py:func:`drjit.schedule()`. In fact, :py:func:`drjit.eval()`
    internally calls :py:func:`drjit.schedule()`, as

    .. code-block:: python

        dr.eval(arg_1, arg_2, ...)

    is equivalent to

    .. code-block:: python

        dr.schedule(arg_1, arg_2, ...)
        dr.eval()

    This function accepts a variable-length keyword argument and processes all
    input arguments. It recursively traverses PyTrees :ref:`PyTrees <pytrees>`
    (sequences, mappings, custom data structures, etc.).

    During this recursive traversal, the function collects all unevaluated Dr.Jit
    arrays, while ignoring previously evaluated arrays along and non-array types.
    The function also does not evaluate *literal constant* arrays (this refers to
    potentially large arrays that are entirely uniform), as this is generally not
    wanted. Use the function :py:func:`drjit.make_opaque` if you wish to evaluate
    literal constant arrays as well.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances or
          :ref:`PyTrees <pytrees>` (they will be recursively traversed to discover
          all Dr.Jit arrays.)

    Returns:
        bool: ``True`` if a variable was evaluated, ``False`` if the operation did
        not do anything.

.. topic:: make_opaque

    Forcefully evaluate arrays (including literal constants).

    This function implements a more drastic version of :py:func:`drjit.eval` that
    additionally converts literal constant arrays into evaluated (device
    memory-based) representations.

    It is related to the function :py:func:`drjit.opaque` that can be used to
    directly construct such opaque arrays. Please see the documentation of this
    function regarding the rationale of making array contents opaque to Dr.Jit's
    symbolic tracing mechanism.

    Args:
        *args (tuple): A variable-length list of Dr.Jit array instances or
          :ref:`PyTrees <pytrees>` (they will be recursively traversed to discover
          all Dr.Jit arrays.)

.. topic:: dlpack_device

    Returns a tuple containing the DLPack device type and device ID associated with
    the given array

.. topic:: dlpack

    Returns a DLPack capsule representing the data in this array.

    This operation may potentially perform a copy. For example, nested arrays like
    :py:class:`drjit.llvm.Array3f` or :py:class:`drjit.cuda.Matrix4f` need to be
    rearranged into a contiguous memory representation before they can be exposed.

    In other case, e.g. for :py:class:`drjit.llvm.Float`,
    :py:class:`drjit.scalar.Array3f`, or :py:class:`drjit.scalar.ArrayXf`, the data
    is already contiguous and a zero-copy approach is used instead.

.. topic:: array

    Returns a NumPy array representing the data in this array.

    This operation may potentially perform a copy. For example, nested arrays like
    :py:class:`drjit.llvm.Array3f` or :py:class:`drjit.cuda.Matrix4f` need to be
    rearranged into a contiguous memory representation before they can be wrapped.

    In other case, e.g. for :py:class:`drjit.llvm.Float`,
    :py:class:`drjit.scalar.Array3f`, or :py:class:`drjit.scalar.ArrayXf`, the data
    is already contiguous and a zero-copy approach is used instead.

.. topic:: torch

    Returns a PyTorch tensor representing the data in this array.

    For :ref:`flat arrays <flat_arrays>` and :ref:`tensors <tensors>`, Dr.Jit
    performs a *zero-copy* conversion, which means that the created tensor provides
    a *view* of the same data that will reflect later modifications to the Dr.Jit
    array. :ref:`Nested arrays <nested_arrays>` require a temporary copy to rearrange
    data into a compatible form.

    .. warning::

       This operation converts the numerical representation but does *not* embed the
       resulting tensor into the automatic differentiation graph of the other
       framework. This means that gradients won't correctly propagate through
       programs combining multiple frameworks. Take a look at the function
       :py:func:`drjit.wrap` for further information on how to accomplish this.

.. topic:: jax

    Returns a JAX tensor representing the data in this array.

    For :ref:`flat arrays <flat_arrays>` and :ref:`tensors <tensors>`, Dr.Jit
    performs a *zero-copy* conversion, which means that the created tensor provides
    a *view* of the same data that will reflect later modifications to the Dr.Jit
    array. :ref:`Nested arrays <nested_arrays>` require a temporary copy to rearrange
    data into a compatible form.

    .. warning::
       This operation converts the numerical representation but does *not* embed the
       resulting tensor into the automatic differentiation graph of the other
       framework. This means that gradients won't correctly propagate through
       programs combining multiple frameworks. Take a look at the function
       :py:func:`drjit.wrap` for further information on how to accomplish this.

.. topic:: tf

    Returns a TensorFlow tensor representing the data in this array.

    For :ref:`flat arrays <flat_arrays>` and :ref:`tensors <tensors>`, Dr.Jit
    performs a *zero-copy* conversion, which means that the created tensor provides
    a *view* of the same data that will reflect later modifications to the Dr.Jit
    array. :ref:`Nested arrays <nested_arrays>` require a temporary copy to rearrange
    data into a compatible form.

    .. warning::

       This operation converts the numerical representation but does *not* embed the
       resulting tensor into the automatic differentiation graph of the other
       framework. This means that gradients won't correctly propagate through
       programs combining multiple frameworks. Take a look at the function
       :py:func:`drjit.wrap` for further information on how to accomplish this.

.. topic:: detach

    Transforms the input variable into its non-differentiable version (*detaches* it
    from the AD computational graph).

    This function supports arbitrary Dr.Jit arrays/tensors and :ref:`PyTrees
    <pytrees>` as input. In the latter case, it applies the transformation
    recursively. When the input variable is not a PyTree or Dr.Jit array, it is
    returned as it is.

    While the type of the returned array is preserved by default, it is possible to
    set the ``preserve_type`` argument to false to force the returned type to be
    non-differentiable. For example, this will convert an array of type
    :py:class:`drjit.llvm.ad.Float` into one of type :py:class:`drjit.llvm.Float`.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        preserve_type (bool): Defines whether the returned variable should preserve
            the type of the input variable.
    Returns:
        object: The detached variable.

.. topic:: set_grad_enabled

    Enable or disable gradient tracking on the provided variables.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor,
            :ref:`PyTree <pytrees>`, sequence, or mapping.

        value (bool): Defines whether gradient tracking should be enabled or
            disabled.

.. topic:: enable_grad

    Enable gradient tracking for the provided variables.

    This function accepts a variable-length keyword argument and processes all
    input arguments. It recursively traverses PyTrees :ref:`PyTrees <pytrees>`
    (sequences, mappings, custom data structures, etc.).

    During this recursive traversal, the function enables gradient tracking for all
    encountered Dr.Jit arrays. Variables of other types are ignored.

    Args:
        *args (tuple): A variable-length list of Dr.Jit arrays/tensors or
            :ref:`PyTrees <pytrees>`.

.. topic:: disable_grad

    Disable gradient tracking for the provided variables.

    This function accepts a variable-length keyword argument and processes all
    input arguments. It recursively traverses PyTrees :ref:`PyTrees <pytrees>`
    (sequences, mappings, custom data structures, etc.).

    During this recursive traversal, the function disables gradient tracking for all
    encountered Dr.Jit arrays. Variables of other types are ignored.

    Args:
        *args (tuple): A variable-length list of Dr.Jit arrays/tensors or
            :ref:`PyTrees <pytrees>`.

.. topic:: grad_enabled

    Return whether gradient tracking is enabled on any of the given variables.

    Args:
        *args (tuple): A variable-length list of Dr.Jit arrays/tensors instances or
          :ref:`PyTrees <pytrees>`. The function recursively traverses them to
          all differentiable variables.

    Returns:
        bool: ``True`` if any of the input variables has gradient tracking enabled,
        ``False`` otherwise.

.. topic:: grad

    Return the gradient value associated to a given variable.

    When the variable doesn't have gradient tracking enabled, this function returns ``0``.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor or :ref:`PyTree <pytrees>`.

        preserve_type (bool): Should the operation preserve the input type in the
            return value? (This is the default). Otherwise, Dr.Jit will, e.g.,
            return a type of `drjit.cuda.Float` for an input of type
            `drjit.cuda.ad.Float`.

    Returns:
        object: the gradient value associated to the input variable.

.. topic:: set_grad

    Set the gradient associated with the provided variable.

    This operation internally decomposes into two sub-steps:

    .. code-block:: python

       dr.clear_grad(target)
       dr.accum_grad(target, source)

    When ``source`` is not of the same type as ``target``, Dr.Jit will try to broadcast
    its contents into the right shape.

    Args:
        target (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        source (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

.. topic:: accum_grad

    Accumulate the contents of one variable into the gradient of another variable.

    When ``source`` is not of the same type as ``target``, Dr.Jit will try to broadcast
    its contents into the right shape.

    Args:
        target (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        source (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

.. topic:: clear_grad

    Clear the gradient of the given variable.

    Args:
        arg (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

.. topic:: replace_grad

    Replace the gradient value of ``arg0`` with the one of ``arg1``.

    This is a relatively specialized operation to be used with care when
    implementing advanced automatic differentiation-related features.

    One example use would be to inform Dr.Jit that there is a better way to compute
    the gradient of a particular expression than what the normal AD traversal of
    the primal computation graph would yield.

    The function promotes and broadcasts ``arg0`` and ``arg1`` if they are not of the
    same type.

    Args:
        arg0 (object): An arbitrary Dr.Jit array, tensor, Python arithmetic type, or :ref:`PyTree <pytrees>`.

        arg1 (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

    Returns:
        object: a new Dr.Jit array combining the *primal* value of ``arg0`` and the
        derivative of ``arg1``.

.. topic:: enqueue

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
        b = f(a) # some computation involving 'a'

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

    The same naturally also works in the reverse direction. Dr.Jit provides a
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
            :ref:`PyTree <pytrees>`.

.. topic:: traverse

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
    :py:class:`drjit.ADFlag` via the ``flags`` parameter.

    Args:
        mode (drjit.ADMode): Specifies the direction in which gradients should be
            propgated. :py:attr:`drjit.ADMode.Forward`
            and:py:attr:`drjit.ADMode.Backward` refer to forward and backward
            traversal.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph are cleared
            during traversal. The default value is :py:attr:`drjit.ADFlag.Default`.

.. topic:: forward_from

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

    When :py:attr:`drjit.JitFlag.SymbolicCalls` is set, the implementation
    raises an exception when the provided array does not support gradient
    tracking, or when gradient tracking was not previously enabled via
    :py:func:`drjit.enable_grad()`, as this generally indicates the presence of
    a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoGrad` flag (e.g. by
    passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

    Args:
        args (object): A Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
            during traversal, and whether or not to fail when the input is not
            differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

.. topic:: forward_to

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

    When :py:attr:`drjit.JitFlag.SymbolicCalls` is set, the implementation
    raises an exception when the provided array does not support gradient
    tracking, or when gradient tracking was not previously enabled via
    :py:func:`drjit.enable_grad()`, as this generally indicates the presence of
    a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoGrad` flag (e.g. by
    passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

    Args:
        *args (tuple): A variable-length list of Dr.Jit differentiable array, tensors,
            or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
            during traversal, and whether or not to fail when the input is not
            differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

    Returns:
        object: the gradient value(s) associated with ``*args`` following the
        traversal.

.. topic:: forward

    Forward-propagate gradients from the provided Dr.Jit array or tensor

    This function is an alias of :py:func:`drjit.forward_from()`. Please refer to
    the documentation of this function.

    Args:
        args (object): A Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph are cleared
            during traversal. The default value is :py:attr:`drjit.ADFlag.Default`.

.. topic:: backward_from

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

    When :py:attr:`drjit.JitFlag.SymbolicCalls` is set, the implementation
    raises an exception when the provided array does not support gradient
    tracking, or when gradient tracking was not previously enabled via
    :py:func:`drjit.enable_grad()`, as this generally indicates the presence of
    a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoGrad` flag (e.g. by
    passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

    Args:
        args (object): A Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
            during traversal, and whether or not to fail when the input is not
            differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

.. topic:: backward_to

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

    When :py:attr:`drjit.JitFlag.SymbolicCalls` is set, the implementation
    raises an exception when the provided array does not support gradient
    tracking, or when gradient tracking was not previously enabled via
    :py:func:`drjit.enable_grad()`, as this generally indicates the presence of
    a bug. Specify the :py:attr:`drjit.ADFlag.AllowNoGrad` flag (e.g. by
    passing ``flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad``) to the function.

    Args:
        *args (tuple): A variable-length list of Dr.Jit differentiable array, tensors,
            or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
            during traversal, and whether or not to fail when the input is not
            differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

    Returns:
        object: the gradient value(s) associated with ``*args`` following the
        traversal.

.. topic:: backward

    Backpropgate gradients from the provided Dr.Jit array or tensor.

    This function is an alias of :py:func:`drjit.backward_from()`. Please refer to
    the documentation of this function.

    Args:
        args (object): A Dr.Jit array, tensor, or :ref:`PyTree <pytrees>`.

        flags (drjit.ADFlag | int): Controls what parts of the AD graph to clear
            during traversal, and whether or not to fail when the input is not
            differentiable. The default value is :py:attr:`drjit.ADFlag.Default`.

.. topic:: graphviz

    Return a GraphViz diagram describing registered JIT variables and their connectivity.

    This function returns a representation of the computation graph underlying the
    Dr.Jit just-in-time compiler, which is separate from the automatic
    differentiation layer. See the :py:func:`graphviz_ad()` function to visualize
    the computation graph of the latter.

    Run ``dr.graphviz().view()`` to open up a PDF viewer that shows the resulting
    output in a separate window.

    The function depends on the ``graphviz`` Python package when
    ``as_string=False`` (the default).

    Args:
        as_string (bool): if set to ``True``, the function will return raw GraphViz markup
            as a string. (Default: ``False``)

    Returns:
        object: GraphViz object or raw markup.

.. topic:: graphviz_ad

    Return a GraphViz diagram describing variables registered with the automatic
    differentiation layer, as well as their connectivity.

    This function returns a representation of the computation graph underlying the
    Dr.Jit AD layer, which one architectural layer above the just-in-time compiler.
    See the :py:func:`graphviz()` function to visualize the computation graph of
    the latter.

    Run ``dr.graphviz_ad().view()`` to open up a PDF viewer that shows the
    resulting output in a separate window.

    The function depends on the ``graphviz`` Python package when
    ``as_string=False`` (the default).

    Args:
        as_string (bool): if set to ``True``, the function will return raw GraphViz markup
            as a string. (Default: ``False``)

    Returns:
        object: GraphViz object or raw markup.

.. topic:: whos

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
        None | str: a human-readable list (if requested).

.. topic:: whos_ad

    Return/print a list of live variables registered with the automatic differentiation layer.

    This function provides information about the set of variables that are
    currently registered with the Dr.Jit automatic differentiation layer,
    which one architectural layer above the just-in-time compiler.
    See the :py:func:`whos()` function to obtain information about
    the latter.

    Args:
        as_string (bool): if set to ``True``, the function will return the list in
            string form. Otherwise, it will print directly onto the console and return
            ``None``. (Default: ``False``)

    Returns:
        None | str: a human-readable list (if requested).

.. topic:: suspend_grad

    Context manager for temporally suspending derivative tracking.

    Dr.Jit's AD layer keeps track of a set of variables for which derivative
    tracking is currently enabled. Using this context manager is it possible to
    define a scope in which variables will be subtracted from that set, thereby
    controlling what derivative terms shouldn't be generated in that scope.

    The variables to be subtracted from the current set of enabled variables can be
    provided as function arguments. If none are provided, the scope defined by this
    context manager will temporally disable all derivative tracking.

    .. code-block:: python

        a = dr.llvm.ad.Float(1.0)
        b = dr.llvm.ad.Float(2.0)
        dr.enable_grad(a, b)

        with suspend_grad(): # suspend all derivative tracking
            c = a + b

        assert not dr.grad_enabled(c)

        with suspend_grad(a): # only suspend derivative tracking on 'a'
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

    The optional ``when`` boolean keyword argument can be defined to specified a
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
            instances or :ref:`PyTrees <pytrees>`. The function will recursively
            traverse them to all differentiable variables.

        when (bool): An optional Python boolean determining whether to suspend
          derivative tracking.

.. topic:: resume_grad

    Context manager for temporally resume derivative tracking.

    Dr.Jit's AD layer keeps track of a set of variables for which derivative
    tracking is currently enabled. Using this context manager is it possible to
    define a scope in which variables will be added to that set, thereby controlling
    what derivative terms should be generated in that scope.

    The variables to be added to the current set of enabled variables can be
    provided as function arguments. If none are provided, the scope defined by this
    context manager will temporally resume derivative tracking for all variables.

    .. code-block:: python

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

    The optional ``when`` boolean keyword argument can be defined to specified a
    condition determining whether to resume the tracking of derivatives or not.

    .. code-block:: python

        a = dr.llvm.ad.Float(1.0)
        dr.enable_grad(a)

        cond = condition()

        with suspend_grad():
            with resume_grad(when=cond):
                b = 4.0 * a

        assert dr.grad_enabled(b) == cond

    Args:
        *args (tuple): A variable-length list of differentiable Dr.Jit array
            instances or :ref:`PyTrees <pytrees>`. The function will recursively
            traverse them to all differentiable variables.

        when (bool): An optional Python boolean determining whether to resume
          derivative tracking.

.. topic:: isolate_grad

    Context manager to temporarily isolate outside world from AD traversals.

    Dr.Jit provides isolation boundaries to postpone AD traversals steps leaving a
    specific scope. For instance this function is used internally to implement
    differentiable loops and polymorphic calls.

.. topic:: has_backend

    Check if the specified Dr.Jit backend was successfully initialized.

.. topic:: set_label

    Assign a label to the provided Dr.Jit array.

    This can be helpful to identify computation in GraphViz output (see
    :py:func:`drjit.graphviz`, :py:func:`graphviz_ad`).

    The operations assumes that the array is tracked by the just-in-time compiler.
    It has no effect on unsupported inputs (e.g., arrays from the ``drjit.scalar``
    package). It recurses through :ref:`PyTrees <pytrees>` (tuples, lists,
    dictionaries, custom data structures) and appends names (indices, dictionary
    keys, field names) separated by underscores to uniquely identify each element.

    The following ``**kwargs``-based shorthand notation can be used to assign
    multiple labels at once:

    .. code-block:: python

       set_label(x=x, y=y)

    Args:
        *arg (tuple): a Dr.Jit array instance and its corresponding label ``str`` value.

        **kwarg (dict): A set of (keyword, object) pairs.

.. topic:: ADMode

    Enumeration to distinguish different types of primal/derivative computation.

    See also :py:func:`drjit.enqueue()`, :py:func:`drjit.traverse()`.

.. topic:: ADMode_Primal

    Primal/original computation without derivative tracking. Note that this
    is *not* a valid input to Dr.Jit AD routines, but it is sometimes useful
    to have this entry when to indicate to a computation that derivative
    propagation should not be performed.

.. topic:: ADMode_Forward

    Propagate derivatives in forward mode (from inputs to outputs)

.. topic:: ADMode_Backward

    Propagate derivatives in backward/reverse mode (from outputs to inputs

.. topic:: ADFlag

    By default, Dr.Jit's AD system destructs the enqueued input graph during
    forward/backward mode traversal. This frees up resources, which is useful
    when working with large wavefronts or very complex computation graphs.
    However, this also prevents repeated propagation of gradients through a
    shared subgraph that is being differentiated multiple times.

    To support more fine-grained use cases that require this, the following
    flags can be used to control what should and should not be destructed.

.. topic:: ADFlag_ClearNone

    Clear nothing.

.. topic:: ADFlag_ClearEdges

    Delete all traversed edges from the computation graph

.. topic:: ADFlag_ClearInput

    Clear the gradients of processed input vertices (in-degree == 0)

.. topic:: ADFlag_ClearInterior

    Clear the gradients of processed interior vertices (out-degree != 0)

.. topic:: ADFlag_ClearVertices

    Clear gradients of processed vertices only, but leave edges intact. Equal to ``ClearInput | ClearInterior``.

.. topic:: ADFlag_Default

    Default: clear everything (edges, gradients of processed vertices). Equal to ``ClearEdges | ClearVertices``.

.. topic:: ADFlag_AllowNoGrad

    Don't fail when the input to a ``drjit.forward`` or ``backward`` operation is not a differentiable array.

.. topic:: JitBackend

    List of just-in-time compilation backends supported by Dr.Jit. See also :py:func:`drjit.backend_v()`.

.. topic:: JitBackend_Invalid

    Indicates that a type is *not* handled by a Dr.Jit backend (e.g., a scalar type)

.. topic:: JitBackend_LLVM

    Dr.Jit backend targeting various processors via the LLVM compiler infrastructure.

.. topic:: JitBackend_CUDA

    Dr.Jit backend targeting NVIDIA GPUs using PTX ("Parallel Thread Execution") IR.

.. topic:: VarType

    List of possible scalar array types (not all of them are supported).

.. topic:: VarType_Void

    Unknown/unspecified type.

.. topic:: VarType_Bool

    Boolean/mask type.

.. topic:: VarType_Int8

    Signed 8-bit integer.

.. topic:: VarType_UInt8

    Unsigned 8-bit integer.

.. topic:: VarType_Int16

    Signed 16-bit integer.

.. topic:: VarType_UInt16

    Unsigned 16-bit integer.

.. topic:: VarType_Int32

    Signed 32-bit integer.

.. topic:: VarType_UInt32

    Unsigned 32-bit integer.

.. topic:: VarType_Int64

    Signed 64-bit integer.

.. topic:: VarType_UInt64

    Unsigned 64-bit integer.

.. topic:: VarType_Pointer

    Pointer to a memory address.

.. topic:: VarType_Float16

    16-bit floating point format (IEEE 754).

.. topic:: VarType_Float32

    32-bit floating point format (IEEE 754).

.. topic:: VarType_Float64

    64-bit floating point format (IEEE 754).

.. topic:: ReduceOp

    List of different atomic read-modify-write (RMW) operations supported by :py:func:`drjit.scatter_reduce()`.

.. topic:: ReduceOp_Identity

    Perform an ordinary scatter operation that ignores the current entry.

.. topic:: ReduceOp_Add

    Addition.

.. topic:: ReduceOp_Mul

    Multiplication.

.. topic:: ReduceOp_Min

    Minimum.

.. topic:: ReduceOp_Max

    Maximum.

.. topic:: ReduceOp_And

    Binary AND operation.

.. topic:: ReduceOp_Or

    Binary OR operation.

.. topic:: CustomOp

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

       # Add two numbers 'x' and 'y'. Calls our '.eval()' callback with detached arguments
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

.. topic:: CustomOp_eval

    Evaluate the custom operation in primal mode.

    You must implement this method when subclassing :py:class:`CustomOp`, since the
    default implementation raises an exception. It should realize the original
    (non-derivative-aware) form of a computation and may take an arbitrary sequence
    of positional, keyword, and variable-length positional/keyword arguments.

    You should not need to call this function yourself---Dr.Jit will automatically do so
    when performing custom operations through the :py:func:`drjit.custom()` interface.

    Note that the input arguments passed to ``.eval()`` will be *detached* (i.e.
    they don't have derivative tracking enabled). This is intentional, since
    derivative tracking is handled by the custom operation along with the other
    callbacks :py:func:`forward` and :py:func:`backward`.

.. topic:: CustomOp_forward

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

.. topic:: CustomOp_backward

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

.. topic:: CustomOp_grad_out

    Query the gradient of the return value.

    Returns an object, whose type matches the original return value produced in
    :py:func:`eval()`. This function should only be used within the
    :py:func:`backward()` callback.

.. topic:: CustomOp_set_grad_out

    Accumulate a gradient into the return value.

    This function should only be used within the :py:func:`forward()` callback.

.. topic:: CustomOp_grad_in

    Query the gradient of a specified input parameter.

    The second argument specifies the parameter name as string. Gradients of
    variable-length positional arguments (``*args``) can be queried by providing an
    integer index instead.

    This function should only be used within the :py:func:`forward()` callback.

.. topic:: CustomOp_set_grad_in

    Accumulate a gradient into the specified input parameter.

    The second argument specifies the parameter name as string. Gradients of
    variable-length positional arguments (``*args``) can be assigned by providing
    an integer index instead.

    This function should only be used within the :py:func:`backward()` callback.

.. topic:: CustomOp_add_input

    Register an implicit input dependency of the operation on an AD variable.

    This function should be called by the :py:func:`eval()` implementation when an
    operation has a differentiable dependence on an input that is not a ordinary
    input argument of the function (e.g., a global program variable or a field of a
    class).

.. topic:: CustomOp_add_output

    Register an implicit output dependency of the operation on an AD variable.

    This function should be called by the :py:func:`eval()` implementation when an
    operation has a differentiable dependence on an output that is not part of the
    function return value (e.g., a global program variable or a field of a
    class)."

.. topic:: CustomOp_name

    Return a descriptive name of the ``CustomOp`` instance.

    Amongst other things, this name is used to document the presence of the
    custom operation in GraphViz debug output. (See :py:func:`graphviz_ad`.)

.. topic:: custom

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

.. topic:: switch

    Selectively invoke functions based on a provided index array.

    When called with a *scalar* ``index`` (of type ``int``), this function
    is equivalent to the following Python expression:

    .. code-block:: python

       targets[index](*args, **kwargs)

    When called with a Dr.Jit *index array* (specifically, 32-bit unsigned
    integers), it performs the vectorized equivalent of the above and assembles an
    array of return values containing the result of all referenced functions. It
    does so efficiently using at most a single invocation of each function in
    ``targets``.

    .. code-block:: python

        from drjit.llvm import UInt32

        res = dr.switch(
            index=UInt32(0, 0, 1, 1), # <-- selects the function
            targets=[                 # <-- arbitrary list of callables
                lambda x: x,
                lambda x: x*10
            ],
            UInt32(1, 2, 3, 4)        # <-- argument passed to function
        )

        # res now contains [0, 10, 20, 30]

    The function traverses the set of positional (``*args``) and keyword arguments
    (``**kwargs``) to find all Dr.Jit arrays including arrays contained within
    :ref:`PyTrees <pytrees>`. It routes a subset of array entries to each function
    as specified by the ``index`` argument.

    Dr.Jit will use one of two possible strategies to compile this operation
    depending on the active compilation flags (see :py:func:`drjit.set_flag`,
    :py:func:`drjit.scoped_set_flag`):

    1. **Symbolic mode**: Dr.Jit transcribes every function into a counterpart in the
       generated low-level intermediate representation (LLVM IR or PTX) and targets
       them via an indirect jump instruction.

       This mode is used when :py:attr:`drjit.JitFlag.SymbolicCalls` is set, which
       is the default.

    2. **Evaluated mode**: Dr.Jit *evaluates* the inputs  ``index``, ``args``,
       ``kwargs`` via :py:func:`drjit.eval`, groups them by ``index``, and invokes
       each function with with the subset of inputs that reference it. Callables
       that are not referenced by any element of ``index`` are ignored.

       In this mode, a :py:func:`drjit.switch` statement will cause Dr.Jit to
       launch a series of kernels processing subsets of the input data (one per
       function).

    A separate section about :ref:`symbolic and evaluated modes <sym-eval>`
    discusses these two options in detail.

    To switch the compilation mode locally, use :py:func:`drjit.scoped_set_flag` as
    shown below:

    .. code-block:: python

       with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, False):
           result = dr.switch(..)

    When a boolean Dr.Jit array (e.g., :py:class:`drjit.llvm.Bool`,
    :py:class:`drjit.cuda.ad.Bool`, etc.) is specified as last positional argument
    or as a keyword argument named ``active``, that argument is treated specially:
    entries of the input arrays associated with a ``False`` mask entry are ignored
    and never passed to the functions. Associated entries of the return value will
    be zero-initialized. The function will still receive the mask argument as
    input, but it will always be set to ``True``.

    .. danger::

        The indices provided to this operation are unchecked by default. Attempting
        to call functions beyond the end of the ``targets`` array is undefined
        behavior and may crash the application, unless such calls are explicitly
        disabled via the ``active`` parameter. Negative indices are not permitted.

        If *debug mode* is enabled via the :py:attr:`drjit.JitFlag.Debug` flag,
        Dr.Jit will insert range checks into the program. These checks disable
        out-of-bound calls and furthermore report warnings to identify problematic
        source locations:

        .. code-block:: pycon
           :emphasize-lines: 2-3

           >>> print(dr.switch(UInt32(0, 100), [lambda x:x], UInt32(1)))
           Attempted to invoke callable with index 100, but this↵
           value must be smaller than 1. (<stdin>:2)

    Args:
        index (int|drjit.ArrayBase): a list of indices to choose the functions

        targets (Sequence[Callable]): a list of callables to which calls will be
          dispatched based on the ``index`` argument.

        mode (Optional[str]): Specify this parameter to override the evaluation mode.
          Possible values besides ``None`` are: ``"symbolic"``, ``"evaluated"``.
          If not specified, the function first checks if the index is
          potentially scalar, in which case it uses a trivial fallback
          implementation. Otherwise, it queries the state of the Jit flag
          :py:attr:`drjit.JitFlag.SymbolicCalls` and then either performs a
          symbolic or an evaluated call.

        label (Optional[str]): An optional descriptive name. If specified, Dr.Jit
          will include this label in generated low-level IR, which can be helpful
          when debugging the compilation of large programs.

        *args (tuple): a variable-length list of positional arguments passed to the
          functions. :ref:`PyTrees <pytrees>` are supported.

        **kwargs (dict): a variable-length list of keyword arguments passed to the
          functions. :ref:`PyTrees <pytrees>` are supported.

    Returns:
        object: When ``index`` is a scalar Python integer, the return value simply
        forwards the return value of the selected function. Otherwise, the function
        returns a Dr.Jit array or :ref:`PyTree <pytrees>` combining the results from
        each referenced callable.

.. topic:: while_loop

    Repeatedly execute a function while a loop condition holds.

    .. rubric:: Motivation

    This function provides a *vectorized* generalization of a standard Python
    ``while`` loop. For example, consider the following Python snippet

    .. code-block:: python

       i: int = 1
       while i < 10:
           x *= x
           i += 1

    This code would fail when ``i`` is replaced by an array with multiple entries
    (e.g., of type :py:class:`drjit.llvm.Int`). In that case, the loop condition
    evaluates to a boolean array of per-component comparisons that are not
    necessarily consistent with each other. In other words, each entry of the array
    may need to run the loop for a *different* number of iterations. A standard
    Python ``while`` loop is not able to do so.

    The :py:func:`drjit.while_loop` function realizes such a fine-grained looping
    mechanism. It takes three main input arguments:

    1. ``state``, a tuple of *state variables* that are modified by the loop
       iteration,

       Dr.Jit optimizes away superfluous state variables, so there isn't any harm
       in specifying variables that aren't actually modified by the loop.

    2. ``cond``, a function that takes the state variables as input and uses them to
       evaluate and return the loop condition in the form of a boolean array,

    3. ``body``, a function that also takes the state variables as input and runs one
       loop iteration. It must return an updated set of state variables.

    The function calls ``cond`` and ``body`` to execute the loop. It then returns a
    tuple containing the final version of the ``state`` variables. With this
    functionality, a vectorized version of the above loop can be written as
    follows:

    .. code-block:: python

       i, x = dr.while_loop(
           state=(i, x),
           cond=lambda i, x: i < 10,
           body=lambda i, x: (i+1, x*x)
       )

    Lambda functions are convenient when the condition and body are simple enough
    to fit onto a single line. In general you may prefer to define local functions
    (``def loop_cond(i, x): ...``) and pass them to the ``cond`` and ``body``
    arguments.

    Dr.Jit also provides the :py:func:`@drjit.syntax <drjit.syntax>` decorator,
    which automatically rewrites standard Python control flow constructs into the
    form shown above. It combines vectorization with the readability of natural
    Python syntax and is the recommended way of (indirectly) using
    :py:func:`drjit.while_loop`. With this decorator, the above example would be
    written as follows:

    .. code-block:: python

       @dr.syntax
       def f(i, x):
           while i < 10:
               x *= x
               i += 1
            return i, x

    .. rubric:: Evaluation modes

    Dr.Jit uses one of *three* different modes to compile this operation depending
    on the inputs and active compilation flags (the text below this overview will
    explain how this mode is automatically selected).

    1. **Scalar mode**: Scalar loops that don't need any vectorization can
       fall back to a simple Python loop construct.

       .. code-block:: python

          while cond(state):
              state = body(*state)

       This is the default strategy when ``cond(state)`` returns a scalar Python
       ``bool``.

       The loop body may still use Dr.Jit types, but note that this effectively
       unrolls the loop, generating a potentially long sequence of instructions
       that may take a long time to compile. Symbolic mode (discussed next) may be
       advantageous in such cases.

    2. **Symbolic mode**: Here, Dr.Jit runs a single loop iteration to capture its
       effect on the state variables. It embeds this captured computation into
       the generated machine code. The loop will eventually run on the device
       (e.g., the GPU) but unlike a Python ``while`` statement, the loop *does not*
       run on the host CPU (besides the mentioned tentative evaluation for symbolic
       tracing).

       When loop optimizations are enabled
       (:py:attr:`drjit.JitFlag.OptimizeLoops`), Dr.Jit may re-trace the loop body
       so that it runs twice in total. This happens transparently and has no
       influence on the semantics of this operation.

    3. **Evaluated mode**: in this mode, Dr.Jit will repeatedly *evaluate* the loop
       state variables and update active elements using the loop body function
       until all of them are done. Conceptually, this is equivalent to the
       following Python code:

       .. code-block:: python

          active = True
          while True:
             dr.eval(state)
             active &= cond(state)
             if not dr.any(active):
                 break
             state = dr.select(active, body(state), state)

       (In practice, the implementation does a few additional things like
       suppressing side effects associated with inactive entries.)

       Dr.Jit will typically compile a kernel when it runs the first loop
       iteration. Subsequent iterations can then reuse this cached kernel since
       they perform the same exact sequence of operations. Kernel caching tends to
       be crucial to achieve good performance, and it is good to be aware of
       pitfalls that can effectively disable it.

       For example, when you update a scalar (e.g. a Python ``int``) in each loop
       iteration, this changing counter might be merged into the generated program,
       forcing the system to re-generate and re-compile code at every iteration,
       and this can ultimately dominate the execution time. If in doubt, increase
       the log level of Dr.Jit (:py:func:`drjit.set_log_level` to
       :py:attr:`drjit.LogLevel.Info`) and check if the kernels being
       launched contain the term ``cache miss``. You can also inspect the *Kernels
       launched* line in the output of :py:func:`drjit.whos`. If you observe soft
       or hard misses at every loop iteration, then kernel caching isn't working
       and you should carefully inspect your code to ensure that the computation
       stays consistent across iterations.

       When the loop processes many elements, and when each element requires a
       different number of loop iterations, there is question of what should be
       done with inactive elements. The default implementation keeps them around
       and does redundant calculations that are, however, masked out. Consequently,
       later loop iterations don't run faster despite fewer elements being active.

       Alternatively, you may specify the parameter ``compress=True`` or set the
       flag :py:attr:`drjit.JitFlag.CompressLoops`, which causes the removal of
       inactive elements after every iteration. This reorganization is not for free
       and does not benefit all use cases, which is why it isn't enabled by
       default.

    A separate section about :ref:`symbolic and evaluated modes <sym-eval>`
    discusses these two options in further detail.

    The :py:func:`drjit.while_loop()` function chooses the evaluation mode as follows:

    1. When the ``mode`` argument is set to ``None`` (the *default*), the
       function examines the loop condition. It uses *scalar* mode when this
       produces a Python `bool`, otherwise it inspects the
       :py:attr:`drjit.JitFlag.SymbolicLoops` flag to switch between *symbolic* (the default)
       and *evaluated* mode.

       To change this automatic choice for a region of code, you may specify the
       ``mode=`` keyword argument, nest code into a
       :py:func:`drjit.scoped_set_flag` block, or change the behavior globally via
       :py:func:`drjit.set_flag`:

       .. code-block:: python

          with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
              # .. nested code will use evaluated loops ..

    2. When ``mode`` is set to ``"scalar"`` ``"symbolic"``, or ``"evaluated"``,
       it directly uses that method without inspecting the compilation flags or
       loop condition type.

    When using the :py:func:`@drjit.syntax <drjit.syntax>` decorator to
    automatically convert Python ``while`` loops into :py:func:`drjit.while_loop`
    calls, you can also use the :py:func:`drjit.hint` function to pass keyword
    arguments including ``mode``, ``label``, or ``max_iterations`` to the generated
    looping construct:

    .. code-block:: python

      while dr.hint(i < 10, name='My loop', mode='evaluated'):
         # ...

    .. rubric:: Assumptions

    The loop condition function must be *pure* (i.e., it should never modify the
    state variables or any other kind of program state). The loop body
    should *not* write to variables besides the officially declared state
    variables:

    .. code-block:: python

       y = ..
       def loop_body(x):
           y[0] += x     # <-- don't do this. 'y' is not a loop state variable

       dr.while_loop(body=loop_body, ...)

    Dr.Jit automatically tracks dependencies of *indirect reads* (done via
    :py:func:`drjit.gather`) and *indirect writes* (done via
    :py:func:`drjit.scatter`, :py:func:`drjit.scatter_reduce`,
    :py:func:`drjit.scatter_add`, :py:func:`drjit.scatter_inc`, etc.). Such
    operations create implicit inputs and outputs of a loop, and these *do not*
    need to be specified as loop state variables (however, doing so causes no
    harm.) This auto-discovery mechanism is helpful when performing vectorized
    methods calls (within loops), where the set of implicit inputs and outputs can
    often be difficult to know a priori. (in principle, any public/private field in
    any instance could be accessed in this way).

    .. code-block:: python

       y = ..
       def loop_body(x):
           # Scattering to 'y' is okay even if it is not declared as loop state
           dr.scatter(target=y, value=x, index=0)

    Another important assumption is that the loop state remains *consistent* across
    iterations, which means:

    1. The type of state variables is not allowed to change. You may not declare a
       Python ``float`` before a loop and then overwrite it with a
       :py:class:`drjit.cuda.Float` (or vice versa).

    2. Their structure/size must be consistent. The loop body may not turn
       a variable with 3 entries into one that has 5.

    3. Analogously, state variables must always be initialized prior to the
       loop. This is the case *even if you know that the loop body is guaranteed to
       overwrite the variable with a well-defined result*. An initial value
       of ``None`` would violate condition 1 (type invariance), while an empty
       array would violate condition 2 (shape compatibility).

    The implementation will check for violations and, if applicable, raise an
    exception identifying problematic state variables.

    .. rubric:: Potential pitfalls

    1. **Long compilation times**.

       In the example below, ``i < 100000`` is *scalar*, causing
       :py:func:`drjit.while_loop()` to use the scalar evaluation strategy that
       effectively copy-pastes the loop body 100000 times to produce a *giant*
       program. Code written in this way will be bottlenecked by the CUDA/LLVM
       compilation stage.

       .. code-block:: python

          @dr.syntax
          def f():
              i = 0
              while i < 100000:
                  # .. costly computation
                  i += 1

    2. **Incorrect behavior in symbolic mode**.

       Let's fix the above program by casting the loop condition into a Dr.Jit type
       to ensure that a *symbolic* loop is used. Problem solved, right?

       .. code-block:: python

          from drjit.cuda import Bool

          @dr.syntax
          def f():
              i = 0
              while Bool(i < 100000):
                  # .. costly computation
                  i += 1

       Unfortunately, no: this loop *never terminates* when run in symbolic mode.
       Symbolic mode does not track modifications of scalar/non-Dr.Jit types across
       loop iterations such as the ``int``-valued loop counter ``i``. It's as if we
       had written ``while Bool(0 < 100000)``, which of course never finishes.

       Evaluated mode does not have this problem---if your loop behaves differently
       in symbolic and evaluated modes, then some variation of this mistake is
       likely to blame. To fix this, we must declare the loop counter as a vector
       type *before* the loop and then modify it as follows:

       .. code-block:: python

          from drjit.cuda import Int

          @dr.syntax
          def f():
              i = Int(0)
              while i < 100000:
                  # .. costly computation
                  i += 1

    .. warning::

       This new implementation of the :py:func:`drjit.while_loop` abstraction still
       lacks the functionality to ``break`` or ``return`` from the loop, or to
       ``continue`` to the next loop iteration. We plan to add these capabilities
       in the near future.

    .. rubric:: Interface

    Args:
        state (tuple): A tuple containing the initial values of the loop's state
          variables. This tuple normally consists of Dr.Jit arrays or :ref:`PyTrees
          <pytrees>`. Other values are permissible as well and will be forwarded to
          the loop body. However, such variables will not be captured by the
          symbolic tracing process.

        cond (Callable): a function/callable that will be invoked with ``*state``
          (i.e., the state variables will be *unpacked* and turned into
          function arguments). It should return a scalar Python ``bool`` or a
          boolean-typed Dr.Jit array representing the loop condition.

        body (Callable): a function/callable that will be invoked with ``*state``
          (i.e., the state variables will be *unpacked* and turned into
          function arguments). It should update the loop state and then return a
          new tuple of state variables that are *compatible* with the previous
          state (see the earlier description regarding what such compatibility entails).

        mode (Optional[str]): Specify this parameter to override the evaluation mode.
          Possible values besides ``None`` are: ``"scalar"``, ``"symbolic"``, ``"evaluated"``.
          If not specified, the function first checks if the loop is
          potentially scalar, in which case it uses a trivial fallback
          implementation. Otherwise, it queries the state of the Jit flag
          :py:attr:`drjit.JitFlag.SymbolicLoops` and then either performs a
          symbolic or an evaluated loop.

        compress (Optional[bool]): Set this this parameter to ``True`` or ``False``
          to enable or disable *loop state compression* in evaluated loops (see the
          text above for a description of this feature). The function
          queries the value of :py:attr:`drjit.JitFlag.CompressLoops` when the
          parameter is not specified. Symbolic loops ignore this parameter.

        labels (list[str]): An optional list of labels associated with each
          ``state`` entry. Dr.Jit uses this to provide better error messages in
          case of a detected inconsistency. The :py:func:`@drjit.syntax <drjit.syntax>`
          decorator automatically provides these labels based on the transformed
          code.

        label (Optional[str]): An optional descriptive name. If specified, Dr.Jit
          will include this label in generated low-level IR, which can be helpful
          when debugging the compilation of large programs.

        max_iterations (int): The maximum number of loop iterations (default: ``-1``).
          You must specify a correct upper bound here if you wish to differentiate
          the loop in reverse mode. In that case, the maximum iteration count is used
          to reserve memory to store intermediate loop state.

        strict (bool): You can specify this parameter to reduce the strictness
          of variable consistency checks performed by the implementation. See
          the documentation of :py:func:`drjit.hint` for an example. The
          default is ``strict=True``.

    Returns:
        tuple: The function returns the final state of the loop variables following
        termination of the loop.

.. topic:: if_stmt

    Conditionally execute code.

    .. rubric:: Motivation

    This function provides a *vectorized* generalization of a standard Python
    ``if`` statement. For example, consider the following Python snippet

    .. code-block:: python

       i: int = .. some expression ..
       if i > 0:
           x = f(i) # <-- some costly function 'f' that depends on 'i'
       else:
           y += 1

    This code would fail if ``i`` is replaced by an array containing multiple
    entries (e.g., of type :py:class:`drjit.llvm.Int`). In that case, the
    conditional expression produces a boolean array of per-component comparisons
    that are not necessarily consistent with each other. In other words, some of
    the entries may want to run the body of the ``if`` statement, while others must
    skip to the ``else`` block. This is not compatible with the semantics of a
    standard Python ``if`` statement.

    The :py:func:`drjit.if_stmt` function realizes a more fine-grained conditional
    operation that accommodates these requirements, while avoiding execution of the
    costly branch unless this is truly needed. It takes the following input
    arguments:

    1. ``cond``, a boolean array that specifies whether the body of the ``if``
       statement should execute.

    2. A tuple of input arguments (``args``) that will be forwarded to
       ``true_fn`` and ``false_fn``. It is important to specify all inputs to
       ensure correct derivative tracking of the operation.

    3. ``true_fn``, a callable that implements the body of the ``if`` block.

    4. ``false_fn``, a callable that implements the body of the ``else`` block.

    The implementation will invoke ``true_fn(*args)`` and ``false_fn(*args)`` to
    trace their contents. The return values of these functions must be compatible
    with each other (a precise definition of compatibility is described below). A
    vectorized version of the earlier example can then be written as follows:

    .. code-block:: python

       x, y = dr.if_stmt(
           args=(i, x, y),
           cond=i > 0,
           true_fn=lambda i, x, y: (f(i), y),
           false_fn=lambda i, x, y: (x, y + 1)
       )

    Lambda functions are convenient when ``true_fn`` and ``false_fn`` are simple
    enough to fit onto a single line. In general you may prefer to define local
    functions (``def true_fn(i, x, y): ...``) and pass them to the ``true_fn`` and
    ``false_fn`` arguments.

    Dr.Jit later optimizes away superfluous inputs/outputs of
    :py:func:`drjit.if_stmt`, so there isn't any harm in, e.g., specifying an
    identical element of a return value in both ``true_fn`` and ``false_fn``.

    Dr.Jit also provides the :py:func:`@drjit.syntax <drjit.syntax>` decorator,
    which automatically rewrites standard Python control flow constructs into the
    form shown above. It combines vectorization with the readability of natural
    Python syntax and is the recommended way of (indirectly) using
    :py:func:`drjit.if_stmt`. With this decorator, the above example would be
    written as follows:

    .. code-block:: python

       @dr.syntax
       def f(i, x, y):
           if i > 0:
               x = f(i)
           else:
               y += 1
           return x, y

    .. rubric:: Evaluation modes

    Dr.Jit uses one of *three* different modes to realize this operation depending
    on the inputs and active compilation flags (the text below this overview will
    explain how this mode is automatically selected).

    1. **Scalar mode**: Scalar ``if`` statements that don't need any
       vectorization can be reduced to normal Python branching constructs:

       .. code-block:: python

          if cond:
              state = true_fn(*args)
          else:
              state = false_fn(*args)

       This strategy is the default when ``cond`` is a scalar Python ``bool``.

    2. **Symbolic mode**: Dr.Jit runs ``true_fn`` and ``false_fn`` to
       capture the computation performed by each function, which allows it to
       generate an equivalent branch in the generated kernel. Symbolic mode
       preserves the control flow structure of the original program by replicating
       it within Dr.Jit's intermediate representation.

    3. **Evaluated mode**: in this mode, Dr.Jit runs both branches of the ``if``
       statement and then combines the results via :py:func:`drjit.select`. This is
       nearly equivalent to the following Python code:

       .. code-block:: python

          true_state = true_fn(*state)
          false_state = false_fn(*state) if false_fn else state
          state = dr.select(cond, true_fn, false_fn)

       (In practice, the implementation does a few additional things like
       suppressing side effects associated with inactive entries.)

       Evaluated mode is conceptually simpler but also slower, since the device
       executes both sides of a branch when only one of them is actually needed.

    The mode is chosen as follows:

    1. When the ``mode`` argument is set to ``None`` (the *default*), the
       function examines the type of the ``cond`` input and uses scalar
       mode if the type is a builtin Python ``bool``.

       Otherwise, it chooses between symbolic and evaluated mode based on the
       :py:attr:`drjit.JitFlag.SymbolicConditionals` flag, which is set by default.
       To change this choice for a region of code, you may specify the ``mode=``
       keyword argument, nest it into a :py:func:`drjit.scoped_set_flag` block, or
       change the behavior globally via :py:func:`drjit.set_flag`:

       .. code-block:: python

          with dr.scoped_set_flag(dr.JitFlag.SymbolicConditionals, False):
              # .. nested code will use evaluated mode ..

    2. When ``mode`` is set to ``"scalar"`` ``"symbolic"``, or ``"evaluated"``,
       it directly uses that mode without inspecting the compilation flags or
       condition type.

    When using the :py:func:`@drjit.syntax <drjit.syntax>` decorator to
    automatically convert Python ``if`` statements into :py:func:`drjit.if_stmt`
    calls, you can also use the :py:func:`drjit.hint` function to pass keyword
    arguments including the ``mode`` and ``label`` parameters.

    .. code-block:: python

      if dr.hint(i < 10, mode='evaluated'):
         # ...

    .. rubric:: Assumptions

    The return values of ``true_fn`` and ``false_fn`` must be of the same type.
    This requirement applies recursively if the return value is a :ref:`PyTree
    <pytrees>`.

    Dr.Jit will refuse to compile vectorized conditionals, in which ``true_fn``
    and ``false_fn`` return a scalar that is inconsistent between the branches.

    .. code-block:: pycon
       :emphasize-lines: 10-15

       >>> @dr.syntax
       ... def (x):
       ...    if x > 0:
       ...        y = 1
       ...    else:
       ...        y = 0
       ...    return y
       ...
       >>> print(f(dr.llvm.Float(-1,2)))
       RuntimeError: dr.if_stmt(): detected an inconsistency when comparing the return
       values of 'true_fn' and 'false_fn': drjit.detail.check_compatibility(): inconsistent
       scalar Python object of type 'int' for field 'y'.

       Please review the interface and assumptions of dr.if_stmt() as explained in the
       Dr.Jit documentation.

    The problem can be solved by assigning an instance of a capitalized Dr.Jit type
    (e.g., ``y=Int(1)``) so that the operation can be tracked.

    The functions ``true_fn`` and ``false_fn`` should *not* write to variables
    besides the explicitly declared return value(s):

    .. code-block:: python

       vec = drjit.cuda.Array3f(1, 2, 3)
       def true_fn(x):
           vec.x += x     # <-- don't do this. 'y' is not a declared output

       dr.if_stmt(args=(x,), true_fun=true_fn, ...)

    This example can be fixed as follows:

    .. code-block:: python

       def true_fn(x, vec):
           vec.x += x
           return vec

       vec = dr.if_stmt(args=(x, vec), true_fun=true_fn, ...)

    :py:func:`drjit.if_stmt()` is differentiable in both forward and reverse modes.
    Correct derivative tracking requires that regular differentiable inputs are
    specified via the ``args`` parameter. The :py:func:`@drjit.syntax
    <drjit.syntax>` decorator ensures that these assumptions are satisfied.

    Dr.Jit also tracks dependencies of *indirect reads* (done via
    :py:func:`drjit.gather`) and *indirect writes* (done via
    :py:func:`drjit.scatter`, :py:func:`drjit.scatter_reduce`,
    :py:func:`drjit.scatter_add`, :py:func:`drjit.scatter_inc`, etc.). Such
    operations create implicit inputs and outputs, and these *do not* need to be
    specified as part of ``args`` or the return value of ``true_fn`` and
    ``false_fn`` (however, doing so causes no harm.) This auto-discovery mechanism
    is helpful when performing vectorized methods calls (within conditional
    statements), where the set of implicit inputs and outputs can often be
    difficult to know a priori. (in principle, any public/private field in any
    instance could be accessed in this way).

    .. code-block:: python

       y = ..
       def true_fn(x):
           # 'y' is neither declared as input nor output of 'f', which is fine
           dr.scatter(target=y, value=x, index=0)

       dr.if_stmt(args=(x,), true_fn=true_fn, ...)

    .. rubric:: Interface

    Args:
        cond (bool|drjit.ArrayBase): a scalar Python ``bool`` or a boolean-valued
          Dr.Jit array.

        args (tuple): A list of positional arguments that will be forwarded to
          ``true_fn`` and ``false_fn``.

        true_fn (Callable): a callable that implements the body of the ``if`` block.

        false_fn (Callable): a callable that implements the body of the ``else`` block.

        mode (Optional[str]): Specify this parameter to override the evaluation
          mode. Possible values besides ``None`` are: ``"scalar"``, ``"symbolic"``,
          ``"evaluated"``.

        arg_labels (list[str]): An optional list of labels associated with each
          input argument. Dr.Jit uses this feature in combination with
          the :py:func:`@drjit.syntax <drjit.syntax>` decorator to provide better
          error messages in case of detected inconsistencies.

        rv_labels (list[str]): An optional list of labels associated with each
          element of the return value. This parameter should only be specified when
          the return value is a tuple. Dr.Jit uses this feature in combination with
          the :py:func:`@drjit.syntax <drjit.syntax>` decorator to provide better
          error messages in case of detected inconsistencies.

        label (Optional[str]): An optional descriptive name. If specified, Dr.Jit
          will include this label in generated low-level IR, which can be helpful
          when debugging the compilation of large programs.

        strict (bool): You can specify this parameter to reduce the strictness
          of variable consistency checks performed by the implementation. See
          the documentation of :py:func:`drjit.hint` for an example. The
          default is ``strict=True``.

    Returns:
        object: Combined return value mixing the results of ``true_fn`` and
        ``false_fn``.

.. topic:: dispatch

    Invoke a provided Python function for each instance in an instance array.

    This function invokes the provided ``target`` for each instance
    in the instance array ``inst`` and assembles the return values into
    a result array. Conceptually, it does the following:

    .. code-block:: python

       def dispatch(inst, target, *args, **kwargs):
           result = []
           for in in inst:
               result.append(target(inst, *args, **kwargs))

    However, the implementation accomplishes this more efficiently using only a
    single call per unique instance. Instead of a Python ``list``, it returns a
    Dr.Jit array or :ref:`PyTree <pytrees>`.

    In practice, this function is mainly good for two things:

    - Dr.Jit instance arrays contain C++ instance, and these will typically expose
      a set of methods. Adding further methods requires re-compiling C++ code and
      adding bindings, which may impede quick prototyping. With
      :py:func:`drjit.dispatch()`, a developer can quickly implement additional
      vectorized method calls within Python (with the caveat that these can only
      access public members of the underlying type).

    - Dynamic dispatch is a relatively costly operation. When multiple calls are
      performed on the same set of instances, it may be preferable to merge them
      into a single and potentially significantly faster use of
      :py:func:`drjit.dispatch()`. An example is shown below:

      .. code-block:: python

         inst = # .. Array of C++ instances ..
         result_1 = inst.func_1(arg1)
         result_2 = inst.func_2(arg2)

      The following alternative implementation instead uses :py:func:`drjit.dispatch()`:

      .. code-block:: python

         def my_func(self, arg1, arg2):
             return (self.func_1(arg1),
                     self.func_2(arg2))

         result_1, result_2 = dr.dispatch(inst, my_func, arg1, arg2)

    This function is otherwise very similar to :py:func:`drjit.switch()`
    and similarly provides two different compilation modes, differentiability,
    and special handling of mask arguments. Please review the documentation
    of :py:func:`drjit.switch()` for details.

    Args:
        inst (drjit.ArrayBase): a Dr.Jit instance array.

        target (Callable): function to dispatch on all instances

        mode (Optional[str]): Specify this parameter to override the evaluation mode.
          Possible values besides ``None`` are: ``"symbolic"``, ``"evaluated"``.
          If not specified, the function first checks if the index is
          potentially scalar, in which case it uses a trivial fallback
          implementation. Otherwise, it queries the state of the Jit flag
          :py:attr:`drjit.JitFlag.SymbolicCalls` and then either performs a
          symbolic or an evaluated call.

        label (Optional[str]): An optional descriptive name. If specified, Dr.Jit
          will include this label in generated low-level IR, which can be helpful
          when debugging the compilation of large programs.

        *args (tuple): a variable-length list of positional arguments passed to the
          function. :ref:`PyTrees <pytrees>` are supported.

        **kwargs (dict): a variable-length list of keyword arguments passed to the
          function. :ref:`PyTrees <pytrees>` are supported.

    Returns:
        object: A Dr.Jit array or :ref:`PyTree <pytrees>` containing the
        result of each performed function call.

.. topic:: detail_copy

    Create a deep copy of a PyTree

    This function recursively traverses PyTrees and replaces Dr.Jit arrays with
    copies created via the ordinary copy constructor. It also rebuilds tuples,
    lists, dictionaries, and custom data structures. The purpose of this function
    is isolate the inputs of :py:func:`drjit.while_loop()` and
    :py:func:`drjit.if_stmt()` from changes.

    This function exists for Dr.Jit-internal use. You probably should not call
    it in your own application code.

.. topic:: detail_check_compatibility

    Traverse two PyTrees in parallel and ensure that they have an identical
    structure.

    Raises an exception is a mismatch is found (e.g., different types, arrays with
    incompatible numbers of elements, dictionaries with different keys, etc.)

.. topic:: detail_collect_indices

    Return Dr.Jit variable indices associated with the provided data structure.

    This function traverses Dr.Jit arrays, tensors, :ref:`PyTree <pytrees>` (lists,
    tuples, dictionaries, custom data structures) and returns the indices of all detected
    variables (in the order of traversal, may contain duplicates). The index
    information is returned as a list of encoded 64 bit integers, where each
    contains the AD variable index in the upper 32 bits and the JIT variable
    index in the lower 32 bit.

    This function exists for Dr.Jit-internal use. You probably should not
    call it in your own application code.

.. topic:: detail_update_indices

    Create a copy of the provided input while replacing Dr.Jit variables
    with new ones based on a provided set of indices.

    This function works analogously to ``collect_indices``, except that it
    consumes an index array and produces an updated output.

    It recursively traverses and copies an input object that may be a Dr.Jit
    array, tensor, or :ref:`PyTree <pytrees>` (list, tuple, dict, custom data
    structure) while replacing any detected Dr.Jit variables with new ones based
    on the provided index vector. The function returns the resulting object,
    while leaving the input unchanged. The output array object borrows the
    provided array references as opposed to stealing them.

    This function exists for Dr.Jit-internal use. You probably should not call
    it in your own application code.

.. topic:: detail_reset

    Release all Jit variables in a PyTree

    This function recursively traverses PyTrees and replaces Dr.Jit arrays with
    empty instances of the same type. :py:func:`drjit.while_loop` uses this
    function internally to release references held by a temporary copy of the
    state tuple.

.. topic:: flag

    Query whether the given Dr.Jit compilation flag is active.

.. topic:: set_flag

    Set the value of the given Dr.Jit compilation flag.

.. topic:: scoped_set_flag

    Context manager, which sets or unsets a Dr.Jit compilation flag in a local
    execution scope.

    For example, the following snippet shows how to temporarily disable a flag:

    .. code-block:: python

       with dr.scoped_set_flag(dr.JitFlag.SymbolicCalls, False):
           # Code affected by the change should be placed here

       # Flag is returned to its original status

.. topic:: detail_reduce_identity

   Return the identity element for a reduction with the desired variable type
   and operation.

.. topic:: detail_can_scatter_reduce

   Check if the underlying backend supports a desired flavor of
   scatter-reduction for the given array type.

.. topic:: JitFlag

    Flags that control how Dr.Jit compiles and optimizes programs.

    This enumeration lists various flag that control how Dr.Jit compiles and
    optimizes programs, most of which are enabled by default. The status of each
    flag can be queried via :py:func:`drjit.flag` and enabled/disabled via the
    :py:func:`drjit.set_flag` or the recommended :py:func:`drjit.scoped_set_flag`
    functions, e.g.:

    .. code-block:: python

      with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
          # code that has this flag disabled goes here

    The most common reason to update the flags is to switch between *symbolic* and
    *evaluated* execution of loops and functions. The former eagerly executes
    programs by breaking them into many smaller kernels, while the latter records
    computation symbolically to assemble large *megakernels*. See explanations
    below along with the documentation of :py:func:`drjit.switch` and
    :py:class:`drjit.while_loop` for more details on these two modes.

    Dr.Jit flags are a thread-local property. This means that multiple independent
    threads using Dr.Jit can set them independently without interfering with each
    other.

.. topic:: JitFlag_Debug

    **Debug mode**: Enable functionality to uncover errors in application code.

    When debug mode is enabled, Dr.Jit inserts a number of additional runtime
    checks to locate sources of undefined behavior.

    Debug mode comes at a *significant* cost: it interferes with kernel caching,
    reduces tracing performance, and produce kernels that run slower. We recommend
    that you only use it periodically before a release, or when encountering a
    serious problem like a crashing kernel.

    First, debug mode enables assertion checks in user code such as those performed by
    :py:func:`drjit.assert_true`, :py:func:`drjit.assert_false`, and
    :py:func:`drjit.assert_equal`.

    Second, Dr.Jit inserts additional checks to intercept out-of-bound reads
    and writes performed by operations such as :py:func:`drjit.scatter`,
    :py:func:`drjit.gather`, :py:func:`drjit.scatter_reduce`,
    :py:func:`drjit.scatter_inc`, etc. It also detects calls to invalid callables
    performed via :py:func:`drjit.switch`, :py:func:`drjit.dispatch`. Such invalid
    operations are masked, and they generate a warning message on the console,
    e.g.:

    .. code-block:: pycon
       :emphasize-lines: 2-3

       >>> dr.gather(dtype=UInt, source=UInt(1, 2, 3), index=UInt(0, 1, 100))
       RuntimeWarning: drjit.gather(): out-of-bounds read from position 100 in an array↵
       of size 3. (<stdin>:2)

    Finally, Dr.Jit also installs a `python tracing hook
    <https://docs.python.org/3/library/sys.html#sys.settrace>`__ that
    associates all Jit variables with their Python source code location, and
    this information is propagated all the way to the final intermediate
    representation (PTX, LLVM IR). This is useful for low-level debugging and
    development of Dr.Jit itself. You can query the source location
    information of a variable ``x`` by writing ``x.label``.

    Due to limitations of the Python tracing interface, this handler becomes active
    within the *next* called function (or Jupyter notebook cell) following
    activation of the :py:attr:`drjit.JitFlag.Debug` flag. It does not apply to
    code within the same scope/function.

    C++ code using Dr.Jit also benefits from debug mode but will lack accurate
    source code location information. In mixed-language projects, the reported file
    and line number information will reflect that of the last operation on the
    Python side of the interface.

.. topic:: JitFlag_ReuseIndices

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

    Index reuse is *enabled* by default.

.. topic:: JitFlag_ConstantPropagation

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

    Constant propagation is *enabled* by default.

.. topic:: JitFlag_ValueNumbering

    **Local value numbering**: a simple variant of common subexpression elimination
    that collapses identical expressions within basic blocks. For example, the
    following assertion holds when value numbering is enabled in Dr.Jit.

    .. code-block:: python

       from drjit.llvm import Int

       # Create two non-literal arrays stored in device memory
       a, b = Int(1, 2, 3), Int(4, 5, 6)

       # Perform the same arithmetic operation twice
       c1 = a + b
       c2 = a + b

       # Verify that c1 and c2 reference the same Dr.Jit variable
       assert c1.index == c2.index

    Local value numbering is *enabled* by default.

.. topic:: JitFlag_FastMath

    **Fast Math**: this flag is analogous to the ``-ffast-math`` flag in C
    compilers. When set, the system may use approximations and simplifications
    that sacrifice strict IEEE-754 compatibility.

    Currently, it changes two behaviors:

    - expressions of the form ``a * 0`` will be simplified to ``0`` (which is
      technically not correct when ``a`` is infinite or NaN-valued).

    - Dr.Jit will use slightly approximate division and square root
      operations in CUDA mode. Note that disabling fast math mode is costly
      on CUDA devices, as the strict IEEE-754 compliant version of these
      operations uses software-based emulation.

    Fast math mode is *enabled* by default.

.. topic:: JitFlag_SymbolicCalls

    Dr.Jit provides two main ways of compiling function calls targeting
    *instance arrays*.

    1. **Symbolic mode** (the default): Dr.Jit invokes each callable with
       *symbolic* (abstract) arguments. It does this to capture a transcript
       of the computation that it can turn into a function in the generated
       kernel. Symbolic mode preserves the control flow structure of the
       original program by replicating it within Dr.Jit's intermediate
       representation.

    2. **Evaluated mode**: Dr.Jit evaluates all inputs and groups them by instance
       ID. Following this, it launches a kernel *per instance* to process the
       rearranged inputs and assemble the function return value.

    A separate section about :ref:`symbolic and evaluated modes <sym-eval>`
    discusses these two options in detail.

    Besides calls to instance arrays, this flag also controls the behavior of
    the functions :py:func:`drjit.switch` and :py:func:`drjit.dispatch`.

    Symbolic calls are *enabled* by default.

.. topic:: JitFlag_OptimizeCalls

    Perform basic optimizations
    for function calls on instance arrays.

    This flag enables two optimizations:

    - *Constant propagation*: Dr.Jit will propagate literal constants across
      function boundaries while tracing, which can unlock simplifications within.
      This is especially useful in combination with automatic differentiation,
      where it helps to detect code that does not influence the computed
      derivatives.

    - *Devirtualization*: When an element of the return value has the same
      computation graph in all instances, it is removed from the function call
      interface and moved to the caller.

    The flag is enabled by default. Note that it is only meaningful in combination
    with  :py:attr:`SymbolicCalls`. Besides calls to instance arrays, this flag
    also controls the behavior of the functions :py:func:`drjit.switch` and
    :py:func:`drjit.dispatch`.

.. topic:: JitFlag_MergeFunctions

    Deduplicate code generated
    by function calls on instance arrays.

    When ``arr`` is an instance array (potentially with thousands of instances),
    a function call like

    .. code-block:: python

       arr.f(inputs...)

    can potentially generate vast numbers of different functions in the generated
    code. At the same time, many of these functions may contain identical code
    (or code that is identical except for data references).

    Dr.Jit can exploit such redundancy and merge such functions during code
    generation. Besides generating shorter programs, this also helps to reduce
    thread divergence.

    This flag is *enabled* by default. Note that it is only meaningful in
    combination with :py:attr:`SymbolicCalls`. Besides calls to instance arrays,
    this flag also controls the behavior of the functions :py:func:`drjit.switch`
    and :py:func:`drjit.dispatch`.

.. topic:: JitFlag_SymbolicLoops

    Dr.Jit provides two main ways of compiling loops involving Dr.Jit arrays.

    1. **Symbolic mode** (the default): Dr.Jit executes the loop a single
       time regardless of how many iterations it requires in practice. It
       does so with *symbolic* (abstract) arguments to capture the loop
       condition and body and then turns it into an equivalent loop in the
       generated kernel. Symbolic mode preserves the control flow structure
       of the original program by replicating it within Dr.Jit's intermediate
       representation.

    2. **Evaluated mode**: Dr.Jit evaluates the loop's state variables and
       reduces the loop condition to a single element (``bool``) that
       expresses whether any elements are still alive. If so, it runs the
       loop body and the process repeats.

    A separate section about :ref:`symbolic and evaluated modes <sym-eval>`
    discusses these two options in detail.

    Symbolic loops are *enabled* by default.

.. topic:: JitFlag_OptimizeLoops

    Perform basic optimizations
    for loops involving Dr.Jit arrays.

    This flag enables two optimizations:

    - *Constant arrays*: loop state variables that aren't modified by
      the loop are automatically removed. This shortens the generated code, which can
      be helpful especially in combination with the automatic transformations
      performed by :py:func:`@drjit.syntax <drjit.syntax>` that can be somewhat
      conservative in classifying too many local variables as potential loop state.

    - *Literal constant arrays*: In addition to the above point, constant
      loop state variables that are *literal constants* are propagated into
      the loop body, where this may unlock further optimization opportunities.

      This is useful in combination with automatic differentiation, where
      it helps to detect code that does not influence the computed derivatives.

    A practical implication of this optimization flag is that it may cause
    :py:func:`drjit.while_loop` to run the loop body twice instead of just once.

    This flag is *enabled* by default. Note that it is only meaningful
    in combination with :py:attr:`SymbolicLoops`.

.. topic:: JitFlag_CompressLoops

    Compress the loop state of evaluated loops after every iteration.

    When an evaluated loop processes many elements, and when each element requires a
    different number of loop iterations, there is question of what should be done
    with inactive elements. The default implementation keeps them around and does
    redundant calculations that are, however, masked out. Consequently, later loop
    iterations don't run faster despite fewer elements being active.

    Setting this flag causes the removal of inactive elements after every
    iteration. This reorganization is not for free and does not benefit all use
    cases.

    This flag is *disabled* by default. Note that it only applies to *evaluated*
    loops (i.e., when :py:attr:`SymbolicLoops` is disabled, or the
    ``mode='evaluted'`` parameter as passed to the loop in question).

.. topic:: JitFlag_SymbolicConditionals

    Dr.Jit provides two main ways of compiling conditionals involving Dr.Jit arrays.

    1. **Symbolic mode** (the default): Dr.Jit captures the computation
       performed by the ``True`` and ``False`` branches and generates an
       equivalent branch in the generated kernel. Symbolic mode preserves the
       control flow structure of the original program by replicating it
       within Dr.Jit's intermediate representation.

    2. **Evaluated mode**: Dr.Jit always executes both branches and blends
       their outputs.

    A separate section about :ref:`symbolic and evaluated modes <sym-eval>`
    discusses these two options in detail.

    Symbolic conditionals are *enabled* by default.

.. topic:: JitFlag_ForceOptiX

    Force execution through OptiX even if a kernel doesn't use ray tracing. This
    only applies to the CUDA backend is mainly helpful for automated tests done by
    the Dr.Jit team.

    This flag is *disabled* by default.

.. topic:: JitFlag_PrintIR

    Print the low-level IR representation when launching a kernel.

    If enabled, this flag causes Dr.Jit to print the low-level IR (LLVM IR,
    NVIDIA PTX) representation of the generated code onto the console (or
    Jupyter notebook).

    This flag is *disabled* by default.

.. topic:: JitFlag_KernelHistory

    Maintain a history of kernel launches to profile/debug programs.

    Programs written on top of Dr.Jit execute in an *extremely* asynchronous
    manner. By default, the system postpones the computation to build large fused
    kernels. Even when this computation eventually runs, it does so asynchronously
    with respect to the host, which can make benchmarking difficult.

    In general, beware of the following benchmarking *anti-pattern*:

    .. code-block:: python

        import time
        a = time.time()
        # Some Dr.Jit computation
        b = time.time()
        print("took %.2f ms" % ((b-a) * 1000))

    In the worst case, the measured time interval may only capture the *tracing
    time*, without any actual computation having taken place. Another common
    mistake with this pattern is that Dr.Jit or the target device may still be busy
    with computation that started *prior* to the ``a = time.time()`` line, which is
    now incorrectly added to the measured period.

    Dr.Jit provides a *kernel history* feature, where it creates an entry in a list
    whenever it launches a kernel or related operation (memory copies, etc.). This
    not only gives accurate and isolated timings (measured with counters on the
    CPU/GPU) but also reveals if a kernel was launched at all. To capture the
    kernel history, set this flag just before the region to be benchmarked and call
    :py:func:`drjit.kernel_history()` at the end.

    Capturing the history has a (very) small cost and is therefore  *disabled* by
    default.

.. topic:: JitFlag_LaunchBlocking

    Force synchronization after every kernel launch. This is useful to
    isolate severe problems (e.g. crashes) to a specific kernel.

    This flag has a severe performance impact and is *disabled* by default.

.. topic:: JitFlag_ScatterReduceLocal

    Reduce locally before performing atomic scatter-reductions.

    Atomic memory operations are expensive when many writes target the same
    region of memory. This leads to a phenomenon called *contention* that is
    normally associated with significant slowdowns (10-100x aren't unusual).

    This issue is particularly common when automatically differentiating
    computation in *reverse mode* (e.g. :py:func:`drjit.backward`), since
    this transformation turns differentiable global memory reads into atomic
    scatter-additions. A differentiable scalar read is all it takes to create
    such an atomic memory bottleneck.

    To reduce this cost, Dr.Jit can perform a *local reduction* that
    uses cooperation between SIMD/warp lanes to resolve all requests
    targeting the same address and then only issuing a single atomic memory
    transaction per unique target. This can reduce atomic memory traffic
    32-fold on the GPU (CUDA) and 16-fold on the CPU (AVX512).
    On the CUDA backend, local reduction is currently only supported for
    32-bit operands (signed/unsigned integers and single precision variables).

    The section on :ref:`optimizations <reduce-local>` presents plots that
    demonstrate the impact of this optimization.

    The JIT flag :py:attr:`drjit.JitFlag.ScatterReduceLocal` affects the
    behavior of :py:func:`scatter_add`, :py:func:`scatter_reduce` along with
    the reverse-mode derivative of :py:func:`gather`.
    Setting the flag to ``True`` will usually cause a ``mode=`` argument
    value of :py:attr:`drjit.ReduceOp.Auto` to be interpreted as
    :py:attr:`drjit.ReduceOp.Local`. Another LLVM-specific optimization
    takes precedence in certain situations,
    refer to the discussion of this flag for details.

    This flag is *enabled* by default.

.. topic:: JitFlag_SymbolicScope

    This flag is set to ``True`` when Dr.Jit is currently capturing symbolic
    computation. The flag is automatically managed and should not be updated
    by application code.

    User code may query this flag to check if it is legal to perform certain
    operations (e.g., evaluating array contents).

    Note that this information can also be queried in a more fine-grained
    manner (per variable) using the :py:attr:`drjit.ArrayBase.state` field.

.. topic:: JitFlag_Default

    The default set of optimization flags consisting of

    - :py:attr:`drjit.JitFlag.ConstantPropagation`,
    - :py:attr:`drjit.JitFlag.ValueNumbering`,
    - :py:attr:`drjit.JitFlag.FastMath`,
    - :py:attr:`drjit.JitFlag.SymbolicLoops`,
    - :py:attr:`drjit.JitFlag.OptimizeLoops`,
    - :py:attr:`drjit.JitFlag.SymbolicCalls`,
    - :py:attr:`drjit.JitFlag.MergeFunctions`,
    - :py:attr:`drjit.JitFlag.OptimizeCalls`,
    - :py:attr:`drjit.JitFlag.SymbolicConditionals`,
    - :py:attr:`drjit.JitFlag.ReuseIndices`, and
    - :py:attr:`drjit.JitFlag.ScatterReduceLocal`.

.. topic:: JitFlag_LoopRecord

    Deprecated. Replaced by :py:attr:`SymbolicLoops`.

.. topic:: JitFlag_LoopOptimize

    Deprecated. Replaced by :py:attr:`OptimizeLoops`.

.. topic:: JitFlag_VCallRecord

    Deprecated. Replaced by :py:attr:`SymbolicCalls`.

.. topic:: JitFlag_VCallOptimize

    Deprecated. Replaced by :py:attr:`OptimizeCalls`.

.. topic:: JitFlag_VCallDeduplicate

    Deprecated. Replaced by :py:attr:`MergeFunctions`.

.. topic:: JitFlag_Recording

    Deprecated. Replaced by :py:attr:`Symbolic`.

.. topic:: VarState

    The :py:attr:`drjit.ArrayBase.state` property returns one of the following enumeration values describing possible evaluation states of a Dr.Jit variable.

.. topic:: VarState_Invalid

    The variable has length 0 and effectively does not exist.

.. topic:: VarState_Undefined

    An undefined memory region. Does not (yet) consume device memory.

.. topic:: VarState_Literal

    A literal constant. Does not consume device memory.

.. topic:: VarState_Unevaluated

    An ordinary unevaluated variable that is neither a literal constant nor symbolic.

.. topic:: VarState_Evaluated

    Evaluated variable backed by an device memory region.

.. topic:: VarState_Dirty

    An evaluated variable backed by a device memory region. The variable
    furthermore has pending *side effects* (i.e. the user has performed a
    :py:func`:drjit.scatter`, :py:func:`drjit.scatter_reduce`
    :py:func`:drjit.scatter_inc`, :py:func`:drjit.scatter_add`, or
    :py:func`:drjit.scatter_add_kahan` operation, and the effect of this operation
    has not been realized yet). The array's status will automatically change to
    :py:attr:`Evaluated` the next time that Dr.Jit evaluates computation, e.g. via
    :py:func:`drjit.eval`.

.. topic:: VarState_Symbolic

    A symbolic variable that could take on various inputs. Cannot be evaluated.

.. topic:: VarState_Mixed

    This is a nested array, and the components have mixed states.

.. topic:: ArrayBase_state

    This read-only property returns an enumeration value describing the evaluation state of this Dr.Jit array.

    :type: drjit.VarState

.. topic:: reinterpret_array

    Reinterpret the provided Dr.Jit array or tensor as a different type.

    This operation reinterprets the input type as another type provided that it has
    a compatible in-memory layout (this operation is also known as a *bit-cast*).

    Args:
        dtype (type): Target type.

        value (object): A compatible Dr.Jit input array or tensor.

    Returns:
        object: Result of the conversion as described above.

.. topic:: PCG32

    Implementation of PCG32, a member of the PCG family of random number generators
    proposed by Melissa O'Neill.

    PCG combines a Linear Congruential Generator (LCG) with a permutation function
    that yields high-quality pseudorandom variates while at the same time requiring
    very low computational cost and internal state (only 128 bit in the case of
    PCG32).

    More detail on the PCG family of pseudorandom number generators can be found
    `here <https://www.pcg-random.org/index.html>`__.

    The :py:class:`PCG32` class is implemented as a :ref:`PyTree <pytrees>`, which
    means that it is compatible with symbolic function calls, loops, etc.

.. topic:: PCG32_PCG32

    Initialize a random number generator that generates ``size`` variates in parallel.

    The ``initstate`` and ``initseq`` inputs determine the initial state and increment
    of the linear congruential generator. Their defaults values are based on the
    original implementation.

    The implementation of this routine internally calls py:func:`seed`, with one
    small twist. When multiple random numbers are being generated in parallel, the
    constructor adds an offset equal to :py:func:`drjit.arange(UInt64, size)
    <drjit.arange>` to both ``initstate`` and ``initseq`` to de-correlate the
    generated sequences.

.. topic:: PCG32_PCG32_2

    Copy-construct a new PCG32 instance from an existing instance.

.. topic:: PCG32_seed

    Seed the random number generator with the given initial state and sequence ID.

    The ``initstate`` and ``initseq`` inputs determine the initial state and increment
    of the linear congruential generator. Their values are the defaults from the
    original implementation.

.. topic:: PCG32_next_uint32

    Generate a uniformly distributed unsigned 32-bit random number

    Two overloads of this function exist: the masked variant does not advance
    the PRNG state of entries ``i`` where ``mask[i] == False``.

.. topic:: PCG32_next_uint64

    Generate a uniformly distributed unsigned 64-bit random number

    Internally, the function calls :py:func:`next_uint32` twice.

    Two overloads of this function exist: the masked variant does not advance
    the PRNG state of entries ``i`` where ``mask[i] == False``.

.. topic:: PCG32_next_float32

    Generate a uniformly distributed single precision floating point number on the
    interval :math:`[0, 1)`.

    Two overloads of this function exist: the masked variant does not advance
    the PRNG state of entries ``i`` where ``mask[i] == False``.

.. topic:: PCG32_next_float64

    Generate a uniformly distributed double precision floating point number on the
    interval :math:`[0, 1)`.

    Two overloads of this function exist: the masked variant does not advance
    the PRNG state of entries ``i`` where ``mask[i] == False``.

.. topic:: PCG32_next_uint32_bounded

    Generate a uniformly distributed 32-bit integer number on the
    interval :math:`[0, \texttt{bound})`.

    To ensure an unbiased result, the implementation relies on an iterative
    scheme that typically finishes after 1-2 iterations.

.. topic:: PCG32_next_uint64_bounded

    Generate a uniformly distributed 64-bit integer number on the
    interval :math:`[0, \texttt{bound})`.

    To ensure an unbiased result, the implementation relies on an iterative
    scheme that typically finishes after 1-2 iterations.

.. topic:: PCG32_add

    Advance the pseudorandom number generator.

    This function implements a multi-step advance function that is equivalent to
    (but more efficient than) calling the random number generator ``arg`` times
    in sequence.

    This is useful to advance a newly constructed PRNG to a certain known state.

.. topic:: PCG32_iadd

    In-place addition operator based on :py:func:`__add__`.

.. topic:: PCG32_sub

    Rewind the pseudorandom number generator.

    This function implements the opposite of ``__add__`` to step a PRNG backwards.
    It can also compute the *difference* (as counted by the number of internal
    ``next_uint32`` steps) between two :py:class:`PCG32` instances. This assumes
    that the two instances were consistently seeded.

.. topic:: PCG32_isub

    In-place subtraction operator based on :py:func:`__sub__`.

.. topic:: PCG32_inc

    Sequence increment of the PCG32 PRNG (an unsigned 64-bit integer or integer array). Please see the original paper for details on this field.

.. topic:: PCG32_state

    Sequence state of the PCG32 PRNG (an unsigned 64-bit integer or integer array). Please see the original paper for details on this field.

.. topic:: Texture_init

    Create a new texture with the specified size and channel count

    On CUDA, this is a slow operation that synchronizes the GPU pipeline, so
    texture objects should be reused/updated via :py:func:`set_value()` and
    :py:func:`set_tensor()` as much as possible.

    When ``use_accel`` is set to ``False`` on CUDA mode, the texture will not
    use hardware acceleration (allocation and evaluation). In other modes
    this argument has no effect.

    The ``filter_mode`` parameter defines the interpolation method to be used
    in all evaluation routines. By default, the texture is linearly
    interpolated. Besides nearest/linear filtering, the implementation also
    provides a clamped cubic B-spline interpolation scheme in case a
    higher-order interpolation is needed. In CUDA mode, this is done using a
    series of linear lookups to optimally use the hardware (hence, linear
    filtering must be enabled to use this feature).

    When evaluating the texture outside of its boundaries, the ``wrap_mode``
    defines the wrapping method. The default behavior is ``drjit.WrapMode.Clamp``,
    which indefinitely extends the colors on the boundary along each dimension.

.. topic:: Texture_init_tensor

    Construct a new texture from a given tensor.

    This constructor allocates texture memory with the shape information
    deduced from ``tensor``. It subsequently invokes :py:func:`set_tensor(tensor)`
    to fill the texture memory with the provided tensor.

    When both ``migrate`` and ``use_accel`` are set to ``True`` in CUDA mode, the texture
    exclusively stores a copy of the input data as a CUDA texture to avoid
    redundant storage. Note that the texture is still differentiable even when migrated.

.. topic:: Texture_set_value

    Override the texture contents with the provided linearized 1D array.

    In CUDA mode, when both the argument ``migrate`` and :py:func:`use_accel()` are ``True``,
    the texture exclusively stores a copy of the input data as a CUDA texture to avoid
    redundant storage.Note that the texture is still differentiable even when migrated.

.. topic:: Texture_set_tensor

    Override the texture contents with the provided tensor.

    This method updates the values of all texels. Changing the texture
    resolution or its number of channels is also supported. However, on CUDA,
    such operations have a significantly larger overhead (the GPU pipeline
    needs to be synchronized for new texture objects to be created).

    In CUDA mode, when both the argument ``migrate`` and :py:func:`use_accel()` are ``True``,
    the texture exclusively stores a copy of the input data as a CUDA texture to avoid
    redundant storage.Note that the texture is still differentiable even when migrated.

.. topic:: Texture_value

    Return the texture data as an array object

.. topic:: Texture_tensor

    Return the texture data as a tensor object

.. topic:: Texture_filter_mode

    Return the filter mode

.. topic:: Texture_wrap_mode

    Return the wrap mode

.. topic:: Texture_use_accel

    Return whether texture uses the GPU for storage and evaluation

.. topic:: Texture_migrated

    Return whether textures with :py:func:`use_accel()` set to ``True`` only store
    the data as a hardware-accelerated CUDA texture.

    If ``False`` then a copy of the array data will additionally be retained .

.. topic:: Texture_shape

    Return the texture shape

.. topic:: Texture_eval

    Evaluate the linear interpolant represented by this texture.

.. topic:: Texture_eval_fetch

    Fetch the texels that would be referenced in a texture lookup with
    linear interpolation without actually performing this interpolation.

.. topic:: Texture_eval_cubic

    Evaluate a clamped cubic B-Spline interpolant represented by this
    texture

    Instead of interpolating the texture via B-Spline basis functions, the
    implementation transforms this calculation into an equivalent weighted
    sum of several linear interpolant evaluations. In CUDA mode, this can
    then be accelerated by hardware texture units, which runs faster than
    a naive implementation. More information can be found in:

        GPU Gems 2, Chapter 20, "Fast Third-Order Texture Filtering"
        by Christian Sigg.

    When the underlying grid data and the query position are differentiable,
    this transformation cannot be used as it is not linear with respect to position
    (thus the default AD graph gives incorrect results). The implementation
    calls :py:func:`eval_cubic_helper()` function to replace the AD graph with a
    direct evaluation of the B-Spline basis functions in that case.

.. topic:: Texture_eval_cubic_grad

    Evaluate the positional gradient of a cubic B-Spline

    This implementation computes the result directly from explicit
    differentiated basis functions. It has no autodiff support.

    The resulting gradient and hessian have been multiplied by the spatial extents
    to count for the transformation from the unit size volume to the size of its
    shape.

.. topic:: Texture_eval_cubic_hessian

    Evaluate the positional gradient and hessian matrix of a cubic B-Spline

    This implementation computes the result directly from explicit
    differentiated basis functions. It has no autodiff support.

    The resulting gradient and hessian have been multiplied by the spatial extents
    to count for the transformation from the unit size volume to the size of its
    shape.

.. topic:: Texture_eval_cubic_helper

    Helper function to evaluate a clamped cubic B-Spline interpolant

    This is an implementation detail and should only be called by the
    :py:func:`eval_cubic()` function to construct an AD graph. When only the cubic
    evaluation result is desired, the :py:func:`eval_cubic()` function is faster
    than this simple implementation

.. topic:: scatter_inc

    Atomically increment a value within an unsigned 32-bit integer array and return
    the value prior to the update.

    This operation works just like the :py:func:`drjit.scatter_reduce()` operation
    for 32-bit unsigned integer operands, but with a fixed ``value=1`` parameter
    and ``op=ReduceOp::Add``.

    The main difference is that this variant additionally returns the *old* value
    of the target array prior to the atomic update in contrast to the more general
    scatter-reduction, which just returns ``None``. The operation also supports
    masking---the return value in the unmasked case is undefined. Both ``target``
    and ``index`` parameters must be 1D unsigned 32-bit arrays.

    This operation is a building block for stream compaction: threads can
    scatter-increment a global counter to request a spot in an array and then write
    their result there. The recipe for this is look as follows:

    .. code-block:: python

       data_1 = ...
       data_2 = ...
       active = drjit.ones(Bool, len(data_1)) # .. or a more complex condition

       # This will hold the counter
       ctr = UInt32(0)

       # Allocate output buffers
       max_size = 1024
       data_compact_1 = dr.empty(Float, max_size)
       data_compact_2 = dr.empty(Float, max_size)

       idx = dr.scatter_inc(target=ctr, index=UInt32(0), mask=active)

       # Disable dr.scatter() operations below in case of a buffer overflow
       active &= idx < max_size

       dr.scatter(
           target=data_compact_1,
           value=data_1,
           index=my_index,
           mask=active
       )

       dr.scatter(
           target=data_compact_2,
           value=data_2,
           index=my_index,
           mask=active
       )

    When following this approach, be sure to provide the same mask value to the
    :py:func:`drjit.scatter_inc()` and subsequent :py:func:`drjit.scatter()`
    operations.

    The function :py:func:`drjit.reshape` can be used to resize the resulting
    arrays to their compacted size. Please refer to the documentation of this
    function, specifically the code example illustrating the use of the
    ``shrink=True`` argument.

    The function :py:func:`drjit.scatter_inc()` exhibits the following unusual
    behavior compared to regular Dr.Jit operations: the return value references the
    instantaneous state during a potentially large sequence of atomic operations.
    This instantaneous state is not reproducible in later kernel evaluations, and
    Dr.Jit will refuse to do so when the computed index is reused. In essence, the
    variable is "consumed" by the process of evaluation.

    .. code-block:: python

       my_index = dr.scatter_inc(target=ctr, index=UInt32(0), mask=active)
       dr.scatter(
           target=data_compact_1,
           value=data_1,
           index=my_index,
           mask=active
       )

       dr.eval(data_compact_1) # Run Kernel #1

       dr.scatter(
           target=data_compact_2,
           value=data_2,
           index=my_index, # <-- oops, reusing my_index in another kernel.
           mask=active     #     This raises an exception.
       )

    To get the above code to work, you will need to evaluate ``my_index`` at the
    same time to materialize it into a stored (and therefore trivially
    reproducible) representation. For this, ensure that the size of the ``active``
    mask matches ``len(data_*)`` and that it is not the trivial ``True`` default
    mask (otherwise, the evaluated ``my_index`` will be scalar).

    .. code-block:: python

       dr.eval(data_compact_1, my_index)

    Such multi-stage evaluation is potentially inefficient and may defeat the
    purpose of performing stream compaction in the first place. In general, prefer
    keeping all scatter operations involving the computed index in the same kernel,
    and then this issue does not arise.

    The implementation of :py:func:`drjit.scatter_inc()` performs a local reduction
    first, followed by a single atomic write per SIMD packet/warp. This is done to
    reduce contention from a potentially very large number of atomic operations
    targeting the same memory address. Fully masked updates do not cause memory
    traffic.

    There is some conceptual overlap between this function and
    :py:func:`drjit.compress()`, which can likewise be used to reduce a stream to a
    smaller subset of active items. The downside of :py:func:`drjit.compress()` is
    that it requires evaluating the variables to be reduced, which can be very
    costly in terms of of memory traffic and storage footprint. Reducing through
    :py:func:`drjit.scatter_inc()` does not have this limitation: it can operate on
    symbolic arrays that greatly exceed the available device memory. One advantage
    of :py:func:`drjit.compress()` is that it essentially boils down to a
    relatively simple prefix sum, which does not require atomic memory operations
    (these can be slow in some cases).

.. topic:: scatter_add_kahan

    Perform a Kahan-compensated atomic scatter-addition.

    Atomic floating point accumulation can incur significant rounding error when
    many values contribute to a single element. This function implements an
    error-compensating version of :py:func:`drjit.scatter_add` based on the
    `Kahan-BabuÅ¡ka-Neumeier algorithm <https://en.wikipedia.org/wiki/Kahan_summation_algorithm>`__
    that simultaneously accumulates into *two* target buffers.

    In particular, the operation first accumulates a values into entries of a
    dynamic 1D array ``target_1``. It tracks the round-off error caused by this
    operation and then accumulates this error into a *second* 1D array named
    ``target_2``. At the end, the buffers can simply be added together to obtain
    the error-compensated result.

    This function has a few limitations: in contrast to
    :py:func:`drjit.scatter_reduce` and :py:func:`drjit.scatter_add`, it does not
    perform a local reduction (see flag :py:attr:`JitFlag.ScatterReduceLocal`),
    which can be an important optimization when atomic accumulation is a
    performance bottleneck.

    Furthermore, the function currently works with flat 1D arrays. This is an
    implementation limitation that could in principle be removed in the future.

    Finally, the function is differentiable, but derivatives currently only
    propagate into ``target_1``. This means that forward derivatives don't enjoy
    the error compensation of the primal computation. This limitation is of no
    relevance for backward derivatives.

.. topic:: format

    Return a formatted string representation.

    This function generates a formatted string representation as specified by a
    *format string* ``fmt`` and then returns it as a Python ``str`` object. The
    operation fetches referenced positional and keyword arguments and pretty-prints
    Dr.Jit arrays, tensors, and :ref:`PyTrees <pytrees>` with indentation, field
    names, etc.

    .. code-block:: pycon

       >>> from drjit.cuda import Array3f
       >>> s = dr.format("{}:\n{foo}",
       ...               "A PyTree containing an array",
       ...               foo={ 'a' : Array3f(1, 2, 3) })
       >>> print(s)
       A PyTree containing an array:
       {
         'a': [[1, 2, 3]]
       }

    Dynamic arrays with more than 20 entries will be abbreviated. Specify the
    ``limit=..`` argument to reveal the contents of larger arrays.

    .. code-block:: pycon

       >>> dr.format(dr.arange(dr.llvm.Int, 30))
       [0, 1, 2, .. 24 skipped .., 27, 28, 29]

       >>> dr.format(dr.arange(dr.llvm.Int, 30), limit=30)
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,↵
        23, 24, 25, 26, 27, 28, 29]

    This function lacks many features of Python's (rather intricate)
    `format string mini language <https://docs.python.org/3/library/string.html#formatspec>`__ and
    `f-string interpolation <https://peps.python.org/pep-0498/>`__. However, a subset of the
    functionality is supported:

    - Positional arguments (in ``*args``) can be referenced implicitly (``{}``), or
      using indices (``{0}``, ``{1}``, etc.). Those conventions should not be mixed.
      Unreferenced positional arguments will be silently ignored.

    - Keyword arguments (in ``**kwargs``) can be referenced via their keyword name
      (``{foo}``). Unreferenced keywords will be silently ignored.

    - A trailing ``=`` in a brace expression repeats the string within the braces followed
      by the output:

      .. code-block:: pycon

         >>> dr.format('{foo=}', foo=1)
         foo=1

    When the format string ``fmt`` is omitted, it is implicitly set to ``{}``, and the
    function formats a single positional argument.

    In contrast to the related :py:func:`drjit.print()` , this function does not
    output the result on the console, and it cannot support symbolic inputs. This
    is because returning a string right away is incompatible with the requirement
    of evaluating/formatting symbolic inputs in a delayed fashion. If you wish to
    format symbolic arrays, you must call :py:func:`drjit.print()` with a custom
    ``file`` object that implements the ``.write()`` function. Dr.Jit will call
    this function with the generated string when it is ready.

    Args:
        fmt (str): A format string that potentially references input arguments
          from ``*args`` and ``**kwargs``.

        limit (int): The operation will abbreviate dynamic arrays with more than
          ``limit`` (default: 20) entries.

    Returns:
        str: The formatted string representation created as specified above.

.. topic:: print

    Generate a formatted string representation and print it immediately or
    in a delayed fashion (if any of the inputs are symbolic).

    This function combines the behavior of the built-in Python ``format()`` and
    ``print()`` functions: it generates a formatted string representation as
    specified by a *format string* ``fmt`` and then outputs it on the console.
    The operation fetches referenced positional and keyword arguments and
    pretty-prints Dr.Jit arrays, tensors, and :ref:`PyTrees <pytrees>` with
    indentation, field names, etc.

    .. code-block:: pycon

       >>> from drjit.cuda import Array3f
       >>> dr.print("{}:\n{foo}",
       ...          "A PyTree containing an array",
       ...          foo={ 'a' : Array3f(1, 2, 3) })
       A PyTree containing an array:
       {
         'a': [[1, 2, 3]]
       }

    The key advance of :py:func:`drjit.print` compared to the built-in Python
    ``print()`` statement is that it can run asynchronously, which allows it to
    print *symbolic variables* without requiring their evaluation. Dr.Jit uses
    symbolic variables to trace loops (:py:func:`drjit.while_loop`), conditionals
    (:py:func:`drjit.if_stmt`), and calls (:py:func:`drjit.switch`,
    :py:func:`drjit.dispatch`). Such symbolic variables represent values that are
    unknown at trace time, and which cannot be printed using the built-in Python
    ``print()`` function (attempting to do so will raise an exception).

    When the print statement does not reference any symbolic arrays or tensors, it
    will execute immediately. Otherwise, the output will appear after the next
    :py:func:`drjit.eval()` statement, or whenever any subsequent computation is
    evaluated. Here is an example from an interactive Python session demonstrating
    printing from a symbolic call performed via :py:func:`drjit.switch()`:

    .. code-block:: pycon

       >>> from drjit.llvm import UInt, Float
       >>> def f1(x):
       ...     dr.print("in f1: {x=}", x=x)
       ...
       >>> def f2(x):
       ...     dr.print("in f2: {x=}", x=x)
       ...
       >>> dr.switch(index=UInt(0, 0, 0, 1, 1, 1),
       ...           targets=[f1, f2],
       ...           x=Float(1, 2, 3, 4, 5, 6))
       >>> # No output (yet)
       >>> dr.eval()
       in f1: x=[1, 2, 3]
       in f2: x=[4, 5, 6]

    Dynamic arrays with more than 20 entries will be abbreviated. Specify the
    ``limit=..`` argument to reveal the contents of larger arrays.

    .. code-block:: pycon

       >>> dr.print(dr.arange(dr.llvm.Int, 30))
       [0, 1, 2, .. 24 skipped .., 27, 28, 29]

       >>> dr.format(dr.arange(dr.llvm.Int, 30), limit=30)
       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,↵
        23, 24, 25, 26, 27, 28, 29]

    This function lacks many features of Python's (rather intricate)
    `format string mini language <https://docs.python.org/3/library/string.html#formatspec>`__ and
    `f-string interpolation <https://peps.python.org/pep-0498/>`__. However, a subset of the
    functionality is supported:

    - Positional arguments (in ``*args``) can be referenced implicitly (``{}``), or
      using indices (``{0}``, ``{1}``, etc.). Those conventions should not be mixed.
      Unreferenced positional arguments will be silently ignored.

    - Keyword arguments (in ``**kwargs``) can be referenced via their keyword name
      (``{foo}``). Unreferenced keywords will be silently ignored.

    - A trailing ``=`` in a brace expression repeats the string within the braces followed
      by the output:

      .. code-block:: pycon

         >>> dr.print('{foo=}', foo=1)
         foo=1

    When the format string ``fmt`` is omitted, it is implicitly set to ``{}``, and the
    function formats a single positional argument.

    The function implicitly appends ``end`` to the format string, which is set to a
    newline by default. The final result is sent to ``sys.stdout`` (by default) or
    ``file``. When a ``file`` argument is given, it must implement the method
    ``write(arg: str)``.

    A related operation :py:func:`drjit.format()` admits the same format string
    syntax but returns a Python ``str`` instead of printing to the console. This
    operation, however, does not support symbolic inputs---use
    :py:func:`drjit.print()` with a custom ``file`` argument to stringify symbolic
    inputs asynchronously.

    .. note::

       This operation is not suitable for extracting large amounts of data from
       Dr.Jit kernels, as the conversion to a string representation incurs a
       nontrivial runtime cost.

    .. note::

       **Technical details on symbolic printing**

       When Dr.Jit compiles and executes queued computation on the target device,
       it includes additional code for symbolic print operations that that captures
       referenced arguments and copies them back to the host (CPU). The information
       is then printed following the end of that process.

       Only a limited amount of memory is set aside to capture the output of
       symbolic print operations. This is because the amount of data produced
       within long-running symbolic loops can often exceed the total device
       memory. Also, printing gigabytes of ASCII text into a Python console or
       Jupyter notebook is likely not a good idea.

       For the electronically inclined, the operation is best thought of as hooking
       up an oscilloscope to a high-frequency circuit. The oscilloscope provides a
       limited view into a vast torrent of data to assist the user, who would be
       overwhelmed if the oscilloscope worked by capturing and showing everything.

       The operation warns when the size of the buffers was insufficient. In this
       case, the output is still printed in the correct order, but chunks of the
       data are missing. The position of the resulting holes is unspecified and
       non-deterministic.

       .. code-block:: pycon
          :emphasize-lines: 5-8

          >>> dr.print(dr.arange(Float, 10000000), method='symbolic')
          >>> dr.eval()
          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

          RuntimeWarning: dr.print(): symbolic print statement only captured 20 of 10000000 available outputs.
          The above is a non-deterministic sample, in which entries are in the right order but not necessarily
          contiguous. Specify `limit=..` to capture more information and/or add the special format field
          `{thread_id}` show the thread ID/array index associated with each entry of the captured output.

       This is because the (many) parallel threads of the program all try to append
       their state to the output buffer, but only the first ``limit`` (20 by
       default) succeed. The host subsequently re-sorts the captured data by thread
       ID. This means that the output ``[5, 6, 102, 188, 1026, ..]`` would also be
       a valid result of the prior command. When a print statement references
       multiple arrays, then the operations either shows all array entries
       associated with a particular execution thread, or none of them.

       To refine what is captured, you can specify the ``active`` argument to
       disable the print statement for a subset of the entries (a "trigger" in
       the oscilloscope analogy). Printing from an inactive thread within a
       symbolic loop (:py:func:`drjit.while_loop`), conditional
       (:py:func:`drjit.if_stmt`), or call (:py:func:`drjit.switch`,
       :py:func:`drjit.dispatch`) will likewise not generate any output.

       A potential gotcha of the current design is that a symbolic print within a
       symbolic loop counts as one print statement and will only generate a single
       combined output string. The output of each thread is arranged in one
       contiguous block. You can add the special format string keyword
       ``{thread_id}`` to reveal the mapping between output values and the
       execution thread that generated them:

       .. code-block:: pycon

          >>> from drjit.llvm import Int
          >>> @dr.syntax
          >>> def f(j: Int):
          ...     i = Int(0)
          ...     while i < j:
          ...         dr.print('{thread_id=} {i=}', i=i)
          ...         i += 1
          ...
          >>> f(Int(2, 3))
          >>> dr.eval();
          thread_id=[0, 0, 1, 1, 1], i=[0, 1, 0, 1, 2]

       The example above runs a symbolic loop twice in parallel: the first thread
       runs for for 2 iterations, and the second runs for 3 iterations. The loop
       prints the iteration counter ``i``, which then leads to the output ``[0, 1,
       0, 1, 2]`` where the first two entries are produced by the first thread, and
       the trailing three belong to the second thread. The ``thread_id`` output
       clarifies this mapping.

    Args:
        fmt (str): A format string that potentially references input arguments
          from ``*args`` and ``**kwargs``.

        active (drjit.ArrayBase | bool): A mask argument that can be used to
          disable a subset of the entries. The print statement will be completely
          suppressed when there is no output. (default: ``True``).

        end (str): This string will be appended to the format string. It is
          set to a newline character (``"\n"``) by default.

        file (object): The print operation will eventually invoke
          ``file.write(arg:str)`` to print the formatted string. Specify this
          argument to route the output somewhere other than the default output
          stream ``sys.stdout``.

        mode (str): Specify this parameter to override the evaluation mode.
          Possible values are: ``"symbolic"``, ``"evaluated"``, or ``"auto"``. The
          default value of ``"auto"`` causes the function to use evaluated mode
          (which prints immediately) unless a symbolic input is detected, in which
          case printing takes place symbolically (i.e., in a delayed fashion).

        limit (int): The operation will abbreviate dynamic arrays with more than
          ``limit`` (default: 20) entries.

.. topic:: thread_count

    Return the number of threads that Dr.Jit uses to parallelize computation on the CPU

.. topic:: set_thread_count

    Adjust the number of threads that Dr.Jit uses to parallelize computation on the CPU.

    The thread pool is primarily used by Dr.Jit's LLVM backend. Other projects using
    underlying `nanothread <https://github.com/mitsuba-renderer/nanothread>`__
    thread pool library will also be affected by changes performed using by this
    function. It is legal to call it even while parallel computation is currently
    ongoing.

.. topic:: intrusive_base

    Base class with intrusive combined C++/Python reference counting.

.. topic:: detail_clear_registry

    Clear all instances that are currently registered with Dr.Jit's instance
    registry. This is may be needed in a very specific corner case: when a large
    program (e.g., a test suite) dispatches function calls via instance arrays, and
    when such a test suite raises exceptions internally and holds on to them (which
    is e.g., what PyTest does to report errors all the way at the end), then the
    referenced instances may remain alive beyond their usual lifetime. This can
    have an unintended negative effect by influencing subsequent tests that must
    now also consider the code generated by these instances (in particular,
    failures due to unimplemented functions

.. topic:: width

    Returns the *vectorization width* of the provided input(s), which is defined as
    the length of the last dynamic dimension.

    When working with Jit-compiled CUDA or LLVM-based arrays, this corresponds to
    the number of items being processed in parallel.

    The function raises an exception when the input(s) is ragged, i.e., when it
    contains arrays with incompatible sizes. It returns ``1`` if if the input is
    scalar and/or does not contain any Dr.Jit arrays.

    Args:
        arg (object): An arbitrary Dr.Jit array or :ref:`PyTree <pytrees>`.

    Returns:
        int: The width of the provided input(s).

.. topic:: popcnt

    Return the number of nonzero zero bits.

    This function evaluates the component-wise population count of the input
    scalar, array, or tensor. This function assumes that ``arg`` is either an
    arbitrary Dr.Jit integer array or a 32 bit-sized scalar integer value.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of nonzero zero bits in ``arg``

.. topic:: lzcnt

    Return the number of leading zero bits.

    This function evaluates the component-wise leading zero count of the input
    scalar, array, or tensor. This function assumes that ``arg`` is either an
    arbitrary Dr.Jit integer array or a 32 bit-sized scalar integer value.

    The operation is well-defined when ``arg`` is zero.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of leading zero bits in ``arg``

.. topic:: tzcnt

    Return the number of trailing zero bits.

    This function evaluates the component-wise trailing zero count of the input
    scalar, array, or tensor. This function assumes that ``arg`` is either an
    arbitrary Dr.Jit integer array or a 32 bit-sized scalar integer value.

    The operation is well-defined when ``arg`` is zero.

    Args:
        arg (int | drjit.ArrayBase): A Python or Dr.Jit array

    Returns:
        int | drjit.ArrayBase: number of trailing zero bits in ``arg``

.. topic:: brev

    Reverse the bit representation of an integer value or array.

    This function assumes that ``arg`` is either an arbitrary Dr.Jit integer
    array or a 32 bit-sized scalar integer value.

    Args:
        arg (int | drjit.ArrayBase): A Python ``int`` or Dr.Jit integer array.

    Returns:
        int | drjit.ArrayBase: the bit-reversed version of ``arg``.

.. topic:: compress

    Compress a mask into an array of nonzero indices.

    This function takes an boolean array as input and then returns an unsigned
    32-bit integer array containing the indices of nonzero entries.

    It can be used to reduce a stream to a subset of active entries via the
    following recipe:

    .. code-block:: python

       # Input: an active mask and several arrays data_1, data_2, ...
       dr.schedule(active, data_1, data_2, ...)
       indices = dr.compress(active)
       data_1 = dr.gather(type(data_1), data_1, indices)
       data_2 = dr.gather(type(data_2), data_2, indices)
       # ...

    There is some conceptual overlap between this function and
    :py:func:`drjit.scatter_inc()`, which can likewise be used to reduce a
    stream to a smaller subset of active items. Please see the documentation of
    :py:func:`drjit.scatter_inc()` for details.

    .. danger::
       This function internally performs a synchronization step.

    Args:
        arg (bool | drjit.ArrayBase): A Python or Dr.Jit boolean type

    Returns:
        Array of nonzero indices

.. topic:: count

    Compute the number of active entries along the given axis.

    Given a boolean-valued input array, tensor, or Python sequence, this function
    reduces elements using the ``+`` operator (interpreting ``True`` elements as
    ``1`` and ``False`` elements as ``0``). It returns an unsigned 32-bit version
    of the input array.

    By default, it reduces along index ``0``, which refers to the outermost axis.
    Negative indices (e.g. ``-1``) count backwards from the innermost axis. The
    special argument ``axis=None`` causes a simultaneous reduction over all axes.
    Note that the reduced form of an *empty* array is considered to be zero.

    See the section on :ref:`horizontal reductions <horizontal-reductions>` for
    important general information about their properties.

    Args:
        value (bool | Sequence | drjit.ArrayBase): A Python or Dr.Jit mask type

        axis (int | None): The axis along which to reduce. The default value of
          ``0`` refers to the outermost axis. Negative values count backwards from
          the innermost axis. A value of ``None`` causes a simultaneous reduction
          along all axes.

    Returns:
        int | drjit.ArrayBase: Result of the reduction operation

.. topic:: sync_thread

    Wait for all currently running computation to finish.

    This function synchronizes the device (e.g. the GPU) with the host (CPU) by
    waiting for the termination of all computation enqueued by the current host
    thread.

    One potential use of this function is to measure the runtime of a kernel
    launched by Dr.Jit. We instead recommend the use of the
    :py:func:`drjit.kernel_history()`, which exposes more accurate device timers.

    In general, calling this function in user code is considered **bad practice**.
    Dr.Jit programs "run ahead" of the device to keep it fed with work. This is
    important for performance, and :py:func:`drjit.sync_thread` breaks this
    optimization.

    All operations sent to a device (including reads) are strictly ordered, so
    there is generally no reason to wait for this queue to empty. If you find
    that :py:func:`drjit.sync_thread` is needed for your program to run correctly,
    then you have found a bug. Please report it on the project's
    `GitHub issue tracker <https://github.com/mitsuba-renderer/drjit>`__.

.. topic:: flush_malloc_cache

    Free the memory allocation cache maintained by Dr.Jit.

    Allocating and releasing large chunks of memory tends to be relatively
    expensive, and Dr.Jit programs often need to do so at high rates.

    Like most other array programming frameworks, Dr.Jit implements an internal
    cache to reduce such allocation-related costs. This cache starts out empty and
    grows on demand. Allocated memory is never released by default, which can be
    problematic when using multiple array programming frameworks within the same
    Python session, or when running multiple processes in parallel.

    The :py:func:`drjit.flush_malloc_cache` function releases all currently unused
    memory back to the operating system. This is a relatively expensive step: you
    likely don't want to use it within a performance-sensitive program region (e.g.
    an optimization loop).

.. topic:: flush_kernel_cache

    Release all currently cached kernels.

    When Dr.Jit evaluates a previously unseen computation, it compiles a kernel and
    then maps it into the memory of the CPU or GPU. The kernel stays resident so
    that it can be immediately reused when that same computation reoccurs at a
    later point.

    In long development sessions (e.g. a Jupyter notebook-based prototyping),
    this cache may eventually become unreasonably large, and calling
    :py:func:`flush_kernel_cache` to free it may be advisable.

    Note that this does not free the *disk cache* that also exists to preserve compiled
    programs across sessions. To clear this cache as well, delete the directory
    ``$HOME/.drjit`` on Linux/macOS, and ``%AppData%\Local\Temp\drjit`` on Windows.
    (The ``AppData`` folder is typically found in ``C:\Users\<your username>``).

.. topic:: kernel_history

    Return the history of captured kernel launches.

    Dr.Jit can optionally capture performance-related metadata. To do so, set the
    :py:attr:`drjit.JitFlag.KernelHistory` flag as follows:

    .. code-block:: python

       with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
          # .. computation to be analyzed ..

       hist = dr.kernel_history()

    The :py:func:`drjit.kernel_history()` function returns a list of dictionaries
    characterizing each major operation performed by the analyzed region. This
    dictionary has the following entries

    - ``backend``: The used JIT backend.

    - ``execution_time``: The time (in microseconds) used by this operation.

      On the CUDA backend, this value is captured via CUDA events. On the LLVM
      backend, this involves querying ``CLOCK_MONOTONIC`` (Linux/macOS) or
      ``QueryPerformanceCounter`` (Windows).

    - ``type``: The type of computation expressed by an enumeration value of type
      :py:class:`drjit.KernelType`. The most interesting workload generated by Dr.Jit
      are just-in-time compiled kernels, which are identified by :py:attr:`drjit.KernelType.JIT`.

      These have several additional entries:

      - ``hash``: The hash code identifying the kernel. (This is the same hash code is
        also shown when increasing the log level via :py:func:`drjit.set_log_level`).

      - ``ir``: A capture of the intermediate representation used in this kernel.

      - ``operation_count``: The number of low-level IR operations. (A rough
        proxy for the complexity of the operation.)

      - ``cache_hit``: Was this kernel present in Dr.Jit's in-memory cache?
        Otherwise, it as either loaded from memory or had to be recompiled from scratch.

      - ``cache_disk``: Was this kernel present in Dr.Jit's on-disk cache?
        Otherwise, it had to be recompiled from scratch.

      - ``codegen_time``: The time (in microseconds) which Dr.Jit needed to
        generate the textual low-level IR representation of the kernel. This
        step is always needed even if the resulting kernel is already cached.

      - ``backend_time``: The time (in microseconds) which the backend (either the
        LLVM compiler framework or the CUDA PTX just-in-time compiler) required to
        compile and link the low-level IR into machine code. This step is only
        needed when the kernel did not already exist in the in-memory or on-disk cache.

      - ``uses_optix``: Was this kernel compiled by the
        `NVIDIA OptiX <https://developer.nvidia.com/rtx/ray-tracing/optix>`__
        ray tracing engine?

    Note that :py:func:`drjit.kernel_history()` clears the history while extracting
    this information. A related operation :py:func:`drjit.kernel_history_clear()`
    *only* clears the history without returning any information.

.. topic:: kernel_history_clear

    Clear the kernel history.

    This operation clears the kernel history without returning any information
    about it. See :py:func:`drjit.kernel_history` for details.

.. topic:: detail_any_symbolic

    Returns ``true`` if any of the values in the provided PyTree are symbolic variables.

.. topic:: slice

    Select a subset of the input array or PyTree along the trailing dynamic dimension.

    Given a Dr.Jit array ``value`` with shape ``(..., N)`` (where ``N`` represents
    a dynamically sized dimension), this operation effectively evaluates the
    expression ``value[..., index]``. It recursively traverses :ref:`PyTrees
    <pytrees>` and transforms each compatible array element. Other values are
    returned unchanged.

    The following properties of ``index`` determine the return type:

    - When ``index`` is a 1D integer array, the operation reduces to one or more
      calls to :py:func:`drjit.gather`, and :py:func:`slice` returns a reduced
      output object of the same type and structure.

    - When ``index`` is a scalar Python ``int``, the trailing dimension is entirely
      removed, and the operation returns an array from the ``drjit.scalar``
      namespace containing the extracted values.

.. topic:: profile_mark

    Mark an event on the timeline of profiling tools.

    Currently, this function uses `NVTX <https://github.com/NVIDIA/NVTX>`__ to report
    events that can be captured using `NVIDIA Nsight Systems
    <https://developer.nvidia.com/nsight-systems>`__. The operation is a no-op when
    no profile collection tool is attached.

    Note that this event will be recorded on the CPU timeline.

.. topic:: profile_range

    Context manager to mark a region (e.g. a function call) on the timeline of
    profiling tools.

    You can use this context manager to wrap parts of your code and track when and
    for how long it runs. Regions can be arbitrarily nested, which profiling tools
    visualize as a stack.

    .. code-block: python

       with dr.profile_range("Costly preprocessing"):
           init_data_structures()

    Note that this function is intended to track activity on the CPU timeline. If
    the wrapped region launches asynchronous GPU kernels, then those won't
    generally be included in the length of the range unless
    :py:func:`drjit.sync_thread` or some other type of synchronization operation
    waits for their completion (which is generally not advisable, since keeping CPU
    and GPU asynchronous with respect to each other improves performance).

    Currently, this function uses `NVTX <https://github.com/NVIDIA/NVTX>`__ to report
    events that can be captured using `NVIDIA Nsight Systems
    <https://developer.nvidia.com/nsight-systems>`__. The operation is a no-op when
    no profile collection tool is attached.

.. topic:: ReduceMode

    Compilation strategy for atomic scatter-reductions.

    Elements of of this enumeration determine how Dr.Jit executes *atomic
    scatter-reductions*, which refers to indirect writes that update an existing
    element in an array, while avoiding problems arising due to concurrency.

    Atomic scatter-reductions can have a *significant* detrimental impact on
    performance. When many threads in a parallel computation attempt to modify the
    same element, this can lead to *contention*---essentially a fight over which
    part of the processor **owns** the associated memory region, which can slow
    down a computation by many orders of magnitude.

    This parameter also plays an important role for :py:func:`drjit.gather`, which
    is nominally a read-only operation. This is because the reverse-mode derivative
    of a gather turns it into an atomic scatter-addition, where further context on
    how to compile the operation is needed.

    Dr.Jit implements several strategies to address contention, which can be
    selected by passing the optional ``mode`` parameter to
    :py:func:`drjit.scatter_reduce`, :py:func:`drjit.scatter_add`, and
    :py:func:`drjit.gather`.

    If you find that a part of your program is bottlenecked by atomic writes, then
    it may be worth explicitly specifying some of the strategies below to see which
    one performs best.

.. topic:: ReduceMode_Auto

    Select the first valid option from the following list:

    - use :py:attr:`drjit.ReduceMode.Expand` if the computation uses the LLVM
      backend and the ``target`` array storage size is smaller or equal
      than than the value given by :py:func:`drjit.expand_threshold`. This
      threshold can be changed using the :py:func:`drjit.set_expand_threshold`
      function.

    - use :py:attr:`drjit.ReduceMode.Local` if
      :py:attr:`drjit.JitFlag.ScatterReduceLocal` is set.

    - fall back to :py:attr:`drjit.ReduceMode.Direct`.

.. topic:: ReduceMode_Direct

    Insert an ordinary atomic reduction operation into the program.

    This mode is ideal when no or little contention is expected, for example
    because the target indices of scatters are well spread throughout the target
    array. This mode generates a minimal amount of code, which can help improve
    performance especially on GPU backends.

.. topic:: ReduceMode_Local

    Locally pre-reduce operands.

    In this mode, Dr.Jit adds extra code to the compiled program to examine the
    target indices of atomic updates. For example, CUDA programs run with an
    instruction granularity referred to as a *warp*, which is a group of 32
    threads. When some of these threads want to write to the same location, then
    those operands can be pre-processed to reduce the total number of necessary
    atomic memory transactions (potentially to just a single one!)

    On the CPU/LLVM backend, the same process works at the granularity of
    *packets*. The details depends on the underlying instruction set---for
    example, there are 16 threads per packet on a machine with AVX512, so
    there is a potential for reducing atomic write traffic by that factor.

.. topic:: ReduceMode_NoConflicts

    Perform a non-atomic read-modify-write operation.

    This mode is only safe in specific situations. The caller must guarantee that
    there are no conflicts (i.e., scatters targeting the same elements). If
    specified, Dr.Jit will generate a *non-atomic* read-modify-update operation
    that potentially runs significantly faster, especially on the LLVM backend.

.. topic:: ReduceMode_Expand

    Expand the target array to avoid write conflicts, then scatter non-atomically.

    This feature is only supported on the LLVM backend. Other backends interpret
    this flag as if :py:attr:`drjit.ReduceMode.Auto` had been specified.

    This mode internally expands the storage underlying the target array to a much
    larger size that is proportional to the number of CPU cores. Scalar (length-1)
    target arrays are expanded even further to ensure that each CPU gets an
    entirely separate cache line.

    Following this one-time expansion step, the array can then accommodate an
    arbitrary sequence of scatter-reduction operations that the system will
    internally perform using *non-atomic* read-modify-write operations (i.e.,
    analogous to the :py:attr:`NoConflicts` mode). Dr.Jit automatically
    re-compress the array into the ordinary representation.

    On bigger arrays and on machines with many cores, the storage costs resulting
    from this mode can be prohibitive.

.. topic:: ReduceMode_Permute

    In contrast to prior enumeration entries, this one modifies plain
    (non-reductive) scatters and gathers. It exists to enable internal
    optimizations that Dr.Jit uses when differentiating vectorized function
    calls and compressed loops. You likely should not use it in your own code.

    When setting this mode, the caller guarantees that there will be no
    conflicts, and that every entry is written exactly single time using an
    index vector representing a permutation (it's fine this permutation is
    accomplished by multiple separate write operations, but there should be no
    more than 1 write to each element).

    Giving 'Permute' as an argument to a (nominally read-only) gather
    operation is helpful because we then know that the reverse-mode derivative
    of this operation can be a plain scatter instead of a more costly
    atomic scatter-add.

    Giving 'Permute' as an argument to a scatter operation is helpful
    because we then know that the forward-mode derivative does not depend
    on any prior derivative values associated with that array, as all
    current entries will be overwritten.

.. topic:: set_expand_threshold

    Set the threshold for performing scatter-reductions via expansion.

    The documentation of :py:class:`drjit.ReduceOp` explains the cost of atomic
    scatter-reductions and introduces various optimization strategies.

    One particularly effective optimization (the section on :ref:`optimizations
    <reduce-local>` for plots) named :py:attr:`drjit.ReduceOp.Expand` is specific
    to the LLVM backend. It replicates the target array to avoid write conflicts
    altogether, which enables the use of non-atomic memory operations. This is
    *significantly* faster but also *very memory-intensive*. The storage cost of an
    1MB array targeted by a :py:func:`drjit.scatter_reduce` operation now grows to
    ``N`` megabytes, where ``N`` is the number of cores.

    For this reason, Dr.Jit implements a user-controllable threshold exposed via
    the functions :py:func:`drjit.expand_threshold` and
    :py:func:`drjit.set_expand_threshold`. When the array has more entries than the
    value specified here, the :py:attr:`drjit.ReduceOp.Expand` strategy will *not* be used
    unless specifically requested via the ``mode=`` parameter of operations like
    :py:func:`drjit.scatter_reduce`, :py:func:`drjit.scatter_add`, and
    :py:func:`drjit.gather`.

    The default value of this parameter is `1000000` (1 million entries).

.. topic:: expand_threshold

    Query the threshold for performing scatter-reductions via expansion.

    Getter for the quantity set in :py:func:`drjit.set_expand_threshold()`

.. topic:: reshape

    Converts ``value`` into an array of type ``dtype`` by rearranging the contents
    according to the specified shape.

    The parameter ``shape`` may contain a single ``-1``-valued target dimension, in
    which case its value is inferred from the remaining shape entries and the size
    of the input. When ``shape`` is of type ``int``, it is interpreted as a
    1-tuple ``(shape,)``.

    This function supports the following behaviors:

    1. **Reshaping tensors**: Dr.Jit :ref:`tensors <tensors>` admit arbitrary
       shapes. The :py:func:`drjit.reshape` can convert between them as long as the
       total number of elements remains unchanged.

       .. code-block:: pycon

          >>> from drjit.llvm.ad import TensorXf
          >>> value = dr.arange(TensorXf, 6)
          >>> dr.reshape(dtype=TensorXf, value=value, shape=(3, -1))
          [[0, 1]
           [2, 3]
           [4, 5]]

    2. **Reshaping nested arrays**: The function can ravel and unravel nested
       arrays (which have some static dimensions). This provides a high-level
       interface that subsumes the functions :py:func:`drjit.ravel` and
       :py:func:`drjit.unravel`.

       .. code-block:: pycon

          >>> from drjit.llvm.ad import Array2f, Array3f
          >>> value = Array2f([1, 2, 3], [4, 5, 6])
          >>> dr.reshape(dtype=Array3f, value=value, shape=(3, -1), order='C')
          [[1, 4, 2],
           [5, 3, 6]]
          >>> dr.reshape(dtype=Array3f, value=value, shape=(3, -1), order='F')
          [[1, 3, 5],
           [2, 4, 6]]

       (By convention, Dr.Jit nested arrays are :ref:`always printed in transposed
       <nested_array_transpose>` form, which explains the difference in output
       compared to the identically shaped Tensor example just above.)

       The ``order`` argument can be used to specify C (``"C"``) or Fortran
       (``"F"``)-style ordering when rearranging the array. The default value
       ``"A"`` corresponds to Fortran-style ordering.

    3. **PyTrees**: When ``value`` is a :ref:`PyTree <pytrees>`, the operation
       recursively threads through the tree's elements.

    3. **Stream compression and loops that fork recursive work**. When called with
       ``shrink=True``, the function creates a view of the original data that
       potentially has a smaller number of elements.

       The main use of this feature is to implement loops that process large
       numbers of elements in parallel, and which need to occasionally "fork"
       some recursive work. On modern compute accelerators, an efficient way
       to handle this requirement is to append this work into a queue that is
       processed in a subsequent pass until no work is left. The reshape operation
       with ``shrink=True`` then resizes the preallocated queue to the actual
       number of collected items, which are the input of the next iteration.

       Please refer to the following example  that illustrates how
       :py:func:`drjit.scatter_inc`, :py:func:`drjit.scatter`, and
       :py:func:`drjit.reshape(..., shrink=True) <drjit.reshape>` can be combined
       to realize a parallel loop with a fork condition

       .. code-block:: python

          @drjit.syntax
          def f():
              # Loop state variables (an arbitrary array or PyTree)
              state = ...

              # Determine how many elements should be processed
              size = dr.width(loop_state)

              # Run the following loop until no work is left
              while size > 0:
                  # 1-element array used as an atomic counter
                  queue_index = UInt(0)

                  # Preallocate memory for the queue. The necessary
                  # amount of memory is task-dependent
                  queue_size = size
                  queue = dr.empty(dtype=type(state), shape=queue_size)

                  # Create an opaque variable representing the number 'loop_state'.
                  # This keeps this changing value from being baked into the program,
                  # which is needed for proper kernel caching
                  queue_size_o = dr.opaque(UInt32, queue_size)

                  while not stopping_criterion(state):
                      # This line represents the loop body that processes work
                      state = loop_body(state)

                      # if the condition 'fork' is True, spawn a new work item that
                      # will be handled in a future iteration of the parent loop.

                      if fork(state):
                          # Atomically reserve a slot in 'queue'
                          slot = dr.scatter_inc(target=queue_index, index=0)

                          # Work item for the next iteration, task dependent
                          todo = state

                          # Be careful not to write beyond the end of the queue
                          valid = slot < queue_size_o

                          # Write 'todo' into the reserved slot
                          dr.scatter(target=queue, index=slot, value=todo, active=valid)

                 # Determine how many fork operations took place
                 size = queue_index[0]
                 if size > queue_size:
                     raise RuntimeError('Preallocated queue was too small: tried to store '
                                        f'{size} elements in a queue of size {queue_size}')

                 # Reshape the queue and re-run the loop
                 state = dr.reshape(dtype=type(state), value=queue, shape=size, shrink=True)

    Args:
        dtype (type): Desired output type of the reshaped array. This could
          equal ``type(value)`` or refer to an entirely different array type.

        value (object): An arbitrary Dr.Jit array, tensor, or :ref:`PyTree
          <pytrees>`. The function returns unknown objects of other types
          unchanged.

        shape (int|tuple[int, ...]): The target shape.

        order (str): A single character indicating the index order used to
          reinterpret the input. ``'F'`` indicates column-major/Fortran-style
          ordering, in which case the first index changes at the highest frequency.
          The alternative ``'C'`` specifies row-major/C-style ordering, in which
          case the last index changes at the highest frequency. The default value
          ``'A'`` (automatic) will use F-style ordering for arrays and C-style
          ordering for tensors.

        shrink (bool): Cheaply construct a view of the input that potentially
            has a smaller number of elements. The main use case of this method
            is explained above.

    Returns:
        object: The reshaped array or PyTree.

.. topic:: tile

    Tile the input array ``count`` times along the trailing dimension.

    This function replicates the input ``count`` times along the trailing dynamic
    dimension. It recursively threads through nested arrays and :ref:`PyTree
    <pytrees>`. Static arrays and tensors currently aren't supported.
    When ``count==1``, the function returns the input without changes.

    An example is shown below:

    .. code-block:

       from drjit.cuda import Float

       >>> x = Float(1, 2)
       >>> drjit.tile(x, 3)
       [1, 2, 1, 2, 1, 2]

    Args:
        value (drjit.ArrayBase): A Dr.Jit type or :ref:`PyTree <pytrees>`.

        count (int): Number of repetitions

    Returns:
        object: The tiled input as described above. The return type matches that of ``value``.

.. topic:: repeat

    Repeat each successive entry of the input ``count`` times along the trailing dimension.

    This function replicates the input ``count`` times along the trailing dynamic
    dimension. It recursively threads through nested arrays and :ref:`PyTree
    <pytrees>`. Static arrays and tensors currently aren't supported.
    When ``count==1``, the function returns the input without changes.

    An example is shown below:

    .. code-block:

       from drjit.cuda import Float

       >>> x = Float(1, 2)
       >>> drjit.tile(x, 3)
       [1, 1, 1, 2, 2, 2]

    Args:
        value (drjit.ArrayBase): A Dr.Jit type or :ref:`PyTree <pytrees>`.

        count (int): Number of repetitions

    Returns:
        object: The repeated input as described above. The return type matches that of ``value``.

.. topic:: block_reduce

    Reduce elements within blocks.

    This function reduces all elements within contiguous blocks of size
    ``block_size`` along the trailing dimension of the input array ``value``,
    returning a correspondingly smaller output array. Various types of
    reductions are supported (see :py:class:`drjit.ReduceOp` for details).

    For example, a sum reduction of a hypothetical array ``[a, b, c, d, e, f]``
    with ``block_size=2`` produces the output ``[a+b, c+d, e+f]``.

    The function raises an exception when the length of the trailing dimension
    is not a multiple of the block size.  It recursively threads through nested
    arrays and :ref:`PyTrees <pytrees>`.

    Dr.Jit uses one of two strategies to realize this operation, which can be
    optionally forced by specifying the ``mode`` parameter.

    - ``mode="evaluated"`` first evaluates the input array via
      :py:func:`drjit.eval()` and then launches a specialized reduction kernel.

      On the CUDA backend, this kernel makes efficient use of shared memory and
      cooperative warp instructions with the limitation that it requires
      ``block_size`` to be a power of two. The LLVM backend parallelizes the
      operation via the built-in thread pool and has no ``block_size``
      limitations.

    - ``mode="symbolic"`` uses :py:func:`drjit.scatter_reduce()` to atomically
      scatter-reduce values into the output array. This strategy can be
      advantageous when the input array is symbolic (making evaluation
      impossible) or both unevaluated and extremely large (making evaluation
      costly or impossible if there isn't enough memory).

      Disadvantages of this mode are that

      - Atomic scatters can suffer from memory contention (though
        :py:func:`drjit.scatter_reduce()` takes steps to reduce contention, see
        its documentation for details).

      - Atomic floating point scatter-addition is subject to non-deterministic
        rounding errors that arise from its non-commutative nature. Coupled
        with the scheduling-dependent execution order, this can lead to small
        variations across program runs. Integer and floating point min/max
        reductions are unaffected by this.

    - ``mode=None`` (default) automatically picks a reasonable strategy
      according to the following logic:

      - Symbolic mode is admissible when the necessary atomic reduction
        :ref:`is supported by the backend <scatter_reduce_supported>`.

      - Evaluated mode is admissible when the input does not involve symbolic
        variables. On the CUDA backend ``block_size`` must furthermore be a
        power of two.

      - If only one strategy remains, then pick that one. Raise an exception
        when no strategy works out.

      - Otherwise, use evaluated mode when the input array is already
        evaluated, or when evaluating it would consume less than 1 GiB of
        memory.

      - Use symbolic mode in all other cases.

    For some inputs, no strategy works out (e.g., multiplicative reduction of
    an array with a non-power-of-two block size on the CUDA backend). The
    function will raise an exception in such cases.

    Since evaluated mode can be quite a bit faster and is guaranteed to be
    deterministic, it is recommended that you design your program so that it
    invokes :py:func:`drjit.block_reduce` with a power-of-two ``block_size``.

    .. note::

        Tensor inputs are not supported. To reduce blocks within tensors, apply
        the regular axis-wide reductions (:py:func:`drjit.sum`,
        :py:func:`drjit.prod`, :py:func:`drjit.min`, :py:func:`drjit.max`) to
        reshaped tensors. For example, to sum-reduce a ``(16, 16)`` tensor by a
        factor of ``(4, 2)`` (i.e., to a ``(4, 8)``-sized tensor), write
        ``dr.sum(dr.reshape(value, shape=(4, 4, 8, 2)), axis=(1, 3))``.

    Args:
        value (object): A Dr.Jit array or PyTree

        block_size (int): size of the block

        mode (str | None): optional parameter to force an evaluation strategy.

    Returns:
        The block-reduced array or PyTree as specified above.

.. topic:: block_sum

    Sum elements within blocks.

    This is a convenience alias for :py:func:`drjit.block_reduce` with
    ``op`` set to :py:attr:`drjit.ReduceOp.Add`.

.. topic:: ArrayBase

    This is the base class of all Dr.Jit arrays and tensors. It provides an
    abstract version of the array API that becomes usable when the type is extended
    by a concrete specialization. :py:class:`ArrayBase` itself cannot be
    instantiated.

    See the section on Dr.Jit `type signatures <type_signatures>` to learn about
    the type parameters of :py:class:`ArrayBase`.

.. topic:: ArrayBase_init

    Construct a Dr.Jit array.

    Arrays can be constructed ..

    - from a sequence: ``Array3f([1, 2, 3])``
    - from individual components: ``Array3f(1, 2, 3)``
    - from a convertible type: ``Array3f(Array3i(1, 2, 3))``
    - from an tensor in another framework like NumPy, PyTorch, etc: ``Array3f(np.array([1, 2, 3]))``
    - via a broadcast: ``Array3f(1)`` (equivalent to ``Array3f(1, 1, 1)``)

    Note that this constructor is only available in *subclasses* of the
    :py:class:`drjit.ArrayBase` type.

.. topic:: ArrayBase_init_2

    Construct a Dr.Jit tensor.

    The function expects a linearized 1D array (in C-style order) and a shape
    descriptor as inputs.

    Note that this constructor is only available in *tensor subclasses* of the
    :py:class:`drjit.ArrayBase` type.

.. topic:: detail_IndexVector

   Reference-counted index vector. This class stores references to Dr.Jit
   variables and generally behaves like a ``list[int]``. The main difference
   is that it holds references to the elements so that they cannot expire.

   The main purpose of this class is to represent the inputs and outputs of
   :py:func:`drjit.detail.VariableTracker.read` and
   :py:func:`drjit.detail.VariableTracker.write`.

.. topic:: detail_VariableTracker

   Helper class for tracking state variables during control flow operations.

   This class reads and writes state variables as part of control flow
   operations such as :py:func:`dr.while_loop() <while_loop>` and
   :py:func:`dr.if_stmt() <if_stmt>`. It checks that each variable remains
   consistent across this multi-step process.

   Consistency here means that:

   - The tree structure of the :ref:`PyTree <pytrees>` PyTree is preserved
     across calls to :py:func:`read()`` and :py:func:`write()``.

   - The type of every PyTree element is similarly preserved.

   - The sizes of Dr.Jit arrays in the PyTree remain compatible across calls to
     :py:func:`read()` and :py:func:`write()`. The sizes of two arrays ``a``
     and ``b`` are considered compatible if ``a+b`` is well-defined (it's okay
     if this involves an intermediate broadcasting step.)

   In the case of an inconsistency, the implementation generates an error
   message that identifies the problematic variable by name.


.. topic:: detail_VariableTracker_VariableTracker

   Create a new variable tracker.

   The constructor accepts two parameters:

   - ``strict``: Certain types of Python objects (e.g. custom Python classes
     without ``DRJIT_STRUCT`` field, scalar Python numeric types) are not
     traversed by the variable tracker. If ``strict`` mode is enabled, any
     inconsistency here will cause the implementation to immediately give up
     with an error message. This is not always desired, hence this behavior
     is configurable.

   - ``check_size``: If set to ``true``, the tracker will ensure that
     variables remain size-compatible. The one case in Dr.Jit where this is
     not desired are evaluated loops with compression enabled (i.e.,
     inactive elements are pruned, which causes the array size to
     progressively shrink).

.. topic:: detail_VariableTracker_read

   Traverse a PyTree and read its variable indices.

   This function recursively traverses the PyTree ``state`` and appends the
   indices of encountered Dr.Jit arrays to the reference-counted output
   vector ``indices``. It performs numerous consistency checks during this
   process to ensure that variables remain consistent over time.

   The ``labels`` argument optionally identifies the top-level variable
   names tracked by this instance. This is recommended to obtain actionable
   error messages in the case of inconsistencies. Otherwise,
   ``default_label`` is prefixed to variable names.

.. topic:: detail_VariableTracker_write

   Traverse a PyTree and write its variable indices.

   This function recursively traverses the PyTree ``state`` and updates the
   encountered Dr.Jit arrays with indices from the ``indices`` argument.
   It performs numerous consistency checks during this
   process to ensure that variables remain consistent over time.

   When ``preserve_dirty`` is set to ``true``, the function leaves
   dirty arrays (i.e., ones with pending side effects) unchanged.

   The ``labels`` argument optionally identifies the top-level variable
   names tracked by this instance. This is recommended to obtain actionable
   error messages in the case of inconsistencies. Otherwise,
   ``default_label`` is prefixed to variable names.


.. topic:: detail_VariableTracker_clear

   Clear all variable state stored by the variable tracker.

.. topic:: detail_VariableTracker_restore

   Undo all changes and restore tracked variables to their original state.

.. topic:: detail_VariableTracker_rebuild

   Create a new copy of the PyTree representing the final
   version of the PyTree following a symbolic operation.

   This function returns a PyTree representing the latest state. This PyTree
   is created lazily, and it references the original one whenever values
   were unchanged. This function also propagates in-place updates when
   they are detected.

.. topic:: detail_VariableTracker_verify_size

   Check that the PyTree is compatible with size ``size``.

.. topic:: set_backend

   Adjust the ``drjit.auto.*`` module so that it refers to types from the
   specified backend.
