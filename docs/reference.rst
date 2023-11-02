.. py:module:: drjit

.. _reference:

Reference
=========

Array creation
--------------

.. autofunction:: zeros
.. autofunction:: empty
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: arange
.. autofunction:: linspace

Control flow
------------

.. autofunction:: function
.. autofunction:: switch
.. autofunction:: dispatch
.. autofunction:: while_loop

.. _horizontal-reductions-ref:

Horizontal operations
---------------------

These operations are *horizontal* in the sense that [..]

.. autofunction:: gather
.. autofunction:: scatter

.. autoclass:: ReduceOp

   Denotes the type of atomic read-modify-write (RMW) operation (for use with
   :py:func:`drjit.scatter_reduce`) or aggregation to be performed by a
   horizontal reduction.

   .. autoattribute:: None
      :annotation:

      Perform an ordinary scatter operation that ignores the current entry
      (only applies to scatter-reductions).

   .. autoattribute:: Add
      :annotation:

      Addition.

   .. autoattribute:: Mul
      :annotation:

      Multiplication.

   .. autoattribute:: Min
      :annotation:

      Minimum

   .. autoattribute:: Max
      :annotation:

      Maximum

   .. autoattribute:: And
      :annotation:

      Binary AND operation

   .. autoattribute:: Or
      :annotation:

      Binary OR operation

.. autofunction:: scatter_reduce
.. autofunction:: ravel
.. autofunction:: unravel
.. autofunction:: min
.. autofunction:: max
.. autofunction:: sum
.. autofunction:: prod
.. autofunction:: dot
.. autofunction:: norm
.. autofunction:: all
.. autofunction:: any
.. autofunction:: prefix_sum
.. autofunction:: cumsum
.. autofunction:: reverse

Mask operations
---------------

Also relevant here are :py:func:`any`, :py:func:`all`.

.. autofunction:: select
.. autofunction:: isinf
.. autofunction:: isnan
.. autofunction:: isfinite
.. autofunction:: allclose

Miscellaneous operations
------------------------

.. autofunction:: shape
.. autofunction:: slice_index
.. autofunction:: meshgrid

Just-in-time compilation
------------------------

.. autoclass:: JitBackend

   List of just-in-time compilation backends supported by Dr.Jit
   See also :py:func:`drjit.backend_v()`.

   .. autoattribute:: None
      :annotation:

      Indicates that a type is not handled by a Dr.Jit backend (e.g., a scalar type)

   .. autoattribute:: LLVM
      :annotation:

      Dr.Jit backend targeting various processors via the LLVM compiler infractructure.

   .. autoattribute:: CUDA
      :annotation:

      Dr.Jit backend targeting NVIDIA GPUs using PTX ("Parallel Thread Excecution") IR.

.. autoclass:: VarType

   List of possible scalar array types (not all of them are supported).

   .. autoattribute:: Void
      :annotation:

      Unknown/unspecified type.

   .. autoattribute:: Bool
      :annotation:

      Boolean/mask type.

   .. autoattribute:: Int8
      :annotation:

      Signed 8-bit integer.

   .. autoattribute:: UInt8
      :annotation:

      Unsigned 8-bit integer.

   .. autoattribute:: Int16
      :annotation:

      Signed 16-bit integer.

   .. autoattribute:: UInt16
      :annotation:

      Unsigned 16-bit integer.

   .. autoattribute:: Int32
      :annotation:

      Signed 32-bit integer.

   .. autoattribute:: UInt32
      :annotation:

      Unsigned 32-bit integer.

   .. autoattribute:: Int64
      :annotation:

      Signed 64-bit integer.

   .. autoattribute:: UInt64
      :annotation:

      Unsigned 64-bit integer.

   .. autoattribute:: Pointer
      :annotation:

      Pointer to a memory address.

   .. autoattribute:: Float16
      :annotation:

      16-bit floating point format (IEEE 754).

   .. autoattribute:: Float32
      :annotation:

      32-bit floating point format (IEEE 754).

   .. autoattribute:: Float64
      :annotation:

      64-bit floating point format (IEEE 754).

.. autoclass:: VarState

   The :py:attr:`drjit.ArrayBase.state` property returns one of the following
   enumeration values describing possible evaluation states of a Dr.Jit
   variable.

   .. autoattribute:: Invalid
      :annotation:

      The variable has length 0 and effetively does not exist.

   .. autoattribute:: Normal
      :annotation:

      An ordinary unevaluated variable that is neither a literal constant nor symbolic.

   .. autoattribute:: Literal
      :annotation:

      A literal constant. Does not consume device memory.

   .. autoattribute:: Evaluated
      :annotation:

      An evaluated variable backed by a device memory region.

   .. autoattribute:: Symbolic
      :annotation:

      A symbolic variable that could take on various inputs. Cannot be evaluated.

   .. autoattribute:: Mixed
      :annotation:

      This is a nested array, and the components have mixed states.

.. autofunction:: has_backend
.. autofunction:: schedule
.. autofunction:: eval


.. autoclass:: JitFlag

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

   .. autoattribute:: ConstantPropagation
      :annotation:

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

      Enabled by default.

   .. autoattribute:: ValueNumbering
      :annotation:

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

       Enabled by default.

   .. autoattribute:: VCallRecord
      :annotation:

      **Recorded function calls**: Dr.Jit provides two main ways of compiling
      *indirect function calls* (also known as *virtual function calls* or
      *dynamic dispatch*).

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

      Recorded mode is enabled by default.

   .. autoattribute:: IndexReuse
      :annotation:

      **Index reuse**: Dr.Jit consists of two main parts: the just-in-time
      compiler, and the automatic differentiation layer. Both maintain an
      internal data structure representing captured computation, in which each
      variable is associated with an index (e.g., ``r1234`` in the JIT
      compiler, and ``a1234`` in the AD graph).

      The index of a Dr.Jit array in these graphs can be queried via the
      :py:attr:`drjit.index` and :py:attr:`drjit.index_ad` variables, and they
      are also visible in debug messages (if :py:func:`drjit.set_log_level` is
      set to a more verbose debug level).

      Dr.Jit aggressively reuses the indices of expired variables by default,
      but this can make debug output difficult to interpret. When when
      debugging Dr.Jit itself, it is often helpful to investigate the history
      of a particular variable. In such cases, set this flag to ``False`` to
      disable variable reuse both at the JIT and AD levels. This comes at a
      cost: the internal data structures keep on growing, so it is not suitable
      for long-running computations.

      Index reuse is enabled by default.

   .. autoattribute:: Default
      :annotation:

      The default set of flags.

.. autofunction:: set_flag
.. autofunction:: flag
.. autoclass:: scoped_set_flag

   .. automethod:: __init__
   .. automethod:: __enter__
   .. automethod:: __exit__

Type traits
-----------

The functions in this section can be used to infer properties or types of
Dr.Jit arrays.

The naming convention with a trailing ``_v`` or ``_t`` indicates whether a
function returns a value or a type. Evaluation takes place at runtime within
Python. In C++, these expressions are all  ``constexpr`` (i.e., evaluated at
compile time.).

Array type tests
________________

.. autofunction:: is_array_v
.. autofunction:: is_mask_v
.. autofunction:: is_float_v
.. autofunction:: is_integral_v
.. autofunction:: is_arithmetic_v
.. autofunction:: is_signed_v
.. autofunction:: is_unsigned_v
.. autofunction:: is_dynamic_v
.. autofunction:: is_jit_v
.. autofunction:: is_diff_v
.. autofunction:: is_vector_v
.. autofunction:: is_complex_v
.. autofunction:: is_matrix_v
.. autofunction:: is_quaternion_v
.. autofunction:: is_tensor_v
.. autofunction:: is_special_v
.. autofunction:: is_struct_v

Array properties (shape, type, etc.)
____________________________________

.. autofunction:: type_v
.. autofunction:: backend_v
.. autofunction:: size_v
.. autofunction:: depth_v
.. autofunction:: itemsize_v

.. py:data:: Dynamic
    :type: int
    :value: -1

    Special size value used to identify dynamic arrays in
    :py:func:`size_v`.

.. py:data:: newaxis
    :type: NoneType
    :value: None

    Special size value used to create new axes in slicing
    expressions (analogous to a similar feature in NumPy).

Access to related types
_______________________

.. autofunction:: mask_t
.. autofunction:: value_t
.. autofunction:: scalar_t
.. autofunction:: array_t
.. autofunction:: int_array_t
.. autofunction:: uint_array_t
.. autofunction:: int32_array_t
.. autofunction:: uint32_array_t
.. autofunction:: int64_array_t
.. autofunction:: uint64_array_t
.. autofunction:: float_array_t
.. autofunction:: float32_array_t
.. autofunction:: float64_array_t
.. autofunction:: replace_type_t
.. autofunction:: detached_t
.. autofunction:: expr_t

Standard mathematical functions
-------------------------------

.. autofunction:: abs
.. autofunction:: minimum
.. autofunction:: maximum
.. autofunction:: clip
.. autofunction:: fma
.. autofunction:: ceil
.. autofunction:: floor
.. autofunction:: trunc
.. autofunction:: round
.. autofunction:: sqrt
.. autofunction:: cbrt
.. autofunction:: rcp
.. autofunction:: rsqrt
.. autofunction:: reinterpret_array

Transcendental functions
------------------------

Dr.Jit implements the most common transcendental functions using methods that
are based on the CEPHES math library. The accuracy of these approximations is
documented in a set of :ref:`tables <transcendental-accuracy>` below.

Trigonometric functions
_______________________

.. autofunction:: sin
.. autofunction:: cos
.. autofunction:: sincos
.. autofunction:: tan
.. autofunction:: asin
.. autofunction:: acos
.. autofunction:: atan
.. autofunction:: atan2

Hyperbolic functions
____________________

.. autofunction:: sinh
.. autofunction:: cosh
.. autofunction:: sincosh
.. autofunction:: tanh
.. autofunction:: asinh
.. autofunction:: acosh
.. autofunction:: atanh

Exponentials, logarithms, power function
________________________________________

.. autofunction:: log2
.. autofunction:: log
.. autofunction:: exp2
.. autofunction:: exp
.. autofunction:: power


Automatic differentiation
-------------------------

.. autoclass:: ADMode

   Enumeration to distinguish different types of primal/derivative computation.

   See also :py:func:`drjit.enqueue()`, :py:func:`drjit.traverse()`.

   .. autoattribute:: Primal
      :annotation:

      Primal/original computation without derivative tracking. Note that this
      is *not* a valid input to Dr.Jit AD routines, but it is sometimes useful
      to have this entry when to indicate to a computation that derivative
      propagation should not be performed.

   .. autoattribute:: Forward
      :annotation:

      Propagate derivatives in forward mode (from inputs to outputs)

   .. autoattribute:: Backward
      :annotation:

      Propagate derivatives in backward/reverse mode (from outputs to inputs)

.. autoclass:: ADFlag

   By default, Dr.Jit's AD system destructs the enqueued input graph during
   forward/backward mode traversal. This frees up resources, which is useful
   when working with large wavefronts or very complex computation graphs.
   However, this also prevents repeated propagation of gradients through a
   shared subgraph that is being differentiated multiple times.

   To support more fine-grained use cases that require this, the flags in the
   following enumeration can be used to control what should and should not be
   destructed.

   See also :py:func:`drjit.traverse()`, :py:func:`drjit.forward_from()`,
   :py:func:`drjit.forward_to()`, :py:func:`drjit.backward_from()`, and
   :py:func:`drjit.backward_to()`.

   .. autoattribute:: ClearNone
      :annotation:

      Clear nothing.

   .. autoattribute:: ClearEdges
      :annotation:

      Delete all traversed edges from the computation graph

   .. autoattribute:: ClearInput
      :annotation:

      Clear the gradients of processed input vertices (in-degree == 0)

   .. autoattribute:: ClearInterior
      :annotation:

      Clear the gradients of processed interior vertices (out-degree != 0)

   .. autoattribute:: ClearVertices
      :annotation:

      Clear gradients of processed vertices only, but leave edges intact. Equal
      to ``ClearInput | ClearInterior``.

   .. autoattribute:: AllowNoGrad
      :annotation:

      Don't fail when the input to a ``drjit.forward`` or ``backward``
      operation is not a differentiable array.";

   .. autoattribute:: Default
      :annotation:

      Default: clear everything (edges, gradients of processed vertices). Equal
      to ``ClearEdges | ClearVertices``.

.. autofunction:: detach
.. autofunction:: enable_grad
.. autofunction:: disable_grad
.. autofunction:: set_grad_enabled
.. autofunction:: grad_enabled
.. autofunction:: grad
.. autofunction:: set_grad
.. autofunction:: accum_grad
.. autofunction:: replace_grad
.. autofunction:: clear_grad
.. autofunction:: traverse
.. autofunction:: enqueue
.. autofunction:: forward_from
.. autofunction:: forward_to
.. autofunction:: forward
.. autofunction:: backward_from
.. autofunction:: backward_to
.. autofunction:: backward
.. autofunction:: suspend_grad
.. autofunction:: resume_grad
.. autofunction:: isolate_grad

.. autoclass:: CustomOp

   .. automethod:: eval
   .. automethod:: forward
   .. automethod:: backward
   .. automethod:: name
   .. automethod:: grad_out
   .. automethod:: set_grad_out
   .. automethod:: grad_in
   .. automethod:: set_grad_in
   .. automethod:: add_input
   .. automethod:: add_output

.. autofunction:: custom


Safe mathematical functions
---------------------------

Dr.Jit provides "safe" variants of a few standard mathematical operations that
are prone to out-of-domain errors in calculations with floating point rounding
errors.  Such errors could, e.g., cause the argument of a square root to become
negative, which would ordinarily require complex arithmetic. At zero, the
derivative of the square root function is infinite. The following operations 
clamp the input to a safe range to avoid these extremes.

.. autofunction:: safe_sqrt
.. autofunction:: safe_asin
.. autofunction:: safe_acos

Constants
---------

.. autodata:: e

   The exponential constant :math:`e` represented as a Python ``float``.

.. autodata:: log_two

   The value :math:`\log(2)` represented as a Python ``float``.

.. autodata:: inv_log_two

   The value :math:`\frac{1}{\log(2)}` represented as a Python ``float``.

.. autodata:: pi

   The value :math:`\pi` represented as a Python ``float``.

.. autodata:: inv_pi

   The value :math:`\frac{1}{\pi}` represented as a Python ``float``.

.. autodata:: sqrt_pi

   The value :math:`\sqrt{\pi}` represented as a Python ``float``.

.. autodata:: inv_sqrt_pi

   The value :math:`\frac{1}{\sqrt{\pi}}` represented as a Python ``float``.

.. autodata:: two_pi

   The value :math:`2\pi` represented as a Python ``float``.

.. autodata:: inv_two_pi

   The value :math:`\frac{1}{2\pi}` represented as a Python ``float``.

.. autodata:: sqrt_two_pi

   The value :math:`\sqrt{2\pi}` represented as a Python ``float``.

.. autodata:: inv_sqrt_two_pi
   :annotation:

   The value :math:`\frac{1}{\sqrt{2\pi}}` represented as a Python ``float``.

.. autodata:: four_pi

   The value :math:`4\pi` represented as a Python ``float``.

.. autodata:: inv_four_pi

   The value :math:`\frac{1}{4\pi}` represented as a Python ``float``.

.. autodata:: sqrt_four_pi

   The value :math:`\sqrt{4\pi}` represented as a Python ``float``.

.. autodata:: sqrt_two

   The value :math:`\sqrt{2\pi}` represented as a Python ``float``.

.. autodata:: inv_sqrt_two

   The value :math:`\frac{1}{\sqrt{2\pi}}` represented as a Python ``float``.

.. autodata:: inf

   The value ``float('inf')`` represented as a Python ``float``.

.. autodata:: nan

   The value ``float('nan')`` represented as a Python ``float``.

.. autofunction:: epsilon
.. autofunction:: one_minus_epsilon
.. autofunction:: recip_overflow
.. autofunction:: smallest
.. autofunction:: largest

Array base class
----------------

.. autoclass:: ArrayBase

    .. autoproperty:: array
    .. autoproperty:: ndim
    .. autoproperty:: shape
    .. autoproperty:: state
    .. autoproperty:: x
    .. autoproperty:: y
    .. autoproperty:: z
    .. autoproperty:: w
    .. autoproperty:: index
    .. autoproperty:: index_ad
    .. automethod:: __len__
    .. automethod:: __iter__
    .. automethod:: __repr__
    .. automethod:: __bool__

       Casts the array to a Python ``bool`` type. This is only permissible when
       ``self`` represents an boolean array of both depth and size 1.

    .. automethod:: __add__
    .. automethod:: __radd__
    .. automethod:: __iadd__
    .. automethod:: __sub__
    .. automethod:: __rsub__
    .. automethod:: __isub__
    .. automethod:: __mul__
    .. automethod:: __rmul__
    .. automethod:: __imul__
    .. automethod:: __truediv__
    .. automethod:: __rtruediv__
    .. automethod:: __itruediv__
    .. automethod:: __floordiv__
    .. automethod:: __rfloordiv__
    .. automethod:: __ifloordiv__
    .. automethod:: __mod__
    .. automethod:: __rmod__
    .. automethod:: __imod__
    .. automethod:: __rshift__
    .. automethod:: __rrshift__
    .. automethod:: __irshift__
    .. automethod:: __lshift__
    .. automethod:: __rlshift__
    .. automethod:: __ilshift__
    .. automethod:: __and__
    .. automethod:: __rand__
    .. automethod:: __iand__
    .. automethod:: __or__
    .. automethod:: __ror__
    .. automethod:: __ior__
    .. automethod:: __xor__
    .. automethod:: __rxor__
    .. automethod:: __ixor__
    .. automethod:: __abs__
    .. automethod:: __le__
    .. automethod:: __lt__
    .. automethod:: __ge__
    .. automethod:: __gt__
    .. automethod:: __ne__
    .. automethod:: __eq__
    .. automethod:: __dlpack__
    .. automethod:: __array__

Concrete array classes
----------------------

Scalar array namespace (``drjit.scalar``)
_________________________________________

The scalar backend directly operates on individual floating point/integer
values without the use of parallelization or vectorization.

For example, a :py:class:`drjit.scalar.Array3f` instance represents a
simple 3D vector with 3 ``float``-valued entries. In the JIT-compiled
backends (CUDA, LLVM), the same ``Array3f`` type represents an array of 3D
vectors partaking in a parallel computation.

Scalars
^^^^^^^
.. py:data:: Bool
    :type: type
    :value: bool
.. py:data:: Float
    :type: type
    :value: float
.. py:data:: Float64
    :type: type
    :value: float
.. py:data:: Int
    :type: type
    :value: int
.. py:data:: Int64
    :type: type
    :value: int
.. py:data:: UInt
    :type: type
    :value: int
.. py:data:: UInt64
    :type: type
    :value: int

1D arrays
^^^^^^^^^
.. autoclass:: drjit.scalar.Array0b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4b
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXb
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4f
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXf
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0u
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1u
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2u
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3u
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4u
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXu
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0i
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1i
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2i
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3i
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4i
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXi
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4f64
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXf64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0u64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1u64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2u64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3u64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4u64
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXu64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array0i64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array1i64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array2i64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array3i64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array4i64
    :show-inheritance:
.. autoclass:: drjit.scalar.ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: drjit.scalar.Array22b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array33b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array44b
    :show-inheritance:
.. autoclass:: drjit.scalar.Array22f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array33f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array44f
    :show-inheritance:
.. autoclass:: drjit.scalar.Array22f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array33f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: drjit.scalar.Complex2f
    :show-inheritance:
.. autoclass:: drjit.scalar.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.scalar.Quaternion4f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: drjit.scalar.TensorXb
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXf
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXu
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXi
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXf64
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXu64
    :show-inheritance:
.. autoclass:: drjit.scalar.TensorXi64
    :show-inheritance:

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: drjit.scalar.PCG32

   .. automethod:: __init__
   .. automethod:: seed
   .. automethod:: next_uint32
   .. automethod:: next_uint64
   .. automethod:: next_float32
   .. automethod:: next_float64
   .. automethod:: next_uint32_bounded
   .. automethod:: next_uint64_bounded
   .. automethod:: __add__
   .. automethod:: __iadd__
   .. automethod:: __sub__
   .. automethod:: __isub__
   .. autoproperty:: inc
   .. autoproperty:: state


LLVM array namespace (``drjit.llvm``)
_______________________________________

The LLVM backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalar
^^^^^^

.. autoclass:: drjit.llvm.Bool
    :show-inheritance:
.. autoclass:: drjit.llvm.Float
    :show-inheritance:
.. autoclass:: drjit.llvm.Float64
    :show-inheritance:
.. autoclass:: drjit.llvm.UInt
    :show-inheritance:
.. autoclass:: drjit.llvm.UInt64
    :show-inheritance:
.. autoclass:: drjit.llvm.Int
    :show-inheritance:
.. autoclass:: drjit.llvm.Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: drjit.llvm.Array0b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4b
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXb
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXf
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0u
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1u
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2u
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3u
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4u
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXu
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0i
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1i
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2i
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3i
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4i
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXi
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXf64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0u64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1u64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2u64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3u64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXu64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array0i64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array1i64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array2i64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array3i64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array4i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: drjit.llvm.Array22b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array33b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array44b
    :show-inheritance:
.. autoclass:: drjit.llvm.Array22f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array33f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array44f
    :show-inheritance:
.. autoclass:: drjit.llvm.Array22f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array33f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: drjit.llvm.Complex2f
    :show-inheritance:
.. autoclass:: drjit.llvm.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.llvm.Quaternion4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: drjit.llvm.TensorXb
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXf
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXu
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXi
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXf64
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXu64
    :show-inheritance:
.. autoclass:: drjit.llvm.TensorXi64
    :show-inheritance:

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: drjit.llvm.PCG32

   .. automethod:: __init__
   .. automethod:: seed
   .. automethod:: next_uint32
   .. automethod:: next_uint64
   .. automethod:: next_float32
   .. automethod:: next_float64
   .. automethod:: next_uint32_bounded
   .. automethod:: next_uint64_bounded
   .. automethod:: __add__
   .. automethod:: __iadd__
   .. automethod:: __sub__
   .. automethod:: __isub__
   .. autoproperty:: inc
   .. autoproperty:: state

LLVM array namespace with automatic differentiation (``drjit.llvm.ad``)
_______________________________________________________________________

The LLVM AD backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: drjit.llvm.ad.Bool
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Float
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Float64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.UInt
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.UInt64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Int
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: drjit.llvm.ad.Array0b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXb
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXf
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXu
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXi
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXf64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4u64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXu64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array0i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array1i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array2i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array3i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array4i64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: drjit.llvm.ad.Array22b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array33b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array44b
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array22f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array33f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array44f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array22f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array33f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: drjit.llvm.ad.Complex2f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Quaternion4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: drjit.llvm.ad.TensorXb
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXf
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXu
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXi
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXf64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXu64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.TensorXi64
    :show-inheritance:

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: drjit.llvm.ad.PCG32

   .. automethod:: __init__
   .. automethod:: seed
   .. automethod:: next_uint32
   .. automethod:: next_uint64
   .. automethod:: next_float32
   .. automethod:: next_float64
   .. automethod:: next_uint32_bounded
   .. automethod:: next_uint64_bounded
   .. automethod:: __add__
   .. automethod:: __iadd__
   .. automethod:: __sub__
   .. automethod:: __isub__
   .. autoproperty:: inc
   .. autoproperty:: state

Miscellaneous
-------------

.. autofunction:: graphviz
.. autofunction:: graphviz_ad
.. autofunction:: label
.. autofunction:: set_label
.. py:data:: None
   :type: NoneType

   This is just a copy of the builtin Python ``None`` value.
