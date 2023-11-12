.. py:module:: drjit

.. _reference:

API Reference
=============

Array creation
--------------

.. autofunction:: zeros
.. autofunction:: empty
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: opaque
.. autofunction:: arange
.. autofunction:: linspace

Control flow
------------

.. autofunction:: syntax
.. autofunction:: hint
.. autofunction:: while_loop
.. autofunction:: if_stmt
.. autofunction:: switch
.. autofunction:: dispatch

.. _horizontal-reductions-ref:

Horizontal operations
---------------------

These operations are *horizontal* in the sense that [..]

.. autofunction:: gather
.. autofunction:: scatter

.. autoclass:: ReduceOp

   List of different atomic read-modify-write (RMW) operations supported by
   :py:func:`drjit.scatter_reduce()`.

   .. autoattribute:: None
      :annotation:

      Perform an ordinary scatter operation that ignores the current entry.
.
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
.. autofunction:: scatter_add
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
.. autofunction:: make_opaque

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

   .. autoattribute:: Undefined
      :annotation:

      An undefined memory region. Does not (yet) consume device memory.

   .. autoattribute:: Literal
      :annotation:

      A literal constant. Does not consume device memory.

   .. autoattribute:: Unevaluated
      :annotation:

      An ordinary unevaluated variable that is neither a literal constant nor symbolic.

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

   .. For Sphinx-related technical reasons, the below comment is replicated
      in docstr.h. Please keep the two in sync when making changes.

   Flags that control how Dr.Jit compiles and optimizes programs.

   This enumeration lists various flag that control how Dr.Jit compiles and
   optimizes programs, most of which are enabled by default. The status of each
   flag can be queried via :py:func:`drjit.flag` and enabled/disabled via the
   :py:func:`drjit.set_flag` or the recommended
   :py:func:`drjit.scoped_set_flag` functions, e.g.:

   .. code-block:: python

      with dr.scoped_set_flag(dr.JitFlag.SymbolicLoops, False):
          # code that has this flag disabled goes here

   The most common reason to update the flags is to switch between *symbolic*
   and *evaluated* execution of loops and functions. The former eagerly
   executes programs by breaking them into many smaller kernels, while the
   latter records computation symbolically to assemble large *megakernels*. See
   explanations below along with the documentation of :py:func:`drjit.switch`
   and :py:class:`drjit.while_loop` for more details on these two modes.

   Dr.Jit flags are a thread-local property. This means that multiple independent
   threads using Dr.Jit can set them independently without interfering with each
   other.

   .. autoattribute:: IndexReuse
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

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

      Index reuse is *enabled* by default.

   .. autoattribute:: ConstantPropagation
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      **Constant propagation**: immediately evaluate arithmetic involving
      literal constants on the host and don't generate any device-specific code
      for them.

      For example, the following assertion holds when value numbering is
      enabled in Dr.Jit.

      .. code-block:: python

         from drjit.llvm import Int

         # Create two literal constant arrays
         a, b = Int(4), Int(5)

         # This addition operation can be immediately performed and does
         # not need to be recorded
         c1 = a + b

         # Double-check that c1 and c2 refer to the same Dr.Jit variable
         c2 = Int(9)
         assert c1.index == c2.index

      Constant propagation is *enabled* by default.

   .. autoattribute:: ValueNumbering
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      **Local value numbering**: a simple variant of common subexpression
      elimination that collapses identical expressions within basic blocks. For
      example, the following assertion holds when value numbering is enabled in
      Dr.Jit.

      .. code-block:: python

         from drjit.llvm import Int

         # Create two nonliteral arrays stored in device memory
         a, b = Int(1, 2, 3), Int(4, 5, 6)

         # Perform the same arithmetic operation twice
         c1 = a + b
         c2 = a + b

         # Verify that c1 and c2 reference the same Dr.Jit variable
         assert c1.index == c2.index

      Local value numbering is *enabled* by default.

   .. autoattribute:: SymbolicCalls
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Dr.Jit provides two main ways of compiling function calls targeting
      *instance arrays*.

      1. **Symbolic mode** (the default): Dr.Jit captures callables by invoking them
         with *symbolic* (abstract) arguments. By doing so, it can capture a
         transcript of each callable and then turn it into a function in the
         generated kernel. Symbolic mode preserves the control flow structure of the
         original program by replicating it within Dr.Jit's intermediate
         representation.

         The main advantage of recorded mode is:

         * It is very efficient in terms of device memory storage and bandwidth, since
           function call arguments and return values can be exchanged through fast
           CPU/GPU registers.

         Its main downsides are:

         * Symbolic arrays cannot be evaluated, printed, etc. Attempting to
           perform such operations will raise an exception.

           This limitation may be inconvenient especially when debugging code, in
           which case evaluated mode is preferable.

         * Thread divergence: neighboring SIMD lanes may target different callables,
           which can have a negative impact on efficiency.

         * A kernel with many callables can become quite large and costly to compile.

      2. **Evaluated mode**: Dr.Jit evaluates all inputs and groups them by instance
         ID. Following this, it launches a a kernel *per instance* to process the
         rearranged inputs and assemble the function return value.

         The main advantages of evaluated mode are:

         * *It is easier to debug*: evaluating and processing intermediate results
           (e.g. via Python's ``print`` statement or more advanced plotting tools)
           is legal.  You may also use a debugger to step through the program.

         * Kernels are smaller and avoid thread divergence, since Dr.Jit reorders
           computation by callable.

         The main downsides are:

         * Each callable essentially turns its own kernel that reads its input and
           writes outputs via device memory. The required memory bandwidth and
           storage often make evaluated mode impractical.

      Note that the behavior of the functions :py:func:`drjit.switch` and
      :py:func:`drjit.dispatch` is also controlled by this flag.

   .. autoattribute:: OptimizeCalls
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Perform basic optimizations for function calls on instance arrays.

      This flag enables two optimizations:

      - *Constant propagation*: Dr.Jit will propagate literal constants across
        function boundaries while tracing, which can unlock simplifications
        within. This is especially useful in combination with automatic
        differentiation, where it helps to detect code that does not influence
        the computed derivatives.

      - *Devirtualization*: When an element of the return value has the same
        computation graph in all instances, it is removed from the function
        call interface and moved to the caller.

      The flag is *enabled* by default. Note that it is only effective in
      combination with  :py:attr:`SymbolicCalls`. The behavior of the functions
      :py:func:`drjit.switch` and :py:func:`drjit.dispatch` is also controlled
      by this flag.

   .. autoattribute:: MergeFunctions
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Deduplicate code generated by function calls on instance arrays.

      When ``arr`` is an instance array (potentially with thousands of
      instances), a function call like

      .. code-block:: python

         arr.f(inputs...)

      can potentially generate vast numbers of different callables in the
      generated code. At the same time, many of these callables may contain
      identical code (or code that is identical except for data references).

      Dr.Jit can exploit such redundancy and merge such callables during
      computation. Besides generating shorter programs, this also helps to
      reduce thread divergence.

      This flag is enabled by default. Note that it is only effective in
      combination with  :py:attr:`SymbolicCalls`. The behavior of the functions
      :py:func:`drjit.switch` and :py:func:`drjit.dispatch` is also controlled
      by this flag.

   .. autoattribute:: SymbolicLoops
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Dr.Jit provides two main ways of compiling loops involving Dr.Jit arrays.

      1. **Symbolic mode** (the default): Dr.Jit executes the loop a single
         time regardless of how many iterations it requires in practice. It
         does so with *symbolic* (abstract) arguments to capture the loop
         condition and body and then turns it into an equivalent loop in the
         generated kernel. Symbolic mode preserves the control flow structure
         of the original program by replicating it within Dr.Jit's intermediate
         representation.

         The main advantage of recorded mode is:

         * It is very efficient in terms of device memory storage and
           bandwidth, since loop state variables can be exchanged through fast
           CPU/GPU registers.

         Its main downsides is:

         * Symbolic arrays cannot be evaluated, printed, etc. Attempting to
           perform such operations within the loop body will raise an
           exception.

           This limitation may be inconvenient especially when debugging code,
           in which case evaluated mode is preferable.

      2. **Evaluated mode**: Dr.Jit evaluates the loop's state variables and
         reduces the loop condition to a single element (``bool``) that
         expresses whether any elements are still alive. If so, it runs the
         loop body and the process repeats. The main advantages of evaluated
         mode is:

         * *It is easier to debug*: evaluating and processing intermediate results
           (e.g. via Python's ``print`` statement or more advanced plotting
           tools) is legal.  You may also use a debugger to step through the
           program.

         The main downsides are:

         * Each iteration generates at least one kernel that reads its input and
           writes outputs via device memory. The required memory bandwidth and
           storage often make evaluated mode impractical.

      Symbolic mode is the default.

   .. autoattribute:: OptimizeLoops
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Perform basic optimizations for loops involving Dr.Jit arrays.

      This flag enables two optimizations:

      - *Constant arrays*: variables in the *loop state* set that aren't
        modified by a loop are removed from this set. This shortens the
        generated code, which can be helpful especially in combination with the
        automatic transformations performed by :py:func:`drjit.function` that
        may be somewhat conservative in classifying too many local variables as
        potential loop state.

      - *Literal constant arrays*: In addition to the above point, constant
        loop state variables that are *literal constants* are propagated into
        the loop body, where this may reveal optimization opportunities.

        This is useful in combination with automatic differentiation, where it
        helps to detect code that does not influence the computed derivatives.

      One practical implication of this optimization is that it may cause
      :py:func:`drjit.while_loop` to run the loop body twice instead of just
      once.

      This flag is enabled by default. Note that it is only effective in
      combination with  :py:attr:`SymbolicLoops`.

   .. autoattribute:: ForceOptiX
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Force execution through OptiX even if a kernel doesn't use ray tracing.
      This only applies to the CUDA backend is mainly helpful for automated
      tests done by the Dr.Jit team.

      This flag is *disabled* by default.

   .. autoattribute:: PrintIR
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Print the low-level IR representation when launching a kernel.

      If enabled, this flag causes Dr.Jit to print the low-level IR (LLVM IR,
      NVIDIA PTX) representation of the generated code onto the console (or
      Jupyter notebook).

      This flag is *disabled* by default.

   .. autoattribute:: KernelHistory
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Maintain a history of kernel launches to profile/debug programs.

      Programs written on top of Dr.Jit execute in an *extremely* asynchronous
      manner. By default, the system postpones the computation to build large
      fused kernels. Even when this computation eventually runs, it does so
      asynchronously with respect to the host, which can make benchmarking
      difficult.

      In general, beware of the following benchmarking *anti-pattern*:

      .. code-block::

          import time
          a = time.time()
          # Some Dr.Jit computation
          b = time.time()
          print("took %.2f ms" % ((b-a) * 1000))

      In the worst case, the measured time interval may only capture the
      *tracing time*, without any actual computation having taken place.
      Another common mistake with this pattern is that Dr.Jit or the target
      device may still be busy with computation that started *prior* to the ``a
      = time.time()`` line, which is now incorrectly added to the measured
      period.

      Dr.Jit provides a *kernel history* feature, where it creates an entry in
      a list whenever it launches a kernel or related operation (memory copies,
      etc.). This not only gives accurate and isolated timings (measured with
      counters on the CPU/GPU) but also reveals if a kernel was launched at
      all. To capture the kernel history, set this flag just before the region
      to be benchmarked and call :py:func:`drjit.kernel_history()` at the end.

      Capturing the history has a (very) small cost and is therefore
      *disabled* by default.

   .. autoattribute:: LaunchBlocking
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Force synchronization after every kernel launch. This is useful to
      isolate severe problems (e.g. crashes) to a specific kernel.

      This flag has a severe performance impact and is *disabled* by default.

   .. autoattribute:: AtomicReduceLocal
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      Reduce locally before performing atomic memory operations.

      Atomic operations targeting global memory can be very expensive,
      especially when many writes target the same memory address leading to
      *contention*.

      This is a common problem when automatically differentiating computation
      in *reverse mode* (e.g. :py:func:`drjit.backward`), since this
      transformation turns differentiable global memory reads into atomic
      scatter-additions.

      To reduce this cost, Dr.Jit can optionally perform a local reduction that
      uses cooperation between SIMD/warp lanes to resolve all requests
      targeting the same address and then only issuing a single atomic memory
      transaction per unique target. This can reduce atomic memory traffic by
      up to a factor of 32 (CUDA) or 16 (LLVM backend with AVX512).

      This operation only affects the behavior of the :py:func:`scatter_reduce`
      function (and the reverse-mode derivative of :py:func:`gather`).

      This flag is *enabled* by default.

   .. autoattribute:: Symbolic
      :annotation:

      .. For Sphinx-related technical reasons, the below comment is replicated
         in docstr.h. Please keep the two in sync when making changes.

      This flag should not be set in user code. Dr.Jit sets it whenever it is
      capturing computation symbolically.

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

.. data:: e

   The exponential constant :math:`e` represented as a Python ``float``.

.. data:: log_two

   The value :math:`\log(2)` represented as a Python ``float``.

.. data:: inv_log_two

   The value :math:`\frac{1}{\log(2)}` represented as a Python ``float``.

.. data:: pi

   The value :math:`\pi` represented as a Python ``float``.

.. data:: inv_pi

   The value :math:`\frac{1}{\pi}` represented as a Python ``float``.

.. data:: sqrt_pi

   The value :math:`\sqrt{\pi}` represented as a Python ``float``.

.. data:: inv_sqrt_pi

   The value :math:`\frac{1}{\sqrt{\pi}}` represented as a Python ``float``.

.. data:: two_pi

   The value :math:`2\pi` represented as a Python ``float``.

.. data:: inv_two_pi

   The value :math:`\frac{1}{2\pi}` represented as a Python ``float``.

.. data:: sqrt_two_pi

   The value :math:`\sqrt{2\pi}` represented as a Python ``float``.

.. data:: inv_sqrt_two_pi
   :annotation:

   The value :math:`\frac{1}{\sqrt{2\pi}}` represented as a Python ``float``.

.. data:: four_pi

   The value :math:`4\pi` represented as a Python ``float``.

.. data:: inv_four_pi

   The value :math:`\frac{1}{4\pi}` represented as a Python ``float``.

.. data:: sqrt_four_pi

   The value :math:`\sqrt{4\pi}` represented as a Python ``float``.

.. data:: sqrt_two

   The value :math:`\sqrt{2\pi}` represented as a Python ``float``.

.. data:: inv_sqrt_two

   The value :math:`\frac{1}{\sqrt{2\pi}}` represented as a Python ``float``.

.. data:: inf

   The value ``float('inf')`` represented as a Python ``float``.

.. data:: nan

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


Miscellaneous
-------------

.. autofunction:: graphviz
.. autofunction:: graphviz_ad
.. autofunction:: whos
.. autofunction:: whos_ad
.. autofunction:: label
.. autofunction:: set_label
.. py:data:: None
   :type: NoneType

   This is just a copy of the builtin Python ``None`` value.
