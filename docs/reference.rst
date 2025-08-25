.. py:module:: drjit

.. _reference:

API Reference (Main)
====================

This document explains the public API (behaviors, signatures) in exhaustive
detail. If you're new to Dr.Jit, it may be easier to start by reading the other
sections first.

The reference documentation is also exposed through docstrings, which many
visual editors (e.g., `VS Code <https://code.visualstudio.com>`__, `neovim
<https://neovim.io>`__ with `LSP
<https://en.wikipedia.org/wiki/Language_Server_Protocol>`__) will show during
code completion, or when hovering over an expression.

The reference extensively use `type variables
<https://docs.python.org/3/library/typing.html#typing.TypeVar>`__ which can be
recognized because their name equals or ends with a capital ``T`` (e.g., ``T``,
``ArrayT``, ``MaskT``, etc.). Type variables serve as placeholders that show
how types propagate through function calls. For example, a function with
signature

.. code-block:: python

   def f(arg: T, /) -> tuple[T, T]: ...

will return a pair of ``int`` instances when called with an ``int``-typed
``arg`` value.

Array creation
--------------

.. autofunction:: zeros
.. autofunction:: empty
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: opaque
.. autofunction:: arange
.. autofunction:: linspace
.. autofunction:: zeros_like
.. autofunction:: empty_like
.. autofunction:: ones_like

Control flow
------------

.. autofunction:: syntax
.. autofunction:: hint
.. autofunction:: while_loop
.. autofunction:: if_stmt
.. autofunction:: switch
.. autofunction:: dispatch

Scatter/gather operations
-------------------------

.. autofunction:: gather
.. autofunction:: scatter

.. autofunction:: scatter_reduce
.. autofunction:: scatter_add
.. autofunction:: scatter_add_kahan
.. autofunction:: scatter_inc
.. autofunction:: slice

Reductions
----------

.. autoenum:: ReduceOp
.. autoenum:: ReduceMode

.. autofunction:: reduce
.. autofunction:: sum
.. autofunction:: prod
.. autofunction:: min
.. autofunction:: max
.. autofunction:: mean

.. autofunction:: all
.. autofunction:: any
.. autofunction:: none
.. autofunction:: count

.. autofunction:: dot
.. autofunction:: abs_dot
.. autofunction:: squared_norm
.. autofunction:: norm

Prefix reductions
-----------------

.. autofunction:: prefix_reduce
.. autofunction:: prefix_sum
.. autofunction:: cumsum

Block reductions
----------------

.. autofunction:: block_reduce
.. autofunction:: block_sum
.. autofunction:: block_prefix_reduce
.. autofunction:: block_prefix_sum

Rearranging array contents
--------------------------

.. autofunction:: concat
.. autofunction:: reverse
.. autofunction:: moveaxis
.. autofunction:: take
.. autofunction:: take_interp
.. autofunction:: compress
.. autofunction:: ravel
.. autofunction:: unravel
.. autofunction:: reshape
.. autofunction:: tile
.. autofunction:: repeat
.. autofunction:: resample
.. autofunction:: convolve

Random number generation
------------------------

.. autofunction:: rng

.. py:module:: drjit.random

.. autoclass:: Generator

   .. automethod:: random
   .. automethod:: uniform
   .. automethod:: normal
   .. automethod:: clone

.. py:module:: drjit

Mask operations
---------------

Also relevant here are :py:func:`any`, :py:func:`all`, :py:func:`none`, and :py:func:`count`.

.. autofunction:: select
.. autofunction:: isinf
.. autofunction:: isnan
.. autofunction:: isfinite
.. autofunction:: allclose

Miscellaneous operations
------------------------

.. autofunction:: shape
.. autofunction:: width
.. autofunction:: slice_index
.. autofunction:: meshgrid
.. autofunction:: binary_search
.. autofunction:: make_opaque
.. autofunction:: copy
.. autofunction:: linear_to_srgb
.. autofunction:: srgb_to_linear
.. autofunction:: reorder_threads

Just-in-time compilation
------------------------

.. autoenum:: JitBackend
.. autoenum:: VarType
.. autoenum:: VarState
.. autoenum:: JitFlag

.. autofunction:: has_backend
.. autofunction:: schedule
.. autofunction:: eval
.. autofunction:: set_flag
.. autofunction:: flag

.. autoclass:: scoped_set_flag

   .. automethod:: __init__
   .. automethod:: __enter__
   .. automethod:: __exit__

Function freezing
-----------------

.. autofunction:: freeze

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
.. autofunction:: is_half_v
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

    This variable stores an alias of ``None``. It is used to create new axes in
    tensor slicing operations (analogous to ``np.newaxis`` in NumPy). See the
    discussion of :ref:`tensors <tensors>` for an example.

Access to related types
_______________________

.. autofunction:: mask_t
.. autofunction:: value_t
.. autofunction:: scalar_t
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
.. autofunction:: array_t
.. autofunction:: tensor_t
.. autofunction:: matrix_t
.. autofunction:: leaf_t

Bit-level operations
--------------------
.. autofunction:: reinterpret_array
.. autofunction:: popcnt
.. autofunction:: lzcnt
.. autofunction:: tzcnt
.. autofunction:: brev

.. autofunction:: log2i

Standard mathematical functions
-------------------------------

.. autofunction:: fma
.. autofunction:: abs
.. autofunction:: minimum
.. autofunction:: maximum
.. autofunction:: sqrt
.. autofunction:: cbrt
.. autofunction:: rcp
.. autofunction:: rsqrt
.. autofunction:: clip
.. autofunction:: ceil
.. autofunction:: floor
.. autofunction:: trunc
.. autofunction:: round
.. autofunction:: sign
.. autofunction:: copysign
.. autofunction:: mulsign
.. autofunction:: step
.. autofunction:: mul_hi
.. autofunction:: mul_wide

Operations for vectors and matrices
-----------------------------------

.. autofunction:: cross
.. autofunction:: det
.. autofunction:: diag
.. autofunction:: trace
.. autofunction:: matmul
.. autofunction:: hypot
.. autofunction:: normalize
.. autofunction:: lerp
.. autofunction:: sh_eval
.. autofunction:: frob
.. autofunction:: rotate
.. autofunction:: polar_decomp
.. autofunction:: matrix_to_quat
.. autofunction:: quat_to_matrix
.. autofunction:: transform_decompose
.. autofunction:: transform_compose
.. autofunction:: unit_angle

Operations for complex values and quaternions
---------------------------------------------

.. autofunction:: conj
.. autofunction:: arg
.. autofunction:: real
.. autofunction:: imag
.. autofunction:: quat_to_euler
.. autofunction:: euler_to_quat

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

Other
_____

.. autofunction:: erf
.. autofunction:: erfinv
.. autofunction:: lgamma
.. autofunction:: rad2deg
.. autofunction:: deg2rad
.. autofunction:: sphdir

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

Automatic differentiation
-------------------------

.. autoenum:: ADMode
.. autoenum:: ADFlag

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
.. autofunction:: wrap


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
    .. autoproperty:: T
    .. autoproperty:: index
    .. autoproperty:: index_ad
    .. autoproperty:: grad
    .. automethod:: item
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
    .. automethod:: __matmul__
    .. automethod:: __rmatmul__
    .. automethod:: __imatmul__
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
    .. automethod:: numpy
    .. automethod:: torch
    .. automethod:: jax
    .. automethod:: tf
    .. automethod:: memview


Computation graph analysis
--------------------------

The following operations visualize the contents of Dr.Jit's computation graphs
(of which there are *two*: one for Jit compilation, and one for automatic
differentiation).

.. autofunction:: graphviz
.. autofunction:: graphviz_ad
.. autofunction:: whos
.. autofunction:: whos_ad
.. autofunction:: set_label

Debugging
---------

.. autoenum:: LogLevel
.. autofunction:: assert_true
.. autofunction:: assert_false
.. autofunction:: assert_equal
.. autofunction:: print
.. autofunction:: format
.. autofunction:: log_level
.. autofunction:: set_log_level

Profiling
---------

.. autoclass:: profile_enable
.. autofunction:: profile_mark
.. autoclass:: profile_range

Textures
--------

The texture implementations are defined in the various backends.
(e.g. :py:class:`drjit.llvm.ad.Texture3f16`). However, they reference
enumerations provided here

.. autoenum:: WrapMode
.. autoenum:: FilterMode

Low-level bits
--------------

.. autofunction:: set_backend
.. autofunction:: thread_count
.. autofunction:: set_thread_count
.. autofunction:: sync_thread
.. autofunction:: flush_kernel_cache
.. autofunction:: flush_malloc_cache
.. autofunction:: expand_threshold
.. autofunction:: set_expand_threshold
.. autofunction:: kernel_history
.. autofunction:: kernel_history_clear

.. py:currentmodule:: drjit.detail
.. autofunction:: set_leak_warnings
.. autofunction:: leak_warnings
.. autofunction:: llvm_version
.. autofunction:: cuda_version
.. py:currentmodule:: drjit

Typing
------

.. autoattribute:: T
.. autoattribute:: Ts
.. autoattribute:: AnyArray

Local memory
------------

.. autofunction:: alloc_local
.. autoclass:: Local

   .. automethod:: __init__
   .. automethod:: read
   .. automethod:: write
   .. automethod:: __getitem__
   .. automethod:: __setitem__
   .. automethod:: __len__

Digital Differential Analyzer
-----------------------------

.. py:module:: drjit.dda

The :py:mod:`drjit.dda` module provides a general implementation of a *digital
differential analyzer* (DDA) that steps through the intersection of a ray
segment and a N-dimensional grid, performing a custom computation at every
cell.

The :py:func:`drjit.integrate` function builds on this functionality to compute
differentiable line integrals of bi- or trilinearly interpolants stored on a
grid.

.. autofunction:: dda
.. autofunction:: integrate

Optimizers
----------

.. py:module:: drjit.opt

The :py:mod:`drjit.opt` module implements basic infrastructure for
gradient-based optimization and adaptive mixed-precision training.

.. autoclass:: Optimizer

   .. automethod:: step
   .. automethod:: reset
   .. automethod:: update
   .. automethod:: learning_rate
   .. automethod:: set_learning_rate
   .. automethod:: __setitem__
   .. automethod:: __getitem__
   .. automethod:: __delitem__
   .. automethod:: __contains__
   .. automethod:: __len__
   .. automethod:: keys
   .. automethod:: values
   .. automethod:: items

.. autoclass:: SGD

   .. automethod:: __init__

.. autoclass:: Adam

   .. automethod:: __init__

.. autoclass:: RMSProp

   .. automethod:: __init__

.. autoclass:: GradScaler

   .. automethod:: __init__
   .. automethod:: step
   .. automethod:: scale
   .. automethod:: unscale

.. _coop_vec_ref:

Cooperative Vectors
-------------------

.. py:module:: drjit.nn

The :py:mod:`drjit.nn` module provides infrastructure to implement small
neural networks and revolves around the notion of *cooperative vectors* that
facilitate code generation of matrix-vector products. Please see the separate
:ref:`documentation section <coop_vec>` for an introduction.

.. autoclass:: CoopVec

   .. automethod:: __init__
   .. automethod:: __add__
   .. automethod:: __sub__
   .. automethod:: __mul__
   .. automethod:: __len__
   .. automethod:: __repr__

   .. property:: index
      :type: int

      Stores the Dr.Jit variable index of the cooperative vector.

   .. property:: type
      :type: type[drjit.ArrayBase]

      Stores the element type

.. autoclass:: MatrixView

   .. automethod:: __getitem__

   .. property:: dtype
      :type: drjit.VarType

      Scalar type underlying the view.

   .. property:: shape
      :type: tuple[int, int]

      Number of rows/columns. Vectors are stored as matrices with one column.

   .. property:: layout
      :type: MatrixLayout

      One of several possible matrix layouts (training/inference-optimal and
      row-major).

   .. property:: stride
      :type: int

      Row stride (in # of elements)

   .. property:: size
      :type: int

      Total number of elements

   .. property:: offset
      :type: int

      Offset of the matrix data within ``buffer`` (counted in # of elements)

   .. property:: transpose
      :type: bool

      The ``MatrixView.T`` property flips this flag (all other
      values stay unchanged).

   .. property:: buffer
      :type: drjit.ArrayBase

      The underlying buffer, which may contain additional matrices/vectors
      besides the data referenced by the :py:class:`MatrixView`.

   .. property:: T
      :type: MatrixView

      Return a transposed view.

   .. property:: grad
      :type: MatrixView

      Return an analogous view of the gradient.

.. autofunction:: view
.. autofunction:: pack
.. autofunction:: unpack
.. autofunction:: matvec
.. autofunction:: cast

Neural Networks
---------------

Besides :ref:`cooperative vector classes <coop_vec_ref>`, the
:py:mod:`drjit.nn` module also provides convenient abstractions to declare,
evaluate, and train networks. Please see the separate :ref:`documentation
section <neural_nets>` for an introduction.

.. autoclass:: Module

   .. automethod:: __call__
   .. automethod:: alloc

.. autoclass:: Sequential

   .. automethod:: __len__
   .. automethod:: __getitem__

.. autoclass:: Linear
.. autoclass:: ReLU
.. autoclass:: LeakyReLU
.. autoclass:: SinEncode
.. autoclass:: TriEncode
.. autoclass:: Exp
.. autoclass:: Exp2
.. autoclass:: Tanh
.. autoclass:: Cast
.. autoclass:: ScaleAdd
.. autoclass:: HashEncodingLayer
.. autoclass:: HashGridEncoding

    .. autoproperty:: params
    .. automethod:: __call__

.. autoclass:: PermutoEncoding

    .. autoproperty:: params
    .. automethod:: __call__


CUDA / GL interoperability
--------------------------

.. py:currentmodule:: drjit.cuda

High-level interface
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: GLInterop


Low-level interface
^^^^^^^^^^^^^^^^^^^

.. autofunction:: register_gl_buffer
.. autofunction:: register_gl_texture
.. autofunction:: unregister_cuda_resource
.. autofunction:: map_graphics_resource_ptr
.. autofunction:: map_graphics_resource_array
.. autofunction:: unmap_graphics_resource
.. autofunction:: memcpy_2d_to_array_async
