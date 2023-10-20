Reference
=========

.. py:module:: drjit

Array creation
--------------

.. autofunction:: zeros
.. autofunction:: ones
.. autofunction:: full
.. autofunction:: identity
.. autofunction:: arange
.. autofunction:: linspace
.. autofunction:: tile
.. autofunction:: repeat
.. autofunction:: meshgrid

.. _horizontal-reductions:

Horizontal operations
---------------------

These operations are *horizontal* in the sense that [..]

.. autofunction:: gather
.. autofunction:: scatter
.. autofunction:: scatter_reduce
.. autofunction:: scatter_inc
.. autofunction:: ravel
.. autofunction:: unravel
.. autofunction:: slice
.. autofunction:: min
.. autofunction:: max
.. autofunction:: sum
.. autofunction:: prod
.. autofunction:: dot
.. autofunction:: norm
.. autofunction:: shuffle
.. autofunction:: compress
.. autofunction:: count
.. autofunction:: block_sum

Mask operations
---------------

.. autofunction:: select
.. autofunction:: all
.. autofunction:: any
.. autofunction:: all_nested
.. autofunction:: any_nested
.. autofunction:: eq
.. autofunction:: neq
.. autofunction:: isinf
.. autofunction:: isnan
.. autofunction:: isfinite
.. autofunction:: allclose

Miscellaneous operations
------------------------

.. autofunction:: shape
.. autofunction:: width
.. autofunction:: resize
.. autofunction:: slice_index
.. autofunction:: binary_search
.. autofunction:: upsample
.. autofunction:: tzcnt
.. autofunction:: lzcnt
.. autofunction:: popcnt

Function dispatching
--------------------

.. autofunction:: switch
.. autofunction:: dispatch

Just-in-time compilation
------------------------

.. autofunction:: schedule
.. autofunction:: eval
.. autofunction:: printf_async
.. autofunction:: graphviz
.. autofunction:: label
.. autofunction:: set_label

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
.. autofunction:: is_jit_v
.. autofunction:: is_llvm_v
.. autofunction:: is_diff_v
.. autofunction:: is_complex_v
.. autofunction:: is_matrix_v
.. autofunction:: is_quaternion_v
.. autofunction:: is_tensor_v
.. autofunction:: is_special_v
.. autofunction:: is_struct_v

Array shape
___________

.. autofunction:: size_v
.. autofunction:: depth_v
.. autofunction:: itemsize_v

.. py:data:: Dynamic
    :type: int
    :value: -1

    Special size value used to identify dynamic arrays in
    :py:func:`array_size_v`.

Access to related types
_______________________

.. autofunction:: mask_t
.. autofunction:: value_t
.. autofunction:: scalar_t
.. autofunction:: array_t
.. autofunction:: bool_array_t
.. autofunction:: int_array_t
.. autofunction:: uint_array_t
.. autofunction:: int32_array_t
.. autofunction:: uint32_array_t
.. autofunction:: int64_array_t
.. autofunction:: uint64_array_t
.. autofunction:: float_array_t
.. autofunction:: float32_array_t
.. autofunction:: float64_array_t
.. autofunction:: leaf_array_t
.. autofunction:: expr_t
.. autofunction:: diff_array_t
.. autofunction:: detached_t

Others
______

.. autofunction:: reinterpret_array_v

Standard mathematical functions
-------------------------------

.. autofunction:: abs
.. autofunction:: minimum
.. autofunction:: maximum
.. autofunction:: clip
.. autofunction:: clamp
.. autofunction:: fma
.. autofunction:: ceil
.. autofunction:: floor
.. autofunction:: trunc
.. autofunction:: round
.. autofunction:: sqrt
.. autofunction:: cbrt
.. autofunction:: rcp
.. autofunction:: rsqrt
.. autofunction:: frexp
.. autofunction:: ldexp
.. autofunction:: lerp
.. autofunction:: normalize
.. autofunction:: log2i
.. autofunction:: erf
.. autofunction:: erfinv
.. autofunction:: lgamma
.. autofunction:: tgamma
.. autofunction:: hypot
.. autofunction:: sign
.. autofunction:: copysign
.. autofunction:: mulsign
.. autofunction:: arg
.. autofunction:: real
.. autofunction:: imag
.. autofunction:: conj
.. autofunction:: cross
.. autofunction:: sh_eval

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

.. _transcendental-accuracy:

Accuracy (single precision)
___________________________

.. note::

    The trigonometric functions *sin*, *cos*, and *tan* are optimized for low
    error on the domain :math:`|x| < 8192` and don't perform as well beyond
    this range.

.. list-table::
    :widths: 5 8 8 10 8 10
    :header-rows: 1
    :align: center

    * - Function
      - Tested domain
      - Abs. error (mean)
      - Abs. error (max)
      - Rel. error (mean)
      - Rel. error (max)
    * - :math:`\text{sin}()`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\text{ulp})`
      - :math:`1.8 \cdot 10^{-6}\,(19\,\text{ulp})`
    * - :math:`\text{cos}()`
      - :math:`-8192 < x < 8192`
      - :math:`1.2 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`1.9 \cdot 10^{-8}\,(0.25\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(47\,\text{ulp})`
    * - :math:`\text{tan}()`
      - :math:`-8192 < x < 8192`
      - :math:`4.7 \cdot 10^{-6}`
      - :math:`8.1 \cdot 10^{-1}`
      - :math:`3.4 \cdot 10^{-8}\,(0.42\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-6}\,(30\,\text{ulp})`
    * - :math:`\text{asin}()`
      - :math:`-1 < x < 1`
      - :math:`2.3 \cdot 10^{-8}`
      - :math:`1.2 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\text{ulp})`
      - :math:`2.3 \cdot 10^{-7}\,(2\,\text{ulp})`
    * - :math:`\text{acos}()`
      - :math:`-1 < x < 1`
      - :math:`4.7 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`2.9 \cdot 10^{-8}\,(0.33\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{atan}()`
      - :math:`-1 < x < 1`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`6 \cdot 10^{-7}`
      - :math:`4.2 \cdot 10^{-7}\,(4.9\,\text{ulp})`
      - :math:`8.2 \cdot 10^{-7}\,(12\,\text{ulp})`
    * - :math:`\text{sinh}()`
      - :math:`-10 < x < 10`
      - :math:`2.6 \cdot 10^{-5}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.8 \cdot 10^{-8}\,(0.34\,\text{ulp})`
      - :math:`2.7 \cdot 10^{-7}\,(3\,\text{ulp})`
    * - :math:`\text{cosh}()`
      - :math:`-10 < x < 10`
      - :math:`2.9 \cdot 10^{-5}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.9 \cdot 10^{-8}\,(0.35\,\text{ulp})`
      - :math:`2.5 \cdot 10^{-7}\,(4\,\text{ulp})`
    * - :math:`\text{tanh}()`
      - :math:`-10 < x < 10`
      - :math:`4.8 \cdot 10^{-8}`
      - :math:`4.2 \cdot 10^{-7}`
      - :math:`5 \cdot 10^{-8}\,(0.76\,\text{ulp})`
      - :math:`5 \cdot 10^{-7}\,(7\,\text{ulp})`
    * - :math:`\text{asinh}()`
      - :math:`-30 < x < 30`
      - :math:`2.8 \cdot 10^{-8}`
      - :math:`4.8 \cdot 10^{-7}`
      - :math:`1 \cdot 10^{-8}\,(0.13\,\text{ulp})`
      - :math:`1.7 \cdot 10^{-7}\,(2\,\text{ulp})`
    * - :math:`\text{acosh}()`
      - :math:`1 < x < 10`
      - :math:`2.9 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\text{ulp})`
      - :math:`2.4 \cdot 10^{-7}\,(3\,\text{ulp})`
    * - :math:`\text{atanh}()`
      - :math:`-1 < x < 1`
      - :math:`9.9 \cdot 10^{-9}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`1.5 \cdot 10^{-8}\,(0.18\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{exp}()`
      - :math:`-20 < x < 30`
      - :math:`0.72 \cdot 10^{4}`
      - :math:`0.1 \cdot 10^{7}`
      - :math:`2.4 \cdot 10^{-8}\,(0.27\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{log}()`
      - :math:`10^{-20} < x < 2\cdot 10^{30}`
      - :math:`9.6 \cdot 10^{-9}`
      - :math:`7.6 \cdot 10^{-6}`
      - :math:`1.4 \cdot 10^{-10}\,(0.0013\,\text{ulp})`
      - :math:`1.2 \cdot 10^{-7}\,(1\,\text{ulp})`
    * - :math:`\text{erf}()`
      - :math:`-1 < x < 1`
      - :math:`3.2 \cdot 10^{-8}`
      - :math:`1.8 \cdot 10^{-7}`
      - :math:`6.4 \cdot 10^{-8}\,(0.78\,\text{ulp})`
      - :math:`3.3 \cdot 10^{-7}\,(4\,\text{ulp})`
    * - :math:`\text{erfc}()`
      - :math:`-1 < x < 1`
      - :math:`3.4 \cdot 10^{-8}`
      - :math:`2.4 \cdot 10^{-7}`
      - :math:`6.4 \cdot 10^{-8}\,(0.79\,\text{ulp})`
      - :math:`1 \cdot 10^{-6}\,(11\,\text{ulp})`

Accuracy (double precision)
___________________________

.. list-table::
    :widths: 5 8 8 10 8 10
    :header-rows: 1
    :align: center

    * - Function
      - Tested domain
      - Abs. error (mean)
      - Abs. error (max)
      - Rel. error (mean)
      - Rel. error (max)
    * - :math:`\text{sin}()`
      - :math:`-8192 < x < 8192`
      - :math:`2.2 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`3.6 \cdot 10^{-17}\,(0.25\,\text{ulp})`
      - :math:`3.1 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{cos}()`
      - :math:`-8192 < x < 8192`
      - :math:`2.2 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`3.6 \cdot 10^{-17}\,(0.25\,\text{ulp})`
      - :math:`3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{tan}()`
      - :math:`-8192 < x < 8192`
      - :math:`6.8 \cdot 10^{-16}`
      - :math:`1.2 \cdot 10^{-10}`
      - :math:`5.4 \cdot 10^{-17}\,(0.35\,\text{ulp})`
      - :math:`4.1 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{cot}()`
      - :math:`-8192 < x < 8192`
      - :math:`4.9 \cdot 10^{-16}`
      - :math:`1.2 \cdot 10^{-10}`
      - :math:`5.5 \cdot 10^{-17}\,(0.36\,\text{ulp})`
      - :math:`4.4 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{asin}()`
      - :math:`-1 < x < 1`
      - :math:`1.3 \cdot 10^{-17}`
      - :math:`2.2 \cdot 10^{-16}`
      - :math:`1.5 \cdot 10^{-17}\,(0.098\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{acos}()`
      - :math:`-1 < x < 1`
      - :math:`5.4 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`3.5 \cdot 10^{-17}\,(0.23\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{atan}()`
      - :math:`-1 < x < 1`
      - :math:`4.3 \cdot 10^{-17}`
      - :math:`3.3 \cdot 10^{-16}`
      - :math:`1 \cdot 10^{-16}\,(0.65\,\text{ulp})`
      - :math:`7.1 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{sinh}()`
      - :math:`-10 < x < 10`
      - :math:`3.1 \cdot 10^{-14}`
      - :math:`1.8 \cdot 10^{-12}`
      - :math:`3.3 \cdot 10^{-17}\,(0.22\,\text{ulp})`
      - :math:`4.3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{cosh}()`
      - :math:`-10 < x < 10`
      - :math:`2.2 \cdot 10^{-14}`
      - :math:`1.8 \cdot 10^{-12}`
      - :math:`2 \cdot 10^{-17}\,(0.13\,\text{ulp})`
      - :math:`2.9 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{tanh}()`
      - :math:`-10 < x < 10`
      - :math:`5.6 \cdot 10^{-17}`
      - :math:`3.3 \cdot 10^{-16}`
      - :math:`6.1 \cdot 10^{-17}\,(0.52\,\text{ulp})`
      - :math:`5.5 \cdot 10^{-16}\,(3\,\text{ulp})`
    * - :math:`\text{asinh}()`
      - :math:`-30 < x < 30`
      - :math:`5.1 \cdot 10^{-17}`
      - :math:`8.9 \cdot 10^{-16}`
      - :math:`1.9 \cdot 10^{-17}\,(0.13\,\text{ulp})`
      - :math:`4.4 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{acosh}()`
      - :math:`1 < x < 10`
      - :math:`4.9 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`2.6 \cdot 10^{-17}\,(0.17\,\text{ulp})`
      - :math:`6.6 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{atanh}()`
      - :math:`-1 < x < 1`
      - :math:`1.8 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`3.2 \cdot 10^{-17}\,(0.21\,\text{ulp})`
      - :math:`3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{exp}()`
      - :math:`-20 < x < 30`
      - :math:`4.7 \cdot 10^{-6}`
      - :math:`2 \cdot 10^{-3}`
      - :math:`2.5 \cdot 10^{-17}\,(0.16\,\text{ulp})`
      - :math:`3.3 \cdot 10^{-16}\,(2\,\text{ulp})`
    * - :math:`\text{log}()`
      - :math:`10^{-20} < x < 2\cdot 10^{30}`
      - :math:`1.9 \cdot 10^{-17}`
      - :math:`1.4 \cdot 10^{-14}`
      - :math:`2.7 \cdot 10^{-19}\,(0.0013\,\text{ulp})`
      - :math:`2.2 \cdot 10^{-16}\,(1\,\text{ulp})`
    * - :math:`\text{erf}()`
      - :math:`-1 < x < 1`
      - :math:`4.7 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`9.6 \cdot 10^{-17}\,(0.63\,\text{ulp})`
      - :math:`5.9 \cdot 10^{-16}\,(5\,\text{ulp})`
    * - :math:`\text{erfc}()`
      - :math:`-1 < x < 1`
      - :math:`4.8 \cdot 10^{-17}`
      - :math:`4.4 \cdot 10^{-16}`
      - :math:`9.6 \cdot 10^{-17}\,(0.64\,\text{ulp})`
      - :math:`2.5 \cdot 10^{-15}\,(16\,\text{ulp})`

Safe mathematical functions
---------------------------

.. autofunction:: safe_sqrt
.. autofunction:: safe_asin
.. autofunction:: safe_acos

Constants
---------

.. autoattribute:: const.e
.. autoattribute:: const.log_two
.. autoattribute:: const.inv_log_two
.. autoattribute:: const.pi
.. autoattribute:: const.inv_pi
.. autoattribute:: const.sqrt_pi
.. autoattribute:: const.inv_sqrt_pi
.. autoattribute:: const.two_pi
.. autoattribute:: const.inv_two_pi
.. autoattribute:: const.sqrt_two_pi
.. autoattribute:: const.inv_sqrt_two_pi
.. autoattribute:: const.four_pi
.. autoattribute:: const.inv_four_pi
.. autoattribute:: const.sqrt_four_pi
.. autoattribute:: const.sqrt_two
.. autoattribute:: const.inv_sqrt_two
.. autoattribute:: const.inf
.. autoattribute:: const.nan
.. autofunction:: epsilon
.. autofunction:: one_minus_epsilon
.. autofunction:: recip_overflow
.. autofunction:: smallest
.. autofunction:: largest

Array base class
----------------

.. autoclass:: ArrayBase

    .. autoproperty:: array
    .. autoproperty:: shape
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
    .. automethod:: __dlpack_device__
    .. automethod:: __array__



Automatic differentiation
-------------------------

.. autofunction:: detach
.. autofunction:: enable_grad
.. autofunction:: disable_grad
.. autofunction:: set_grad_enabled
.. autofunction:: grad_enabled
.. autofunction:: grad
.. autofunction:: set_grad
.. autofunction:: accum_grad
.. autofunction:: replace_grad
.. autofunction:: traverse
.. autofunction:: enqueue
.. autofunction:: forward_from
.. autofunction:: forward
.. autofunction:: forward_to
.. autofunction:: backward_from
.. autofunction:: backward
.. autofunction:: backward_to

.. .. autofunction:: ad_scope_enter
.. .. autofunction:: ad_scope_leave
.. autofunction:: suspend_grad
.. autofunction:: resume_grad
.. autofunction:: isolate_grad
.. autofunction:: graphviz_ad

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
.. autofunction:: wrap_ad

Matrix and quaternion related functions
---------------------------------------

.. autofunction:: rotate
.. autofunction:: transpose
.. autofunction:: inverse_transpose
.. autofunction:: det
.. autofunction:: inverse
.. autofunction:: diag
.. autofunction:: trace
.. autofunction:: frob
.. autofunction:: polar_decomp
.. autofunction:: quat_to_matrix
.. autofunction:: matrix_to_quat
.. autofunction:: quat_to_euler
.. autofunction:: euler_to_quat
.. autofunction:: transform_decompose
.. autofunction:: transform_compose

Concrete array classes
----------------------

Scalar array namespace (``drjit.scalar``)
_________________________________________

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
.. autoclass:: drjit.scalar.Matrix2u
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix2i
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3u
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3i
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4u
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4i
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.scalar.Matrix4f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Complex2f
    :show-inheritance:
.. autoclass:: drjit.scalar.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.scalar.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.scalar.Quaternion4f64
    :show-inheritance:


CUDA array namespace (``drjit.cuda``)
_______________________________________

.. autoclass:: drjit.cuda.Bool
    :show-inheritance:
.. autoclass:: drjit.cuda.Float
    :show-inheritance:
.. autoclass:: drjit.cuda.Float64
    :show-inheritance:
.. autoclass:: drjit.cuda.UInt
    :show-inheritance:
.. autoclass:: drjit.cuda.UInt64
    :show-inheritance:
.. autoclass:: drjit.cuda.Int
    :show-inheritance:
.. autoclass:: drjit.cuda.Int64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4b
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXb
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXf
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0u
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1u
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2u
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3u
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4u
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXu
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0i
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1i
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2i
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3i
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4i
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXi
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXf64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0u64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1u64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2u64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3u64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXu64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array0i64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array1i64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array2i64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array3i64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array4i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ArrayXi64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2u
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2i
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3u
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3i
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4u
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4i
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Complex2f
    :show-inheritance:
.. autoclass:: drjit.cuda.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.cuda.Quaternion4f64
    :show-inheritance:

CUDA array namespace with automatic differentiation (``drjit.cuda.ad``)
_______________________________________________________________________

.. autoclass:: drjit.cuda.ad.Bool
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Float
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Float64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.UInt
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.UInt64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Int
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Int64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXb
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXf
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXu
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXi
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXf64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4u64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXu64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array0i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array1i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array2i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array3i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array4i64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.ArrayXi64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4u
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4i
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Complex2f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Quaternion4f64
    :show-inheritance:

LLVM array namespace (``drjit.llvm``)
_______________________________________

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
.. autoclass:: drjit.llvm.Matrix2u
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix2i
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3u
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3i
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4u
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4i
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.llvm.Matrix4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Complex2f
    :show-inheritance:
.. autoclass:: drjit.llvm.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.llvm.Quaternion4f64
    :show-inheritance:

LLVM array namespace with automatic differentiation (``drjit.llvm.ad``)
_______________________________________________________________________
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
.. autoclass:: drjit.llvm.ad.Matrix2u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix2i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4u
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4i
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Matrix4f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Complex2f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.llvm.ad.Quaternion4f64
    :show-inheritance:
