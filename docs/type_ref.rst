.. py:currentmodule:: drjit

Array types
===========

This section of the documentation lists all array classes that are available
in the various namespaces.

TODO: insert general information about how types are organized here.

Dr.Jit types derive from :py:class:`drjit.ArrayBase` and generally do not
implement any methods beyond those of the base class, which makes this section
rather repetitious.


Scalar array namespace (``drjit.scalar``)
_________________________________________

.. py:module:: drjit.scalar

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
.. py:data:: Float16
    :type: type
    :value: half
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
.. autoclass:: Array0b
.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Textures
^^^^^^^^
.. autoclass:: drjit.scalar.Texture1f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture2f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture3f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture1f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture2f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture3f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture1f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture2f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.scalar.Texture3f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

.. py:module:: drjit.llvm

The LLVM backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalar
^^^^^^

.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Textures
^^^^^^^^
.. autoclass:: drjit.llvm.Texture1f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture2f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture3f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture1f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture2f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture3f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture1f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture2f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.Texture3f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

.. py:module:: drjit.llvm.ad

The LLVM AD backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Textures
^^^^^^^^
.. autoclass:: drjit.llvm.ad.Texture1f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture2f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture3f16

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture1f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture2f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture3f

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture1f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture2f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

.. autoclass:: drjit.llvm.ad.Texture3f64

   .. automethod:: __init__
   .. automethod:: set_value
   .. automethod:: set_tensor
   .. automethod:: value
   .. automethod:: tensor
   .. automethod:: filter_mode
   .. automethod:: wrap_mode
   .. automethod:: use_accel
   .. automethod:: migrated
   .. autoproperty:: shape
   .. automethod:: eval
   .. automethod:: eval_fetch
   .. automethod:: eval_cubic
   .. automethod:: eval_cubic_grad
   .. automethod:: eval_cubic_hessian
   .. automethod:: eval_cubic_helper

Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

CUDA array namespace (``drjit.cuda``)
_______________________________________

.. py:module:: drjit.cuda

The CUDA backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

CUDA array namespace with automatic differentiation (``drjit.cuda.ad``)
_______________________________________________________________________

.. py:module:: drjit.cuda.ad

The CUDA AD backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

Automatic array namespace (``drjit.cuda``)
__________________________________________

.. py:module:: drjit.auto

The automatic backend by default wraps `drjit.cuda` when an CUDA-capable device
was detected, otherwise it wraps `drjit.llvm`.

You can use the function :py:func:`drjit.set_backend` to redirect this module.

This backend is always vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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

Automatic array namespace with automatic differentiation (``drjit.auto.ad``)
____________________________________________________________________________

.. py:module:: drjit.auto.ad

The automatic AD backend by default wraps `drjit.cuda.ad` when an CUDA-capable
device was detected, otherwise it wraps `drjit.llvm.ad`.

You can use the function :py:func:`drjit.set_backend` to redirect this module.

This backend is always vectorized, hence types listed as *scalar* actually represent an
array of scalars partaking in a parallel computation (analogously, 1D arrays
are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
.. autoclass:: Bool

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Float64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: UInt64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Int64

   Derives from :py:class:`drjit.ArrayBase`.


1D arrays
^^^^^^^^^
.. autoclass:: Array0b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4u64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array0i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array1i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array2i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array3i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array4i64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: ArrayXi64

   Derives from :py:class:`drjit.ArrayBase`.


2D arrays
^^^^^^^^^
.. autoclass:: Array22b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44b

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array22f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array33f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Array44f64

   Derives from :py:class:`drjit.ArrayBase`.


Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Complex2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Quaternion4f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix2f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix3f64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: Matrix4f64

   Derives from :py:class:`drjit.ArrayBase`.


Tensors
^^^^^^^
.. autoclass:: TensorXb

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf16

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXf64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXu64

   Derives from :py:class:`drjit.ArrayBase`.

.. autoclass:: TensorXi64

   Derives from :py:class:`drjit.ArrayBase`.


Random number generators
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: PCG32

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
