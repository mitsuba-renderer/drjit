.. py:module:: drjit

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
