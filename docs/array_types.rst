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
    :show-inheritance:
.. autoclass:: Array1b
    :show-inheritance:
.. autoclass:: Array2b
    :show-inheritance:
.. autoclass:: Array3b
    :show-inheritance:
.. autoclass:: Array4b
    :show-inheritance:
.. autoclass:: ArrayXb
    :show-inheritance:
.. autoclass:: Array0f
    :show-inheritance:
.. autoclass:: Array1f
    :show-inheritance:
.. autoclass:: Array2f
    :show-inheritance:
.. autoclass:: Array3f
    :show-inheritance:
.. autoclass:: Array4f
    :show-inheritance:
.. autoclass:: ArrayXf
    :show-inheritance:
.. autoclass:: Array0u
    :show-inheritance:
.. autoclass:: Array1u
    :show-inheritance:
.. autoclass:: Array2u
    :show-inheritance:
.. autoclass:: Array3u
    :show-inheritance:
.. autoclass:: Array4u
    :show-inheritance:
.. autoclass:: ArrayXu
    :show-inheritance:
.. autoclass:: Array0i
    :show-inheritance:
.. autoclass:: Array1i
    :show-inheritance:
.. autoclass:: Array2i
    :show-inheritance:
.. autoclass:: Array3i
    :show-inheritance:
.. autoclass:: Array4i
    :show-inheritance:
.. autoclass:: ArrayXi
    :show-inheritance:
.. autoclass:: Array0f64
    :show-inheritance:
.. autoclass:: Array1f64
    :show-inheritance:
.. autoclass:: Array2f64
    :show-inheritance:
.. autoclass:: Array3f64
    :show-inheritance:
.. autoclass:: Array4f64
    :show-inheritance:
.. autoclass:: ArrayXf64
    :show-inheritance:
.. autoclass:: Array0u64
    :show-inheritance:
.. autoclass:: Array1u64
    :show-inheritance:
.. autoclass:: Array2u64
    :show-inheritance:
.. autoclass:: Array3u64
    :show-inheritance:
.. autoclass:: Array4u64
    :show-inheritance:
.. autoclass:: ArrayXu64
    :show-inheritance:
.. autoclass:: Array0i64
    :show-inheritance:
.. autoclass:: Array1i64
    :show-inheritance:
.. autoclass:: Array2i64
    :show-inheritance:
.. autoclass:: Array3i64
    :show-inheritance:
.. autoclass:: Array4i64
    :show-inheritance:
.. autoclass:: ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: Array22b
    :show-inheritance:
.. autoclass:: Array33b
    :show-inheritance:
.. autoclass:: Array44b
    :show-inheritance:
.. autoclass:: Array22f
    :show-inheritance:
.. autoclass:: Array33f
    :show-inheritance:
.. autoclass:: Array44f
    :show-inheritance:
.. autoclass:: Array22f64
    :show-inheritance:
.. autoclass:: Array33f64
    :show-inheritance:
.. autoclass:: Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f
    :show-inheritance:
.. autoclass:: Complex2f64
    :show-inheritance:
.. autoclass:: Quaternion4f
    :show-inheritance:
.. autoclass:: Quaternion4f64
    :show-inheritance:
.. autoclass:: Matrix2f
    :show-inheritance:
.. autoclass:: Matrix3f
    :show-inheritance:
.. autoclass:: Matrix4f
    :show-inheritance:
.. autoclass:: Matrix2f64
    :show-inheritance:
.. autoclass:: Matrix3f64
    :show-inheritance:
.. autoclass:: Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: TensorXb
    :show-inheritance:
.. autoclass:: TensorXf
    :show-inheritance:
.. autoclass:: TensorXu
    :show-inheritance:
.. autoclass:: TensorXi
    :show-inheritance:
.. autoclass:: TensorXf64
    :show-inheritance:
.. autoclass:: TensorXu64
    :show-inheritance:
.. autoclass:: TensorXi64
    :show-inheritance:

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
    :show-inheritance:
.. autoclass:: Float
    :show-inheritance:
.. autoclass:: Float64
    :show-inheritance:
.. autoclass:: UInt
    :show-inheritance:
.. autoclass:: UInt64
    :show-inheritance:
.. autoclass:: Int
    :show-inheritance:
.. autoclass:: Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: Array0b
    :show-inheritance:
.. autoclass:: Array1b
    :show-inheritance:
.. autoclass:: Array2b
    :show-inheritance:
.. autoclass:: Array3b
    :show-inheritance:
.. autoclass:: Array4b
    :show-inheritance:
.. autoclass:: ArrayXb
    :show-inheritance:
.. autoclass:: Array0f
    :show-inheritance:
.. autoclass:: Array1f
    :show-inheritance:
.. autoclass:: Array2f
    :show-inheritance:
.. autoclass:: Array3f
    :show-inheritance:
.. autoclass:: Array4f
    :show-inheritance:
.. autoclass:: ArrayXf
    :show-inheritance:
.. autoclass:: Array0u
    :show-inheritance:
.. autoclass:: Array1u
    :show-inheritance:
.. autoclass:: Array2u
    :show-inheritance:
.. autoclass:: Array3u
    :show-inheritance:
.. autoclass:: Array4u
    :show-inheritance:
.. autoclass:: ArrayXu
    :show-inheritance:
.. autoclass:: Array0i
    :show-inheritance:
.. autoclass:: Array1i
    :show-inheritance:
.. autoclass:: Array2i
    :show-inheritance:
.. autoclass:: Array3i
    :show-inheritance:
.. autoclass:: Array4i
    :show-inheritance:
.. autoclass:: ArrayXi
    :show-inheritance:
.. autoclass:: Array0f64
    :show-inheritance:
.. autoclass:: Array1f64
    :show-inheritance:
.. autoclass:: Array2f64
    :show-inheritance:
.. autoclass:: Array3f64
    :show-inheritance:
.. autoclass:: Array4f64
    :show-inheritance:
.. autoclass:: ArrayXf64
    :show-inheritance:
.. autoclass:: Array0u64
    :show-inheritance:
.. autoclass:: Array1u64
    :show-inheritance:
.. autoclass:: Array2u64
    :show-inheritance:
.. autoclass:: Array3u64
    :show-inheritance:
.. autoclass:: Array4u64
    :show-inheritance:
.. autoclass:: ArrayXu64
    :show-inheritance:
.. autoclass:: Array0i64
    :show-inheritance:
.. autoclass:: Array1i64
    :show-inheritance:
.. autoclass:: Array2i64
    :show-inheritance:
.. autoclass:: Array3i64
    :show-inheritance:
.. autoclass:: Array4i64
    :show-inheritance:
.. autoclass:: ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: Array22b
    :show-inheritance:
.. autoclass:: Array33b
    :show-inheritance:
.. autoclass:: Array44b
    :show-inheritance:
.. autoclass:: Array22f
    :show-inheritance:
.. autoclass:: Array33f
    :show-inheritance:
.. autoclass:: Array44f
    :show-inheritance:
.. autoclass:: Array22f64
    :show-inheritance:
.. autoclass:: Array33f64
    :show-inheritance:
.. autoclass:: Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f
    :show-inheritance:
.. autoclass:: Complex2f64
    :show-inheritance:
.. autoclass:: Quaternion4f
    :show-inheritance:
.. autoclass:: Quaternion4f64
    :show-inheritance:
.. autoclass:: Matrix2f
    :show-inheritance:
.. autoclass:: Matrix3f
    :show-inheritance:
.. autoclass:: Matrix4f
    :show-inheritance:
.. autoclass:: Matrix2f64
    :show-inheritance:
.. autoclass:: Matrix3f64
    :show-inheritance:
.. autoclass:: Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: TensorXb
    :show-inheritance:
.. autoclass:: TensorXf
    :show-inheritance:
.. autoclass:: TensorXu
    :show-inheritance:
.. autoclass:: TensorXi
    :show-inheritance:
.. autoclass:: TensorXf64
    :show-inheritance:
.. autoclass:: TensorXu64
    :show-inheritance:
.. autoclass:: TensorXi64
    :show-inheritance:

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
    :show-inheritance:
.. autoclass:: Float
    :show-inheritance:
.. autoclass:: Float64
    :show-inheritance:
.. autoclass:: UInt
    :show-inheritance:
.. autoclass:: UInt64
    :show-inheritance:
.. autoclass:: Int
    :show-inheritance:
.. autoclass:: Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: Array0b
    :show-inheritance:
.. autoclass:: Array1b
    :show-inheritance:
.. autoclass:: Array2b
    :show-inheritance:
.. autoclass:: Array3b
    :show-inheritance:
.. autoclass:: Array4b
    :show-inheritance:
.. autoclass:: ArrayXb
    :show-inheritance:
.. autoclass:: Array0f
    :show-inheritance:
.. autoclass:: Array1f
    :show-inheritance:
.. autoclass:: Array2f
    :show-inheritance:
.. autoclass:: Array3f
    :show-inheritance:
.. autoclass:: Array4f
    :show-inheritance:
.. autoclass:: ArrayXf
    :show-inheritance:
.. autoclass:: Array0u
    :show-inheritance:
.. autoclass:: Array1u
    :show-inheritance:
.. autoclass:: Array2u
    :show-inheritance:
.. autoclass:: Array3u
    :show-inheritance:
.. autoclass:: Array4u
    :show-inheritance:
.. autoclass:: ArrayXu
    :show-inheritance:
.. autoclass:: Array0i
    :show-inheritance:
.. autoclass:: Array1i
    :show-inheritance:
.. autoclass:: Array2i
    :show-inheritance:
.. autoclass:: Array3i
    :show-inheritance:
.. autoclass:: Array4i
    :show-inheritance:
.. autoclass:: ArrayXi
    :show-inheritance:
.. autoclass:: Array0f64
    :show-inheritance:
.. autoclass:: Array1f64
    :show-inheritance:
.. autoclass:: Array2f64
    :show-inheritance:
.. autoclass:: Array3f64
    :show-inheritance:
.. autoclass:: Array4f64
    :show-inheritance:
.. autoclass:: ArrayXf64
    :show-inheritance:
.. autoclass:: Array0u64
    :show-inheritance:
.. autoclass:: Array1u64
    :show-inheritance:
.. autoclass:: Array2u64
    :show-inheritance:
.. autoclass:: Array3u64
    :show-inheritance:
.. autoclass:: Array4u64
    :show-inheritance:
.. autoclass:: ArrayXu64
    :show-inheritance:
.. autoclass:: Array0i64
    :show-inheritance:
.. autoclass:: Array1i64
    :show-inheritance:
.. autoclass:: Array2i64
    :show-inheritance:
.. autoclass:: Array3i64
    :show-inheritance:
.. autoclass:: Array4i64
    :show-inheritance:
.. autoclass:: ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: Array22b
    :show-inheritance:
.. autoclass:: Array33b
    :show-inheritance:
.. autoclass:: Array44b
    :show-inheritance:
.. autoclass:: Array22f
    :show-inheritance:
.. autoclass:: Array33f
    :show-inheritance:
.. autoclass:: Array44f
    :show-inheritance:
.. autoclass:: Array22f64
    :show-inheritance:
.. autoclass:: Array33f64
    :show-inheritance:
.. autoclass:: Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f
    :show-inheritance:
.. autoclass:: Complex2f64
    :show-inheritance:
.. autoclass:: Quaternion4f
    :show-inheritance:
.. autoclass:: Quaternion4f64
    :show-inheritance:
.. autoclass:: Matrix2f
    :show-inheritance:
.. autoclass:: Matrix3f
    :show-inheritance:
.. autoclass:: Matrix4f
    :show-inheritance:
.. autoclass:: Matrix2f64
    :show-inheritance:
.. autoclass:: Matrix3f64
    :show-inheritance:
.. autoclass:: Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: TensorXb
    :show-inheritance:
.. autoclass:: TensorXf
    :show-inheritance:
.. autoclass:: TensorXu
    :show-inheritance:
.. autoclass:: TensorXi
    :show-inheritance:
.. autoclass:: TensorXf64
    :show-inheritance:
.. autoclass:: TensorXu64
    :show-inheritance:
.. autoclass:: TensorXi64
    :show-inheritance:

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
    :show-inheritance:
.. autoclass:: Float
    :show-inheritance:
.. autoclass:: Float64
    :show-inheritance:
.. autoclass:: UInt
    :show-inheritance:
.. autoclass:: UInt64
    :show-inheritance:
.. autoclass:: Int
    :show-inheritance:
.. autoclass:: Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: Array0b
    :show-inheritance:
.. autoclass:: Array1b
    :show-inheritance:
.. autoclass:: Array2b
    :show-inheritance:
.. autoclass:: Array3b
    :show-inheritance:
.. autoclass:: Array4b
    :show-inheritance:
.. autoclass:: ArrayXb
    :show-inheritance:
.. autoclass:: Array0f
    :show-inheritance:
.. autoclass:: Array1f
    :show-inheritance:
.. autoclass:: Array2f
    :show-inheritance:
.. autoclass:: Array3f
    :show-inheritance:
.. autoclass:: Array4f
    :show-inheritance:
.. autoclass:: ArrayXf
    :show-inheritance:
.. autoclass:: Array0u
    :show-inheritance:
.. autoclass:: Array1u
    :show-inheritance:
.. autoclass:: Array2u
    :show-inheritance:
.. autoclass:: Array3u
    :show-inheritance:
.. autoclass:: Array4u
    :show-inheritance:
.. autoclass:: ArrayXu
    :show-inheritance:
.. autoclass:: Array0i
    :show-inheritance:
.. autoclass:: Array1i
    :show-inheritance:
.. autoclass:: Array2i
    :show-inheritance:
.. autoclass:: Array3i
    :show-inheritance:
.. autoclass:: Array4i
    :show-inheritance:
.. autoclass:: ArrayXi
    :show-inheritance:
.. autoclass:: Array0f64
    :show-inheritance:
.. autoclass:: Array1f64
    :show-inheritance:
.. autoclass:: Array2f64
    :show-inheritance:
.. autoclass:: Array3f64
    :show-inheritance:
.. autoclass:: Array4f64
    :show-inheritance:
.. autoclass:: ArrayXf64
    :show-inheritance:
.. autoclass:: Array0u64
    :show-inheritance:
.. autoclass:: Array1u64
    :show-inheritance:
.. autoclass:: Array2u64
    :show-inheritance:
.. autoclass:: Array3u64
    :show-inheritance:
.. autoclass:: Array4u64
    :show-inheritance:
.. autoclass:: ArrayXu64
    :show-inheritance:
.. autoclass:: Array0i64
    :show-inheritance:
.. autoclass:: Array1i64
    :show-inheritance:
.. autoclass:: Array2i64
    :show-inheritance:
.. autoclass:: Array3i64
    :show-inheritance:
.. autoclass:: Array4i64
    :show-inheritance:
.. autoclass:: ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: Array22b
    :show-inheritance:
.. autoclass:: Array33b
    :show-inheritance:
.. autoclass:: Array44b
    :show-inheritance:
.. autoclass:: Array22f
    :show-inheritance:
.. autoclass:: Array33f
    :show-inheritance:
.. autoclass:: Array44f
    :show-inheritance:
.. autoclass:: Array22f64
    :show-inheritance:
.. autoclass:: Array33f64
    :show-inheritance:
.. autoclass:: Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f
    :show-inheritance:
.. autoclass:: Complex2f64
    :show-inheritance:
.. autoclass:: Quaternion4f
    :show-inheritance:
.. autoclass:: Quaternion4f64
    :show-inheritance:
.. autoclass:: Matrix2f
    :show-inheritance:
.. autoclass:: Matrix3f
    :show-inheritance:
.. autoclass:: Matrix4f
    :show-inheritance:
.. autoclass:: Matrix2f64
    :show-inheritance:
.. autoclass:: Matrix3f64
    :show-inheritance:
.. autoclass:: Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: TensorXb
    :show-inheritance:
.. autoclass:: TensorXf
    :show-inheritance:
.. autoclass:: TensorXu
    :show-inheritance:
.. autoclass:: TensorXi
    :show-inheritance:
.. autoclass:: TensorXf64
    :show-inheritance:
.. autoclass:: TensorXu64
    :show-inheritance:
.. autoclass:: TensorXi64
    :show-inheritance:

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
    :show-inheritance:
.. autoclass:: Float
    :show-inheritance:
.. autoclass:: Float64
    :show-inheritance:
.. autoclass:: UInt
    :show-inheritance:
.. autoclass:: UInt64
    :show-inheritance:
.. autoclass:: Int
    :show-inheritance:
.. autoclass:: Int64
    :show-inheritance:

1D arrays
^^^^^^^^^
.. autoclass:: Array0b
    :show-inheritance:
.. autoclass:: Array1b
    :show-inheritance:
.. autoclass:: Array2b
    :show-inheritance:
.. autoclass:: Array3b
    :show-inheritance:
.. autoclass:: Array4b
    :show-inheritance:
.. autoclass:: ArrayXb
    :show-inheritance:
.. autoclass:: Array0f
    :show-inheritance:
.. autoclass:: Array1f
    :show-inheritance:
.. autoclass:: Array2f
    :show-inheritance:
.. autoclass:: Array3f
    :show-inheritance:
.. autoclass:: Array4f
    :show-inheritance:
.. autoclass:: ArrayXf
    :show-inheritance:
.. autoclass:: Array0u
    :show-inheritance:
.. autoclass:: Array1u
    :show-inheritance:
.. autoclass:: Array2u
    :show-inheritance:
.. autoclass:: Array3u
    :show-inheritance:
.. autoclass:: Array4u
    :show-inheritance:
.. autoclass:: ArrayXu
    :show-inheritance:
.. autoclass:: Array0i
    :show-inheritance:
.. autoclass:: Array1i
    :show-inheritance:
.. autoclass:: Array2i
    :show-inheritance:
.. autoclass:: Array3i
    :show-inheritance:
.. autoclass:: Array4i
    :show-inheritance:
.. autoclass:: ArrayXi
    :show-inheritance:
.. autoclass:: Array0f64
    :show-inheritance:
.. autoclass:: Array1f64
    :show-inheritance:
.. autoclass:: Array2f64
    :show-inheritance:
.. autoclass:: Array3f64
    :show-inheritance:
.. autoclass:: Array4f64
    :show-inheritance:
.. autoclass:: ArrayXf64
    :show-inheritance:
.. autoclass:: Array0u64
    :show-inheritance:
.. autoclass:: Array1u64
    :show-inheritance:
.. autoclass:: Array2u64
    :show-inheritance:
.. autoclass:: Array3u64
    :show-inheritance:
.. autoclass:: Array4u64
    :show-inheritance:
.. autoclass:: ArrayXu64
    :show-inheritance:
.. autoclass:: Array0i64
    :show-inheritance:
.. autoclass:: Array1i64
    :show-inheritance:
.. autoclass:: Array2i64
    :show-inheritance:
.. autoclass:: Array3i64
    :show-inheritance:
.. autoclass:: Array4i64
    :show-inheritance:
.. autoclass:: ArrayXi64
    :show-inheritance:

2D arrays
^^^^^^^^^
.. autoclass:: Array22b
    :show-inheritance:
.. autoclass:: Array33b
    :show-inheritance:
.. autoclass:: Array44b
    :show-inheritance:
.. autoclass:: Array22f
    :show-inheritance:
.. autoclass:: Array33f
    :show-inheritance:
.. autoclass:: Array44f
    :show-inheritance:
.. autoclass:: Array22f64
    :show-inheritance:
.. autoclass:: Array33f64
    :show-inheritance:
.. autoclass:: Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: Complex2f
    :show-inheritance:
.. autoclass:: Complex2f64
    :show-inheritance:
.. autoclass:: Quaternion4f
    :show-inheritance:
.. autoclass:: Quaternion4f64
    :show-inheritance:
.. autoclass:: Matrix2f
    :show-inheritance:
.. autoclass:: Matrix3f
    :show-inheritance:
.. autoclass:: Matrix4f
    :show-inheritance:
.. autoclass:: Matrix2f64
    :show-inheritance:
.. autoclass:: Matrix3f64
    :show-inheritance:
.. autoclass:: Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: TensorXb
    :show-inheritance:
.. autoclass:: TensorXf
    :show-inheritance:
.. autoclass:: TensorXu
    :show-inheritance:
.. autoclass:: TensorXi
    :show-inheritance:
.. autoclass:: TensorXf64
    :show-inheritance:
.. autoclass:: TensorXu64
    :show-inheritance:
.. autoclass:: TensorXi64
    :show-inheritance:

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
