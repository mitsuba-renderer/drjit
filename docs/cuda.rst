CUDA array namespace (``drjit.cuda``)
_______________________________________

The CUDA backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
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

1D arrays
^^^^^^^^^
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

2D arrays
^^^^^^^^^
.. autoclass:: drjit.cuda.Array22b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array33b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array44b
    :show-inheritance:
.. autoclass:: drjit.cuda.Array22f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array33f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array44f
    :show-inheritance:
.. autoclass:: drjit.cuda.Array22f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array33f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: drjit.cuda.Complex2f
    :show-inheritance:
.. autoclass:: drjit.cuda.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.cuda.Quaternion4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: drjit.cuda.TensorXb
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXf
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXu
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXi
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXf64
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXu64
    :show-inheritance:
.. autoclass:: drjit.cuda.TensorXi64
    :show-inheritance:

CUDA array namespace with automatic differentiation (``drjit.cuda.ad``)
_______________________________________________________________________

The CUDA AD backend is vectorized, hence types listed as *scalar* actually
represent an array of scalars partaking in a parallel computation
(analogously, 1D arrays are arrays of 1D arrays, etc.).

Scalars
^^^^^^^
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

1D arrays
^^^^^^^^^
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

2D arrays
^^^^^^^^^
.. autoclass:: drjit.cuda.ad.Array22b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array33b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array44b
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array22f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array33f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array44f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array22f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array33f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Array44f64
    :show-inheritance:

Special (complex numbers, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: drjit.cuda.ad.Complex2f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Complex2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Quaternion4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Quaternion4f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4f
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix2f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix3f64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.Matrix4f64
    :show-inheritance:

Tensors
^^^^^^^
.. autoclass:: drjit.cuda.ad.TensorXb
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXf
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXu
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXi
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXf64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXu64
    :show-inheritance:
.. autoclass:: drjit.cuda.ad.TensorXi64
    :show-inheritance:
