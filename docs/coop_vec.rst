.. py:currentmodule:: drjit

.. cpp:namespace:: drjit

.. _coop_vec:

Cooperative vectors
===================

Cooperative vectors enable efficient compilation and evaluation of expressions
involving matrix multiplication. They cater to a specific use case, where each
execution thread performs a sequence of independent multiplications by
reasonably small matrices (e.g., :math:`64\times 64`). This enables the fully
fused evaluation of small `multilayer perceptrons
<https://en.wikipedia.org/wiki/Multilayer_perceptron>`__ (MLPs) within a larger
program. That said, the feature isn't specific to MLPs and could also be used
in other ways.

On NVIDIA GPUs (Turing or newer), cooperative vectors map to the OptiX
`cooperative vector API
<https://raytracing-docs.nvidia.com/optix9/guide/index.html#cooperative_vectors#neural-rendering-with-cooperative-vectors>`__,
leveraging the builtin `tensor core
<https://www.nvidia.com/en-us/data-center/tensor-cores/>`__ for acceleration.
On the CPU (LLVM) backend, Dr.Jit compiles cooperative vector operations using
available instruction set extensions (AVX512, NEON, etc.).


Overview
--------

The cooperative vector API is exposed the :py:mod:`drjit.coop` submodule. Here
is an example use to evaluate a simple MLP:

.. code-block:: python

   import drjit as dr

   # Import 16-bit tensor and floating point types
   from drjit.auto.ad import TensorXf16, Array3f16

   # MLP layer shapes
   shapes = [(16, 3), (16, 16), (16, 16), (3, 16)]

   # Create a set of weight matrices and bias vectors
   for shape in shapes:
       A.append(dr.rand(TensorXf16, shape))
       b.append(dr.rand(TensorXf16, shape[0]))

   # Pack layers into an inference-optimal layout
   A, b = dr.coop.pack(A, b)

   # MLP input
   x, y, z = ...

   # Cast into a cooperative vector
   x = dr.coop.Vector(x, y, z)

   # Evaluate the MLP
   for i in range(len(shapes)):
       x = dr.matvec(A[i], x, b[i])

       # Activation for interior layers
       if i < len(shapes) - 1:
           x = dr.relu(x)

The main type is :py:class:`drjit.coop.Vector`, a dynamically sized vector similar
to the builtin :py:class:`ArrayXf <drjit.cuda.ad.ArrayXf>`. It can represent
elements of various types, though all elements must have the same type.

.. code-block:: python

   # Initialize from individual elements
   a, b = Float16(...), Float16(...)
   x = dr.coop.Vector(a, b)

   # .. from nested array
   x = dr.coop.Vector(Array2f16(...))

   # Element access
   y: Float16 = x[0]

   # Element assignment
   x[0] = 1.5

Cooperative vectors admit a restricted set of arithmetic operations. These
include:

- Elementary arithmetic operations: ``+``, ``-``, ``*`` (but no division)
- :py:func:`dr.fma() <fma>`,
- :py:func:`dr.min() <min>`, :py:func:`dr.max() <max>`,
- :py:func:`dr.log2() <log2>`, :py:func:`dr.exp2() <exp2>`,
- :py:func:`dr.tanh() <tanh>`,
- :py:func:`dr.relu() <relu>`, :py:func:`dr.step() <relu>`.

Operations outside of this set can be realized via element access,
e.g.:

.. code-block::

   x : dr.coop.Vector = ...
   y = dr.coop.Vector(dr.sin(v) for v in x)

However, this may have a negative impact on performance. Only the operations
above map to native cooperative vector operations on the CUDA/OptiX backend.


Performance considerations
--------------------------

On the CUDA/OptiX backend, use of the tensor cores (specifically, the
:py:func:`dr.coop.matvec <drjit.coop.matvec>` operation) currently requires
16-bit floating point arguments. The tensor core in principle supports 8-bit
floating point formats, which might be added to Dr.Jit in the future.

On the LLVM backend, 32-bit floating point formats are preferable. (The FP16
fused multiply-add operation required for matrix multiplication on the CPU is
part of AVX512FP16. However, no consumer CPUs provide this extension as of
2025. Use of FP16 without this extension causes costly roundtrips to cast to
FP32 and back.)
