.. py:currentmodule:: drjit

.. cpp:namespace:: drjit

.. _coop_vec:

Cooperative vectors
===================

*Cooperative vectors* are a `new API
<https://github.com/KhronosGroup/GLSL/blob/main/extensions/nv/GLSL_NV_cooperative_vector.txt>`__
for evaluating matrix-vector products in certain types of GPU workloads. They
are specifically designed for situations, where the matrices in question are
relatively small (e.g., 64x64 or smaller), and where each thread of a parallel
program needs to multiply a different vector by such a matrix. By working
together, threads can perform these multiplications more efficiently, which is
why this approach is called *cooperative*.

Cooperative vectors are especially useful for evaluating small `multilayer
perceptrons <https://en.wikipedia.org/wiki/Multilayer_perceptron>`__ (MLPs)
within larger programs while fully *fusing* all steps of the process into a
single kernel. Other workloads that heavily rely on matrix-vector products may
benefit as well.

Dr.Jit supports cooperative vectors on both backends:

- On **NVIDIA GPUs (Turing or newer)**, cooperative vectors map to the OptiX
  `cooperative vector API
  <https://raytracing-docs.nvidia.com/optix9/guide/index.html#cooperative_vectors#neural-rendering-with-cooperative-vectors>`__,
  leveraging built-in `tensor cores
  <https://www.nvidia.com/en-us/data-center/tensor-cores/>`__ for acceleration.

- On the **CPU (LLVM) backend**, compilation of cooperative vector operations
  targets the available instruction set extensions (AVX512, NEON, etc.).

Note: code examples in the remainder of this section assume the following
include directive:

.. code-block:: python

   import drjit.nn as nn
   from drjit.auto.ad import Float16

Motivation
----------

The cooperative vector API is available via the :py:mod:`drjit.nn` submodule.
Below is an example demonstrating how to evaluate a simple MLP.

.. code-block:: python

   import drjit as dr
   from drjit.auto.ad import TensorXf16  # Import a FP16 tensor type

   # Define MLP layer shapes
   shapes = [(16, 3), (16, 16), (16, 16), (3, 16)]

   # Initialize weights and biases
   A, b = [], []
   for m, n in shapes:
       A.append(dr.rand(TensorXf16, (m, n), scale='xavier'))
       b.append(dr.zeros(TensorXf16, m))

   # Pack layers into an inference-optimal layout
   A, b = nn.pack(A, b)

   # Define MLP input values
   x, y, z = ... # (type: Float16)

   # Pack inputs into a cooperative vector
   v = nn.CoopVector(x, y, z)

   # Forward pass through the MLP
   for i in range(len(shapes)):
       v = nn.matvec(A[i], v, b[i])

       # Apply activation for hidden layers
       if i != len(shapes) - 1:
           v = dr.relu(v)

   # Unpack the resulting cooperative vector
   x, y, z = v

This involves the following steps:

- Initializing weight matrices (e.g, via :py:func:`dr.zeros() <drjit.zeros>`,
  :py:func:`dr.rand() <drjit.rand>` and/or :py:func:`dr.normal()
  <drjit.normal>`).

- Packing coefficients into an optimized memory layout using
  :py:func:`nn.pack() <drjit.nn.pack>`.

- Constructing a :py:class:`nn.CoopVector` containing the MLP inputs.

- Performing matrix-vector multiplications and other arithmetic, while keeping
  the state in cooperative vector form.

- Unpacking the final cooperative vector into regular Dr.Jit arrays.

Cooperative vectors
-------------------

The central type of this API is the *cooperative vector* class
:py:class:`nn.CoopVector`. This is a dynamically sized vector with uniformly
typed elements.

Unlike regular Dr.Jit arrays (e.g. :py:class:`drjit.cuda.ArrayXf`), cooperative
vectors *do not allow indexed element access*. For example, the following
operation raises an exception:

.. code-block:: pycon

   >>> vec = nn.CoopVector(Float16(1), Float16(2))
   >>> vec[1]
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: 'drjit.nn.CoopVector' object is not subscriptable

This restriction exists because the compiler may arbitrarily distribute
cooperative vector components across threads for efficiency. Allowing direct
indexing would interfere with this optimization.

The :py:class:`drjit.nn.CoopVector` constructor accepts an arbitrary sequence
of :ref:`PyTrees <pytrees>` containing Dr.Jit array and Python scalars and
flattens them into a cooperative vector:

.. code-block:: python

   vec = nn.CoopVector( # Construct a 4D vector
       Float16(1),
       3.0,
       Array2f(4, 5)
    )

Use the standard Python unpacking syntax to turn cooperative vectors back into
their components:

.. code-block:: python

   x, y, z = vec      # Unpack a cooperative 3D vector
   x, y, *extra = vec # Unpack first 2 components, put rest into 'extra'

The same syntax can also be used to concatenate vectors:

.. code-block:: python

   vec_3 = nn.CoopVector(*vec_1, *vec_2)

Cooperative vectors can also be converted into nested arrays, tensors, or
Python lists:

.. code-block:: python

   vec_arr = Array3f(vec)
   vec_ten = TensorXf(vec)
   vec_lst = list(vec)

Cooperative vectors are compatible with Dr.Jit's symbolic tracing
infrastructure and may be used as state variables in
:py:func:`drjit.while_loop` and :py:func:`drjit.if_stmt`.

Arithmetic
^^^^^^^^^^

Cooperative vectors support a restricted set of arithmetic operations:

- Elementary arithmetic operations: ``+``, ``-``, ``*`` (but no division)
- :py:func:`dr.fma() <fma>`,
- :py:func:`dr.minimum() <minimum>`, :py:func:`dr.maximum() <maximum>`,
- :py:func:`dr.log2() <log2>`, :py:func:`dr.exp2() <exp2>`,
- :py:func:`dr.tanh() <tanh>`,
- :py:func:`dr.relu() <relu>`, :py:func:`dr.step() <relu>`.
- :py:func:`nn.matvec() <drjit.nn.matvec>`

These operations directly map to hardware-optimized operations on CUDA/OptiX.
Operations outside of this set can be realized via unpacking/repacking, e.g.:

.. code-block::

   x : nn.CoopVector = ...
   y = nn.CoopVector(dr.sin(v) for v in x)

However, this may degrade performance. It is best to keep cooperative vectors
in their opaque layout whenever possible.

Arithmetic operations may mix cooperative vectors and regular Dr.Jit arrays or
Python scalars, which will undergo implicit broadcasting.

.. code-block::

   x: nn.CoopVector[dr.cuda.Float16] = ...
   y: dr.cuda.Float16 = ...
   z = dr.maximum(x, 0) + y

.. _matrix_views:

Matrix views
------------

Input matrices and bias vectors should generally be converted into a
hardware-dependent layout to improve performance compared to the default
row-major representation (also, many operations raise exceptions on the
OptiX/CUDA backend when matrices are not in such an optimal layout).

The function :py:func:`nn.pack() <drjit.nn.pack>` performs this conversion and
furthermore packs data into a shared buffer for optimal efficiency. The
function takes an arbitrary sequence of :ref:`PyTrees <pytrees>` as input and
returns a result with the same structure.

.. code-block:: python

   A: TensorXf = ...
   b: Float = ...
   A_view, b_view = nn.pack(A, b, layout='inference')

Every Dr.Jit array or tensor will be replaced by a
:py:class:`drjit.nn.MatrixView`, which is a thin pointer into a shared buffer
annotated with layout and type metadata. The function can generate optimal
memory layouts for either *inference* (the default) and *training*. You must
specify ``layout='training'`` if you wish to differentiate matrix
multiplication in reverse mode.

Following this step, the layout-converted ``A`` and ``b`` are stored in the
same buffer, (accessible via :py:attr:`MatrixView.buffer
<drjit.nn.MatrixView.buffer>`), and ``A_view`` and ``b_view`` merely encode the
offset and layout.

Matrix views cannot be used in arithmetic expressions (e.g., ``A_view + 1``
fails) and are best thought of an opaque handle. They exist to characterize the
input of the matrix-vector multiplication operation explained next.

Two other view-related operations be useful in certain situations, please
see the linked documentation for details.

- :py:func:`drjit.nn.unpack` converts optimal-layout data back into a row-major layout.
- :py:func:`drjit.nn.view` creates row-major views.

Matrix-vector products
----------------------

The main purpose of cooperative vectors is the matrix-vector multiplication
operation :py:func:`nn.matvec() <drjit.nn.matvec>`:

.. code-block:: python

   y = nn.matvec(A, x, b) # Compute y = A @ x + b

Here,

- ``A`` and ``b`` are *views* (:py:class:`nn.MatrixView`) created by
  :py:func:`nn.pack() <drjit.nn.pack>` or :py:func:`nn.view()
  <drjit.nn.view>`.
- ``x`` and ``y`` are cooperative vectors. They are interpreted as *column
  vectors*, i.e., ``y = A[:, 0] * x[0] + A[:, 1] * x[1] + ... + b``.
- the ``b`` term is optional.

The function also accepts an optional ``transpose=True`` parameter to compute
:math:`A^Tx + b`.

The standard Python ``A @ x`` and ``A.T @ x`` matrix multiplication syntax
works as well. However, if your computation requires the addition of a ``b``
vector, prefer :py:func:`nn.matvec() <drjit.nn.matvec>` over this syntax, since
it merges both steps into a single operation.

Differentiation
---------------

Cooperative vectors are compatible with automatic differentiation. Simply pack
variables with tracked gradients into cooperative vectors---the system will
then propagate derivatives through cooperative operations. Here is an example:

.. code-block:: python

   a, b = Float16(1), Float16(2)
   dr.enable_grad(a, b)

   vec = nn.CoopVector(a, b) # Pack grad-enabled variables

   # ... Computation involving 'vec' ...

   x, y = vec                    # Unpack
   loss = x**2 + y**2            # Continue calculation and ..
   dr.backward_from(loss)        # .. eventually backpropagate

The layout conversion functions :py:func:`nn.pack() <drjit.nn.pack()>`
and :py:func:`nn.unpack() <drjit.nn.unpack()>` are *not differentiable*,
which is intentional. To train a neural network, convert the initial
coefficient values into training-optimal layout and optimize this
representation directly. Doing so is more efficient than changing layouts twice
in every optimization step (once for the weights and once for their
derivatives).

The following AD operations recognize :py:func:`nn.CoopVector
<drjit.nn.CoopVector>` and :py:func:`nn.MatrixView <drjit.nn.MatrixView>` objects:

- :py:func:`grad_enabled`, :py:func:`enable_grad`, :py:func:`disable_grad`.
- :py:func:`detach`, :py:func:`grad` (TODO not yet for CoopVec..)

Performance considerations
--------------------------

- **CUDA/OptiX** backend:

  - :py:func:`nn.matvec() <drjit.nn.matvec>` currently requires 16-bit
    floating point arguments. FP8 formats may be added in the future.

  - Tensor cores work with 8x8 and 16x16 blocks. Matrices, whose row or column
    counts are not a multiples of 8 or 16 will be zero-padded internally. There
    is no performance benefit in working with such intermediate sizes.

- **LLVM** backend:

  - There is no difference between row-major and training/inference-optimal
    layouts on the CPU. However, using :py:func:`nn.pack()
    <drjit.nn.pack>` is still recommended , since packing multiple arrays
    into a shared buffer has a small performance benefit.

  - On Intel-compatible processors, using half precision cooperative vectors is
    not recommended. FP16 matrix multiplication requires ``AVX512FP16``, an
    extension not yet available on consumer CPUs as of 2025. Without this
    extension, FP16 computation involves many costly FP16 ↔ FP32 roundtrips.
