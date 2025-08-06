.. py:currentmodule:: drjit

.. cpp:namespace:: drjit

.. _coop_vec:

Cooperative vectors
===================

*Cooperative vectors* are a `new API
<https://github.com/KhronosGroup/GLSL/blob/main/extensions/nv/GLSL_NV_cooperative_vector.txt>`__
for evaluating matrix-vector products in certain types of GPU workloads. They
are designed to handle cases, where each thread of a parallel program needs
to multiply a vector by a reasonably small matrix (e.g., 128x,128, 64x64 or fewer
entries). By working together, the threads can perform these multiplications
more efficiently, which is why the approach is called *cooperative*.

Cooperative vectors are especially useful for evaluating small `multilayer
perceptrons <https://en.wikipedia.org/wiki/Multilayer_perceptron>`__ (MLPs)
within larger programs while fully *fusing* all steps of the process into a
single kernel. Other workloads that heavily rely on matrix-vector products may
benefit as well.

Dr.Jit supports cooperative vectors on both of its backends:

- On **NVIDIA GPUs (Turing or newer)**, cooperative vectors map to the OptiX
  `cooperative vector API
  <https://raytracing-docs.nvidia.com/optix9/guide/index.html#cooperative_vectors#neural-rendering-with-cooperative-vectors>`__,
  leveraging built-in `tensor cores
  <https://www.nvidia.com/en-us/data-center/tensor-cores/>`__ for acceleration.
  Driver version R570 or newer is required to use this feature.

- On the **CPU (LLVM) backend**, compilation of cooperative vector operations
  targets the available instruction set extensions (AVX512, NEON, etc.).

Code snippets in the remainder of this section assume the following include
directives:

.. code-block:: python

   import drjit as dr
   import drjit.nn as nn
   from drjit.auto.ad import Float16, TensorXf16

Motivation
----------

The cooperative vector API is available via the :py:mod:`drjit.nn` submodule.
Below is an example demonstrating how to use it to perform a matrix
multiplication.

.. code-block:: python

   # Matrix shape
   m, n = 3, 16

   # Create a random matrix + offset
   rng = dr.rng(seed=0)
   A = rng.normal(TensorXf, (m, n))
   b = rng.random(TensorXf, m)

   # Pack 'A' and 'b' into a buffer with an optimal layout
   buffer, A_view, b_view = nn.pack(A, b)

   # Create a cooperative vector
   x = nn.CoopVec(... 16 values ...)

   # Evaluate A @ x + b
   v_out = nn.matvec(A_view, v_in, b_view)

   # Unpack the resulting cooperative vector
   x, y, z = v_out

This involves the following steps:

- Initializing matrix data and packing it into an optimized memory layout using
  :py:func:`nn.pack() <drjit.nn.pack>`.

- Constructing a :py:class:`nn.CoopVec` containing the inputs to the matrix
  multiplication.inputs.

- Performing one or more matrix-vector multiplications and other arithmetic,
  while keeping the state in cooperative vector form.

- Unpacking the final cooperative vector into regular Dr.Jit arrays.

Cooperative vectors
-------------------

The central type of this API is the *cooperative vector* class
:py:class:`nn.CoopVec`. This is a dynamically sized vector with uniformly
typed elements.

Unlike regular Dr.Jit arrays (e.g. :py:class:`drjit.cuda.ArrayXf`), cooperative
vectors *do not allow indexed element access*. For example, the following
operation raises an exception:

.. code-block:: pycon

   >>> vec = nn.CoopVec(Float16(1), Float16(2))
   >>> vec[1]
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: 'drjit.nn.CoopVec' object is not subscriptable

This restriction exists because the compiler may arbitrarily distribute
cooperative vector components across threads for efficiency. Allowing direct
indexing would interfere with this optimization.

The :py:class:`drjit.nn.CoopVec` constructor accepts an arbitrary sequence
of :ref:`PyTrees <pytrees>` containing Dr.Jit array and Python scalars and
flattens them into a cooperative vector:

.. code-block:: python

   vec = nn.CoopVec( # Construct a 4D vector
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

   vec_3 = nn.CoopVec(*vec_1, *vec_2)

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
- :py:func:`dr.step() <step>`.
- :py:func:`nn.matvec() <drjit.nn.matvec>`

These operations directly map to hardware-optimized operations on CUDA/OptiX.
Operations outside of this set can be realized via unpacking/repacking, e.g.:

.. code-block::

   x : nn.CoopVec = ...
   y = nn.CoopVec(dr.sin(v) for v in x)

However, this may degrade performance. It is best to keep cooperative vectors
in their opaque layout whenever possible.

Arithmetic operations may mix cooperative vectors and regular Dr.Jit arrays or
Python scalars, which will undergo implicit broadcasting.

.. code-block::

   x: nn.CoopVec[dr.cuda.Float16] = ...
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
memory layouts for either *inference* (the default) or *training*. You must
specify ``layout='training'`` if you wish to differentiate matrix
multiplication in reverse mode.

Following this step, ``A`` and ``b`` have been merged into ``buffer``, and
``A_view`` and ``b_view`` encode the offset and layout within this larger
buffer. Matrix views *cannot* be used in arithmetic expressions and are best
thought of as opaque handles. They only exist to describe the input of the
matrix-vector multiplication operation explained next.

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

Cooperative vectors support automatic differentiation. Simply pack variables
with tracked gradients into cooperative vectors---the system will then
propagate derivatives through subsequent operations. Here is an example:

.. code-block:: python

   # Differentiable input
   a = Array2f16(..)
   dr.enable_grad(a)

   # Differentiable matrix + bias vector
   buffer, A_view, b_view = nn.pack(A, b)
   dr.enable_grad(buffer)

   # Pack grad-enabled variables into a cooperative vector
   x = nn.CoopVec(a)

   # Differentiable matrix-vector multiplication
   y = dr.matvec(A_view, x, b_view)

   r0, r1 = y                    # Unpack
   loss = r0**2 + r1**2          # Continue calculation and ..
   dr.backward_from(loss)        # .. eventually backpropagate

Specific views or cooperative vectors can also be detached via
:py:func:`drjit.detach()` to inhibit gradient propagation, e.g.:

.. code-block:: python

   y = nn.matvec(A_view, dr.detach(x), dr.detach(b_view))

Note that the conversion functions :py:func:`nn.pack() <drjit.nn.pack()>` and
:py:func:`nn.unpack() <drjit.nn.unpack()>` are *not differentiable*. This is
intentional: to train a neural network, convert the initial coefficient values
into training-optimal layout and optimize this representation directly. Doing
so is more efficient than changing layouts twice in every optimization step
(once for the weights and once for their derivatives).

The following AD operations recognize :py:func:`nn.CoopVec
<drjit.nn.CoopVec>` and :py:func:`nn.MatrixView <drjit.nn.MatrixView>` objects:

- :py:func:`grad_enabled`, :py:func:`enable_grad`, :py:func:`disable_grad`.
- :py:func:`detach`.

Performance considerations
--------------------------

- **CUDA/OptiX** backend:

  - When calling :py:func:`nn.matvec() <drjit.nn.matvec>`, expect significantly
    reduced performance when only a subset of threads participate in the
    operation. When neural networks are evaluated in loops or conditional
    expressions, it may be advisable to incorporate reordering (via
    :py:func:`dr.reorder_threads() <drjit.reorder_threads>`) to obtain coherent groups of threads.

  - :py:func:`nn.matvec() <drjit.nn.matvec>` currently requires 16-bit
    floating point arguments. FP8 formats may be added in the future.

  - Tensor cores work with 8x8 and 16x16 blocks. Matrices, whose row or column
    counts are not a multiples of 8 or 16 will be zero-padded internally. There
    is no performance benefit in working with such intermediate sizes.

    Unpacking cooperative vectors may degrade performance. It is best to keep
    them in their opaque layout whenever possible.

- **LLVM** backend:

  - The LLVM code path is mainly provided as an alternative implementation
    for testing. The cooperative vector computation model is unfortuantely not
    very efficient on x86_64 CPUs due to the limited number of available
    registers.

  - There is no difference between row-major and training/inference-optimal
    layouts on the CPU. However, using :py:func:`nn.pack()
    <drjit.nn.pack>` is still recommended, since packing multiple arrays
    into a shared buffer has a small performance benefit.

  - On Intel-compatible processors, using half precision cooperative vectors is
    not recommended. FP16 matrix multiplication requires ``AVX512FP16``, an
    extension not yet available on consumer CPUs as of 2025. Without this
    extension, FP16 computation involves many costly FP16 ↔ FP32 roundtrips.
