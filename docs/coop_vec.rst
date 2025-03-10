.. py:currentmodule:: drjit

.. cpp:namespace:: drjit

.. _coop_vec:

Cooperative vectors
===================

Cooperative vectors provide an efficient way to evaluate matrix-vector products
in programs. They are designed for scenarios, where the underlying matrices are
relatively small (e.g., 64x64 or smaller), and where each execution thread
provides its own input vector. By working together, threads can perform these
multiplications more efficiently, which is why this approach is called
*cooperative vectors*.

This feature is especially useful for evaluating small `multilayer perceptrons
<https://en.wikipedia.org/wiki/Multilayer_perceptron>`__ (MLPs) within larger
programs while fully *fusing* all steps of the process into a single kernel.
However, cooperative vectors are not exclusive to MLPs and can be applied to
other use cases as well.

- On **NVIDIA GPUs (Turing or newer)**, cooperative vectors map to the OptiX
  `cooperative vector API
  <https://raytracing-docs.nvidia.com/optix9/guide/index.html#cooperative_vectors#neural-rendering-with-cooperative-vectors>`__,
  leveraging built-in `tensor cores
  <https://www.nvidia.com/en-us/data-center/tensor-cores/>`__ for acceleration.

- On the **CPU (LLVM) backend**, Dr.Jit compiles cooperative vector operations
  using available instruction set extensions (AVX512, NEON, etc.).

Motivation
----------

The cooperative vector API available via the :py:mod:`drjit.coop` submodule.
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
   A, b = dr.coop.pack(A, b)

   # Define MLP input values
   x, y, z = ... # (type: Float16)

   # Pack inputs into a cooperative vector
   v = dr.coop.Vector(x, y, z)

   # Forward pass through the MLP
   for i in range(len(shapes)):
       v = dr.matvec(A[i], v, b[i])

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
  :py:func:`dr.coop.pack() <drjit.coop.pack>`.

- Packing input values into a :py:func:`drjit.coop.Vector`.

- Performing matrix-vector multiplications and other arithmetic, while keeping
  the state in cooperative vector form.

- Unpacking the final result.

Cooperative vectors
-------------------

The central type of this API is the *cooperative vector* class
:py:class:`drjit.coop.Vector`. This is a dynamically sized vector with
uniformly typed elements.

Unlike regular Dr.Jit arrays (e.g. :py:class:`drjit.cuda.ArrayXf`), cooperative
vectors *do not allow indexed element access*. For example, the following
operation raises an exception:

.. code-block:: pycon

   >>> from drjit.cuda import Float16
   >>> x, y = Float16(1), Float16(2)
   >>> vec = drjit.coop.Vector(x, y)
   >>> vec[1]
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   TypeError: 'drjit.coop.Vector' object is not subscriptable

This restriction exists because the compiler may arbitrarily distribute
cooperative vector components across threads for efficiency. Allowing direct
indexing would interfere with this optimization.

Packing
^^^^^^^

The :py:class:`drjit.coop.Vector` constructor accepts a mix of Dr.Jit arrays
and Python scalars and packs them into a cooperative vector:

.. code-block:: python

   x, z = Float16(1), Float16(2)
   vec = drjit.coop.Vector(x, 0.0, z) # 3D vector

Nested arrays are permitted as well and will be automatically flattened:

.. code-block:: python

   p, uv = Array3f(...), Array2f(...)
   vec = drjit.coop.Vector(p, uv) # 5D vector

Unpacking
^^^^^^^^^

To unpack a cooperative into its components, use the following standard Python
syntax:

.. code-block:: python

   x, y, z = vec # Unpack a 3D vector
   x, y, *extra = vec # Unpack first 2 components, put rest into 'extra'

Alternatively, cooperative vectors can be converted into lists or tensors:

.. code-block:: python

   vec_l = list(vec)
   vec_t = TensorXf(vec)

Arithmetic
^^^^^^^^^^

Cooperative vectors support a restricted set of arithmetic operations:

- Elementary arithmetic operations: ``+``, ``-``, ``*`` (but no division)
- :py:func:`dr.fma() <fma>`,
- :py:func:`dr.minimum() <minimum>`, :py:func:`dr.maximum() <maximum>`,
- :py:func:`dr.log2() <log2>`, :py:func:`dr.exp2() <exp2>`,
- :py:func:`dr.tanh() <tanh>`,
- :py:func:`dr.relu() <relu>`, :py:func:`dr.step() <relu>`.
- :py:func:`dr.coop.matvec() <drjit.coop.matvec>`

These operations directly map to hardware-optimized operations on CUDA/OptiX.
Operations outside of this set can be realized via unpacking/repacking, e.g.:

.. code-block::

   x : dr.coop.Vector = ...
   y = dr.coop.Vector(dr.sin(v) for v in x)

However, this may degrade performance. It is best to keep cooperative vectors
in their opaque layout whenever possible.

Arithmetic operations may mix cooperative vectors and regular Dr.Jit arrays or
Python scalars, which will undergo implicit broadcasting.

.. code-block::

   x: dr.coop.Vector[dr.cuda.Float16] = ...
   y: dr.cuda.Float16 = ...
   z = dr.maximum(x, 0) + y

.. _matrix_views:

Matrix views
------------

For optimal performance, input matrices and biases should be packed into a
shared buffer in an implementation-defined optimal layout. Dr.Jit provides
:py:class:`drjit.coop.View` for this purpose---a structure that holds a pointer
to a buffer along with layout and type metadata.

Packing
^^^^^^^

Use :py:func:`dr.coop.pack() <drjit.coop.pack>` to optimize memory layout for
inference (which is the default) or training:

.. code-block:: python

   A: TensorXf = ...
   b: Float = ...
   A_view, b_view = dr.coop.pack(A, b, layout='inference')

The function returns views into the resulting shared buffer.
It also supports :ref:`PyTrees <pytrees>`:

.. code-block:: python

   data_dict = { 'A_1': A, 'b_1': b }
   view_dict = dr.coop.pack(data_dict, layout='training')

The training-optimal layout should be used used if the program *backpropagates*
(as in :py:func:`dr.backward*() <drjit.backward>`) gradients through
matrix-vector products. Forward derivative propagation (as in
:py:func:`dr.forward*() <drjit.forward>`) does not require a training-optimal
layout.

If the input matrices are already packed in a row-major layout, call
:py:func:`dr.coop.view() <drjit.coop.view>` to create an efficient reference
and then pass slices of the view to :py:func:`dr.coop.pack()
<drjit.coop.pack>`. This avoids additional copies.

.. code-block::

   mat: TensorXf = ...
   mat_view = dr.coop.view(mat)

   A1_view, A2_view = dr.coop.pack(
       mat_view[0:32, :],
       mat_view[32:64, :]
   )

Unpacking
^^^^^^^^^

The function :py:func:`dr.coop.unpack() <drjit.coop.unpack>` transforms a
sequence (or :ref:`PyTree <pytrees>`) of vectors and optimal-layout matrices
back into row-major layout.

.. code-block:: python

   A_out, b_out = dr.coop.unpack(A_opt, b_opt)

Note that the output of this function are (row-major) *views* into a shared
buffer. These views can be converted back into regular tensors:

.. code-block:: python

   A = TensorXf16(A)

Matrix-vector products
----------------------

The main use purpose of cooperative vectors is matrix-vector multiplication.
The function :py:func:`dr.coop.matvec() <drjit.coop.matvec>` performs this
operation:

.. code-block:: python

   y = dr.coop.matvec(A, x, b) # Compute y = A @ x + b

Here,

- ``A`` and ``b`` are *views* (:py:class:`dr.coop.View`) created by
  :py:func:`dr.coop.pack() <drjit.coop.pack>` or :py:func:`dr.coop.view()
  <drjit.coop.view>`.
- ``x`` and ``y`` are cooperative vectors.
- the ``b`` term is optional.

The function also accepts an optional ``transpose=True`` parameter to compute
:math:`A^Tx + b`.

Differentiation
---------------

Cooperative vectors are compatible with automatic differentiation. Simply pack
variables with tracked gradients into cooperative vectors---the system will
then track subsequent operations. Here is an example:

.. code-block:: python

   a, b = Float16(1), Float16(2)
   dr.enable_grad(a, b)

   vec = drjit.coop.Vector(a, b) # Pack grad-enabled variables

   # ... computation involving 'vec' ...

   x, y = vec                    # Unpack
   loss = x**2 + y**2            # Continue calculation and ..
   dr.backward_from(loss)        # .. eventually backpropagate


Performance considerations
--------------------------

On CUDA/OptiX, :py:func:`dr.coop.matvec() <drjit.coop.matvec>` currently requires 16-bit
floating point arguments. Support for 8-bit formats may be added in the future.

On LLVM, 32-bit floating point is preferable, as FP16 matrix multiplication
requires AVX512FP16--an extension not yet available on consumer CPUs (as of
2025). Without this extension, using FP16 may cause costly FP32 conversions.
