.. py:currentmodule:: drjit

.. _special_arrays:

Array types
===========

Dr.Jit exposes a *large* (~500) variety of different type bindings, which include

- :ref:`Flat arrays <flat_arrays>` (e.g., :py:class:`drjit.cuda.Float`),
- :ref:`Nested arrays <nested_arrays>` (e.g., :py:class:`drjit.cuda.Array4f`),
- :ref:`Matrices <matrices>` (e.g., :py:class:`drjit.cuda.Matrix4f`),
- :ref:`Complex numbers <complex_numbers>` (e.g., :py:class:`drjit.cuda.Complex2f`), and
- :ref:`Quaternions <quaternions>` (e.g., :py:class:`drjit.cuda.Quaternion4f`).
- :ref:`Tensors <tensors>` (e.g., :py:class:`drjit.cuda.TensorXf`).

Each flavor exists for a variety of different dimensions, backends, and
numerical representations. Every type also has a corresponding C++
analogue, which enables tracing and automatic differentiation of large
codebases involving a mixture of C++ and Python code.

The remainder of this section reviews commonalities and differences between
the various available array types.

.. _backends:

Backends
--------

Dr.Jit types are organized into five different *backend*-specific Python
packages named

- :py:mod:`drjit.scalar`,
- :py:mod:`drjit.cuda`,
- :py:mod:`drjit.cuda.ad`,
- :py:mod:`drjit.llvm`, and
- :py:mod:`drjit.llvm.ad`.

(Additional backends are likely to be added in the future.)

Additionally, there is an *automatic backend* that simply redirects to
one of the above depending on the backends detected at runtime.

- :py:mod:`drjit.auto`, and
- :py:mod:`drjit.auto.ad`.

Any given array type (e.g. ``Array3f``) actually exists in *all five* of these
packages (e.g., ``drjit.scalar.Array3f``, ``drjit.llvm.ad.Array3f``). However,
there are notable differences between them:

- **Scalar backend**: types contained within ``drjit.scalar.*`` represent a
  single element of the underlying concept. For example,
  :py:class:`drjit.scalar.Complex2f` stores the real and imaginary part of a
  single complex value in an array of shape shape ``(2,)``.

- **Vectorized backends**: types within all of the other packages are
  *vectorized*, i.e., they represent arbitrarily many elements. For example,
  :py:class:`drjit.cuda.Complex2f` stores a dynamically sized sequence of
  complex values in an array of shape shape ``(2, N)``. Program execution along
  the dynamic dimension runs in parallel, which is important for efficiency.

  Dr.Jit uses an approach denoted as *tracing* to execute programs
  involving these vectorized types. Every operation conceptually appends
  instructions to a progressively growing computational *kernel*. Variable
  evaluation eventually compiles and executes this kernel on a target device.

  The LLVM backend uses the `LLVM Compiler Infrastructure
  <https://llvm.org>`__ to compile kernels targeting the CPU. It vectorizes
  the program to use available instruction set extensions such as `Intel AVX512
  <https://en.wikipedia.org/wiki/AVX-512>`__ or `ARM NEON
  <https://developer.arm.com/Architectures/Neon>`__ and parallelizes their
  execution across multiple cores.

  The CUDA backend launches parallel kernels on NVIDIA GPUs, which involves
  `CUDA <https://developer.nvidia.com/cuda-toolkit>`__ and the `PTX
  intermediate representation
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`__. This
  backend only depends on users having having a suitable graphics card and
  driver (notably, users do not need to install the CUDA SDK.)

- **Automatic differentiation**: types within packages having the ``.ad`` suffix
  additionally track differentiable operations to enable subsequent forward- or
  reverse-mode differentiation. They are only needed when the computation
  actually uses such derivatives.

Programs can mix and match types from these different backends. In particular,
it is normal for a program to simultaneously use the ``drjit.scalar.*`` package
(for *uniform* values) along with types from a vectorized backend.

.. _flat_arrays:

Flat arrays
-----------

Dr.Jit programs are ultimately composed of operations involving *flat arrays*.
In vectorized backends, these are dynamically sized 1D arrays. In the scalar
backend, they are aliases of native Python types (``bool``, ``int``,
``float``).

The following kinds of flat arrays are available

.. list-table::
   :header-rows: 1

   * - Type name
     - Interpretation
   * - ``Bool``
     - Boolean-valued array
   * - ``Int`` (or ``Int32``)
     - 32-bit signed integer array
   * - ``UInt`` (or ``UInt32``)
     - 32-bit unsigned integer array
   * - ``Int64``
     - 64-bit signed integer array
   * - ``UInt64``
     - 64-bit unsigned integer array
   * - ``Float16``
     - Half precision array
   * - ``Float`` (or ``Float32``)
     - Single precision array
   * - ``Float64``
     - Double precision array

The register file of GPUs is 32 bit-valued, which motivates this naming convention.

The following example constructs an 1D array with 3 elements, prints its
contents, and then performs a simple computation.

.. code-block:: pycon

   >>> import drjit as dr
   >>> from drjit.llvm import Float
   >>> x = Float(1, .5, .25)
   >>> print(x)
   [1, 0.5, 0.25]
   >>> y = dr.sqrt(1-x**2)
   >>> print(y)
   [0, 0.866025, 0.968246]

The last statement compiles a kernel that implements the expression
:math:`\sqrt{1-x^2}` using both SIMD-style and multi-core parallelism.
Conceptually, this corresponds to the following C code:

.. code-block:: cpp

   // Loop parallelized using SIMD + multicore parallelism
   for (size_t i = 0; i < N; ++i) {
       float v0 = x[i];
       float v1 = v0*v0;
       float v2 = sqrtf(v1);
       y[i] = v2;
   }

The kernel is *fused*, which means that temporaries like ``v1`` and ``v2`` are
kept in registers instead of being written to CPU/GPU memory. Naturally, such
optimizations aren't needed when the input only consists of three elements, but
they can greatly accelerate more costly workloads. Dr.Jit caches this kernel
and reuses it when it detects the same computational pattern at a later point.

A flat array of size 1 will implicitly broadcast to any other size. Other sizes
are incompatible and will raise an error.

.. code-block:: pycon

   >>> Float(1, 2, 3) == Float(2)
   [False, True, False]

   >>> Float(1, 2, 3) == Float(2, 3)
   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   RuntimeError: drjit.llvm.Float.__richcmp__(): jit_var_lt(r1, r2): operands ↵
   have incompatible sizes! (sizes: 3, 2)

Everything in Dr.Jit ultimately boils down to operations on flat arrays. The
various specialized types discussed in the remainder of this section are merely
containers that *wrap* one or more flat arrays, e.g., to endow them with
different semantics. A single operation on such a container then triggers a
sequence of flat array operations.

For example, the following snippet computes the angle between two :ref:`nested
arrays <nested_arrays>` representing 3D vectors. The call to
:py:func:`drjit.set_log_level` reveals the underlying tracing process, in which
each line corresponds to the creation of an internal flat array.

.. code-block:: pycon

   >>> a = dr.llvm.Array3f(...)
   >>> b = dr.llvm.Array3f(...)
   >>> dr.set_log_level(dr.LogLevel.Debug)
   >>> angle = dr.acos(a @ b)
   jit_var_new(): float32 r7 = mul(r1, r4)
   jit_var_new(): float32 r8 = fma(r2, r5, r7)
   jit_var_new(): float32 r9 = fma(r3, r6, r8)
   jit_var_new(): float32 r10 = abs(r9)
   jit_var_new(): float32 r11 = mul(r9, r9)
   jit_var_new(): float32 r12 = 0.5
   jit_var_new(): bool r13 = gt(r10, r12)
   ...

.. _nested_arrays:

Nested arrays
-------------

Types like :py:class:`drjit.scalar.Array2f` or
:py:class:`drjit.cuda.ad.Array4f` implement *nested* arrays, which are arrays
of flat arrays.

They typically represent N-dimensional quantities like 3D positions or
velocities. Dr.Jit provides these from 0 to 4 dimensions, along with
generically sized variants denoted by a capital ``X``
(:py:class:`drjit.scalar.ArrayXf`, :py:class:`drjit.cuda.ad.ArrayXf`, etc.).
The entries of statically sized versions can be accessed via the ``.x``,
``.y``, ``.z``, and ``.w`` properties. An example use is shown below:

.. code-block:: python

   def norm_2d(v: drjit.cuda.Array2f):
       return dr.sqrt(v.x**2 + v.y**2)

Nested arrays match the standard broadcasting behavior of other
array programming frameworks:

.. code-block:: pycon

   >>> dr.scalar.Array3f(1)
   [1, 1, 1]

   >>> dr.scalar.Array3f(1, 2, 3) + 1
   [2, 3, 4]

The naming convention of nested arrays (and other types discussed in the
remainder of this section) is based on a suffix characterizing the number of
dimensions, numeric type, and the number of bits. For example, the following
flavors of 2D arrays are available:

.. list-table::
   :header-rows: 1

   * - Type name
     - Interpretation
   * - ``Array2b``
     - Boolean-valued 2D array
   * - ``Array2i``
     - 32-bit signed integer 2D array
   * - ``Array2u``
     - 32-bit unsigned integer 2D array
   * - ``Array2i64``
     - 64-bit signed integer 2D array
   * - ``Array2u64``
     - 64-bit unsigned integer 2D array
   * - ``Array2f16``
     - Half precision 2D array
   * - ``Array2f``
     - Single precision 2D array
   * - ``Array2f64``
     - Double precision 2D array

It is legal build nested arrays from flat arrays of different sizes. Usually,
some of the elements will have size ``1``, which means that they can broadcast
to any other size as needed. Operations like ``print()`` already perform this
broadcasting step internally:

.. code-block:: pycon

   >>> vec = dr.llvm.Array2f()
   >>> vec.x = [1, 2, 3]
   >>> vec.y = 10
   >>> print(vec) # <-- array of three 2D vectors, whose 'y' component is identical
   [[1, 10],
    [2, 10],
    [3, 10]]

Other combinations make less sense and will cause errors:

.. code-block:: pycon

   >>> vec = dr.llvm.Array2f()
   >>> vec.x = [1, 2, 3]
   >>> vec.y = [1, 2]
   >>> print(vec)
   [ragged array]

   >>> drjit.sum(x)
   RuntimeError: drjit.llvm.Float.__add__(): jit_var_add(r1, r2): operands have incompatible sizes! (sizes: 2, 3)

   The above exception was the direct cause of the following exception:

   Traceback (most recent call last):
     File "<stdin>", line 1, in <module>
   RuntimeError: drjit.sum(<drjit.llvm.Array2f>): failed (see above)!

.. _matrices:

Matrices
--------

Matrix types like :py:class:`drjit.scalar.Matrix2f` or
:py:class:`drjit.cuda.ad.Matrix4f` represent square matrices stored in
row-major format, typically encoding linear transformations that can be applied
to :ref:`nested arrays <nested_arrays>`.

Matrices change the behavior of various operations:

- **Broadcasting**: The implicit or explicit construction of a matrix from a
  scalar broadcasts to the identity element:

  .. code-block:: pycon

     >>> dr.scalar.Matrix2f(1, 2, 3, 4) + 10
     [[10, 2],
      [3, 14]]

- The multiplication operator ``*`` coincides with the matrix multiplication
  operator ``@`` (see :py:func:`drjit.matmul()` for details). Depending on
  the nature of the arguments, this operation carries out a
  matrix-matrix, matrix-vector, vector-matrix, or scalar product.

- True division (``arg0 / arg1``) with a matrix-valued denominator ``arg1``
  involves a matrix inverse.

- Additionally, the following operations generalize by internally replace
  ordinary multiplication and division operations with their matrix analogs:

  - :py:func:`drjit.fma`
  - :py:func:`drjit.rcp`

- The following operations Reciprocation via :py:func:`drjit.rcp()` returns the matrix inverse.


To give an example, if ``a``, ``b``, ``c`` below were all matrices, then the
expression below would right-multiply ``b`` by the inverse of ``c``,
left-multiply by ``a``, and finally add the identity matrix.

.. code-block:: python

   a * b / c + 1

If you prefer to work with matrix-*shaped* types while preserving standard
array semantics during arithmetic and broadcasting operations, you can use
nested :ref:`nested arrays <nested_arrays>` such as
:py:class:`drjit.cuda.Array44f`, which has the same shape as
:py:class:`drjit.cuda.Matrix4f`. The type trait :py:func:`drjit.array_t`
returns the "plain array" form associated with any given Dr.Jit type, including
matrices.

Dr.Jit does not provide bindings for non-square matrices or matrices larger
than ``4x4``. While additional bindings can easily be added, doing so for large
matrices is inadvisable: everything is ultimately unrolled into flat array
operations, hence multiplying two ``1000x1000`` matrices would, e.g., produce
an unusably large kernel with ~1'000'000'000 instructions.

.. _complex_numbers:

Complex numbers
---------------

Types like :py:class:`drjit.scalar.Complex2f` or
:py:class:`drjit.cuda.ad.Complex2f64` represent complex-valued scalars and
arrays. The use of these types changes the behavior of various standard
operations:

- **Broadcasting**: The implicit or explicit construction of a complex type
  from a non-complex scalar broadcasts to the identity element:

  .. code-block:: pycon

     >>> dr.scalar.Complex2f(1 + 2j) + 3
     4+2j

- The multiplication operator ``*`` performs a complex product.

- True division (``arg0 / arg1``) with a complex-valued denominator ``arg1``
  involves a complex inverse.

- Additionally, many builtin mathematical operations implement generalizations
  that correctly handle complex-valued inputs. These currently include:

  - :py:func:`drjit.fma`
  - :py:func:`drjit.rcp`
  - :py:func:`drjit.abs`
  - :py:func:`drjit.sqrt`
  - :py:func:`drjit.rsqrt`
  - :py:func:`drjit.log2`
  - :py:func:`drjit.log`
  - :py:func:`drjit.exp2`
  - :py:func:`drjit.exp`
  - :py:func:`drjit.power`
  - :py:func:`drjit.sin`
  - :py:func:`drjit.cos`
  - :py:func:`drjit.sincos`
  - :py:func:`drjit.tan`
  - :py:func:`drjit.asin`
  - :py:func:`drjit.acos`
  - :py:func:`drjit.atan`
  - :py:func:`drjit.sinh`
  - :py:func:`drjit.cosh`
  - :py:func:`drjit.sincosh`
  - :py:func:`drjit.tanh`
  - :py:func:`drjit.asinh`
  - :py:func:`drjit.acosh`
  - :py:func:`drjit.atanh`

  Complex implementations of other transcendental functions such as the error
  function and its inverse have not been added (yet). Their behavior is
  considered undefined. External contributions to add them are welcomed.

.. _quaternions:

Quaternions
-----------

Types like :py:class:`drjit.scalar.Quaternion4f` or
:py:class:`drjit.cuda.ad.Quaternion4f64` represent quaternion-valued scalars
and arrays. The use of these types changes the behavior of various standard
operations:

- **Broadcasting**: The implicit or explicit construction of a quaternions
  from non-quaternionic values or arrays broadcasts to the identity element:

  .. code-block:: pycon

     >>> dr.scalar.Quaternion4f(1, 2, 3, 4) + 10
     1i+2j+3k+14

- The multiplication operator ``*`` performs a quaternion product.

- True division (``arg0 / arg1``) with a quaternion-valued denominator ``arg1``
  involves a quaternion inverse.

- Additionally, a few mathematical operations implement generalizations that
  correctly handle quaternion-valued inputs. These currently include:

  - :py:func:`drjit.fma`
  - :py:func:`drjit.rcp`
  - :py:func:`drjit.abs`
  - :py:func:`drjit.sqrt`
  - :py:func:`drjit.rsqrt`
  - :py:func:`drjit.log2`
  - :py:func:`drjit.log`
  - :py:func:`drjit.exp2`
  - :py:func:`drjit.exp`
  - :py:func:`drjit.power`

  Quaternionic implementations of other transcendental functions such as
  ordinary and hyperbolic trigonometric functions have not been added (yet).
  Their behavior is considered undefined. External contributions to add them
  are welcomed.

.. _tensors:

Tensors
-------

Dr.Jit also includes a general n-dimensional array type colloquially referred
to as a `tensor <https://en.wikipedia.org/wiki/Tensor>`__. The tensor types all
have a capital ``X`` in their name to denote their dynamic shape (e.g.,
:py:class:`drjit.cuda.TensorXf16`).

A tensor is internally represented by a :ref:`flat array <flat_arrays>` and a
shape tuple. It can be constructed manually, or using various other array
creation operations.

.. code-block:: pycon

   >>> from drjit.llvm import TensorXf
   >>> drjit.zeros(TensorXf, shape=(1, 2, 3, 4))
   [[[[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]],
     [[0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0]]]]
   >>> t = TensorXf([1,2,3,4,5,6], shape=(3, 2))
   >>> print(t)
   [[1, 2],
    [3, 4],
    [5, 6]]

The shape and flat array underlying a tensor can be accessed using its
:py:attr:`.shape <drjit.ArrayBase.shape>` and :py:attr:`.array <drjit.ArrayBase.array>` members.

.. code-block:: pycon

   >>> t.shape
   (3, 2)
   >>> t.array
   [1, 2, 3, 4, 5, 6]

Tensors directly convert to other Dr.Jit types, and vice versa. A potential
surprise here is that this changes the output of operations like
``print``, :py:func:`drjit.print`, :py:func:`drjit.format`, and
:py:func:`drjit.ArrayBase.__repr__`:

.. code-block:: pycon

   >>> a = Array3f(t)
   >>> t = TensorXf(a)
   >>> a
   [[1, 3, 5],
    [2, 4, 6]]
   >>> t
   [[1, 2],
    [3, 4],
    [5, 6]]
   >>> a.shape
   (3, 2)
   >>> t.shape
   (3, 2)

.. _nested_array_transpose:

This is intentional and merely cosmetic: the string conversion of non-tensor
arrays actually prints the *transpose*, which rearranges the data so that all
information associated with one thread of the parallel program is shown next to
each other (e.g. to display a complete 3D vector on each line in the above
example). In contrast, the string conversion of tensors matches that of other
array programming libraries and does not transpose the input.

Tensors support all normal mathematical operations along with automatic
differentiation. They share the broadcasting behavior known from other array
programming frameworks.

.. code-block:: pycon

   >>> t = drjit.pi - drjit.atan2(TensorXf([1],   shape=(1,1)),
   ...                            TensorXf([1,2], shape=(1,2)))
   >>> t.shape
   (1, 2)
   >>> t
   [[2.35619, 2.67795]]

Tensors support the full spectrum of slicing operations: slicing using fixed indices,
ranges, integer arrays, ellipsis (``...``), and adding new axes by
indexing with :py:attr:`drjit.newaxis` (or equivalently, ``None``).

.. code-block:: pycon

   >>> t = ...
   >>> t.shape
   (10, 20, 30, 40)
   >>> t2 = t[UInt32(5,6), 10:20:4, drjit.newaxis, 1, ...]
   >>> t2.shape
   (2, 3, 1, 40)

Slicing internally turns into a :py:func:`drjit.gather` operation that reads
from the underlying flat array, while slice assignment turns into
:py:func:`drjit.scatter`. The conversion from a slice tuple into concrete
indices is performed by the function :py:func:`drjit.slice_index` that can also be
used directly.

.. _tensor_limitations:

Limitations
^^^^^^^^^^^

It should be noted that Dr.Jit is *not* a general array/tensor programming
library. It currently lacks many standard operations found in frameworks like
PyTorch or TensorFlow. This includes

- Operations to split or concatenate tensors and rearrange their axes in various ways.
- General matrix/tensor product operations, convolutions, FFT, Einstein sums, etc.

While we intend to make the interface more feature-complete in the future
(external help is welcomed!), tensors are best used sparingly in actual
programs.

The reason for this is that tensor-based programs tend to make frequent use of
slicing operations. For example, the following snippet adds even and
odd-numbered entries of a 1D tensor and would not feel out of place in a
typical NumPy/PyTorch/TensorFlow program.

.. code-block:: pycon

   >>> t = t[0::2] + t[1::2]

In a Dr.Jit program, the entries of this tensor would be computed by different
threads of a parallel program. Correct sequencing of the operation then
generally requires a *barrier* realized by an intermediate variable evaluation,
which prevents the compilation of a fully fused kernel. In other words, the use
of tensors can interfere with one of Dr.Jit's key optimizations, which is its
ability to aggressively fuse operations into large kernels.

We recommend the use of tensors mainly as storage representation of shaped data
(images, volumes), and as a container to exchange data with other libraries,
e.g. via :py:func:`drjit.wrap`.
