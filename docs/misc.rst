.. py:currentmodule:: drjit

Miscellaneous
=============

.. _pytrees:

PyTrees
-------

The word *PyTree* (borrowed from `JAX
<https://jax.readthedocs.io/en/latest/pytrees.html>`_) refers to a tree-like
data structure made of Python container types including

- ``list``,
- ``tuple``,
- ``dict``,
- `data classes <https://docs.python.org/3/library/dataclasses.html>`__.
- custom Python classes or C++ bindings with a ``DRJIT_STRUCT`` annotation.

Various Dr.Jit operations will automatically traverse such PyTrees to process
any Dr.Jit arrays or tensors found within. For example, it might be convenient
to store differentiable parameters of an optimization within a dictionary and
then batch-enable gradients:

.. code-block:: python

   from drjit.cuda.ad import Array3f, Float

   params = {
       'foo': Array3f(...),
       'bar': Float(...)
   }

   dr.enable_grad(params)

PyTrees can similarly be used as state variables in symbolic loops and
conditionals, as arguments and return values of symbolic calls, as arguments of
scatter/gather operations, and many others (the :ref:`reference <reference>`
explicitly lists the word *PyTree* in all supported operations).

Limitations
^^^^^^^^^^^

You may not use Dr.Jit types as *keys* of a dictionary occurring within a
PyTree. Furthermore, PyTrees may not contain cycles. For example, the following
data structure will cause PyTree-compatible operations to fail with a
``RecursionError``.

.. code-block:: python

   x = []
   x.append(x)

Finally, Dr.Jit automatically traverses tuples, lists, and dictionaries,
but it does not traverse subclasses of basic containers and other generalized
sequences or mappings. This is intentional.

.. _custom_types_py:

Custom types
^^^^^^^^^^^^

There are two ways of extending PyTrees with custom data types. The first is to
register a Python `data class
<https://docs.python.org/3/library/dataclasses.html>`__.

.. code-block:: python

   from drjit.cuda.ad import Float
   from dataclasses import dataclass

   @dataclass
   class MyPoint2f:
       x: Float
       y: Float

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)

The second option is to annotate an existing non-dataclass type (e.g. a
standard Python class or a C++ binding) with a static ``DRJIT_STRUCT`` member.
This is simply a dictionary describing the names and types of all fields.
Such custom types must be default-constructible (i.e., the constructor
should work if called without arguments).

.. code-block:: python

   from drjit.cuda.ad import Float

   class MyPoint2f:
       DRJIT_STRUCT = { 'x' : Float, 'y': Float }

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)

Fields don't exclusively have to be containers or Dr.Jit types. For example, we
could have added an extra ``datetime`` entry to record when a set of points was
captured. Such fields will be ignored by traversal operations.

.. _local_memory:

Local memory
------------

*Local memory* is a relatively advanced feature of Dr.Jit. You may need it
it if you encounter the following circumstances:

1. A symbolic loop in your program must *both read and write* the same
   memory buffer using computed indices.

2. The buffer is *entirely local* to a thread of the computation (i.e., local
   to an element of an array program).

3. The buffer is *small* (e.g., a few 100-1000s of entries).

The :py:func:`drjit.alloc_local` function allocates a local memory buffer of
size ``n`` and type ``T``:

.. code-block:: python

   buf: dr.Local[T] = dr.alloc_local(T, n)

You may further specify an optional ``value=...`` argument to
default-initialize the buffer entries. The returned instance of type
:py:class:`drjit.Local` represents the allocation. Its elements can be accessed
using the regular ``[]`` indexing syntax.

Example uses of local memory might include a local stack to traverse a tree
data structure, `insertion sort
<https://en.wikipedia.org/wiki/Insertion_sort>`__ to maintain a small sorted
list, or a `LU factorization
<https://en.wikipedia.org/wiki/LU_decomposition>`__ of a small (e.g. 32×32)
matrix with column pivoting. In contrast to what the name might suggest, local
memory is neither particularly fast nor local to the processor. In fact, it is
based on standard global device memory. Local memory is also not to be confused
with *shared memory* present on CUDA architectures.

The purpose of local memory is that it exposes global memory in a different
way to provide a *local scratch space* within a larger parallel computation.
Normally, one would use :py:func:`drjit.gather` and :py:func:`drjit.scatter` to
dynamically read and write memory. However, they cannot be used in this
situation because *read-after-write* (RAW) dependencies would trigger variable
evaluations that aren't permitted in a symbolic context. Local memory legalizes
such programs because RAW dependencies among multiple threads are simply not possible.

Local memory may also appear similar to dynamic array types like
:py:class:`drjit.cuda.ArrayXf`, which group multiple variables/registers into
an array for convenience. The key difference is that ``ArrayXf`` does not
support element access with computed indices, while local memory buffers do.

Allocating, reading, and writing local memory are all symbolic operations that
don't consume any memory by themselves. However, when local memory appears in a
kernel being launched, the system must conceptually allocate extra memory for
the duration of the kernel (the details of this are backend-dependent). While
this does not contribute to the long-term memory requirements of a program, the
short term memory requirements can be *significant* because local memory is
separately allocated for each thread. On a CUDA device, there could be as many
as 1 million simultaneously resident threads across thread blocks. A seemingly
small local 1024-element single precision array then expands into a whopping 4
GiB of memory.

See the snippet below for an example that calls a function ``f()``  ``n`` times
to compute a histogram (stored in local memory) of its outputs to then find the
largest histogram bucket.

.. code-block:: python

   from drjit.auto import UInt32

   # A function returning results in the range 0..9
   def f(i: UInt32) -> UInt32: ....

   @dr.syntax
   def g(n: UInt32):
       # Create zero-initialized buffer of type 'drjit.Local[UInt32]'
       hist = dr.alloc_local(UInt32, 10, value=dr.zeros(UInt32))

       # Fill histogram
       i = UInt32(0)
       while i < n:        # <-- symbolic loop
           hist[f(i)] += 1 # <-- read+write with computed index
           i += 1

       # Get the largest histogram entry
       i, maxval = UInt32(0), UInt32(0)
       while i < 10:
           maxval = dr.maximum(maxval, hist[i])
           i += 1
       return maxval

When this function is evaluated with an *array* of inputs (e.g. ``n=UInt32(n1,
n2, ...)``) it will create several histograms with different numbers of
functions evaluations in parallel. Each evaluation conceptually gets its own
``hist`` variable in this case.

Dr.Jit can also create local memory over :ref:`PyTrees <pytrees>` (for
example, instead of ``dtype=Float``, we could have called
:py:func:`drjit.alloc_local` with a complex number, 3x3 matrix, tuple, or
dataclass). Indexing into the :py:class:`drjit.Local` instance then fetches or
stores one instance of the PyTree.

.. note::

   Local memory reads/writes are *not* tracked by Dr.Jit's automatic
   differentiation layer. However, you *may* use local memory in
   implementations of custom differentiable operations based on the
   :py:class:`drjit.CustomOp` interface.

   The implication of the above two points it that when you want to
   differentiate a local memory-based computation, you have to realize the
   forward/backward derivative yourself. This is intentional because the
   default AD-provided derivative would be extremely bad (it will increase the
   size of the scratch space many-fold).

.. _transcendental-accuracy:

Accuracy of transcendental operations
-------------------------------------

Single precision
^^^^^^^^^^^^^^^^

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

Double precision
^^^^^^^^^^^^^^^^

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

.. _type_signatures:

Type signatures
---------------

The :py:class:`drjit.ArrayBase` class and various core functions have
relatively complicated-looking type signatures involving Python `generics and
type variables <https://docs.python.org/3/library/typing.html#generics>`__.
This enables type-checking of arithmetic expressions and improves visual
autocomplete in editors such as `VS Code <https://code.visualstudio.com>`__.
This section explains how these type annotations work.

The :py:class:`drjit.ArrayBase` class is both an *abstract* and a *generic*
Python type parameterized by several auxiliary type parameters. They help
static type checkers like `MyPy <https://github.com/python/mypy>`__ and
`PyRight <https://github.com/microsoft/pyright>`__ make sense how subclasses of
this type transform when passed to various builtin operations. These auxiliary
parameters are:

- ``SelfT``: the type of the array subclass (i.e., a forward reference of the
  type to itself).
- ``SelfCpT``: a union of compatible types, for which ``self + other`` or
  ``self | other`` produce a result of type ``SelfT``.
- ``ValT``: the *value type* (i.e., the type of ``self[0]``)
- ``ValCpT``: a union of compatible types, for which ``self[0] + other`` or
  ``self[0] | other`` produce a result of type ``ValT``.
- ``RedT``: type following reduction by :py:func:`drjit.sum` or
  :py:func:`drjit.all`.
- ``PlainT``: the plain type underlying a special array (e.g.
  ``dr.scalar.Complex2f -> dr.scalar.Array2f``, ``dr.llvm.TensorXi ->
  dr.llvm.Int``).
- ``MaskT``: type produced by comparisons such as ``__eq__``.

For example, here is the declaration of ``llvm.ad.Array2f`` shipped as part of
Dr.Jit's `stub file
<https://nanobind.readthedocs.io/en/latest/typing.html#stubs>`__
``drjit/llvm/ad.pyi``:

.. code-block:: python

   class Array2f(drjit.ArrayBase['Array2f', '_Array2fCp', Float, '_FloatCp', Float, Array2b]):
       pass

String arguments provide *forward references* that the type checker will
resolve at a later point. So here, we have

- ``SelfT``: :py:class:`drjit.llvm.ad.Array2f`,
- ``SelfCp``: a forward reference to ``drjit.llvm.ad._Array2fCp`` (more on this shortly),
- ``ValT``: :py:class:`drjit.llvm.ad.Float`,
- ``ValCpT``: a forward reference to ``drjit.llvm.ad._FloatCp`` (more on this shortly),
- ``RedT``: :py:class`drjit.llvm.ad.Float`,
- ``PlainT``: :py:class:`drjit.llvm.ad.Array2f`, and
- ``MaskT``: :py:class:`drjit.llvm.ad.Array2b`.

The mysterious-looking underscored forward references can be found at the
bottom of the same stub, for example:

.. code-block:: python

   _Array2fCp: TypeAlias = Union['Array2f', '_FloatCp', 'drjit.llvm._Array2fCp',
                                 'drjit.scalar._Array2fCp', 'Array2f', '_Array2f16Cp']

This alias creates a union of types that are *compatible* (as implied by the
``"Cp"`` suffix) with the type ``Array2f``, for example when encountered in an
arithmetic operations like an addition. This includes:

- Whatever is compatible with the *value type* of the array (``drjit.llvm.ad._FloatCp``)
- Types compatible with the *non-AD* version of the array (``drjit.llvm._Array2fCp``)
- Types compatible with the *scalar* version of the array (``drjit.scalar._Array2fCp``)
- Types compatible with a representative *lower-precision* version of that same
  array type (``drjit.llvm.ad._Array2f16Cp``)

These are all themselves type aliases representing unions continuing in the
same vein, and so this in principle expands up a quite huge combined union.
This enables static type inference based on Dr.Jit's promotion rules.

With this background, we can now try to understand a type signature such as
that of :py:func:`drjit.maximum`:

.. code-block:: python

   @overload
   def maximum(a: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], b: SelfCpT, /) -> SelfT: ...
   @overload
   def maximum(a: SelfCpT, b: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, PlainT, MaskT], /) -> SelfT: ...
   @overload
   def maximum(a: T, b: T, /) -> T: ...

Suppose we are computing the maximum of two 3D arrays:

.. code-block:: python

   a: Array3u = ...
   b: Array3f = ...
   c: WhatIsThis = dr.maximum(a, b)

In this case, ``WhatIsThis`` is ``Array3f`` due to the type promotion rules, but how
does the type checker know this? When it tries the first overload, it
realizes that ``b: Array3f`` is *not* part of the ``SelfCpT`` (compatible
with *self*) type parameter of ``Array3u``. In second overload, the test is
reversed and succeeds, and the result is the ``SelfT`` of ``Array3f``, which is
also ``Array3f``. The third overload exists to handle cases where neither input
is a Dr.Jit array type. (e.g. ``dr.maximum(1, 2)``)
