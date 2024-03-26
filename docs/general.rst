.. py:currentmodule:: drjit

General information
===================


Optimizations
-------------

This section lists optimizations performed by Dr.Jit while tracing code. The
examples all use the following import:

.. code-block:: pycon

   >>> from drjit.llvm import Int

Vectorization and parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dr.Jit automatically *vectorizes* and *parallelizes* traced code. The
implications of these transformations are backend-specific.

Consider the following simple calculation, which squares an integer
sequence with 10000 elements.

.. code-block:: pycon

   >>> dr.arange(dr.llvm.Int, 10000)**2
   [0, 1, 4, .. 9994 skipped .., 99940009, 99960004, 99980001]

On the LLVM backend, *vectorization* means that generated code uses instruction
set extensions such as Intel AVX/AVX2/AVX512, or ARM NEON when they are
available. For example, when the machine supports the `AVX512
<https://en.wikipedia.org/wiki/AVX-512>`__ extensions, each machine
instruction processes a *packet* of 16 values, which means that a total of 625
packets need to be evaluated.

The system uses the built-in `nanothread
<https://github.com/mitsuba-renderer/nanothread>`__ thread pool to distribute
packets to be processed among the available processor cores. In this simple
example, there is not enough work to truly benefit from multi-core parallelism,
but this approach pays off in more complex examples.

You can use the functions :py:func:`drjit.thread_count`,
:py:func:`drjit.set_thread_count` to specify the number of threads used for parallel processing.

On the CUDA backend, the system automatically determines a number of *threads*
that maximize occupancy along with a suitable number of *blocks* and then
launches a parallel program that spreads out over the entire GPU (assuming that
there is enough work to do so).

.. _cow:

Copy-on-Write
^^^^^^^^^^^^^

Arrays are reference-counted and use a `Copy-on-Write
<https://en.wikipedia.org/wiki/Copy-on-write>`__ (CoW) strategy. This means
that copying an array is cheap since the copy can reference the original array
without requiring a device memory copy. The matching variable indices in the
example below demonstrate the lack of an actual copy.

.. code-block:: pycon

   >>> a = Int(1, 2, 3)
   >>> b = Int(a)        # <- create a copy of 'a'
   >>> a.index, b.index
   (1, 1)

However, subsequent modification causes this copy to be made.

.. code-block:: pycon

   >>> b[0] = 0
   >>> (a.index, b.index)
   (1, 2)

This optimization is always active and cannot be disabled.

Constant propagation
^^^^^^^^^^^^^^^^^^^^

Dr.Jit immediately performs arithmetic involving *literal constant* arrays:

.. code-block:: pycon

   >>> a = Int(4) + Int(5)
   >>> a.state
   dr.VarState.Literal

In other words, the addition does not become part of the generated device code.
This optimization reduces the size of the generated LLVM/PTX IR and can be
controlled via :py:attr:`drjit.JitFlag.ConstantPropagation`.

Dead code elimination
^^^^^^^^^^^^^^^^^^^^^

When generating code, Dr.Jit excludes unnecessary operations that do not
influence arrays evaluated by the kernel. It also removes dead branches in
loops and conditional statements.

This optimization is always active and cannot be disabled.

Value numbering
^^^^^^^^^^^^^^^

Dr.Jit collapses identical expressions into the same variable (this is safe
given the :ref:`CoW <cow>` strategy explained above).

.. code-block:: pycon

   >>> a, b = Int(1, 2, 3), Int(4, 5, 6)
   >>> c = a + b
   >>> d = a + b
   >>> c.index == d.index
   True

This optimization reduces the size of the generated LLVM/PTX IR and can be
controlled via :py:attr:`drjit.JitFlag.ValueNumbering`.

.. _reduce-local:

Local atomic reduction
^^^^^^^^^^^^^^^^^^^^^^

Atomic memory operations can be a bottleneck when they encounter *write
contention*, which refers to a situation where many threads attempt to write to
the same array element at once.

For example, the following operation causes 1'000'000 threads to write to
``a[0]``.

.. code-block:: pycon

   >>> a = dr.zeros(Int, 10)
   >>> dr.scatter_add(target=a, index=dr.zeros(Int, 1000000), value=...)

Since Dr.Jit vectorizes the program during execution, the computation is
grouped into *packets* that typically contain 16 to 32 elements. By locally
pre-accumulating the values within each packet and then only performing 31-62K
atomic memory operations (instead of 1'000'000), performance can be
considerably improved.

This issue is particularly important when automatically differentiating
computation in *reverse mode* (e.g. :py:func:`drjit.backward`), since
this transformation turns differentiable global memory reads into atomic
scatter-additions. A differentiable scalar read is all it takes to create
such an atomic memory bottleneck.

The following plots illustrate the expected level performance in a
microbenchmark that scatters-adds :math:`10^8` random integers into a buffer at
uniformly distributed positions. The size of the target buffer varies along the
horizontal axis. Generally, we expect to see significant contention on the
left, since this involves a large number of writes to only a few elements. The
behavior of GPU and CPU atomics are somewhat different, hence we look at them
in turn starting with the CUDA backend.

The :py:attr:`drjit.ReduceMode.Direct` strategy generates a plain atomic
operation without additional handling. This generally performs badly except for
two special cases: when writing to a scalar array, the NVIDIA compiler detects
this and performs a specialized optimization (that is, however, quite specific
to this microbenchmark and unlikely to work in general). Towards the right,
there is essentially no contention and multiple writes to the same destination
are unlikely to appear within the same warp, hence
:py:attr:`drjit.ReduceMode.Direct` outperforms the other methods.

.. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_cuda.svg
  :class: only-light

.. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_cuda_dark.svg
  :class: only-dark

The :py:attr:`drjit.ReduceMode.Local` strategy in the above plot performs a
`butterfly reduction <https://en.wikipedia.org/wiki/Butterfly_network>`__ to
locally pre-reduce writes targeting the same region of memory, which
significantly reduces the dangers of atomic memory contention.

On the CPU (LLVM) backend, :py:attr:`Direct` mode can become so slow that this
essentially breaks the program. The :py:attr:`Local` strategy is analogous to
the CUDA backend and improves performance by an order of magnitude when many
writes target the same element. In this benchmark, that becomes less likely as
the target array grows, and the optimization becomes ineffective.

.. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_llvm.svg
  :class: only-light

.. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_llvm_dark.svg
  :class: only-dark

The :py:attr:`drjit.ReduceMode.Expand` strategy produces a near-flat profile.
It replicates the target array to avoid write conflicts altogether, which
enables the use of non-atomic memory operations. This is *significantly* faster
but also *very memory-intensive*, as the storage cost of an 1 MiB array targeted
by a :py:func:`drjit.scatter_reduce` operation now grows to *N* MiB,
where *N* is the number of cores. The functions :py:func:`expand_threshold`
and :py:func:`set_expand_threshold` can be used to set thresholds that
determine when Dr.Jit is willing to automatically use this strategy.


Other
^^^^^

Some other optimizations are specific to symbolic operations, such as

- :py:attr:`drjit.JitFlag.OptimizeCalls`,
- :py:attr:`drjit.JitFlag.MergeFunctions`,
- :py:attr:`drjit.JitFlag.OptimizeLoops`,
- :py:attr:`drjit.JitFlag.CompressLoops`.

Please refer the documentation of these flags for details.

.. _horizontal-reductions:

Horizontal reductions
---------------------

Dr.Jit offers the following *horizontal operations* that reduce the dimension
of an input array, tensor, or Python sequence:

- :py:func:`drjit.sum` and :py:func:`drjit.mean`, which reduce using ``+``,
- :py:func:`drjit.prod`, which reduces using ``*``,
- :py:func:`drjit.min`, which reduces using ``min()``,
- :py:func:`drjit.max`, which reduces using ``max()``,
- :py:func:`drjit.all`, which reduces using ``&``, and
- :py:func:`drjit.any` and :py:func:`drjit.none`, which reduce using ``|``.

By default, these functions reduce along the outermost dimension and return an
instance of the array's element type. For instance, sum-reducing an array ``a`` of
type :py:class:`drjit.cuda.Array3f` would just be a convenient abbreviation for
the expression ``a[0] + a[1] + a[2]`` of type :py:class:`drjit.cuda.Float`.
Dr.Jit can trace this operation and include it in the generated kernel (i.e.,
it is *symbolic*).

Reductions of JIT-compiled 1D arrays (e.g., :py:class:`drjit.cuda.Float`) are an
important special case. Since each value of such an array represents a
different execution thread of a parallel program, Dr.Jit must first invoke
:py:func:`drjit.eval` to evaluate and store the array in memory and then launch
a device-specific implementation of a horizontal reduction. This interferes with
Dr.Jit's regular mode of operation, which is to capture a maximally large
program without intermediate evaluation. In other words, use of such 1D
reductions may have a negative effect on performance. The operation will fail
in execution contexts where evaluation is forbidden, e.g., while capturing
symbolic loops and function calls. Atomic operations like
:py:func:`drjit.scatter_add` can be an interesting alternative in such cases.

Furthermore Dr.Jit does *not* reduce such JIT-compiled 1D arrays to their
element type (e.g., a standard Python `float`). Instead, it returns a dynamic
array of the same type, containing only a single element. This is
intentional--unboxing the array into a Python scalar would require transferring
the value to the CPU, which would incur GPU<->CPU synchronization overheads.
You must explicitly index into the result (``result[0]``) to obtain a value
with the underlying element type. Boolean arrays define a ``__bool__`` method
so that such indexing can be avoided. For example, the following works as
expected:

.. code-block:: python

   a = drjit.cuda.Float(...)
   # The line below is simply a nicer way of writing "if dr.any(a < 0)[0]:"
   if dr.any(a < 0):
      # ...

All reduction operations take an optional argument ``axis`` that specifies the
axis of the reduction. Its default value ``0`` implies a reduction over the
outermost axis. Negative indices (e.g. ``-1``) count backwards from the
innermost axis. The special argument ``axis=None`` causes a simultaneous
reduction over all axes.

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
<https://docs.python.org/3/library/dataclasses.html>`__. Note that this type
must be default-constructible, which means that its members should have default
initializers.

.. code-block:: python

   from drjit.cuda.ad import Float
   from dataclasses import dataclass

   @dataclass
   class MyPoint2f:
       x: Float = Float(0)
       y: Float = Float(0)

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)

The second option is to annotate an existing non-dataclass type (e.g. a
standard Python class or a C++ binding) with a static ``DRJIT_STRUCT`` member.
This is simply a dictionary describing the names and types of all fields.
Such custom types must also be default-constructible (i.e., the constructor
should work if called without arguments). The following is analogous to the
above dataclass version:

.. code-block:: python

   from drjit.cuda.ad import Float

   class MyPoint2f:
       DRJIT_STRUCT = { 'x' : Float, 'y': Float }

       def __init__(self, x: Float | None = None, y: Float | None = None):
           self.x = x or Float()
           self.y = y or Float()

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)

Fields don't exclusively have to be containers or Dr.Jit types. For example, we
could have added an extra ``datetime`` entry to record when a set of points was
captured. Such fields will be ignored by traversal operations.


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
- ``RedT``: :py:class`drjit.llvm.ad.Float`, and
- ``MaskT``: :py:class:`drjit.llvm.ad.Array2b`.

The mysterious-looking underscored forward references can be found at the
bottom of the same stub, for example:

.. code-block:: python

   _Array2fCp: TypeAlias = Union['Array2f', '_FloatCp', 'drjit.llvm._Array2fCp',
                                 'drjit.scalar._Array2fCp', '_Array2f16Cp']

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
   def maximum(a: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], b: SelfCpT, /) -> SelfT: ...
   @overload
   def maximum(a: SelfCpT, b: ArrayBase[SelfT, SelfCpT, ValT, ValCpT, RedT, MaskT], /) -> SelfT: ...
   @overload
   def maximum(a: T, b: T, /) -> T: ...

Suppose we are computing the maximum of two 3D arrays:

.. code-block:: python

   a: Array3u = ...
   b: Array3f = ...
   c: ??? = dr.maximum(a, b)

In this case, ``???`` is ``Array3f`` due to the type promotion rules, but how
does the type checker know this? When it tries the first overload, it
realizes that ``b: Array3f`` is *not* part of the ``SelfCpT`` (compatible
with *self*) type parameter of ``Array3u``. In second overload, the test is
reversed and succeeds, and the result is the ``SelfT`` of ``Array3f``, which is
also ``Array3f``. The third overload exists to handle cases where neither input
is a Dr.Jit array type. (e.g. ``dr.maximum(1, 2)``)
