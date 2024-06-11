.. py:currentmodule:: drjit

.. _optim:

Optimizations
-------------

This section lists optimizations performed by Dr.Jit while tracing code. The
examples all use the following import:

.. code-block:: pycon

   >>> from drjit.auto import Int

Vectorization and parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Dr.Jit automatically *vectorizes* and *parallelizes* traced code. The
implications of these transformations are backend-specific.

Consider the following simple calculation, which squares an integer
sequence with 10000 elements.

.. code-block:: pycon

   >>> dr.arange(Int, 10000)**2
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
:py:func:`drjit.set_thread_count` to specify the number of threads used for
parallel processing.

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

.. only:: not latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_cuda.svg
     :class: only-light

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_cuda_dark.svg
     :class: only-dark

.. only:: latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_cuda.svg

The :py:attr:`drjit.ReduceMode.Local` strategy in the above plot performs a
`butterfly reduction <https://en.wikipedia.org/wiki/Butterfly_network>`__ to
locally pre-reduce writes targeting the same region of memory, which
significantly reduces the dangers of atomic memory contention.

On the CPU (LLVM) backend, :py:attr:`Direct` mode can become so slow that this
essentially breaks the program. The :py:attr:`Local` strategy is analogous to
the CUDA backend and improves performance by an order of magnitude when many
writes target the same element. In this benchmark, that becomes less likely as
the target array grows, and the optimization becomes ineffective.

.. only:: not latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_llvm.svg
     :class: only-light

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_llvm_dark.svg
     :class: only-dark

.. only:: latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/01/scatter_add_llvm.svg

The :py:attr:`drjit.ReduceMode.Expand` strategy produces a near-flat profile.
It replicates the target array to avoid write conflicts altogether, which
enables the use of non-atomic memory operations. This is *significantly* faster
but also *very memory-intensive*, as the storage cost of an 1 MiB array targeted
by a :py:func:`drjit.scatter_reduce` operation now grows to *N* MiB,
where *N* is the number of cores. The functions :py:func:`expand_threshold`
and :py:func:`set_expand_threshold` can be used to set thresholds that
determine when Dr.Jit is willing to automatically use this strategy.

Packet memory operations
^^^^^^^^^^^^^^^^^^^^^^^^

The functions :py:func:`drjit.gather`, :py:func:`drjit.scatter`, and
:py:func:`drjit.scatter_reduce` can be used to access vectors in a flat array.

For example,

.. code-block:: pycon

   >>> buffer = Float(...)
   >>> vec4_out = dr.gather(dtype=Array4f, source=buffer, index=..)

is equivalent to (but *more efficient* than) four subsequent gathers that access
elements ``index4*0`` to ``index*4+3``. Dr.Jit compiles such operations into
*packet memory operations* whenever the size of the output array is a power of
two. This yields a small performance improvement on the GPU (on the order of
5-30%) and a massive speedup on the LLVM CPU backend especially for scatters.
See the flag :py:attr:`drjit.JitFlag.PacketOps` for details.

Other
^^^^^

Some other optimizations are specific to symbolic operations, such as

- :py:attr:`drjit.JitFlag.OptimizeCalls`,
- :py:attr:`drjit.JitFlag.MergeFunctions`,
- :py:attr:`drjit.JitFlag.OptimizeLoops`,
- :py:attr:`drjit.JitFlag.CompressLoops`.

Please refer the documentation of these flags for details.
