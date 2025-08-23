.. py:currentmodule:: drjit

.. _eval:

Evaluation
==========

The previous sections explained how Dr.Jit traces computation for later
evaluation. We now examine what situations trigger this evaluation step,
potential performance pitfalls, and how to take manual control if needed.

Why is it needed?
-----------------

Certain operations *simply cannot* be traced. When Dr.Jit encounters them in a
program, they force the system to compile and run a kernel before continuing.

The following operations all exhibit this behavior:

1. **Printing the contents of arrays**: printing requires knowing the actual
   array contents at that moment, hence evaluation can no longer be postponed.
   Here is an example:

   .. code-block:: python

      a = Float(1, 2) + 3 # <-- traced
      print(a)            # <-- evaluated

   (Dr.Jit offers an alternative :py:func:`drjit.print()` statement that can
   print in a delayed fashion to be compatible with tracing.)

2. **Accessing arrays along their trailing dimension**: recall that the
   trailing dimension of Jit-compiled arrays plays a special role, as the
   system uses it to parallelize the computation. Accessing specific entries
   or gathering along this dimension creates a data dependency that cannot be
   satisfied within the same parallel phase. An example:

   .. code-block:: python

      a = Float(1, 2) + 3 # <-- traced

      b = a[0]            # <-- evaluated

      # Also generally evaluated:
      b = dr.gather(Float, source=a, index=UInt32(0))

   Dr.Jit will split such programs into multiple kernels, which it does eagerly
   by evaluating the source array whenever it detects this type of access.

   Note that it is specifically the trailing dimension that causes this
   behavior. Accesses to components of a nested array type like
   :py:class:`drjit.cuda.Array3f` are unproblematic and can be traced.

   .. code-block:: python

      a = Array3f(1, 2, 3) + 4 # <-- traced
      b = a[0] + a.y           # <-- traced

3. **Side effects**: operations such as :py:func:`drjit.scatter`,
   :py:func:`drjit.scatter_reduce`, etc., modify existing arrays. While such
   operations are traced (i.e. postponed for later evaluation), subsequent
   access of the modified array triggers a variable evaluation. An example:

   .. code-block:: python

      a = dr.empty(Float, 3)
      dr.scatter(target=a, value=Float(0, 1, 2), index=UInt32(2, 1, 0)) # <-- traced
      b = a + 1 # <-- evaluates 'a' and traces the addition

   Here, evaluation enforces an ordering constraint that is in general needed
   to ensure correctness in a parallel execution context.

4. **Reductions**: when using an operation such as :py:func:`dr.sum() <sum>` or
   :py:func:`dr.all() <all>` to reduce along the trailing dimension of an array,
   a prior evaluation is required. Some reductions accept an optional
   ``mode="symbolic"`` parameter to postpone evaluation.

5. **Data exchange**: casting Dr.Jit variables into nd-arrays of other
   frameworks (e.g. NumPy, PyTorch, etc.) requires their evaluation, as other
   libraries don't have a compatible concept of traced computation.

6. **Manual**: variable evaluation can also be triggered *manually* using the
   operation :py:func:`drjit.eval()`:

   .. code-block:: python

      dr.eval(a)

How does it work?
-----------------

Suppose that a traced computation has the following dependence structure.

.. only:: not latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph1-light.svg
     :width: 300
     :class: only-light
     :align: center

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph1-dark.svg
     :width: 300
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph1-light.svg
     :width: 300
     :align: center

If we now evaluate ``x`` via

.. code-block:: python

   dr.eval(x)

this generates a kernel that also computes the dependent variables ``a`` and
``b``. Running this kernel turns ``x`` from an *implicit* representation (a
computation graph node) into an *explicit* one (a memory region stored on the
CPU/GPU).

.. only:: not latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph2-light.svg
     :class: only-light
     :width: 300
     :align: center

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph2-dark.svg
     :width: 300
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph2-light.svg
     :width: 300
     :align: center

This evaluated ``x`` no longer needs its dependencies---any parts of the
computation graph that become unreferenced as a consequence of this are
automatically removed.

Suppose that we now evaluate ``y``:

.. code-block:: python

   dr.eval(y)

This will compile another kernel that includes the step ``b`` a *second
time*. If this redundant computation is costly, we could instead also have
explicitly evaluated both ``x`` and ``y`` as part of the same kernel.

.. code-block:: python

   dr.eval(x, y)

Unevaluated arrays specify how something can be computed without consuming any
device memory. In contrast, a large evaluated array can easily take up
gigabytes of device memory. Because of this, some care is often advisable to
avoid superfluous variable evaluation.

Once evaluated, variables behave exactly the same way in subsequent
computations except that any use in kernels causes them to be *loaded* instead
of *recomputed*. Passing an already evaluated array to :py:func:`dr.eval()
<eval>` a second time is a no-op.

Asynchronous execution
----------------------

Dr.Jit is *asynchronous* in two different ways:

1. operations are traced for later evaluation as previously explained.

2. evaluation itself also takes place asynchronously.

For example, a statement like

.. code-block:: python

   dr.eval(x)

appends a work item to a GPU/CPU command queue and returns right away instead
of waiting for this evaluation to complete. This way, we can immediately begin
tracing the next block of code, which improves performance by keeping both host
and target device busy. A more accurate version of the previous flow diagram
therefore looks as follows:

.. only:: not latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph3-light.svg
     :class: only-light
     :align: center

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph3-dark.svg
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cgraph3-light.svg
     :align: center

This behavior is transparent, which means that no special steps need to be
taken on the user's side (e.g., to wait for computation to finish or to
synchronize with the queue---Dr.Jit will do so automatically if needed).

.. _caching:

Kernel caching
--------------

When Dr.Jit evaluates an expression, it must generate and compile a *kernel*,
i.e., a self-contained parallel program that can run on the target device. This
compilation step is not free---in fact, compilation can sometimes take *longer*
than the actual runtime of an associated kernel.

To mitigate this cost, Dr.Jit implements a *kernel cache*. Roughly speaking,
the idea is that we often end up repeating the same kind of computation with
different data. Whenever the system detects that it already has a suitable
kernel at hand, it reuses this kernel instead of compiling it again
(this is called a cache *hit*).

Cache *misses* fall into two categories: a *soft* miss means that we already
encountered this kernel in a previous session, and a compiled version can be
loaded from disk. A *hard* miss means that this computation was never seen
before and requires a costly compilation step is needed. The following flow
diagram visualizes the role of the cache:

.. only:: not latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cache-light.svg
     :class: only-light
     :align: center

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cache-dark.svg
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/cache-light.svg
     :align: center

In a gradient-based optimization, typically only the first gradient step will
compile kernels (causing either soft or hard misses), which are subsequently
reused many times.

The location of the on-disk cache depends on the backend, operating system, and
type of kernel. It can be found in the following location (where ``~`` refers
to the user's home directory):

1. **LLVM Backend**:

   - **Linux** and **macOS**: ``~/.drjit/*.llvm.bin``
   - **Windows**: ``~\AppData\Local\Temp\drjit\*.llvm.bin``

1. **CUDA Backend**:

   The CUDA environment already provides an on-disk kernel caching mechanism,
   which is reused by Dr.Jit. The cache files can be found here:


   - **Linux**: ``~/.nv/ComputeCache\*``
   - **Windows**: ``~\AppData\Roaming\NVIDIA\ComputeCache\*``

   Kernels that perform hardware-accelerated ray tracing go through a different
   compilation pipeline named `OptiX
   <https://developer.nvidia.com/rtx/ray-tracing/optix>`__. In this case, they
   are cached in a single file at the following location:

   - **Linux**: ``~/.drjit/optix7cache.db``
   - **Windows**: ``~\AppData\Local\Temp\drjit\optix7cache.db``

Analyzing JIT behavior
----------------------

Tracing and evaluation run silently behind the scenes, but sometimes it can be
useful to watch this process as it occurs. For this, call
:py:func:`dr.set_log_level() <drjit.log_level>` to set the log level to a value
of :py:attr:`Info <drjit.LogLevel.Info>` or lower (the default is
:py:attr:`Warn <drjit.LogLevel.Warn>`).

.. code-block:: pycon
   :emphasize-lines: 6, 8, 9, 10, 11

   >>> import drjit as dr
   >>> from drjit.auto import Float

   >>> x = dr.arange(Float, 1024)
   >>> y = x + 1
   >>> dr.set_log_level(dr.LogLevel.Info)
   >>> y
   jit_eval(): launching 1 kernel.
     -> launching c77f588e6b5e7e2f (n=1024, in=0, out=1, ops=8, jit=33.12 us):
        cache miss, build: 4.1462 ms.
   jit_eval(): done.
   [1, 2, 3, .. 1018 skipped .., 1022, 1023, 1024]

With this increased level, every kernel launch now triggers an explicit log
message. Here, it shows that this computation was encountered for the first
time (``cache miss``), requiring a backend compilation step that took much
longer than the time spent within Dr.Jit (~4 ms vs ~33 μs). This compilation
step is, however, only needed once.

Other statistics shown here are the kernel ID (hexadecimal number), number of
elements processed in parallel (``n=1024``), number of input (``in=0``) and
output (``out=1``) arrays and IR operations (``ops=8``), which is a simple
proxy for the complexity of a generated kernel.

The function :py:func:`dr.whos() <drjit.whos>` lists all currently registered
JIT variables along with statistics about compilation and memory allocation. It
is also possible to assign labels to specific variables to identify them in
this list.

.. code-block:: pycon

   >>> x.label = "x"
   >>> y.label = "y"
   >>> dr.whos()

     ID       Type       Status     Refs       Size      Storage   Label
     ========================================================================
     1        cuda u32                 1       1024
     2        cuda f32                 1       1024
     4        cuda f32   device 0      1       1024        4 KiB
     ========================================================================

     JIT compiler
     ============
      - Storage           : 4 KiB on device, 12 KiB unevaluated.
      - Variables created : 5 (peak: 6, table size: 832 B).
      - Kernel launches   : 1 (0 cache hits, 0 soft, 1 hard misses).

     Memory allocator
     ================
      - host              : 0 B/0 B used (peak: 0 B).
      - host-async        : 0 B/0 B used (peak: 0 B).
      - host-pinned       : 0 B/0 B used (peak: 0 B).
      - device            : 4 KiB/4 KiB used (peak: 4 KiB).


It shows that three variables are registered with the system, of which one
(index 4, label ``y``) is evaluated and occupies 4 KiB of device memory on CUDA
device 0.

The message also shows the total number of kernel launches and corresponding
hits or soft/hard misses. The launch statistics in particular can be helpful to
investigate performance pitfalls (please see the next section for details).

Finally, it is possible to visualize the complete graph of traced computation
via :py:func:`dr.graphviz() <graphviz>` (this requires installing the
``graphviz`` `PyPI package <https://pypi.org/project/graphviz/>`__).
Let's include one more operation to make this a bit more interesting:

.. code-block:: pycon

   >>> z = dr.sinh(x*y)
   >>> z.label = "z"
   >>> dr.graphviz()  # <-- Alternatively, dr.graphviz().view() opens a separate window

This produces a graph combining the previous expression with the implementation
of :py:func:`dr.sinh() <drjit.sinh>`.

.. only:: not latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/gv-light.svg
     :class: only-light
     :align: center

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/gv-dark.svg
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://d38rqfq1h7iukm.cloudfront.net/media/uploads/wjakob/2024/06/gv-light.svg
     :align: center

Pitfalls
--------

Please aware of the following cases that can lead to poor performance.

Caching
~~~~~~~

Dr.Jit treats *literal constants* (i.e., known scalars such as ``1.234``) as
code rather than data. This is generally a good thing, but it also means that
kernels that are identical except for such a literal constant do not benefit
from the kernel cache, as they are considered to be distinct.

This can turn into a rather severe performance bottleneck when launching
kernels from a loop. Consider the following example:

.. code-block:: python

   y = Float(....)
   for i in range(1000):
       y = f(y, i)
       dr.eval(y)
       # ...

If ``f()`` depends on ``i``, this code will likely compile 1000 separate
kernels that are identical except for the number 0, 1, 2, ..., that is baked
into the traced code of ``f()``. Such repeated kernel compilation steps often
end up dominating the computation time and lead to poor device utilization.

Here, it would have been better to compile a single kernel that can handle
any possible value of ``i``.

To do so, use the function :py:func:`dr.opaque() <drjit.opaque>`, which creates
an evaluated variable containing the given constant. With this change, the
counter is no longer a literal constant, which collapses all loop iterations to
a single consistent cache entry.

.. code-block:: python

   for i in range(1000):
       i2 = dr.opaque(Int, i)
       y = f(y, i2)
       dr.eval(y)

Alternatively, the following also works

.. code-block:: python

   i = dr.opaque(Int, 0)
   for _ in range(1000):
       y = f(y, i)
       i += 1
       dr.eval(y, i)

To track down such issues, use the function :py:func:`drjit.whos()` (and
possibly, increase the log level as explained above). If you notice that every
iteration of a loop generates soft or hard cache misses, the issue is
potentially due to a changing literal constant.

Caching, continued
~~~~~~~~~~~~~~~~~~

Another pattern that can break kernel caching are growing dependency chains.
Consider a function ``f`` that fetches numbers from a random number generator
(in this case, using the builtin :py:class:`PCG32 <drjit.auto.PCG32>`
pseudorandom number generator provided by Dr.Jit -- however, note that the
issue explained here is not specific to random number generation).

A loop calls this function 1000 times in a row and accumulates the results:

.. code-block:: python

   from drjit.auto import PCG32

   rng = PCG32(100000)
   y = Float(0)

   for _ in range(1000):
       y += f(rng)
       dr.eval(y)

:ref:`Benchmarking <bench>` a program of the above form will likely
reveal that kernels launched the loop become *progressively slower*. Furthermore,
kernel caching doesn't work, and 1000 separate costly compilation steps are
needed. What is going on?!

The problem is that the random number generator has an internal state variable
:py:attr:`rng.state <drjit.auto.PCG32.state>` that evolves whenever the
function ``f`` fetches a new sample (e.g., using :py:func:`rng.next_float32()
<drjit.auto.PCG32.next_float32>`). If we never explicitly evaluate the random
number generator, then these updates remain in computation graph form.
Consequently, each iteration of the loop will repeat all (up to 1000) steps to
re-play the ``rng`` state up to the current iteration, which breaks caching and
causes the progressive slowdown. The fix is easy:

.. code-block:: python

   dr.eval(x, rng)

we must simply remember to also evaluate the ``rng`` variable.

Element-wise access
~~~~~~~~~~~~~~~~~~~

Element-wise access along the trailing (vectorized) dimension of Dr.Jit arrays
is best avoided. For example, the following is inefficient:

.. code-block:: python

   x: Float = f(...) # Some function, which produces a large output array
   for i in range(len(x)):
       if x[i] < 0:
           raise Exception('Negative element found!')

If ``x`` is a 1-million entry :py:class:`drjit.cuda.Float` array, the loop
will generate 1 million separate PCI-Express transactions that each copy a
tiny 4-byte value to the CPU. This is on top of the general inefficiency of
iterating over many values in an interpreted programming language.

Instead, prefer vectorized constructions:

.. code-block:: python

   if dr.any(x < 0):
       raise Exception('Negative element found!')

If you absolutely must iterate over scalar elements, it's better to use
a host-centric array framework, e.g., by calling :py:func:`x.numpy()
<drjit.ArrayBase.numpy>` to convert ``x`` into a NumPy array.
