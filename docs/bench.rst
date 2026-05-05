.. py:currentmodule:: drjit

.. _bench:

Benchmarking
============

Program execution using Dr.Jit is highly asynchronous. Because of this, some
care must be taken when benchmarking programs. For example, a code fragment
like

.. code-block:: python

   start = time.end()
   y = f(x)
   end = time.end()

can run miraculously fast despite ``f(x)`` being an expensive computation. This
is because Dr.Jit merely traced ``f(x)`` but did not yet run it. Adding a call
to :py:func:`dr.eval() <eval>` triggers an immediate evaluation:

.. code-block:: python

   start = time.end()
   y = f(x)
   dr.eval(y)
   end = time.end()

However, this is still not be enough: variable evaluation itself is also
asynchronous. This is normally not something that users have to think about,
but it does become apparent when timing Python code. To force a synchronization,
we could add a call to :py:func:`dr.sync_thread()`.

.. code-block:: python

   start = time.end()
   y = f(x)
   dr.eval(y)
   dr.sync_thread()
   end = time.end()

However, explicit synchronization is generally an *anti-pattern* and not
recommended. Measuring kernels timings on the CPU will also add considerable
noise due to OS scheduling, etc.

The recommended way to measure the runtime of a set of kernels is the
:py:func:`drjit.kernel_history` API, which returns a list of kernel calls with
high-resolution timing data.

Integration
-----------

Dr.Jit has builtin support for the `NVIDIA Nsight System
<https://developer.nvidia.com/nsight-systems>`__ performance analysis tool. You
can use the following two functions to visually mark points and regions of the
program execution so that they can be more easily identified.

- :py:func:`drjit.profile_mark`
- :py:func:`drjit.profile_range`

High-level helpers
------------------

The :py:mod:`drjit.bench` submodule wraps the patterns above into three
ready-made entry points. All three honour the global
:py:func:`drjit.log_level` — at ``Info`` they print a one-block summary, at
``Debug`` they also print one line per iteration, and at ``Warn`` or below
they stay silent.

:py:func:`drjit.bench.repeat` is a decorator that runs a function ``runs``
times with synchronous launches (so the codegen / backend / execution split
is accurate) and, optionally, ``runs`` more times with asynchronous launches
to expose the async speed-up. Per-call timings are appended to an optional
``records`` list so multiple benchmarks can be compared side-by-side:

.. code-block:: python

   records = []

   @dr.bench.repeat(label='Render', runs=8, records=records)
   def render(spp):
       return mi.render(scene, spp=spp, seed=0)

   img = render(32, label='@32 spp')

:py:func:`drjit.bench.measure` is a context manager for one-shot timing of a
single block:

.. code-block:: python

   with dr.bench.measure(label='Rendering'):
       img = mi.render(scene, spp=512, seed=0)

:py:func:`drjit.bench.reset` drops in-process JIT state (kernel history,
malloc cache, allocator stats) and, when ``clear_cache=True``, also wipes
the on-disk kernel cache so that the next run measures backend-compile
time from scratch.
