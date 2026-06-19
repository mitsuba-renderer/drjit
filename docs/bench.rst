.. py:currentmodule:: drjit

.. _bench:

Benchmarking
============

Program execution using Dr.Jit is highly asynchronous. Because of this, some
care must be taken when benchmarking programs.

How to do it
------------

The recommended way to measure GPU kernel runtimes is the
:py:func:`drjit.kernel_history` API. While the :py:attr:`drjit.JitFlag.KernelHistory`
flag is set, Dr.Jit records performance metadata for every launched kernel,
including a high-resolution, *device-side* timing (captured via CUDA events on
CUDA, and GPU timestamps on Metal). This sidesteps the pitfalls described below.

.. code-block:: python

   # Record kernel metadata for the region of interest.
   with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
       y = f(x)
       dr.eval(y)

   hist = dr.kernel_history()
   total = sum(k["execution_time"] for k in hist)
   print(f"{total:.3f} ms across {len(hist)} operations(s)")

Each entry is a dictionary; ``execution_time`` is given in milliseconds. See
:py:func:`drjit.kernel_history` for the full set of fields (kernel hash, IR,
cache hits, code generation and backend compilation times, etc.). Note that
calling :py:func:`drjit.kernel_history` also *clears* the recorded history.

How **not** to do it
--------------------

A code fragment like

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

However, the above is an *anti-pattern*. It is bad because:

1. It conflates CPU tracing and GPU kernel costs. Tracing cost can
   be avoided via :ref:`function freezing <freeze>`.

2. It tends to mistakenly include kernel compilation costs and synchronization
   latencies caused by uncached kernel loads and memory allocations at the
   start of an application. These would be avoided in practical usage involving
   a warm kernel and allocation cache.

3. Measurements are noisy because of OS scheduling and CPU/GPU communication.

The recommended way to measure the runtime of a set of kernels is the
:py:func:`drjit.kernel_history` API, which returns a list kernel calls with
high-resolution timing data.

Events
------

For finer-grained control over synchronization, each JIT backend provides an
``Event`` class (:py:class:`drjit.llvm.Event`, and the analogous
``drjit.cuda.Event`` / ``drjit.metal.Event``; also reachable through the active
backend as ``drjit.auto.Event``). An event marks a point in the command stream
and can be used to wait for, or poll, the completion of all work enqueued before
it:

.. code-block:: python

   from drjit.auto import Event

   event = Event()
   y = f(x)
   dr.eval(y)        # enqueue the work
   event.record()    # mark this point in the stream

   if not event.query():   # non-blocking completion check
       event.wait()        # block until the preceding work has finished

Beyond timing, events are a useful mechanism for *handing off* GPU work between
two threads that drive the same GPU through Dr.Jit at the same time. One thread
records an event once it has enqueued a batch of work; the other thread waits on
that event before consuming the results, establishing the ordering without
forcing a full :py:func:`drjit.sync_thread` (which would stall *all* outstanding
work on the device). Because :py:func:`wait <drjit.llvm.Event.wait>` releases the
GIL, the waiting thread does not block unrelated Python execution.

On the CUDA and LLVM backends, two timing-enabled events also measure the
elapsed device-side time between them:

.. code-block:: python

   start, end = Event(), Event()
   start.record()
   y = f(x); dr.eval(y)
   end.record()
   print(f"{start.elapsed_time(end):.3f} ms")   # waits for 'end', then reports

.. note::

   The Metal backend supports event synchronization but *not* timing:
   ``elapsed_time()`` raises, and the ``enable_timing`` constructor flag is
   ignored. Use :py:func:`drjit.kernel_history` for portable kernel timing.

Integration
-----------

Dr.Jit integrates with platform-native performance analysis tools, specifically
`NVIDIA Nsight Systems <https://developer.nvidia.com/nsight-systems>`__
(CUDA) and `Apple Instruments <https://developer.apple.com/instruments/>`__
(Metal). You can use the following functions to visually mark points
and regions of the program execution to more easily identify them in traces.

- :py:func:`drjit.profile_mark`
- :py:func:`drjit.profile_range`

These annotate the CPU timeline. On CUDA they appear as `NVTX
<https://github.com/NVIDIA/NVTX>`__ marks/ranges in Nsight Systems; on Apple
platforms they are emitted as ``os_signpost`` events and (nestable) intervals,
which appear in the *Points of Interest* track of Instruments. This track is
part of templates such as *Time Profiler* and *System Trace* (but *not* *Metal
System Trace*, which records GPU activity only). All of these are no-ops when no
profiling tool is attached, so they can be left in place in production code.

GPU capture
^^^^^^^^^^^

The :py:func:`drjit.profile_enable` context manager marks a region of the
program for *targeted* GPU capture (via ``cuProfilerStart``/``cuProfilerStop``
on CUDA, and a ``MTLCaptureScope`` on Apple Metal). In both cases, the capture
tool is told to record exactly this region rather than the whole program:

.. code-block:: python

   with dr.profile_enable():
       code_to_be_profiled()

The optional ``active`` argument enables or disables the region based on a
runtime condition, which is convenient for capturing a single iteration of a
loop without restructuring the surrounding code:

.. code-block:: python

   for i in range(n):
       with dr.profile_enable(active=(i == 2)):
           code_to_be_profiled()

Capturing from the command line
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both platforms can record a trace directly from a terminal. The resulting trace
file can be analyzed in the respective GUI tools afterwards.

On CUDA, launch the program with the Nsight Systems CLI. The
``--capture-range=cudaProfilerApi`` option ties the recording to the
:py:func:`drjit.profile_enable` region (i.e. the ``cuProfilerStart`` /
``cuProfilerStop`` pair), so only that region is captured:

.. code-block:: bash

   # Profile the whole run (NVTX ranges from profile_range appear on the timeline)
   nsys profile -o my-capture python myscript.py

   # Restrict the recording to the dr.profile_enable() region(s)
   nsys profile -c cudaProfilerApi -o my-capture python myscript.py

   # Inspect afterwards
   nsys-ui my-capture.nsys-rep

On Apple platforms, you must first set ``MTL_CAPTURE_ENABLED=1`` in the
environment. The :py:func:`drjit.profile_enable` region is then written to a
``.gputrace`` document on disk (default path ``drjit.gputrace``, override with
``DRJIT_METAL_CAPTURE_PATH``), which can be opened in Xcode:

.. code-block:: bash

   # Capture the dr.profile_enable() region to drjit.gputrace
   MTL_CAPTURE_ENABLED=1 python myscript.py
   open drjit.gputrace

For a timeline view, record with the Instruments CLI instead. This does not
require ``profile_enable`` or ``MTL_CAPTURE_ENABLED``. Use the *Metal System
Trace* template for GPU activity, or a template carrying the *Points of Interest*
track (e.g. *Time Profiler* or *System Trace*) to see the ``os_signpost`` marks
and ranges from :py:func:`drjit.profile_mark` / :py:func:`drjit.profile_range`:

.. code-block:: bash

   # GPU activity timeline
   xctrace record --template "Metal System Trace" \
       --output gpu.trace --launch -- python myscript.py

   # profile_mark / profile_range signposts (Points of Interest)
   xctrace record --template "Time Profiler" \
       --output poi.trace --launch -- python myscript.py

   open gpu.trace

Independently of either tool, the ``MTLCaptureScope`` created by
``profile_enable`` is also installed as the capture manager's *default scope*.
This is only relevant for interactive use: if you instead attach Xcode to the
process and click the capture button in its toolbar, the capture is restricted
to the ``profile_enable`` region.
