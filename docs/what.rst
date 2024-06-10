.. _what_is_drjit:

What is Dr.Jit?
===============

Dr.Jit is a library to run massively parallel programs on the GPU or CPU,
potentially along with derivatives for gradient-based optimization. It shares
this purpose with `many <https://cupy.dev>`__ `currently
<https://github.com/google/jax>`__ `existing <https://www.tensorflow.org>`__
`languages <https://www.taichi-lang.org>`__ `and
<https://github.com/NVIDIA/warp>`__ `tools <https://pytorch.org>`__.

Using Dr.Jit involves two steps:

1. You write some code and run it.

2. Dr.Jit captures what your program does, converts it into one or more
   parallel *kernels*, and then launches them on a compute accelerator.

**That's it**.  It doesn't do much, but it does this *very efficiently*.

Perhaps the most significant difference is that Dr.Jit is *not* a machine
learning library. Its sweet spot are non-neural programs characterized by
*embarrassing parallelism*, i.e., programs with large data-parallel regions. A
good example of this are `Monte Carlo
<https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ methods with their
parallel sample evaluation---indeed, the reason why this project was originally
created was to provide the foundation of `Mitsuba 3
<https://mitsuba.readthedocs.io/en/latest/>`__, a differentiable Monte Carlo
renderer. That said, Dr.Jit is a general tool that also supports other kinds of
embarrassingly parallel computation.

This documentation focuses on Python, but Dr.Jit also has a C++ interface. A
separate :ref:`documentation section <cpp_iface>` explains how to convert
code between the two languages.

Let's now take a look at the two steps from above in the context of a simple
example.

Capturing computation
---------------------

The following `NumPy <https://numpy.org>`__ array program computes an
approximation of the Fresnel integral
:math:`\int_0^1\sin(t^2)\,\mathrm{d}t\approx 0.3102683` with 1 million function
evaluations.

.. code-block:: python

   import numpy as np
   a = np.linspace(0, 1, 1000000, dtype=np.float32)
   b = np.sin(a**2)
   print(np.mean(b)) # prints 0.3102684

While this code can be adapted into a superficially similar Dr.Jit program

.. code-block:: python

   import drjit as dr
   from drjit.auto import Float

   a = dr.linspace(Float, 0, 1, 1000000)
   b = dr.sin(a**2)
   print(dr.mean(b))

there are fundamental differences between the two:

1. In NumPy, operations like ``np.linspace``, ``np.sin``, ``**``, etc.,
   load and store memory-backed arrays. Accessing memory is slow, which turns
   this part into a bottleneck rather than the actual math.

2. Dr.Jit *traces* the computation instead of executing it right away. This
   means that it *pretends* to execute until reaching the last line, at which
   point it launches a kernel combining all the collected operations. Not only
   does this avoid loading and storing temporaries: it also makes it easy to
   parallelize the program on compute accelerators.

This is just a toy example, but the key mechanism shown in the example scales:
Dr.Jit can trace large and complicated programs with side effects, loops,
conditionals, polymorphic indirection, atomic memory operations, texture
fetches, hardware-accelerated ray tracing operations, etc. The principle is
always the same: the system captures what operations are needed to calculate a
result, postponing them for as long as possible.

Users of `JAX <https://github.com/google/jax>`__ may find this familiar: JAX
combines tracing with tensor-based optimizations for machine learning
workloads. JAX is generally amazing, but we find that its optimization often
tend to backfire in large non-ML workloads, causing `crashes or timeouts
<https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2022DrJit.pdf>`__.
Dr.Jit is tiny in comparison (~20K LOC for the compiler part versus > 1 million
for the JAX XLA backend) and what it does is simple: it really just captures
and later replays computation in parallel without trying to be overly clever
about it.

With this added context, let's revisit the previous example line by line to
examine the differences. The first one imports the library into an abbreviated
``dr`` namespace containing all functions.

.. code-block:: python

   import drjit as dr

Just below, there was a second
``import`` statement that requires an explanation:

.. code-block:: python

   from drjit.auto import Float

This line fetches an array type named ``Float`` representing a sequence of
single-precision numbers. The module ``drjit.auto`` refers to a computational
*backend* where computation is to be performed (e.g., the CPU, GPU)---``auto``
means that Dr.Jit should choose automatically.

This highlights another fundamental difference to NumPy, JAX, etc: these
frameworks all build on a single *nd-array* type (aka. *tensor*) to represent
data with different shapes and representations. In contrast, Dr.Jit is *not* a
tensor library. It uses types to emphasize these properties. For example, here
are just a few of the many different :ref:`array types <special_arrays>` provided by
the system:

- :py:class:`Int <drjit.auto.Int>` (or ``Int32``): a 32-bit signed integer.
- :py:class:`Complex2f64 <drjit.auto.Complex2f64>`: a 2D array with complex
  number semantics represented in double precision.
- :py:class:`Array3u64 <drjit.auto.Array3u64>`: 3D array of unsigned 64-bit integers.
- :py:class:`Matrix4f16 <drjit.auto.Matrix4f16>`: a half precision 4x4 matrix.

All of these are furthermore *arrays* of the concept they represent; the system
automatically vectorizes and parallelizes along this added dimension. Basically
you write code that "looks" like a scalar program, and Dr.Jit will efficiently
run it many times in parallel. In contrast to tensor-based systems, there is no
ambiguity about how this parallelization should take place. Because of its
typed nature, operations like :py:func:`drjit.linspace` take the data
type as a mandatory first argument.

Now let's look at how this idea of tracing computation to assemble a parallel
program works. Conceptually, a line like

.. code-block:: python

   a = dr.linspace(Float, 0, 1, 1000000)

can be thought of as expanding into device code equivalent to:

.. code-block:: python

   a = malloc(...) # reserve memory for output array 'a'

   # Parallel loop (SIMD and multi-core)
   for i in range(1000000):
       a[i] = i * 1.0 / 999999.0

Continuing the Python program simply appends more code to the loop body.
The next line of the original program was

.. code-block:: python

   b = dr.sin(a**2)

Since the we never end up accessing ``a`` explicitly, Dr.Jit generates a
program that avoids storing this variable:

.. code-block:: python

   b = malloc(...) # reserve memory for output array 'b'

   # Parallel loop (SIMD and multi-core)
   for i in range(1000000):
       a_temp = i * (1.0 / 999999.0)
       b[i] = sin(a_temp * a_temp)

The final line of the original program

.. code-block:: python

   print(dr.mean(b))

performs a reduction that adds values computed by different threads. At this
point, Dr.Jit compiles and launches a kernel containing the previous steps.

Metaprogramming
---------------

This was an example of an idea called *metaprogramming*: we are writing a
program that will write a program, and this second program is what ultimately
runs on the target device. Often, the program and metaprogram are essentially
the same, in which case the difference is very subtle.


Dr.Jit automatically takes care of memory allocations partitions code into
kernel launches, and pipes input/output data to these kernels. You can take
control of these steps if needed.

