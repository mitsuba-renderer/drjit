.. py:currentmodule:: drjit

.. _what_is_drjit:

What is Dr.Jit?
===============

Dr.Jit is a library to run massively parallel programs on the GPU or CPU, and
to optionally compute derivatives of such programs for gradient-based
optimization. It shares this purpose with `many <https://cupy.dev>`__
`currently <https://github.com/google/jax>`__ `existing
<https://www.tensorflow.org>`__ `languages <https://www.taichi-lang.org>`__
`and <https://github.com/NVIDIA/warp>`__ `tools <https://pytorch.org>`__.

Using Dr.Jit involves two steps:

1. You write some code and run it.

2. Dr.Jit captures what your program does, converts it into one or more
   parallel *kernels*, and then launches them on a compute accelerator.

**That's it**.  It doesn't do much, but it does this *very efficiently*.

Perhaps the most significant difference to the majority of existing tools is
that Dr.Jit is *not primarily* a machine learning library. While it does
provide support for neural network :ref:`evaluation and training <neural_nets>`,
it its sweet spot are non-neural programs characterized by *embarrassing
parallelism*---that is to say, programs with large data-parallel regions. A
good example of this are `Monte Carlo
<https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ methods with their
parallel sample evaluation (indeed, the reason why this project was originally
created was to provide the foundation of `Mitsuba 3
<https://mitsuba.readthedocs.io/en/latest/>`__, a differentiable Monte Carlo
renderer). Over time, Dr.Jit has become a general tool that supports many other
kinds of parallel workloads.

This documentation centers around the Python interface, but Dr.Jit can also be
used from C++. A separate :ref:`documentation section <cpp_iface>` explains how
to convert code between the two languages.

To install the latest version of Dr.Jit for Python, run the following shell command:

.. code-block:: bash

   $ python -m pip install --upgrade drjit

With that taken care of, let's see how Dr.Jit works in the context of a simple
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
   load and store memory-backed arrays. Accessing memory is slow, hence
   this turns into the main bottleneck rather than the actual math.

2. Dr.Jit *traces* the computation instead of executing it right away. This
   means that it *pretends* to execute until reaching the last line, at which
   point it launches a kernel combining all the collected operations. Not only
   does this avoid loading and storing intermediate results: it also makes it
   easy to parallelize the program on compute accelerators.

This is just a toy example, but the idea that it demonstrates is far more general.
Dr.Jit can trace large and complicated programs with side effects, loops,
conditionals, polymorphic indirection, atomic memory operations, texture
fetches, ray tracing operations, etc. The principle is always the same: the
system captures what operations are needed to calculate a result, postponing
them for as long as possible.

Users of `JAX <https://github.com/google/jax>`__ may find this familiar: JAX
combines tracing with tensor-based optimizations for machine learning
workloads. JAX is generally amazing, but we find that its optimization often
tend to backfire in large non-ML workloads, causing `crashes or timeouts
<https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2022DrJit.pdf>`__.
Dr.Jit is tiny compared to JAX (~20K LOC for the compiler part versus > 1 million
for the JAX XLA backend) and what it does is simple: it really just captures
and later replays computation in parallel without trying to be overly clever
about it.

With this added context, let's revisit the previous example to
examine the differences in more detail. The first line imports the library into
an abbreviated ``dr`` namespace containing all functions.

.. code-block:: python

   import drjit as dr

Just below, there is a second ``import`` statement that requires an
explanation:

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
are just a few of the :ref:`many different types <special_arrays>` provided by
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
ambiguity about how this parallelization should take place. Because of the
typed nature of Dr.Jit, many operations (e.g., :py:func:`drjit.linspace`)
take the desired return type as a mandatory first argument.

Let's now look at how *tracing* can be used to assemble a parallel
program. Conceptually, a line like

.. code-block:: python

   a = dr.linspace(Float, 0, 1, 1000000)

can be thought of as expanding into device code equivalent to:

.. code-block:: python

   a = malloc(...) # reserve memory for output array 'a'

   # Accelerate via multi-core + SIMD parallelism:
   for i in range(1000000):
       a[i] = i * (1.0 / 999999.0)

Recall that our original program contained a few more lines of code, so this
device program is still incomplete. Continuing execution in Python conceptually
*appends* further instructions to the parallel loop. The next line of the
original Python program was

.. code-block:: python

   b = dr.sin(a**2)

Since the we never end up accessing ``a`` explicitly, Dr.Jit generates a more
efficient device program that avoids storing this intermediate variable altogether:

.. code-block:: python

   b = malloc(...) # reserve memory for output array 'b'

   # Accelerate via multi-core + SIMD parallelism:
   for i in range(1000000):
       a_temp = i * (1.0 / 999999.0)
       b[i] = sin(a_temp * a_temp)

The final line of the original Python program

.. code-block:: python

   print(dr.mean(b))

performs a reduction that adds values computed by different threads. It is at
this point that Dr.Jit compiles and launches a kernel containing the previous
steps.

Metaprogramming
---------------

This was an example of more general design pattern called *metaprogramming*: we
wrote code in Python (called the *metaprogram*) that subsequently generated
*another* program, and this is what finally ran on the target device.

.. only:: not latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/pipeline-light.svg
     :class: only-light
     :align: center

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/pipeline-dark.svg
     :class: only-dark
     :align: center

.. only:: latex

   .. image:: https://rgl.s3.eu-central-1.amazonaws.com/media/uploads/wjakob/2024/06/pipeline-light.svg
     :align: center

Dr.Jit took care of partitioning this generated program into computational
units (called *kernels*) and piping inputs/outputs to them as needed. The
program and metaprogram often do the essentially same thing, in which case the
difference between the two can be quite subtle.

However, the program and metaprogram could also be different. For example,
let's modify the code so that it asks the user to enter a number on the
keyboard that is then used to to raise the integrand to a custom power:

.. code-block:: python
   :emphasize-lines: 3

   a = np.linspace(0, x, 1000000, dtype=np.float32)
   print('Enter exponent: ', end='')
   i = int(input())
   print(np.mean(np.sin(a**i)))

This extra step is only part of the metaprogram, but it is *not* part of the
generated device program. Dr.Jit only "sees" operations done on capitalized
types imported from a backend (e.g., ``Int``, ``Array3f``, etc.), and
everything else is just regular Python code that is interpreted as usual. This
means that the metaprogram compiles to different device programs depending on
what happens at runtime. This simple idea enables specialization of otherwise
very general programs to a given task or dataset to improve performance.

Backends
--------

Dr.Jit provides two backends with feature parity:

1. The `CUDA <https://en.wikipedia.org/wiki/CUDA>`__ backend targets `NVIDIA
   <https://www.nvidia.com>`__ GPUs with compute capability 5.0 or newer.
   You can explicitly request this backend by importing types from
   ``drjit.cuda`` or ``drjit.cuda.ad`` (add ``.ad`` if derivative computation is needed).

2. The `LLVM <https://llvm.org>`__ backend targets Intel (``x86_64``) and ARM
   (``aarch64``) CPUs. It parallelizes the program using the available CPU
   cores and vector instruction set extensions such as AVX, AVX512, NEON, etc.
   You can explicitly request this backend by importing types from
   ``drjit.llvm`` or ``drjit.llvm.ad`` (add ``.ad`` if derivative computation is needed).

   Note that LLVM >= 11.0 must be installed on your machine for this backend to
   be available. LLVM can be installed as follows:

   - **macOS**: Install `Homebrew <https://brew.sh>`__ and then enter the following
     command:

     .. code-block:: bash

        $ brew install llvm

   - **Linux**: Install the LLVM package using your distribution's package
     manager. On Debian/Ubuntu, you would, e.g., type:

     .. code-block:: bash

        $ sudo apt install llvm

   - **Windows**: Run one of the `official installers
     <https://github.com/llvm/llvm-project/releases/>`__, for example version `18.1.6
     <https://github.com/llvm/llvm-project/releases/download/llvmorg-18.1.6/LLVM-18.1.6-win64.exe>`__.

The previously mentioned ``drjit.auto`` and ``drjit.auto.ad`` backends redirect
to the CUDA backend if a compatible GPU was found, otherwise they fall back to
the LLVM backend.

Other backends may be added in the future.

Wrap-up
-------

This concludes our discussion of a first simple example. Subsequent parts of
this documentation explain how Dr.Jit generalizes to bigger programs:

1. :ref:`Basics <basics>`: a fast-paced review of the various ways in which
   Dr.Jit arrays can be created and modified.

2. :ref:`Control flow <cflow>`: how to trace ``while`` loops, ``if``
   statements, and polymorphic indirection.

3. :ref:`Evaluation <eval>`: Certain operations (such as printing the contents
   of an array) cannot be traced and trigger an *evaluation* step. We review
   what steps require evaluation, and how to tune this process.

4. :ref:`Automatic differentiation <autodiff>`: How to compute gradients of
   differentiable programs.

5. :ref:`Array types <special_arrays>`: A review of the various available
   array types.

6. :ref:`Interoperability <interop>`: How to integrate Dr.Jit with other
   frameworks (e.g. PyTorch or JAX) and backpropagate gradients through
   mixed-framework programs.

..
   Dr.Jit automatically takes care of memory allocations partitions code into
   kernel launches, and pipes input/output data to these kernels.
   When does Dr.Jit evaluate variables?
   Taking control of variable evaluation
   Type traits
   custom data structures
   random number generation
   debugging, printing, benchmarking, pitfalls
   how to clear the cache for benchmarking
   faq

