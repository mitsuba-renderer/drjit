.. image:: ../ext/drjit-core/resources/drjit-logo-dark.svg
  :width: 500
  :align: center
  :alt: Dr.Jit logo
  :class: only-light

.. image:: ../ext/drjit-core/resources/drjit-logo-light.svg
  :width: 500
  :align: center
  :alt: Dr.Jit logo
  :class: only-dark

About this project
==================

**Dr.Jit** is a *just-in-time* (JIT) compiler for ordinary and differentiable
computation. It was originally created as the numerical foundation of `Mitsuba
3 <https://github.com/mitsuba-renderer/mitsuba3>`_, a differentiable `Monte
Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`_ renderer. However,
*Dr.Jit* is a general-purpose tool that can also help with various other types
of embarrassingly parallel computation.

*Dr.Jit* principally facilitates three steps:

- **Vectorization and tracing**: When *Dr.Jit* encounters an operation (e.g. an
  addition ``a + b``) it does not execute it right away: instead, it remembers
  that an addition will be needed at some later point by recording it into a
  graph representation (this is called *tracing*). Eventually, it will
  *just-in-time* (JIT) compile the recorded operations into a *fused* kernel
  using either `LLVM <https://en.wikipedia.org/wiki/LLVM>`_ (when targeting the
  CPU) or `CUDA <https://en.wikipedia.org/wiki/CUDA>`_ (when targeting the
  GPU). The values ``a`` and ``b`` will typically be arrays with many elements,
  and the system parallelizes their evaluation using multi-core parallelism and
  vector instruction sets like `AVX512
  <https://en.wikipedia.org/wiki/AVX-512>`_ or `ARM Neon
  <https://developer.arm.com/architectures/instruction-sets/simd-isas/neon>`_.

  *Dr.Jit* is ideal for Monte Carlo methods, where the same computation must be
  repeated for millions of random samples. *Dr.Jit* dynamically generates
  specialized parallel code for the target platform.
  As a fallback, Dr.Jit can also be used without JIT-compilation, which turns
  the project into a header-only vector library without external dependencies.

- **Differentiation**: If desired, *Dr.Jit* can compute derivatives using
  *automatic differentiation* (AD), using either `forward or reverse-mode
  accumulation <https://en.wikipedia.org/wiki/Automatic_differentiation>`_.
  Differentiation and tracing go hand-in-hand to produce specialized derivative
  evaluation code.

- **Python**: *Dr.Jit* types are accessible within C++17 and Python. Code can be
  developed in either language, or even both at once. Combinations of Python
  and C++ code can be jointly traced and differentiated.

*Dr.Jit* handles large programs with custom data structures, side effects, and
polymorphism. It includes a mathematical support library including
transcendental functions and types like vectors, matrices, complex numbers,
quaternions, etc.

Difference to machine learning frameworks
-----------------------------------------

Why did we create *Dr.Jit*, when dynamic derivative compilation is already
possible using Python-based ML frameworks like `JAX
<https://github.com/google/jax>`_, `Tensorflow <https://www.tensorflow.org>`_,
and `PyTorch <https://github.com/pytorch/pytorch>`_ along with backends like
`XLA <https://www.tensorflow.org/xla>`_ and `TorchScript
<https://pytorch.org/docs/stable/jit.html>`_?

The reason is related to the typical workloads: machine learning involves
small-ish computation graphs that are, however, made of arithmetically intense
operations like convolutions, matrix multiplications, etc. The application
motivating *Dr.Jit* (differentiable rendering) creates giant and messy
computation graphs consisting of 100K to millions of "trivial" nodes
(elementary arithmetic operations). In our experience, ML compilation backends
use internal representations and optimization passes that are *too rich* for
this type of input, causing them to crash or time out during compilation. If
you have encountered such issues, you may find *Dr.Jit* useful.

Cloning
-------

Dr.Jit recursively depends on two other repositories: `pybind11
<https://github.com/pybind/pybind11>`_ for Python bindings, and `drjit-core
<https://github.com/mitsuba-renderer/drjit-core>`_ providing core components of
the JIT-compiler.

To fetch the entire project including these dependencies, clone the project
using the ``--recursive`` flag as follows:

.. code-block:: bash

    $ git clone --recursive https://github.com/mitsuba-renderer/drjit

References, citations
---------------------

Please see the paper `Dr.Jit: A Just-In-Time Compiler for Differentiable
Rendering <https://rgl.epfl.ch/publications/Jakob2020DrJit>`_ for the
nitty-gritty details and details on the problem motivating this project. There
is also a `video presentation
<https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2020DrJit.mp4>`_
explaining the design decisions at a higher level.

If you use *Dr.Jit* in your own research, please cite it using the following
BibTeX entry:

.. code-block:: bibtex

    @article{Jakob2020DrJit,
      author = {Wenzel Jakob and S{\'e}bastien Speierer and Nicolas Roussel and Delio Vicini},
      title = {Dr.Jit: A Just-In-Time Compiler for Differentiable Rendering},
      journal = {Transactions on Graphics (Proceedings of SIGGRAPH)},
      volume = {41},
      number = {4},
      year = {2022},
      month = jul,
      doi = {10.1145/3528223.3530099}
    }

Logo and history
----------------

The *Dr.Jit* logo was generously created by `Otto Jakob
<https://ottojakob.com>`_. The "*Dr*." prefix simultaneously abbreviates
*differentiable rendering* with the stylized partial derivative *D*, while also
conveying a medical connotation that is emphasized by the `Rod of Asclepius
<https://en.wikipedia.org/wiki/Rod_of_Asclepius>`_. Differentiable rendering
algorithms are growing beyond our control in terms of conceptual and
implementation-level complexity. A doctor is a person, who can offer help in
such a time of great need. *Dr.Jit* tries to fill this role to to improve the
well-being of differentiable rendering researchers.

*Dr.Jit* is the successor of the `Enoki
<https://github.com/mitsuba-renderer/enoki>`_ project, and its high-level API
still somewhat resembles that of Enoki. The system evolved towards a different
approach and has an all-new implementation, hence the decision to switch to a
different project name.
