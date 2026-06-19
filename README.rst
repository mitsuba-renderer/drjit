Dr.Jit — A Just-In-Time-Compiler for Differentiable Rendering
=============================================================

.. image:: https://readthedocs.org/projects/drjit/badge/?version=latest
   :target: http://drjit.readthedocs.org/en/latest
   :alt: Documentation

.. image:: https://rgl-ci.epfl.ch/app/rest/builds/aggregated/strob:(buildType:(project:(id:DrJit)))/statusIcon.svg
   :target: https://rgl-ci.epfl.ch/project/DrJit?mode=trends&guest=1
   :alt: Continuous Integration

.. image:: https://img.shields.io/pypi/v/drjit.svg?color=green
   :target: https://pypi.org/pypi/drjit
   :alt: PyPI

.. raw:: html

    <p align="center">
    <img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-dark.svg#gh-light-mode-only" alt="Dr.Jit logo" width="500"/>
    <img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-light.svg#gh-dark-mode-only" alt="Dr.Jit logo" width="500"/>
    </p>

About this project
------------------

**Dr.Jit** is a *just-in-time* (JIT) compiler for ordinary and differentiable
computation. It was originally created as the numerical foundation of `Mitsuba
3 <https://github.com/mitsuba-renderer/mitsuba3>`__, a differentiable `Monte
Carlo <https://en.wikipedia.org/wiki/Monte_Carlo_method>`__ renderer. However,
*Dr.Jit* is a general-purpose tool that can also help with various other types
of embarrassingly parallel computation.

*Dr.Jit* helps with three steps:

- **Tracing and vectorization**: Dr.Jit executes arithmetic (e.g. ``a + b``) by
  recording it into a computation graph instead of performing the operation
  right away. It then *just-in-time* (JIT) compiles
  this graph into fused kernels targeting GPUs, using either `Metal
  <https://developer.apple.com/metal/>`__ on macOS or `CUDA
  <https://en.wikipedia.org/wiki/CUDA>`__ on other platforms. It can also
  target the host's CPU using vector instruction sets like `AVX512
  <https://en.wikipedia.org/wiki/AVX-512>`__ or `NEON
  <https://developer.arm.com/architectures/instruction-sets/simd-isas/neon>`__
  via `LLVM <https://llvm.org/>`__. The generated kernels are very efficient.
  Dr.Jit can also be used without JIT-compilation, which turns the project into
  a header-only vector library without external dependencies.

- **Differentiation**: If desired, Dr.Jit can compute derivatives using
  *automatic differentiation* (AD), using either `forward or reverse-mode
  accumulation <https://en.wikipedia.org/wiki/Automatic_differentiation>`__.
  Differentiation and tracing go hand-in-hand to produce specialized derivative
  evaluation code.

- **Python**: Dr.Jit types are accessible within C++17 and Python. Code can be
  developed in either language, or even both at once. Combinations of Python
  and C++ code can be jointly traced and differentiated.

Dr.Jit handles large programs with custom data structures, side effects, and
polymorphism. It includes a mathematical support library including
transcendental functions and types like vectors, matrices, complex numbers,
quaternions, etc.

Difference to machine learning frameworks
-----------------------------------------

Why did we create Dr.Jit, when dynamic derivative compilation is already
possible using Python-based ML frameworks like `JAX
<https://github.com/google/jax>`__, `Tensorflow <https://www.tensorflow.org>`__,
and `PyTorch <https://github.com/pytorch/pytorch>`__ along with backends like
`XLA <https://www.tensorflow.org/xla>`__ and `TorchScript
<https://pytorch.org/docs/stable/jit.html>`__?

The reason is related to the typical workloads: machine learning involves
smallish computation graphs that are, however, made of arithmetically intense
operations like convolutions, matrix multiplications, etc. The application
motivating *Dr.Jit* (differentiable rendering) creates giant and messy
computation graphs consisting of 100K to millions of "trivial" nodes
(elementary arithmetic operations). In our experience, ML compilation backends
use internal representations and optimization passes that are *too rich* for
this type of input, causing them to crash or time out during compilation. If
you have encountered such issues, you may find *Dr.Jit* useful.

Cloning
-------

Dr.Jit recursively depends on two other repositories: `nanobind
<https://github.com/wjakob/nanobind>`__ for Python bindings, and `drjit-core
<https://github.com/mitsuba-renderer/drjit-core>`__ providing core components of
the JIT-compiler.

To fetch the entire project including these dependencies, clone the project
using the ``--recursive`` flag as follows:

.. code-block:: bash

    $ git clone --recursive https://github.com/mitsuba-renderer/drjit

Documentation
-------------

Please see Dr.Jit's page on `readthedocs.io <https://drjit.readthedocs.io>`__
for example code and reference documentation.

References, citations
---------------------

Please see the paper `Dr.Jit: A Just-In-Time Compiler for Differentiable
Rendering <https://rgl.epfl.ch/publications/Jakob2022DrJit>`__ for the
nitty-gritty details and details on the problem motivating this project. There
is also a `video presentation
<https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2022DrJit.mp4>`__
explaining the design decisions at a higher level.

If you use *Dr.Jit* in your own research, please cite it using the following
BibTeX entry:

.. code-block:: bibtex

    @article{Jakob2022DrJit,
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
<https://ottojakob.com>`__. The "*Dr*." prefix simultaneously abbreviates
*differentiable rendering* with the stylized partial derivative *D*, while also
conveying a medical connotation that is emphasized by the `Rod of Asclepius
<https://en.wikipedia.org/wiki/Rod_of_Asclepius>`__. Differentiable rendering
algorithms are growing beyond our control in terms of conceptual and
implementation-level complexity. A doctor is a person, who can offer help in
such a time of great need. *Dr.Jit* tries to fill this role to improve the
well-being of differentiable rendering researchers.

*Dr.Jit* is the successor of the `Enoki
<https://github.com/mitsuba-renderer/enoki>`__ project, and its high-level API
still somewhat resembles that of Enoki. The system evolved towards a different
approach and has an all-new implementation, hence the decision to switch to a
different project name.
