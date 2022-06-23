<p align="center">
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-dark.svg#gh-light-mode-only" alt="Dr.Jit logo" width="500"/>
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-light.svg#gh-dark-mode-only" alt="Dr.Jit logo" width="500"/>
</p>

# Dr.Jit — A Just-In-Time-Compiler for Differentiable Rendering

| Documentation   | Continuous Integration |       PyPI      |
|      :---:      |          :---:         |       :---:     |
| [![docs][1]][2] |    [![rgl-ci][3]][4]   | [![pypi][5]][6] |


[1]: https://readthedocs.org/projects/drjit/badge/?version=latest
[2]: http://drjit.readthedocs.org/en/latest
[3]: https://rgl-ci.epfl.ch/app/rest/builds/aggregated/strob:(buildType:(project:(id:DrJit)))/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/project/DrJit?mode=trends&guest=1
[5]: https://img.shields.io/pypi/v/drjit.svg
[6]: https://pypi.org/pypi/drjit

## Introduction

**Dr.Jit** is a _just-in-time_ (JIT) compiler for ordinary and differentiable
computation. It was originally created as the numerical foundation of [Mitsuba
3](https://github.com/mitsuba-renderer/mitsuba3), a differentiable [Monte
Carlo](https://en.wikipedia.org/wiki/Monte_Carlo_method) renderer. However,
_Dr.Jit_ is a general-purpose tool that can also help with various other types
of embarrassingly parallel computation.

_Dr.Jit_ principally facilitates three steps

- **Vectorization and tracing**: When _Dr.Jit_ encounters an arithmetic
  operation (e.g. an addition `a + b`) it does not execute it right away:
  instead, it remembers that an addition will be needed at some later point by
  recording it into a graph representation (this is called _tracing_).
  Eventually, it will _just-in-time_ (JIT)-compile the recorded operations into
  a _fused_ kernel using either [LLVM](https://en.wikipedia.org/wiki/LLVM) (for
  CPUs) or [CUDA](https://en.wikipedia.org/wiki/CUDA) (for GPUs). The values
  `a` and `b` will typically be arrays with many elements, and the system thus
  parallelizes the evaluation using both multi-core parallelism and vector
  instruction sets like [AVX512](https://en.wikipedia.org/wiki/AVX-512) or [ARM
  Neon](https://developer.arm.com/architectures/instruction-sets/simd-isas/neon).

  _Dr.Jit_ is ideal for Monte Carlo methods, where the same computation must be
  repeated for millions of random samples. _Dr.Jit_ dynamically generates
  specialized parallel code for various target platforms that would be
  challenging maintain using traditional software development techniques.

  As a fallback, Dr.Jit can also be used without JIT-compilation, which turns
  the project into a header-only vector library without external dependencies.

- **Differentiation**: If desired, _Dr.Jit_ can compute derivatives using
  _automatic differentiation_ (AD), using either [forward or reverse-mode
  accumulation](https://en.wikipedia.org/wiki/Automatic_differentiation).
  Differentiation and tracing go hand-in-hand to produce specialized derivative
  evaluation code.

- **Python**: _Dr.Jit_ types are accessible within C++17 and Python. Code can be
  developed in either language, or even both at once. Combinations of Python
  and C++ code can be jointly traced and differentiated.

_Dr.Jit_ handles large programs with custom data structures, side effects,
virtual method calls, lambda functions, loadable modules. It includes a
mathematical support library including transcendental functions and types like
vectors, matrices, complex numbers, quaternions, etc.

## Difference to machine learning frameworks

Why did we create _Dr.Jit_, when dynamic derivative compilation is already
possible using Python-based ML frameworks like
[JAX](https://github.com/google/jax), [Tensorflow](https://www.tensorflow.org),
and [PyTorch](https://github.com/pytorch/pytorch) along with backends like
[XLA](https://www.tensorflow.org/xla) and
[TorchScript](https://pytorch.org/docs/stable/jit.html)? 

The reason is related to the typical workloads: machine learning involves
small-ish computation graphs that are, however, made of arithmetically intense
operations like convolutions, matrix multiplications, etc. The application
motivating _Dr.Jit_ (differentiable rendering) creates giant and messy
computation graphs consisting of 100K to millions of "trivial" nodes
(elementary arithmetic operations). In our experience, ML compilation backends
use internal representations and optimization passes that are _too rich_ for
this type of input, causing them to crash or time out during compilation. If
you have encountered such issues, you may find _Dr.Jit_ useful.

## Cloning

Dr.Jit recursively depends on two other repositories:
[pybind11](https://github.com/pybind/pybind11) for Python bindings, and
[drjit-core](https://github.com/mitsuba-renderer/drjit-core) providing the JIT
compiler.

To fetch the entire project including these dependencies, clone the project
using the ``--recursive`` flag as follows:

```bash
$ git clone --recursive https://github.com/mitsuba-renderer/drjit
```

## Documentation

Please see Dr.Jit's page on [readthedocs](https://drjit.readthedocs.io) for
example code and reference documentation.

## References, citing

Please see the paper [Dr.Jit: A Just-In-Time Compiler for Differentiable
Rendering](https://rgl.epfl.ch/publications/Jakob2020DrJit) for the
nitty-gritty details and details on the problem motivating this project. There
is also a [video
presentation](https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2020DrJit.mp4).

If you use _Dr.Jit_ in your own research, please cite it using the following
BibTeX entry:
```bibtex
@article{Jakob2020DrJit,
  author = {Wenzel Jakob and Sébastien Speierer and Nicolas Roussel and Delio Vicini},
  title = {Dr.Jit: A Just-In-Time Compiler for Differentiable Rendering},
  journal = {Transactions on Graphics (Proceedings of SIGGRAPH)},
  volume = {41},
  number = {4},
  year = {2022},
  month = jul,
  doi = {10.1145/3528223.3530099}
}
```

## Logo and history

The _Dr.Jit_ logo was generously created by [Otto
Jakob](https://ottojakob.com). The "_Dr_." prefix simultaneously abbreviates
_differentiable rendering_ with the stylized partial derivative _D_, while also
conveying a medical connotation that is emphasized by the [Rod of
Asclepius](https://en.wikipedia.org/wiki/Rod_of_Asclepius). Differentiable
rendering algorithms are growing beyond our control in terms of conceptual and
implementation-level complexity. A doctor is a person, who can offer help in
such a time of great need. _Dr.Jit_ tries to fill this role to to improve the
well-being of differentiable rendering researchers.

_Dr.Jit_ is the successor of the
[Enoki](https://github.com/mitsuba-renderer/enoki) project, and its high-level
API still somewhat resembles that of Enoki. The system evolved towards a
different approach and has an all-new implementation, hence the decision
to switch to a different name.
