<p align="center">
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-dark.svg#gh-light-mode-only" alt="Dr.Jit logo" width="500"/>
<img src="https://github.com/mitsuba-renderer/drjit-core/raw/master/resources/drjit-logo-light.svg#gh-dark-mode-only" alt="Dr.Jit logo" width="500"/>
</p>

# Dr.Jit — A Just-In-Time-Compiler for Differentiable Rendering

| Documentation   | Continuous Integration |       PyPI      |
|      :---:      |          :---:         |       :---:     |
| [![docs][1]][2] |    [![rgl-ci][3]][4]   | [![pypi][5]][6] |


[1]: https://readthedocs.org/projects/drjit-wjakob/badge/?version=latest
[2]: http://drjit-wjakob.readthedocs.org/en/latest
[3]: https://rgl-ci.epfl.ch/app/rest/builds/buildType(id:DrJit_Build)/statusIcon.svg
[4]: https://rgl-ci.epfl.ch/viewType.html?buildTypeId=DrJit_Build&guest=1
[5]: https://img.shields.io/pypi/v/drjit.svg
[6]: https://pypi.org/pypi/drjit

## Introduction

**Dr.Jit** is a C++17 template library that dramatically simplifies several
types of program transformations that are often applied to numerical software:

* **Vectorization**. Converting a scalar program into into one that
  simultaneously processes many inputs to leverage parallelism on modern
  processor architectures. Here, "many" could refer to a packet with 16 values
  (AVX512) or millions of entries processed on a GPU.

* **Forward and reverse-mode automatic differentiation (AD)**. Computing
  derivatives of an arbitrary computation with respect to its inputs or
  outputs.

* **Python bindings**. Exposing C++ code within a Python environment so that it
  becomes usable along widely used software from this ecosystem (NumPy,
  Matplotlib, PyTorch, etc).

All features are "opt-in" and activated by including specific header files,
keeping compilation times short.

Algorithms designed using Dr.Jit are expressed a generic way (using *templates*)
and specialized to specific requirements by "lifting" them onto a computational
backend. By stacking up such transformations, it becomes possible to create
elaborate architectures that would be very tedious to develop by hand—for
example, a GPU implementation of an algorithm that can be differentiated
end-to-end along with other code running in an interactive Python session.
The following backends are currently included:

* **SIMD**. Dr.Jit can express computation using efficient SIMD instructions
  available on modern CPUs (AVX512, AVX2, AVX, and SSE4.2). In this mode, Dr.Jit
  processes packets (typically 4, 8, or 16 elements) and turns into a pure
  header-file library (i.e. no extra compilation steps needed for Dr.Jit
  itself.)

* **JIT Compiler (CUDA)**. Dr.Jit includes a just-in-time compiler that
  dynamically transforms algorithms into efficient kernels that run on NVIDIA
  GPUs. Dr.Jit is able to do this without any compile-time dependencies on the
  usual CUDA toolchain (``nvcc``, etc.): it simply looks for the graphics
  driver at runtime and talks to it using NVIDIA's *Parallel Thread Execution*
  (PTX) intermediate language.

* **JIT Compiler (LLVM)**. The same JIT compiler can also generate vectorized
  CPU kernels via LLVM's intermediate representation. The difference to the
  SIMD mode mentioned above is that these kernels operate on large arrays (e.g.
  millions of entries), and that computation is automatically partitioned over
  all cores in your system. In essence, this mode enables using your CPU as if
  it was a GPU. Once more, Dr.Jit can do this without a any build-time
  dependency on LLVM, which is detected dynamically at runtime (any non-ancient
  version > 7.0 works).

* **Automatic Differentiation**. The above transformations can all be combined
  with automatic differentiation (AD) in either forward and reverse mode to
  compute high-dimensional derivatives, e.g., for gradient-based optimization.

* **Fallback option**. Dr.Jit is designed so that programs can also be lifted onto
  simple builtin types (``float``, ``int``, etc.) and retain their
  functionality. In this case, the algorithm will behave like a standard C++
  implementation.

In addition to the above, Dr.Jit is designed to be

* **Unobtrusive**. Code written using Dr.Jit's abstractions must remain easy
  to read and maintain.

* **Structured**. Dr.Jit handles complex programs with
  custom data structures, virtual method calls, lambda functions, loadable
  modules, and many other modern C++ features. Tedious steps like conversion of
  data structures into a *Structure of Arrays* (SoA) format are offloaded onto
  the C++ type system.

* **Complete**. Dr.Jit ships with a library of special functions and data
  structures that facilitate implementation of numerical code (vectors,
  matrices, complex numbers, quaternions, etc.).

* **Non-viral**. Dr.Jit is licensed under the terms of the 3-clause BSD license.

To the author's knowledge, nothing quite like it exists, although there are of
course many vectorization techniques (Autovectorization, expression templates),
frameworks (Eigen, XLA, Numba, Agner Fog's vector classes) and AD tools
(PyTorch, Tensorflow, Jax) that provide subsets of the above functionality.

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

Please see Dr.Jit's page on
[readthedocs](https://drjit.readthedocs.io/en/master/demo.html) for example
code and reference documentation.
