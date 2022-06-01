.. _building-drjit:

Building Dr.Jit
--------------

The core parts of this library are of the *header-only* type requiring no
separate compilation step. If you are only working with the builtin array and
packet types, you can thus ignore all of the following. However, the following three optional
features depend on shared libraries that must be compiled ahead of time:

1. CUDA/LLVM arrays based on *just-in-time* (JIT) compilation
2. Automatic differentiation
3. Python bindings

Here is how you can clone the project and compile it with support for all of the
above using `ninja <https://ninja-build.org/>`_, though old-fashioned build
tools like GNU Make also work:

.. code-block:: text

    $ git clone --recursive https://github.com/mitsuba-renderer/drjit
    $ mkdir drjit/build
    $ cd drjit/build
    $ cmake -GNinja -DDRJIT_ENABLE_JIT=1 -DDRJIT_ENABLE_AUTODIFF=1 -DDRJIT_ENABLE_PYTHON=1 ..
    $ ninja

A visual frontend like `cmake-gui <https://cmake.org/runningcmake/>`_ can be
also used to enable the `DRJIT_ENABLE_JIT`, `DRJIT_ENABLE_AUTODIFF`, and
`DRJIT_ENABLE_PYTHON` flags as desired.
