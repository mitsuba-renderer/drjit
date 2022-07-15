.. _firststeps-cpp:

First steps in C++
==================

.. code-block:: bash

   $ git clone --recursive https://github.com/mitsuba-renderer/drjit

.. code-block:: bash

   $ cd drjit
   $ cmake -S . -B build
   $ cmake --build build

This will by default build everything (JIT, automatic differentiation library,
Python bindings). Alternatively, the previous ``cmake`` command can be invoked
with extra parameters for more fine-grained control:

.. code-block:: bash

   $ cmake -S . -B build -DDRJIT_ENABLE_JIT=1 -DDRJIT_ENABLE_AUTODIFF=1 -DDRJIT_ENABLE_PYTHON=1


.. _cpp-iface:

C++ interface
-------------

TBD
