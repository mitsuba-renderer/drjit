.. py:currentmodule:: drjit

.. _interop:

Interoperability
================

Dr.Jit can exchange data with various other array programming frameworks.
Currently, the following ones are officially supported:

- `NumPy <https://numpy.org>`__,
- `PyTorch <https://pytorch.org>`__,
- `JAX <https://jax.readthedocs.io/en/latest/installation.html>`__,
- `TensorFlow <https://tensorflow.org>`__. 

There isn't much to it: given an input array from another framework, simply
pass it to the constructor of the Dr.Jit array or tensor type you wish to
construct.

.. code-block:: python

   import numpy as np
   from drjit.llvm import Array3f, TensorXf

   a = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)

   # Load into a dynamic 3D array with shape (3, N)
   b = Array3f(a)

   # Load into a generic tensor, can represent any input shape.
   c = TensorXf(a)

Dr.Jit uses a *zero-copy* strategy whenever possible, by simply exposing the
existing data using a different type. This is possible thanks to the `DLPack
<https://github.com/dmlc/dlpack>`__ data exchange protocol.

The reverse direction is in principle analogous, though not all frameworks
correctly detect that Dr.Jit arrays implements the DLPack specification. To
avoid unnecessary copies, use the :py:func:`.numpy() <ArrayBase.numpy>`,
:py:func:`.torch() <ArrayBase.torch>`, :py:func:`.jax() <ArrayBase.jax>`, or
:py:func:`.tf() <ArrayBase.tf>` members that always do the right thing for each
target.

.. code-block:: python

   b_np    = b.numpy()
   c_torch = c.torch()

Note that these operations evaluate the input Dr.Jit array if this has not
already been done before.

.. _interop_ad:

Differentiability
-----------------

The former operations only convert data but do not track derivatives.

.. code-block:: pycon

   >>> import torch, drjit as dr
   >>> from drjit.cuda.ad import Float
   >>> a = torch.tensor([1.0], requires_grad=True)
   >>> b = drjit.llvm.Float(a)
   >>> dr.grad_enabled(b)
   False
   >>> :-(

Multi-framework differentiation requires a clear interface within the AD system
of each participant. The :py:func:`@drjit.wrap <wrap>` decorator provides such
an interface.

This decorator can either expose a differentiable Dr.Jit function in another
framework or the reverse, and it supports both forward and reverse-mode
differentiation.

You can combine it with further decorators such as :py:func:`@drjit.syntax <syntax>` and
use the full set of symbolic or evaluated operations available in normal Dr.Jit
programs.

Below is an example computing the derivative of a Dr.Jit subroutine within a
larger PyTorch program:

.. code-block:: pycon

   >>> from drjit.cuda import Int
   >>> @dr.wrap(source='torch', target='drjit')
   ... @dr.syntax
   ... def pow2(n, x):
   ...    i, n = Int(0), Int(n)
   ...    while dr.hint(i < n, max_iterations=10):
   ...        x *= x
   ...        i += 1
   ...    return x
   ...
   >>> n = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
   >>> x = torch.tensor([4, 4, 4, 4], dtype=torch.float32, requires_grad=True)
   >>> y = pow2(n, x)
   >>> print(y)
   tensor([4.0000e+00, 1.6000e+01, 2.5600e+02, 6.5536e+04],
          grad_fn=<TorchWrapperBackward>)
   >>> y.sum().backward()
   >>> print(x.grad)
   tensor([1.0000e+00, 8.0000e+00, 2.5600e+02, 1.3107e+05])

See the documentation of :py:func:`@drjit.wrap <wrap>` for further details.

.. _interop_caveats:

Caveats
-------

Some frameworks are *extremely greedy* in their use of resources especially
when working with CUDA. They must be reined in to build software that
effectively combines multiple frameworks. This is where things stand as of
early 2024:

- Dr.Jit behaves nicely and only allocates memory on demand.

- PyTorch behaves nicely and only allocates memory on demand.

- JAX `preallocates 75% of the total GPU memory
  <https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html>`__ when the
  first JAX operation is run, which only leaves a small remainder for Dr.Jit
  and the operating system.

  To disable this behavior, you must set the environment variable
  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` *before launching Python or the
  Jupyter notebook*.

  Alternatively, you can run

  .. code-block:: python

     import os
     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

  before importing JAX.

- TensorFlow preallocates `"nearly all" of the GPU memory
  <https://www.tensorflow.org/guide/gpu>`__ visible to the process,
  which will likely prevent Dr.Jit from functioning correctly.

  To disable this behavior, you must call the `set_memory_growth
  <https://www.tensorflow.org/api_docs/python/tf/config/experimental/set_memory_growth>`__
  function before using any other TensorFlow API, which will cause it to
  use a less aggressive on-demand allocation policy.

Once they allocate memory, these frameworks also *keep it to themselves*: for
example, if your program temporarily creates a huge PyTorch tensor that uses
nearly all GPU memory, then that memory is blocked from further use in Dr.Jit.

This behavior is technically justified: allocating and releasing memory is a
rather slow operation especially on CUDA, so every framework (*including*
Dr.Jit) implements some type of internal memory cache. These caches can be
manually freed if necessary. Here is how this can be accomplished:

- Dr.Jit: call :py:func:`drjit.flush_malloc_cache()`.

- PyTorch: call `torch.cuda.empty_cache()
  <https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html>`__.

- JAX: there is `no way to do it
  <https://github.com/google/jax/issues/1222>`__ besides setting
  ``XLA_PYTHON_CLIENT_ALLOCATOR=platform`` *before launching Python or
  the Jupyter notebook* or setting the variable via ``os.environ`` at
  the beginning of the program/Jupyter notebook. This disables the JAX
  memory cache, which may have a negative impact on performance.

- TensorFlow: there is `no way to do it
  <https://github.com/tensorflow/tensorflow/issues/36465>`__ besides
  setting ``TF_GPU_ALLOCATOR=cuda_malloc_async`` *before launching
  Python or the Jupyter notebook* or setting the variable via
  ``os.environ`` at the beginning of the program/Jupyter notebook. This
  disables the TensorFlow memory cache, which may have a negative impact
  on performance.

A side remark is that clearing such allocations caches is an expensive
operation in any of these frameworks. You likely don't want to do so within a
performance-sensitive program region (e.g., an optimization loop).
