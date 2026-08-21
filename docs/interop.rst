.. py:currentmodule:: drjit

.. _interop:

Interoperability
================

Dr.Jit can exchange data with various other array programming frameworks.
Currently, the following ones are officially supported:

- `NumPy <https://numpy.org>`__,
- `PyTorch <https://pytorch.org>`__,
- `JAX <https://jax.readthedocs.io/en/latest/installation.html>`__,
- `SymPy <https://www.sympy.org>`__ (symbolic computation, see :ref:`below
  <interop_sympy>`).

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
:py:func:`.torch() <ArrayBase.torch>`, or :py:func:`.jax() <ArrayBase.jax>`
members that always do the right thing for each target.

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

.. _interop_sympy:

SymPy
-----

Dr.Jit can also interoperate with `SymPy <https://www.sympy.org>`__, a
computer algebra system. This integration works differently from the
PyTorch/JAX/TensorFlow targets described above: rather than exchanging *data*
at runtime, the SymPy target operates at *compile time*. It converts Dr.Jit
types into SymPy symbols, runs the decorated function in SymPy to obtain an
expression, and then compiles that expression into Dr.Jit code.

.. code-block:: python

   import sympy as sp
   from drjit.cuda.ad import Float, Array3f

   @dr.wrap(source='drjit', target='sympy')
   def norm(v):
       return v.norm()

   result = norm(Array3f(3, 4, 0))  # Float(5.0)

The key idea is that the compiled result consists of *ordinary Dr.Jit
operations*. All of the usual Dr.Jit semantics apply: the code is traced,
parallelized across all array elements, and—if AD-enabled types are
used—automatically differentiable:

.. code-block:: pycon

   >>> x = Float(3.0)
   >>> dr.enable_grad(x)
   >>> @dr.wrap(source='drjit', target='sympy')
   ... def f(x):
   ...     return x**2 + 2*x + 1
   >>> y = f(x)
   >>> dr.backward(y)
   >>> x.grad
   [8]

Type promotion
^^^^^^^^^^^^^^

Dr.Jit array types are automatically mapped to their natural SymPy equivalents
when they enter the decorated function:

- Scalar types (:py:class:`Float <drjit.auto.ad.Float>`,
  :py:class:`UInt32 <drjit.auto.ad.UInt32>`, …) become ``sp.Symbol(real=True)``.
  Note that SymPy does not distinguish integer and floating-point symbols—both
  map to the same kind of symbol. Integer-specific operations like floor
  division and modulo will produce SymPy expressions (``floor(x/2)``,
  ``Mod(x, 3)``) that are correctly compiled to Dr.Jit code, but SymPy
  treats all values as real numbers.
- Arrays (:py:class:`Array2f <drjit.auto.ad.Array2f>`,
  :py:class:`Array3f <drjit.auto.ad.Array3f>`, :py:class:`ArrayXf
  <drjit.auto.ad.ArrayXf>`) become ``sp.Matrix`` column vectors.
- Matrix types (:py:class:`Matrix3f <drjit.auto.ad.Matrix3f>`,
  :py:class:`Matrix4f <drjit.auto.ad.Matrix4f>`, ...) become square
  ``sp.Matrix`` objects.
- Tensor types (:py:class:`TensorXf <drjit.auto.ad.TensorXf>`) cannot be used
  in the SymPy wrapper.

Caching
^^^^^^^

Compiled functions are cached at two levels. An in-memory cache, keyed by the
argument type signature, avoids recompilation when the same function is called
repeatedly with arguments of the same types. A disk cache in
``~/.drjit/sympy/`` persists compiled bytecode across interpreter restarts,
following Dr.Jit's kernel cache convention (``~/.drjit/``). To clear the
cache, simply delete that directory.

Nested calls
^^^^^^^^^^^^

Functions decorated with ``@dr.wrap(source='drjit', target='sympy')`` can call
each other. When a wrapped function is invoked during another wrapped
function's compilation, it runs in SymPy mode and its expression is inlined
into the outer function rather than triggering a separate compilation.

Limitations
^^^^^^^^^^^

The SymPy target has several limitations to be aware of:

- The decorated function must be **pure** and use only SymPy operations on its
  arguments. Since the arguments are SymPy symbols (not Dr.Jit arrays), calling
  Dr.Jit functions like :py:func:`dr.gather() <gather>`,
  :py:func:`dr.scatter() <scatter>` will not work.

- **No control flow on arguments.** Python ``if``/``else`` statements cannot
  branch on SymPy symbols because they are not booleans. Use ``sp.Piecewise``
  for conditional expressions instead.

- **Expression complexity.** SymPy manipulates expressions algebraically, which
  can become slow for large expressions. Operations like matrix inversion on
  matrices larger than about 4×4 may produce very large expressions and take a
  long time to compile. The compiled Dr.Jit code will be fast regardless, so
  this is purely a one-time compilation cost.

- **First-call overhead.** The first call to a wrapped function traces through
  SymPy, performs CSE optimization, and compiles the result. This can take
  noticeably longer than subsequent calls, which hit the cache.

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

A side remark is that clearing such allocations caches is an expensive
operation in any of these frameworks. You likely don't want to do so within a
performance-sensitive program region (e.g., an optimization loop).
