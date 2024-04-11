.. py:currentmodule:: drjit

.. _changelog:

Changelog
#########

DrJit 1.0.0 (TBA)
-----------------

The 1.0 release of Dr.Jit marks major new phase of this project. We addressed
long-standing limitations and thoroughly documented every part of Dr.Jit.
Due to the magnitude of the changes, some incompatibilities are unavoidable:
bullet points with an exclamation mark highlight changes with an impact on
source-level compatibility.

Here is what's new:

- **Python bindings**: Dr.Jit comes with an all-new set of Python bindings
  created using the `nanobind <https://github.com/wjakob/nanobind>`__ library.
  This has several consequences:

  - Tracing Dr.Jit code written in Python is now *significantly* faster (we've
    observed speedups by a factor of ~10-20×). This should help in situations
    where performance is limited by tracing rather than kernel evaluation.

  - Dr.Jit can now target Python 3.12's `stable ABI
    <https://docs.python.org/3/c-api/stable.html#stable-abi>`__. This means
    that binary wheels will work on future versions of Python without
    recompilation.

  - thorough type annotations enable static type checking and better code
    completion in editors like `VS Code <https://code.visualstudio.com>`__.

- **Natural syntax**: vectorized loops and conditionals can now be expressed
  using natural Python syntax. To see what this means, consider the following
  function that computes an integer power of a floating point array:

  .. code-block:: python

     from drjit.cuda import Int, Float

     @dr.syntax
     def ipow(x: Float, n: Int):
         result = Float(1)

         while n != 0:
             if n & 1 != 0:
                 result *= x
             x *= x
             n >>= 1

         return result

  Given that this function processes arrays, we expect that condition of the
  ``if`` statement may disagree among elements. Also, each element may need a
  different number of loop iterations. However, such component-wise
  conditionals and loops aren't supported by stock Python. Previously, Dr.Jit
  provided ways of expressing such code using masking and a special
  ``dr.cuda.Loop`` object, but this was rather tedious.

  The new :py:func:`@drjit.syntax <drjit.syntax>` decorator greatly simplifies
  the development of programs with complex control flow. It performs an
  automatic source code transformation that replaces conditionals and loops
  with array-compatible variants (:py:func:`drjit.while_loop`,
  :py:func:`drjit.if_stmt`). The transformation leaves everything else as-is,
  including line number information that is relevant for debugging.

- **Differentiable control flow**: symbolic control flow constructs (loops)
  previously failed with an error message when they detected differentiable
  variables. All symbolic operations (loops, function calls, and conditionals)
  now support differentiation in forward and reverse modes.

- **Documentation**: every Dr.Jit function now comes with extensive reference
  documentation that clearly specifies its behavior and accepted inputs. The
  behavior with respect to tensors and arbitrary object graphs (referred to as
  :ref:`"PyTrees" <pytrees>`) was made consistent.

- **Half-precision arithmetic**: Dr.Jit now provides ``float16``-valued arrays
  and tensors on both the LLVM and CUDA backends (e.g.,
  :py:class:`drjit.cuda.ad.TensorXf16` or :py:class:`drjit.llvm.Float16`).

- **Mixed-precision optimization**: Dr.Jit now maintains one global AD graph
  for all variables, enabling differentiation of computation combining single-,
  double, and half precision variables. Previously, there was a separate graph
  per type, and gradients did not propagate through casts between them.

- **Multi-framework computations**: The :py:func:`@drjit.wrap <drjit.wrap>` decorator
  provides a differentiable bridge to other AD frameworks. In this new release
  of Dr.Jit, its capabilities were significantly revamped. Besides PyTorch, it
  now also supports JAX, and it consistently handles both forward and backward
  derivatives. The new interface admits functions with arbitrary
  fixed/variable-length positional and keyword arguments containing arbitrary
  PyTrees of differentiable and non-differentiable arrays, tensors, etc.

- **Debug mode**: A new debug validation mode (:py:attr:`drjit.JitFlag.Debug`)
  inserts a number of additional checks to identify sources of undefined
  behavior. Enable it to catch out-of-bounds reads, writes, and calls to
  undefined callables. Such operations will trigger a warning that includes the
  responsible source code location.

  The following built-in assertion checks are also active in debug mode. They
  support both regular and symbolic inputs in a consistent fashion.

  - :py:func:`drjit.assert_true`,
  - :py:func:`drjit.assert_false`,
  - :py:func:`drjit.assert_equal`.

- **Symbolic print statement**: A new high-level *symbolic* print operation
  :py:func:`drjit.print` enables deferred printing from any symbolic context
  (i.e., within symbolic loops, conditionals, and function calls). It is
  compatible with Jupyter notebooks and displays arbitrary :ref:`PyTrees
  <pytrees>` in a structured manner. This operation replaces the function
  ``drjit.print_async()`` provided in previous releases.

- **Swizzling**: swizzle access and assignment operator are now provided. You
  can use them to arbitrarily reorder, grow, or shrink the input array.

  .. code-block:: python

     a = Array4f(...), b = Array2f(...)
     a.xyw = a.xzy + b.xyx

- Reductions operations previously existed as *ordinary* (e.g.,
  :py:func:`drjit.all`) and *nested* (e.g. ``drjit.all_nested``) variants. Both
  are now subsumed by an optional ``axis`` argument similar to how this works
  in other array programming frameworks like NumPy.

  The reduction functions (:py:func:`drjit.all` :py:func:`drjit.any`,
  :py:func:`drjit.sum`, :py:func:`drjit.prod`, :py:func:`drjit.min`,
  :py:func:`drjit.max`) reduce over the outermost axis (``axis=0``) by default,
  Specify ``axis=None`` to reduce the entire array recursively analogous to the
  previous nested reduction.

  Aliases for the ``_nested`` function variants still exist to facilitate
  porting but are deprecated and will be removed in a future release.

- The performance of atomic scatter-reductions
  (:py:func:`drjit.scatter_reduce`, :py:func:`drjit.scatter_add`) has been
  *significantly* improved. Both functions now provide a ``mode=`` parameter to
  select between different implementation strategies. The new strategy
  :py:attr:`drjit.ReduceMode.Expand` offers a speedup of *over 10×* on the LLVM
  backend compared to the previously used local reduction strategy.
  Furthermore, improved code generation for :py:attr:`drjit.ReduceMode.Local`
  brings a roughly 20-40% speedup on the CUDA backend. See the documentation
  section on :ref:`atomic reductions <reduce-local>` for details and
  benchmarks with plots.

- **DDA**: a newly added *digital differential analyzer*
  (:py:func:`drjit.dda.dda`) can be used to traverse the intersection of a ray
  segment and an n-dimensional grid. The function :py:func`drjit.dda.integrate`
  builds on this functionality to compute analytic differentiable line
  integrals of bi- and trilinear interpolants.

- **Loop compression**: the implementation of evaluated loops (previously
  referred to as wavefront mode) visits all entries of the loop state variables
  at every iteration, even when most of them have already finished executing the
  loop. Dr.Jit now provides an optional ``compress=True`` parameter in
  :py:func:`drjit.while_loop` to prune away inactive entries and accelerate
  later loop iterations.

- The new release has a strong focus on error resilience and leak avoidance.
  Exceptions raised in custom operations, function dispatch, symbolic loops,
  etc., should not cause failures or leaks. Both Dr.Jit and nanobind are very
  noisy if they detect that objects are still alive when the Python interpreter
  shuts down.

- **Terminology cleanup**: Dr.Jit has two main ways of capturing control flow
  (conditionals, loops, function calls): it can evaluate each possible outcome
  eagerly, causing it to launch many small kernels (this is now called:
  *evaluated mode*). The second is to capture control flow and merge it into
  the same kernel (this is now called *symbolic mode*). Previously,
  inconsistent and rendering-specific terminology was used to refer to these
  two concepts.

  Several entries of the :py:class:`drjit.JitFlag` enumeration were renamed to
  reflect this fact (for example, ``drjit.JitFlag.VCallRecord`` is now called
  :py:attr:`drjit.JitFlag.SymbolicCalls`). The former entries still exist as
  (deprecated) aliases.

- **Index reuse**: variable indices (:py:attr:`drjit.ArrayBase.index`,
  :py:attr:`drjit.ArrayBase.index_ad`) used to monotonically increase as
  variables were being created. Internally, multiple hash tables were needed to
  associate these ever-growing indices with locations in an internal variable
  array, which which had a surprisingly large impact on tracing performance.
  Dr.Jit removes this mapping both at the AD and JIT levels and eagerly reuses
  variable indices.

  This change can be inconvenient for low-level debugging, where it was often
  helpful to inspect the history of operations involving a particular variable
  by searching a trace dump for mentions of its variable index. Such trace dumps
  were generated by setting :py:func:`drjit.set_log_level` to a level of
  :py:attr:`drjit.LogLevel.Debug` or even :py:attr:`drjit.LogLevel.Trace`. A
  new flag was introduced to completely disable variable reuse and help such
  debugging workflows:

  .. code-block:: python

     dr.set_flag(dr.JitFlag.ReuseIndices, False)

  Note that this causes the internal variable array to steadily grow, hence
  this feature should only be used for brief debugging sessions.

- The :py:func:`drjit.empty` function used to immediate allocate an array of
  the desired shape (compared to, say, :py:func:`drjit.zero` which creates a
  literal constant array that consumes no device memory). Users found this
  surprising, so the behavior was changed so that :py:func:`drjit.empty`
  similarly delays allocation.

- **Fast math**: Dr.Jit now has an optimization flag named
  :py:attr:`drjit.JitFlag.FastMath` that is reminiscent of ``-ffast-math`` in
  C/C++ compilers. It enables program simplifications such as ``a*0 == 0`` that
  are not always valid. For example, equality in this example breaks when ``a``
  is infinite or equal to NaN. The flag is on by default since it can
  considerably improve performance especially when targeting GPUs.

⚠️ Compatibility ⚠️
------------------

- **Symbolic loop syntax**: the old "recorded loop" syntax is no longer
  supported. Existing code will need adjustments to use
  :py:func:`drjit.while_loop`.

- **Comparison operators**: The ``==`` and ``!=`` comparisons previously
  reduced the result of to a single Python ``bool``. They now return an array
  of component-wise comparisons to be more consistent with other array
  programming frameworks. Use :py:func:`dr.all(a == b) <all>` or
  :py:func:`dr.all(a == b, axis=None) <all>` to get the previous behavior.

  The functions ``drjit.eq()`` and ``drjit.neq()`` for element-wise
  equality and inequality tests were removed, as their behavior is now subsumed
  by the builtin ``==`` and ``!=`` operators.

- **Matrix layout**: The Dr.Jit matrix type switched from column-major to
  row-major storage. Your code will need to be updated if it indexes into
  matrices first by column and then row (``matrix[col][row]``) instead of
  specifying the complete location ``matrix[row, col]``. The latter convention
  is consistent between both versions.


Internals
---------

This section documents lower level changes that don't directly impact the
Python API.

- Compilation of Dr.Jit is faster and produces smaller binaries. Downstream
  projects built on top of Dr.Jit will also see improvements on both metrics.

- Dr.Jit now builds a support library (``libdrjit-extra.so``) containing large
  amounts of functionality that used to be implemented using templates. The
  disadvantage of the previous template-heavy approach was that this code ended
  up getting compiled over and over again especially when Dr.Jit was used
  within larger projects such as `Mitsuba 3 <https://mitsuba-renderer.org>`__,
  where this caused very long compilation times.

  The following features were moved into this library:

  * Transcendental functions (:py:func:`drjit.log`, :py:func:`drjit.atan2`,
    etc.) now have pre-compiled implementations for Jit arrays. Automatic
    differentiation of such operations was also moved into
    ``libdrjit-extra.so``.

  * The AD layer was rewritten to reduce the previous
    backend (``drjit/autodiff.h``) into a thin wrapper around
    functionality in ``libdrjit-extra.so``. The previous AD-related shared
    library ``libdrjit-autodiff.so`` no longer exists.

  * The template-based C++ interface to perform vectorized method calls on
    instance arrays (``drjit/vcall.h``, ``drjit/vcall_autodiff.h``,
    ``drjit/vcall_jit_reduce.h``, ``drjit/vcall_jit_record.h``) was removed and
    turned into generic implementation within the ``libdrjit-extra.so``
    library. All functionality (symbolic/evaluated model, automatic
    differentiation) is now exposed through a single statically precompiled
    function (``ad_call``). The same function is also used to realize the Python
    interface (:py:func:`drjit.switch`, :py:func:`drjit.dispatch`).

    To de-emphasize C++ *virtual* method calls (the interface is more broadly
    about calling things in parallel), the header file was renamed to
    ``drjit/call.h``. All macro uses of ``DRJIT_VCALL_*`` should be renamed to
    ``DRJIT_CALL_*``.

  * Analogous to function calls, the Python and C++ interfaces to
    symbolic/evaluated loops and conditionals are each implemented through a
    single top-level function (``ad_loop`` and ``ad_cond``) in
    ``libdrjit-extra.so``. This removes large amounts of template code and
    accelerates compilation.

- Improvements to CUDA and LLVM backends kernel launch configurations that
  more effectively use the available parallelism.

- The packet mode backend (``include/drjit/packet.h``) now includes support
  for ``aarch64`` processors via NEON intrinsics. This is actually an old
  feature from a predecessor project (Enoki) that was finally revived.

- The ``nb::setattr()`` function that was previously used to update modified
  fields queried by a *getter* no longer exists. Dr.Jit now uses a simpler way
  to deal with getters. The technical reason that formerly required the
  presence of this function doesn't exist anymore.


Removals
--------

- Packet-mode virtual function call dispatch (``drjit/vcall_packet.h``)
  was removed.

- The legacy string-based IR in Dr.Jit-core has been removed.

- The ability to instantiate a differentiable array on top of a
  non-JIT-compiled type (e.g., ``dr::DiffArray<float>``) was removed. This was
  in any case too inefficient to be useful besides debugging.

Other minor technical improvements
----------------------------------

- :py:func:`drjit.switch` and :py:func:`drjit.dispatch` now support all
  standard Python calling conventions (positional, keyword, variable length).

- There is a new C++ interface named :cpp:func:`drjit::dispatch` that works
  analogously to the Python version.

- The ``drjit.reinterpret_array_v`` function was renamed to
  :py:func:`drjit.reinterpret_array`.

- The :py:func:`drjit.llvm.PCG32.seed` function (and other backend variants)
  were modified to add the lane counter to both `initseq` and `initstate`.
  Previously, the counter was only added to the former, which led to noticeable
  correlation artifacts.
