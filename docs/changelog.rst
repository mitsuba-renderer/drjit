.. py:module:: drjit

.. _changelog:

Changelog
#########

DrJit 1.0.0 (TBA)
-----------------

Dr.Jit 1.0 represents a major redesign of many parts of this project. The
following list covers the most important changes and their impact on
source-level compatibility. Points with an exclamation sign cover
incompatibilities and potential stumbling blocks.

- ⚠️ **Python bindings**: Dr.Jit comes with an all-new set of Python bindings
  created using the `nanobind <https://github.com/wjakob/nanobind>`__ library.
  This was also an opportunity to fix many long-standing binding-related
  problems:

  - Tracing Dr.Jit code written in Python is now *significantly* faster. Expect
    speedups by a factor of ~10-20×. The shared libraries containing the
    bindings have also become much smaller (from ~10MB to just over ~1MB).

  - All functions now have a reference documentation that clearly specifies
    their behavior and accepted inputs. Their behavior with respect to less
    common inputs (tensors, :ref:`Pytrees <pytrees>`) was made consistent
    and documented across the codebase.

  - Due to the magnitude of these changes, you may observe occasional
    incompatibilities. If they are not reported here, please open a ticket so
    that they can either be documented (if intentional) or fixed.

- ⚠️ **Control flow**: You can now express vectorized loops and conditionals
  using natural Python syntax. Consider the following snippet to compute an
  integer power of a floating point value:

  .. code-block:: python

     from drjit.cuda import Int, Float

     @drjit.function
     def ipow(x: Float, n: Int):
         result = Float(1)

         while n != 0:
             if n & 1 != 0:
                 result *= x
             x *= x
             n >> = 1

         return result

  This function processes arrays---it is likely that the condition of the
  ``if`` statement will disagree among elements, and that each element will
  furthermore require a different number of loop operations. Needless to say,
  this is not supported by stock Python.

  Handling such constraints previously required the construction of a special
  ``dr.cuda.Loop`` object along with masked assignments to imitate conditional
  execution (e.g., ``result[dr.neq(n&1, 0)] *= x``). The natural notation shown
  above improves readability and even efficiency, as Dr.Jit is now able to
  retain conditional statements and translate them into conditional jumps in
  the generated CPU/GPU program.

  Note the presence of the :py:func:`@drjit.function <drjit.function>`
  decorator, which transforms the ``while`` loop into a call to the
  array-compatible function :py:func:`drjit.while_loop` and the ``if``
  statement into a call to :py:func:`drjit.if_stmt`. Besides these two effects,
  the transformation is minimal and preserves other code along with line number
  information to aid debugging.

- ⚠️ **Comparison operators**: The ``==`` and ``!=`` comparisons previously
  reduced the result of to a single Python ``bool``. They now return an array
  of component-wise comparisons to be more consistent with other array
  programming frameworks. Use :py:func:`dr.all(a == b) <all>` or
  :py:func:`dr.all(a == b, axis=None) <all>` to get the previous behavior.

  The functions ``drjit.eq()`` and ``drjit.neq()`` for element-wise
  equality and inequality tests were removed, as their behavior is now subsumed
  by the builtin ``==`` and ``!=`` operators.

- ⚠️ **Matrix layout**: The Dr.Jit matrix type switched from column-major to
  row-major storage. Your code will need to be updated if it indexes into
  matrices first by column and then row (``matrix[col][row]``) instead of
  specifying the complete location ``matrix[row, col]``. The latter convention
  is consistent between both versions.

- **Mixed-precision optimization**: Dr.Jit now maintains one global AD graph
  for all variables, enabling differentiation of computation combining single-,
  double, and half precision variables. Previously, there was a separate graph
  per type, and gradients did not propagate through casts between them.

- **Half-precision arithmetic**: Dr.Jit now provides ``float16``-valued arrays
  and tensors on both the LLVM and CUDA backends (e.g.,
  :py:class:`drjit.cuda.ad.TensorXf16` or :py:class:`drjit.llvm.Float16`).

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

- The new release has a strong focus on error resilience and leak avoidance.
  Exceptions raised in custom operations, function dispatch, symbolic loops,
  etc., should not cause failures or leaks. Both Dr.Jit and nanobind are very
  noisy if they detect that objects are still alive when the Python interpreter
  shuts down. You may occasionally still see such leak warnings.

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

- Variable indices (:py:attr:`drjit.ArrayBase.index`,
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

     dr.set_flag(dr.JitFlag.IndexReuse, False)

  Note that this causes the internal variable array to steadily grow, hence
  this feature should only be used for brief debugging sessions.

- Dr.Jit can now target the Python 3.12+ stable ABI. This means that binary
  wheels will work on future versions of Python without recompilation.

- The :py:func:`drjit.empty` function used to immediate allocate an array of
  the desired shape (compared to, say, :py:func:`drjit.zero` which creates a
  literal constant array that consumes no device memory). Users found this
  surprising, so the behavior was changed so that :py:func:`drjit.empty`
  similarly delays allocation.

Internals
---------

This section documents lower level changes that don't directly impact the
Python API.

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

- the ``drjit.reinterpret_array_v`` function was renamed to
  :py:func:`drjit.reinterpret_array`.

- The :py:func:`drjit.llvm.PCG32.seed` function (and other backend variants)
  were modified to add the lane counter to both `initseq` and `initstate`.
  Previously, the counter was only added to the former, which led to noticeable
  correlation artifacts.
