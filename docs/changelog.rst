.. py:module:: drjit

.. _changelog:

Changelog
#########

- ⚠️ The nanobind Python bindings were completely rewritten using a new
  architecture centered around the `nanobind
  <https://github.com/wjakob/nanobind>`__ library. This has the following
  consequences:

  - Tracing Dr.Jit code written in Python is *significantly* faster. Expect
    speedups of ~10-20×. The shared libraries containing the bindings have also
    become much smaller (from ~10MB to ~1MB).

  - All functions now have a reference documentation that clearly specifies
    their behavior and accepted inputs. Their behavior with respect to less
    common inputs (tensors, :ref:`Pytrees <pytrees>`) was made consistent
    across the codebase.

- Reductions operations previously existed as ordinary (e.g.,
  :py:func:`drjit.all`) and nested (e.g. ``drjit.all_nested``) variants. Both
  are now subsumed by an optional ``axis:int|NoneType`` argument similar to
  other array programming frameworks like NumPy.

  All reduction functions (:py:func:`drjit.all` :py:func:`drjit.any`,
  :py:func:`drjit.sum`, :py:func:`drjit.prod`, :py:func:`drjit.min`,
  :py:func:`drjit.max`) reduce over the outermost axis (``axis=0``) by default.
  Specify ``axis=None`` to reduce the entire array recursively.

  The original ``_nested`` function variants continue to exist to facilitate
  porting but are deprecated and will be removed in a future release.

- ⚠️ The Dr.Jit matrix type witched from column-major to row-major storage. If you
  previously manually indexed into matrices (e.g., ``matrix[col][row]``), then
  your code will need to be updated.

