.. py:module:: drjit

.. _changelog:

Changelog
#########

DrJit 2.0.0 (TBA)
-----------------

Dr.Jit 2 represents a major redesign of many parts of this project. The
following list covers the most important changes and their impact on
source-level compatibility. Points with an exclamation sign cover
incompatibilities and potential stumbling blocks.

- ⚠️ The Dr.Jit Python bindings were completely rewritten using a new
  architecture centered around the `nanobind
  <https://github.com/wjakob/nanobind>`__ library. This has the following
  consequences:

  - Tracing Dr.Jit code written in Python is *significantly* faster. Expect
    speedups of ~10-20×. The shared libraries containing the bindings have also
    become much smaller (from ~10MB to ~1MB).

  - All functions now have a reference documentation that clearly specifies
    their behavior and accepted inputs. Their behavior with respect to less
    common inputs (tensors, :ref:`Pytrees <pytrees>`) was made consistent
    and documented across the codebase.

- ⚠️ The Dr.Jit matrix type switched from column-major to row-major storage. If
  you previously manually indexed into matrices (e.g., ``matrix[col][row]``),
  then your code will need to be updated.

- Variable indices (:py:func:`drjit.ArrayBase.index`,
  :py:func:`drjit.ArrayBase.ad_index`) used to monotonically increase as
  variables were being created. Internally, multiple hash tables were needed to
  associate these ever-growing indices to locations in an internal variable
  array, which which had a surprisingly large impact on tracing performance.
  Dr.Jit removes this mapping and eagerly reuses variable indices.

  This may be inconvenient for low-level debugging of trace dumps (via
  :py:func:`drjit.set_log_level`). To force a linear ordering,  **TODO**

- Reductions operations previously existed as *ordinary* (e.g.,
  :py:func:`drjit.all`) and *nested* (e.g. ``drjit.all_nested``) variants. Both
  are now subsumed by an optional ``axis`` argument similar to other array
  programming frameworks like NumPy.

  The reduction functions (:py:func:`drjit.all` :py:func:`drjit.any`,
  :py:func:`drjit.sum`, :py:func:`drjit.prod`, :py:func:`drjit.min`,
  :py:func:`drjit.max`) reduce over the outermost axis (``axis=0``) by default,
  Specify ``axis=None`` to reduce the entire array recursively analogous to the
  previous nested reduction.

  Aliases for the ``_nested`` function variants still exist to facilitate
  porting but are deprecated and will be removed in a future release.
