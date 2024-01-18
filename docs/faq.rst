.. _faq:

Frequently asked questions
==========================

Many tensor operations are missing, will you add them?
------------------------------------------------------

While Dr.Jit does have a tensors and certain operations that operate on them,
it is *not* a general array programming framework in the style of NumPy or
PyTorch. If you try to use it like one, you likely won't have a good time.

Fast Dr.Jit programs perform many parallel evaluations of programs built from
:ref:`flat <flat_arrays>` and :ref:`nested <nested_arrays>` array operations,
which the system fuses into a large self-contained kernel. See the section on
:ref:`limitations involving tensors <tensor_limitations>` to see why
tensor-heavy development style tends to interfere with this process.
