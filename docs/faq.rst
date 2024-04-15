.. py:currentmodule:: drjit

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


Why can I not use PyTorch/TensorFlow/JAX inside a symbolic loop/conditional/call?
---------------------------------------------------------------------------------

This question arises periodically when users wish to embed a neural computation
into an algorithm written using Dr.Jit. If this algorithm uses symbolic control
flow operations either directly (via :py:func:`dr.while_loop <while_loop>`,
etc.) or indirectly (via the :py:func:`@dr.syntax <syntax>` decorator), the
attempt to connect the two frameworks fails with an error message.

Operations in PyTorch et al. aren't compatible with Dr.Jit's symbolic variable
representation---they need concrete inputs. The only way to connect them is to
evaluate the array contents, typically by switching the operations from
symbolic to evaluated mode. This connection can be further formalized
using the decorator discussed in the section on :ref:`interoperability
<interop>`. This isn't a great solution since evaluated mode comes at a
significant additional cost.
