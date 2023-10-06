.. py:module:: drjit

General information
===================

.. _horizontal-reductions:

Horizontal reductions
---------------------

Dr.Jit offers the following *horizontal operations* that reduce the dimension
of an input array, tensor, or Python sequence:

- :py:func:`drjit.sum`, which reduces using ``+``,
- :py:func:`drjit.prod`, which reduces using ``*``,
- :py:func:`drjit.min`, which reduces using ``min()``,
- :py:func:`drjit.max`, which reduces using ``max()``,
- :py:func:`drjit.all`, which reduces using ``&``, and
- :py:func:`drjit.any`, which reduces using ``|``.

By default, these functions reduce along the outermost dimension and return an
instance of the array's element type. For instance, sum-reducing an array ``a`` of
type :py:class:`drjit.cuda.Array3f` would just be a convenient abbreviation for
the expression ``a[0] + a[1] + a[2]`` of type :py:class:`drjit.cuda.Float`.
Dr.Jit can execute this operation symbolically.

Reductions of 1D JIT-compiled arrays (e.g., :py:class:`drjit.cuda.Float`) are
an important special case. Since each value of such an array represents a
different thread of execution of a program, Dr.Jit must first invoke
:py:func:`drjit.eval` to evaluate and store the array in memory and then launch
an device-specific implementation of a horizontal reduction. This interferes
with symbolic execution and is even forbidden in certain execution contexts
(e.g., when recording symbolic loops and functions).

Furthermore, in this mode of operation, Dr.Jit does *not* return the result
using the array's element type (e.g., a standard Python `float`). Instead, it
returns the sum as a dynamic array (:py:class:`drjit.cuda.Float`) containing
single element. This is an intentional design decision to avoid transferring
the value to the CPU, which would incur GPU<->CPU synchronization overheads on
some backends. You must explicitly index into the result (``result[0]``) to
obtain a value with the underlying element type.

All reduction operations take an optional argument ``axis`` that specifies the
axis of the reduction (default: ``0``). The value ``None`` implies a reduction
over all array axes. Arguments other than ``0`` and ``None`` are currently
unsupported.

.. _pytrees:

Pytrees
-------

The word *Pytree* (borrowed from `JAX
<https://jax.readthedocs.io/en/latest/pytrees.html>`_) refers to a tree-like
data structure made of Python container types including ``list``, ``tuple``,
and ``dict``, which can be further extended to encompass user-defined classes.

Various Dr.Jit operations will automatically traverse such Pytrees to process
any Dr.Jit arrays or tensors found within. For example, it might be convenient
to store differentiable parameters of an optimization within a dictionary and
then batch-enable gradients:

.. code-block:: python

   from drjit.cuda.ad import Array3f, Float

   params = {
       'foo': Array3f(...),
       'bar': Float(...)
   }

   dr.enable_grad(params)

Pytrees can similarly be used as variables in recorded loops, arguments and
return values of polymorphic function calls, arguments in scatter/gather
operations, and many others (the :ref:`reference <reference>` explicitly lists
the word *Pytree* in all supported operations).

To turn a user-defined type into a Pytree, define a static ``DRJIT_STRUCT``
member dictionary describing the names and types of all fields. It should also
be default-constructible without the need to specify any arguments. For
instance, the following snippet defines a named 2D point, containing (amongst
others) two nested Dr.Jit arrays.

.. code-block:: python

   from drjit.cuda.ad import Float

   class MyPoint2f:
       DRJIT_STRUCT = { 'name': str, 'x' : Float, 'y': Float }

       def __init__(self, name: str = "", x: Float = 0, y: Float = 0):
           self.name = name
           self.x = x
           self.y = y

   # Create a vector representing 100 2D points. Dr.Jit will
   # automatically populate the 'x' and 'y' members
   value = dr.zeros(MyPoint2f, 100)
