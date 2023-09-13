.. _pytrees:

Pytrees
=======

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

To turn a user-defined into a Pytree, define a static ``DRJIT_STRUCT`` member
dictionary describing the names and types of all fields. For instance, the
following snippet defines a named 2D point, containing (amongst others) two
nested Dr.Jit arrays.

.. code-block:: python

   from drjit.cuda.ad import Float

   class MyPoint2f:
       DRJIT_STRUCT = { 'name': str, 'x' : Float, 'y': Float }

       def __init__(self, name: str, x: Float, y: Float):
           self.name = name
           self.x = x
           self.y = y
