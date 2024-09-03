.. py:currentmodule:: drjit

.. _autodiff:

Differentiation
===============

This section explains how to use Dr.Jit's `automatic differentiation
<https://en.wikipedia.org/wiki/Automatic_differentiation>`__ feature to compute
derivatives of arbitrary computation.

Basics
------

Before delving into the API, let us first review a few general principles of
automatic derivative computation.

Consider a program that consumes certain inputs :math:`x_1`, :math:`x_2`, etc.,
performs a computation, and then generates outputs :math:`y_1`, :math:`y_2`,
etc. Systems for *automatic differentiation* (henceforth AD) make it possible
to compute derivatives of the form :math:`\partial y_i/\partial x_i` in a fully
automatic way, for example so that the program can be optimized to accomplish a
certain task.

AD does so by decomposing the program into a sequence of steps that are
individually easy to differentiate, and it then uses the `chain rule
<https://en.wikipedia.org/wiki/Chain_rule>`__ to stitch these components into
derivatives of the larger program.

We shall think of the computation

We shall assume at first that
this computation is *pure*, meaning that the computation consistently produces
the same result even if it is 

Implementation
--------------

To use this feature, make sure that you are working with AD-enabled array
types:

   .. code-block:: python

      # ❌ Lacks the ".ad" suffix
      from drjit.auto import Float, Array3f, UInt

      # ✅ AD-enabled array types
      from drjit.auto.ad import Float, Array3f, UInt

To compute derivatives with respect to an input parameter of a computation,


..
   Common mistakes: overwriting or mutating a
   grad-enabled variable and then not being able
   to get its derivative when backpropagating
   fwd mode isn't as efficient as it could be
   An output isn't a leaf.

Note that while Dr.Jit compute first-order derivatives in forward and backward
mode, it lacks support for higher-order differentiation (e.g. Hessian-vector products).



Differentiating loops
---------------------

(Most of this section still needs to be written)


Backward derivative of simple loops
-----------------------------------

Dr.Jit provides a specialized reverse-mode differentiation strategy for certain
types of loops that is more efficient than the default, in particular by
avoiding potentially significant storage overheads. It can be used to handle
simple summation loops such as

.. code-block:: python

   from drjit.auto.ad import Float, Int

   @dr.syntax
   def loop(x: Float, n: int):
       y, i = Float(0), UInt(0)

       while i < n:
           y += f(x, i)
           i += 1

       return y

Here, ``f`` represents an arbitrary pure computation that depends on
``x`` and the loop counter ``i``.

Normally, the reverse-mode derivative of a loop is a complicated and
costly affair: it must run the loop twice, store all intermediate
variable state, and then re-run the loop a second time *in reverse*.

However, the example above admits a simpler and significantly more
efficient solution: we can run the loop just once without reversal and
storage overheads. Conceptually, this reverse-mode derivative looks as
follows:

.. code-block:: python

   def grad_loop(x: Float, grad_y: Float, n: int):
       grad_x, i = Float(0), UInt(0)

       while i < n:
           dr.enable_grad(x)

           y_i = f(x, i)
           y_i.grad = grad_y
           grad_x += dr.backward_to(x)
           i += 1

           dr.disable_grad(x)

       return grad_x

For this optimization to be legal, the loop state must consist of

1. Arbitrary variables that don't carry derivatives
2. Differentiable inputs, which remain constant during the loop
3. Differentiable outputs computed by accumulating a function
   of variables in categories 1 and 2.

These three sets *may not overlap*. In the above example,

1. ``i`` does not carry derivatives.
2. ``x`` is a differentiable input
3. ``y`` is a differentiable output accumulating an expression that depends on
   the variables in categories 1 and 2 (``y += f(x, i)``).

In contrast is *not* important that the loop counter ``i`` linearly increases,
that there is a loop counter at all, or that the loop runs for a uniform number
of iterations.

When the conditions explained above are satisfied, specify
``max_iterations=-1`` to :py:func:`dr.while_loop() <while_loop>`. This tells
Dr.Jit that it can automatically perform the explained optimization to generate
an efficient reverse-mode derivative.

In :py:func:`@dr.syntax <syntax>`-decorated functions, you can equivalently
wrap the loop condition into a :py:func:`dr.hint(..., max_iterations=-1)
<hint>` annotation). The original example then looks as follows:

.. code-block:: python

   @dr.syntax
   def loop(x: Float, n: int):
       y, i = Float(0), UInt(0)

       while dr.hint(i < n, max_iterations=-1):
           y += f(x, i)
           i += 1

       return y

