.. py:currentmodule:: drjit

.. _autodiff:

Automatic differentiation
=========================

(Most of this section still needs to be written)


Differentiating loops
---------------------

(Most of this section still needs to be written)


Reverse-mode derivative of "sum" loops
--------------------------------------

Dr.Jit also provides a specialized reverse-mode differentiation strategy for
certain types of loops that is more efficient and avoids severe storage
overheads. It can be used to handle simple "sum"-style loops such as

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
3. ``y`` is a differentiable output computed from variables
   in categories 1/2 via accumulation (i.e., ``y += f(x, i)``).

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

