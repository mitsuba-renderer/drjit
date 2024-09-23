.. py:currentmodule:: drjit

.. _autodiff:

Differentiation
===============

This section explains the use of `automatic differentiation
<https://en.wikipedia.org/wiki/Automatic_differentiation>`__ to compute
derivatives of arbitrary computation.

Introduction
------------

Before delving into the Python interface, let us first review mathematical
principles of derivative computation.

Consider a program that consumes certain inputs :math:`x_1`, :math:`x_2`, etc.,
performs a computation, and then generates outputs :math:`y_1`, :math:`y_2`,
etc. Systems for *automatic differentiation* (AD) automate the computation of
derivatives :math:`\partial y_i/\partial x_i`, which is instrumental when the
program should be be optimized to accomplish a certain task.

AD does this by decomposing the program into a sequence of steps that are
individually easy to differentiate, and it then using the `chain rule
<https://en.wikipedia.org/wiki/Chain_rule>`__ to stitch per-step derivatives
into derivatives of the larger program.

For simplicity, let's assume at first that the computation is *pure*, i.e.,
that it consistently produces the same output if re-run with the same input. In
this case, we can think of the program as a function
:math:`f:\mathbb{R}^m\to\mathbb{R}^n` with an associated `Jacobian matrix
<https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant>`__

.. math::

   \mathbf{J}_f = \begin{bmatrix}
   \frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_n}\\
   \vdots &\ddots& \vdots\\
   \frac{\partial f_m}{\partial x_1}&\cdots&\frac{\partial f_m}{\partial x_n}
   \end{bmatrix}

If :math:`f` has many inputs and outputs (e.g., with :math:`m` and :math:`n` on
the order of a few millions), its Jacobian matrix with shape :math:`m\times n` is
*incredibly* large, making it too costly to store or compute.

A surprising insight of automatic derivative computation is that although
:math:`J_f` is often impossibly expensive to use on its own, one can cheaply
compute *matrix-vector* products with :math:`\mathbf{J}_f` on the fly. The cost
of this is dramatically lower than the naive strategy of first computing
:math:`\mathbf{J}_f` and then doing the matrix-vector multiplication with the
stored matrix. The key feature of AD systems is that  they can automatically
implement these kinds of matrix-vector products for a given algorithm
:math:`f`.

Two kinds of products are mainly of interest: the **forward mode**
right-multiplies :math:`\mathbf{J}_f` with an arbitrary :math:`n`-dimensional
vector :math:`\boldsymbol{\delta}_\mathbf{x}`:

.. math::
   :name: eq:1

   \boldsymbol{\delta}_\mathbf{y} = \mathbf{J}_{\!f}\,\boldsymbol{\delta}_\mathbf{x}

The result :math:`\boldsymbol{\delta}_\mathbf{y}` provides a first-order
approximation of the change in :math:`f` when shifting the evaluation point
:math:`\mathbf{x}` into direction :math:`\boldsymbol{\delta}_\mathbf{x}` (in
other words, a `directional derivative
<https://en.wikipedia.org/wiki/Directional_derivative>`__).

Forward mode is great whenever we need to compute many output derivatives with
respect to a single input parameter :math:`x_j`. In this case, we would simply
set the input perturbation :math:`\boldsymbol{\delta}_\mathbf{x}` to a
zero-filled vector with a :math:`1` in the :math:`j`-th component so that the
expression in Equation :math:numref:`eq:1` extracts the :math:`j`-th column of
:math:`J_{\!f}`. Setting :math:`\boldsymbol{\delta}_\mathbf{x}` to other values
can be used to cheaply evaluate arbitrary linear combinations of the columns of
:math:`\mathbf{J}_f`. Extracting multiple columns requires multiple independent
passes with a proportional increase in computation time, which is why forward
mode isn't a good choice when a function should be separately differentiated
with respect to many inputs.

*Reverse*, or *backward mode* instead goes the other way around and is often
more appropriate in this case. It right-multiplies the *transpose Jacobian*
:math:`\mathbf{J}_f^T` with an arbitrary :math:`m`-dimensional *output
perturbation* :math:`\boldsymbol{\delta}_\mathbf{y}`:

.. math::
   :name: eq:2

   \boldsymbol{\delta}_\mathbf{x} = \mathbf{J}^T_{\!f}\,\boldsymbol{\delta}_\mathbf{y}

With a suitable choice of :math:`\boldsymbol{\delta}_\mathbf{y}`, this
expression can extract a row or compute more general linear
combinations of the rows of :math:`\mathbf{J}_f`.

(Note that :math:`\boldsymbol{\delta}_\mathbf{x}` and
:math:`\boldsymbol{\delta}_\mathbf{y}` should be considered different symbols
in Equations :math:numref:`eq:1` and :math:numref:`eq:2`. In other words, this
is not a coupled system of equations).

Reverse mode is widely used to train neural networks in the area of machine
learning, where it is known as *backpropagation*. In this case, the function
:math:`f` computes a single *loss value* from a large set of neural network
parameters, and :math:`\mathbf{J}_f` turns into a large row vector containing
all parameter derivatives. Reverse mode efficiently computes all of these
derivatives at once in a single pass.

.. note::

   There is a somewhat common misconception about these two modes: reverse mode
   does *not* compute derivatives of the function's inverse :math:`f^{-1}`.
   Similarly, forward and reverse derivatives are not mathematical inverses of
   each other. For example, they compute exactly the same value when :math:`f`
   is scalar (i.e., :math:`m=n=1`). Instead, the main difference between them
   is their *efficiency* in obtaining desired derivative values, which depends
   on the target application and shape of the underlying Jacobian (i.e.,
   :math:`m` and :math:`n`).


Basics
------

Differentiable computation requires importing AD-enabled array types from a
separate set of namespaces (:py:mod:`drjit.cuda.ad`, :py:mod:`drjit.llvm.ad`,
and :py:mod:`drjit.auto.ad`). This even includes *non-differentiable* integer
types (e.g. :py:class:`drjit.auto.ad.UInt`) that are replicated there for
convenience.

.. code-block:: pycon

   >>> # ❌ Lacks the ".ad" suffix
   >>> from drjit.auto import Float, Array3f, UInt

   >>> # ✅ AD-enabled array types
   >>> from drjit.auto.ad import Float, Array3f, UInt

Tracking derivatives has a computational cost and is not always desired. You
must use :py:func:`drjit.enable_grad`, to explicitly mark every differentiable
input of a computation:

.. code-block:: pycon

   >>> x = Float(10)
   >>> dr.enable_grad(x)
   >>> dr.grad_enabled(x) # Check that 'x' tracks derivatives
   True

Note that functions in this section generally take multiple arguments and
recurse through :ref:`PyTrees <pytrees>`, which is convenient when
differentiating many variables at once.

Following this step, we can perform a computation and then either differentiate
it operation in *forward mode* via :py:func:`dr.forward <forward>`

.. code-block:: pycon

   >>> y = x**2
   >>> dr.forward(x)
   >>> y.grad
   [20]

or *reverse mode* via :py:func:`dr.backward <backward>`:

.. code-block:: pycon

   >>> y = x**2
   >>> dr.backward(y)
   >>> x.grad
   [20]

The final gradient can be obtained via the :py:attribute:`.grad
<ArrayBase.grad>` member or :py:func:`dr.grad <grad>` (which also works for
:ref:`PyTrees <pytrees>`).


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

