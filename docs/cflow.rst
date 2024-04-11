.. py:currentmodule:: drjit

.. _cflow:

Control flow
============

This section explains how Dr.Jit handles control flow constructs such as ``if``
statements, ``while`` loops, and dynamic dispatch.

``if`` statements
-----------------

The :py:func:`drjit.if_stmt` function provides a generalized ``if`` statement
that works even when the conditional expression is a boolean-valued array or
tensor. The resulting vectorized code only calls the taken branch when
possible.

.. code-block:: python

   x = dr.cuda.Float(...)

   # Return true_fn(*args) or false_fn(*args) depending on the value of 'cond'
   x = dr.if_stmt(
       args = (x,),
       cond = x > 1,
       true_fn = lambda x: x - 1,
       false_fn = lambda x: x
   )


Since it can be tedious to write larger programs in this functional style, the
library also provides the :py:func:`@drjit.syntax <drjit.syntax>` decorator
that automatically rewrites ordinary Python code into the above form:

.. code-block:: python

   @dr.syntax
   def f(x):
       if x > 1:
           x -= 1

See the section on :ref:`symbolic and evaluated modes <sym-eval>` for an
overview of how the system compiles such control flow statements. The reference
of :py:func:`drjit.if_stmt` discusses the ``if`` statement in full detail.

``while`` loops
---------------

The :py:func:`drjit.while_loop` function provides a generalized ``while`` loop
that works even when the loop condition is a boolean-valued array or
tensor. The resulting vectorized code can run the loop for a varying number of
operations per array/tensor element when needed.

.. code-block:: python

   # Find the square root of a=1, 2, .., 10 using the Babylonian algorithm / Newton's method
   a = dr.arange(dr.cuda.Float, 10) + 1
   x = dr.cuda.Float(1)

   # Run body(*state) while cond(*state) is True, then return 'state'
   _, x = dr.while_loop(
       state = (a, x),
       cond = lambda a, x: abs((x*x - a) / a) > 1e-6,
       body = lambda a, x: .5 * (x + a/x)
   )

Since it can be tedious to write larger programs in this functional style, the
library also provides the :py:func:`@drjit.syntax <drjit.syntax>` decorator
that automatically rewrites ordinary Python code into the above form:

.. code-block:: python

   @dr.syntax
   def f(a, x):
       while abs((x*x - a) / a) > 1e-6:
           x = .5 * (x + a/x)
       return x

See the section on :ref:`symbolic and evaluated modes <sym-eval>` for an
overview of how the system compiles such control flow statements. The reference
of :py:func:`drjit.while_loop` discusses the ``while`` loop in full detail.

Dynamic dispatch
----------------

The term `dynamic dispatch <https://en.wikipedia.org/wiki/Dynamic_dispatch>`__
refers to a type of function call that targets multiple possible
implementations based on runtime information. The functions
:py:func:`drjit.switch` and :py:func:`drjit.dispatch` realize this type of
control flow primitive within Dr.Jit.

.. code-block:: python

   def f1(a, b, c):
      # ...
      return x, y

   def f2(a, b, c):
      # ...
      return x, y

   # Call either 'f1' or 'f2' based on 'index', an integer array with values 0 and 1
   x, y = dr.switch(
      targets = [f1, f2],
      index = index,
      a, b, c
   )

See the next section on :ref:`symbolic and evaluated modes <sym-eval>` for an
overview of how the system compiles such control flow statements. The reference
of :py:func:`drjit.switch` and :py:func:`drjit.dispatch` explains these two
operations in full detail.

.. _sym-eval:

Symbolic versus evaluated modes
-------------------------------

All control flow operations support compilation in either *symbolic* or
*evaluated* modes. This section discusses them in turn.

Symbolic mode
_____________

*Symbolic mode* captures the complete structure of a program and turns it into
a single large kernel that eventually runs on the target device.

Symbolic mode exists to avoid unwanted intermediate evaluation of variables,
which would split the large kernel into multiple smaller ones. The resulting
inter-kernel communication via device memory tends to have a *significant cost*
in terms of both storage requirements and memory bandwidth.

This is no big surprise: Dr.Jit already traces computation to generate fused
kernels that specifically avoid these communication overheads. However, control
flow constructs (loops, conditionals, dynamic dispatch) present a difficulty
during this tracing process. Consider the following example:

.. code-block:: python

   while x > 0:
       x = f(x)

Knowing when to stop this loop requires access to the contents of ``x``. To
keep evaluation of ``f(x)`` on the target device (e.g. the GPU) while at the
same time avoiding intermediate evaluation, Dr.Jit must capture a loop that
runs for an *unknown* number of iterations. Doing so preserves the control flow
structure of the original program, by effectively replicating it within
Dr.Jit's intermediate representation.

To accomplish these goals, Dr.Jit invokes the loop body with *symbolic*
variables to capture the change from one iteration to the next. Symbolic
variables represent unknown information that will only become available later
when the generated code runs on the device.

Advantages
~~~~~~~~~~

Symbolic mode is highly efficient with regards to of device storage
requirements and memory bandwidth. This is because function call arguments,
return values, loop state variables, etc., can all be exchanged via fast
CPU/GPU registers.

The difference is particularly pronounced when compiling code for the CPU,
where memory bandwidth can quickly become a bottleneck.

Disadvantages
~~~~~~~~~~~~~

Executing code in symbolic mode can be somewhat restrictive. For example, any
attempt to reveal the contents of a symbolic variable is doomed to fail since
it literally does not exist (yet). Other operations requiring variable
evaluation (:py:func:`drjit.eval`) are likewise not permitted:

.. code-block::

   >>> @dr.syntax
   ... def f(i: dr.cuda.Int, x: dr.cuda.Array2f):
   ...     while i < 10:
   ...         x *= x
   ...         i += 1
   ...         print(x)                # <-- fails
   ...         y: dr.cuda.Float = x[0] # <-- OK
   ...         z: float         = y[0] # <-- fails
   ...
   >>> f(dr.cuda.Int(1, 2), dr.cuda.Array2f(3, 4))
   You performed an operation that tried to evalute a *symbolic*
   variable which is not permitted.

   Tracing operations like dr.while_loop(), dr.if_stmt(), dr.switch(),
   dr.dispatch(), etc., employ such symbolic variables to call code with
   abstract inputs and record the resulting computation. It is also
   possible that you used ordinary Python loops/if statements together
   with the @dr.syntax decorator, which automatically rewrites code to
   use such tracing operations. The following operations cannot be
   performed on symbolic variables:

    - You cannot use dr.eval() or dr.schedule() to evaluate them.

    - You cannot print() their contents. (But you may use dr.print() to
      print them *asynchronously*)

    - You cannot perform reductions that would require evaluation of the
      entire input array (e.g. dr.all(x > 0, axis=None) to check if the
      elements of an array are positive).

    - You cannot access specific values in 1D arrays (this would require
      the contents to be known.)

   The common pattern of these limitation is that the contents of symbolic
   of variables are *simply not known*. Any attempt to access or otherwise
   reveal their contents is therefore doomed to fail.

   Symbolic variables can be inconvenient for debugging, where it is nice
   to be able to stick a print() call into code, or to single-step through
   a program and investigate intermediate results. If you wish to do this,
   then you should switch Dr.Jit from *symbolic* into *evaluated* mode.

   This mode tends to be more expensive in terms of memory storage and
   bandwidth, which is why it is not enabled by default. Please see the
   Dr.Jit documentation for more information on symbolic and evaluated
   evaluation modes:
   https://drjit.readthedocs.io/en/latest/cflow.html#symbolic-versus-evaluated-modes

It is perfectly valid to index into nested Dr.Jit arrays like
:py:class:`drjit.cuda.Array2f`, but the end result should *not* be a Python
``int`` or ``float`` since that would require knowing the actual array
contents.

As the message above indicated, printing array contents is possible, but this
requires a *symbolic* print statement implemented by :py:func:`drjit.print`
that delays the output until the content of all referenced variables is
available.

Other Python array programming frameworks do not support Dr.Jit's symbolic
inputs---this means that you cannot, e.g., use PyTorch or Tensorflow to
evaluate a neural network within a Dr.Jit loop or indirect function call.

Loops (:py:func:`drjit.while_loop`), conditionals (:py:func:`drjit.if_stmt`),
and dynamic dispatch (:py:func:`drjit.switch`, :py:func:`drjit.dispatch`) may
be arbitrarily nested. However, it is not legal to nest *evaluated* operations
within *symbolic* ones, as this would require the evaluation of symbolic
variables.

Some of the above limitations may be inconvenient especially when debugging
code, in which case you may prefer to temporarily switch to evaluated mode.

Besides these points, symbolic mode has several additional disadvantages that
we mention for completeness:

- Symbolic mode tends to create large kernels, which can be costly
  to compile. However, this cost is generally offset by *kernel caching*.

- Large kernels also tend to use a large number of registers, and this may
  impede the latency-hiding capabilities especially when targeting GPUs.

- Dr.Jit vectorizes computation (SIMD-style). Divergence in highly
  branching code may eliminate the benefits of this optimization.

Symbolic mode is the default, since the performance benefits usually outweigh
all of the above points.

Evaluated mode
______________

Evaluated mode executes programs in such a way that control flow decisions such
as the loop iteration count from the earlier example are known at trace time.
This involves frequent kernel launches to evaluate variable contents (via
:py:func:`drjit.eval`).

Advantages
~~~~~~~~~~

Programs that use evaluated mode are easier to debug. It is possible to
single-step through programs and examine the contents of temporary variables.
You may use Python's built-in ``print`` statement or more advanced
graphical plotting tools to construct visualizations from within loops and
dynamically called functions. The program may freely mix Dr.Jit computation
with other array programming frameworks like PyTorch, Tensorflow, JAX, etc.
Kernels are smaller and avoid thread divergence. (For example, Dr.Jit reorders
the inputs of calls so that the computation is 100% converged).

Disadvantages
~~~~~~~~~~~~~

Evaluated mode tends to be *significantly* slower than symbolic mode, as data
is constantly read and written from/to device memory. The required memory
bandwidth and storage can make this mode impractical.
