.. py:currentmodule:: drjit

.. _cflow:

Control flow
============

Dr.Jit can trace *control flow* statements like loops, conditionals, and
indirect jumps if they are expressed in a compatible manner.

First, let's see what can go wrong when doing this naively. The Python snippet below is meant to
compute the `population count <https://en.wikipedia.org/wiki/Hamming_weight>`__ (i.e., the
number of bits set to *1*) per element of an integer sequence:

.. code-block:: python

   from drjit.auto import Int

   def popcnt(i: Int):
       '''Count the number of active bits in ``i``'''
       j = Int(0)
       while i != 0:  # While there are remaining active bits
           j += i & 1 # Increment counter 'j' if current bit active
           i = i // 2 # Shift bits of 'i' to the right
       return j

   print(popcnt(dr.arange(Int, 1024)))

However, running it fails with an error message:

.. code-block:: pycon

   Traceback (most recent call last):
     File "popcnt.py", line 12, in <module>
       print(popcnt(dr.arange(Int, 1024)))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     File "popcnt.py", line 7, in popcnt
       while i != 0:
             ^^^^^^
   RuntimeError: drjit.llvm.Bool.__bool__(): implicit conversion to 'bool' ↵
   requires an array with at most 1 element (this one has 1024 elements).

Why does this happen? In the example above, ``i`` is an *array* of integers,
hence ``i != 0`` produces a Boolean array of component-wise comparisons.

If these values aren't all identical, it implies that elements of ``i`` require
different numbers of loop iterations. Unfortunately, this is something that
regular Python *simply does not support*. Dr.Jit raises the alarm when it
notices the user's attempt to interpret the condition ``i != 0`` as a Python
``bool``.

Annotating the function with the :py:func:`@dr.syntax <drjit.syntax>` decorator
(change highlighted) fixes this problem:

.. code-block:: python
   :emphasize-lines: 1

   @dr.syntax
   def popcnt(i: Int):
       ...

The script now terminates and prints the following correct output:

.. code-block:: pycon

   [0, 1, 1, .. 1018 skipped .., 9, 9, 10]

How does this work?
-------------------

The :py:func:`@dr.syntax <drjit.syntax>` decorator provides *syntax sugar*: it
takes a function as input and returns a slightly modified version of it. Most
code passes through unchanged, with two exceptions: the decorator rewrites the
declarations of loops and conditionals to make them compatible with tracing.

In the example above, it does this by

1. identifying the variables modified by the loop (``i`` and ``j``)

2. encapsulating the loop condition and body in separate functions, and

3. performing a call to :py:func:`dr.while_loop() <drjit.while_loop>` with all
   of this information.

This produces code equivalent to:

.. code-block:: python

   def popcnt(i: Int):
       j = Int(0)
       i, j = dr.while_loop(
           state=(i, j),
           cond=lambda i, j: i != 0,
           body=lambda i, j: (i // 2, j + (i & 1))
       )
       return j

The function :py:func:`dr.while_loop() <drjit.while_loop>` *generalizes* the
built-in Python ``while`` loop: when the condition is a Python ``bool``, it
doesn't do anything special and just reproduces the normal behavior. When the
loop condition is an array, it runs the loop *separately for each element*,
potentially for different numbers of iterations.

In the same manner, ``if`` statements will be turned into calls to
:py:func:`dr.if_stmt() <drjit.if_stmt>` that serves the same purpose for
conditionals.

The main feature of :py:func:`@dr.syntax <drjit.syntax>` is to free users from
having to perform this transformation themselves.

Symbolic mode
-------------

The default way in which Dr.Jit handles control flow is called *symbolic mode*,
which has certain limitations. Let's make a small change to the code from
before to illustrate one of them.

.. code-block:: python
   :emphasize-lines: 6

   @dr.syntax
   def popcnt(i: Int):
       '''Count the number of active bits'''
       j = Int(0)
       while i != 0:
           print(f"{i=}")
           j += i & 1
           i = i // 2
       return j

(the added ``print()`` statement is meant to show the state of variables at
intermediate steps.)

Running this modified code produces a *long* error message:

.. code-block:: pycon

    Traceback (most recent call last):
      File "popcnt.py", line 9, in _loop_body
        print(f"{i=}")

    RuntimeError: You performed an operation that tried to evalute a *symbolic*↵
    variable, which is not permitted.

    [lots of explanation text omitted here]

The message explains that ``i`` and ``j`` are considered **symbolic** while
inside the loop. Certain operations are not allowed in this context, and printing
their contents is one of them.

To understand *why* this is forbidden, recall that Dr.Jit embraces the idea of
*tracing*, i.e., postponing computation for later evaluation. In the case of
``popcnt()``, this means that Dr.Jit will execute the loop body *only once* to
understand how it modifies the variables ``i`` and ``j``, but without doing any
actual computation. Even the number of loop iterations is unknown at this
point. All of these details are postponed to when the traced computation
actually runs on the target device (e.g., the GPU).

The implication of this design is that ``i`` and ``j`` are *symbols* that don't
have explicit values within the loop body, which is why the ``print()``
operation failed.

This way of capturing control flow is the default behavior of Dr.Jit and called
**symbolic mode**. Dr.Jit also supports a second approach called **evaluated
mode** that we will examine next.

Evaluated mode
--------------

The inability to access the contents of symbolic variables can be inconvenient.
We might need to print or plot intermediate steps, or to step through a program
using a visual debugger.

To do so, let's switch the loop to **evaluated mode**. We can do so at a
statement level by annotating the loop condition with :py:func:`dr.hint(...,
mode='evaluated') <drjit.hint>`.

.. code-block:: python
   :emphasize-lines: 5

   @dr.syntax
   def popcnt(i: Int):
       '''Count the number of active bits'''
       j = Int(0)
       while dr.hint(i != 0, mode='evaluated'):
           print(f"{i=}")
           j += i & 1
           i = i // 2
       return j

   popcnt(dr.arange(Int, 1024))

With this change, Dr.Jit now executes all loop iterations explicitly. Accessing
the contents of ``i`` also works without problems, and the script produces
the following output:

.. code-block:: text

    i=[0, 1, 2, .. 1018 skipped .., 1021, 1022, 1023]
    i=[0, 0, 1, .. 1018 skipped .., 510, 511, 511]
    i=[0, 0, 0, .. 1018 skipped .., 255, 255, 255]
    i=[0, 0, 0, .. 1018 skipped .., 127, 127, 127]
    i=[0, 0, 0, .. 1018 skipped .., 63, 63, 63]
    i=[0, 0, 0, .. 1018 skipped .., 31, 31, 31]
    i=[0, 0, 0, .. 1018 skipped .., 15, 15, 15]
    i=[0, 0, 0, .. 1018 skipped .., 7, 7, 7]
    i=[0, 0, 0, .. 1018 skipped .., 3, 3, 3]
    i=[0, 0, 0, .. 1018 skipped .., 1, 1, 1]
    [0, 1, 1, .. 1018 skipped .., 9, 9, 10]

Evaluated mode can also be enabled globally by disabling the flags
:py:attr:`dr.JitFlag.SymbolicLoops <drjit.JitFlag.SymbolicLoops>` and
:py:attr:`dr.JitFlag.SymbolicConditionals <drjit.JitFlag.SymbolicConditionals>`
via :py:func:`dr.set_flag() <set_flag>` or :py:func:`dr.scoped_set_flag()
<scoped_set_flag>`.

.. _sym-eval:

Discussion
----------

Let's take a step back and compare the properties of these two different modes.

Evaluated mode
~~~~~~~~~~~~~~

As the name suggests, this mode evaluates loop variables to store them in
memory. Each loop iteration then loads variable state and writes out new state
at the end. The *host* (i.e., the CPU) is in charge of all control flow, which
makes this mode simple to understand:

- Debugging programs is straightforward. The user can step through program line
  by line and examine variable contents via Python's built-in ``print()``
  statement or more advanced graphical plotting tools to construct
  visualizations from within loops, conditionals, and calls (tracing calls is
  described at the bottom of this section).

- The program can freely mix Dr.Jit computation with other array programming
  frameworks like PyTorch, Tensorflow, JAX, etc.

The main *disadvantage* of evaluated mode are overheads from constantly reading
and writing from/to device memory. The resulting memory bandwidth and storage
costs can be prohibitive.

Symbolic mode
~~~~~~~~~~~~~

Symbolic mode moves the control flow onto the target device. This is a natural
choice: Dr.Jit already traces computation to generate fused kernels, and
this simply extends that idea to include control flow as well. For this, Dr.Jit
must trace loops that run for an *unknown* number of iterations,
which it does by introducing symbolic variables to capture the change from one
iteration to the next. Symbolic variables represent unknown information that
will only become available later when the generated code runs on the device.

The advantage of symbolic mode is that it can keep variable state in fast
CPU/GPU registers, which improves performance and reduces storage costs.

The main *disadvantage* is that symbolic variables cannot be evaluated while
tracing. Likewise, they cannot be passed to other frameworks like PyTorch or
Tensorflow. Indeed, *any* attempt to reveal the content of symbolic variables
is doomed to fail since it literally does not exist (yet). The upcoming section
on :ref:`variable evaluation <eval>` clarifies what operations require
evaluation. Symbolic mode is the default, since the performance benefits
usually outweigh these disadvantages.

.. note::

   Here are a few more detailed notes about symbolic and evaluated loops for
   advanced users. Feel free to skip these if you are new to Dr.Jit.

   - Loops (:py:func:`drjit.while_loop`), conditionals
     (:py:func:`drjit.if_stmt`), and dynamic dispatch (:py:func:`drjit.switch`,
     :py:func:`drjit.dispatch`) may be arbitrarily nested. However, it is not
     legal to nest *evaluated* operations within *symbolic* ones, as this would
     require the evaluation of symbolic variables.

   - Printing array contents is not permitted in symbolic mode, but Dr.Jit
     also provides a requires a *symbolic* print statement implemented by
     :py:func:`dr.print() <drjit.print>` that prints in a delayed manner
     (i.e., asynchronously from the device) to avoid this problem.

   - Symbolic mode tends to create much larger kernels. Indeed, the idea is to
     preserve the entire program and generate one giant output kernel (a
     *megakernel*). Such large kernels can be costly to compile, though
     this cost is usually offset by *kernel caching* discussed in the next
     section.

   - Large kernels produced by symbolic mode also tend to use a large number of
     registers, and this may impede the latency-hiding capabilities of GPUs.
     Simlarly, Dr.Jit always vectorizes computation (SIMD-style). Divergence in
     highly branching code produced by symbolic tracing may reduce performance.

Indirect calls
--------------

Dr.Jit provides the functions :py:func:`dr.switch() <drjit.switch>` and
:py:func:`dr.dispatch() <drjit.dispatch>` to capture indirect function calls
that target multiple possible targets. Here is an example:

.. code-block:: python

   # A sequence of fucntions with the same argument and return value signature
   def f1(a, b, c):
      # ...
      return x, y

   def f2(a, b, c):
      # ...
      return x, y

   x, y = dr.switch(
      targets=[f1, f2], # <-- call functions from the provided list ('f1' or 'f2')
      index=index,      # <-- choose based on the integer array 'index' (indices must be < 2 in this example)
      a, b, c           # <-- function parameters to forward to 'f1' and 'f2'
   )

The reference of :py:func:`dr.switch() <drjit.switch>` and
:py:func:`dr.dispatch() <drjit.dispatch>` explains these two operations in full
detail. As with the previous control flow operations, they support compilation
in either *symbolic* or *evaluated* modes.

Pitfalls
--------

Please be aware of the following potential issues involving tracing of control
flow.

1. **Unrolling loops**. Consider a function ``f(x)``, which calls another
   expensive function ``g(x)`` many times in a loop.

   .. code-block:: python

      @dr.syntax
      def f(x):
          for i in range(1000):
              x = g(x)
          return x

   This will likely not yield the expected behavior: first, Dr.Jit's
   :py:func:`@dr.syntax <drjit.syntax>` decorator ignores ``for`` loops and
   only considers ``while`` loops. Furthermore, it only processes loops with
   array-valued loop stopping conditions, which is not the case here.
   Therefore, this function actually unrolls the computation graph of ``g``
   1000 times and is equivalent to

   .. code-block:: python

      def f(x):
          x = g(x)
          x = g(x)
          # .. (998 repetitions) ..
          return x

   Compiling the resulting giant kernel can be very inefficient. Instead,
   consider rewriting the function as follows so that the loop can be traced:

   .. code-block:: python

      from drjit.auto import Int

      @dr.syntax
      def f(x):
          i = Int(0)
          while i < 1000:
              x = g(x)
              i += 1
          return x

2. **Type constancy**. Tracing control flow requires the type of state
   variables to remain consistent. For example, the following fails with an
   error message because the body of the ``if`` statement changes ``x`` from
   ``drjit.*.Int`` (a traced Dr.Jit type) to a lower case ``int`` (a built-in
   Python type).

   .. code-block:: python

      @dr.syntax
      def f(x: Int):
          if x < 0:
              x = 0
          # ...

   The problem is easily fixed by casting the assigned value to the expected
   type:

   .. code-block:: python

      @dr.syntax
      def f(x: Int):
          if x < 0:
              x = Int(0)
          # ...

3. **Traversal of nested objects**. The :py:func:`@dr.syntax <drjit.syntax>`
   decorator transforms loops and conditionals into calls to
   :py:func:`dr.while_loop() <drjit.while_loop>` and :py:func:`dr.if_stmt()
   <drjit.if_stmt>`.

   This involves traversing local variables to detect potential changes
   during the loop or conditional statement. In the ``Accum.add_positive()``
   example function below, both ``y`` and ``self`` are automatically identified
   as such local variables.

   .. code-block:: python

      from drjit.auto import Int

      class Accum:
          def __init__(self):
              """Create a zero-initialized accumulator"""
              self.value = Int(0)

          @dr.syntax
          def add_positive(self, x: Int):
              """Accumulate 'x', but only if it is positive"""
              if x > 0:
                  self.value += x

      a = Accum()
      a.add_positive(Int(1, -1))
      print(a.value) # Prints: [1, -1]    :-(

   Unfortunately, there is a subtle bug in the above code: symbolic control
   flow operations only traverse :ref:`PyTrees <pytrees>`, and ``self`` (which
   is of type ``Accum``) is *not* a PyTree. The implementation therefore misses
   the conditional nature of the change of ``self.value`` and produces the
   incorrect output ``[1, -1]`` instead of the expected ``[1, 0]``.

   So what is a :ref:`PyTree <pytrees>`? Besides Dr.Jit arrays, they can
   consist of arbitrarily nested Python containers (``list``, ``tuple``,
   ``dict``), `data classes
   <https://docs.python.org/3/library/dataclasses.html>`__, and custom classes
   with a ``DRJIT_STRUCT`` annotation. To fix the problem, we can, e.g., add a
   ``DRJIT_STRUCT`` annotation to ``Accum`` to explain its sub-elements:

   .. code-block:: python

      class Accum:
          DRJIT_STRUCT = { 'value' : Int }

   Alternatively, we can switch the implementation of ``Accum`` to a `data
   class <https://docs.python.org/3/library/dataclasses.html>`__:

   .. code-block:: python
      :emphasize-lines: 3, 5

      from dataclasses import dataclass

      @dataclass
      class Accum:
          value: Int = Int(0)

          @dr.syntax
          def add_positive(self, y: Int):
              ...
