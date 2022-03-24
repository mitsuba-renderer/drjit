.. cpp:namespace:: drjit

.. _recording-loops:

Recording loops
===============

Numerical software frequently involves iterative root-finding or optimization
steps that present challenges during vectorization, especially when working
with backends processing millions of entries at once. This section presents
Dr.Jit's facilities for recording loop constructs to considerably improve
performance in many such situations. You can skip this section if you are not
using JIT-compiled types (i.e., :cpp:struct:`CUDAArray` or
:cpp:struct:`LLVMArray`)

Motivation
----------

To fully understand the trade-offs, we will look at three different ways to
implement a simple loop (two "bad" ones, and the better alternative). The
example uses CUDA arrays, but everything applies equally to the LLVM case.

1. **Multiple small kernels**: consider the following problematic loop, which
   runs until all elements of the ``value`` vector satisfy a stopping condition
   given by the mask array ``done``.

   .. code-block:: cpp

       dr::CUDAArray<float> value = ...;
       dr::CUDAArray<bool> done = false;

       while (!dr::all(done)) {
           value = f(value); // f() is a placeholder for some complex calculation
           done = stopping_criterion(value); // Are we done yet?
           dr::eval(value, done);
       }

   In this example, each loop iteration will launch a CUDA kernel that reads in
   the current contents of ``value``, updates them, and then writes them back.
   Following this, :cpp:func:`all` triggers an expensive horizontal reduction
   that must wait for all queued GPU computation to conclude before it can
   determine whether the stopping criterion is satisfied for all entries. The
   reduction runs on the GPU and copies its result (a single boolean value) to
   the CPU via a PCI express transaction, which finally informs the CPU whether
   or not to run next loop iteration. The horizontal reduction and GPU ↔ CPU
   synchronization constitute a pipeline flush that breaks the asynchronicity
   generally needed to achieve 100% GPU utilization. That said, this may all
   be fine if each loop iteration does a significant amount of work.


2. **One large kernel**:

   In cases where it is possible to know (or bound) the required number of loop
   iterations, we can simply switch to a regular ``for`` loop that evaluates
   the function ``f`` some number of times. Suitable bounds can often be
   determined for standard methods like binary search, Newton's method
   (obviously dependent on the definition and domain of ``f``), etc.

   .. code-block:: cpp

       dr::CUDAArray<float> value = ...;

       for (int i = 0; i < 50; ++i)
           value = f(value); // f() is a placeholder for some complex calculation

   This is a major improvement over the previous case, because we can generate
   a single large kernel that performs the entire calculation without
   intermediate synchronization barriers. The expensive global memory reads and
   writes are also no longer needed: intermediate results are simply stored in
   GPU registers.

   However, this approach has its limits: it cannot be used when the maximum
   iteration count is unknown. Significant inefficiencies also arise when some
   array entries require many fewer iterations, since the fixed global
   iteration count prevents an early exit optimization.

   Finally, consider what happens when the iteration count is very large (e.g.
   1 million): in this case, the generated kernel will become correspondingly
   large (e.g. 10-100 MiB of CUDA PTX or LLVM IR), at which point compilation
   to machine code will likely fail.

3. **One small kernel with recorded loop**:

   This section presents a new primitive for *recording the loop itself*. The
   feature must first be enabled explicitly in the JIT compiler, otherwise it
   will fall back to approach (1), i.e. many small loops:

   .. code-block:: cpp

       dr::enable_flag(JitFlag::LoopRecord);

   To record loops, you must also include an extra header file

   .. code-block:: cpp

       #include <drjit/loop.h>

   providing the :cpp:struct:`Loop` class. The class must be instantiated with
   the list of variables that are modified by the loop iteration, and the loop
   stopping condition should then be wrapped into its :cpp:func:`Loop::cond()`
   method:

   .. code-block:: cpp

       dr::CUDAArray<float> value = ...;
       dr::CUDAArray<bool> done = false;

       dr::Loop loop(value, done);
       while (loop.cond(!done)) {
           value = f(value); // f() is a placeholder for some complex calculation
           done = stopping_criterion(value); // Are we done yet?
       }

   This does something quite surprising: it runs the loop *a single time* on
   the CPU, which has the sole purpose of recording all involved arithmetic
   symbolically. In contrast, the generated GPU kernel will include
   additional branch statements that cause the iteration associated with each
   entry to run just until stopping condition is satisfied (and no longer!).
   Like in the previous example, this approach uses registers to propagate
   information from one loop iteration to the next (i.e. without costly global
   memory accesses), and it has the added benefit of producing small kernels
   that terminate as soon as the iteration has converged.

   Importantly, none of the previous steps triggered a kernel evaluation: we
   can continue to use ``value`` and queue up further computation, e.g., to
   create interdependent or nested loops.



Usage and limitations
---------------------

Dr.Jit's :cpp:struct:`Loop` primitive will run your loop once, record everything
that it does, and then surround the captured instruction sequence
with additional loop instructions (branch statements, `Phi functions
<https://en.wikipedia.org/wiki/Static_single_assignment_form>`_ in SSA form).
When evaluated on the target device, the resulting kernel will then run the
loop until the specified condition is satisfied.

The involved machinery makes this process more fragile than a standard C++ or
Python ``while`` loop, and you must carefully adhere to the set of rules
outlined below. Failure to do so may result in undefined behavior: ideally
LLVM/CUDA failing due to an invalid PTX/LLVM intermediate representation, but
potentially also crashes or incorrect results.

- **Variable usage**: The loop is allowed to read any variable that was before
  or inside the loop. However, writing variables requires extra precautions:

  - **Local variables**: You do not need to do anything special when your loop
    writes to a local variable that does not propagate information between loop
    iterations. However, stashing this variable somewhere and accessing it
    later on outside of the loop is not allowed (it's not local in that case).

  - **Loop variables**: Variables that propagate state between iterations, or
    from inside to outside of the loop are called *loop variables*. They must
    be passed to the :cpp:struct:`Loop` constructor so that Dr.Jit can insert
    instructions that ensure the correct flow of computed information.

    Loop variables must be LLVM or CUDA arrays or more complex types built from
    them. Builtin C++ or Python types (e.g. an ``int``) do not work, because
    writes to such variables cannot be intercepted by Dr.Jit.

  - **Scatter operations**: the target of a scatter operation
    (:cpp:func:`scatter` and :cpp:func:`scatter_add`) is a special case: it
    does not count as a loop variable despite being the target of a write, and
    it should not be passed to the :cpp:struct:`Loop` constructor.

- **No automatic differentiation**: Dr.Jit will raise an exception when your loop involves
  differentiable variables for which :cpp:func:`grad_enabled()` evaluates to
  ``true``. See the section on :ref:`differentiating loops <diff-loop>` to see
  how to work around this limitation.

- **No eval()**: certain Dr.Jit operations trigger an immediate kernel
  evaluation. These include

  - Horizontal operations: :cpp:func:`all`, :cpp:func:`hsum`, etc..

  - Virtual function calls involving arrays of instance pointers

  - Performing arithmetic involving an unevaluated variable that was previously
    the target of one or more a scatter operations.

  - Other access to unevaluated array contents, e.g. a ``print()`` statement.

  You are not allowed to do any of the above, both within the :cpp:struct:`Loop`
  condition and the body. Dr.Jit will raise an exception when a kernel
  evaluation is triggered while recording a loop.

- **No side effects in condition**: the following loop is okay:

  .. code-block:: cpp

      while (loop.cond(i < 10)) {
          i += 1;
          // .. other code ..
      }

  However, the next one one is not, because the loop condition changes a loop
  variable:

  .. code-block:: cpp

      while (loop.cond(i++ < 10)) {
          // .. other code ..
      }

  This is currently not supported---simply move the side effect to the loop body.

- **Other deviations**:

  The :cpp:struct:`Loop` constructor modifies the supplied loop variables to
  intercept arithmetic involving them, which assumes that this declaration is
  immediately followed by a directive of the form ``while (loop.cond(...))``.
  Deviations from this pattern are not permitted:

  .. code-block:: cpp
     :emphasize-lines: 2, 3

      dr::Loop loop(x);
      x += 1; // Do not  modify loop variables between dr::Loop and the loop body
      while (!loop.cond(x > 0)) { // Negate argument (x > 0) instead of loop.cond()
          //...
      }


C++ example
-----------

The following simple C++ example counts the number of iterations needed to
reach the value 1 in the sequence underlying the `Collatz conjecture
<https://en.wikipedia.org/wiki/Collatz_conjecture>`_. This involves two loop
variables ``value`` and ``cond`` that are both written and read in each
iteration. In contrast, the variable ``is_even`` is only temporary and does not
need to be provided to the :cpp:struct:`Loop` constructor.

.. code-block:: cpp

    using UInt32 = dr::CUDAArray<uint32_t>;

    // Collatz conjecture: count # of iterations to reach 1
    UInt32 collatz(UInt32 value) {
        UInt32 counter = 0;

        dr::Loop loop(value, counter);
        while (loop.cond(dr::neq(value, 1))) {
            dr::mask_t<UInt32> is_even = dr::eq(value & 1, 0);
            value = dr::select(is_even, value / 2, 3*value + 1);
            counter++;
        }

        return counter;
    }




Python example
--------------

There is a major complication in Python that does not appear in C++: an
assignment statement (``a = b``) does not overwrite the contents of ``a``.
Instead, it modifies the local scope to refer to the new value while updating
reference counts. This is normally perfectly fine, but here it interferes with
:cpp:struct:`Loop`'s ability to understand how a variable was modified by a
symbolically executed loop iteration (the original ``a`` will appear
unchanged!)

To avoid this issue in Python, you can use the ``.assign()`` member of the Dr.Jit
array class. It is not needed for in-place updates like ``+=``.

.. code-block:: python
   :emphasize-lines: 6, 11, 12

    import drjit as dr
    from drjit.cuda import UInt32, Loop

    def collatz(value: UInt32):
        counter = UInt32(0)
        value = UInt32(value) # Copy input to avoid modifying array of caller

        loop = Loop(value, counter)
        while loop.cond(dr.neq(value, 1)):
            is_even = dr.eq(value & 1, 0)
            # Use .assign() to update 'value' array instead of creating a new array
            value.assign(dr.select(is_even, value // 2, 3*value + 1))
            counter += 1

        return counter

Apart from this caveat, everything should be have exactly the same as in C++.

Scalar fallback
---------------

The C++ and Python versions of :cpp:struct:`Loop` class provide a scalar
fallback mode: suppose that we replace all CUDA arrays of the previous C++
example by builtin scalar types:

.. code-block:: cpp

    uint32_t collatz(uint32_t value) {
        uint32_t counter = 0;

        dr::Loop loop(value, counter);
        while (loop.cond(dr::neq(value, 1))) {
            dr::mask_t<uint32_t> is_even = dr::eq(value & 1, 0);
            value = dr::select(is_even, value / 2, 3*value + 1);
            counter++;
        }

        return counter;
    }

In this case, ``dr::Loop()`` turns into a no-op, and ``loop.cond()`` simply returns
its input argument. This is useful in template programs that support
compilation to several different backends.

C++ Reference
-------------

.. cpp:struct:: Loop

   Mechanism for JIT-compiling loops with dynamic stopping criteria

   .. cpp:function:: template <typename... Args> Loop(Args&... args)

      Captures the supplied loop variables and modifies them to intercept
      modifications. Loop variables must be LLVM or CUDA arrays, or nested arrays
      thereof. The C++ interface also permits passing custom data structures
      here, as long as their contents were exposed to Dr.Jit via a
      :c:macro:`DRJIT_STRUCT` declaration.

      Construction can occur either in one step:

      .. code-block:: cpp

          dr::Loop loop(arg_1, arg_2);

      Alternative, the class can also be constructed in multiple steps. In this
      case the type of one of the loop variables (does not matter which one)
      must be specified as a template parameter:

      .. code-block:: cpp

          dr::Loop<Float> loop;
          look.put(arg_1);
          look.put(arg_2);
          loop.init();


   .. cpp:function:: template <typename Value> put(Value &value)

       Register a loop variable with the loop.

   .. cpp:function:: void init()

       Finish creating the loop class. Must be called after all loop variables
       are registered, and before :cpp:func:`cond` is invoked.

   .. cpp:function:: bool cond(const Mask &m)

       This function will be called exactly twice in practice: the first time,
       it returns ``true`` indicating that the loop condition should be
       evaluated a second time. At this point, it adjusts all loop variables
       to capture subsequent modifications.

       The second time, it returns ``false`` and updates the loop variables
       to reflect the (still unevaluated) result following loop termination.

   .. cpp:function:: const Mask mask()

       Return the mask value that was previously supplied to the
       :cpp:func:`cond()` function. This is only relevant when the loop
       recording feature is disabled---in all other cases, the return value is
       ``Mask(true)``.
