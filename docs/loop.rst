enoki::Loop -- the sharp bits
=============================

Enoki's ``enoki::Loop`` primitive is quite unusual: it will run your loop once,
record everything that it does symbolically, and then surround the captured
instruction sequence with additional loop instructions (branch statements, `Phi
functions <https://en.wikipedia.org/wiki/Static_single_assignment_form>`_ in
SSA form). When evaluated on the target device, the resulting kernel will then
run the loop many times until the specified condition is satisfied.

This process is far more fragile than a standard C++ or Python ``while`` loop,
and you must carefully adhere to the set of rules outlined below. Failure to do
so will result in undefined behavior: ideally LLVM/CUDA failing due to invalid
PTX or LLVM IR code, but potentially also crashes or incorrect results.

- The loop may not involve differentiable variables with enabled gradients.
  See the section on :ref:`differentiating loops <diff-loop>` to see how to
  work around this limitation. Enoki will raise an exception if differentiable
  variables are detected.

- Certain Enoki operations trigger an immediate kernel evaluation. These
  include

  - Horizontal operations: ``ek::all``, ``ek::hsum``, etc..

  - Reading from an unevaluated variable that was the target of a scatter
    operation.

  - Virtual function calls involving arrays of instance pointers

  - Other access to array contents, e.g. a ``print()`` statement.

  You are not allowed to do any of the above, both within the ``ek::Loop``
  condition and the loop body. Enoki will raise an exception when a kernel
  evaluation is triggered while recording a loop.

- The loop condition should not have side effects. In particular, ``while
  (loop.cond(i < 10)) { i += 1; .. }`` is okay, but ``while (loop.cond(i++ <
  10)) { .. }`` is not. Simply move the side effect to the loop body.

- The loop is allowed to read any variable that was before or inside the loop.
  However, writing variables requires extra precautions:

  - **Local variables**: You don't need to do anything special when your loop
    writes to a local variable that does not propagate information between loop
    iterations. However, stashing this variable somewhere and accessing it
    later on outside of the loop is not allowed (it's not local in that case).

  - **Loop variables**: Variables that propagate state between iterations, or
    from inside to outside of the loop are called *loop variables*. They must
    be passed to the ``enoki::Loop`` constructor so that Enoki can insert
    instructions that ensure the correct flow of computed information.

  - **Scatter operations**: the target of a scatter operation (``ek::scatter``
    and ``ek::scatter_add``) does not count as a loop variable despite being
    the target of a write, and it should not be passed to the
    ``ek::Loop`` constructor. For the reasons outlined above regarding kernel
    evaluation, you cannot access an array using both scatter and read/gather
    instructions within the same loop.

For an example of temporary and loop variables, see the following simple C++
example of a loop that measures the number of iterations needed to reach the
value 1 in the sequence underlying the `Collatz conjecture
<https://en.wikipedia.org/wiki/Collatz_conjecture>`_. It makes use of a local
variable ``is_even``, and only declares two loop variables ``value`` and
``cond`` that are both written and read in each iteration.

.. code-block:: cpp

    using UInt32 = CUDAArray<uint32_t>;

    // Collatz conjecture: count # of iterations to reach 1
    UInt32 collatz(UInt32 value) {
        UInt32 counter = 0;
        ek::Loop loop(value, counter);
        while (loop.cond(ek::neq(value, 1))) {
            mask_t<UInt32> is_even = ek::eq(value & 1, 0);
            value = ek::select(is_even, value / 2, 3*value + 1);
            counter++;
        }
        return counter;
    }

There is a major complication in Python that does not appear in C++: an
assignment statement (``a = b``) does not overwrite the contents of ``a``.
Instead, it modifies the local scope to refer to the new value while updating
reference counts. This is normally perfectly, but here it interferes with
``ek.Loop``'s ability to understand how a variable was modified by a
symbolically executed loop iteration (the original ``a`` will appear
unchanged!)

To avoid this issue in Python, you can use the `.assign()` member of the Enoki
array class. It is not needed for in-place updates like ``+=``.

.. code-block:: python
   :emphasize-lines: 9, 10

    import enoki as ek
    from enoki.cuda import UInt32, Loop

    def collatz(value: UInt32):
        counter = UInt32(0)
        loop = Loop(value, counter)
        while (loop.cond(ek.neq(value, 1))):
            is_even = ek.eq(value & 1, 0)
            value.assign(ek.select(is_even, value // 2, 3*value + 1))
            counter += 1
        return counter
