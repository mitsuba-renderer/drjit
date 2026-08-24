
.. py:currentmodule:: drjit

.. _freeze:

Function Freezing
=================

In typical usage, Dr.Jit traces a computation, compiles it into executable
machine code, and then launches the resulting kernels on the target device. Of
these three steps, only the kernel launch is truly desired, whereas tracing and
compilation represent *overheads*.

Compilation tends to be the most costly, but can normally be avoided thanks to a
cache of previously compiled kernels. This leaves tracing as the remaining
source of overhead. Although tracing is generally fast, it can dominate when a
program processes a small amount of data using relatively complex expressions.
This is especially problematic when the same code runs repeatedly, for example,
inside an optimization loop.

Function freezing addresses this problem by eliminating the cost of tracing and
compilation on repeated calls. The first call to a function
decorated with :py:func:`@dr.freeze <freeze>` executes the Python function and
creates a recording. Subsequent calls with compatible inputs can replay the
recording directly without executing the function body again. Dr.Jit can
maintain multiple recordings when a function is called with different input
configurations.

Freezing works best for functions that are expensive to trace and whose input
structure remains stable across calls. It provides little benefit when tracing
is inexpensive or when most calls require a new recording.

Basic use
---------

To use this feature, place the computation in a function and apply the
:py:func:`@dr.freeze <freeze>` decorator:

.. code-block:: python
   :emphasize-lines: 4

   import drjit as dr
   from drjit.cuda import Float

   @dr.freeze
   def f(x: Float):
       y = seriously_complicated_function(x)
       dr.eval(y) # ..intermediate evaluations..
       return another_huge_function(y, x)

   value_1 = f(Float(...)) # 🔴 record
   value_2 = f(Float(...)) # ▶️ replay

Methods can be frozen in the same way using :py:func:`@dr.freeze <freeze>`.
Here, ``self`` is treated like any other argument. Dr.Jit only tracks its
attributes when the object is a :ref:`PyTree <pytrees>`. With an ordinary Python
class, changes to these attributes may be missed, causing a recording to replay
with stale values.

Make the class a dataclass or list its traversable members using a
:ref:`DRJIT_STRUCT annotation <custom_types_py>`:

.. code-block:: python

   class Scaler:
       DRJIT_STRUCT = { "scale": Float }

       def __init__(self, scale: Float):
           self.scale = scale

       @dr.freeze
       def apply(self, x: Float):
           return self.scale * x

   scaler = Scaler(Float(2))
   scaler.apply(Float(1))  # returns [2]

   scaler.scale = Float(3)
   scaler.apply(Float(1))  # returns [3]

The frozen function ``f`` behaves like the wrapped Python callable and also
provides the following attributes and methods:

- ``f.enabled``: set this to ``False`` to disable function freezing
  so that calls directly reach the wrapped function.

- ``f.n_recordings``: reports the total number of recordings created.

- ``f.n_cached_recordings``: reports the number of stored recordings,
  which may differ from ``n_recordings`` when cache entries are evicted.

- ``f.clear()``: clears the cache and resets both counters.

The :py:attr:`drjit.JitFlag.KernelFreezing` flag can be used to disable
function freezing globally:

.. code-block:: python

   dr.set_flag(dr.JitFlag.KernelFreezing, False)

Structural compatibility
------------------------

Inputs and return values of frozen functions may be arbitrary
:ref:`PyTrees <pytrees>`, including Dr.Jit arrays, tuples and lists,
dictionaries, dataclasses, or :ref:`custom classes <custom_types_py>` with a
``DRJIT_STRUCT`` annotation.
Custom classes returned by a frozen function must be default-constructible so
that Dr.Jit can recreate them during replay.

Dr.Jit can only replay a recording when the new input is *structurally
equivalent* to the input that produced it. The following examples illustrate
what this means:

The contents and size of a Dr.Jit array may change between calls without
triggering a new recording:

.. code-block:: python

   f(Float(1, 2))     # 🔴 record
   f(Float(3, 4, 5))  # ▶️ replay

When a function has multiple inputs, inputs of the same size form a group. A
recording made with one set of input sizes generalizes to other sizes, as long
as the inputs within each group continue to have the same size.

.. code-block:: python

   f(Float(1, 2),    Float(3, 4))    # 🔴 record
   f(Float(1, 2, 3), Float(4, 5, 6)) # ▶️ replay
   f(Float(1, 2, 3), Float(4, 5))    # ⚠️ miss, record again

All of the following changes will also trigger a new recording:

- Changing the type of an argument or the value of a Python scalar.
  The following snippet does both:

  .. code-block:: python

     f(x, 1)    # 🔴 record
     f(x, 3.0)  # ⚠️ miss, record again

- Replacing a Python object that is not a PyTree with an unequal object.
  Equality is determined using ``__eq__``:

  .. code-block:: python

     f(x, BigInt(1)) # 🔴 record
     f(x, BigInt(2)) # ⚠️ miss, record again

- Changing the length of a Python container or the key set of a dictionary:

  .. code-block:: python

     f(x, [1],    {'y' : 'h'}) # 🔴 record
     f(x, [1, 2], {'_y': 'y'}) # ⚠️ miss, record again

- Changing any Dr.Jit compiler flag:

  .. code-block:: python

     f(x) # 🔴 record
     dr.set_flag(...)
     f(x) # ⚠️ miss, record again

- Changing a tensor's rank or any dimension other than its leading dimension:

  .. code-block:: python

     f(dr.zeros(TensorXf, (10, 20, 30))) # 🔴 record
     f(dr.zeros(TensorXf, (15, 20, 30))) # ▶️ replay
     f(dr.zeros(TensorXf, (18, 24)))     # ⚠️ miss, record again

Repeated recordings
-------------------

By default, a frozen function warns after it has created more than ten
recordings. The ``warn_after`` parameter changes this threshold. Setting it to
``1`` can be useful when diagnosing a function:

.. code-block:: pycon

   >>> @dr.freeze(warn_after=1)
   ... def f(x, iteration):
   ...     return x + iteration
   ...
   >>> f(Float(1, 2), 0)
   >>> f(Float(1, 2), 1)
   This frozen function was traced 2 times. Repeated tracing defeats the purpose
   of function freezing and is caused by structural changes of the function's
   inputs. The change that triggered this recording was: 'iteration' (the value
   changed from 0 to 1).

Here, ``iteration`` is a Python ``int``, whose value forms part of the
recording's configuration. If a value is expected to change between calls,
pass it as an opaque Dr.Jit array to avoid another recording:

.. code-block:: python

   for i in range(n):
       f(x, dr.opaque(UInt32(i)))

Dr.Jit normally detects changing literal arrays such as ``UInt32(i)`` and makes
them opaque automatically. This requires one additional recording. Making a
value that is known to change opaque from the start avoids that cost.

Captured state
--------------

Not every value used by a function appears in its argument list:

.. code-block:: python

   scale = 2

   @dr.freeze
   def f(x):
       return x * scale

   f(Float(1, 2))  # 🔴 record

   scale = 3
   f(Float(1, 2))  # ⚠️ miss, record again

Although ``scale`` is not an argument of ``f``, Dr.Jit detects that the
function reads it and includes its value when selecting a recording. The same
applies to variables defined in an enclosing function, which Python calls
*closure variables*. Dr.Jit also detects such reads inside comprehensions,
generator expressions, lambdas, and nested functions.

Dr.Jit does not automatically detect state accessed through a helper function
or stored in an object that is not a :ref:`PyTree <pytrees>`. Pass such state
as an argument or expose it using the ``state_fn`` argument. The
:py:func:`dr.freeze <freeze>` API reference includes an example of the latter
approach.

Mutable inputs
--------------

A frozen function may update Dr.Jit arrays stored in a mutable
:ref:`PyTree <pytrees>`. Dr.Jit writes the updated values back after replay:

.. code-block:: python

   @dr.freeze
   def increment(state: dict):
       state["value"] = state["value"] + 1

   state = { "value": Float(1, 2) }

   increment(state)  # 🔴 record
   increment(state)  # ▶️ replay

   assert dr.all(state["value"] == Float(3, 4))

A replay assigns new arrays to the leaves of the input, but it cannot reproduce
a change of the input's *structure*, such as adding or removing container
entries or changing Python values stored in the input. The next section covers
the case where such a change happens only once.

Inputs initialized on first use
-------------------------------

A callable often builds part of its input the first time it runs: a dictionary
gains a key, a tensor is reshaped, or an object populates a member on first
use. Such an initialization typically happens once, and the frozen function
recovers from it by recording the callable a second time. The call has already
happened at that point, so the input now holds the initialized state that every
later call sees as well, and the second recording describes the callable in its
steady state.

.. code-block:: python

   @dr.freeze
   def func(d):
      d["y"] = d["x"] + 1

   d = {"x": UInt32(1, 2, 3)}
   func(d)                      # Adds the key, hence two recordings
   assert func.n_recordings == 2

   func(d)                      # ▶️ replay, since the key now exists
   assert func.n_recordings == 2

The callable therefore runs **twice** on the call that triggers this, which is
observable when it advances a random number generator or writes to a file. A
callable that keeps changing the structure of its input on every call cannot be
frozen and raises instead.

Automatic differentiation
--------------------------

Frozen functions can be differentiated as usual:

.. code-block:: python

   @dr.freeze
   def loss(x: Float):
       return dr.mean(dr.square(x))

   x = dr.arange(Float, 4)
   dr.enable_grad(x)

   value = loss(x)
   dr.backward(value)

The first call records the primal computation. The first derivative pass in
each mode records the corresponding derivative computation. Later passes in
that mode can replay it.

A frozen function can also propagate gradients internally. This is useful when
the function computes a loss and the caller only needs the accumulated
gradients:

.. code-block:: python

   @dr.freeze
   def accumulate_gradient(y: Float):
       dr.backward(dr.mean(y))

   x = dr.arange(Float, 4)
   dr.enable_grad(x)

   accumulate_gradient(dr.square(x))

Gradients produced this way propagate through the function's inputs to the
variables on which they depend. The backward pass becomes part of the recording
and is replayed together with the primal computation.

The two forms have different performance characteristics. When
``dr.backward()`` is applied to a frozen function's return value, the
reverse-mode pass evaluates the primal computation again. The primal therefore
runs twice, once when the frozen function is called and once during the
reverse-mode pass. Calling ``dr.backward()`` inside the frozen function records
the primal and reverse computations together and avoids this duplicated work.
Prefer the latter form when the loss can be computed inside the function and
the caller only needs the resulting gradients.

🔪 The sharp bits 🔪
--------------------

Most users will not encounter the limitations below. They become relevant when
a frozen function reads values back to Python, derives array sizes in unusual
ways, or relies on low-level operations outside Dr.Jit's regular kernel
launches.

Reading array values in Python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function freezing rejects operations that transfer the contents of an evaluated
Dr.Jit array from the device to the host and read them into Python. Such a read
can influence Python control flow and therefore change the recorded
computation. In the example below, the value of ``x[0]`` would determine which
of two kernels is generated:

.. code-block:: pycon

   >>> @dr.freeze
   ... def func(x: Float):
   ...    if x[0] > 0:
   ...       return x
   ...    else:
   ...       return -x

   >>> func(Float(0, 1))
   RuntimeError: drjit.cuda.Float.__getitem__(): jit_var_read(): reading the contents of an↵
   evaluated variable while recording a frozen function is not permitted, as this operation↵
   cannot be recorded. See https://drjit.readthedocs.io/en/latest/freeze.html for details.

Array sizes
~~~~~~~~~~~

A recording can usually be replayed with inputs of different widths. When a
function creates an array whose width depends on an input, Dr.Jit must be able
to recover that relationship. Simple integer multiples and fractions are
supported:

.. code-block:: python

   @dr.freeze
   def return_even(x: Float):
       indices = 2 * dr.arange(UInt32, dr.width(x) // 2)
       return dr.gather(Float, x, indices)

Other relationships, such as ``dr.width(x) - 1``, cannot be inferred reliably.
The size observed while recording may then be reused during replay, potentially
causing incorrect results or out-of-bounds accesses. Avoid such relationships
when a frozen function will receive inputs of different widths.

Size inference can also become ambiguous when a kernel reads multiple inputs
of different widths:

.. code-block:: python

   @dr.freeze
   def combine(x: Float, y: Float):
       indices = dr.arange(UInt32, dr.width(x) // 2)
       return (dr.gather(Float, x, indices) +
               dr.gather(Float, y, indices))

   combine(dr.arange(Float, 8),
           dr.arange(Float, 16))  # 🔴 record, returns 4 elements

   combine(dr.arange(Float, 16),
           dr.arange(Float, 16))  # incorrect: still returns 4 elements

During the first call, the output width of four could be described as either
half the width of ``x`` or one quarter the width of ``y``. Dr.Jit may record
the latter relationship. The second call then returns four elements even
though the function derives its indices from ``x`` and should return eight.

Avoid this ambiguity by keeping the relative input sizes fixed across calls.
Otherwise, do not freeze a computation whose output size could be inferred
from more than one input.

Advanced tensor indexing
~~~~~~~~~~~~~~~~~~~~~~~~

Tensor indexing is generally supported, but combining an array index with a
slice requires care. Dr.Jit lowers such an expression to a flat index
calculation while tracing the Python function. This calculation uses the
current length of the index array as a Python integer, which becomes fixed in
the recording:

.. code-block:: python

   @dr.freeze
   def select_rows(t: TensorXf, rows: UInt32):
       return t[rows, :]

   select_rows(t, UInt32(0, 1))     # 🔴 record with two rows
   select_rows(t, UInt32(0, 1, 2))  # incorrect: still assumes two rows

Avoid this pattern when the index array may change size. Computing flat indices
and using :py:func:`dr.gather <gather>` directly avoids the problem.

Unsupported low-level operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function freezing captures regular Dr.Jit operations, including kernel
launches, reductions, and data movement. It cannot capture low-level operations
such as custom CUDA kernel launches.

Existing textures can be passed to frozen functions and used for lookups and
gradient calculations. Creating or updating a texture inside a frozen function
is not supported on the CUDA and Metal backends:

.. code-block:: python

   @dr.freeze
   def func(data, pos):
      tex = Texture1f([dr.width(data)], 1)
      tex.set_value(data)  # <--- unsupported!
      return tex.eval(pos)

Texture initialization uses backend-specific memory operations that are not
captured by function freezing, such as ``cuMemcpy2DAsync`` on CUDA.
Building acceleration data structures for GPU ray tracing is likewise
unsupported and must be done outside the frozen function.
