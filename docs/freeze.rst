.. py:currentmodule:: drjit

.. _freeze:

Function Freezing
=================

This feature is still experimental, and we list a number of unsupported cases
in the :ref:`pitfalls` section. If you encounter any issues please feel free to
open an issue `here <https://github.com/mitsuba-renderer/drjit/issues>`__.

Introduction
------------

When working with Dr.Jit, your code typically is first traced, to obtain a
computation graph of the operations you intended to perform. When calling
:py:func:`drjit.eval`, this graph is assembled into an intermediary
representation, either LLVM IR or CUDA PTX. Finnaly, this assembly is compiled
into actual binary code that can be run on the specified hardware. This last
step can be very expensive, since the underlying compilers perform a lot of
optimization on the intermediary code. Dr.Jit therefore caches this step by
default using a hash of the assembled IR code. As mentioned in the :ref:`_eval`
page, changing literal values can cause re-compilation of the kernel and result
in a significant performance bottleneck. Memoization of compilation
significantly reduces the overhead that otherwise would be encountered.
However, the first two steps of tracing the Python code and generating the
intermediary representation can still be expensive. This feature tries to
address this performance bottleneck, by introducing the :py:func:`drjit.freeze`
decorator. If a function is annotated with this decorator, Dr.Jit will try to
cache the tracing and assembly steps as well. When a frozen function is called
the first time, Dr.Jit will analyze the inputs, and then trace the
function once, capturing all kernels lauched. On subsequent calls to the
function Dr.Jit will try to find previous recordings with compatible input
layouts. If such a recording is found, it will launch it instead of re-tracing
the function. This skips tracing and assembly of kernels, as well as
compilation, reducing the time spent not executing kernels.

.. code-block:: python

   import drjit as dr
   from drjit.cuda import Float, UInt32

   # Without freezing - traces every time
   def func(x):
       # Complex operations...
       y = x + 1
       dr.eval(y)
       z = x * 2
       return z

   # With freezing - traces only once
   @dr.freeze
   def frozen(x):
       # Same complex operations...
       y = x + 1
       dr.eval(y)
       z = x * 2
       return z


How Function Freezing Works
---------------------------

Every time the function is called, the input is analyzed and all JIT variables
are extracted into a flat-deduplicated array. Additionally, a key of the layout
in which the variables where stored in the input is generated. The key is used
to find recordings of previous calls to the function in a hashmap. If none are
found, the inner function is called and the backend is put into a recording
mode. In this mode, all device level operations, such as kernel launches are
record. When the function is called again, the input is traversed, and the
layout is used to lookup compatible recordings. If such a recording is found,
it is used to replay the kernel launches.

Traversal
~~~~~~~~~

In order to map the variables provided to a frozen function in its inputs to
the to the kernel slots, Dr.Jit has to be able to traverse the input of the
function. In addition to basic python containers such as lists, tuples and
dictionaries, the following containers are traversable and can be part of the
input of a frozen function.

*Dataclasses* are traversable by Dr.Jit and their fields are automatically made
visible to the traversal algorithm.

.. code-block:: python

   # Fields of dataclasses are traversable
   @dataclass
   class MyClass:
      x: Float

Classes can be annotated with a static *DRJIT_STRUCT* field to make classes
traversable.

.. code-block:: python

   class MyClass:
      x: Float

      # Annotating the class with DRJIT_STRUCT will make the members listed
      # available to traversal.
      DRJIT_STRUCT = {
         "x": Float
      }

Classes inheriting from trampoline classes are automatically traversed. This is
useful when implementing your own subclasses with vcalls.

.. code-block:: python

   # If BSDF is a traversable trampoline class member variables of MyClass will
   # also be traversed.
   class MyClass(BSDF):
      x: Float


Unsupported Operations
----------------------

Since frozen functions record kernel launches and have to be able to replay
them later, certian operations are not supported inside frozen functions.

Array Access
~~~~~~~~~~~~

The input of a frozen function can consist of two kinds of variables. Cached
variables such as Python integers, strings etc. These are able to influence
which kernels are recorded. Opaque JIT variables are able to change from one
call to another without requiring re-tracing of the function. The recorded
kernels can therefore not depend on the content of such variables. To prevent
incorrect outputs, reading the contents from such variables is prohibited
inside of a frozen function.

.. code-block:: python

   @dr.freeze
   def func(x, y):
      # Depending on the content of x, one or the other kernel would be generated.
      # This cannot be replayed and accessing x is therefore prohibited.
      if x[1] > 0:
         return y + 1
      else:
         return y - 1

   x = Float(0, 1)
   y = Float(0, 1, 2)

   func(x, y)


Non Recordable Operations
~~~~~~~~~~~~~~~~~~~~~~~~~

Whenever a device level operation is called inside a frozen function, Dr.Jit
has to be made aware of it. Kernel launches and other common operations such as
reductions, are supported by hooking into a low-level abstraction in the core
library. Applying any operation, not known to Dr.Jit, on the memory underlying
a variable is not supported and might result in incorrect outputs or
exceptions. Such operations are used in the initialization of CUDA textures or
acceleration structure building in Mitsuba3.

.. code-block:: python

   @dr.freeze
   def func(data, pos):
      # On CUDA backends, this will call ``cuMemcpy2DAsync`` on the texture
      # memory, without notifying the frozen function, and therefore fail.
      tex = Texture1f([dr.width(data)], 1
      tex.set_value(data)
      return tex.eval(pos)

   data = Float(0, 1)
   pos = Float(0.3, 0.6)
   func(data, pos)


Offset Pointers
~~~~~~~~~~~~~~~

When recording a frozen function, kernel inputs are handled through the
pointers, held by evaluated variables. To find the variable associated with a
pointer, this pointer has to point to the start of a memory region. Therefore,
handling of pointers pointing into the middle of such memory regions is not
supported.

..code-block:: cpp

   # This pattern is not supported inside of frozen functions.
   UInt32::load_(x.data() + 4)

This pattern might be used in C++ code called by the frozen function and can
result in exceptions being raised.

.. _pitfalls:

Pitfalls
--------

When using the :py:func:`drjit.freeze` decorator, certain caveats have to be
considered. The following section will explain the most common pitfalls that
can be encountered when using this feature.

Hidden Inputs
~~~~~~~~~~~~~

When calling a frozen function with a custom class, that contains Array
variables, these need to be made visible to the frozen function. Dr.Jit
provides two options to do so:

1. Annotating the class with the :py:func:`dataclass` decorator
2. Adding a static ``DRJIT_STRUCT`` dictionary to the class definition, that
   specifies its members, and types

If some of the inputs are not discoverable, Dr.Jit might raise exceptions when
recording a frozen function. Additional variables can be provided using the
optional ``state_fn`` argument of the :py:func:`drjit.freeze` decorator. It
is called before the frozen function is either recorded or replayed, and
receives the same inputs as the annotated function. It should return a
traversible variant, such as a tuple, of the hidden variables.

.. code-block:: python

   class PartiallyDiscoverable:
      x: Float
      y: Float

      DRJIT_STRUCT = {
         "x": Float,
         # ``y`` is not discoverable by Dr.Jit
      }
      ...

   # Calling the frozen function with a partially traversible input, and
   using hidden variables in the function results in exceptions being raised.
   @dr.freeze
   def func(x)
      return x.y + 1

   input = PartiallyHidden(x, y)
   func(input)

   # The ``state_fn`` argument can be used to make the hidden variables visible
   # without re-defining the input class.
   @dr.freeze(state_fn = lambda input, **__: input.y)
   def func(x)
      return x.y + 1

   input = PartiallyHidden(x, y)
   func(input)

Excessive Recordings
~~~~~~~~~~~~~~~~~~~~

A common pattern, used when rendering scenes or optimizing them is to use the
iteration index as a seed, to initialize a random number generator. This is
also supported in a frozen function, however passing the iteration count as a
plain Python integer will cause the function to be recorded multiple times,
resulting in lower performance than not using frozen functions.

.. code-block:: python

   @dr.freeze
   def func(scene, it):
      return render(scene, seed = it)

   for i in range(n):
      # When this function is called with different seed values, the frozen
      # function will be re-traced, because the input key has changed.
      func(scene, i)

   for i in range(n):
      # Re-tracing can be prevented by using a JIT variable instead.
      func(scene, UInt32(i))

By default the argument ``auto_opaque`` of :py:func:`drjit.freeze` is set to
``True``. This will indicate to the frozen function that only changing JIT
literals should be made opaque. In the above case, the frozen function will
realize that the second argument of the function is a Literal and that it
changed from the first to the second call. Therefore it will be made opaque in
subsequent calls. This causes one more recording to be generated than would be
necessary, however it prevents excessive memory consumption. Dr.Jit will
provide a list of such changing literals if they are detected, as well as the
python paths to find them. To prevent this extra recording, the changing input
can be made opaque inbeforehand.

.. code-block:: python

   for i in range(n):
      # Re-tracing can be prevented by using a JIT variable instead.
      seed = UInt32(i)
      dr.make_opaque(seed)
      func(scene, seed)


Dry Run Replay
~~~~~~~~~~~~~~

Some operations, such as block reductions require the recording to be replayed
in a dry-run mode before executing it. This calculates the size of variables
and ensures that it is possible to replay the recording. If a dry-run failed,
the function will have to be re-traced, however instead of adding a new
recording to the function, the old one will be overwritten.

.. code-block:: python

   dr.freeze
   def func(x):
      return dr.block_reduce(dr.ReduceOp.Add, x, 2)

   # The first time the function is called, a new recording is made
   x = dr.arange(Float, 4)
   y = func(x)

   # The block reduction will require a dry-run before launching kernels. In
   # this case, it is detected that the size of x is not divisible by 2. The
   # function will be re-traced, however this will overwrite the old recording.
   x = dr.arange(Float, 5)
   y = func(x)

Using such functions with changing input sizes could cause excessive re-tracing
of the underlying function.

Prefix sums will generate variables, whos size is dependent on the content of
the variable. Therefore the output size cannot be determined in a dry-run.
Using such reductions with any other function that requires a dry-run will
always be re-traced.

.. code-block:: python

   @dr.freeze
   func(x):
      y = dr.block_reduce(dr.ReduceOp.Add, x, 2)
      return dr.prefix_sum(y)

   # Calling the function the first time, will cause it to be traced.
   x = dr.arange(Float, 4)
   func(x)

   # Succesive calls will also re-trace the function, even when called with the
   # same input. A warning will also be printed, to notify of such cases.
   x = dr.arange(Float, 4)
   func(x)

Kernel Size Inference
~~~~~~~~~~~~~~~~~~~~~

Dr.Jit supports calling frozen functions with varying input sizes without
requiring the function to be re-traced. If the input size is used to determine
the size of variables created within the function, Dr.Jit must infer the
appropriate kernel sizes when replaying the function. When the size of a
created variable is a direct multiple or fraction of the input size, Dr.Jit
uses a heuristic to estimate the output size.

.. code-block:: python

   @dr.freeze
   def func(x):
      return dr.gather(type(x), x, dr.arange(UInt32, dr.width(x) / 2))

   # When calling the function, Dr.Jit will notice that the size of the output
   # is a whole fraction of the input. This fact will be recorded by the frozen
   # function.
   x = dr.arange(Float, 8)
   y1 = func(x)

   # When replaying the function with a differently sized input, the size of
   # the resulting variable will be estimated according to this fraction.
   x = dr.arange(Float, 16)
   y1 = func(x)

On the other hand, if the output size is computed using other operations, such
as subtraction, the kernel size is fixed during subsequent calls to the
function. The resulting output is therefore undefined, and some kernels might
access memory regions outside of the input bounds.

.. code-block:: python

   @dr.freeze
   def func(x):
      return dr.gather(type(x), x, dr.arange(UInt32, dr.width(x) - 1))

   # When first calling the function with an input of size 4, the kernel size
   # is embedded into the frozen function.
   x = dr.arange(Float, 8)
   y1 = func(x)

   # When replaying the function, a kernel with the same size will be replayed,
   # resulting in an output of size 4 being computed.
   x = dr.arange(Float, 16)
   y1 = func(x)


Unsupported Closures
~~~~~~~~~~~~~~~~~~~~

Even though functions with closures are supported in general, certain cases
might not be. When calling a function with a closure, the frozen function
decorator will try to find the closure variables of the function to pass them
as part of the input.

.. code-block:: python

   spp = 2

   @dr.freeze
   def func(scene, it):
      # Using the nonlocal variable spp is supported
      image = render(scene, spp = spp)

      # The reference image can also be a non-local variable
      loss = loss_fn(image, ref_image)

      loss.backward()

      return loss, image

   for i in range(1000):
      loss, image = func(scene, UInt32(it))

      opt.step()

      if i == 700:
         # Changing the closure variable will cause the function to be
         # re-traced once
         spp = 4

However, the decorator is not able to walk the closures of functions or methods
called from outer functions. The reason is that these might be dynamically
determined at runtime, and inspecting them is not always possible. In such
cases, some variables might be missed during traversal, and either cause
incorrect output if a changing Python variable was missed or raise an exception
if a JIT variable was missed.

.. code-block:: python

   y = 1

   def inner(x):
      # The variable y will be missed during traversal, causing incorrect
      # output if it changes between calls.
      return x + y

   @dr.freeze
   def outer(x):
      return inner(x)

Unsupported Inputs
~~~~~~~~~~~~~~~~~~



Virtual Function Calls
~~~~~~~~~~~~~~~~~~~~~

As symbollic virtual function calls are an important feature of Dr.Jit they are
supported by frozen functions however, some limitations apply. The following
example shows a supported use of virtual function calls in frozen functions.

.. code-block:: python

   a, b = A(), B()

   @dr.freeze
   def func(base, x):
      return base.f(x)

   base = BasePtr(a, a, None, b, b)
   x = Float(1, 2, 3, 4, 5)

   func(base, x)

In the above case, all JIT variables, used in the frozen function, including
inside of the virtual function calls.

*TODO:* nested vcalls

