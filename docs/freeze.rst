.. py:currentmodule:: drjit

.. _freeze:

Function Freezing
=================

.. warning:: This feature is still considered experimental, please refer to the
   sections on :ref:`pitfalls <pitfalls>` and :ref:`unsupported operations
   <unsupported_operations>` section. If you encounter any issues please open
   an issue `here <https://github.com/mitsuba-renderer/drjit/issues>`__ with
   a minimal reproducer.

Introduction
------------

Dr.Jit traces code to obtain computation graphs that are later compiled into
into executable machine code. Compilation can be very costly due to a number of
sophisticated optimizations performed by LLVM/CUDA backends. Fortunately,
compilation is often avoidable thanks to a cache of previously compiled
kernels, which leaves tracing as the main source of overheads.

While tracing costs are often negligible, there are situations where this part
actually ends up dominating. This can happen when the program evaluates complex
expressions with relatively little data so that tracing takes longer than the
actual kernel runtime. This can be especially problematic when the code runs
repeatedly, e.g., as part of an optimization loop.

Dr.Jit's *function freezing* feature addresses this performance bottleneck
using the :py:func:`@dr.freeze <freeze>` decorator. Calls to functions using
this decorator query a cache and potentially avoid tracing altogether. The
first time such a function is called, Dr.Jit will analyze the inputs and then
trace its body, taking note of all kernel launches. On subsequent calls, Dr.Jit
only checks that the new inputs are still compatible with the previously
recorded kernels. In that case, it skips tracing and assembly and launches the
kernels directly.

Usage
-----

Using this feature is as simple as annotating the function with the
:py:func:`@dr.freeze <freeze>` decorator:

.. code-block:: python
   :emphasize-lines: 11

   import drjit as dr
   from drjit.cuda import Float, UInt32

   # Without freezing - traces every time
   def func(x):
       y = seriously_complicated_code(x)
       dr.eval(y) # ..intermediate evaluations..
       return huge_function(y, x)

   # With freezing - traces only once
   @dr.freeze
   def frozen(x):
       y = seriously_complicated_code(x)
       dr.eval(y) # ..intermediate evaluations..
       return huge_function(y, x)

Calls to :py:func:`@dr.freeze <freeze>`-decorated functions still involves
small overheads related to examining their inputs, mapping them to function
outputs, and performing checks to ensure correctness. However, these costs are
proportional to the number of function inputs/outputs rather than the
complexity of the computation.

For debugging purposes, the freezing feature can easily be disabled by setting
the :py:attr:`drjit.JitFlag.KernelFreezing` to ``False``.

.. code-block:: python

   @dr.freeze
   def func(x):
      ...

   # By default the function is recorded and replayed on subsequent calls.
   func(x)

   # Function freezing can be disabled by setting a flag to False. Subsequent
   # calls will not use the recording and run the function as if it was not
   # annotated.
   dr.set_flag(dr.JitFlag.KernelFreezing, False)
   func(x)

To re-enable function freezing, the flag can simply be set to ``True`` again.
Previous recordings made while the flag was set will still be available and
can be used when replaying the function.

Additional arguments can be specified when using the decorator. These are
documented in the API-level documentation :py:func:`@dr.freeze <freeze>`.

More implementation details are given :ref:`below
<freezing_implementation_details>`.

.. _unsupported_operations:

Unsupported operations
----------------------

Frozen functions only support operations that can be replayed seamlessly
with new inputs. We describe *unsupported* operations below.

Array access
~~~~~~~~~~~~

Frozen functions can accept arbitrary :ref:`PyTrees <pytrees>` as input, which
ultimately consist of the following leaf elements:

- Scalar Python variables (``int``, ``str``, etc.). The freezing feature makes a
  note of their value and detects changes in subsequent calls. Because they can
  influence the generated kernel code, any changes here trigger re-tracing of
  the function body.

- Dr.Jit arrays. The contents of evaluated JIT array arguments may change
  between calls without requiring re-tracing.

The contents of Dr.Jit array variables will generally change when a frozen
function is later replayed. Operations that extract scalar array elements,
(e.g, to influence control flow) are not legal, since the freezing would bake
the observed constants and decisions into the generated kernels instead of
responding to changes in subsequent replays. Dr.Jit detects such attempts and
raises an exception.

.. code-block:: python

   @dr.freeze
   def func(x: Float, y: Float):
      # Depending on the content of x, one of two possible kernels could be generated.
      # This cannot be replayed. Accessing elements of x is therefore prohibited.
      if x[1] > 0:
         return y + 1
      else:
         return y - 1

   func(Float(0, 1), Float(0, 1, 2))

.. _non_recordable_operations:

Unsupported low-level operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The freezing feature captures regular Dr.Jit kernel launches, reductions, and
various data movement operations. It does not track operations that fall
outside of this pattern, such as custom CUDA kernel launches.

Some parts of Dr.Jit itself currently remain unsupported. For example,
constructing a new texture as in

.. code-block:: python

   @dr.freeze
   def func(data, pos):
      tex = Texture1f([dr.width(data)], 1)
      tex.set_value(data)  # <--- unsupported!
      return tex.eval(pos)

does not produce correct results. This is because initialization of device
textures requires a call to a CUDA texture-specific operation
(``cuMemcpy2DAsync``), which is essentially a custom (non-Dr.Jit) kernel.
Another unsupported case is building acceleration data structures for GPU ray
tracing. Such steps must be performed outside of the frozen function.

Note: Dr.Jit provides basic abstractions to capture even such steps in principle,
so it possible that these limitations will be lifted in the future.

Gradient propagation
~~~~~~~~~~~~~~~~~~~~

Tracing the backward pass of differentiable computation is often at least as
complex as the forward pass. Freezing derivatives is therefore desirable. The
:py:func:`@dr.freeze <freeze>` decorator supports propagating gradients within
the function and can propagate gradients to variables that the function's
inputs depend on.

However, propagating gradients from the result of a frozen function *through*
the function is not supported. All gradient backpropagation has to start
within the recorded function.

In terms of automatic differentiation, annotating a function with the
:py:func:`dr.freeze` decorator is equivalent to wrapping the content with an
isolated gradient scope.

.. code-block:: python

   @dr.freeze
   def func(y):
      # Some differentiable operation...
      z = dr.mean(y)
      # Propagate the gradients to the input of the function...
      dr.backward(z)

   x = dr.arange(Float, 3)
   dr.enable_grad(x)

   y = dr.square(x)

   # The first time the function is called, it will be recorded and the correct
   # gradients will be accumulated into x.
   func(y)

   y = x * 2

   # On subsequent calls the function will be replayed, and gradients will
   # be accumulated in x.
   func(y)

The :py:func:`@dr.freeze <freeze>` decorator adds an implicit
:py:func:`drjit.isolate_grad` context to the function. The above function is
then equivalent to the following function.

.. code-block:: python

   def func(y):
      # The isolate grad scope is added implicitly by the freezing decorator
      with dr.isolate_grad():
         # Some differentiable operation...
         z = dr.mean(y)
         # Propagate the gradients to the input of the function...
         dr.backward(z)


Compress
~~~~~~~~

Compress operations (:py:func:`drjit.compress`) generate results, whose size
(number of entries) depends on the content of the input. Therefore the output
size cannot be determined ahead of time. Using :py:func:`drjit.compress` with
any other function that needs to know array sizes in advance will cause the
function to be re-traced on every call, effectively rendering the freezing
mechanism useless.

Examples of such functions include :py:func:`drjit.block_reduce`,
:py:func:`drjit.block_prefix_reduce`, and :py:func:`drjit.scatter_reduce` when
using the LLVM backend.

.. code-block:: python

   @dr.freeze
   func(x):
      y = dr.block_reduce(dr.ReduceOp.Add, x, 2)
      return dr.compress(y > 2)

   # Calling the function the first time, will cause it to be traced.
   x = dr.arange(Float, 4)
   func(x)

   # Successive calls will also re-trace the function, even when called with the
   # same input. A warning will also be printed, to notify of such cases.
   x = dr.arange(Float, 4)
   func(x)


Pointers with offsets
~~~~~~~~~~~~~~~~~~~~~

The following comment mainly applies to custom C++ code using Dr.Jit.

Internally, new inputs to pre-recorded kernels are passed using the variables'
data pointer. This is also how variables are identified and disambiguated
in the function freezing implementation.

However, this identification mechanism will not work for addresses pointing
*inside* of a memory region. Therefore, such pointers are not supported inside
of frozen functions.

.. code-block:: cpp

   // This pattern is not supported inside of frozen functions.
   UInt32::load_(x.data() + 4)

Note that this pattern might be used in existing C++ code which is called inside
of the frozen function, which would result in an exception.


.. _pitfalls:

Pitfalls
--------

Watch out for following pitfalls when using :py:func:`@dr.freeze <freeze>` decorator.

Implicit inputs
~~~~~~~~~~~~~~~

A class can hold JIT arrays as members, and its methods can use them. Likewise,
a function can access variables of the outer scope (closures). These types of
implicit inputs to a frozen function are generally not supported:

.. code-block:: python

   class MyClass:
      def __init__(self, state: Float):
         self.state = state

      @dr.freeze
      def method(self, a: Float):
         # The `self.state` variable is an implicit input to the frozen function.
         # Attempting to record this function will raise an exception!
         return self.state + a

   ...

   local_var = Float([1, 2, 3])
   def func(a: Float):
      # `local_var` is an implicit input to the frozen function (closure variable).
      return local_var + a

   @dr.freeze
   def func2(b: Float):
      return func(b) + b

   # This will raise an exception. Closure variables are not supported except
   # in the most straightforward cases.
   func2(Float([4, 5, 6]))

When freezing such a method or function, these implicit inputs need to be made
visible to the freezing mechanism. There are two recommended ways to do so:

1. Turn the class into a valid :ref:`PyTree <pytrees>`, e.g., a dataclass
   (:py:class:`@dataclass`) or a  ``DRJIT_STRUCT``.

2. Or, use the ``state_fn`` argument of the :py:func:`@dr.freeze <freeze>` decorator to
   manually specify the implicit inputs. ``state_fn`` will be called as a
   function with the same arguments as the annotated function, and should return
   a tuple of all extra inputs to be considered when recording and replaying.

The following snippet illustrates correct usage:

.. code-block:: python

   @dataclass
   class MyDataClass:
      # Dataclasses are valid PyTrees, which make these fields visible to Dr.Jit
      # and the freezing mechanism.
      x: Float
      y: Float

      @dr.freeze
      def func(self, z: Float):
         return self.y + z

   def other_func(obj: MyDataClass, z: Float):
      return obj.x + obj.y + x

   ...

   class OpaqueClass:
      def __init__(self, x: Float):
         # This field is not visible to Dr.Jit.
         self.x = x

   # The ``state_fn`` argument can be used to make implicit inputs visible
   # without modifying the class.
   @dr.freeze(state_fn=(lambda obj, **_: obj.x))
   def func(obj: OpaqueClass):
      return obj.x + 1



Kernel size inference
~~~~~~~~~~~~~~~~~~~~~

As explained above, frozen functions can in general be called many times with
JIT inputs of varying sizes (number of elements) without requiring re-tracing.

In some situations, the size of an input may be used to determine the size of
another variable:

.. code-block:: python

   @dr.freeze
   def func(x):
      indices = dr.arange(UInt32, dr.width(x) // 2)
      # The size of the result depends on the size of input `x`.
      return dr.gather(type(x), x, indices)

The freezing mechanism uses a simple heuristic to detect variables whose size
is a direct multiple or fraction of the input size.

.. code-block:: python

   # When calling the function, Dr.Jit will notice that the size of the output
   # is a whole fraction of the input. This fact will be recorded by the frozen
   # function.
   x = dr.arange(Float, 8)
   y1 = func(x)
   assert dr.width(y1) == 4

   # When replaying the function with a differently sized input, the size of
   # the resulting variable will be derived according to this fraction.
   x = dr.arange(Float, 16)
   y2 = func(x)
   assert dr.width(y2) == 8

Unfortunately, if this heuristic does not succeed (e.g., creating a variable with 3
more entries than the input), the size of the new variable will be assumed to be
a constant, and will always be set to the size observed during the first recording,
even in subsequent calls.

.. warning::

   Because there is no way for Dr.Jit to reliably detect it, freezing a function
   containing this pattern can result in unsafe code or undefined behavior! In
   particular, there may be out-of-bounds accesses due to the incorrect variable
   size.

.. code-block:: python

   @dr.freeze
   def func(x):
      # The size of `indices` is not a simple multiple or fraction of the size
      # of input `x`.
      indices = dr.arange(UInt32, dr.width(x) - 1)
      return dr.gather(type(x), x, indices)

   # When first calling the function with an input of size 8, the constant size
   # of (8 - 1) = 7 is baked into the frozen function.
   x = dr.arange(Float, 8)
   y1 = func(x)

   # When replaying the function, a kernel of the hardcoded size 7 be replayed,
   # resulting in an incorrect output. This is unsafe!
   x = dr.arange(Float, 16)
   y2 = func(x)

When more than one variable are accessed using :py:func:`drjit.gather` or
:py:func:`drjit.scatter`, and the kernel size has to be inferred, it is
possible that Dr.Jit picks the wrong variable to base the kernel size on.
Such cases might also lead to undefined behavior and may cause out-of-bounds
memory accesses. In general, Dr.Jit will try to use the largest variable that
is either a fraction or multiple of the kernel input size.

.. code-block:: python

   @dr.freeze
   def func(x, y):
      # The size of `indices` is not a simple multiple or fraction of the size
      # of input `x`.
      indices = dr.arange(UInt32, dr.width(x) // 2)
      return dr.gather(type(x), x, indices) + dr.gather(type(y), y, indices)

   # When calling the function, Dr.Jit will notice, that the size of the output
   # is a whole fraction of the size of ``x`` as well as ``y``.
   x = dr.arange(Float, 8)
   y = dr.arange(Float, 16)
   z1 = func(x, y)
   assert dr.width(z1) == 4

   # When replaying the function, Dr.Jit will use the larger of the two inputs
   # to determine the size of the output.
   x = dr.arange(Float, 16)
   y = dr.arange(Float, 32)
   z2 = func(x, y)
   assert dr.width(z2) == 8

Excessive recordings
~~~~~~~~~~~~~~~~~~~~

A common pattern when rendering scenes or running an optimization loop is to use
the iteration index, e.g., as a seed to initialize a random number generator.
This is also supported in a frozen function, however passing the iteration count
as a plain Python integer will cause the function to be re-recorded each time,
resulting in lower performance than not using frozen functions.

.. code-block:: python

   @dr.freeze
   def func(scene, it):
      return render(scene, seed = it)

   for i in range(n):
      # When this function is called with different int-typed seed values, the
      # frozen function will be re-traced for each new value of ``i``!
      func(scene, i)

   for i in range(n):
      # Re-tracing can be prevented by using an opaque JIT variable instead.
      i = dr.opaque(UInt32, i)
      func(scene, i)


Auto-opaque
~~~~~~~~~~~

There is one more subtlety when using a *literal* JIT variable (:py:obj:`UInt32(i)`)
instead of an opaque one (:py:obj:`dr.opaque(UInt32, i)`). The "auto-opaque"
feature, which is enabled by default, will detect literal JIT inputs that
change between calls and make them opaque. However, this means that the function
has to be traced at least twice, which incurs additional overhead at the start.

.. code-block:: python

   for i in range(n):
      # By default, this literal JIT variable (non-opaque) will be made opaque
      # when passed to the frozen function at the second call only.
      # This means the function is traced twice instead of once.
      i = UInt32(i)
      func(scene, i)

Disabling auto-opaque (:py:obj:`drjit.freeze(auto_opaque=False)`) will result
in a single recording, but all literal inputs will be made opaque regardless of
whether they would later remain constant or not. This will lead to higher memory
usage and may also worsen performance of the kernel itself.

When possible, it is therefore recommended to **use opaque JIT variables for
inputs that are known to change across calls**.

To help track changing inputs, Dr.Jit can provide a list of such changing
literals and their "paths" in the input arguments if they are detected:

.. code-block:: python

   # For the literal "paths" to be printed the log level has to be set to ``Info``
   dr.set_log_level(dr.LogLevel.Info)

   @dr.freeze
   def frozen(x, y, l, c):
      return x +  1
      ...

   # Members of classes will be printed
   @dataclass
   class MyClass:
      z: Float

   # We call the function twice. The first call will leave all literals untouched.
   # In the second call, changing literals will be detected and their paths will
   # be printed.
   for i in range(2):
      x = dr.arange(Float, i+2)
      y = Float(i)
      l = [Float(1), Float(i)]
      c = MyClass(Float(i))

      # The function can be called with arguments and keyword arguments. They will
      # show up as a tuple in the path.
      frozen(x, y, l, c = c)

The above code will print the following message, when the function is called the second time:

.. code-block:: text

   While traversing the frozen function input, new literal variables have
   been discovered which changed from one call to another. These will be made
   opaque, and the input will be traversed again. This will incur some
   overhead. To prevent this, make those variables opaque in beforehand. Below,
   a list of variables that changed will be shown.
   args[1][0]: The literal value of this variable changed from 0x0 to 0x3f800000
   args[2][1][0]: The literal value of this variable changed from 0x0 to 0x3f800000
   kwargs["c"].z[0]: The literal value of this variable changed from 0x0 to 0x3f800000

This output can be used to determine which literal where made opaque.
As stated above, it can be beneficial to make these literals opaque beforehand.
In this case, the second argument of the function, the second argument of the
list and the member ``z`` of the class have been detected as changing literals.


Dry-run replay
~~~~~~~~~~~~~~

Some operations, such as block reductions, require the recording to be replayed
in a dry-run mode before executing it. This calculates the size of variables and
ensures that it will be possible to replay the recording later. If such a
dry-run fails, the function will have to be re-traced, however instead of adding
a new recording to the function, the old one will be overwritten. It is not
possible to add another recording, to the cache, since the condition that
causes a dry-run to fail can be dependent on the size (number of elements) of
JIT input variables, which is allowed to change.

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

   # Calling the function in a loop with changing input sizes can cause all
   # dry-runs to fail, rendering the freezing mechanism useless.
   for i in range(5, 10):
      x = dr.arange(Float, i)
      y = func(x)

A warning will be printed after more than 10 iterations have been re-traced.
This limit can be changed using the ``warn_after`` argument of the decorator.

Such functions should therefore be used with caution and only called with
inputs that do not lead to a dry-run failure.

Tensor shapes
~~~~~~~~~~~~~

When a frozen function is called with a tensor, the first dimension of the
tensor is assumed to be dynamic. It can change from one call to another without
triggering re-tracing of the function. However, changes in any other dimension
will cause it to be re-traced.

This is due to the way tensors are indexed: computing indices to access tensor
entries does not involve the first (outermost) dimension, which makes it
possible to reuse the same code as long as trailing dimensions do not change.

.. code-block:: python

   @dr.freeze
   def func(t: TensorXf, i: UInt, j: UInt, k: UInt):
      # Indexes into the tensor array, getting the entry at (row, col)
      index = i * dr.shape(t)[1] * dr.shape(t)[2] + j * dr.shape(t)[2] + k
      return dr.gather(Float, t.array, index)

   # The first call will record the function
   t = TensorXf(dr.arange(Float, 10*7*3), shape=(10, 7, 3))
   func(t, UInt(1), UInt(1), UInt(1))

   # Subsequent calls with the same trailing dimensions will be replayed
   t = TensorXf(dr.arange(Float, 25*7*3), shape=(25, 7, 3))
   func(t, UInt(1), UInt(1), UInt(1))

   # Changes in trailing dimensions will cause the function to be re-traced
   t = TensorXf(dr.arange(Float, 10*3*7), shape=(10, 3, 7))
   func(t, UInt(1), UInt(1), UInt(1))

Dr.Jit also supports advanced tensor indexing, allowing you to use arrays to
index into a tensor e.g. ``t[UInt(1, 2, 3), :]``. This syntax can also be
used inside of frozen functions, however it might lead to kernels with baked-in
kernel sizes, and therefore incorrect outputs. If tensor indexing with indices
of changing sizes is required, calculating the array index manually with the
formula in the above example is recommended.

.. code-block:: python

   @dr.freeze
   def func(t: TensorXf, i: UInt, j: UInt, k: UInt):
      # Indexes into the tensor array, getting the entry at (row, col)
      return t[i, j, k]

   t = TensorXf(dr.arange(Float, 10*7*3), shape=(10, 7, 3))

   # The first call will record the function, and will return a tensor of shape
   # (3, 2, 1)
   func(t, UInt(1, 2, 3), UInt(1, 2), UInt(1))

   # Calling the function with a different number of index elements will be
   # correct, as long as only the array with the largest number of indices
   # changes.
   func(t, UInt(1, 2, 3, 4), UInt(1, 2), UInt(1))

   # Calling the function with a different number of index elements on multiple
   # dimensions can lead  to incorrect outputs. The heuristic will use the larger
   # array to infer the size of the kernel, by multiplication with the recorded
   # fraction (in this case 2). This call will (incorrectly) return a tensor of
   # shape (4, 2, 1).
   func(t, UInt(1, 2, 3, 4), UInt(1, 2, 3), UInt(1))


.. warning::
   Using indexing or slicing inside of a frozen function can easily lead to
   baked-in kernel sizes and as a result to incorrect outputs without any
   warning. This should be used with caution when replaying frozen functions
   with JIT inputs of varying sizes (number of elements).

Textures
~~~~~~~~

:ref:`Textures <textures>` can be used inside of frozen functions for lookups,
as well as for gradient calculations. However because they require special
memory operations on the CUDA backend, it is not possible to update or
initialize CUDA textures inside of frozen functions.
This is a special case of :ref:`non-recordable operation <non_recordable_operations>`.

.. code-block:: python

   @dr.freeze
   def func(tex: Texture1f, pos: Float):
     return tex.eval(pos)

   tex = Texture1f([2], 1)
   tex.set_value(t(0, 1))

   pos = dr.arange(Float, 4) / 4

   # The texture can be evaluated inside the frozen function.
   func(tex, pos)


Indirect function calls
~~~~~~~~~~~~~~~~~~~~~~~

As symbolic indirect function calls are generally supported by frozen functions.
However, some limitations apply. The following example shows a supported use of
indirect function calls in frozen functions.

.. code-block:: python

   # `A` and `B` derive from `Base`
   a, b = A(), B()

   @dr.freeze
   def func(base: BasePtr, x: Float):
      return base.f(x)

   base = BasePtr(a, a, None, b, b)
   x = Float(1, 2, 3, 4, 5)
   func(base, x)

When a frozen function is called with a variable that can point to a virtual
base class, Dr.Jit's pointer registry is traversed to find all variables used
in the frozen function call. Since some objects can be registered, but not
referenced by the pointer, member JIT variables of these objects are traversed
**and evaluated**, even though they are not used in the function.
This side-effect can be unexpected.

.. code-block:: python

   # `A` and `B` derive from `Base`
   # These objects are registered with Dr.Jit's pointer registry
   a, b = A(), B()

   @dr.freeze
   def func(base: BasePtr, x: Float):
      return base.f(x)

   # Even though only `a` is referenced, we have to traverse member variables
   # of `b`. These can be evaluated by the frozen function call.
   base = BasePtr(a, a, None)
   x = Float(1, 2, 3, 4, 5)
   func(base, x)

Nested indirect function calls are supported when the inner base class pointer
is passed as an argument to the outer function. However, due to implementation
details nested calls are not supported when the outer function retrieves the
callee pointer from class member variables

.. code-block:: python

   # Even though `A` is traversable, a frozen function with a call to
   # ``nested_member`` will fail.
   class A(Base):
      DRJIT_STRUCT = {
         "s": BasePtr,
      }

      s: BasePtr

      def nested(self, s, x):
         s.f(x)

      def nested_member(self, x):
         self.s.f(x)

   a, b = A(), B()

   # This nested vcall is supported because the nested base pointer is an
   # argument to the nested function.
   @dr.freeze
   def supported(base: BasePtr, nested_base: BasePtr, x: Float):
      return base.nested(nested_base, x)

   a.s = BasePtr(b)
   dr.make_opaque(a.s)

   # This nested vcall is unsupported because the nested base pointer is an
   # opaque member of the class `A`.
   @dr.freeze
   def unsupported(base: BasePtr, x: Float):
      return base.nested_member(x)

Runaway recursion
~~~~~~~~~~~~~~~~~

Passing inputs to a frozen function that contain basic reference cycles is
supported. However, reference cycles going through C++ classes can lead to a
runaway recursion when traversing the function inputs, and raise an exception.

.. code-block:: python

   @dr.freeze
   def frozen(l):
      return l[0] + 1

   # This constructs a list with a reference cycle.
   l = [Float(1)]
   l.append(l)

   # Passing an object with a simple reference cycle is supported.
   frozen(l)

However, this more complex example shows an *unsupported* case of reference cycles that
can occur when using custom BSDFs in Mitsuba 3.

.. code-block:: python

   # A class inheriting from a trampoline class is automatically traversed.
   class MyBSDF(mi.BSDF):
      def set_scene(self, scene):
         self.scene = scene
      ...

   @dr.freeze
   def frozen(scene):
      ...

   # Construct a scene that includes ``MyBSDF`` as an element.
   scene = ...
   # Setting the scene reference in the BSDF completes the reference cycle.
   mybsdf.set_scene(scene)

   # Calling the function with such an object, will lead to a runaway
   # recursion, and the frozen function will raise an exception.
   frozen(scene)


.. _freezing_implementation_details:

Implementation details
----------------------

Every time the annotated function is called, its inputs are analyzed. All JIT
variables are extracted into a flattened and de-duplicated array. Additionally,
a key describing the "layout" of the inputs is generated. This key will be used
to distinguish between different recordings of the same frozen function, in case
some of its inputs qualitatively change in subsequent calls.

If no recording is found for the current key, Dr.Jit enters a "kernel recording"
mode (:py:obj:`drjit.JitFlag.FreezingScope`) and the actual function code is
executed. In this mode, all device level operations, such as kernel launches are
recorded as well as executed normally.

The next time the function is called, the newly-provided inputs are traversed,
and the layout is used to look up compatible recordings. If such a recording is
found, any tracing is skipped: the various recorded operations and kernels are
directly replayed.

Traversal
~~~~~~~~~

In order to map the variables provided to a frozen function as arguments to the
actual kernel inputs (slots), Dr.Jit must be able to traverse these arguments.
In addition to basic Python containers such as lists, tuples and dictionaries,
the following :ref:`PyTrees <pytrees>` are traversable and can be part of the
input of a frozen function.

*Dataclasses* are traversable by Dr.Jit and their fields are automatically made
visible to the traversal algorithm.

.. code-block:: python

   # Fields of dataclasses are traversable
   @dataclass
   class MyClass:
      x: Float

Classes can be annotated with a static ``DRJIT_STRUCT`` field to make classes
traversable.

.. code-block:: python

   class MyClass:
      x: Float

      # Annotating the class with DRJIT_STRUCT will make the members listed
      # available to traversal.
      DRJIT_STRUCT = {
         "x": Float
      }

Finally, C++ classes may additionally implement the ``TraversableBase`` class
to make them traversable. Python classes, inheriting from these classes through
trampolines are automatically traversed. This is useful when implementing your
own subclasses with indirect function calls.

.. code-block:: python

   # If BSDF is a traversable trampoline class,
   # then member variables of MyClass will also be traversed.
   class MyClass(mi.BSDF):
      x: Float


Output construction
~~~~~~~~~~~~~~~~~~~

After a frozen function has been replayed, the outputs of the replayed operation
(kernel launches, reductions, etc) have to be mapped back to outputs of the
frozen function, respecting the layout observed in the first launch.

Since this output must be constructible at replay time, only a subset of
traversable types can be returned from frozen functions. This includes:

- JIT and AD variables,
- Dr.Jit Tensors and Arrays,
- Python lists, tuples and dictionaries,
- Dataclasses,
- ``DRJIT_STRUCT`` annotated classes with a default constructor.

The following example shows an *unsupported* return type: because the constructor
of ``MyClass`` expects a variable, an object of type ``MyClass`` cannot be
created at replay time.

.. code-block:: python

   class MyClass:
      x: Float

      DRJIT_STRUCT = {
         "x": Float,
      }

      # Non-default constructor (requires argument `x`)
      def __init__(self, x: Float):
         self.x = x

   @dr.freeze
   def func(x):
      return MyClass(x + 1)

   # Calling the function will fail, as the output of the frozen function
   # cannot be constructed without a default constructor.
   func(Float(1, 2, 3))
