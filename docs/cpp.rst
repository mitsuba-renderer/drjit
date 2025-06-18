.. cpp:namespace:: drjit

.. _cpp_iface:

C++ Interface
=============

The C++ and Python interfaces of Python are designed to be as similar as
possible. The following subsections explain how to translate from one language
to the other along with a few unavoidable differences.

Use as a header library
-----------------------

By default, Dr.Jit operates as a header-only library without further
dependencies.


.. code-block:: cpp

   #include <drjit/array.h>

   namespace dr = drjit;

   using Array3f  = dr::Array<float, 3>;
   using Matrix3f = dr::Matrix<float, 3>;

   int main(int, char**) {
       Array3f x(1, 2, 3),
               y(1, 0, 1);

       Array3f z = dr::normalize(dr::cross(x, y)) * .5f;
       printf("Result = %s\n", dr::string(z).c_str());
   }

Vectorized loops
----------------

Analogous to the vectorized loop interface in Python
(:py:func:`drjit.while_loop`), it is possible to run symbolic and evaluated
loops in C++ via the :cpp:func:`drjit::while_loop()` function. Use of this API
requires the header file

.. code-block:: cpp

   #include <drjit/while_loop.h>

An example usage is shown below, which is essentially a translation of the
Python interface to C++.

.. code-block:: cpp

   using Int = dr::CUDAArray<int>;

   Int i = ..., j = ....;

   std::tie(i, j) = dr::while_loop(
       // Initial loop state
       std::make_tuple(i, j),

       // while(..) condition of the loop
       [](const UInt &i, const UInt &) {
           return i < 5;
       },

       /// Loop body
       [](UInt &i, UInt &j) {
           i += 1;
           j = j * j;
       }
   );

The short-hand notation provided through the :py:func:`@drjit.syntax
<drjit.syntax>` decorator is not available in C++.

This syntax can be cumbersome when the loop state consists of many variables,
since the associated type and argument names must be repeated several times. In
this case, it can be preferable to locally define a custom ``struct``
representing the entirety of the loop state. See the explanation of
:ref:`custom traversable types <custom_types_cpp>` for details.
Dr.Jit optimizes away superfluous loop state variables, so there isn't any
harm in specifying loop state that isn't actually modified by the loop.

The detailed interface of this function is as follows:

.. cpp:function:: template <typename State, typename Cond, typename Body> std::decay_t<State> while_loop(State&& state, Cond &&cond, Body &&body, const char * label = nullptr)

   This function takes an instance ``state`` of the tuple type ``State`` (which
   could be a ``std::pair``, ``std::tuple``, or the lighter-weight alternative
   :cpp:class:`drjit::tuple` created via :cpp:func:`drjit::make_tuple`).

   It invokes the loop body ``body`` with an unpacked version of the tuple elements
   (i.e., ``body(std::get<0>(state), ...)``) until the *loop condition*
   ``cond(std::get<0>(state), ...)`` equals ``false``.

   When the loop condition returns a scalar C++ ``bool``, the operation
   compiles into an ordinary C++ loop. When it is a Dr.Jit array, the loop
   either runs in *symbolic* or *evaluated* mode. Please see the Python
   equivalent of this function (:py:func:`drjit.while_loop`) for details on
   what this means.

   The ``label`` argument can be used to optionally specify a human-readable
   name that will be included in both low-level IR and GraphViz output.

   Both ``cond`` and ``body`` may specify arbitrary callables (lambda
   functions, types with a custom ``operator()`` implementation). When such
   callables capture state from the surrounding call frame, it is important to
   note that Dr.Jit's AD system may need to re-evaluate the loop at a later
   time, at which point the function which originally called
   :cpp:func:`drjit::while_loop` has itself returned. The `&alpha` variable
   captured by reference below would lead to undefined behavior in this case
   (i.e., it would likely crash your program).

   .. code-block:: cpp

      int step = 123;

      dr::while_loop(
          ...
          /// Loop body
          [&step](UInt &i) {
              i += step;
              ...
          }
          ...
      );

   Instead, capture relevant variable state *by value* or include it as part of
   ``state``. Dr.Jit will move the two functions (``cond`` and ``body``
   including captured state) into a persistent object that will eventually be
   released by the AD backend when it is no longer needed.

Vectorized conditionals
-----------------------

Analogous to the vectorized conditional statement interface in Python
(:py:func:`drjit.if_stmt`), it is possible to evaluate symbolic and evaluated
conditionals in C++ via the :cpp:func:`drjit::if_stmt()` function. Use of this API
requires the header file

.. code-block:: cpp

   #include <drjit/if_stmt.h>

An example usage is shown below, which is essentially a translation of the
Python interface to C++.

.. code-block:: cpp

   using Int = dr::CUDAArray<int>;

   Int i = ..., j = ....;

   Int abs_diff = dr::if_stmt(
       // 'args': arguments to forward to 'true_fn' and 'false_fn'
       std::make_tuple(i, j),

       // 'cond': conditional expression
       i < j,

       // 'true_fn': to be called for elements with 'cond == true'
       [](UInt i, UInt j) {
           return j - i;
       }

       // 'false_fn': to be called for elements with 'cond == false'
       [](UInt i, UInt j) {
           return i - j;
       }
   );

The argument ``args`` must always be a tuple that will be unpacked and passed
as arguments of ``true_fn`` and ``false_fn``. The return value of these
function can be any tree of arbitrarily nested arrays, tuples, and other
:ref:`custom data structures <custom_types_cpp>`.

The short-hand notation provided through the :py:func:`@drjit.syntax
<drjit.syntax>` decorator is not available in C++.

As with the loop, the C++ syntax can be cumbersome when there are many
input/output variables, in which case it is preferable to locally define custom
``struct`` types representing all inputs/outputs, or both. Dr.Jit later
optimizes away superfluous variables of :py:func:`drjit.if_stmt`, so there
isn't any harm in being redundant.

The detailed interface of this function is as follows:

.. cpp:function:: template <typename Args, typename Mask, typename Body> auto if_stmt(Args&& state, const Mask &cond, TrueFn &&true_fn, FalseFn &&false_fn, const char * label = nullptr)

   This function takes an instance ``args`` of the tuple type ``Args`` (which
   could be a ``std::pair``, ``std::tuple``, or the lighter-weight alternative
   :cpp:class:`drjit::tuple` created via :cpp:func:`drjit::make_tuple`).

   It invokes ``true_fn`` and ``false_fn`` with an unpacked version of the
   tuple elements (i.e., ``true_fn(std::get<0>(state), ...)``) and combines
   them based on the values of ``cond``.

   When the loop condition returns a scalar C++ ``bool``, the operation
   compiles into an ordinary C++ conditional statement. When it is a Dr.Jit
   array, the loop either runs in *symbolic* or *evaluated* mode. Please see
   the Python equivalent of this function (:py:func:`drjit.if_stmt`) for
   details on what this means.

   The ``label`` argument can be used to optionally specify a human-readable
   name that will be included in both low-level IR and GraphViz output.

   The arguments ``true_fn`` and ``false_fn`` can be used to pass arbitrary
   callables (lambda functions, types with a custom ``operator()``
   implementation). When such callables capture state from the surrounding call
   frame, it is important to note that Dr.Jit's AD system may need to
   re-evaluate the conditional statement at a later time, at which point the
   function which originally called :cpp:func:`drjit::if_stmt` has itself
   returned. The `&step` variable captured by reference below would lead to
   undefined behavior in this case (i.e., it would likely crash your program).

   .. code-block:: cpp

      int step = 123;

      dr::if_stmt(
          ...
          /// true_fn
          [&step](UInt i) {
              return i + step;
          }
          ...
      );

   Instead, capture relevant variable state *by value* or include it as part of
   ``args``. Dr.Jit will move the two functions (``true_fn`` and ``false_fn``
   including captured state) into a persistent object that will eventually be
   released by the AD backend when it is no longer needed.

Vectorized method calls
-----------------------

Given a pointer ``Foo*`` to a user-defined type ``Foo``, a common operation in
C++ entails dispatching a *method* or *virtual method* call:

.. code-block:: cpp

   Foo *ptr = ...;
   float result = ptr->method(arg_1, arg_2, ...);

Dr.Jit also supports this operation in a *vectorized* form to dispatch method
or virtual method calls to a large set instances in parallel:

.. code-block:: cpp

   using FooPtr = dr::CUDAArray<Foo *>;
   using Float = dr::CUDAArray<float>;

   FooPtr ptr = ...;
   Float result = ptrs->f(arg_1, arg_2, ...);

It does so efficiently using at most a single invocation of each callable.

A limitation of array-based method calls is that input/output or output-only
parameters passed using mutable references or pointers are not supported.
Parameters are all inputs, and the function return value is the sole
output---use pairs, tuples, or custom structures to return multiple values.

To enable this functionality for a new class, the following changes to its
implementation are necessary:

First, include the header file

.. code-block:: cpp

   #include <drjit/call.h>

Next, modify the constructors and destructor of the class so that they
register/unregister themselves with the Dr.Jit instance registry.

.. code-block:: cpp

    struct Foo {
        using Float = CUDAArray<float>;

        Foo() {
            jit_registry_put(dr::backend_v<Float>, "Foo", this);
        }

        virtual ~Foo() { jit_registry_remove(this); }

        /// Suppose this is a function implemented by subclasses of the ``Foo`` interface.
        virtual Float f(Float x) = 0;
    };

The call to ``jit_registry_put`` must pass the backend (which can be manually
specified or determined from a Dr.Jit array type via
:cpp:var:`drjit::backend_v`), a class name, and the ``this`` pointer.

Next, you use the following macros to describe the interface of the type. They
must appear at the top level (i.e., outside of classes and namespaces) and
simply list all function names that Dr.Jit should intercept.

.. code-block:: cpp

   DRJIT_CALL_BEGIN(Foo)
       DRJIT_CALL_METHOD(f)
       // Specify other methods here
   DRJIT_CALL_END()

There is no need to specify return values, argument types, or multiple
overloads. Just be sure to list each function that you want to be able to call
on a Dr.Jit instance arrays. Below is an overview of the available macros:

.. c:macro:: DRJIT_CALL_BEGIN(Name)

   Demarcates the start of an interface block. The `Name` parameter must refer
   to the type in question. The ``jit_registry_put`` call in the earlier
   snippet should provide the string-quoted equivalent of `Name` including
   namespace prefixes.

.. c:macro:: DRJIT_CALL_TEMPLATE_BEGIN(Name)

   A variant of the above macro that should be used when ``Name`` refers to a
   template class.

.. c:macro:: DRJIT_CALL_END()

   Demarcates the end of an interface block.

.. c:macro:: DRJIT_CALL_TEMPLATE_INHERITED_BEGIN(Name, Parent)

   A variant of the above macro that should be used when ``Name`` refers to a
   template class that inherits from another template class ``Parent``.
   The class ``Name`` should only be registered under the registry domain of ``Parent``.
   The interface of ``Parent`` must have already been exposed with a call to
   ``DRJIT_CALL_TEMPLATE_BEGIN(Parent) ... DRJIT_CALL_END()``.
   The interface will then be extended with any methods listed in this block.
   Note this feature is only safe to use with polymorphic types, where C++ can
   use RTTI to determine if a particular instance is of type ``Name``.

.. c:macro:: DRJIT_CALL_METHOD(Name)

   Indicates to Dr.Jit that `Name` is the name of a method provided by
   the orginal type.

.. c:macro:: DRJIT_CALL_GETTER(Name)

   This is an optimized form of the above macro that should be used when the
   function in question is a *getter*. This refers to a function that does not
   take in put arguments, and which is pure (i.e., causes no side effects). The
   implementation can then avoid the cost of an actual indirect jump.

Following these declarations, the following code performs a vectorized method
or virtual method call.

.. code-block:: cpp

   dr::CUDAArray<Foo*> instances = ...;
   Float x = ....;
   Float y = instances->f(x);

All of the commentary about function calls in Python (see
:py:func:`drjit.switch()`) applies here as well. The call can be done in
symbolic or evaluated mode, and it supports derivative propagation in forward
and reverse modes.

Masks passed as the last function argument are treated specially and apply to
the entire operation. Masked elements of the call effectively don't perform the
function call at all, and their return value is zero. Side effects performed by
the called functions are also disabled for these elements.

It is legal to perform a function call on an array containing ``nullptr``
pointers. These elements are considered to be masked as well.

Besides vectorizing method calls, there is an alternative call interface named
``drjit::dispatch``:

.. cpp:function:: template <typename Self, typename Func, typename... Args> auto dispatch(const Self& self, const Func &func, const Args&... args)

   This C++ interface is analogous the Python API function
   :py:func:`drjit.dispatch()`. Please review the documentation of the Python
   variant first.

   The C++ interface takes an instance array ``self`` and invokes a provided
   callable ``func`` once for each unique instance. The callable should take a
   scalar instance pointer as first input argument, followed by ``args...``.
   Dr.Jit assembles the traced computation into an indirect jump to one of
   several subroutines.

   The dispatch interface is convenient whenever adding a method or virtual
   method to a class is undesirable. Also, dynamic dispatch is a relatively
   costly operation. When multiple calls are performed on the same set of
   instances, it may be preferable to merge them into a single and potentially
   signficantly faster use of :cpp:func:`drjit::dispatch() <dispatch>`.


The following snippet shows how merge two method calls into a combined call:

.. code-block:: cpp

   dr::CUDAArray<Foo*> instances = ...;
   Float x = ....;

   Float y = dr::dispatch(
       instances,
       [](Foo *self, Float x) { return self->f(x) + self->g(x); },
       x
   );

Python bindings
---------------

Regular arrays
^^^^^^^^^^^^^^

It is possible to register additional bindings for custom array types from
within user code. The test case ``tests/custom_type_ext.cpp`` shows how to
define a ``Color3f`` type that exposes ``.r``, ``.g``, and ``.b`` members.

Instance arrays
^^^^^^^^^^^^^^^

Suppose you have created a C++ type with the following signature:

.. code-block:: cpp

   using Float = dr::DiffArray<JitBackend::CUDA, float>;

   struct Foo {
       virtual Float f(Float input) const = 0;
       virtual ~Foo() = default;
   };

The nanobind description to expose this type in Python is as follows:

.. code-block:: cpp

   nb::class_<Foo>(m, "Foo")
       .def("f", &Foo::f);

It can also be useful to create similar bindings for Dr.Jit ``Foo`` instance
arrays that automatically dispatch function calls to the ``f`` method. To do
so, include

.. code-block:: cpp

   #include <drjit/python.h>

and append the following binding declarations:

.. code-block:: cpp

    using FooPtr = dr::CUDADiffArray<Foo *>;

    dr::ArrayBinding b;
    auto base_ptr = dr::bind_array_t<FooPtr>(b, m, "FooPtr")
        .def("f", [](FooPtr &self, Float a) { return self->f(a); })

.. _custom_types_cpp:

Custom data structures
----------------------

The ability to traverse through members of custom data structures was
previously discussed :ref:`here in the context of Python <custom_types_py>`.

This feature also exists on the C++ side. For this, you must include the header
file

.. code-block:: cpp

   #include <drjit/struct.h>

Following this, you can use the variable-argument ``DRJIT_STRUCT(...)`` macro
to list the available fields.

.. code-block:: cpp

   using Float = dr::CUDADiffArray<float>;

   struct MyPoint2f {
       Float x;
       Float y;

       DRJIT_STRUCT(x, y);
   };

Custom data structures can be defined globally, locally, and they can be
arbitrarily nested. Dr.Jit static arrays, tuples (:cpp:class:`drjit::tuple`),
STL ``std::tuple<...>`` and ``std::pair<T1, T2>`` are all automatically
traversable.

It is also easy to define custom functions that recursively process arbitrary
trees:

.. code-block:: cpp

    template <typename T> void visit_jit_pairs(T &v0, T &v1) {
        if constexpr (dr::is_jit_v<T> && dr::depth_v<T> == 1) {
            /// Do something with 'v0' and 'v1'
        } else if constexpr (dr::is_traversable_v<T>) {
            /// Recurse and try again if the object is traversable
            dr::traverse_2(
                /// Extract the fields of 'v0' and 'v1'
                dr::fields(v0), dr::fields(v1),
                // .. and call the following lambda function on them
                [&](auto &x, auto &y) { visit_jit_pairs(x, y); }
            );
        }
    }

The type trait :cpp:var:`dr::is_traversable\<T\> <drjit::is_traversable_v>`
checks if an instance of a particular type can be traversed.

The helper functions :cpp:func:`drjit::traverse_1`
:cpp:func:`drjit::traverse_2`, :cpp:func:`drjit::traverse_3`, respectively
traverse individual objects, pairs, or triples, at the same time. They take an
arbitrary stateless or stateful lambda function and simply apply it to the
input tuples that can be extracted using the function :cpp:func:`drjit::fields`
that takes a traversable as input.
