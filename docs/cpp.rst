.. cpp:namespace:: drjit

C++ Interface
=============

The C++ and Python interfaces of Python are designed to be as similar as
possible, to the extent that this is possible given the different natures of
these two languages. The following subsections detail noteworthy differences.

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

   #include <drjit/vcall.h>

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

   DRJIT_VCALL_BEGIN(Foo)
       DRJIT_VCALL_METHOD(f)
       // Specify other methods here
   DRJIT_VCALL_END()

There is no need to specify return values, argument types, or multiple
overloads. Just be sure to list each function that you want to be able to call
on a Dr.Jit instance arrays. Below is an overview of the available macros:

.. c:macro:: DRJIT_VCALL_BEGIN(Name)

   Demarcates the start of an interface block. The `Name` parameter must refer
   to the type in question, and the ``jit_registry_put`` call mentioned above
   should provide a string version of `Name` (including namespace prefixes).

.. c:macro:: DRJIT_VCALL_TEMPLATE_BEGIN(Name)

   A variant of the above macro that should be used when ``Name`` refers to a
   template class.

.. c:macro:: DRJIT_END()

   Demarcates the end of an interface block.

.. c:macro:: DRJIT_VCALL_METHOD(Name)

   Indicates to Dr.Jit that `Name` is the name of a method provided by
   the orginal type.

.. c:macro:: DRJIT_VCALL_GETTER(Name)

   This is an optimized form of the above macro that should be used when the
   function in question is a *getter*. This refers to a function that does not
   take in put arguments, and which is pure (i.e., causes no side effects). The
   implementation can then avoid the cost of an actual indirect jump.

Following these declarations, the following code performs a vectorized (virtual)
function call.

.. code-block:: cpp

   dr::CUDAArray<Foo*> instances = ...;
   Float x = ....;
   Float y = instances->f(x);

All of the commentary about function calls in Python (see
:py:func:`drjit.switch()`) applies here as well. The call can be done
symbolically, using wavefronts, and it can propagate derivatives in forward and
reverse mode.

Masks passed as the last function argument are treated specially and apply to
the entire operation. Masked elements of the call effectively don't perform the
function call at all, and their return value is zero. Side effects performed by
the called functions are also disabled for these elements.

It is legal to perform a function call on an array containing ``nullptr``
pointers. These elements are considered to be masked as well.

Exposing instance arrays in Python
----------------------------------

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

    using FooPtr = dr::DiffPtr<JitBackend::CUDA, Foo *>;

    dr::ArrayBinding b;
    auto base_ptr = dr::bind_array_t<FooPtr>(b, m, "FooPtr")
        .def("f", [](FooPtr &self, Float a) { return self->f(a); })
    base_ptr.attr("Domain") = "Foo";

The ``Domain`` attribute at the end should match the name passed to
``jit_registry_put`` and enables use of the instance array with
:py:func:`drjit.dispatch`.
