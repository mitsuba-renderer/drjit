.. cpp:namespace:: drjit

C++ Interface
=============

The C++ and Python interfaces of Python are designed to be as similar as
possible, to the extent that this is possible given the different natures of
these two languages. The following subsections detail noteworthy differences.

Virtual function calls
----------------------

Given a custom class ``T``, Dr.Jit can dispatch function calls to arrays of
``T`` instances. To do so, you must include the header file

.. code-block:: cpp

   #include <drjit/vcall.h>

To enable virtual function call dispatch for a new class, several changes to
your implementation are necessary. First, the constructors and destructor of
the class must register/unregister instances with the Dr.Jit instance registry.

.. code-block:: cpp

    struct T {
        using Float = CUDAArray<float>;

        T() {
            jit_registry_put(dr::backend_v<Float>, "T", this);
        }

        virtual ~T() { jit_registry_remove(this); }

        /// Suppose this is a function implemented by subclasses of the ``T`` interface.
        virtual Float f(Float x) = 0;
    };

The call to ``jit_registry_put`` must pass the backend (which can be
manually specified or determined from a Dr.Jit array type via
:cpp:var:`drjit::backend_v`), a class name, and the ``this`` pointer.

Next, you must use the following macros to describe the interface of the type.
(These should appear at the top level and outside of any namespaces)

.. code-block:: cpp

   DRJIT_VCALL_BEGIN(T)
       DRJIT_VCALL_METHOD(f)
       // Specify other methods here
   DRJIT_VCALL_END()

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
   the orginal type. There is no need to specify return values, argument types,
   or multiple overloads. Just be sure to list each function that you want to
   be able to call on a Dr.Jit instance arrays.

.. c:macro:: DRJIT_VCALL_GETTER(Name)

   This is an optimized form of the above macro that should be used when the
   function in question is a *getter*. This refers to a function that does not
   take in put arguments, and which is pure (i.e., causes no side effects). The
   implementation can then avoid the cost of an actual indirect jump.

Following these declarations, the following code performs a vectorized virtual
function call.

.. code-block:: cpp
   
   dr::CUDAArray<T*> instances = ...;
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
