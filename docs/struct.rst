.. cpp:namespace:: drjit

Custom data structures
======================

Many Dr.Jit operations can be applied to custom data structures, causing them to
recursively propagate through all of the data structure's fields. The remainder
of this sections explains several use cases of this functionality, and how to
enable it via a suitable :c:macro:`DRJIT_STRUCT` declaration. This feature
requires the following optional header file:

.. code-block:: cpp

    #include <drjit/struct.h>

Motivation
----------

One of the main purposes of Dr.Jit is to convert a piece of software into a
corresponding "wide" vectorized version that processes many inputs at once.
Simply replacing all scalar types (e.g. ``float``, ``int32_t``) by Dr.Jit arrays
may be enough to accomplish this goal in some cases. However, this strategy
tends to fail when the program relies on more complex types. Consider the
following example:

.. code-block:: cpp

    struct MyStruct {
        float a[3];
        int b;
        SomeOtherStruct c;
    };

    MyStruct data = /* .. */;
    if (condition)
        data = func(data);

Once vectorized, ``data`` will contain many parallel evaluations, in which case
``condition`` may be simultaneously ``true`` and ``false`` for different
entries. Another common situation involves read or write access to elements of
an array:

.. code-block:: cpp

    MyStruct *storage = /* .. */;
    uint32_t index = /* .. */;
    MyStruct data = storage[index];

After vectorization, ``index`` turns into an array of indices, and the element
access needs to be converted into an equivalent *gather* operation that fetches
many entries in parallel. Of course, all of this could be accomplished by
inserting numerous calls (one *per field*) to Dr.Jit's :cpp:func:`select()` and
:cpp:func:`gather()` functions, but this would be *tedious*.


How it works
------------

Following the pattern outlined in this section, `MyStruct` should instead be
declared as follows:

.. code-block:: cpp

    // MyStruct parameterized by representative element
    template <typename Float> struct MyStruct {
        // Derive suitable types from 'Float'
        using Array3f = dr::Array<Float, 3>;
        using UInt32  = dr::uint32_array_t<Float>;

        // Field declarations
        Array3f a;
        UInt32 b;
        SomeOtherStruct<Float> c;

        // Inform Dr.Jit about the members of 'MyStruct'
        DRJIT_STRUCT(MyStruct, a, b, c)
    };


Note in particular the following changes:

1. ``MyStruct`` is now a *template* that is parameterized by a representative
   member type (``Float`` in this case, but that choice was arbitrary).

2. Importantly, all other types occurring within ``MyStruct`` are now
   *derived* from ``Float``, for example by

   - building larger arrays (``Array3f``).

   - changing the type underlying an array via traits like
     :cpp:type:`uint32_array_t`.

   - instantiating other custom types (``SomeOtherStruct<Float>``) following
     the same pattern.

2. The :c:macro:`DRJIT_STRUCT` declaration at the end informs Dr.Jit about the
   data structure's fields.

Benefits
--------

This new template version of ``MyStruct`` is slightly longer, but it is also
significantly more general. First, it adds compatibility for the various
backends of Dr.Jit. For example,

- ``MyStruct<float>`` reproduces the original behavior.

- ``MyStruct<Packet<float>>`` results in a *structure of arrays* (SoA) version
  that represents entries using SIMD registers.

- ``MyStruct<DiffArray<CUDAArray<<float>>>`` will JIT-compile kernels
  that run on CUDA-capable GPUs, while keeping track of derivatives.

Second, the :c:macro:`DRJIT_STRUCT` declaration at the end makes the type
transparent to :ref:`various standard operations <struct-supported>`.

For instance, consider the previous ``if``-guarded assignment that only made
sense in scalar mode

.. code-block:: cpp

    MyStruct data = /* .. */;
    if (condition)
        data = func(data);

This can now be turned into a *masked* assignment that correctly handles
vectorization:

.. code-block:: cpp

    data[condition] = func(data);

Note that this is essentially syntax sugar to avoid having to write a long
sequence of equivalent assignments of the form

.. code-block:: cpp

    MyStruct temp = func(data);
    data.a = dr::select(condition, data.a, temp.a);
    data.b = dr::select(condition, data.b, temp.b);
    // ... (one per field) ...


.. note::

    **Loops and virtual function calls**: When a custom data structure is an
    argument or return value of a :ref:`virtual function call
    <virtual-functions>`, or when it is a loop variable of a :ref:`symbolic
    loop <recording-loops>`, then Dr.Jit must inspect the data structure's
    individual fields. In such cases, an :c:macro:`DRJIT_STRUCT` declaration is
    mandatory.

.. _struct-supported:

Interface (C++)
---------------

In the following, suppose that the following declarations are available:

.. code-block:: cpp

   using Float    = dr::CUDAArray<float>;
   using UInt32   = dr::CUDAArray<uint32_t>;
   using Mask     = dr::CUDAArray<bool>;
   using MyStruct = ::MyStruct<Float>;

   Mask mask;
   UInt32 index;
   MyStruct x, y, z;

A number of operations support recursive propagation through custom data
structures.

1. **Initialization**: :cpp:func:`zero()`, and :cpp:func:`empty()`. Example:
   dynamic allocation of a data structure with 1000 entries:

   .. code-block:: cpp

       x = dr::empty<MyStruct>(1000);

2. **Mask-based selection**: The function :cpp:func:`select()` can blend
   the fields of two data structures based on a provided mask.

   .. code-block:: cpp

       z = dr::select(mask, x, y);

3. **Masked assignment**: :cpp:func:`masked()` and the indexing operator.

   The :c:macro:`DRJIT_STRUCT` macro installs a convenient ``operator[]`` overload
   that can be used to perform mask-based assignment

   .. code-block:: cpp

       x[x.b < 0] = dr::zero<MyStruct>();

   The following alternative syntax is also provided.

   .. code-block:: cpp

       dr::masked(x, x.b < 0) = dr::zero<MyStruct>();

   This second variant is more portable to other situations: for example
   ``var[mask] = ..`` does not compile when ``var`` is a builtin C++ type like
   ``int``, but the :cpp:func:`masked()` variant still works.

4.  **Vectorized scatter/gather**: :cpp:func:`scatter()`,
    :cpp:func:`scatter_add()`, and :cpp:func:`gather()`.

    The following code gathers a number of elements and scatters them back

    .. code-block:: cpp

        y = dr::gather<MyStruct>(/* source = */ x, index, mask);

        dr::scatter(/* target = */ x, /* source = */ y, index, mask);

5. **Operations specific to dynamic arrays**:

   The size of a dynamic data structure can be queried using
   :cpp:func:`width()` and changed using :cpp:func:`resize()`.

6. **Operations specific to JIT (CUDA/LLVM) arrays**:

   - **Scheduling/evaluation**: Passing a custom data structure to
     :cpp:func:`schedule()` or :cpp:func:`eval()` causes all fields to be
     scheduled or simultaneously evaluated.

   - **Migration**: The function :cpp:func:`migrate()` can migrate entire data
     structures between different memory regions (device/host/managed memory,
     etc.)

7. **Operations specific to differentiable arrays**:

   - Enabling and disabling gradients: :cpp:func:`grad_enabled()`,
     :cpp:func:`enable_grad()`, :cpp:func:`disable_grad()`, and
     :cpp:func:`set_grad_enabled()`.

   - Suspending and resuming gradients: :cpp:func:`grad_suspended()`,
     :cpp:func:`suspend_grad()`, :cpp:func:`resume_grad()`, and
     :cpp:func:`set_grad_suspended()`.

   - Getting and setting gradients: :cpp:func:`grad()`,
     :cpp:func:`set_grad()`, and :cpp:func:`accum_grad()`.

   - Returning a copy that is detached from the AD graph: :cpp:func:`detach()`.

   - Scheduling data structures for forward/reverse-mode traversal:
     :cpp:func:`enqueue()`.

8. **Other**: Custom data structures can be passed through :ref:`virtual
   function calls <virtual-functions>`, and they can be used as loop variables
   in :ref:`symbolic loops <recording-loops>`.

Adding support to further operations is easy, and patches to this end are
welcomed.

Pairs and tuples (C++)
----------------------

The mechanism for traversing custom data structures including all of the
operations discussed above, is fully compatible with the ``std::pair`` and
``std::tuple`` standard containers without the need for any additional
declarations.

Interface (Python)
------------------

Custom data structures are also supported in the Python bindings, though the
:c:macro:`DRJIT_STRUCT` specification takes on a different form here. In a
class defined within Python, you will need to specify a top-level static
attribute documenting the fields and their types. It is also important for that
class to be constructible using the default constructor (e.g. no arguments).

.. code-block:: python

    from drjit.cuda import UInt32, Array3f

    class MyStruct:
        DRJIT_STRUCT = { 'a' : Array3f, 'b' : UInt32 }

        def __init__(self, a=Array3f(), b=UInt32()):
            self.a = a
            self.b = b

In classes exposed via `pybind11 <https://pybind11.readthedocs.io>`_, follow
the following pattern:

.. code-block:: cpp

    auto mystruct = py::class_<MyStruct>(m, "MyStruct")
        .def(py::init<>()) // default constructor (important!)
        .def_readwrite("a", &MyStruct::a)
        .def_readwrite("b", &MyStruct::b);

    py::dict fields;
    fields["a"] = py::type::of<Array3f>();
    fields["b"] = py::type::of<Float>();

    mystruct.attr("DRJIT_STRUCT") = fields;

The set of compatible operations is currently much smaller than in the C++
interface.

1. **Initialization**: :cpp:func:`zero()`, and :cpp:func:`empty()`.

2. **Mask-based selection**: :cpp:func:`select()`.

3.  **Vectorized scatter/gather**: :cpp:func:`scatter()`,
    :cpp:func:`scatter_add()`, and :cpp:func:`gather()`.

4. **Operations specific to dynamic arrays**: :cpp:func:`width()` and
   :cpp:func:`resize()`.

5. **Operations specific to JIT (CUDA/LLVM) arrays**: :cpp:func:`schedule()`
   and :cpp:func:`eval()`.

6. **Operations specific to differentiable arrays**:
   :cpp:func:`grad_enabled()`, :cpp:func:`enable_grad()`, :cpp:func:`disable_grad()`,
   :cpp:func:`set_grad_enabled()`, :cpp:func:`suspend_grad()`, :cpp:func:`resume_grad()`,
   :cpp:func:`set_grad_suspended()`, :cpp:func:`set_grad()`, :cpp:func:`accum_grad()`, and :cpp:func:`detach()`.

7. **Other**: Custom data structures can be passed through :ref:`virtual
   function calls <virtual-functions>`, and they can be used as loop variables
   in :ref:`symbolic loops <recording-loops>`.

Adding support to further operations is easy, and patches to this end are
welcomed.

C++ Reference
-------------

.. c:macro:: DRJIT_STRUCT(Name, ...)

    This macro makes a data structure transparent to Dr.Jit so that operations
    can propagate through the various fields. It must be specified *within* a
    templated ``struct`` or ``class`` declaration, and its first argument
    (``Name``) must repeat the data structure's name. The remaining arguments
    (``...``) must be the names of its fields (in any order, though declaration
    order should be preferred for clarity).

    .. warning::

        Dr.Jit assumes that the data structure can be moved and copied like
        ordinary data, and it explicitly specifies that default variants of

        - default constructor
        - copy assignment constructor and operator
        - move assignment constructor and operator

        must be used. In particular, the beginning of the macro expands into

        .. code-block:: cpp

            Name() = default;
            Name(const Name &) = default;
            Name(Name &&) = default;
            Name &operator=(const Name &) = default;
            Name &operator=(Name &&) = default;

        You will likely encounter compiler errors if your code contains
        duplicates or custom variations of these declarations.
