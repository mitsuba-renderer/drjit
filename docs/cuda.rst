.. cpp:namespace:: drjit

.. _cuda:

Dynamic arrays (CUDA/LLVM)
==========================

The array types discussed so far all operate on fixed-size arrays, such as 3D
vectors or AVX512 packets storing sixteen separate 3D vectors. When the task
at hand involves many more (e.g. millions) of elements, it is often preferable
to work with dynamically sized arrays that can handle vast amounts of data at once.

Available backends
------------------

Dr.Jit provides three different array backends for this purpose:

1. :cpp:struct:`DynamicArray` represents a heap-allocated memory region on the
   CPU not unlike a ``std::vector<T>``.

   It performs arithmetic in an *eager* fashion, which is simple but generally
   leads to extremely poor performance. This is because each and every
   operation  will allocate a temporary array, load source operands, and store
   the result. The cost of these memory accesses greatly exceeds that of the
   underlying arithmetic operation.

   When working with dynamic arrays, it is generally much better to perform
   arithmetic *lazily*, in which case multiple operations can be *fused*
   together to avoid memory traffic caused by temporary results. And that's
   exactly what the next two options do.

2. :cpp:struct:`CUDAArray` parallelizes computation over all `CUDA
   <https://developer.nvidia.com/cuda-zone>`_ cores of compatible NVIDIA GPUs
   (Maxwell or newer). It does so in a *lazy* fashion: what this means is that
   an arithmetic operation like adding two arrays will not be carried out right
   away.

   Instead, Dr.Jit tries to collect as much work as possible until evaluation
   cannot be postponed anymore (e.g. when the user prints the output of a
   computation). At this point, the system generates an efficient fused CUDA kernel that
   contains a transcript of all steps that are needed to obtain the result.

   This *just-in-time* (JIT) compilation step happens transparently: there is
   no need to hand-write CUDA kernels, or even have CUDA installed, for that
   matter. Dr.Jit will look for the graphics driver at runtime and talk to it
   using NVIDIA's *Parallel Thread Execution* (PTX) intermediate representation.


3. :cpp:struct:`LLVMArray` is very similar to the :cpp:struct:`CUDAArray`
   backend but targets the CPU instead of the GPU. It lazily records operations
   using the `LLVM <https://llvm.org/>`_ intermediate representation and
   generates optimized kernels targeting the host processor. If desired, these
   kernels are also parallelized over all CPU cores, which means that just one
   thread issuing computations involving Dr.Jit arrays can keep a large
   multiprocessor system busy.

   There is no compile-time dependency on LLVM when using this backend. Dr.Jit
   will search for a LLVM shared library at runtime, and any non-ancient
   version > 7.0 works.

The remainder of this page only focuses on the JIT-compiled array types
:cpp:struct:`CUDAArray` and :cpp:struct:`LLVMArray`, generally using the CUDA
variant. Unless noted otherwise, these two are completely exchangeable.
The examples are written in Python, and analogous C++ code can be inferred using
the :ref:`conventions <python-cpp-interface>` relating the two language
interfaces.


Lazy Just-In-Time Compilation
-----------------------------

The following examples illustrate the basic operation of Dr.Jit's JIT compiler.
We begin by importing a CUDA array type through the Python bindings followed by
a basic calculation.

.. code-block:: pycon
   :linenos:

   >>> import drjit as dr
   >>> from drjit.cuda import Float
   >>> a = Float(1) + Float(2)
   >>> print(a)
   [3]

This yields no surprises. What is less obvious is that the computation did
not occur in line 3, but rather in line 4 as part the ``print()`` statement. To
understand what is happening here, we can use the :cpp:func:`graphviz()`
function, which visualizes the queued computation graph associated with a
particular variable:

.. code-block:: pycon

   >>> a = Float(1) + Float(2)
   >>> dr.graphviz(a).view()

.. image:: cuda-01.png
    :width: 300px
    :align: center

The graph here consists of two nodes representing the constant literals
followed by an addition. Each node contains a template of an instruction
expressed in in the `PTX
<https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_
intermediate representation, a kind of assembly language that is portable
across NVIDIA GPUs. When the print statement starts to access the array
contents in line 4, this type of lazy execution is no longer possible, at which
point Dr.Jit must *evaluate* the array by compiling and executing a CUDA kernel
containing these three operations.

Other parts of Dr.Jit work hand-in-hand with these JIT-compiled arrays and lazy
evaluation. For example, evaluating a transcendental function operation from
the built-in math library yields a larger graph containing all necessary
operations (click to magnify):

.. code-block:: pycon

   >>> a = dr.asinh(a)
   >>> dr.graphviz(a).view()

.. image:: cuda-02.png
    :width: 400px
    :align: center

Fusing multiple operations can greatly improve performance because the
intermediate results of a larger calculation can be represented in GPU
registers instead of having access them through global memory.

Kernel cache
------------

- JIT compiler is fast
- Second step compilation step is slow, but can be avoided
- Size of arrays doesn't matter

Other design aspects
--------------------

- Reference counting
- Common subexpression elimination

Gotchas
-------

Loop with loop counter or similar
Not evaluating computation in loops
leaving referenced arrays lying around

Diagnostics
-----------

Raising log level
dr.whos()

Horizontal reductions
---------------------

Target device
-------------

Automatic differentiation
-------------------------

Caching memory allocator
------------------------

Similar to the `PyTorch memory allocator
<https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management>`_,
Dr.Jit uses a caching scheme to avoid very costly device synchronizations when
releasing memory. This means that freeing a large GPU variable doesn't cause
the associated memory region to become available for use by the operating
system or other frameworks like Tensorflow or PyTorch. Use the function
:cpp:func:`cuda_malloc_trim` to fully purge all unused memory. The function is
only relevant when working with other frameworks and does not need to be called
to free up memory for use by Dr.Jit itself.

Low level details
-----------------

CUDA: Grid-stride loop

LLVM: nanothread

Usage in C++
------------

Using these array types from C++ requires one of the following three
include directives depending on the desired variant:

.. code-block:: cpp

    #include <drjit/cuda.h>    // <-- For CUDAArray<T>
    #include <drjit/llvm.h>    // <-- For LLVMArray<T>
    #include <drjit/dynamic.h> // <-- For DynamicArray<T>

All of these arrays are composable with other parts of Dr.Jit. For example,
the following type declarations show how to declare a differentiable 3D
array type that will be JIT-compiled to CUDA kernels:

.. code-block:: cpp

    using Float = dr::CUDAArray<float>;
    using FloatD = dr::DiffArray<Float>;
    using Array3f = dr::Array<FloatD, 3>;

.. _custom-cuda:

Dr.Jit ↔ CUDA interoperability
-----------------------------

Dr.Jit's :cpp:struct:`CUDAArray` class dispatches its work to CUDA streams,
making it possible to mix the use of Dr.Jit with standard CUDA kernels. Please
take note of the following points in doing so:

1. CUDA cannot see the effects of computation that has been queued within
   Dr.Jit. Use the :cpp:func:`eval()` function to submit this queued computation
   to the GPU.

2. CUDA kernels run in *streams*: you must submit work to the right stream
   (i.e. the one used by Dr.Jit) to ensure a correct relative ordering of
   operations.

3. C++17 support in NVCC remains limited: it will fail with (incorrect) error
   messages when any Dr.Jit header is included in a file compiled by NVCC. For
   now, it is necessary to partition your project into compilation units
   handled by NVCC and other compilers.

The following example shows what this looks like in practice:

.. code-block:: cpp

   // Forward declaration
   extern void launch_mykernel(cudaStream_t stream, size_t size, const float *in_x,
                               const float *in_y, float *out_x, float *out_y);

   // ...

   using Float   = dr::CUDAArray<float>;
   using Array2f = dr::Array<Float, 2>;

   Array2f in = /* Some Dr.Jit calculation, only symbolic at this point */;

   // Launch CUDA kernel containing queued computation
   dr::eval(in /*, ... other variables ... */);

   // Create empty array (wraps cudaMalloc(), no need to dr::eval() the result)
   Array2f out = dr::empty<Array2f>(1000000);

   // Determine CUDA stream used by Dr.Jit
   cudaStream_t stream = (cudaStream_t) jitc_cuda_stream();

   /// Launch CUDA kernel
   launch_mykernel(
        stream, dr::width(in),
        in.x().data(), in.y().data(),
        out.x().data(), out.y().data()
    );

   // Can now use 'out' in further calculations within Dr.Jit
   out *= 2;

   // Finally, can wrap existing CUDA device pointers into an Dr.Jit array
   float *cuda_device_ptr = ...;
   Float out_2 = dr::map<Float>(cuda_device_ptr,
                                /* # of entries = */ 1000000);

Where the following file containing the kernel is compiled separately by NVCC:

.. code-block:: cpp

    __global__ void my_kernel(size_t size, const float *in_x, const float *in_y,
                              float *out_x, float *out_y) {
        // .. kernel code ..
    }

    // Launcher
    void launch_mykernel(cudaStream_t stream, size_t size, const float *in_x,
                         const float *in_y, float *out_x, float *out_y) {
       my_kernel<<<grid_size, block_size, 0, stream /* <-- important! */>>>(
           size, in_x, in_y, out_x, out_y);
    }

Relationship to other frameworks
--------------------------------

Reference (C++)
---------------

.. cpp:struct:: template <typename Value> DynamicArray : ArrayBase

    This class represents a dynamically sized array using a heap-allocated
    memory region not unlike a ``std::vector<T>``. It it implements all
    arithmetic operations by forwarding them to the underlying ``Value`` type
    and thus behaves like any other Dr.Jit array.

    This class is mainly provided for convenience when storing dynamically
    sized data. It should not be used to perform serious computation, which
    would lead to poor performance. This is because each and every operation
    (e.g. an addition) allocates a new array followed by costly memory reads
    and writes that quickly become the main bottleneck.

.. cpp:struct:: template <typename Value> CUDAArray : ArrayBase

   This array backend just-in-time compiles arithmetic into efficient GPU
   kernels expressed in the CUDA PTX intermediate representation. For details,
   please see the discussion above.

.. cpp:struct:: template <typename Value> LLVMArray : ArrayBase

   This array backend just-in-time compiles arithmetic into efficient CPU
   kernels expressed in the LLVM intermediate representation. For details,
   please see the discussion above.

.. cpp:function:: template <typename Array> const char * graphviz(const Array &array)

   Return GraphViz source code revealing the computation graph associated
   with a particular variable.


.. cpp:function:: template <bool Value, typename Array> auto any_or(const Array &array)

   Test


.. cpp:function:: template <bool Value, typename Array> auto all_or(const Array &array)

   Test

.. cpp:function:: template <bool Value, typename Array> auto none_or(const Array &array)

   Test
