.. cpp:namespace:: enoki

.. _cuda:

CUDA arrays
===========

Many Enoki operations can be applied to custom data structures, causing them to
recursively propagate through all of the data structure's fields. The remainder
of this sections explains several use cases of this functionality, and how to
enable it via a suitable :c:macro:`ENOKI_STRUCT` declaration. This feature
requires the following optional header file:

.. code-block:: cpp

    #include <enoki/cuda.h>

.. _custom-cuda:

Enoki ↔ CUDA interoperability
-----------------------------

Enoki's :cpp:class:`CUDAArray` class dispatches its work to CUDA streams,
making it possible to mix the use of Enoki with standard CUDA kernels. Please
take note of the following points in doing so:

1. Enoki queues up computation for later execution, and its effect won't be
   visible to a CUDA kernel unless you enforce timely evaluation via
   :cpp:func:`eval()`.

2. CUDA kernels run in *streams*: you must submit work to the right stream
   (i.e. the one used by Enoki) to ensure a correct relative ordering of
   operations.

3. C++17 support in NVCC remains limited: it will fail with (incorrect) error
   messages when any Enoki header is included in a file compiled by NVCC. For
   now, it is necessary to partition your project into compilation units
   handled by NVCC and other compilers.

The following example shows what this looks like in practice:

.. code-block:: cpp

   // Forward declaration
   extern void launch_mykernel(cudaStream_t stream, size_t size, const float *in_x,
                               const float *in_y, float *out_x, float *out_y);

   // ...

   using Float   = ek::CUDAArray<float>;
   using Array2f = ek::Array<Float, 2>;

   Array2f in = /* Some Enoki calculation, only symbolic at this point */;

   // Launch CUDA kernel containing queued computation
   ek::eval(in /*, ... other variables ... */);

   // Create empty array (wraps cudaMalloc(), no need to ek::eval() the result)
   Array2f out = ek::empty<Array2f>(1000000);

   // Determine CUDA stream used by Enoki
   cudaStream_t stream = (cudaStream_t) jitc_cuda_stream();

   /// Launch CUDA kernel
   launch_mykernel(
        stream, ek::width(in),
        in.x().data(), in.y().data(),
        out.x().data(), out.y().data()
    );

   // Can now use 'out' in further calculations within Enoki
   out *= 2;

   // Finally, can wrap existing CUDA device pointers into an Enoki array
   float *cuda_device_ptr = ...;
   Float out_2 = ek::map<Float>(cuda_device_ptr,
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
