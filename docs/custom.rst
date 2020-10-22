.. cpp:namespace:: enoki

Customization
=============

Enoki offers several escape hatches to implement custom features that are
difficult to express using builtin functionality. This section explains such
extension mechanisms related to the JIT compiler and automatic differentiation.

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

.. _custom-autodiff:

Custom differentiable operations (C++)
--------------------------------------

Enoki can compute derivatives of builtin operations in both forward and reverse
mode. In rare cases, it may be useful or even necessary to tell Enoki how a
particular operation should be differentiated. Reasons for this may include:

1. The automatic differentiation backend cannot keep track of computation that
   is performed outside of Enoki (e.g. using a highly optimized :ref:`CUDA
   kernel <custom-cuda>`), or when :cpp:class:`DiffArray` is not used for other
   reasons.

2. Multiple frameworks (e.g. PyTorch/TensorFlow and Enoki) may be involved in
   larger projects, in which case gradient propagation requires a clear
   interface between them.

3. The derivative may admit a simplified analytic expression that is superior
   to what direct application of automatic differentiation would produce.

4. Automatic derivative propagation through Enoki's :ref:`symbolic loops
   <symbolic-loops>` is not supported. They will always require extra steps as
   outlined here and in the section on :ref:`differentiating loops
   <diff-loop>`.

Custom differentiable operations require the creation of a class providing
callback functions that are invoked when the AD backend traverses the
associated node in the computation graph. This class also provides a convenient
way of stashing temporary results during the original function evaluation that
can be accessed later on as part of forward or reverse-mode differentiation.

To start, make sure to include the extra header file

.. code-block:: cpp

    #include <enoki/custom.h>

which provides all necessary infrastructure. Suppose, that we are working with
the following types

.. code-block:: cpp

    using Float  = ek::CUDAArray<float>; // JIT-ed CUDA array
    using FloatD = ek::DiffArray<Float>; // .. which furthermore tracks derivatives

We must define the aforementioned callback class deriving from
:cpp:class:`CustomOp`, which is a variadic template class parameterized by the
type underlying automatic differentiation (`FloatD` in this example), and the
function output and input(s).

.. code-block:: cpp

    struct MyOp : ek::CustomOp<FloatD, /* <-- type underlying AD backend */,
                               ...,    /* output type */,
                               ...     /* one or more input type (s) */> { ... };

Suppose that we're interested in computing the derivative of the following operation,
which normalizes a 3D input vector:

.. math::

   N(\mathbf{v}) := \frac{\mathbf{v}}{\|\mathbf{v}\|}

When :math:`\mathbf{v}` depends on an arbitrary parameter :math:`\theta`, the
derivative of the above expression is given by

.. math::

   \frac{\partial}{\partial \theta} N(\mathbf{v}(\theta)) :=
   \frac{1}{\|\mathbf{v}(\theta)\|}
   \frac{\partial\mathbf{v}(\theta)}{\partial \theta}
   - \frac{\mathbf{v}(\theta)}{\|\mathbf{v}(\theta)\|^3}
   \big\langle
   \mathbf{v}(\theta),
   \frac{\partial\mathbf{v}(\theta)}{\partial \theta}
   \big\rangle

Let's define non-differentiable and differentiable 3D vector types first:

.. code-block:: cpp

    using Array3f  = ek::Array<Float, 3>;
    using Array3fD = ek::Array<FloatD, 3>;

The basic structure of the ``Normalize`` class then looks as follows:

.. code-block:: cpp

    struct Normalize : ek::CustomOp<FloatD, Array3fD, Array3fD> {
        using Base = ek::CustomOp<FloatD, Array3fD, Array3fD>;

        // Return a descriptive name that used in GraphViz output
        const char *name() override { return "normalize"; }

        // .. continued shortly ..

    private:
        // Storage for temporary values
        Float m_inv_norm;
        Array3f m_input;
    };

Apart from ``name()``, this declaration must override *three* other virtual
methods: the first, ``eval()``, performs an ordinary (non-differentiable)
function evaluation. Note that its parameter(s) and return value must be
non-differentiable variants of the input/outputs as originally specified via
template parameters of :cpp:struct:`CustomOp`. Non-differentiable is as defined
by :cpp:type:`detached_t`. For example, ``detached_t<Array3fD>`` equals
``Array3f``. Finally, the inputs must be specified as ``const`` references
(see the following note).

.. note::

   The custom function interface assumes that the function's access to
   arguments is read-only, and that it produces all output via a single return
   value. Returning data via parameter references is not allowed.

   Returning multiple things is fine: the return type can be an Enoki array,
   ``std::pair``, ``std::tuple`` or custom data structure exposed via
   :c:macro:`ENOKI_STRUCT`.

The ``eval()`` method also stores two temporary variables (``m_input`` and
``m_inv_norm``) since they are required by in both forward and reverse-mode
derivative propagation.

.. code-block:: cpp

   Array3f eval(const Array3f &input) override {
       m_input = input;
       m_inv_norm = ek::rcp(ek::norm(input));
       return input * m_inv_norm;
   }


The forward-mode callback should query gradients arriving along the function
inputs via :cpp:func:`CustomOp::grad_in()`, where the template parameter
indicates the argument index. If the function only takes one input, it can also
be omitted. Before returning, the function must call
:cpp:func:`CustomOp::set_grad_out()` to assign the output gradient.

.. code-block:: cpp

    void forward() override {
        Array3f grad_in = Base::grad_in<0>(),
                 grad_out = grad_in * m_inv_norm;
        grad_out -= m_input * (ek::dot(m_input, grad_out) *
                               ek::sqr(m_inv_norm));
        Base::set_grad_out(grad_out);
    }

Reverse-mode differentiation via ``backward()`` flips this around: the callback
should query gradients arriving along the function output via the
:cpp:func:`CustomOp::grad_out()` and then invoke
:cpp:func:`CustomOp::set_grad_in()` to assign the input gradient(s). In this
simple example, the two definitions are almost identical, though this is often
not the case.

.. code-block:: cpp

    void backward() override {
        Array3f grad_out = Base::grad_out(),
                 grad_in = grad_out * m_inv_norm;
        grad_in -= m_input * (ek::dot(m_input, grad_in) *
                              ek::sqr(m_inv_norm));
        Base::set_grad_in<0>(grad_in);
    }

Once defined, the custom operation can be invoked as follows:

.. code-block:: cpp

   Array3f d = /* ... */;
   Array3f d2 = ek::custom<Normalize>(d);

.. _custom-autodiff-py:

Custom differentiable operations (Python)
-----------------------------------------

Please first review the section on :ref:`custom differentiable operations in
C++ <custom-autodiff>`. The Python syntax is very similar, except that input 
arguments are referenced by name instead of index.

.. code-block:: python
    :emphasize-lines: 8, 19

    class Normalize(ek.CustomOp):
        def eval(self, value):
            self.value = value
            self.inv_norm = ek.rcp(ek.norm(value))
            return value * self.inv_norm

        def forward(self):
            grad_in = self.grad_in('value')
            grad_out = grad_in * self.inv_norm
            grad_out -= self.value * (ek.dot(self.value, grad_out) *
                                      ek.sqr(self.inv_norm))
            self.set_grad_out(grad_out)

        def backward(self):
            grad_out = self.grad_out()
            grad_in = grad_out * self.inv_norm
            grad_in -= self.value * (ek.dot(self.value, grad_in) *
                                     ek.sqr(self.inv_norm))
            self.set_grad_in('value', grad_in)

        def name(self):
            return "normalize"

Once defined, a custom operation can be invoked as follows:

.. code-block:: python

   import enoki as ek
   from enoki.cuda.ad import Array3f

   d = Array3f(...)
   d2 = ek.custom(Normalize, d)

Differentiable loops
--------------------

For instance, suppose that we are using a loop to compute


.. code-block:: python

   float value = 0.f;
   for (int i = 0; i < 1000; ++i) {
       value += f(value);
   }

Reference
---------

.. cpp:class:: template <typename Type, typename Result, typename... Args> CustomOp

   Callback interface used to integrate custom operations into Enoki's
   graph-based AD implementation.

   .. cpp:function:: virtual detached_t<Result> eval(const detached_t<Args>& ... args) = 0

      This callback function must be provided by implementations of this
      interface. It should perform the underlying "primal" computation using
      detached types, i.e. without keeping track of derivatives.

   .. cpp:function:: virtual void forward() = 0

      This callback function must be provided by implementations of this
      interface. It is invoked during forward-mode AD and should query input
      gradients via :cpp:func:`grad_in()` and then call
      :cpp:func:`set_grad_out()`

   .. cpp:function:: virtual void backward() = 0

      This callback function must be provided by implementations of this
      interface. It is invoked during reverse-mode AD and should query input
      gradients via :cpp:func:`grad_out()` and then call
      :cpp:func:`set_grad_in()`

   .. cpp:function:: virtual const char *name() = 0

      This function must be provided by implementations of this interface. It
      should return a brief descriptive name of the custom operation. It will
      is visible in the graph visualizations obtained via
      :cpp:func:`graphviz()`.

   .. cpp:function:: template <size_t Index = 0> auto grad_in() const

      This protected method queries the gradient of an input argument (`Index`
      zero by default). It should only be called from the :cpp:func:`forward()`
      callback.

   .. cpp:function:: template <size_t Index = 0, typename T> void set_grad_in(const T &value)

      This protected method assigns the gradient of an input argument (`Index`
      zero by default). It should only be called from the :cpp:func:`backward()`
      callback.

   .. cpp:function:: detched_t<Result> grad_out() const

      This protected method queries the gradient of the output argument It
      should only be called from the :cpp:func:`backward()` callback.

   .. cpp:function:: void set_grad_out(const detached_t<Result> &grad)

      This protected method assigns the gradient value of the function output. It
      should only be called from the :cpp:func:`forward()` callback.

.. cpp:function:: template <typename Custom, typename... Input> auto custom(const Input&... inputs)

   This function requires a template parameter providing an implementation of
   the :cpp:class:`CustomOp` interface. It then runs the associated function
   with detached (non-AD) types and splices callback functions into the AD
   graph representation that are invoked during forward and reverse mode
   differentiation.
