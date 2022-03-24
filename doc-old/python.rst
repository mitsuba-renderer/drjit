.. cpp:namespace:: drjit
.. _python:

Python interface
================

Motivation
----------

Dr.Jit is internally a C++ library. The reason for providing an additional
Python interface is two-fold:

1. It enables fast prototyping of numerical code in a Python environment
   (Jupyter, Matplotlib, etc.)

2. Projects can rely on components written in both languages depending on which
   is more suitable for a particular task. Transitions between C++ and Python
   become seamless thanks to `pybind11 <https://github.com/pybind/pybind11>`_.

Installation
------------

The Dr.Jit Python bindings are available on `PyPI
<https://pypi.org/project/drjit/>`_ and can be installed via

.. code-block:: cpp

   python -m pip install drjit

It is also possible to compile these bindings manually for a specific Dr.Jit
release. See the section on :ref:`building Dr.Jit <building-drjit>` for details.

The remainder of this section discusses conventions relating the C++ and Python
interfaces, followed by an example that combines code written in both languages.


.. _python-cpp-interface:

C++ ↔ Python differences
------------------------

Most Dr.Jit functionality is accessible from both C++ and Python, and these
interfaces are also designed to yield similar-looking code. A few simple rules
suffice to translate from one to the other.

Namespace
~~~~~~~~~

Most of the documentation accesses the library through a short-hand namespace
alias ``ek``. To declare this alias, specify (in C++)

.. code-block:: cpp

   namespace dr = drjit;

and in Python:

.. code-block:: cpp

   import drjit as dr

.. _python-types:

Types
~~~~~

In C++, new array types can be created on the fly by instantiating a template, e.g.:

.. code-block:: cpp

   using Float = dr::CUDAArray<float>;
   using Array2f = dr::Array<Float, 2>;

However, this mechanism is not portable to Python, which lacks a notion of
templates. Dr.Jit's bindings instead expose a large variety of specific template
variants, which leads to the following equivalent Python code:

.. code-block:: cpp

   from drjit.cuda import Float
   from drjit.cuda import Array2f

Altogether, there are six top-level packages:

- ``drjit.scalar``: Arrays built on top of scalars (``float``, ``int``, etc.)
- ``drjit.packet``: Arrays built on top of ``Packet<T>``
- ``drjit.llvm``: Arrays built on top of ``LLVMArray<T>``
- ``drjit.cuda``: Arrays built on top of ``CUDAArray<T>``
- ``drjit.llvm.ad``: Arrays built on top of ``DiffArray<LLVMArray<T>>``
- ``drjit.cuda.ad``: Arrays built on top of ``DiffArray<CUDAArray<T>>``

Each of these six namespaces contains the following

- Arithmetic and mask types: ``Bool``, ``Float``, ``Int``, ``UInt``,
  ``Float64``, ``Int64``, ``UInt64``

- Static arrays (0-4 dimensions, 1D shown here): ``Array1b``, ``Array1f``,
  ``Array1i``, ``Array1u``, ``Array1f64``, ``Array1i64``, ``Array1u64``.

- Dynamic arrays (N dimensions): ``ArrayXb``, ``ArrayXf``,
  ``ArrayXi``, ``ArrayXu``, ``ArrayXf64``, ``ArrayXi64``, ``ArrayXu64``.

- Matrices (2-4 dimensions): ``Matrix2f``, ``Matrix2f64``, ``Matrix3f``,
  ``Matrix3f64``, ``Matrix4f``, ``Matrix4f64``.

- Complex numbers: ``Complex2f``, ``Complex2f64``.

- Quaternions: ``Quaternion4f``, ``Quaternion4f64``.

- A pseudorandom number generator: ``PCG32``.

Using this naming convention, ``drjit.llvm.ad.Array3f`` e.g. corresponds to
``Array<DiffArray<LLVMArray<float>>, 3>``.

This approach is convenient because it enables straightforward porting between
Dr.Jit's different computational backends simply by changing an import
directive.

Functions
~~~~~~~~~

All Dr.Jit functions are part of the ``drjit`` namespace in both languages, and
they generally have the same signature. One exception are functions that take
a template type parameter:

.. code-block:: cpp

    Float x = dr::zero<Float>(100);
    Float y = dr::gather<Float>(x, dr::arange<UInt32>(100));

In the Python interface, the template parameters are simply specified as the
first argument of the function:

.. code-block:: python

    x = dr.zero(Float, 100)
    y = dr.gather(Float, x, dr.arange(UInt32, 100))


Conversion from/to other frameworks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dr.Jit arrays are interoperable with `NumPy <https://numpy.org/>`_ `PyTorch
<https://pytorch.org/>`_, `TensorFlow <https://www.tensorflow.org/>`_, and `JAX
<https://github.com/google/jax>`_, which are widely used frameworks for
scientific computing and machine learning. Whenever applicable and possible,
Dr.Jit uses a zero-copy approach that wraps the existing data in device (GPU)
memory.

.. code-block:: python

   x = Array3f(..) # Calculation producing an Dr.Jit array

   # Convert to a NumPy array
   x_np = x.numpy()
   x = Array3f(x_np) # .. and back

   # Convert to a PyTorch array
   x_py = x.torch()
   x = Array3f(x_py) # .. and back

   # Convert to a TensorFlow array
   x_tf = x.tf()
   x = Array3f(x_tf) # .. and back

   # Convert to a JAX array
   x_jax = x.jax()
   x = Array3f(x_jax) # .. and back


Binding C++ code
----------------

The example below details the creation of bindings for a simple computation
that converts spherical to Cartesian coordinates.

The full project including a tiny build system is available `on GitHub
<https://github.com/wjakob/dr_python_test>`_.

.. code-block:: cpp

    #include <drjit/array.h>
    #include <drjit/math.h>
    #include <drjit/cuda.h>

    #include <pybind11/pybind11.h>

    // Import pybind11 and Dr.Jit namespaces
    namespace py = pybind11;
    namespace dr = drjit;

    // The function we want to expose in Python
    template <typename Float>
    dr::Array<Float, 3> sph_to_cartesian(Float theta, Float phi) {
        auto [sin_theta, cos_theta ] = dr::sincos(theta);
        auto [sin_phi,   cos_phi   ] = dr::sincos(phi);

        return { sin_theta * cos_phi,
                 sin_theta * sin_phi,
                 cos_theta };
    }

    /* The function below is called when the extension module is loaded. It performs a
       sequence of m.def(...) calls which define functions in the module namespace 'm' */
    PYBIND11_MODULE(dr_python_test /* <- name of extension module */, m) {
        m.doc() = "Dr.Jit & pybind11 test plugin"; // Set a docstring

        // 1. Bind the scalar version of the function
        m.def("sph_to_cartesian",      // Function name in Python
              sph_to_cartesian<float>, // Function to be  exposed

              // Docstring (shown in the auto-generated help)
              "Convert from spherical to cartesian coordinates [scalar version]",

              // Designate parameter names for help and keyword-based calls
              py::arg("theta"), py::arg("phi"));

        // 2. Bind the GPU version of the function
        m.def("sph_to_cartesian",
              sph_to_cartesian<dr::CUDAArray<float>>,
              "Convert from spherical to cartesian coordinates [GPU version]",
              py::arg("theta"), py::arg("phi"));
    }

pybind11 infers the necessary binding code from the type of the function
provided to the ``def()`` calls.

Using from Python
*****************

The following interactive session shows how to load the extension module and
query its automatically generated help page.

.. code-block:: pycon

    Python 3.8.5 (default, Jul 28 2020, 12:59:40)
    [GCC 9.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.

    >>> import dr_python_test
    >>> help(dr_python_test)

    Help on module dr_python_test:

    NAME
        dr_python_test - Dr.Jit & pybind11 test plugin

    FUNCTIONS
        sph_to_cartesian(...) method of builtins.PyCapsule instance
            sph_to_cartesian(*args, **kwargs)
            Overloaded function.

            1. sph_to_cartesian(theta: float, phi: float) -> drjit.scalar.Array3f

            Convert from spherical to cartesian coordinates [scalar version]

            2. sph_to_cartesian(theta: drjit.cuda.Float, phi: drjit.cuda.Float) -> drjit.cuda.Array3f

            Convert from spherical to cartesian coordinates [GPU version]

    FILE
        /home/wjakob/dr_python_test/dr_python_test.cpython-38-x86_64-linux-gnu.so

As can be seen, the help describes the overloads along with the name and shape
of their input arguments. Let's try calling one of them:

.. code-block:: python

    >>> from dr_python_test import sph_to_cartesian

    >>> r = sph_to_cartesian(theta=1, phi=2)

    >>> r
    [-0.3501754701137543, 0.7651473879814148, 0.5403022766113281]

    >>> type(r)
    <class 'drjit.scalar.Array3f'>

Let's now call the CUDA version of the function. We will use ``dr.linspace`` to
generate generate a few example inputs:

.. code-block:: python

    >>> import drjit as dr
    >>> from drjit.cuda import Float
    >>> from dr_python_test import sph_to_cartesian

    >>> sph_to_cartesian(theta=dr.linspace(Float, 0.0, 1.0, 100),
    ...                  phi=dr.linspace(Float, 1.0, 2.0, 100))

    [[0.0, 0.0, 1.0],
     [0.00537129258736968, 0.008554124273359776, 0.999949038028717],
     [0.010568737052381039, 0.017215078696608543, 0.9997959733009338],
     [0.015590270049870014, 0.02597944624722004, 0.9995409250259399],
     [0.020433608442544937, 0.034843236207962036, 0.9991838932037354],
     .. 90 skipped ..,
     [-0.3104492425918579, 0.7578364610671997, 0.5738508701324463],
     [-0.3203235864639282, 0.7599647045135498, 0.5655494332313538],
     [-0.33023831248283386, 0.7618932127952576, 0.5571902394294739],
     [-0.340190589427948, 0.763620913028717, 0.5487741827964783],
     [-0.3501753807067871, 0.7651472687721252, 0.5403023362159729]]

Note how the C++ code was able to process an Dr.Jit array created via the Python
bindings. All features like JIT compilation and automatic differentiation work
seamlessly across language boundaries: in this case, a single CUDA kernel was
compiled to produce the output, and that kernel contains both the arithmetic
from the ``sph_to_cartesian`` function, and the computation of the inputs via
``dr.linspace`` done on the Python side.
