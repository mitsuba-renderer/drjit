.. cpp:namespace:: drjit

.. _textures:

Textures
========

Dr.Jit further provides the ability to perform texture sampling on array types, 
with the Python interface exposing half, single and double-precision 
floating-point textures in 1, 2 and 3 dimensions. Tensors can be supplied 
to initialize these textures

.. code-block:: python

   import drjit as dr

   n_channels = 3
   tensor = dr.full(dr.cuda.TensorXf, 2, shape=[1024, 768, n_channels])
   tex = dr.cuda.Texture2f(tensor)

In C++, the template class `dr::Texture` can be instantiated 
with any Dr.Jit array or scalar floating-point type, along with the associated 
dimensions

.. code-block:: cpp

   using Float = dr::CUDAArray<float>;

   size_t shape[2] = { 1024, 768 };
   dr::Texture<Float, 2> tex(shape, 3);

Given an array of texture coordinates :math:`p_i \in [0,1]^d`, we can sample a 
texture of :math:`d` dimensions at positions :math:`p_i` using the 
:py:func:`eval()` function

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor)
   pos = dr.cuda.Array2f([0.25, 0.5, 0.9], [0.1, 0.3, 0.5])
   out = tex.eval(pos)

where the texture filtering and wrap-mode methods used for interpolation 
are specified during initialization

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, filter_mode=dr.FilterMode.Linear, 
      wrap_mode=dr.WrapMode.Repeat)

Moreover the :py:func:`eval_cubic()` function provides an independent interface 
for sampling a texture using a clamped cubic B-Spline interpolant.

.. note::

    When evaluating a texture, the numerical precision used during the
    interpolation is dictated by the floating point precision of the query
    point.

Hardware acceleration
---------------------

Dr.Jit textures targeting the CUDA backend can benefit from hardware-accelerated 
texture lookups. Internally, textures initialized with ``use_accel=True`` 
(which is enabled by default) will create an associated *CUDA texture object* 
that leverages GPU hardware intrinsics to perform sampling

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=True)

.. note::

    Only single and half-precision floating-point CUDA texture objects are
    supported. Double-precision textures can be initialized but won't benefit
    from hardware-acceleration.

.. warning::

    Hardware-accelerated lookups use a 9-bit fixed-point format with 8-bits of
    fractional value for storing the *weights* used for linear interpolation. See
    the `CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering>`_
    for more details.

Migration
^^^^^^^^^
When CUDA texture objects aren't utilised, the underlying storage type 
of a Dr.Jit texture is exclusively a tensor,

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=False)

   tensor_data = tex.tensor()
   array_data = tex.value()

however hardware-accelerated Dr.Jit textures can be initialized to retain 
*both* a copy of the data as a CUDA texture object as well as a tensor by 
disabling *migration*

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=True, migrate=False)

While the default behavior of texture intialization is to set ``migrate=True`` to
minimize redundant storage, it's important to note that attempting to fetch
either the :py:func:`tensor()` or :py:func:`value()` data requires converting a
CUDA texture object into a tensor and hence a side effect of these function
calls is to disable migration.

Automatic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^
Suppose we want to compute the gradient of a lookup with respect to the 
input tensor of a texture

.. code-block:: python

   import drjit as dr

   N = 3

   TensorXf = dr.cuda.ad.TensorXf
   Texture1f = dr.cuda.ad.Texture1f
   Array1f = dr.cuda.ad.Array1f

   tensor = TensorXf([3,5,8], shape=(N, 1))

   dr.enable_grad(tensor)

   tex = Texture1f(tensor)
   pos = Array1f(0.4)
   out = Array1f(tex.eval(pos))

   dr.backward(out)

   grad = dr.grad(tensor)

In order to propagate gradients, the associated AD graph needs to track the 
collection of coordinate wrapping, texel fetching and filtering operations that 
are performed on the underlying tensor as part of sampling. While 
hardware-accelerated textures here rely on GPU intrinsics, 
such textures are indeed still differentiable. Internally, while 
the primal lookup operation is hardware-accelerated, a subsequent 
non-accelerated lookup is additionally performed *solely* to record each 
individual operation into the AD graph. More importantly, computing gradients 
does *not* require disabling migration and texture data can continue to 
exclusively be stored as a CUDA texture object.
