.. py:currentmodule:: drjit

.. cpp:namespace:: drjit

.. _textures:

Textures
========

Dr.Jit provides a convenient abstraction around the notion of a *texture*,
i.e., a multidimensional array that can be evaluated at fractional positions.
This feature leverages hardware texture units to accelerate lookups if possible,
and it otherwise reverts to an efficient software implementation.

The texture implementation is fully differentiable and supports half, single,
and double-precision floating-point textures in 1, 2 and 3 dimensions. Each
lookup produces simultaneous evaluations for a set of *channels*, which
conceptually increases the dimension of the underlying storage by one.

The easiest way to create a texture is by initializing it from a compatible
tensor:

.. code-block:: python

   import drjit as dr
   from drjit.auto.ad import TensorXf, Texture2f

   n_channels = 3
   tensor = dr.full(TensorXf, 1, shape=[1024, 768, n_channels])

   # 2D texture with 3 output channels
   tex = Texture2f(tensor)

To use a texture with a different number of dimensions or precision, adopt the
class name appropriately (e.g., :py:class:`Texture3f16
<drjit.auto.ad.Texture3f16>` for 3D half-precision).

You may optionally also specify filter and wrap modes that are used by
subsequent interpolated lookups (see :py:class:`dr.FilterMode <FilterMode>` and
:py:class:`dr.WrapMode <WrapMode>` for details).

.. code-block:: python

   tex = Texture2f(
       tensor,
       filter_mode=dr.FilterMode.Linear,
       wrap_mode=dr.WrapMode.Repeat
   )

The :py:class:`dr.WrapMode <WrapMode>` enum controls how out-of-bounds texture
coordinates are handled:

- ``WrapMode.Clamp``: Clamp coordinates to edge values (edge pixels are repeated)
- ``WrapMode.Repeat``: Wrap coordinates periodically (for tiling textures)
- ``WrapMode.Mirror``: Mirror/reflect coordinates at boundaries

These wrap modes are also used by :py:func:`dr.resample` and
:py:func:`dr.convolve` to control boundary behavior during resampling and
convolution operations.

The :py:func:`.eval() <drjit.auto.Texture2f.eval>` function queries the function
at a position on the unit cube. In this example involving a 2D texture, we must
provide a 2D input point, and the evaluation produces three output channels.

.. code-block:: python

   pos = dr.cuda.Array2f([0.25, 0.5, 0.9],
                         [0.1,  0.3, 0.5])
   out = tex.eval(pos)

Regular lookups use nearest neighbor or linear/bilinear/trilinear
interpolation. The :py:func:`.eval_cubic() <dr.auto.Texture2f.eval_cubic>`
builds on this capability to provide a clamped cubic B-Spline interpolant at
somewhat higher cost.

.. note::

    When evaluating a texture, the numerical precision used during the
    interpolation is dictated by the floating point precision of the query
    point. You may, e.g., want to use a 32-bit position to query a 16-bit
    texture to avoid a loss of accuracy.

Hardware acceleration
---------------------

Dr.Jit can accelerate texture lookups on the CUDA backend using hardware GPU
texture units. Textures initialized with ``use_accel=True`` (the default) will
create an associated *CUDA texture object* that leverages hardware intrinsics
to perform sampling

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=True)

.. note::

    Only single and half-precision floating-point CUDA texture objects are
    supported. Double-precision textures work but won't benefit from
    hardware-acceleration.

.. warning::

    Hardware-accelerated lookups use a 9-bit fixed-point format with 8-bits of
    fractional value for storing the *weights* used for linear interpolation. See
    the `CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#linear-filtering>`_
    for more details.

Migration
^^^^^^^^^
When hardware acceleration is disabled, Dr.Jit textures are a thin wrapper
around the underlying tensor representation, which remains accessible:

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=False)

   tensor_data = tex.tensor() # Return the tensor backing this texture
   array_data = tex.value()   # Same, but in array form

Hardware-accelerated Dr.Jit textures work differently: they *migrate* texture
data into a CUDA texture object that is no longer directly accessible to
Dr.Jit. This makes methods such as :py:func:`.tensor()
<drjit.cuda.Texture2f.tensor>` and :py:func:`.value()
<drjit.cuda.Texture2f.value>` rather expensive, since they must copy the
texture data from the CUDA object back into memory.

If you desire access to a hardware-accelerated texture *and* at the
same time retain the tensor representation, specify ``migrate=False``
to the texture constructor, i.e.,

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=True, migrate=False)

This, however, doubles the storage cost associated with the texture.

Automatic differentiation
^^^^^^^^^^^^^^^^^^^^^^^^^
Suppose we want to compute the gradient of a lookup with respect to the
input tensor of a texture

.. code-block:: python

   import drjit as dr
   from drjit.cuda.ad import TensorXf, Texture1f, Array1f

   N = 3

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
individual operation into the AD graph.

C++ interface
-------------

Textures are also avilable in C++. To do so, instantiate the template class
:cpp:class:`drjit::Texture` with any Dr.Jit array or scalar floating-point type
and specify the desired number of dimensions:

.. code-block:: cpp

   using Float = dr::CUDAArray<float>;

   size_t shape[2] = { 1024, 768 };
   dr::Texture<Float, 2> tex(shape, 3);

