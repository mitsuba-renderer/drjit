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

.. _texture_mipmap:

MIP-mapped filtering
--------------------

Textures constructed with a ``mip_filter`` additionally maintain a sequence of
progressively downscaled copies of the texture known as a *MIP pyramid*.

.. code-block:: python

   tex = Texture2f(tensor, mip_filter=dr.MipFilter.Linear, max_aniso=8)

The ``mip_filter`` parameter (see :py:class:`dr.MipFilter <MipFilter>`) chooses
how a filtered lookup turns a continuous level of detail into a sample of the
pyramid:

- ``MipFilter.Disabled`` is the default and omits the pyramid entirely. The
  filtered lookups below then degrade to an ordinary base level
  :py:func:`.eval() <drjit.auto.Texture2f.eval>`.

- ``MipFilter.Nearest`` rounds to the closest level. This is the cheaper
  option, but the level transitions tend to be visible.

- ``MipFilter.Linear`` blends the two enclosing levels, which removes those
  transitions at the cost of a second lookup.

The ``max_aniso`` parameter bounds the number of taps of an anisotropic lookup
and may range from 1 (isotropic filtering) to the hardware limit of 16.

Calling :py:func:`.set_value() <drjit.auto.Texture2f.set_value>` or
:py:func:`.set_tensor() <drjit.auto.Texture2f.set_tensor>` on an existing
MIP-mapped texture rebuilds the pyramid.

Two lookup methods consume this pyramid:

- :py:func:`.eval_lod() <drjit.auto.Texture2f.eval_lod>` samples the texture at
  an explicit *level of detail*, where a fractional level blends the two
  enclosing pyramid levels.

- :py:func:`.eval_filtered() <drjit.auto.Texture2f.eval_filtered>` implements
  the standard anisotropic filtering scheme of graphics APIs. Given the
  derivatives of the texture coordinate with respect to the two screen
  dimensions, it selects a level of detail so that up to ``max_aniso``
  trilinear taps along the major axis of the pixel footprint cover it without
  aliasing.

  .. code-block:: python

     out = tex.eval_filtered(pos, ddx, ddy)

Both methods are differentiable with respect to the query position and texture
data, which includes derivative propagation through the MIP pyramid
construction.

.. _texture_laplacian:

Laplacian basis
---------------

MIP-mapped textures can optionally adopt a *Laplacian pyramid* basis following
the paper `Practical Inverse Rendering of Textured and Translucent Appearance
<https://doi.org/10.1145/3730855>`__ by Weier et al. This feature targets
workflows involving gradient-based optimization of textures with filtered
texture lookups.

In textures constructed with ``mip_basis=dr.MipBasis.Laplacian``, the
authoritative representation is no longer the base image but a set of per-level
coefficient tensors. The MIP pyramid uploaded to the GPU is then derived from
these tensors by repeated upsampling and summation.

.. code-block:: python

   tex = Texture2f(tensor, migrate=False, mip_filter=dr.MipFilter.Linear,
                   mip_basis=dr.MipBasis.Laplacian)

This basis requires a MIP-mapped texture with floating-point storage on a JIT
backend. The ``migrate=False`` argument is needed because the coefficient
tensors must stay in ordinary memory (see :ref:`migration <texture_migration>`
below).

The coefficients form a coarse-to-fine hierarchy. The coarsest level determines
the overall appearance of the texture, and each finer level adds increasingly
localized detail. An adaptive optimizer such as Adam maintains a separate step
size per level, which turns the decomposition into a multiscale preconditioner.

The coefficient tensors are exposed through level-indexed accessor and setter
overloads. A typical optimization loop registers them with an optimizer, writes
the updated values back, and rebuilds the sampled pyramid once per iteration
via :py:func:`.update_inplace() <drjit.auto.Texture2f.update_inplace>`:

.. code-block:: python

   opt = Adam(lr=1e-2)
   for l in range(tex.mip_levels()):
       opt[f'level_{l}'] = tex.tensor(l)

   for it in range(n_iterations):
       for l in range(tex.mip_levels()):
           tex.set_tensor(l, opt[f'level_{l}'], rebuild=False)
       tex.update_inplace()
       loss = objective(tex.eval_filtered(...))
       dr.backward(loss)
       opt.step()

With ``rebuild=False``, the assignment cheaply rebinds the coefficient tensor,
and the :py:func:`.update_inplace() <drjit.auto.Texture2f.update_inplace>` call
at the end synthesizes the pyramid once for all levels. The default
``rebuild=True`` instead synthesizes it after every assignment, which is
convenient when changing a single level but wasteful in a loop like the one
above.

Assigning the high-resolution base texture via :py:func:`.set_tensor()
<drjit.auto.Texture2f.set_tensor>` or :py:func:`.set_value()
<drjit.auto.Texture2f.set_value>` is also supported and decomposes the image
into the Laplacian representation.

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

.. _texture_migration:

Migration
^^^^^^^^^
When hardware acceleration is disabled, Dr.Jit textures are a thin wrapper
around the underlying tensor representation, which remains accessible:

.. code-block:: python

   tex = dr.cuda.Texture2f(tensor_data, use_accel=False)

   tensor_data = tex.tensor() # Return the tensor backing this texture
   array_data = tex.value()   # Same, but in array form

Hardware-accelerated Dr.Jit textures work differently: they *migrate* texture
data into a CUDA texture object to avoid redundant storage. Methods such as
:py:func:`.tensor() <drjit.cuda.Texture2f.tensor>` and :py:func:`.value()
<drjit.cuda.Texture2f.value>` then return a symbolic view that occupies no
actual storage. Evaluating the view fetches the texel data back from the
hardware texture, and it reflects the texture contents at the time of that
evaluation. Evaluate it before overwriting the texture when the current
contents are needed.

If you desire access to a hardware-accelerated texture *and* at the
same time want the tensor representation to stay in ordinary memory,
specify ``migrate=False`` to the texture constructor, i.e.,

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

8-bit textures
^^^^^^^^^^^^^^

The interface of ``Texture?f8u`` variants (e.g. :py:class:`Texture3f8u
<drjit.auto.ad.Texture3f8u>`) slightly differs from their floating point
counterparts. They store texture data compactly using unsigned 8-bit integers,
but their ``eval_*`` members produce floating point output by transparently
remapping ``0..255`` onto the interval ``[0, 1]``.

When the texture object was created with ``srgb=True``, i.e.,

.. code-block:: python

   tex = Texture2f8u(tensor_u8, srgb=True)

evaluation will apply the nonlinear sRGB transfer function to turn
gamma-encoded 8-bit values back into a linear scale. The CUDA and Metal
backends perform both types of conversions in hardware.

Writing to textures
-------------------

Hardware textures can also be *written* from within a kernel, turning a texture
into a render target. Create the texture with ``writable=True`` and store
per-texel values with :py:func:`.write() <drjit.auto.Texture2f.write>`, which
takes *integer* texel coordinates (one unsigned-integer array per dimension) and
a list of per-channel values:

.. code-block:: python

   tex = Texture2f([height, width], channels=4, writable=True)

   idx = dr.arange(UInt, width * height)
   x, y = idx % width, idx // width
   tex.write([x, y], [r, g, b, a])
   dr.eval()

The access mode follows the *operation*, so a ``writable`` texture may be both
written and sampled (a texture rendered in one kernel can be looked up in
another). Its contents can be read back into a tensor via :py:func:`.tensor()
<drjit.auto.Texture2f.tensor>` / :py:func:`.value()
<drjit.auto.Texture2f.value>` as usual. Writing 8-bit textures clips and
quantizes ``[0, 1]`` float inputs and sRGB-encodes them if needed.

This feature requires a JIT backend; it is unavailable for ``scalar`` textures.

Wrapping native textures
------------------------

To share textures with a GUI or another GPU API, an *existing* native texture
can be wrapped as a Dr.Jit texture with :py:func:`.from_native_handle()
<drjit.auto.Texture2f.from_native_handle>`. The handle is an ``id<MTLTexture>`` pointer
on the Metal backend or an OpenGL texture id on the CUDA backend; the shape,
channel count, and component type are inferred from it (the dimensionality and
precision must match the texture type used):

.. code-block:: python

   tex = dr.metal.Texture2f.from_native_handle(mtl_texture)                 # sample it
   tex = dr.metal.Texture2f.from_native_handle(mtl_texture, writable=True)  # render into it

Pass ``writable=True`` to render into the application's texture via
:py:func:`.write() <drjit.auto.Texture2f.write>`. The inverse,
:py:func:`.native_handle() <drjit.auto.Texture2f.native_handle>`, returns a
Dr.Jit-allocated texture's native handle to hand to a GUI for display: the
``id<MTLTexture>`` on Metal, or the wrapped OpenGL texture id on CUDA (``0`` if
the texture has no OpenGL identity).

A texture wrapping a cross-API handle (an OpenGL texture on the CUDA backend) is
only usable between :py:func:`.map() <drjit.auto.Texture2f.map>` and
:py:func:`.unmap() <drjit.auto.Texture2f.unmap>`, which must bracket each use;
on Metal both are no-ops.

C++ interface
-------------

Textures are also avilable in C++. To do so, instantiate the template class
:cpp:class:`drjit::Texture` with any Dr.Jit array or scalar floating-point type
and specify the desired number of dimensions:

.. code-block:: cpp

   using Float = dr::CUDAArray<float>;

   size_t shape[2] = { 1024, 768 };
   dr::Texture<Float, 2> tex(shape, 3);

