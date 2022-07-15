.. py:module:: drjit

.. _firststeps-py:

First steps in Python
=====================

Dr.Jit offers both Python and C++ interfaces. The majority of this
documentation covers the Python interface. For differences that are specific to
C++, see the :ref:`separate section <firststeps-cpp>` on this.

Installing Dr.Jit
-----------------

The easiest way to obtain Dr.Jit is using `binary wheels
<https://realpython.com/python-wheels/>`_, which we provide for officially
supported Python versions and the most common platforms (Linux x86_64,
Windows x86_64, macOS arm64/x86_64). To install Dr.Jit in this way, run

.. code-block:: bash

   $ pip install drjit

The remainder of this section walks through a simple example that makes use of
various system features. In particular, we will render an image of a bumpy
sphere expressed as a `signed distance function
<https://en.wikipedia.org/wiki/Signed_distance_function>`_ using just-in-time
compilation, random number generation, GPU texturing, loop recording, and automatic
differentiation. You can follow the example by copy-pasting code to a python
file or a `Juypter lab
<https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html>`_
instance (recommended).

Importing Dr.Jit
----------------

Most Dr.Jit functionality resides in the ``drjit`` namespace that we
typically associate with the ``dr`` alias for convenience.

.. code-block:: python

    import drjit as dr

Besides this, we must also choose a set of array types from a specific
computation backend. Several choices are available:

- ``drjit.cuda`` provides data types for GPU-accelerated parallel computing
  using `CUDA <https://en.wikipedia.org/wiki/CUDA>`_.

- ``drjit.llvm`` provides data types for CPU-accelerated parallel computing
  using the `LLVM compiler infrastructure
  <https://en.wikipedia.org/wiki/LLVM>`_.

- ``drjit.scalar`` provides simple data types for serial/scalar computation.

Further backends (e.g. Apple Metal, Intel Xe) are planned in the future. The
CUDA and LLVM backends also provide specialized aliases for derivative tracking
using `automatic differentiation
<https://en.wikipedia.org/wiki/Automatic_differentiation>`_ (``drjit.cuda.ad``
and ``drjit.llvm.ad``). We will discuss them later as part of this tutorial.

We begin by importing various components that will be used in the tutorial:

.. code-block:: python

    from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

LLVM backend
^^^^^^^^^^^^

If you don't have a CUDA-compatible GPU, change ``drjit.cuda`` to
``drjit.llvm`` in the above import statement. In that case, note that LLVM 8
or newer must be installed on the system, which may require additional steps
depending on your platform:

- **Linux**: ``apt-get install llvm`` (or an equivalent command for your distribution.)

- **macOS**: ``brew install llvm`` using `Homebrew <https://brew.sh>`_.

- **Windows**: run one of the `official installers
  <https://releases.llvm.org>`_ (many files can be downloaded from this page,
  look for ones with the pattern ``LLVM-<version>-win64.exe``). It is important
  that you let the installer adjust the ``%PATH%`` variable so that the file
  ``LLVM-C.dll`` can be found by Dr.Jit.

With that out of the way, let's get back to the example.

Signed distance functions and sphere tracing
--------------------------------------------

A `signed distance function
<https://en.wikipedia.org/wiki/Signed_distance_function>`_ is a function that
specifies the distance to the nearest surface. It provides a convenient and
general way of encoding 3D shape information. We will initially start with a
simple SDF of a sphere with radius 1 that is centered around the origin:

.. code-block:: python

    def sdf(p: Array3f) -> Float:
        return dr.norm(p) - 1

The function takes 3D points and returns the associated distance value. The
type annotations are provided for clarity and can be omitted in practice. We
can use an interactive Python prompt to pass an ``Array3f`` instance
representing a 3D point into the function and observe the calculated distance.

.. code-block:: pycon

    >>> sdf(Array3f(1, 2, 3))
    [2.7416574954986572]

The CUDA and LLVM backends of Dr.Jit *vectorize* and *parallelize* computation.
This means that types like ``Float`` and ``Array3f`` typically hold many values
at once that are used to perform simultaneous evaluations of a function. For
example, we can compute the SDF at positions :math:`(0, 0, 0)` and :math:`(1,
2, 3)` in one combined step.

.. code-block:: pycon

    >>> sdf(Array3f([0, 1], [0, 2], [0, 3]))
    [-1.0, 2.7416574954986572]

To visualize the surface encoded by the SDF, we will use an algorithm called
`sphere tracing
<https://graphics.stanford.edu/courses/cs348b-20-spring-content/uploads/hart.pdf>`_.
Given a ray with an origin :math:`\textbf{o}` and direction :math:`\textbf{d}`,
sphere tracing evaluates :math:`\mathrm{sdf}(\textbf{o})` to find the distance
of the nearest surface. The line segment connecting :math:`\textbf{o}` and
:math:`\mathbf{o} + \mathbf{d}\cdot\mathrm{sdf}(\textbf{o})` is free of
surfaces by construction, and the algorithm thus skips to the end of this
interval. Further repetition of this recipe causes the method to either
approach the nearest surface intersection :math:`\textbf{p}` (visualized below)
or escape to infinity.

.. image:: images/sdf.svg
  :width: 500
  :align: center
  :alt: Sphere tracing
  :class: only-light

.. image:: images/sdf-dark.svg
  :width: 500
  :align: center
  :alt: Sphere tracing
  :class: only-dark
   light

The following sphere tracer runs for 10 fixed iteration and lacks various
common optimizations for simplicity. The function :py:func:`fma` performs a
*fused multiply-add*, i.e., it evaluates ``fma(a, b, c) = a*b + c`` with
reduced rounding error and better performance.

.. code-block:: python

    def trace(o: Array3f, d: Array3f) -> Array3f:
        for i in range(10):
            o = dr.fma(d, sdf(o), o)
        return o

So far, so good. Now suppose ``p = trace(o, d)`` finds an intersection
``p``. To use this information to create an image, we must *shade* it (i.e.,
assign an intensity value).

Many different shading models exist; a simple approach is to compute inner
product of the *surface normal* and the direction :math:`\mathbf{l}` towards a
light source. Intuitively, the surface becomes brighter as it more directly
faces the light source. In the case of a signed distance function, the surface
normal at :math:`\mathbf{p}` is given by the gradient vector :math:`\nabla
\mathrm{sdf}(\mathbf{p})` so that this shading model entails computing

.. math::

   \mathrm{max}\{0, \nabla \mathrm{sdf}(\mathbf{p}) \cdot \mathbf{l}\}

The gradient can be estimated using central `finite differences
<https://en.wikipedia.org/wiki/Finite_difference>`_ with step size
``eps=1e-3``, which yields the following rudimentary shading routine (we
will improve upon it shortly).

.. code-block:: python

    def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
        n = Array3f(
            sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
            sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
            sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps])
        ) / (2 * eps)
        return dr.maximum(0, dr.dot(n, l))

To create an image, we must generate a set of rays that will be processed by
these functions. We begin by creating a ``Float`` array with 1000 linearly
spaced elements covering the interval :math:`[-1, 1]` and then expand this into
a set of :math:`1000\times 1000` :math:`x` and :math:`y` grid coordinates. The
:py:func:`linspace` and :py:func:`meshgrid` functions resemble their eponymous
counterparts in array programming libraries like NumPy.

.. code-block:: python

    x = dr.linspace(Float, -1, 1, 1000)
    x, y = dr.meshgrid(x, x)

This is a good point for a small digression to explain a major difference to
tools like NumPy.

Tracing and delayed evaluation
------------------------------

In most array programming frameworks, the previous two commands would have
created arrays representing actual data (grid coordinates in this example).

Dr.Jit uses a different approach termed *tracing* to delay the evaluation of
computation. In particular, no arithmetic took place during the two preceding
steps: instead, Dr.Jit recorded a graph representing the sequence of steps that
are needed to *eventually* compute ``x`` and ``y`` (which are represented by
the bottom two nodes in the visualization below).

.. image:: images/graph.png
  :width: 400
  :align: center
  :alt: Computation graph of previous steps
  :class: only-light

.. image:: images/graph-dark.png
  :width: 400
  :align: center
  :alt: Computation graph of previous steps
  :class: only-dark

.. note::

    To view a computation graph like this on your own machine, you must install
    `GraphViz <https://graphviz.org>`_ on your system along with the `graphviz
    <https://pypi.org/project/graphviz/>`_ Python package. Following this, you
    can run ``dr.graphviz().view()``.

It is clear that the evaluation can not be postponed arbitrarily: we will
eventually want to look at the generated image. At this point, Dr.Jit will take
all recorded steps, compile them into an optimized *kernel*, and run it on the
GPU or CPU. This all happens transparently behind the scenes.

What are the benefits of doing things in this way? Merging multiple steps of a
computation into a kernel (often called *fusion*) means that these steps can
exchange information using fast register memory. This allows them to spend more
time on the actual computation as opposed to reading and writing main memory
(which is slow). Tracing also opens up other optimization opportunities
explained in the `paper and video
<https://rgl.epfl.ch/publications/Jakob2022DrJit>`_ explaining the system's
design. Dr.Jit can trace enormously large programs without interruption and use
the graph representation to simplify them.

Example, continued
------------------

We will now use the previously computed grid points to define a virtual camera
plane with pixel positions :math:`(x, y, 1)` relative to a pinhole at
:math:`(0, 0, -2)` and simultaneously perform sphere tracing along every
associated ray.

.. code-block:: python

    p = trace(o=Array3f(0, 0, -2),
              d=dr.normalize(Array3f(x, y, 1)))

Next, we can shade the intersected points for light arriving from direction
:math:`(0, -1, -1)`. Note the *masked assignment* at the bottom, which disables
shading for rays that did not intersect anything.

.. code-block:: python

    sh = shade(p, l=Array3f(0, -1, -1))
    sh[sdf(p) > .1] = 0

We we multiply and offset the shaded value with an ambient and highlight color.
The resulting variable ``img`` associates an RGB color value with every pixel.

.. code-block:: python

    img = Array3f(.1, .1, .2) + Array3f(.4, .4, .2) * sh

If you are used to array programming frameworks like NumPy/PyTorch, it may be
tempting to think of ``img`` as a tensor that points to a ``3xN`` or
``Nx3``-shaped block of memory (where ``N`` is the pixel count).

Dr.Jit instead traces computation for delayed evaluation, which means that no
actual computation has occurred so far. The 3D array ``img`` (type
:py:class:`drjit.cuda.Array3f`) consists of 3 components (``img.x``, ``img.y``,
and ``img.z``) of type :py:class:`drjit.cuda.Float`, of which each represents
an intermediate variable within a steadily growing program of the following
high-level structure.

.. code-block:: python

    # For illustration only, not part of the running example

    for i in range(1000000): # (in parallel)
        # .. earlier steps ..
        img_x = .1 + .4 * sh
        img_y = .1 + .4 * sh
        img_z = .2 + .2 * sh

This program performs a parallel loop over :math:`1000\times1000` pixels.
Subsequent Dr.Jit operations will simply add further steps to this program. For
example, we can invoke :py:func:`ravel` to flatten the 3D array into a
:py:class:`drjit.cuda.Float` array.

.. code-block:: python

    img_flat = dr.ravel(img)

Conceptually, this adds three more lines to the program

.. code-block:: python

    # For illustration only, not part of the running example

    for i in range(1000000): # (in parallel)
        # .. earlier steps ..
        img_flat[i*3 + 0] = img_x
        img_flat[i*3 + 1] = img_y
        img_flat[i*3 + 2] = img_z

This is essentially *metaprogramming*: running the program generates *another*
program that will run at some later point and perform the actual computation.
This all happens automatically and is key to the efficiency of Dr.Jit.

Dr.Jit also supports arbitrarily sized tensors of various types (for example,
:py:class:`drjit.cuda.TensorXf` for a CUDA ``float32`` tensor). Tensors are
useful for *data exchange* with other array programming frameworks. For
example, we can reshape the flat image buffer into a :math:`1000\times
1000\times 3` image tensor and then visualize it using `matplotlib
<https://matplotlib.org>`_.

.. code-block:: python

    img_t = TensorXf(img_flat, shape=(1000, 1000, 3))

    import matplotlib.pyplot as plt
    plt.imshow(img_t)
    plt.show()

.. warning::

    Despite the presence of a tensor type, Dr.Jit is *not* a tensor/array
    programming library. Heavy use of tensor operations like slice-based
    indexing may lead to poor performance, since they impede Dr.Jit's
    ability to *fuse* many operations into large kernels.

    Programs should be mainly written in terms of 1D arrays
    (:py:class:`drjit.cuda.Float`, :py:class:`drjit.cuda.UInt32`,
    :py:class:`drjit.cuda.Int64`, etc.) and fixed-size combinations. For
    example, :py:class:`drjit.cuda.Matrix4f` wraps :math:`4\times 4=16`
    :py:class:`drjit.cuda.Float` instances, each of which represents
    a variable in the program.

The line ``plt.imshow(img_t)`` will access the image contents, and it is at
this point that the traced program runs on the GPU, producing the following
output:

.. image:: images/sphere.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere
  :class: only-light

.. image:: images/sphere-dark.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere
  :class: only-dark

.. admonition:: Complete example code up to this point.
   :class: dropdown

    .. code-block:: python

        import drjit as dr
        from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

        def sdf(p: Array3f) -> Float:
            return dr.norm(p) - 1

        def trace(o: Array3f, d: Array3f) -> Array3f:
            for i in range(10):
                o = dr.fma(d, sdf(o), o)
            return o

        def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
            n = Array3f(
                sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
                sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
                sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps])
            ) / (2 * eps)
            return dr.maximum(0, dr.dot(n, l))

        x = dr.linspace(Float, -1, 1, 1000)
        x, y = dr.meshgrid(x, x)

        p = trace(o=Array3f(0, 0, -2),
                  d=dr.normalize(Array3f(x, y, 1)))

        sh = shade(p, l=Array3f(0, -1, -1))
        sh[sdf(p) > .1] = 0

        img = Array3f(.1, .1, .2) + Array3f(.4, .4, .2) * sh
        img_flat = dr.ravel(img)

        img_t = TensorXf(img_flat, shape=(1000, 1000, 3))

        import matplotlib.pyplot as plt
        plt.imshow(img_t)
        plt.show()

Textures, random number generation
----------------------------------

This previous example was a little bland—let's make it more interesting!
We will deform the sphere by perturbing the implicitly defined surface with
a noise function.

Dr.Jit was originally designed for `Monte Carlo methods
<https://en.wikipedia.org/wiki/Monte_Carlo_method>`_ that heavily rely on
random sampling, and it ships with Melissa O'Neill's `PCG32
<https://www.pcg-random.org/index.html>`_ pseudorandom number generator to
help with such applications.

Here, we use PCG32 to generate a relatively small set of uniformly distributed
variates covering the interval :math:`[0, 1]`.

.. code-block:: python

    noise = PCG32(size=16*16*16).next_float32()

We can then create a noise texture from these uniform variates. The command
below allocates a 3D texture with a resolution of :math:`16\times16\times 16`
and :math:`1` color channel.

.. code-block:: python

    noise_tex = Texture3f(TensorXf(noise, shape=(16, 16, 16, 1)))

We finally replace the ``sdf()`` function with a modified version that
evaluates the texture with an offset and scaled value of ``p`` to slightly
perturb the level set. This uses the GPU texture units on the CUDA backend and
a software-interpolated lookup in the LLVM backend.

.. code-block:: python

    def sdf(p: Array3f) -> Float:
        sdf_value = dr.norm(p) - 1
        sdf_value += noise_tex.eval(dr.fma(p, 0.5,  0.5))[0] * 0.1
        return sdf_value

Let us also add the following line at the beginning of the program, which
causes Dr.Jit to emit a brief message whenever it compiles and runs a kernel.

.. code-block:: python

    dr.set_log_level(dr.LogLevel.Info)

Re-running the program produces the following output:

.. image:: images/sphere2.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere with trilinear noise
  :class: only-light

.. image:: images/sphere2-dark.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere with trilinear noise
  :class: only-dark

Why does it look so *faceted*? The texture uses trilinear interpolation, and
the surface normal is given by the *derivative* of the interpolant (meaning
that it will be *piecewise constant*). Dr.Jit also provides higher-order
tricubic interpolation that internally reduces to eight hardware-accelerated
texture lookups. We can use it to redefined ``sdf()`` once more:

.. code-block:: python

    def sdf(p: Array3f) -> Float:
        sdf_value = dr.norm(p) - 1
        sdf_value += noise_tex.eval_cubic(dr.fma(p, 0.5,  0.5))[0] * 0.1
        return sdf_value

With this implementation, we obtain a smooth bumpy sphere.

.. image:: images/sphere3.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere with tricubic noise
  :class: only-light

.. image:: images/sphere3-dark.png
  :width: 400
  :align: center
  :alt: Computed image of a sphere with tricubic noise
  :class: only-dark

.. admonition:: Complete example code up to this point.
   :class: dropdown

    .. code-block:: python

        import drjit as dr
        from drjit.cuda import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

        dr.set_log_level(dr.LogLevel.Info)

        noise = PCG32(size=16*16*16).next_float32()
        noise_tex = Texture3f(TensorXf(noise, shape=(16, 16, 16, 1)))

        def sdf(p: Array3f) -> Float:
            sdf_value = dr.norm(p) - 1
            sdf_value += noise_tex.eval_cubic(dr.fma(p, 0.5,  0.5))[0] * 0.1
            return sdf_value

        def trace(o: Array3f, d: Array3f) -> Array3f:
            for i in range(10):
                o = dr.fma(d, sdf(o), o)
            return o

        def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
            n = Array3f(
                sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
                sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
                sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps])
            ) / (2 * eps)
            return dr.maximum(0, dr.dot(n, l))

        x = dr.linspace(Float, -1, 1, 1000)
        x, y = dr.meshgrid(x, x)

        p = trace(o=Array3f(0, 0, -2),
                  d=dr.normalize(Array3f(x, y, 1)))

        sh = shade(p, l=Array3f(0, -1, -1))
        sh[sdf(p) > .1] = 0

        img = Array3f(.1, .1, .2) + Array3f(.4, .4, .2) * sh
        img_flat = dr.ravel(img)

        img_t = TensorXf(img_flat, shape=(1000, 1000, 3))

        import matplotlib.pyplot as plt
        plt.imshow(img_t)
        plt.show()

Kernel launches, caching
------------------------

Besides generating an image, the last experiment also produced several log
messages enabled by the call to :py:func:`dr.set_log_level()`.

.. code-block:: pycon
    :emphasize-lines: 2, 3, 6, 7

    jit_eval(): launching 1 kernel.
      -> launching 17509add1324abde (n=4096, in=0, out=1, ops=41, jit=15.073 us):
         cache miss, build: 576.932 us, 3.375 KiB.
    jit_eval(): done.
    jit_eval(): launching 1 kernel.
      -> launching 87908afce75f85b5 (n=1000000, in=5, out=0, se=3, ops=2114, jit=330.965 us):
         cache miss, build: 1.17021 ms, 30.38 KiB.
    jit_eval(): done.

Several things are noteworthy here:

- Dr.Jit launched *two* kernels: the first one to compute the noise texture
  with ``n=4096`` texels, followed by the main rendering step that computed
  ``n=1000000`` image pixels.

- The second kernel is *big* and contains over two thousand operations (``ops=2114``).

- It generated those kernels for the first time (``cache miss``) and so had to
  perform a somewhat expensive compilation step to generate machine code.

  If you re-run the example a second time, this part of the message will change
  to ``cache hit``, and the compilation is skipped. Dr.Jit stores cached
  kernels on disk in the ``~/.drjit`` directory on Linux/macOS, and in
  ``~/AppData/Local/Temp/drjit`` on Windows. Dr.Jit was originally
  designed to accelerate gradient-based optimization; caching is particularly
  useful in this context, since the expensive compilation step will only run
  once during the first gradient step.

- If you are using the LLVM backend, the kernel will be even larger..

  .. code-block:: pycon
      :emphasize-lines: 2, 3

      jit_eval(): launching 1 kernel.
        -> launching 6e8cadb52477dd91 (n=1000000, in=5, out=0, se=3, ops=7560, jit=2.92385 ms):
           cache miss, build: 2.411 s, 78.25 KiB.
      jit_eval(): done.

  The CPU does not have hardware texturing instructions and must emulate them,
  which causes this size increase to over 7K instructions. While tracing is
  fast (2.9 milliseconds), the one-time compilation step now takes almost 2.5
  seconds!

What leads to these large kernels? Not only does the bumpy sphere SDF generate
more code: Dr.Jit's computation graph also contains it a whopping 17 times: 10
times for sphere tracing steps, 6 times for finite differences-based normal
computation, and one final time for the masked assignment that disables pixels
without valid intersections. This doesn't seem like a good way of using the
system—let's improve the example!

Recorded loops
--------------

A first inefficiency is that a normal Python ``for`` loop will unroll the loop
many times, producing an unnecessarily large trace that is expensive to
compile. It is also inflexible: there is no easy way to to stop the sphere
tracing iteration early when it is sufficiently close to the surface.

Dr.Jit provides a *recorded loop* primitive to address these and related
limitations. To use it, replace the earlier sphere tracing implementation

.. code-block:: python

    # Old version
    def trace(o: Array3f, d: Array3f) -> Array3f:
        for i in range(10):
            o = dr.fma(d, sdf(o), o)
        return o

by the following improved version:

.. code-block:: python

    # Improved version
    def trace(o: Array3f, d: Array3f) -> Array3f:
        i = UInt32(0)
        loop = Loop("Sphere tracing", lambda: (o, i))
        while loop(i < 10):
            o = dr.fma(d, sdf(o), o)
            i += 1
        return o

Expressed in this form, Dr.Jit will only trace the body *once* and make note of
the fact that it must loop on the device while the condition ``i < 10`` holds.
The condition is itself a Dr.Jit array, and elements can therefore run the loop
for different numbers of iterations.

For this all to work correctly, Dr.Jit needs to know what variables are
modified by the loop body. The ``lambda: (o, i)`` parameter serves this role
and allows the system to detect when variables are changed or entirely
overwritten. The label ``"Sphere tracing"`` will be added to generated PTX/LLVM
code and can be helpful when looking at kernels of programs containing many
loops. This simple change reduces the operation count to half.

Automatic differentiation
-------------------------

Next, we can examine the ``shade()`` method that evaluated the SDF 6 times to
compute an approximate derivative, which was a source of inefficiency:

.. code-block:: python

    # Old version
    def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
        n = Array3f(
            sdf(p + [eps, 0, 0]) - sdf(p - [eps, 0, 0]),
            sdf(p + [0, eps, 0]) - sdf(p - [0, eps, 0]),
            sdf(p + [0, 0, eps]) - sdf(p - [0, 0, eps])
        ) / (2 * eps)
        return dr.maximum(0, dr.dot(n, l))

Dr.Jit includes an `automatic differentiation
<https://en.wikipedia.org/wiki/Automatic_differentiation>`_ layer to
analytically differentiate expressions, producing code that is more efficient
*and* more accurate. To use the AD layer, simple append ``.ad`` to the import
directive at the top of the program. For example for the CUDA backend, you
would write:

.. code-block:: python

    from drjit.cuda.ad import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

There is essentially no extra cost for using types from the ``.ad`` namespace
when gradient tracking isn't explicitly enabled for a variable, so you can
simply use them everywhere by default.

The AD version of ``shade()`` invokes :py:func:`drjit.enable_grad()` to track
the differential dependence of subsequent variables on the position ``p``. It
subsequently evaluates the SDF just once, which records the structure of the
computation into a graph representation. The next two lines set an input
gradient at ``p`` and propagate the derivative to the output ``value``, which
results in the desired directional derivative :math:`\nabla
\mathrm{sdf}(\mathbf{p}) \cdot \mathbf{l}`.

.. code-block:: python

    # Improved version
    def shade(p: Array3f, l: Array3f) -> Float:
        dr.enable_grad(p)
        value = sdf(p)
        dr.set_grad(p, l)
        dr.forward_to(value)
        return dr.maximum(0, dr.grad(value))

The :py:func:`dr.forward_to()` call materializes the AD-based
derivatives into ordinary computation that is traced along with
the rest of the program.

This reduces the operation count by another factor of 2, and compilation
time is now consistently between 30-90 milliseconds across backends.

.. admonition:: Complete example code including optimizations
   :class: dropdown

    .. code-block:: python

        import drjit as dr
        from drjit.cuda.ad import Float, UInt32, Array3f, Array2f, TensorXf, Texture3f, PCG32, Loop

        dr.set_log_level(dr.LogLevel.Info)

        noise = PCG32(size=16*16*16).next_float32()
        noise_tex = Texture3f(TensorXf(noise, shape=(16, 16, 16, 1)))

        def sdf(p: Array3f) -> Float:
            sdf_value = dr.norm(p) - 1
            sdf_value += noise_tex.eval_cubic(dr.fma(p, 0.5,  0.5))[0] * 0.1
            return sdf_value

        def trace(o: Array3f, d: Array3f) -> Array3f:
            i = UInt32(0)
            loop = Loop("Sphere tracing", lambda: (o, i))
            while loop(i < 10):
                o = dr.fma(d, sdf(o), o)
                i += 1
            return o

        def shade(p: Array3f, l: Array3f, eps: float = 1e-3) -> Float:
            dr.enable_grad(p)
            value = sdf(p);
            dr.set_grad(p, l)
            dr.forward_to(value)
            return dr.maximum(0, dr.grad(value))

        x = dr.linspace(Float, -1, 1, 1000)
        x, y = dr.meshgrid(x, x)

        p = trace(o=Array3f(0, 0, -2),
                  d=dr.normalize(Array3f(x, y, 1)))

        sh = shade(p, l=Array3f(0, -1, -1))
        sh[sdf(p) > .1] = 0

        img = Array3f(.1, .1, .2) + Array3f(.4, .4, .2) * sh
        img_flat = dr.ravel(img)

        img_t = TensorXf(img_flat, shape=(1000, 1000, 3))

        import matplotlib.pyplot as plt
        plt.imshow(img_t)
        plt.show()

Dr.Jit can propagate derivatives in forward mode (shown here) and reverse mode,
which is useful for gradient-based optimization of programs with many inputs.

Features
--------

Many features weren't covered in this basic tutorial.
Dr.Jit also

- supports polymorphic/virtual function calls, in which a program jumps to one of
  many locations. It can efficiently trace and differentiate such indirection.

- provides a library of transcendental functions (ordinary and hyperbolic trig functions, exponentials, logarithms, elliptic integrals, etc).

- provides types for complex arithmetic, quaternions, and small (< :math:`4\times 4`) matrices.

- provides efficient code for evaluating spherical harmonics.
