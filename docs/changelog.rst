.. py:currentmodule:: drjit

.. _changelog:

Changelog
#########

Upcoming changes
--------------------------------

**New Features**

- **Atomic Scatter Operations**: Added :py:func:`dr.scatter_cas()
  <scatter_cas>` (atomic compare-and-swap) and :py:func:`dr.scatter_exch()
  <scatter_exch>` (atomic exchange) operations. On the CUDA backend, these map
  to native PTX instructions; the LLVM implementation uses a loop over the
  vectorization width.
  (Dr.Jit PR `#450 <https://github.com/mitsuba-renderer/drjit/pull/450>`__,
  Dr.Jit-Core PR `#177 <https://github.com/mitsuba-renderer/drjit-core/pull/177>`__).

- **AdamW Optimizer**: Added the :py:class:`dr.opt.AdamW <opt.AdamW>`
  optimizer with built-in weight decay, equivalent to PyTorch's implementation.
  (PR `#449 <https://github.com/mitsuba-renderer/drjit/pull/449>`__).

- **AMSGrad for Adam/AdamW**: The :py:class:`dr.opt.Adam <opt.Adam>` and
  :py:class:`dr.opt.AdamW <opt.AdamW>` optimizers now support an optional
  ``amsgrad`` parameter. AMSGrad keeps a running maximum of the second moments,
  which can help improve stability near local minima.
  (PR `#467 <https://github.com/mitsuba-renderer/drjit/pull/467>`__).

- **Functions in IR** :py:func:`dr.func`: A new function decorator that
  forces a Python function to also become a callable in the generated IR. This
  can improve compilation times: without it, Dr.Jit emits the function body's
  IR every time it is called within a single kernel. With ``@dr.func``, each
  call resolves to a function call in the IR, emitting the body only once.
  (Dr.Jit PR `#473 <https://github.com/mitsuba-renderer/drjit/pull/473>`__,
  Dr.Jit-Core PR `#183 <https://github.com/mitsuba-renderer/drjit-core/pull/183>`__).

- **Oklab Color Space Conversion**: Added :py:func:`dr.linear_srgb_to_oklab()
  <linear_srgb_to_oklab>` and :py:func:`dr.oklab_to_linear_srgb()
  <oklab_to_linear_srgb>` for perceptually uniform color space conversion.
  (PR `#453 <https://github.com/mitsuba-renderer/drjit/pull/453>`__).

- **Pickling Support**: Dr.Jit arrays can now be natively pickled and
  unpickled via Python's ``pickle`` module.
  (PR `#448 <https://github.com/mitsuba-renderer/drjit/pull/448>`__).

- **Bounded Integer RNG**: Added :py:meth:`dr.rng().integers()
  <random.Generator.integers>` to generate uniformly distributed integers on a
  given interval. (commit `cb09caa
  <https://github.com/mitsuba-renderer/drjit/commit/cb09caac>`__).

- **Symbolic RNG mode**: :py:func:`dr.rng() <rng>` now accepts a
  ``symbolic`` argument for a purely symbolic sampler. (commit `51bacbf
  <https://github.com/mitsuba-renderer/drjit/commit/51bacbf4>`__).

- **ArrayX Initialization from Tensors**: Nested array types with multiple
  dynamic dimensions (like ``ArrayXf``) can now be initialized from Dr.Jit
  tensors or NumPy arrays. (commit `e7e1339
  <https://github.com/mitsuba-renderer/drjit/commit/e7e13399>`__).

- **Type Trait**: Added :py:func:`dr.replace_shape_t() <replace_shape_t>`
  convenience type trait for writing generic functions that need to reshape
  array types. (commit `4643452
  <https://github.com/mitsuba-renderer/drjit/commit/46b24535>`__).

**Hardware/platform-specfic features**

- **NVIDIA Blackwell (SM120+)**: Added support for wide packet loads, gathers,
  and atomics on NVIDIA Blackwell GPUs (SM120+). (commit `879c103
  <https://github.com/mitsuba-renderer/drjit/commit/879c103b>`__).

- **Python 3.14 Compatibility**: Fixed compatibility with PEP 649 deferred
  annotation evaluation, ensuring Dr.Jit works correctly on Python 3.14.
  (commit `7fa6eb4
  <https://github.com/mitsuba-renderer/drjit/commit/7fa6eb4b>`__).

- **Linux ARM Wheels**: Added ``ubuntu-24.04-arm`` to the wheels pipeline.
  (PR `#461 <https://github.com/mitsuba-renderer/drjit/pull/461>`__,
  contributed by `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

**Performance Improvements**

- **Simplified Single-Target Virtual Calls**: When a virtual function call has
  only a single target (as is the case for ``@dr.func``), the JIT backend now
  eliminates the indirection/dispatch loop and calls the function directly,
  producing simpler IR.
  (Dr.Jit-Core PR `#183 <https://github.com/mitsuba-renderer/drjit-core/pull/183>`__).

- **AD Early Exit for Zero Derivatives**: The AD graph traversal now skips
  edges with zero-valued derivatives, avoiding unnecessary computation.
  (commit `06b0a9d
  <https://github.com/mitsuba-renderer/drjit/commit/06b0a9db>`__).

- **GIL Release in __getitem__**: ``dr.ArrayBase.__getitem__()`` now releases
  the GIL while waiting, improving multi-threaded performance.
  (commit `c24be70
  <https://github.com/mitsuba-renderer/drjit/commit/c24be704>`__).

**Bug Fixes**

- Fixed a bug where constructing a cooperative vector inside a
  ``dr.suspend_grad()`` scope could raise an exception.
  (PR `#475 <https://github.com/mitsuba-renderer/drjit/pull/475>`__,
  contributed by `Christian Döring <https://github.com/DoeringChristian>`__).

- Fixed a crash when calling a frozen function with a re-seeded random number
  generator whose seed was a Python integer type.
  (PR `#471 <https://github.com/mitsuba-renderer/drjit/pull/471>`__,
  contributed by `Christian Döring <https://github.com/DoeringChristian>`__).

- Fixed a bug in the C++ ``transform_compose()`` function where the
  translation was placed in the last row of the matrix rather than the last
  column.
  (PR `#451 <https://github.com/mitsuba-renderer/drjit/pull/451>`__,
  contributed by `Delio Vicini <https://github.com/dvicini>`__).

- Fixed multiple issues in the Dr.Jit-Core ``gather`` re-indexing logic: the
  mask stack is now correctly applied during re-indexing, and nested gather
  masks are combined rather than overwritten.
  (Dr.Jit-Core PR `#178 <https://github.com/mitsuba-renderer/drjit-core/pull/178>`__).

- Fixed a bug in virtual call analysis when a target contained a symbolic
  loop — the analysis now accounts for eliminated/optimized-out loop state
  variables.
  (Dr.Jit-Core PR `#184 <https://github.com/mitsuba-renderer/drjit-core/pull/184>`__).

- Fixed LLVM backend compilation of wavefront loops with scalar masks.
  (commit `16a81d0
  <https://github.com/mitsuba-renderer/drjit/commit/16a81d08>`__).

- Fixed lost tensor shapes when a loop or conditional is replayed for AD
  passes, with more robust inference of tensor output shapes.
  (commit `9d201f2
  <https://github.com/mitsuba-renderer/drjit/commit/9d201f20>`__).

- Fixed a regression in ``ArrayX`` initialization from tensors and NumPy
  ndarrays (wrong shape hint order for flipped axes and broken shift loop).
  (commit `df4cf48
  <https://github.com/mitsuba-renderer/drjit/commit/df4cf483>`__).

- Fixed ``Texture::eval_fetch_cuda`` to handle double-precision queries
  gracefully by casting to single-precision when a HW-accelerated texture is
  requested. (commits `83083d8
  <https://github.com/mitsuba-renderer/drjit/commit/83083d8a>`__,
  `054d115
  <https://github.com/mitsuba-renderer/drjit/commit/054d1150>`__).

- Fixed symbolic loop size computation to also account for side-effect sizes.
  (Dr.Jit-Core commit `c6dfc83
  <https://github.com/mitsuba-renderer/drjit-core/commit/c6dfc83>`__).

- Fixed spurious warning when freezing functions with very wide literals.
  (PR `#455 <https://github.com/mitsuba-renderer/drjit/pull/455>`__).

**Other Improvements**

- Updated to nanobind `v2.10.2
  <https://github.com/wjakob/nanobind/releases/tag/v2.10.2>`__.

- Improved documentation and log messages for textures, including
  clarifications regarding numerical precision and extra diagnostics for
  migrated textures. (commit `4edae0a
  <https://github.com/mitsuba-renderer/drjit/commit/4edae0af>`__).

DrJit 1.2.0 (September 17, 2025)
--------------------------------

**New Features**

- **Event API**: Added an event API for fine-grained timing and synchronization
  of GPU kernels. This enables more detailed performance profiling and better
  control over asynchronous operations.
  (Dr.Jit PR `#441 <https://github.com/mitsuba-renderer/drjit/pull/441>`__,
  Dr.Jit-Core PR `#174 <https://github.com/mitsuba-renderer/drjit-core/pull/174>`__).

- **OpenGL Interoperability**: Improved CUDA-OpenGL interoperability with
  simplified APIs. This enables efficient sharing of data between CUDA kernels
  and OpenGL rendering.
  (Dr.Jit PR `#429 <https://github.com/mitsuba-renderer/drjit/pull/429>`__,
  Dr.Jit-Core PR `#164 <https://github.com/mitsuba-renderer/drjit-core/pull/164>`__,
  contributed by `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

- **Enhanced Int8/UInt8 Support**: Improved support for 8-bit integer types
  with better casting and bitcast operations.
  (Dr.Jit PR `#428 <https://github.com/mitsuba-renderer/drjit/pull/428>`__,
  Dr.Jit-Core PR `#163 <https://github.com/mitsuba-renderer/drjit-core/pull/163>`__,
  contributed by `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

**Performance Improvements**

- **Register Spilling to Shared Memory**: CUDA backend now supports spilling
  registers to shared memory, improving performance for kernels with high
  register pressure. (Dr.Jit-Core commit `fdc7cae7`).

- **Memory View Support**: Arrays can now be converted to Python ``memoryview``
  objects for efficient zero-copy data access. (commit `b7039184`).

- **DLPack GIL Release**: The ``dr.ArrayBase.dlpack()`` method now releases
  the GIL while waiting, improving multi-threaded performance. (commit `0adf9b4a`).

- **Thread Synchronization**: ``dr.sync_thread()`` now releases the GIL while
  waiting, preventing unnecessary blocking in multi-threaded applications.
  (commit `956d2f57`).

**API Improvements**

- **Spherical Direction Utilities**: Added Python implementation of spherical
  direction utilities (``dr.sphdir``).
  (PR `#432 <https://github.com/mitsuba-renderer/drjit/pull/432>`__,
  contributed by `Sébastien Speierer <https://github.com/Speierers>`__).

- **Matrix Conversions**: Added support for converting between 3D and 4D
  matrices: ``Matrix4f`` can be constructed from a 3D matrix and ``Matrix3f``
  from a 4D matrix. (commit `7f8ea890`).

- **Quaternion API**: Improved the quaternion Python API for better usability
  and consistency. (commit `282da88a`).

- **Type casts**: Allow casting between Dr.Jit types to properly allow
  AD<->non-AD conversions when required. (commit `72f1e6b2`).

**Bug Fixes**

- Fixed deadlock issues in ``@dr.freeze`` decorator. (commit `e8fc555e`).

- Fixed gradient tracking in ``Texture.tensor()`` to ensure gradients are
  never dropped inadvertently.
  (PR `#444 <https://github.com/mitsuba-renderer/drjit/pull/444>`__).

- Fixed AD support for C++ ``repeat`` and ``tile`` operations with proper
  gradient propagation. (commits `fd693056`, `282da88a`).

- Fixed Python object traversal to check that ``__dict__`` exists before
  accessing it, preventing crashes with certain object types. (commit `433adaf0`).

- Fixed symbolic loop size calculation to properly account for side-effects.
  (Dr.Jit-Core commit `31bf911`).

- Fixed read-after-free issue in OptiX SBT data loading.
  (Dr.Jit-Core commit `009adef`, contributed by `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

**Other Improvements**

- Updated to nanobind `v2.9.2 <https://github.com/wjakob/nanobind/releases/tag/v2.9.2>`__

- Improved error messages by adding function names to vectorized call errors.
  (Dr.Jit-Core PR `#165 <https://github.com/mitsuba-renderer/drjit-core/pull/165>`__,
  contributed by `Sébastien Speierer <https://github.com/Speierers>`__).

- Added missing checks for JIT leak warnings.
  (Dr.Jit-Core PR `#166 <https://github.com/mitsuba-renderer/drjit-core/pull/166>`__,
  contributed by `Sébastien Speierer <https://github.com/Speierers>`__).

- Added warning for LLVM API initialization failures.
  (Dr.Jit-Core PR `#168 <https://github.com/mitsuba-renderer/drjit-core/pull/168>`__,
  contributed by `Sébastien Speierer <https://github.com/Speierers>`__).

- Fixed pytest warnings and improved test infrastructure.
  (PR `#436 <https://github.com/mitsuba-renderer/drjit/pull/436>`__).

DrJit 1.1.0 (August 7, 2025)
----------------------------

The v1.1.0 release of Dr.Jit includes several major new features:

**Major Features**

- **Cooperative Vectors**: Dr.Jit now provides an API to efficiently evaluate
  matrix-vector products in parallel programs. The API targets small matrices
  (e.g., 128x128, 64×64, or smaller) and inlines all computation into the program.
  Threads work cooperatively to perform these operations efficiently. On NVIDIA
  GPUs (Turing or newer), this leverages the OptiX cooperative vector API with
  tensor core acceleration. On the LLVM backend, operations compile to
  sequences of packet instructions (e.g., AVX512). See the :ref:`cooperative
  vector documentation <coop_vec>` for more details. Example:

  .. code-block:: python

     import drjit as dr
     import drjit.nn as nn
     from drjit.cuda.ad import Float16, TensorXf16

     # Create a random number generator
     rng = dr.rng(seed=0)

     # Create a matrix and bias representing an affine transformation
     A = rng.normal(TensorXf16, shape=(3, 16))  # 3×16 matrix
     b = TensorXf16([1, 2, 3])                  # Bias vector

     # Pack into optimized memory layout
     buffer, A_view, b_view = nn.pack(A, b)

     # Create cooperative a vector from 16 inputs
     vec_in = nn.CoopVec(Float16(1), Float16(2), ...)

     # Perform matrix-vector multiplication: A @ vec_in + b
     vec_out = nn.matvec(A_view, vec_in, b_view)

     # Unpack result back to regular arrays
     x, y, z = vec_out

  (Dr.Jit PR `#384 <https://github.com/mitsuba-renderer/drjit/pull/384>`__,
  Dr.Jit-Core PR `#141 <https://github.com/mitsuba-renderer/drjit-core/pull/141>`__).

- **Neural Network Library**: Building on the cooperative vector functionality,
  the new :py:mod:`drjit.nn` module provides modular abstractions for
  constructing, evaluating, and optimizing neural networks, similar to
  PyTorch's ``nn.Module``. This enables fully fused evaluation of small
  multilayer perceptrons (MLPs) within larger programs. See the :ref:`neural
  network module documentation <neural_nets>` for more details. Example:

  .. code-block:: python

     import drjit.nn as nn
     from drjit.cuda.ad import TensorXf16, Float16

     # Define a small MLP for function approximation
     net = nn.Sequential(
         nn.SinEncode(16),                 # Sinusoidal encoding
         nn.Linear(-1, -1, bias=False),    # Hidden layer
         nn.ReLU(),
         nn.Linear(-1, -1, bias=False),    # Hidden layer
         nn.ReLU(),
         nn.Linear(-1, 3, bias=False),     # Output layer (3 outputs)
         nn.Tanh()
     )

     # Instantiate and optimize for 16-bit tensor cores
     rng = dr.rng(seed=0)
     net = net.alloc(dtype=TensorXf16, size=2, rng=rng)
     weights, net = nn.pack(net, layout='training')

     # Evaluate the network
     inputs = nn.CoopVec(Float16(0.5), Float16(0.7))
     outputs = net(inputs)
     x, y, z = outputs  # Three output values

  (PR `#384 <https://github.com/mitsuba-renderer/drjit/pull/384>`__).

- **Hash Grid Encoding**: Added neural network hash grid encoding inspired by
  `Instant NGP <https://nvlabs.github.io/instant-ngp>`__, providing
  multi-resolution spatial encodings. This includes both traditional hash grids
  and `permutohedral encodings <https://radualexandru.github.io/permuto_sdf>`__
  for efficient high-dimensional inputs. (PR `#390
  <https://github.com/mitsuba-renderer/drjit/pull/390>`__, contributed by
  `Christian Döring <https://github.com/DoeringChristian>`__
  and `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

- **Function Freezing**: Added the :py:func:`@dr.freeze <freeze>` decorator
  to eliminate repeated tracing overhead by caching and replaying
  JIT-compiled kernels. Dr.Jit normally traces operations to build
  computation graphs for compilation, which can become a bottleneck
  when the same complex computation is performed repeatedly (e.g., in optimization
  loops). The decorator records kernel launches on the first call and replays
  them directly on subsequent calls, avoiding re-tracing.

  This can dramatically accelerate programs and makes Dr.Jit usable for
  realtime rendering and other applications with strict timing requirements.
  See the :ref:`function freezing documentation <freeze>` for more details.
  Example:

  .. code-block:: python

     import drjit as dr
     from drjit.cuda import Float, UInt32

     # Without freezing - traces every time
     def func(x):
         y = seriously_complicated_code(x)
         dr.eval(y) # ..intermediate evaluations..
         return huge_function(y, x)

     # With freezing - traces only once
     @dr.freeze
     def frozen(x):
         ... # same code as above -- no changes needed

  (Dr.Jit PR `#336 <https://github.com/mitsuba-renderer/drjit/pull/336>`__,
  Dr.Jit-Core PR `#107 <https://github.com/mitsuba-renderer/drjit-core/pull/107>`__,
  contributed by `Christian Döring <https://github.com/DoeringChristian>`__).

- **Shader Execution Reordering (SER)**: Added the function
  :py:func:`dr.reorder_threads() <reorder_threads>` to shuffle threads across
  the GPU to reduce warp-level divergence. When threads in a warp take
  different branches (e.g., in :py:func:`dr.switch() <switch>` statements or
  :ref:`vectorized virtual function calls <cpp-vcall>`) performance can
  degrade significantly. SER can group threads with similar execution paths
  into coherent warps to avoid this. This feature is a no-op in LLVM mode.
  Example:

  .. code-block:: python

     import drjit as dr
     from drjit.cuda import Array3f, UInt32

     arg = Array3f(...) # Prepare data and callable index
     callable_idx = UInt32(...) % 4  # 4 different callables

     # Reorder threads before dr.switch() to reduce divergence
     # The key uses 2 bits (for 4 callables)
     arg = dr.reorder_threads(key=callable_idx, num_bits=2, value=arg)

     # Now, threads with the same callable_idx are grouped together
     callables = [func0, func1, func2, func3]
     out = dr.switch(callable_idx, callables, arg)

  (Dr.Jit PR `#395 <https://github.com/mitsuba-renderer/drjit/pull/395>`__,
  Dr.Jit-Core PR `#145 <https://github.com/mitsuba-renderer/drjit-core/pull/145>`__).

  Related to this, the OptiX backend now requires the OptiX 8.0 ABI
  (specifically, ABI version 87). This is a requirement for SER. (Dr.Jit-Core
  PR `#117 <https://github.com/mitsuba-renderer/drjit-core/pull/117>`__).

- **Random Number Generation API**: Introduced a new random number generation
  API around an abstract :py:class:`Generator <drjit.random.Generator>` object
  analogous to `NumPy
  <https://numpy.org/doc/2.2/reference/random/generator.html>`__. Under the
  hood, this API uses the :py:class:`Philox4x32 <drjit.auto.Philox4x32>`
  counter-based PRNG from `Salmon et al. [2011]
  <https://www.thesalmons.org/john/random123/papers/random123sc11.pdf>`__,
  which provides high-quality random variates that are statistically
  independent within and across parallel streams. Users create generators with
  :py:func:`dr.rng() <rng>` and call methods like :py:meth:`.random()
  <random.Generator.random>` and :py:meth:`.normal() <random.Generator.normal>`. Example:

  .. code-block:: python

     import drjit as dr
     from drjit.cuda import Float, TensorXf

     # Create a random number generator
     rng = dr.rng(seed=42)

     # Generate various random distributions
     uniform = rng.random(Float, 1000)        # Uniform [0, 1)
     normal = rng.normal(Float, 1000)         # Standard normal
     tensor = rng.random(TensorXf, (32, 32))  # Random tensor

  (PR `#417 <https://github.com/mitsuba-renderer/drjit/pull/417>`__).

- **Array Resampling and Convolution**: Added :py:func:`dr.resample() <resample>`
  for changing the resolution of arrays/tensors along specified axes, and
  :py:func:`dr.convolve() <convolve>` for convolution with continuous kernels.
  Both operations are fully differentiable and support various reconstruction
  filters (box, linear, cubic, lanczos, gaussian). Example:

  .. code-block:: python

     # Resample a 2D signal to different resolution
     data = dr.cuda.TensorXf(original_data)  # Shape: (128, 128)
     upsampled = dr.resample(
         data,
         shape=(256, 256),    # Target resolution
         filter='lanczos'     # High-quality filter
     )

     # Apply Gaussian blur via convolution
     blurred = dr.convolve(
         data,
         filter='gaussian',
         radius=2.0
     )

  (PRs `#358 <https://github.com/mitsuba-renderer/drjit/pull/358>`__,
  `#378 <https://github.com/mitsuba-renderer/drjit/pull/378>`__).

- **Gradient-Based Optimizers**: Added an optimization framework
  that includes various standard optimizers inspired by PyTorch. It includes :py:class:`dr.opt.SGD
  <opt.SGD>` with optional momentum and Nesterov acceleration,
  :py:class:`dr.opt.Adam <opt.Adam>` with adaptive learning rates, and
  :py:class:`dr.opt.RMSProp <opt.RMSProp>`. The optimizers own the parameters
  and automatically handle mixed-precision training. An optional helper class
  :py:class:`dr.opt.GradScalar <opt.GradScaler>` implements adaptive gradient
  scaling for low-precision training.

  .. code-block:: python

     from drjit.opt import Adam
     from drjit.cuda import Float

     # Create optimizer and register parameters
     opt = Adam(lr=1e-3)
     rng = dr.rng(seed=0)
     opt['params'] = Float(rng.normal(Float, 100))

     # Optimization loop for unknown function f(x)
     for i in range(1000):
         # Fetch current parameters
         params = opt['params']

         # Compute loss and gradients
         loss = f(params)  # Some function to optimize
         dr.backward(loss)

         # Update parameters
         opt.step()

  (PRs `#345 <https://github.com/mitsuba-renderer/drjit/pull/345>`__, `#402
  <https://github.com/mitsuba-renderer/drjit/pull/402>`__, commit `e3f576
  <https://github.com/mitsuba-renderer/drjit/commit/e3f57620cb58bac14dfd43189aa1bdf8ba0ff8c0>`__).

- **TensorFlow Interoperability**: Added TensorFlow interop via
  :py:func:`@dr.wrap <wrap>`, supporting forward and backward automatic
  differentiation with comprehensive support for variables and tensors. (PR
  `#301 <https://github.com/mitsuba-renderer/drjit/pull/301>`__, contributed by
  `Jakob Hoydis <https://github.com/jhoydis>`__).

**Array and Tensor Operations**

- Added :py:func:`dr.concat() <concat>` to concatenate arrays/tensors
  along a specified axis following the Array API standard. (PR `#354
  <https://github.com/mitsuba-renderer/drjit/pull/354>`__).

- Added :py:func:`dr.take() <take>` and :py:func:`dr.take_interp()
  <take_interp>` for efficient tensor indexing and interpolated indexing
  along specified axes. (PR `#420
  <https://github.com/mitsuba-renderer/drjit/pull/420>`__,
  commit `b59436
  <https://github.com/mitsuba-renderer/drjit/commit/b59436b0f041af1ea7ba04bd508b39e2e9a43ac8>`__).

- Added :py:func:`dr.moveaxis() <moveaxis>` for rearranging tensor
  dimensions, providing NumPy-compatible axis movement. (commit `4d1478
  <https://github.com/mitsuba-renderer/drjit/commit/4d14784696713f398eee6661913ee11e4d6b1934>`__).

- Implemented comprehensive slice operations for regular (non-tensor) arrays,
  supporting advanced patterns like nested slices and integer array indexing.
  (PR `#365
  <https://github.com/mitsuba-renderer/drjit/pull/365>`__).

- Conversion between tensors and nested arrays (e.g. ``Array3f``) now offers an
  option (``flip_axis=True``) of whether or not to flip the axis order (e.g.,
  `Nx3` vs `3xN`). (PR `#348
  <https://github.com/mitsuba-renderer/drjit/pull/348>`__).

**Performance Improvements**

- Packet scatter-add operations now map to specialized GPU operations when
  supported by the hardware and driver. This change also broadens the
  situations where packet operations can be used on the CPU and GPU. Packets of
  size 6 were not supported in the past since their size was not a power of
  two. Now, they are treated as 3 separate size-2 packets. This feature is
  particularly helpful in combination with the new hash grid class, whose
  reverse-mode derivative generates atomic packet scatter-additions.
  (Dr.Jit-Core PR `#151
  <https://github.com/mitsuba-renderer/drjit-core/pull/151>`__, Dr.Jit PR `#406
  <https://github.com/mitsuba-renderer/drjit/pull/406>`__).

- Enabled packet memory operations for texture access, providing speedups when
  accessing multi-channel textures on the LLVM and CUDA backends. (PR `#329
  <https://github.com/mitsuba-renderer/drjit/pull/329>`__).

- Optimized :py:func:`dr.rsqrt() <rsqrt>` to compile to faster instruction
  sequences on the LLVM backend using ``VRSQRTPS`` with Newton-Raphson
  iteration on Intel processors and similar optimizations for ARM Neon. (Dr.Jit
  PR `#343 <https://github.com/mitsuba-renderer/drjit/pull/343>`__,
  Dr.Jit-Core PR `#125
  <https://github.com/mitsuba-renderer/drjit-core/pull/125>`__).

- Made :py:func:`dr.any() <any>`, :py:func:`dr.all() <all>`, and
  :py:func:`dr.none() <none>` asynchronous with respect to the host, improving
  GPU utilization. (Dr.Jit PR `#344
  <https://github.com/mitsuba-renderer/drjit/pull/344>`__, Dr.Jit-Core PR `#126
  <https://github.com/mitsuba-renderer/drjit-core/pull/126>`__).

**Random Number Generation (contd.)**

- Added PCG32 reverse generation capabilities with ``prev_*`` methods for
  all random number generation functions for bidirectional traversal
  of random sequences. (PR `#398
  <https://github.com/mitsuba-renderer/drjit/pull/398>`__).

- Added PCG32 methods for generating normally distributed variates:
  :py:func:`PCG32.next_float_normal() <drjit.llvm.PCG32.next_float_normal>`,
  :py:func:`PCG32.next_float32_normal() <drjit.llvm.PCG32.next_float32_normal>`,
  and :py:func:`PCG32.next_float64_normal() <drjit.llvm.PCG32.next_float64_normal>`.
  (PR `#353 <https://github.com/mitsuba-renderer/drjit/pull/353>`__).

- Added :py:func:`dr.mul_wide() <mul_wide>` and :py:func:`dr.mul_hi() <mul_hi>`
  for wide integer multiplication, essential for implementing the Philox PRNG.
  (Dr.Jit PR `#414 <https://github.com/mitsuba-renderer/drjit/pull/414>`__,
  Dr.Jit-Core PR `#156
  <https://github.com/mitsuba-renderer/drjit-core/pull/156>`__).

**API Improvements**

- Refined semantics of :py:func:`dr.forward_from() <forward_from>` and
  :py:func:`dr.backward_from() <backward_from>` to preserve existing
  gradients instead of unconditionally overriding them.
  (Dr.Jit PR `#351 <https://github.com/mitsuba-renderer/drjit/pull/351>`__).

- Added utility functions :py:func:`dr.zeros_like() <zeros_like>`,
  :py:func:`dr.ones_like() <ones_like>`, and :py:func:`dr.empty_like()
  <empty_like>`.
  (PR `#345 <https://github.com/mitsuba-renderer/drjit/pull/345/files>`__).

- Added :py:meth:`dr.ArrayBase.item() <ArrayBase.item>` method for extracting scalar values from
  single-element arrays/tensors, similar to NumPy/PyTorch. (commit `a142bc
  <https://github.com/mitsuba-renderer/drjit/commit/a142bcdf2143785880cd57c640630abb8b560d9d>`__).

- Added :py:func:`dr.linear_to_srgb() <linear_to_srgb>` and
  :py:func:`dr.srgb_to_linear() <srgb_to_linear>` for color space conversions.
  (commit `a7f138
  <https://github.com/mitsuba-renderer/drjit/commit/a7f1380cb2e684056b51ef6d08e6ea33154a5d62>`__).

- Added :py:attr:`JitFlag.ForbidSynchronization` to catch costly
  synchronization operations during development. (
  Dr.Jit PR `#350 <https://github.com/mitsuba-renderer/drjit/pull/350>`__,
  Dr.Jit-Core PR `#128
  <https://github.com/mitsuba-renderer/drjit-core/pull/128>`__).

- Added C++ bindings for thread-local memory arrays through the
  ``dr::Local<Value, Size>`` template, complementing the existing Python
  functionality. This enables efficient scratch space and stack-like data
  structures in GPU kernels from C++ code. (commit `c30ade
  <https://github.com/mitsuba-renderer/drjit/commit/c30ade7aa596dac838dedece2e73f5a4a3adcec8>`__).

**Notable Bugfixes**

- Fixed ``dr::block_reduce()`` derivative computation for
  arrays not evenly divisible by block size. (commit `df79ed
  <https://github.com/mitsuba-renderer/drjit/commit/df79ed894a110e2255515e9778032ccac38883a9>`__).

- Fixed potential performance cliffs in :py:func:`dr.gather() <gather>`
  by memoizing expressions and limiting expression growth (Dr.Jit-Core PR `#159
  <https://github.com/mitsuba-renderer/drjit-core/pull/159>`__).

- Fixed :py:func:`dr.rotate() <rotate>` quaternion component ordering to match C++
  implementation. (PR `#416
  <https://github.com/mitsuba-renderer/drjit/pull/416>`__).

- Fixed the derivative of :py:func:`dr.unit_angle() <unit_angle>` at signed zero.
  (commit `9d09a9
  <https://github.com/mitsuba-renderer/drjit/commit/9d09a9e9310b29870756faa8b12fa7b1e60c7396>`__).

- Fixed memory leak in Python bindings using dedicated cleanup thread. (PR `#399
  <https://github.com/mitsuba-renderer/drjit/pull/399>`__).

- Preserve tensor shapes in symbolic operations. (commit `74c4d0
  <https://github.com/mitsuba-renderer/drjit/commit/74c4d0313a420a22dd9e2fe0cb11205f051cb762>`__).

- Fixed evaluated loop derivative issues with unchanged differentiable state
  variables. (commit `074cfe
  <https://github.com/mitsuba-renderer/drjit/commit/074cfe9d0c2dc805af00d562a20c6c268477104d>`__).

- Fixed symbolic loop backward derivative compilation for simple loops.
  (commit `01ef10
  <https://github.com/mitsuba-renderer/drjit/commit/01ef10ef3b5cb147c1c3116d089438dfcb97e2c8>`__).

- Fixed broadcasting of tensors and handling of unknown objects in
  :py:func:`dr.select() <select>`. (PRs `#339
  <https://github.com/mitsuba-renderer/drjit/issue/339>`__, PRs `#349
  <https://github.com/mitsuba-renderer/drjit/issue/349>`__).

- Fixed :py:func:`dr.abs() <abs>` derivative at x=0 to match PyTorch behavior. (commit `c597de
  <https://github.com/mitsuba-renderer/drjit/commit/c597de37d98a494e51bd55fc2f40e68d2258691f>`__).

- Fixes for NVIDIA 50-series GPUs and recent driver versions. (Dr.Jit-Core PR
  `#152 <https://github.com/mitsuba-renderer/drjit-core/pull/152>`__).

**Other Improvements**

- Fixed several corner cases in :py:func:`dr.dda.dda() <drjit.dda.dda>` (PR `#311
  <https://github.com/mitsuba-renderer/drjit/pull/311>`__).

- Added support for casting to and from boolean array types in Python. (commit `343d16
  <https://github.com/mitsuba-renderer/drjit/commit/343d16e1305d6c51fcfaaa196ce7737a35768af7>`__).

- Enhanced :py:func:`dr.expr_t() <expr_t>` to preserve custom array types when
  compatible. (commit `85d66c
  <https://github.com/mitsuba-renderer/drjit/commit/85d66c3612190a6b653fc47cd9acbf6be4350e79>`__).

- Improved :py:func:`dr.replace_grad() <replace_grad>` to handle non-differentiable and unknown
  types gracefully. (PR `#364
  <https://github.com/mitsuba-renderer/drjit/pull/364>`__).

- Improved error handling throughout the codebase by replacing ``abort()``
  calls with exceptions for better recovery in interactive environments.
  (commit `27e34c
  <https://github.com/mitsuba-renderer/drjit/commit/27e34c2170af98a08ff25826a5d49238cc5a29a2>`__).

- Added :py:func:`dr.profile_enable() <profile_enable>` context manager for
  selective CUDA profiling using the NSight tools. (commit `e4dda9
  <https://github.com/mitsuba-renderer/drjit/commit/e4dda97b53dba696db40e5a8097310d64fb385f9>`__).

- When compiling Dr.Jit with Clang/Linux, ``libstdc++`` can now also be used.
  Previously, the ``libc++`` standard library was required in this case. (PR
  `#346 <https://github.com/mitsuba-renderer/drjit/pull/346>`__).

DrJit 1.0.5 (February 3, 2025)
------------------------------

- Workaround for OptiX linking issue in driver version R570+. (commit `0c9c54e
  <https://github.com/mitsuba-renderer/drjit-core/commit/0c9c54ec5c2963dd576c5a16d10fb2d63d67166f>`__).

- Tensors can now be used as condition and state variables of
  ``dr.if_stmt/while_loop``. (commit `4691fe
  <https://github.com/mitsuba-renderer/drjit-core/commit/4691fe4421bfd7002cd9c5d998617db0f40cce35>`__).

DrJit 1.0.4 (January 28, 2025)
------------------------------

- Release was retracted

DrJit 1.0.3 (January 16, 2025)
------------------------------

- Fixes to :py:func:`drjit.wrap`. (commit `166be21 <https://github.com/mitsuba-renderer/drjit/pull/326/commits/166be21886e9fc66fe389cbc6f5becec1bfb3417>`__).

DrJit 1.0.2 (January 14, 2025)
------------------------------

- Warning about NVIDIA drivers v565+. (commit `b5fd886 <https://github.com/mitsuba-renderer/drjit-core/commit/b5fd886dcced5b7e5b229e94e2b9e702ae6aba46>`__).
- Support for boolean Python arguments in :py:func:`drjit.select`. (commit `d0c8811 <https://github.com/mitsuba-renderer/drjit/commit/d0c881187c9ec0def50ef3f6cde32dacd86a96b4>`__).
- Backend refactoring: vectorized calls are now also isolated per variant. (commit `17bc707 <https://github.com/mitsuba-renderer/drjit/commit/17bc7078918662b06c6e80c3b5f3ac1d5f9f118f>`__).
- Fixes to :cpp:func:`dr::safe_cbrt() <drjit::safe_cbrt>`. (commit `2f8a3ab <https://github.com/mitsuba-renderer/drjit/commit/2f8a3ab1acbf8e187a0ef4e248d0f65c00e27e3f>`__).

DrJit 1.0.1 (November 23, 2024)
-------------------------------

- Fixes to various edges cases of :py:func:`drjit.dda.dda` (commit `4ce97d
  <https://github.com/mitsuba-renderer/drjit/commit/4ce97dc4a5396c74887a6b123e2219e8def680d6>`__).

DrJit 1.0.0 (November 21, 2024)
-------------------------------

The 1.0 release of Dr.Jit marks a major new phase of this project. We addressed
long-standing limitations and thoroughly documented every part of Dr.Jit.
Due to the magnitude of the changes, some incompatibilities are unavoidable:
bullet points with an exclamation mark highlight changes with an impact on
source-level compatibility.

Here is what's new:

- **Python bindings**: Dr.Jit comes with an all-new set of Python bindings
  created using the `nanobind <https://github.com/wjakob/nanobind>`__ library.
  This has several consequences:

  - Tracing Dr.Jit code written in Python is now *significantly* faster (we've
    observed speedups by a factor of ~10-20×). This should help in situations
    where performance is limited by tracing rather than kernel evaluation.

  - Thorough type annotations improve static type checking and code
    completion in editors like `VS Code <https://code.visualstudio.com>`__.

  - Dr.Jit can now target Python 3.12's `stable ABI
    <https://docs.python.org/3/c-api/stable.html#stable-abi>`__. This means
    that binary wheels will work on future versions of Python without
    recompilation.

- **Natural syntax**: vectorized loops and conditionals can now be expressed
  using natural Python syntax. To see what this means, consider the following
  function that computes an integer power of a floating point array:

  .. code-block:: python

     from drjit.cuda import Int, Float

     @dr.syntax # <-- new!
     def ipow(x: Float, n: Int):
         result = Float(1)

         while n != 0:       # <-- vectorized loop ('n' is an array)
             if n & 1 != 0:  # <-- vectorized conditional
                 result *= x
             x *= x
             n >>= 1

         return result

  Given that this function processes arrays, we expect that condition of the
  ``if`` statement may disagree among elements. Also, each element may need a
  different number of loop iterations. However, such component-wise
  conditionals and loops aren't supported by normal Python. Previously, Dr.Jit
  provided ways of expressing such code using masking and a special
  ``dr.cuda.Loop`` object, but this was rather tedious.

  The new :py:func:`@drjit.syntax <drjit.syntax>` decorator greatly simplifies
  the development of programs with complex control flow. It performs an
  automatic source code transformation that replaces conditionals and loops
  with array-compatible variants (:py:func:`drjit.while_loop`,
  :py:func:`drjit.if_stmt`). The transformation leaves everything else as-is,
  including line number information that is relevant for debugging.

- **Differentiable control flow**: symbolic control flow constructs (loops)
  previously failed with an error message when they detected differentiable
  variables. In the new version of Dr.Jit, symbolic operations (loops, function
  calls, and conditionals) are now differentiable in both forward and reverse
  modes, with one exception: the reverse-mode derivative of loops is still
  incomplete and will be added in the next version of Dr.Jit.

- **Documentation**: every Dr.Jit function now comes with extensive reference
  documentation that clearly specifies its behavior and accepted inputs. The
  behavior with respect to tensors and arbitrary object graphs (referred to as
  :ref:`"PyTrees" <pytrees>`) was made consistent.

- **Half-precision arithmetic**: Dr.Jit now provides ``float16``-valued arrays
  and tensors on both the LLVM and CUDA backends (e.g.,
  :py:class:`drjit.cuda.ad.TensorXf16` or :py:class:`drjit.llvm.Float16`).

- **Mixed-precision optimization**: Dr.Jit now maintains one global AD graph
  for all variables, enabling differentiation of computation combining single-,
  double, and half precision variables. Previously, there was a separate graph
  per type, and gradients did not propagate through casts between them.

- **Multi-framework computations**: The :py:func:`@drjit.wrap <drjit.wrap>` decorator
  provides a differentiable bridge to other AD frameworks. In this new release
  of Dr.Jit, its capabilities were significantly revamped. Besides PyTorch, it
  now also supports JAX, and it consistently handles both forward and backward
  derivatives. The new interface admits functions with arbitrary
  fixed/variable-length positional and keyword arguments containing arbitrary
  PyTrees of differentiable and non-differentiable arrays, tensors, etc.

- **Debug mode**: A new debug validation mode (:py:attr:`drjit.JitFlag.Debug`)
  inserts a number of additional checks to identify sources of undefined
  behavior. Enable it to catch out-of-bounds reads, writes, and calls to
  undefined callables. Such operations will trigger a warning that includes the
  responsible source code location.

  The following built-in assertion checks are also active in debug mode. They
  support both regular and symbolic inputs in a consistent fashion.

  - :py:func:`drjit.assert_true`,
  - :py:func:`drjit.assert_false`,
  - :py:func:`drjit.assert_equal`.

- **Symbolic print statement**: A new high-level *symbolic* print operation
  :py:func:`drjit.print` enables deferred printing from any symbolic context
  (i.e., within symbolic loops, conditionals, and function calls). It is
  compatible with Jupyter notebooks and displays arbitrary :ref:`PyTrees
  <pytrees>` in a structured manner. This operation replaces the function
  ``drjit.print_async()`` provided in previous releases.

- **Swizzling**: swizzle access and assignment operator are now provided. You
  can use them to arbitrarily reorder, grow, or shrink the input array.

  .. code-block:: python

     a = Array4f(...), b = Array2f(...)
     a.xyw = a.xzy + b.xyx

- **Scatter-reductions**: the performance of atomic scatter-reductions
  (:py:func:`drjit.scatter_reduce`, :py:func:`drjit.scatter_add`) has been
  *significantly* improved. Both functions now provide a ``mode=`` parameter to
  select between different implementation strategies. The new strategy
  :py:attr:`drjit.ReduceMode.Expand` offers a speedup of *over 10×* on the LLVM
  backend compared to the previously used local reduction strategy.
  Furthermore, improved code generation for :py:attr:`drjit.ReduceMode.Local`
  brings a roughly 20-40% speedup on the CUDA backend. See the documentation
  section on :ref:`atomic reductions <reduce-local>` for details and
  benchmarks with plots.

* **Packet memory operations**: programs often gather or scatter several memory
  locations that are directly next to each other in memory. In principle, it
  should be possible to do such reads or writes more efficiently.

  Dr.Jit now features improved code generation to realize this optimization
  for calls to :py:func:`dr.gather() <gather>` and :py:func:`dr.scatter()
  <scatter>` that access a power-of-two-sized chunk of contiguous array
  elements. On the CUDA backend, this operation leverages native package memory
  instruction, which can produce small speedups on the order of ~5-30%. On the
  LLVM backend, packet loads/stores now compile to aligned packet loads/stores
  with a transpose operation that brings data into the right shape. Speedups
  here are dramatic (up to >20× for scatters, 1.5 to 2× for gathers). See the
  :py:attr:`drjit.JitFlag.PacketOps` flag for details. On the LLVM backend,
  packet scatter-addition furthermore compose with the
  :py:attr:`drjit.ReduceMode.Expand` optimization explained in the last point,
  which combines the benefits of both steps. This is particularly useful when
  computing the reverse-mode derivative of packet reads.

- **Reductions**: reduction operations previously existed as *regular* (e.g.,
  :py:func:`drjit.all`) and *nested* (e.g. ``drjit.all_nested``) variants. Both
  are now subsumed by an optional ``axis`` argument similar to how this works
  in other array programming frameworks like NumPy. Reductions can now also
  process any number of axes on both regular Dr.Jit arrays and tensors.

  The reduction functions (:py:func:`drjit.all` :py:func:`drjit.any`,
  :py:func:`drjit.sum`, :py:func:`drjit.prod`, :py:func:`drjit.min`,
  :py:func:`drjit.max`) have different default axis values depending on the
  input type. For tensors, ``axis=None`` by default and the reduction is
  performed along the entire underlying array recursively, analogous to the
  previous nested reduction. For all other types, the reduction is performed
  over the outermost axis (``axis=0``) by default.

  Aliases for the ``_nested`` function variants still exist to help porting but
  are deprecated and will be removed in a future release.

- **Prefix reductions**: the functions :py:func:`drjit.cumsum`,
  :py:func:`drjit.prefix_sum` compute inclusive or exclusive prefix sums along
  arbitrary axes of a tensor or array. They wrap for the more general
  :py:func:`drjit.prefix_reduce` that also supports other arithmetic operations
  (e.g. minimum/maximum/product/and/or reductions), reverse reductions, etc.

- **Block reductions**: the new functions :py:func:`drjit.block_reduce` and
  :py:func:`drjit.block_prefix_reduce` compute reductions within contiguous
  blocks of an array.

- **Local memory**: kernels can now allocate temporary thread-local memory and
  perform arbitrary indexed reads and writes. This is useful to implement a
  stack or other types of scratch space that might be needed by a calculation.
  See the separate documentation section about :ref:`local memory
  <local_memory>` for details.

- **DDA**: a newly added *digital differential analyzer*
  (:py:func:`drjit.dda.dda`) can be used to traverse the intersection of a ray
  segment and an n-dimensional grid. The function
  :py:func:`drjit.dda.integrate()` builds on this functionality to compute
  analytic differentiable line integrals of bi- and trilinear interpolants.

- **Loop compression**: the implementation of evaluated loops (previously
  referred to as wavefront mode) visits all entries of the loop state variables
  at every iteration, even when most of them have already finished executing the
  loop. Dr.Jit now provides an optional ``compress=True`` parameter in
  :py:func:`drjit.while_loop` to prune away inactive entries and accelerate
  later loop iterations.

- The new release has a strong focus on error resilience and leak avoidance.
  Exceptions raised in custom operations, function dispatch, symbolic loops,
  etc., should not cause failures or leaks. Both Dr.Jit and nanobind are very
  noisy if they detect that objects are still alive when the Python interpreter
  shuts down.

- **Terminology cleanup**: Dr.Jit has two main ways of capturing control flow
  (conditionals, loops, function calls): it can evaluate each possible outcome
  eagerly, causing it to launch many small kernels (this is now called:
  *evaluated mode*). The second is to capture control flow and merge it into
  the same kernel (this is now called *symbolic mode*). Previously,
  inconsistent and rendering-specific terminology was used to refer to these
  two concepts.

  Several entries of the :py:class:`drjit.JitFlag` enumeration were renamed to
  reflect this fact (for example, ``drjit.JitFlag.VCallRecord`` is now called
  :py:attr:`drjit.JitFlag.SymbolicCalls`). The former entries still exist as
  (deprecated) aliases.

- **Index reuse**: variable indices (:py:attr:`drjit.ArrayBase.index`,
  :py:attr:`drjit.ArrayBase.index_ad`) used to monotonically increase as
  variables were being created. Internally, multiple hash tables were needed to
  associate these ever-growing indices with locations in an internal variable
  array, which had a surprisingly large impact on tracing performance.
  Dr.Jit removes this mapping both at the AD and JIT levels and eagerly reuses
  variable indices.

  This change can be inconvenient for low-level debugging, where it was often
  helpful to inspect the history of operations involving a particular variable
  by searching a trace dump for mentions of its variable index. Such trace dumps
  were generated by setting :py:func:`drjit.set_log_level` to a level of
  :py:attr:`drjit.LogLevel.Debug` or even :py:attr:`drjit.LogLevel.Trace`. A
  new flag was introduced to completely disable variable reuse and help such
  debugging workflows:

  .. code-block:: python

     dr.set_flag(dr.JitFlag.ReuseIndices, False)

  Note that this causes the internal variable array to steadily grow, hence
  this feature should only be used for brief debugging sessions.

- The :py:func:`drjit.empty` function used to immediate allocate an array of
  the desired shape (compared to, say, :py:func:`drjit.zero` which creates a
  literal constant array that consumes no device memory). Users found this
  surprising, so the behavior was changed so that :py:func:`drjit.empty`
  similarly delays allocation.

- **Fast math**: Dr.Jit now has an optimization flag named
  :py:attr:`drjit.JitFlag.FastMath` that is reminiscent of ``-ffast-math`` in
  C/C++ compilers. It enables program simplifications such as ``a*0 == 0`` that
  are not always valid. For example, equality in this example breaks when ``a``
  is infinite or equal to NaN. The flag is on by default since it can
  considerably improve performance especially when targeting GPUs.


⚠️ Compatibility ⚠️
^^^^^^^^^^^^^^^^^^^

- **Symbolic loop syntax**: the old "recorded loop" syntax is no longer
  supported. Existing code will need adjustments to use
  :py:func:`drjit.while_loop`.

- **Comparison operators**: The ``==`` and ``!=`` comparisons previously
  reduced the result of to a single Python ``bool``. They now return an array
  of component-wise comparisons to be more consistent with other array
  programming frameworks. Use :py:func:`dr.all(a == b) <all>` or
  :py:func:`dr.all(a == b, axis=None) <all>` to get the previous behavior.

  The functions ``drjit.eq()`` and ``drjit.neq()`` for element-wise
  equality and inequality tests were removed, as their behavior is now subsumed
  by the builtin ``==`` and ``!=`` operators.

- **Matrix layout**: The Dr.Jit matrix type switched from column-major to
  row-major storage. Your code will need to be updated if it indexes into
  matrices first by column and then row (``matrix[col][row]``) instead of
  specifying the complete location ``matrix[row, col]``. The latter convention
  is consistent between both versions.


Internals
^^^^^^^^^

This section documents lower level changes that don't directly impact the
Python API.

- Compilation of Dr.Jit is faster and produces smaller binaries. Downstream
  projects built on top of Dr.Jit will also see improvements on both metrics.

- Dr.Jit now builds a support library (``libdrjit-extra.so``) containing large
  amounts of functionality that used to be implemented using templates. The
  disadvantage of the previous template-heavy approach was that this code ended
  up getting compiled over and over again especially when Dr.Jit was used
  within larger projects such as `Mitsuba 3 <https://mitsuba-renderer.org>`__,
  where this caused very long compilation times.

  The following features were moved into this library:

  * Transcendental functions (:py:func:`drjit.log`, :py:func:`drjit.atan2`,
    etc.) now have pre-compiled implementations for Jit arrays. Automatic
    differentiation of such operations was also moved into
    ``libdrjit-extra.so``.

  * The AD layer was rewritten to reduce the previous
    backend (``drjit/autodiff.h``) into a thin wrapper around
    functionality in ``libdrjit-extra.so``. The previous AD-related shared
    library ``libdrjit-autodiff.so`` no longer exists.

  * The template-based C++ interface to perform vectorized method calls on
    instance arrays (``drjit/vcall.h``, ``drjit/vcall_autodiff.h``,
    ``drjit/vcall_jit_reduce.h``, ``drjit/vcall_jit_record.h``) was removed and
    turned into generic implementation within the ``libdrjit-extra.so``
    library. All functionality (symbolic/evaluated model, automatic
    differentiation) is now exposed through a single statically precompiled
    function (``ad_call``). The same function is also used to realize the Python
    interface (:py:func:`drjit.switch`, :py:func:`drjit.dispatch`).

    To de-emphasize C++ *virtual* method calls (the interface is more broadly
    about calling things in parallel), the header file was renamed to
    ``drjit/call.h``. All macro uses of ``DRJIT_VCALL_*`` should be renamed to
    ``DRJIT_CALL_*``.

  * Analogous to function calls, the Python and C++ interfaces to
    symbolic/evaluated loops and conditionals are each implemented through a
    single top-level function (``ad_loop`` and ``ad_cond``) in
    ``libdrjit-extra.so``. This removes large amounts of template code and
    accelerates compilation.

- Improvements to CUDA and LLVM backends kernel launch configurations that
  more effectively use the available parallelism.

- The packet mode backend (``include/drjit/packet.h``) now includes support
  for ``aarch64`` processors via NEON intrinsics. This is actually an old
  feature from a predecessor project (Enoki) that was finally revived.

- The ``nb::set_attr()`` function that was previously used to update modified
  fields queried by a *getter* no longer exists. Dr.Jit now uses a simpler way
  to deal with getters. The technical reason that formerly required the
  presence of this function doesn't exist anymore.


Removals
^^^^^^^^

- Packet-mode virtual function call dispatch (``drjit/vcall_packet.h``)
  was removed.

- The legacy string-based IR in Dr.Jit-core has been removed.

- The ability to instantiate a differentiable array on top of a
  non-JIT-compiled type (e.g., ``dr::DiffArray<float>``) was removed. This was
  in any case too inefficient to be useful besides debugging.

Other minor technical improvements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- :py:func:`drjit.switch` and :py:func:`drjit.dispatch` now support all
  standard Python calling conventions (positional, keyword, variable length).

- There is a new C++ interface named :cpp:func:`drjit::dispatch` that works
  analogously to the Python version.

- The ``drjit.reinterpret_array_v`` function was renamed to
  :py:func:`drjit.reinterpret_array`.

- The :py:func:`drjit.llvm.PCG32.seed` function (and other backend variants)
  were modified to add the lane counter to both `initseq` and `initstate`.
  Previously, the counter was only added to the former, which led to noticeable
  correlation artifacts.
