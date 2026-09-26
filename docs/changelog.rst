.. py:currentmodule:: drjit

.. _changelog:

Changelog
#########

DrJit 1.6.0 (unreleased)
------------------------

- **Textures**: The implementation of the texture classes (e.g.,
  :py:class:`Texture2f <drjit.auto.Texture2f>`) was significantly redesigned:

  - They now support anisotropic MIP-mapped filtering in 1-3 dimensions, using
    hardware functionality or a software emulation. The operations are fully differentiable.
    See the associated :ref:`documentation section
    <texture_mipmap>` for more details.
    (commit `9b7191 <https://github.com/mitsuba-renderer/drjit/commit/9b71919be451b8db9625fd06e7ae9785dc10cea1>`__,
    Dr.Jit-Core commit `8574d4 <https://github.com/mitsuba-renderer/drjit-core/commit/8574d491f9e8657ebf19b60dd624249ab8771cc4>`__).

  - MIP-mapped textures can adopt a *Laplacian pyramid* basis following the
    paper `Practical Inverse Rendering of Textured and Translucent Appearance
    <https://doi.org/10.1145/3730855>`__ by Weier et al. This technique
    accelerates and stabilizies optimization of problems that perform
    filtered texture lookups. See the associated
    :ref:`documentation section <texture_laplacian>` for more details.
    (commit `cd0cc2 <https://github.com/mitsuba-renderer/drjit/commit/cd0cc2b343eeced20710f67a63d23e6d82a3496e>`__).

  - 2D 8-bit textures now support *block-compressed* data in the BC4,
    BC5, and BC7 formats. The CUDA and Metal backends decode it in hardware on every lookup, which reduces the memory
    footprint by a factor of 4-8 compared to plain 8-bit storage. The scalar and LLVM backends
    decode the blocks when the texture is created and do not benefit. See the
    :ref:`texture documentation <textures>` for details.
    (commit `0a3063 <https://github.com/mitsuba-renderer/drjit/commit/0a3063c362f63c0245bb1cb5821f8bc612144b2b>`__,
    Dr.Jit-Core commit `7756dc <https://github.com/mitsuba-renderer/drjit-core/commit/7756dc21f343e1549212f6dbd03ddd71432fbd31>`__).

  - The internal state machine of the texture classes was redesigned. Textures
    now migrate their data to the GPU when possible and expose their contents
    readback expression that requires no storage.
    The interaction with :py:func:`@dr.freeze
    <freeze>` was improved so that these symbolic expressions are never evaluated.
    This avoids redundant copies and roughly halves memory usage.
    (commits `df6f7f <https://github.com/mitsuba-renderer/drjit/commit/df6f7f6030303b47e083c7f24d958b4385715a21>`__,
    `92fbff <https://github.com/mitsuba-renderer/drjit/commit/92fbff0de5631e00614de46a16a31918cf2f798a>`__,
    `4dd7ea <https://github.com/mitsuba-renderer/drjit/commit/4dd7ea944513003f8635eb7ea3d2e4367f0b6584>`__,
    `64414a <https://github.com/mitsuba-renderer/drjit/commit/64414a32016a70ffe3ec8ad34af238dddd35e1b0>`__,
    `7eef93 <https://github.com/mitsuba-renderer/drjit/commit/7eef93d8aedf0663d9c71d3b941f22e7c35891c8>`__,
    `ee14bc <https://github.com/mitsuba-renderer/drjit/commit/ee14bcc04d35fcf9cf17d3f0882d886ec112d1f1>`__,
    `74152c <https://github.com/mitsuba-renderer/drjit/commit/74152cff5b2c018b60ce344131bed75c8e35c180>`__,
    `9e110e <https://github.com/mitsuba-renderer/drjit/commit/9e110e21616a19722183fa8350924c3316709467>`__,
    Dr.Jit-Core commit `0c2f1e <https://github.com/mitsuba-renderer/drjit-core/commit/0c2f1ed8751039c18adeff24b828f023fb41da38>`__).

- The frontend part of :py:func:`@dr.freeze <freeze>` frontend was redesigned.
  Launching a previously recorded function is now significantly cheaper. Diagnostics
  and error messages are more intuitive because they refer to user inputs by name. The
  :ref:`documentation <freeze>` was rewritten to be more approachable.
  (commits `628243 <https://github.com/mitsuba-renderer/drjit/commit/628243cf55df4d64cebfe8b1a76b82e30e24af7e>`__,
  `ed08c4 <https://github.com/mitsuba-renderer/drjit/commit/ed08c46d83783c65fb56dcae93dd7c22a9ec2ce5>`__,
  `e7dfc1 <https://github.com/mitsuba-renderer/drjit/commit/e7dfc1969501741f8ad341d947c1c4549f35a1dd>`__,
  `84289c <https://github.com/mitsuba-renderer/drjit/commit/84289c539d5fc392a1ab2174cf41f35ce7b0c988>`__,
  `1347be <https://github.com/mitsuba-renderer/drjit/commit/1347bed180fe8007244a37f0f878930fb81c566c>`__,
  `983c83 <https://github.com/mitsuba-renderer/drjit/commit/983c8353f5e2b1637ac9d0edc36210d416f2c6df>`__,
  `793aee <https://github.com/mitsuba-renderer/drjit/commit/793aee512d5f779a17e0aaae16878d0178cb4398>`__,
  `cb645f <https://github.com/mitsuba-renderer/drjit/commit/cb645f9f917a337443f06584273132a1a56d0763>`__,
  Dr.Jit-Core commits
  `975529 <https://github.com/mitsuba-renderer/drjit-core/commit/975529cf6ca871da8dc1271d6a63aa9c0f04a689>`__,
  `ab9956 <https://github.com/mitsuba-renderer/drjit-core/commit/ab995608f2e6357026c063e52616e43830765eec>`__).

- The backend compilation workflow in Dr.Jit-Core was redesigned. Whereas
  Dr.Jit-Core previously handed one giant source file containing all callables
  to the backend, it now generates many individual compilation units and
  compiles and caches them all in parallel. The LLVM backend switched to
  ORCv2/JITLink and caches LZ4-compressed object standard object files
  (ELF, COFF, Mach-O), which enables :ref:`low-level debugging and inspection
  <inspect_kernels>`. These features require LLVM 18 or newer.
  (Dr.Jit-Core commits
  `f222a2 <https://github.com/mitsuba-renderer/drjit-core/commit/f222a2fef8323361c67ec960d5e33e99ea489097>`__,
  `8c5ec9 <https://github.com/mitsuba-renderer/drjit-core/commit/8c5ec998eaeb10c4eaae9890a3a4a7c782d7b9ed>`__,
  `e31a79 <https://github.com/mitsuba-renderer/drjit-core/commit/e31a79e934d87941975512487f6301320b328a88>`__,
  `a5d4bb <https://github.com/mitsuba-renderer/drjit-core/commit/a5d4bbdcff1872158bed8de5bc8442dbe7ad4385>`__,
  `5190e7 <https://github.com/mitsuba-renderer/drjit-core/commit/5190e7f06e12344d627a376b2bba7a40ebc5586f>`__,
  commits `99f3e8 <https://github.com/mitsuba-renderer/drjit/commit/99f3e86e7e292a84685eeb83381d77d45d1c4494>`__,
  `0c5217 <https://github.com/mitsuba-renderer/drjit/commit/0c521716fe8f02af988cbca5272c47e56cdf58b8>`__).

- As a consequence of the previous change, kernels can now be debugged using LLDB or GDB on Linux and macOS. In debug
  mode (:py:attr:`JitFlag.Debug`), the LLVM backend emits DWARF line tables
  that map machine code to Python source lines. Native debuggers then show
  the Python line that each thread executes inside of a kernel, and
  breakpoints on Python lines resolve to the corresponding kernel code. See
  the :ref:`debugging documentation <debug_kernels>` for details.
  This feature only applies to the LLVM backend for now, but is planned for other backends in the future.
  (Dr.Jit-Core commit `a37430 <https://github.com/mitsuba-renderer/drjit-core/commit/a374309e4a80b28fd36cd94b95bf1d78b3ac343f>`__,
  commit `3442a7 <https://github.com/mitsuba-renderer/drjit/commit/3442a7dbaf79179f69b805f35257790d37a38f3b>`__).

- New kernel history benchmarking API. Measuring kernel runtimes previously
  required changing JIT flags and then extracting fields from dictionaries,
  which was awkward (untyped, no code completion, etc.):

  .. code-block:: python

     with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
         # ... code to be benchmarked ...

     time = 0
     for kernel in dr.kernel_history():
         time += kernel["execution_time"]
     print(time)

  :py:class:`drjit.kernel_history` is now a context manager that automates the
  flag adjustment. It is iterable and provides typed
  :py:class:`drjit.KernelHistoryEntry` instances with attribute access:

  .. code-block:: python

     with dr.kernel_history() as hist:
         # ... code to be benchmarked ...

     time = 0
     for kernel in hist:
         time += kernel.execution_time
     print(time)

  Directly printing the ``hist`` object renders a formatted table of the
  captured launches:

  .. code-block:: text

     Kernel history (2 entries, total device time: 29.8 µs)
     #  Type           Size  In  Out  Ops  Cache  Codegen  Compile  Execute  Hash
     -  -----------  ------  --  ---  ---  -----  -------  -------  -------  ----------------
     0  JIT          100000   0    1    6  hit      41 µs        -  10.6 µs  826339ae739b7c61
     1  BlockReduce  100000   1    1    -  -            -        -  19.2 µs  -

  See the :ref:`benchmarking documentation <bench>` for details. Code using
  the old interface continues to work but raises a ``DeprecationWarning``.
  (commit `926aad <https://github.com/mitsuba-renderer/drjit/commit/926aada92fd3e6ed686677b98c0a384ae0a15e52>`__,
  Dr.Jit-Core commit `b0b4dc <https://github.com/mitsuba-renderer/drjit-core/commit/b0b4dc89de0b07c49233a12ba9b575c094d37f77>`__).

- Many reductions now accept a ``where`` mask that excludes entries from the result.
  This covers :py:func:`dr.sum() <sum>`, :py:func:`dr.prod() <prod>`,
  :py:func:`dr.min() <min>`, :py:func:`dr.max() <max>`, :py:func:`dr.mean()
  <mean>`, :py:func:`dr.var() <var>`, :py:func:`dr.std() <std>`,
  :py:func:`dr.all() <all>`, :py:func:`dr.any() <any>`, :py:func:`dr.none()
  <none>`, :py:func:`dr.count() <count>`, :py:func:`dr.dot() <dot>`,
  :py:func:`dr.norm() <norm>`, :py:func:`dr.reduce() <reduce>`, the prefix
  scans, the block reductions, and :py:func:`dr.median() <median>`.
  (commit `429159 <https://github.com/mitsuba-renderer/drjit/commit/42915957901f0860f28dc0dda6c0a3b727207d86>`__).

- :py:func:`dr.sort() <sort>`, :py:func:`dr.argsort() <argsort>`, and
  :py:func:`dr.median() <median>` no longer run the radix sort for tensor axes
  of length 256 or less. A rank-counting sort in a single kernel now handles
  such blocks, which is hundreds of times faster when the tensor
  has many rows.
  (commit `429159 <https://github.com/mitsuba-renderer/drjit/commit/42915957901f0860f28dc0dda6c0a3b727207d86>`__).

- The CUDA implementation of :py:func:`dr.prefix_reduce() <prefix_reduce>`
  and :py:func:`dr.cumsum() <cumsum>` is now bitwise reproducible and faster.
  (commit `2514a0 <https://github.com/mitsuba-renderer/drjit/commit/2514a062d66cba2a277e50fc3b503cb46b502d8f>`__,
  Dr.Jit-Core commit `2e0c31 <https://github.com/mitsuba-renderer/drjit-core/commit/2e0c31f7fb5088dc04193622b78ddd422d060add>`__).

- New functions :py:func:`dr.expm1() <expm1>` and :py:func:`dr.log1p()
  <log1p>` that evaluate :math:`e^x-1` and :math:`\log(1+x)` without the
  cancellation that affects the naive expressions for small arguments. The
  C++ header ``drjit/math.h`` provides matching ``dr::expm1()`` and
  ``dr::log1p()`` templates. Both follow the corresponding routines of the
  CEPHES library.
  (commit `320c05 <https://github.com/mitsuba-renderer/drjit/commit/320c05d05b4439c334ed7364945ff40705a39917>`__).

- Floating point arrays now support the modulo operator ``%`` with the
  semantics of Python and NumPy, where the result has the sign of the divisor.
  (commit `45aca1 <https://github.com/mitsuba-renderer/drjit/commit/45aca1b152b2a3f692802819a42d6d904bb15991>`__).

- Added uthe color space conversion functions :py:func:`dr.rgb_to_hsv() <rgb_to_hsv>`,
  :py:func:`dr.hsv_to_rgb() <hsv_to_rgb>`, :py:func:`dr.rgb_to_hsl()
  <rgb_to_hsl>`, and :py:func:`dr.hsl_to_rgb() <hsl_to_rgb>`. The header ``drjit/color.h`` provides C++
  versions of these routines and of the Oklab conversions.
  (commits `461b36 <https://github.com/mitsuba-renderer/drjit/commit/461b3654bec7cc4d2b147d6fc9265c5ce718c103>`__,
  `8159a8 <https://github.com/mitsuba-renderer/drjit/commit/8159a87cb21c82fdae9903bcf40e11d1d0c76598>`__).

- A new function
  :py:func:`dr.transform_decompose_qr() <transform_decompose_qr>`
  complements :py:func:`dr.transform_decompose() <transform_decompose>`
  by decomposing an affine transformation into separate scale, shear,
  rotation, and translation components. Both also have C++ conterparts.
  (commit `01ab82 <https://github.com/mitsuba-renderer/drjit/commit/01ab82f33b357f0f953f5749634236a0187cf413>`__).

- Tensor slice assignment (e.g. ``t[1:3, :, 2] = value``) now broadcasts the
  right hand side to the shape of the target region and accepts arbitrary
  tensors. It previously only handled scalars and flat 1D arrays.
  (commit `0120f7 <https://github.com/mitsuba-renderer/drjit/commit/0120f7b72adbd6ab7c919415e708b9e5fe880194>`__).

- Zero-copy DLPack export on the Metal backend. The ``__dlpack__`` method now
  accepts the ``max_version``, ``dl_device``, and ``copy`` arguments of the
  DLPack protocol, and requesting ``dl_device=(8, 0)`` returns a Metal array
  without a host copy.
  (commit `eada35 <https://github.com/mitsuba-renderer/drjit/commit/eada35f582bcf2b478124ffa9144ce670d124e05>`__).

- Enumerations and other objects that implement ``__index__`` now implicitly
  convert to Dr.Jit arrays, which makes expressions like ``dr.select(mask,
  Enum.A, Enum.B)`` legal.
  (commit `de73a6 <https://github.com/mitsuba-renderer/drjit/commit/de73a6a2db3c860903899f4dbf7169d11106ea6c>`__).

- Many functions gained type signatures that were previously missing, which
  improves code completion and static type checking. The
  ``@dr.func`` decorator now preserves the signature of the
  decorated function.
  (commit `4fe61b <https://github.com/mitsuba-renderer/drjit/commit/4fe61b2e1460229af27163989be26a813c1baf0e>`__).

- The Python bindings now build against nanobind 3 and use its new `split mode
  <https://nanobind.readthedocs.io/en/latest/split_mode.html>`__, which
  significantly reduces the number of binary wheels that must be compiled for
  each release. Set the CMake option ``DRJIT_SPLIT_MODE=OFF`` to disable this
  and perform a regular static build. Array types are furthermore frozen
  after their construction, which rules out monkey-patching and speeds up
  attribute access on Python 3.15+.
  (commits `93575f <https://github.com/mitsuba-renderer/drjit/commit/93575f93c6adb45e9b3be6675d8cc117ca9e3a05>`__,
  `43b0a4 <https://github.com/mitsuba-renderer/drjit/commit/43b0a48a91c0eef10c595aa26dca1773e4b2761d>`__,
  `4262b3 <https://github.com/mitsuba-renderer/drjit/commit/4262b399cfdc12abd9f66aea6b0c786c5d47b6dd>`__,
  `821047 <https://github.com/mitsuba-renderer/drjit/commit/8210477c305f072c79c49ab18d7cf407553ddd49>`__).

- Python subclasses of C++ types deriving from ``drjit::TraversableBase`` can
  now participate in reference cycles that Python's garbage collector is
  able to collect. The C++ traversal interface was simplified along the way:
  the ``traverse_1_cb_ro()`` and ``traverse_1_cb_rw()`` callbacks were
  replaced by a single ``traverse_cb()`` method that receives a
  ``TraverseVisitor``.
  (commits `39f935 <https://github.com/mitsuba-renderer/drjit/commit/39f935c5f18cdcf2a1a9115ffe59e2dfdd39ab0f>`__,
  `c15415 <https://github.com/mitsuba-renderer/drjit/commit/c15415ff6c919b7b255d795eaccf3d5043c74554>`__,
  `cce946 <https://github.com/mitsuba-renderer/drjit/commit/cce946d6dc0a5ee0b4470ff2a8410e06707c5f5d>`__).

- Performance improvements: :py:func:`dr.sincos() <sincos>` now uses hardware
  intrinsics on the GPU backends like :py:func:`dr.sin() <sin>` and
  :py:func:`dr.cos() <cos>` already did. The gather reindexing optimization
  now also covers uniform variables, texture lookups, and packet gathers. The
  Metal backend has lower kernel launch overheads. Forward-mode
  differentiation through a sequence of packet scatters no longer launches
  one kernel per scatter. Dr.Jit constant-folds ``x/x`` in fast math mode.
  (commits `e639bd <https://github.com/mitsuba-renderer/drjit/commit/e639bda8c88056d81bb91442032444922888b232>`__,
  `5be4ff <https://github.com/mitsuba-renderer/drjit/commit/5be4ff91d156d5fe65bf3ed4bd2d3658e4ea54e5>`__,
  Dr.Jit-Core commits
  `3dd63d <https://github.com/mitsuba-renderer/drjit-core/commit/3dd63d6cddc6bbea8a6d895771f8ed6c526f53aa>`__,
  `ac975a <https://github.com/mitsuba-renderer/drjit-core/commit/ac975a833d8b3904e49a3d03fa8c81a44bb8a32a>`__,
  `79725c <https://github.com/mitsuba-renderer/drjit-core/commit/79725cf95815258f6c22048a53e87b2a29789d99>`__,
  `55a8fc <https://github.com/mitsuba-renderer/drjit-core/commit/55a8fc8593c243b34f7b8f7474bd935d0fea8f77>`__,
  `78065d <https://github.com/mitsuba-renderer/drjit-core/commit/78065d873ca6c34640c1adbc9e4cd43aaccd132d>`__).

- Ray tracing. Dr.Jit-Core can now compile custom
  shape intersection functions for the Embree, OptiX, and Metal backends,
  shadow rays report which surface ended the traversal, and the Metal and
  OptiX backends support motion transforms, a time argument, instance
  indices, and per-lane visibility masks.
  (Dr.Jit-Core commits
  `5b0817 <https://github.com/mitsuba-renderer/drjit-core/commit/5b0817cbb36606daab5e1cc1e9ffd5c3c25e257c>`__,
  `dadfd5 <https://github.com/mitsuba-renderer/drjit-core/commit/dadfd5e8940abbcddc32ab374fb3680eeb8c79f6>`__,
  `ea9f14 <https://github.com/mitsuba-renderer/drjit-core/commit/ea9f14402a4bdf1d9319931bec87429d1a879ef1>`__,
  `c7672d <https://github.com/mitsuba-renderer/drjit-core/commit/c7672de53b83ba85bb28a2b533946f8d4a48e0e7>`__,
  `75bb6c <https://github.com/mitsuba-renderer/drjit-core/commit/75bb6cbd815d8ed9680ea0c91efb09ed0c38a094>`__,
  `f2d705 <https://github.com/mitsuba-renderer/drjit-core/commit/f2d7057e120b91083fc3e51d660f44cec54550ce>`__,
  `529a4d <https://github.com/mitsuba-renderer/drjit-core/commit/529a4d9ed6c29c34fbfa57d00cb7ba84d25f467e>`__).

- Fixed a series of bugs and corner cases involving symbolic control flow and
  derivatives thereof.
  (commits `ac73bb <https://github.com/mitsuba-renderer/drjit/commit/ac73bb77efd9dfc012154155ca22d7757326d01f>`__,
  `ac4cfd <https://github.com/mitsuba-renderer/drjit/commit/ac4cfd6cdcd7f0f1866b66fe2bb03d1f7d8885fb>`__,
  `7dbb98 <https://github.com/mitsuba-renderer/drjit/commit/7dbb9811a3d4cfbb5a79471cc478eaf8793a0dcb>`__,
  `fcc66d <https://github.com/mitsuba-renderer/drjit/commit/fcc66d1345e252bd73bf251fef0c4806bbc8b973>`__,
  `51bb70 <https://github.com/mitsuba-renderer/drjit/commit/51bb70ee0606341a9b3511dc891275feed56d1a7>`__,
  `52d8f6 <https://github.com/mitsuba-renderer/drjit/commit/52d8f69063ab0be4b1e01da67f03ac50a4a3938c>`__,
  `08c61a <https://github.com/mitsuba-renderer/drjit/commit/08c61a4e95d36f010967c6b0932e8fa9a1edce68>`__,
  `e0fd0c <https://github.com/mitsuba-renderer/drjit/commit/e0fd0c7a984d2c9cabc0e61cf86a3a59fb179720>`__,
  `1c1a40 <https://github.com/mitsuba-renderer/drjit/commit/1c1a40f5d4e4b85f5c8bdf269cfbb90410a19c3d>`__,
  Dr.Jit-Core commits
  `621b05 <https://github.com/mitsuba-renderer/drjit-core/commit/621b0593751edb2fc47ab3857e44cde58d0488bc>`__,
  `c67a99 <https://github.com/mitsuba-renderer/drjit-core/commit/c67a99be0fc93ee63d057f2fc6013194b4505b2c>`__,
  `2c1c98 <https://github.com/mitsuba-renderer/drjit-core/commit/2c1c982cbf1beba355ab485979fdf15fcb04edf9>`__,
  `34433a <https://github.com/mitsuba-renderer/drjit-core/commit/34433aafa3bac3b6e70f574fbf09e80988ff5cc9>`__,
  `410611 <https://github.com/mitsuba-renderer/drjit-core/commit/410611726da9f0bd750aa10ba5088a55a1222aaa>`__).

- Fixed logging-related deadlocks in applications that use Dr.Jit from
  multiple threads.
  (commit `553c0a <https://github.com/mitsuba-renderer/drjit/commit/553c0ad12b3085cf3f2cfb59f76538c64f792f95>`__,
  Dr.Jit-Core commit `40f935 <https://github.com/mitsuba-renderer/drjit-core/commit/40f9352893fb593048601dfc863b332ad7005546>`__).

- :py:func:`dr.frob() <frob>` returns the Frobenius norm instead of its square,
  which matches the convention used by NumPy, PyTorch, MATLAB, and Eigen. This
  improves the reliability of :py:func:`dr.polar_decomp() <polar_decomp>`, which computed a
  wrong scale factor because of the previous behavior.
  (commit `45b483 <https://github.com/mitsuba-renderer/drjit/commit/45b48378be25516ef7507675e4f09d7d433dd008>`__).

- :py:func:`dr.binary_search() <binary_search>` accepts Dr.Jit arrays as
  search bounds.
  (commit `5d2abc <https://github.com/mitsuba-renderer/drjit/commit/5d2abcee91255d6ffd56a75d96e04f135447a69b>`__,
  contributed by `Matteo Santini <https://github.com/matttsss>`__).

- Python 3.9 is no longer supported. Dr.Jit now requires Python 3.10 or newer.

- Miscellaneous Dr.Jit-Core fixes and improvements.
  (Dr.Jit-Core commits
  `b9a65b <https://github.com/mitsuba-renderer/drjit-core/commit/b9a65b2811b9cc34af0a1a5ae9cffccedd3be58b>`__,
  `c2d28e <https://github.com/mitsuba-renderer/drjit-core/commit/c2d28e4fefb1be4397bff63b584bf6704b521c4b>`__,
  `2f53eb <https://github.com/mitsuba-renderer/drjit-core/commit/2f53ebb97041ff2c30d25e33baf92ee322b1e340>`__,
  `7f3943 <https://github.com/mitsuba-renderer/drjit-core/commit/7f3943605b5772bb6c10e5166289212b77c0deb5>`__,
  `33becf <https://github.com/mitsuba-renderer/drjit-core/commit/33becf972a63fd12843c0659d82f0f20f616b030>`__,
  `0d2c96 <https://github.com/mitsuba-renderer/drjit-core/commit/0d2c967d7739968db746ed51305aab2cb675385d>`__,
  `092858 <https://github.com/mitsuba-renderer/drjit-core/commit/0928581aecff595ae78e65b3a71c92db3a7fe194>`__,
  `5da22a <https://github.com/mitsuba-renderer/drjit-core/commit/5da22ae1d76b1b7bd0dc31a9dba8c4c33bf97aca>`__,
  `b26ffb <https://github.com/mitsuba-renderer/drjit-core/commit/b26ffb0736f2b35d9044eec584263d4513ddcc94>`__,
  `e6677f <https://github.com/mitsuba-renderer/drjit-core/commit/e6677faacaa8f7f143e0b3a482782fdcd7b789fe>`__,
  `f5b3f7 <https://github.com/mitsuba-renderer/drjit-core/commit/f5b3f7698723f5e9857991d3615606875ad50902>`__,
  `01645c <https://github.com/mitsuba-renderer/drjit-core/commit/01645cad65a2eb865d2f9c421734b58d93948b7a>`__,
  `dd77a3 <https://github.com/mitsuba-renderer/drjit-core/commit/dd77a33fdb7dc67a2450dbbfcc1f7bf54829ab35>`__,
  `7afd90 <https://github.com/mitsuba-renderer/drjit-core/commit/7afd9080e939b94e1bc0443c2d1d3a39a47e89a7>`__,
  `be9a04 <https://github.com/mitsuba-renderer/drjit-core/commit/be9a04d4fc75f8d03bbc507b6ce921b8e907c066>`__,
  `b25443 <https://github.com/mitsuba-renderer/drjit-core/commit/b25443792120f0fdd7b84bafb740e3db841be8ce>`__,
  `959b4f <https://github.com/mitsuba-renderer/drjit-core/commit/959b4f525081159ef5350846fe6faac7ba9d0f7f>`__,
  `15b269 <https://github.com/mitsuba-renderer/drjit-core/commit/15b26958a8dc5a81f93b5cbd474c71535b3738f6>`__,
  `ab56aa <https://github.com/mitsuba-renderer/drjit-core/commit/ab56aaa279c23114306c879a04a456db4a2a9212>`__,
  `3f7b10 <https://github.com/mitsuba-renderer/drjit-core/commit/3f7b101751238f61a931709fb5728397b964eac4>`__,
  `d4f187 <https://github.com/mitsuba-renderer/drjit-core/commit/d4f187002d8714d73b157f269874f2823699d6b1>`__,
  `b19a59 <https://github.com/mitsuba-renderer/drjit-core/commit/b19a591bd51b44cf3082af764f3cf5f545bf9bbe>`__).

- Miscellaneous minor fixes in Dr.Jit.
  (commits `f8fe71 <https://github.com/mitsuba-renderer/drjit/commit/f8fe719562d0ecffe66bbe5afff8a37caf87f22e>`__,
  `bbde99 <https://github.com/mitsuba-renderer/drjit/commit/bbde99ec989ab00b718953099f568cb195d53d0e>`__,
  `d7b635 <https://github.com/mitsuba-renderer/drjit/commit/d7b6352565cf3ff9c8813e29ced4dbaacb361fcb>`__,
  `8f7b57 <https://github.com/mitsuba-renderer/drjit/commit/8f7b5757fa232e583f7e56610904511a78f4cd4a>`__,
  `c41610 <https://github.com/mitsuba-renderer/drjit/commit/c416104360f7076bc14cd5f166cb1f4a536790eb>`__,
  `3ba418 <https://github.com/mitsuba-renderer/drjit/commit/3ba4180d7fcc2dedcda706f4ce06000cd2182381>`__,
  `2f6ea5 <https://github.com/mitsuba-renderer/drjit/commit/2f6ea54bd7689105f65965371efcb4ac930e5a0d>`__,
  `a25997 <https://github.com/mitsuba-renderer/drjit/commit/a259973b2eadde4237607f778064b4c160a4974b>`__,
  `dfccdd <https://github.com/mitsuba-renderer/drjit/commit/dfccdd7967547efce49fb4a929c39e4474ea73e2>`__,
  `0a40ca <https://github.com/mitsuba-renderer/drjit/commit/0a40ca53e4c18e1b7757184808eb643e79a332e5>`__,
  `1b8fae <https://github.com/mitsuba-renderer/drjit/commit/1b8fae0c74f696549859f339c27f72f35d840199>`__,
  `ef545c <https://github.com/mitsuba-renderer/drjit/commit/ef545c25a54b0683d0ecf3ea57084a7aad5dc17a>`__,
  `4e2be8 <https://github.com/mitsuba-renderer/drjit/commit/4e2be87c97cd26d5a25f884dd2227aa94a49fa63>`__,
  `77fd69 <https://github.com/mitsuba-renderer/drjit/commit/77fd69c7c1c26135380994aa944f91cf48264029>`__,
  `86902f <https://github.com/mitsuba-renderer/drjit/commit/86902f9faf81d430639a60c83108845b4fe8b8b2>`__,
  `39f7fc <https://github.com/mitsuba-renderer/drjit/commit/39f7fca225c7e676f9278efe39239da441a2472d>`__,
  `315af5 <https://github.com/mitsuba-renderer/drjit/commit/315af5ccd7a197f9b99a0b75cfdc82c4e10cd42c>`__,
  `e3979c <https://github.com/mitsuba-renderer/drjit/commit/e3979c81e40125c8116658531e9ad3112a53e84b>`__,
  PRs `#519 <https://github.com/mitsuba-renderer/drjit/pull/519>`__,
  `#525 <https://github.com/mitsuba-renderer/drjit/pull/525>`__,
  `#527 <https://github.com/mitsuba-renderer/drjit/pull/527>`__,
  `#530 <https://github.com/mitsuba-renderer/drjit/pull/530>`__,
  `#531 <https://github.com/mitsuba-renderer/drjit/pull/531>`__).

DrJit 1.5.0 (August 7, 2026)
----------------------------

- Added :py:func:`dr.median() <median>`, which computes the median along one
  or more axes.
  (commit `292dac <https://github.com/mitsuba-renderer/drjit/commit/292dac1ec478cae3c0d6a9385b9ab0cbcdd1b67b>`__).

- :py:func:`dr.minimum() <minimum>` and :py:func:`dr.maximum() <maximum>` now
  consistently propagate NaNs, while the new functions :py:func:`dr.fmin()
  <fmin>` and :py:func:`dr.fmax() <fmax>` suppress them. This is consistent
  with other frameworks (e.g., NumPy/PyTorch). The operations coincide for
  integers.
  (commit `9e0a01 <https://github.com/mitsuba-renderer/drjit/commit/9e0a012bda738514bffff9203745068ee87095b4>`__,
  Dr.Jit-Core commits
  `efdfc1 <https://github.com/mitsuba-renderer/drjit-core/commit/efdfc15de105ce6710f3e1a0d8951e926078dc6b>`__,
  `807937 <https://github.com/mitsuba-renderer/drjit-core/commit/8079374f637554d4f7ff5d5ac1023e040bc7512a>`__).

- :py:func:`dr.count() <count>` now works on nested arrays and reduces along
  user-specified axes.
  (commit `508f9b <https://github.com/mitsuba-renderer/drjit/commit/508f9bab33743f0b3e72cf5c2e6238a78b131038>`__).

- The expression :py:func:`dr.opaque(value) <opaque>` can now be used to make an opaque deep
  copy of a PyTree argument. (commit `10093d <https://github.com/mitsuba-renderer/drjit/commit/10093dad3849cc14bcf5a047142e28691f6cbe63>`__).

- :py:func:`dr.scatter_reduce() <scatter_reduce>` now consistently supports
  :py:attr:`ReduceOp.Min` and :py:attr:`ReduceOp.Max` reductions of floating
  point arrays. The Metal and CUDA backend emulate them using integer min/max
  atomics. The LLVM backend uses a CAS loop with a non-atomic load that
  potentially skips the loop if it would not change the result.
  (commit `4d5103 <https://github.com/mitsuba-renderer/drjit/commit/4d5103a8491286daaf3b92a7e8b9ddf0b3cb58cf>`__,
  Dr.Jit-Core commit `241a64 <https://github.com/mitsuba-renderer/drjit-core/commit/241a6429231ae82b1a40863a16afdd2af6fb2630>`__).

- The on-disk kernel cache in ``~/.drjit`` previously grew without bound.
  Dr.Jit now evicts least recently used entries. The cache directory,
  its size limit, and its verbosity can be configured through environment
  variables. Cache files also shrank to roughly a quarter of their former
  size. See the :ref:`cache configuration <cache_config>` documentation for
  details. (Dr.Jit-Core commit
  `10436e <https://github.com/mitsuba-renderer/drjit-core/commit/10436e7eb958457da636ba9b67c0b76d6d7acb5e>`__).

- :py:func:`dr.copysign() <copysign>` now maps onto a dedicated backend IR node
  on the CUDA, LLVM, and Metal backends.
  (commit `5ac8a8 <https://github.com/mitsuba-renderer/drjit/commit/5ac8a855023d6e90492ac62e63d8edfe197fb669>`__,
  Dr.Jit-Core commit `14d599 <https://github.com/mitsuba-renderer/drjit-core/commit/14d599a2dafac4324f0da2b1ba60fdc7412e480c>`__).

- :py:func:`dr.clip() <clip>` is now defined as ``minimum(maximum(value, min),
  max)`` instead of ``maximum(minimum(value, max), min)`` for consistency with
  NumPy/PyTorch. This is only relevant when the interval is inverted (``min >
  max``). Following the change above, the operation now also propagates NaNs.
  (commit `83a0b1 <https://github.com/mitsuba-renderer/drjit/commit/83a0b1eb6233cde57ba302169e17ef2162d504ba>`__).

- :py:func:`dr.stack() <stack>`, :py:func:`dr.vstack() <vstack>`,
  :py:func:`dr.hstack() <hstack>`, :py:func:`dr.column_stack()
  <column_stack>`, and :py:func:`dr.dstack() <dstack>` now promote
  non-tensor types to tensors.
  (commit `3735e1 <https://github.com/mitsuba-renderer/drjit/commit/3735e172b8f7c84d6dc28e16b474d081f1f57127>`__).

- Constructing a nested array from a 1D dynamic Dr.Jit array (e.g., ``Array3f(Float(1,
  2, 3))``) now consistently broadcasts (commit `b804db <https://github.com/mitsuba-renderer/drjit/commit/b804dbf476924b541a84d8d7c46eb231abd7719c>`__).

- A packet scatter that decomposes into individual scatters no longer
  serializes into separate kernels.
  (commit `b128b6 <https://github.com/mitsuba-renderer/drjit/commit/b128b6ab4dbdae23430df2b5dbc29ab05a0c1613>`__).

- :py:func:`dr.unravel() <unravel>` now casts its input to the flat type
  implied by ``dtype`` instead of rejecting a mismatch, which makes
  expressions like ``Array3f64(TensorXf(...))`` legal.
  (commit `b7db80 <https://github.com/mitsuba-renderer/drjit/commit/b7db80dff36ddbd5d52fbf113bdcf3fe668a30b8>`__).

- Fixed several bugs and changed defaults in :py:func:`dr.convolve() <convolve>` and
  :py:func:`dr.resample() <resample>`:

  - Convolutions with custom continuous filters did not correctly evaluate the
    filter at all integer offsets within ``[-filter_radius, filter_radius]``.
    Filter presets like ``box`` or ``gaussian`` were not affected.
    (commit `232932 <https://github.com/mitsuba-renderer/drjit/commit/23293212a7a5a76627924da8f7a9974021906c3d>`__).

  - A rounding issue could lead to incorrect output for the ``"nearest"``,
    ``"wrap"``, ``"reflect"``, and ``"mirror"`` boundaries. The default
    ``"zero"`` boundary condition was unaffected.
    (commit `232932 <https://github.com/mitsuba-renderer/drjit/commit/23293212a7a5a76627924da8f7a9974021906c3d>`__).

  - :py:func:`dr.convolve() <convolve>` no longer normalizes the filter weights
    by default. With the default arguments, it is now equivalent to
    ``numpy.convolve(..., mode='same')``. Specify ``normalize=True`` to restore
    the previous behavior.
    (commit `715c2d <https://github.com/mitsuba-renderer/drjit/commit/715c2df7f48b932d429c236d0527efdf61554541>`__).

  - Periodic boundary conditions (``"wrap"``, ``"reflect"``, and ``"mirror"``)
    extended the array by a single period. Larger filters could trigger
    undefined behavior by reading beyond the end of the array. The extension
    now repeats as often as needed.
    (commit `6b5f56 <https://github.com/mitsuba-renderer/drjit/commit/6b5f5697e22523bd574c811264ab8de67f5c654f>`__).

  - Other minor fixes. (commits `dbb052 <https://github.com/mitsuba-renderer/drjit/commit/dbb052779d3d2964f93302d63ce855b231b50bd8>`__,
    `1f342a <https://github.com/mitsuba-renderer/drjit/commit/1f342a185b3fd8186d455c8bd7269648ee0c364b>`__).

- Fixed the behavior of :py:func:`dr.dot() <dot>` and ``__rsub__`` on
  cooperative vectors.
  (PRs `#521 <https://github.com/mitsuba-renderer/drjit/pull/521>`__,
  `#529 <https://github.com/mitsuba-renderer/drjit/pull/529>`__,
  contributed by `Lovro Nuic <https://github.com/lnuic>`__).

- Fixed :py:func:`dr.cumsum() <cumsum>` over a tuple of axes, which previously only
  applied a single axis.
  (PR `#522 <https://github.com/mitsuba-renderer/drjit/pull/522>`__,
  contributed by `Lovro Nuic <https://github.com/lnuic>`__).

- Slicing the outer dimension of a nested array (e.g. ``value[3:]``) produced a
  result with the wrong size.
  (PR `#523 <https://github.com/mitsuba-renderer/drjit/pull/523>`__,
  contributed by `Delio Vicini <https://github.com/dvicini>`__).

- Fixed a miscompilation in symbolic ``if`` statements with partially evaluated state.
  (commit `9a7db9 <https://github.com/mitsuba-renderer/drjit/commit/9a7db92b07d162950d2285a44a921477bf819477>`__).

- Fixed a race condition in ``dr.sync_thread()``.
  (Dr.Jit-Core commit `c5a8a9 <https://github.com/mitsuba-renderer/drjit-core/commit/c5a8a95d121e9555bed6168c7254a553e000e2d3>`__).

- Miscellaneous Jit-Core fixes.
  (Dr.Jit-Core commits
  `b1b839 <https://github.com/mitsuba-renderer/drjit-core/commit/b1b83917046e055d93d266b113461a632ba51ab3>`__,
  `26d4a1 <https://github.com/mitsuba-renderer/drjit-core/commit/26d4a14041930632b832f5aef59ed71fcddee61c>`__,
  `546967 <https://github.com/mitsuba-renderer/drjit-core/commit/546967a165632f9e7fe82055f9e36c3795c43eb9>`__,
  `ac46d8 <https://github.com/mitsuba-renderer/drjit-core/commit/ac46d85458e296918a50073a12b636cf7576e787>`__,
  `8eeccd <https://github.com/mitsuba-renderer/drjit-core/commit/8eeccd098ac2bfa159df65f41060ab318bb17124>`__).

- Miscellaneous minor fixes in Dr.Jit.
  (commits `abfa1c <https://github.com/mitsuba-renderer/drjit/commit/abfa1c72cb40266581a84f1faccf1d9537b13a7e>`__,
  `b547d8 <https://github.com/mitsuba-renderer/drjit/commit/b547d867037685b56f93b344af755f0c3eb89df5>`__,
  `564a37 <https://github.com/mitsuba-renderer/drjit/commit/564a37c8cf405ad72f0d1d7d16e3f5cf5f6f4075>`__,
  `479252 <https://github.com/mitsuba-renderer/drjit/commit/4792522871cb0e6143af7be815c0176fa95c3c3d>`__,
  `c478bb <https://github.com/mitsuba-renderer/drjit/commit/c478bb4ec56d477a4dac43fd1797a80b80952ed1>`__,
  `02e2b4 <https://github.com/mitsuba-renderer/drjit/commit/02e2b4c504e59db6daa64077b218a0a00ca9b3d5>`__).

DrJit 1.4.0 (June 25, 2026)
---------------------------

**Major new Features**

- **Metal Backend**: Dr.Jit can now target Apple Silicon GPUs through a new
  Metal backend. It supports the full range of Dr.Jit features including
  symbolic control flow, automatic differentiation, hardware-accelerated
  ray tracing and textures, :ref:`cooperative vectors <coop_vec>`, and
  reductions. (contributed by `Sébastien Speierer
  <https://github.com/Speierers>`__ and `Wenzel Jakob
  <https://github.com/wjakob>`__).

- **Matrix Multiplication for Tensors**: The ``@`` operator and
  :py:func:`dr.matmul() <matmul>` now support tensors of any size and shape,
  fully replicating NumPy / PyTorch semantics including batched matrix
  products, broadcasting, matrix-vector products, and inner products. The
  operation is fully differentiable in both forward and reverse modes. Under
  the hood, this dispatches to efficient :ref:`block-tiled GEMM <matmul_perf>`
  kernels shipped with Dr.Jit-Core.
  (Dr.Jit commit `183dc4 <https://github.com/mitsuba-renderer/drjit/commit/183dc401a355c3190256c7948345befc2d2df41a>`__,
  Dr.Jit-Core PR `#188 <https://github.com/mitsuba-renderer/drjit-core/pull/188>`__,
  Dr.Jit-Core commits
  `0cca8d <https://github.com/mitsuba-renderer/drjit-core/commit/0cca8de7f2f74d9e8782788f221e830cdb94bc22>`__,
  `432ed4 <https://github.com/mitsuba-renderer/drjit-core/commit/432ed4a5c4c082941a00ecf72dde186c676d5555>`__,
  `444c8d <https://github.com/mitsuba-renderer/drjit-core/commit/444c8df706e034ec8ca7812f20b79432671e72f0>`__,
  `4b8864 <https://github.com/mitsuba-renderer/drjit-core/commit/4b88649668c2a91f257e91b9ef4eb9ec7a2947b1>`__,
  `9e5335 <https://github.com/mitsuba-renderer/drjit-core/commit/9e533522035a4c00950553d8c0677b92d780f3b0>`__).

- **Generalized convolution and resampling**: The function
  :py:func:`dr.convolve() <convolve>` now handles discrete filter kernels
  besides continuous ones, making it a Dr.Jit substitute for
  :py:func:`numpy.convolve`. A new ``boundary`` parameter generalizes edge
  handling (``"zero"``, ``"nearest"``, ``"wrap"``, ``"reflect"``, or
  ``"mirror"``). A ``normalize`` flag toggles the renormalization of filter
  weights. The efficiency of both the forward pass and reverse-mode derivative
  was improved via a fast path for non-boundary outputs, and by switching to a
  fast transpose convolution instead of atomic scatters whenever possible. The
  new ``boundary`` argument is also available on :py:func:`dr.resample()
  <resample>`.
  (Dr.Jit commits
  `00b40a <https://github.com/mitsuba-renderer/drjit/commit/00b40a6e873abf2031ed7e11fc19f505b96ec383>`__,
  `d96ba5 <https://github.com/mitsuba-renderer/drjit/commit/d96ba5d933cb805e51f5599b4af020ad5b34250d>`__).

- **Transpose**: Added :py:attr:`dr.ArrayBase.T <ArrayBase.T>` and
  :py:attr:`dr.ArrayBase.mT <ArrayBase.mT>`, matching PyTorch's semantics. (PR
  `#486 <https://github.com/mitsuba-renderer/drjit/pull/486>`__).

- **Muon Optimizer**: Added :py:class:`dr.opt.Muon <opt.Muon>` ("MomentUm
  Orthogonalized by Newton-schulz"), an optimizer for 2D hidden weights of
  neural networks.
  (commit `d205c1 <https://github.com/mitsuba-renderer/drjit/commit/d205c1d4dd57870a54eff0875c2e336a99191317>`__).

- **Redesign of the** :py:mod:`drjit.nn` **API**. Besides
  :ref:`cooperative vectors <coop_vec>`, the :py:mod:`drjit.nn` API now also
  accepts regular tensors as inputs.
  Cooperative vectors fuse with surrounding computation, while tensor
  evaluation enables batched evaluation of large networks.
  See the :ref:`neural network
  documentation <neural_nets>` for details on both modes.

  Previously, it was necessary to extract the packed buffer copy and manually
  cast it between the working and optimizer precision.

  .. code-block:: python

     weights, net = nn.pack(net, layout='training')
     opt = Adam(lr=1e-3, params={'weights': Float32(weights)})

     for i in range(n):
         weights[:] = Float16(opt['weights'])
         ...

  The new API exposes a cleaner interface that automates all of these steps:

  .. code-block:: python

     net = nn.pack(net, layout='training')
     opt = Adam(lr=1e-3)
     opt.update(net)

     for i in range(n):
         net.update(opt)
         ...

  :py:class:`nn.Module <drjit.nn.Module>` subclasses implement a
  :py:class:`MutableMapping <collections.abc.MutableMapping>` keyed by the path
  to each parameter (e.g. ``'layers.0.weights'``). ``opt.update(net)`` pulls
  the parameters into the optimizer, while ``net.update(opt)`` pushes the
  updated state back.
  The :py:func:`nn.pack() <drjit.nn.pack>` function is now differentiable. This enables
  the use of Cooperative Vectors with matrix-level optimizers
  like :py:class:`Muon <drjit.opt.Muon>`.
  (PR `#490 <https://github.com/mitsuba-renderer/drjit/pull/490>`__).

- **Reverse-mode differentiation of symbolic loops**:
  :py:func:`@dr.syntax <syntax>` ``while`` loops and symbolic
  :py:func:`dr.while_loop() <while_loop>` calls are now differentiable in
  reverse mode via trajectory replay. See the
  :ref:`documentation <diff_loops>` for details.
  (PR `#491 <https://github.com/mitsuba-renderer/drjit/pull/491>`__).

- **NumPy-style advanced tensor indexing**: Tensor indexing with multiple
  integer arrays now follows NumPy/PyTorch semantics.
  (PR `#460 <https://github.com/mitsuba-renderer/drjit/pull/460>`__).

- **NumPy-style array/tensor manipulation and sorting**: This release brings a
  large set NumPy-compatible functions for sorting, reshaping, and reindexing
  arrays and tensors. This includes :py:func:`dr.sort() <sort>`,
  :py:func:`dr.argsort() <argsort>`, :py:func:`dr.argmin() <argmin>` and
  :py:func:`dr.argmax() <argmax>` which are backed by an efficient
  GPU-accelerated multi-bit radix sort. Other new shape manipulation functions
  include :py:func:`dr.expand_dims() <expand_dims>`, :py:func:`dr.squeeze()
  <squeeze>`, :py:func:`dr.transpose() <transpose>`, and
  :py:func:`dr.swapaxes() <swapaxes>`. (PR `#496
  <https://github.com/mitsuba-renderer/drjit/pull/496>`__).

- **NumPy-consistent reductions**: The horizontal reductions
  (:py:func:`dr.sum() <sum>`, :py:func:`dr.prod() <prod>`,
  :py:func:`dr.min() <min>`, :py:func:`dr.max() <max>`,
  :py:func:`dr.mean() <mean>`, :py:func:`dr.all() <all>`,
  :py:func:`dr.any() <any>`, :py:func:`dr.none() <none>`,
  :py:func:`dr.count() <count>`, :py:func:`dr.reduce() <reduce>`,
  :py:func:`dr.norm() <norm>`, :py:func:`dr.squared_norm() <squared_norm>`) now
  mirror NumPy more closely by accepting a ``keepdims`` flag, with full tensor
  support.  :py:func:`dr.norm() <norm>` and
  :py:func:`dr.squared_norm() <squared_norm>` additionally gain the ``axis`` and
  ``mode`` parameters shared by the rest of the family. Finally, this release
  adds NumPy-compatible :py:func:`dr.var() <var>` and :py:func:`dr.std() <std>`
  functions.
  (PR `#493 <https://github.com/mitsuba-renderer/drjit/pull/493>`__).

- **Test assertions**: Added :py:func:`dr.assert_allclose() <assert_allclose>`,
  an assertion utility for correctness checks in test cases that complements
  :py:func:`dr.allclose() <allclose>`.
  (PR `#489 <https://github.com/mitsuba-renderer/drjit/pull/489>`__).

**Performance Improvements**

- **Tracing and evaluation**:
  A comprehensive optimization pass targeted Dr.Jit's tracing/code generation
  phases and Python bindings, making them roughly **twice as fast**. This will
  help workloads bottlenecked on tracing/Python-related overheads.
  (Dr.Jit commits
  `534829 <https://github.com/mitsuba-renderer/drjit/commit/534829d88af9f434b0f2da9a798732ade7256e88>`__,
  `3fba39 <https://github.com/mitsuba-renderer/drjit/commit/3fba39d2595121fae88d59f4f47b8dd6e9a000aa>`__,
  `6b212c <https://github.com/mitsuba-renderer/drjit/commit/6b212c235004edfad964665ade3e6f3ec9af6ecb>`__,
  `50986a <https://github.com/mitsuba-renderer/drjit/commit/50986a050625dd88d6ec9b5ab29caaade2cf7027>`__,
  Dr.Jit-Core PR `#194 <https://github.com/mitsuba-renderer/drjit-core/pull/194>`__).

- **Frozen function replay**: The :py:func:`@dr.freeze
  <freeze>` replay path was thoroughly optimized, accelerating it by up to ~2.5x.
  (Dr.Jit commits
  `ff09ee <https://github.com/mitsuba-renderer/drjit/commit/ff09ee9e6de02d02cefcbc103a917ef02febf998>`__,
  `c1282c <https://github.com/mitsuba-renderer/drjit/commit/c1282ca81a14145095f8be16ccd632d6fc7a5a8c>`__,
  `13fe80 <https://github.com/mitsuba-renderer/drjit/commit/13fe80ed2142098179b32c127d0dde7eaba0a506>`__).

- **Faster function calls**: Dr.Jit now generates much better
  code for indirect function calls in kernels (e.g., method calls on arrays of
  object instances, :py:func:`dr.switch() <switch>`, and
  :py:func:`dr.dispatch() <dispatch>`). The
  per-instance data of all callables is now merged into a single per-kernel
  buffer and fetched using vectorized packet loads, rather than being scattered
  across many small buffers and read element by element. On the LLVM backend,
  call inputs and outputs are additionally passed in registers rather than
  stack scratch space, which reduces memory traffic and
  improves performance. Dr.Jit also uses more efficient data structures to
  collect call data, which speeds up the compilation of kernels that dispatch to
  a large number of instances.
  (Dr.Jit-Core commits
  `1ed505 <https://github.com/mitsuba-renderer/drjit-core/commit/1ed505b2ee4a1a9eb98599cc08dd31927f017d4d>`__,
  `bc6d9c <https://github.com/mitsuba-renderer/drjit-core/commit/bc6d9cacb76d83c9725787e19af7ec14d510d972>`__,
  `69120f <https://github.com/mitsuba-renderer/drjit-core/commit/69120ffef47b2cf2b05d74ffbea4c320c833c00e>`__,
  `83207d <https://github.com/mitsuba-renderer/drjit-core/commit/83207d5aeeb8fab27473c606b6a71349bce4157c>`__).

- **LLVM code generation**: Load/store aliasing metadata was improved so that
  non-conflicting memory operations within a kernel can be freely reordered
  or hoisted out of loops, which improves performance of kernels on the LLVM backend.
  (Dr.Jit-Core commit `84c85b <https://github.com/mitsuba-renderer/drjit-core/commit/84c85bd9d07a2de88a73a23dd0bf0baad53df104>`__).

- **Warp-reduction for packet scatter-reduce**: On the CUDA and Metal backends,
  :py:func:`dr.scatter_reduce() <scatter_reduce>` now provides a *packet-aware*
  reduction path that jointly reduces values within the warp/simdgroup before
  issuing scalar or packet atomics depending on hardware/driver support.
  (Dr.Jit-Core PR `#190
  <https://github.com/mitsuba-renderer/drjit-core/pull/190>`__).

- **nanobind optimizations**: Dr.Jit benefits from optimizations introduced in
  `nanobind v2.13
  <https://nanobind.readthedocs.io/en/latest/changelog.html#version-2-13-0-jun-18-2026>`__.
  This release adds *instance pooling*, which provides a cache to recycle
  short-lived objects. Dr.Jit opts into this feature to accelerate tracing,
  which generates large amounts of temporaries. Other optimizations target
  object creation/destruction and nd-array exchange. (Dr.Jit commit `6b212c <https://github.com/mitsuba-renderer/drjit/commit/6b212c235004edfad964665ade3e6f3ec9af6ecb>`__,
  nanobind PRs `#1366 <https://github.com/wjakob/nanobind/pull/1366>`__, `#1374
  <https://github.com/wjakob/nanobind/pull/1374>`__, `#1375
  <https://github.com/wjakob/nanobind/pull/1375>`__).

- **nanothread optimizations**: The thread pool driving parallel evaluation
  was improved:

  - **Faster worker wake-up**: idle worker threads busy-poll for a short while
    and then go to sleep to avoid wasting power. The new version of
    nanothread is more careful to wake only the required number of threads,
    and it does so using efficient OS primitives, such as
    `futex <https://en.wikipedia.org/wiki/Futex>`__ on Linux (commits
    `73efa1 <https://github.com/mitsuba-renderer/nanothread/commit/73efa1367ddd49aa2026b245c6857231eefbb344>`__,
    `366774 <https://github.com/mitsuba-renderer/nanothread/commit/366774d9f92d62bba3b3c7e53503a290e42315b0>`__).

  - **Worker count**: the main thread now "counts" as a member
    of the thread pool, since it pitches in when waiting for work.
    On Apple silicon, workers now only run on "performance cores", as
    parallelization over "efficiency cores" tends to add tail latency
    that slows down parallel workloads.
    (commits `03cacd <https://github.com/mitsuba-renderer/nanothread/commit/03cacd084c58675bb468b08798ad5f0f11dd0608>`__,
    `348404 <https://github.com/mitsuba-renderer/nanothread/commit/3484048050507b2c3813157b4250b85380b6df96>`__.
    `e68a4d <https://github.com/mitsuba-renderer/nanothread/commit/e68a4d827f07fa7620ae3ae235cd43fd8df725f1>`__,
    `098925 <https://github.com/mitsuba-renderer/nanothread/commit/09892587087c27b46db4891fa05bf3a9774ac8c9>`__,
    `beca8c <https://github.com/mitsuba-renderer/nanothread/commit/beca8c6635d458a2db027a7ecc0659a6e32134f3>`__).

  - **Fixed timing glitches**: timing information reported by
    :py:class:`dr.kernel_history() <kernel_history>` would occasionally
    report nonsensical values close to ``2^64`` due to a race condition
    that is now fixed.
    (`f11692 <https://github.com/mitsuba-renderer/nanothread/commit/f1169296bb4af6ee1e553e3b331be8ec4275e399>`__).

**Minor features**

- **CUDA Green Context API**: Added :py:class:`drjit.cuda.green_context`, a
  context manager that isolates kernels to a subset of the GPU's streaming
  multiprocessors. See the :ref:`green context documentation <green_context>`
  for details.
  (Dr.Jit commit `6c69ec <https://github.com/mitsuba-renderer/drjit/commit/6c69ecb75cfc605063502747e9c9264bc739ead9>`__,
  Dr.Jit-Core commit `d4f1a6 <https://github.com/mitsuba-renderer/drjit-core/commit/d4f1a62c6b175af295e857069b1401c36bcf6caa>`__).

- **Command queue flushing**: The new :py:func:`dr.flush_thread()
  <flush_thread>` function flushes queued work to the GPU, which is needed for
  multi-threaded use of Dr.Jit on the Metal backend. (Dr.Jit commit `c68e00 <https://github.com/mitsuba-renderer/drjit/commit/c68e00853de7957912e490245c2036196fe422ff>`__,
  Dr.Jit-Core commit `467dd3 <https://github.com/mitsuba-renderer/drjit-core/commit/467dd3d23ed23129139dcdf557baead32b683e01>`__).

**Bug Fixes**

- Fixed a bug in :py:meth:`dr.rng().integers() <random.Generator.integers>`
  where a symbolic loop was misused, producing invalid LLVM IR.
  (commit `f7054e <https://github.com/mitsuba-renderer/drjit/commit/f7054e1b82d7b930e08fd3bb4a8f091543160f18>`__).

- Fixed a variable shadowing bug in ``_flatten``/``_unflatten`` that caused
  crashes when flattening PyTrees containing custom ``DRJIT_STRUCT`` types.
  (PR `#482 <https://github.com/mitsuba-renderer/drjit/pull/482>`__).

- Fixed a bug in :py:class:`nn.SinEncode <drjit.nn.SinEncode>` where the
  per-octave phase offset did not match the documented formula. Code using
  ``shift=0`` is unaffected.
  (PR `#490 <https://github.com/mitsuba-renderer/drjit/pull/490>`__).

- Fixed incorrect type names in :py:func:`dr.graphviz_ad() <graphviz_ad>`.
  (commit `0c685e <https://github.com/mitsuba-renderer/drjit/commit/0c685e42d553aad27063c6fd756f5cba91ac503c>`__).

- Fixed minor memory leaks due to recorded/frozen kernels.
  (Dr.Jit-Core commit `f0bf64 <https://github.com/mitsuba-renderer/drjit-core/commit/f0bf641ead1ce23697d13dab89f980da467c188e>`__).

- Fixed memory leaks related to kernel histories.
  (Dr.Jit-Core commit `318e55 <https://github.com/mitsuba-renderer/drjit-core/commit/318e554a242a1de11e6a4694de16a6936ef1671f>`__).

- Renamed the conflicting ``KernelRecordingMode.None`` enumerator to
  ``Inactive`` to avoid the collision with Python's ``None``.
  (Dr.Jit-Core PR `#186 <https://github.com/mitsuba-renderer/drjit-core/pull/186>`__,
  Dr.Jit PR `#481 <https://github.com/mitsuba-renderer/drjit/pull/481>`__).

- Fixed several issues involving symbolic loops with aliased state variables.
  (Dr.Jit PRs `#505 <https://github.com/mitsuba-renderer/drjit/pull/505>`__,
  `#510 <https://github.com/mitsuba-renderer/drjit/pull/510>`__,
  Dr.Jit-Core PR `#198 <https://github.com/mitsuba-renderer/drjit-core/pull/198>`__,
  contributed by `Lovro Nuic <https://github.com/lnuic>`__).

- Fixed half-precision ``Min``/``Max`` reductions and the half-precision
  infinity constant.
  (Dr.Jit-Core PR `#199 <https://github.com/mitsuba-renderer/drjit-core/pull/199>`__).

- Various smaller backend fixes: a missing mask predicate in the CUDA packet
  ``scatter_reduce`` path, a crash in Metal cooperative-vector matrix-vector
  products with unsupported output dimensions, a race condition under
  multi-threaded Metal use, incorrect fast-math flag handling on ``Sqrt`` and
  ``Div`` nodes, and more robust handling of failed ``jit_eval()`` calls.
  (Dr.Jit-Core PRs `#191 <https://github.com/mitsuba-renderer/drjit-core/pull/191>`__,
  `#200 <https://github.com/mitsuba-renderer/drjit-core/pull/200>`__,
  `#196 <https://github.com/mitsuba-renderer/drjit-core/pull/196>`__,
  `#192 <https://github.com/mitsuba-renderer/drjit-core/pull/192>`__,
  commits
  `368c53 <https://github.com/mitsuba-renderer/drjit-core/commit/368c539bf5659224a4ad2d9be69a3093e4fa9714>`__,
  `37bbce <https://github.com/mitsuba-renderer/drjit-core/commit/37bbce6ccdf05910c53b4ca234b907b0dbd47845>`__,
  Dr.Jit PR `#503 <https://github.com/mitsuba-renderer/drjit/pull/503>`__).

**Other Improvements**

- Improved documentation and error messages when the Dr.Jit binary fails to
  load. (PR `#485 <https://github.com/mitsuba-renderer/drjit/pull/485>`__).

- Various improvements to Dr.Jit's static type annotations: added missing
  stubs for :py:func:`dr.mean() <mean>`, added type hints for ``PrefixRedOp``,
  and minor stub pattern replacement rule fixes.
  (PRs `#478 <https://github.com/mitsuba-renderer/drjit/pull/478>`__,
  `#480 <https://github.com/mitsuba-renderer/drjit/pull/480>`__,
  `#483 <https://github.com/mitsuba-renderer/drjit/pull/483>`__).

- **Release the GIL while waiting for kernel history**: Retrieving timing data
  via :py:class:`dr.kernel_history() <kernel_history>` now releases the GIL while
  waiting for the asynchronous results to arrive, allowing other Python threads
  to make progress in the meantime.
  (commits `766e1e <https://github.com/mitsuba-renderer/drjit/commit/766e1e9ade4c722a88d1a1dd18d7d5140115ab8f>`__,
  `f90bfd <https://github.com/mitsuba-renderer/drjit/commit/f90bfd6a77364a1dd0fff51ff1ab5e0999ff5c8b>`__).

- **ndarray Cleanup**: ndarray reclamation previously always went through an
  asynchronous cleanup thread. This detour is now skipped for CUDA and Metal arrays
  when the calling thread already holds the GIL.
  (commit `c01a23 <https://github.com/mitsuba-renderer/drjit/commit/c01a235744fe22c64c9a97bc1817a9f49b6b9a78>`__).

**API Breaks and Device Compatibility**

- ⚠️ :py:func:`nn.pack() <drjit.nn.pack>` and :py:func:`nn.unpack()
  <drjit.nn.unpack>` **no longer return the shared buffer as the first
  element of the result tuple**. They now return only the packed/unpacked
  PyTree with matrix views in place of the input tensors. The underlying
  buffer remains available via the :py:attr:`MatrixView.buffer
  <drjit.nn.MatrixView.buffer>` attribute, or, for a packed
  :py:class:`nn.Module <drjit.nn.Module>`, via the ``'weights'`` entry of
  the module's mapping interface (i.e. ``net['weights']``).

  Migration:

  .. code-block:: python

     # Before
     buffer, A_view, b_view = nn.pack(A, b, layout='training')
     dr.enable_grad(buffer)

     # After
     A_view, b_view = nn.pack(A, b, layout='training')
     dr.enable_grad(A_view.buffer)

  For a packed :py:class:`nn.Module <drjit.nn.Module>`:

  .. code-block:: python

     # Before
     buffer, net = nn.pack(net, layout='training')
     dr.enable_grad(buffer)

     # After
     net = nn.pack(net, layout='training')
     dr.enable_grad(net['weights'])

- ⚠️ **Removed TensorFlow support**.
  TensorFlow appears largely unmaintained. Over a year after the launch of
  NVIDIA's Blackwell GPU generation, there is still no official support in the
  official TensorFlow packages. This is a maintenance burden as our CI
  infrastructure uses this GPU. Consequently, we decided to drop Tensorflow
  support (``.tf()`` conversion, support in :py:func:`@dr.wrap <wrap>`).

- ⚠️ **Removed Kahan-compensated atomic scatter**. The
  ``drjit.scatter_add_kahan()`` operation was removed. See commit `f6b4be <https://github.com/mitsuba-renderer/drjit-core/commit/f6b4be02af6714a80fd07f970c9686ac2978e324>`__
  for the rationale.

- ⚠️ **Compute capability**. Dr.Jit-Core's CUDA backend now requires compute
  capability **7.5 or higher** (Turing and later) and NVIDIA driver **R535 or
  newer**.
  (Dr.Jit-Core PR `#188 <https://github.com/mitsuba-renderer/drjit-core/pull/188>`__).

DrJit 1.3.1 (February 23, 2026)
-------------------------------

**Bug Fixes**

- Fixed LLVM library search paths to include ``aarch64`` and WSL-specific
  directories. This resolves failures to locate LLVM on ARM Linux systems and
  Windows Subsystem for Linux.
  (Dr.Jit-Core PR `#185 <https://github.com/mitsuba-renderer/drjit-core/pull/185>`__).

- Fixed ordering of CUDA forward declarations of callables, resolving cases
  where a forward declaration could appear after the actual function definition.
  (Dr.Jit-Core commit `213983 <https://github.com/mitsuba-renderer/drjit-core/commit/213983e47c99db0c6ab5e3dfce952e68bb9a8bd3>`__).

DrJit 1.3.0 (February 16, 2026)
-------------------------------

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
  given interval. (commit `cb09ca <https://github.com/mitsuba-renderer/drjit/commit/cb09caaccbdb36b10b7a20cc140e8e34ae648771>`__).

- **Symbolic RNG mode**: :py:func:`dr.rng() <rng>` now accepts a
  ``symbolic`` argument for a purely symbolic sampler. (commit `51bacb <https://github.com/mitsuba-renderer/drjit/commit/51bacbf4bce75e5a5021d37a8400752b87a7a022>`__).

- **ArrayX Initialization from Tensors**: Nested array types with multiple
  dynamic dimensions (like ``ArrayXf``) can now be initialized from Dr.Jit
  tensors or NumPy arrays. (commit `e7e133 <https://github.com/mitsuba-renderer/drjit/commit/e7e1339921aa0db5dda3f7623684818548004186>`__).

- **Type Trait**: Added :py:func:`dr.replace_shape_t() <replace_shape_t>`
  convenience type trait for writing generic functions that need to reshape
  array types. (commit `46b245 <https://github.com/mitsuba-renderer/drjit/commit/46b24535bb3abd8efc3a2293b56c3bdc8adf6423>`__).

**Hardware/platform-specfic features**

- **NVIDIA Blackwell (SM120+)**: Added support for wide packet loads, gathers,
  and atomics on NVIDIA Blackwell GPUs (SM120+). (commit `879c10 <https://github.com/mitsuba-renderer/drjit/commit/879c103b01cd56b62d5fb0525db23be25b4a2dac>`__).

- **Python 3.14 Compatibility**: Fixed compatibility with PEP 649 deferred
  annotation evaluation, ensuring Dr.Jit works correctly on Python 3.14.
  (commit `7fa6eb <https://github.com/mitsuba-renderer/drjit/commit/7fa6eb4b513a2ce85140593872448960eda59aeb>`__).

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
  (commit `06b0a9 <https://github.com/mitsuba-renderer/drjit/commit/06b0a9db060519af74e795a223ecf49b6f98f4cc>`__).

- **GIL Release in __getitem__**: ``dr.ArrayBase.__getitem__()`` now releases
  the GIL while waiting, improving multi-threaded performance.
  (commit `c24be7 <https://github.com/mitsuba-renderer/drjit/commit/c24be704467a727cffb4716a04f523b068fbfe52>`__).

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
  (commit `16a81d <https://github.com/mitsuba-renderer/drjit/commit/16a81d088685c07b11906176b01bf7646b50893a>`__).

- Fixed lost tensor shapes when a loop or conditional is replayed for AD
  passes, with more robust inference of tensor output shapes.
  (commit `9d201f <https://github.com/mitsuba-renderer/drjit/commit/9d201f20390049a46531b8b6c0d5cca7dc74f854>`__).

- Fixed a regression in ``ArrayX`` initialization from tensors and NumPy
  ndarrays (wrong shape hint order for flipped axes and broken shift loop).
  (commit `df4cf4 <https://github.com/mitsuba-renderer/drjit/commit/df4cf483cb6919b9bdf427c9e33faeb4f68982a7>`__).

- Fixed ``Texture::eval_fetch_cuda`` to handle double-precision queries
  gracefully by casting to single-precision when a HW-accelerated texture is
  requested. (commits `83083d <https://github.com/mitsuba-renderer/drjit/commit/83083d8a8d1805418baa7e4324d1b6c1b73f74e4>`__,
  `054d11 <https://github.com/mitsuba-renderer/drjit/commit/054d11502e559b0c8d971ce4e640905ea94ff4f8>`__).

- Fixed symbolic loop size computation to also account for side-effect sizes.
  (Dr.Jit-Core commit `c6dfc8 <https://github.com/mitsuba-renderer/drjit-core/commit/c6dfc839fac41634ae58e996bcf5185049bc22b4>`__).

- Fixed spurious warning when freezing functions with very wide literals.
  (PR `#455 <https://github.com/mitsuba-renderer/drjit/pull/455>`__).

**Other Improvements**

- Updated to nanobind `v2.10.2
  <https://github.com/wjakob/nanobind/releases/tag/v2.10.2>`__.

- Improved documentation and log messages for textures, including
  clarifications regarding numerical precision and extra diagnostics for
  migrated textures. (commit `4edae0 <https://github.com/mitsuba-renderer/drjit/commit/4edae0afffa79cbba5227b0c0ee4b47f25300e13>`__).

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
  register pressure. (Dr.Jit-Core commit `5cf6d3 <https://github.com/mitsuba-renderer/drjit-core/commit/5cf6d3730ebc931863c89071e16979153e802d09>`__).

- **Memory View Support**: Arrays can now be converted to Python ``memoryview``
  objects for efficient zero-copy data access. (commit `b70391 <https://github.com/mitsuba-renderer/drjit/commit/b7039184ddfd33db6f4ea6a73575ab82c84dbaaa>`__).

- **DLPack GIL Release**: The ``dr.ArrayBase.dlpack()`` method now releases
  the GIL while waiting, improving multi-threaded performance. (commit `0adf9b <https://github.com/mitsuba-renderer/drjit/commit/0adf9b4a87a432cafebe1c62972da4724e6b5613>`__).

- **Thread Synchronization**: ``dr.sync_thread()`` now releases the GIL while
  waiting, preventing unnecessary blocking in multi-threaded applications.
  (commit `956d2f <https://github.com/mitsuba-renderer/drjit/commit/956d2f57ffebe86d1bab1bd4e90936cee87970d7>`__).

**API Improvements**

- **Spherical Direction Utilities**: Added Python implementation of spherical
  direction utilities (``dr.sphdir``).
  (PR `#432 <https://github.com/mitsuba-renderer/drjit/pull/432>`__,
  contributed by `Sébastien Speierer <https://github.com/Speierers>`__).

- **Matrix Conversions**: Added support for converting between 3D and 4D
  matrices: ``Matrix4f`` can be constructed from a 3D matrix and ``Matrix3f``
  from a 4D matrix. (commit `7f8ea8 <https://github.com/mitsuba-renderer/drjit/commit/7f8ea8907174ed85491849c6294bcb1264efe00a>`__).

- **Quaternion API**: Improved the quaternion Python API for better usability
  and consistency. (commit `282da8 <https://github.com/mitsuba-renderer/drjit/commit/282da88a7c07c05483e82238c39a920a1904202e>`__).

- **Type casts**: Allow casting between Dr.Jit types to properly allow
  AD<->non-AD conversions when required. (commit `72f1e6 <https://github.com/mitsuba-renderer/drjit/commit/72f1e6b2d84e360f3899a7e0b7dd2061faad3ef8>`__).

**Bug Fixes**

- Fixed deadlock issues in ``@dr.freeze`` decorator. (commit `e8fc55 <https://github.com/mitsuba-renderer/drjit/commit/e8fc555e182742fd63dcf681c545252d682eb546>`__).

- Fixed gradient tracking in ``Texture.tensor()`` to ensure gradients are
  never dropped inadvertently.
  (PR `#444 <https://github.com/mitsuba-renderer/drjit/pull/444>`__).

- Fixed AD support for C++ ``repeat`` and ``tile`` operations with proper
  gradient propagation. (commits `fd6930 <https://github.com/mitsuba-renderer/drjit/commit/fd69305652ddfccdd8c8fcdb36d07e43718e4c59>`__, `282da8 <https://github.com/mitsuba-renderer/drjit/commit/282da88a7c07c05483e82238c39a920a1904202e>`__).

- Fixed Python object traversal to check that ``__dict__`` exists before
  accessing it, preventing crashes with certain object types. (commit `433ada <https://github.com/mitsuba-renderer/drjit/commit/433adaf077952528437bb286a49223f396669cf3>`__).

- Fixed symbolic loop size calculation to properly account for side-effects.
  (Dr.Jit-Core commit `31bf91 <https://github.com/mitsuba-renderer/drjit-core/commit/31bf9119cdea83bcf039ba7422f9e4703122abc5>`__).

- Fixed read-after-free issue in OptiX SBT data loading.
  (Dr.Jit-Core commit `009ade <https://github.com/mitsuba-renderer/drjit-core/commit/009adef08b368bfdb4c098ded28e1545716559dc>`__, contributed by `Merlin Nimier-David <https://merlin.nimierdavid.fr>`__).

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
  <https://github.com/mitsuba-renderer/drjit/pull/402>`__, commit `e3f576 <https://github.com/mitsuba-renderer/drjit/commit/e3f57620cb58bac14dfd43189aa1bdf8ba0ff8c0>`__).

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
  commit `b59436 <https://github.com/mitsuba-renderer/drjit/commit/b59436b0f041af1ea7ba04bd508b39e2e9a43ac8>`__).

- Added :py:func:`dr.moveaxis() <moveaxis>` for rearranging tensor
  dimensions, providing NumPy-compatible axis movement. (commit `4d1478 <https://github.com/mitsuba-renderer/drjit/commit/4d14784696713f398eee6661913ee11e4d6b1934>`__).

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
  single-element arrays/tensors, similar to NumPy/PyTorch. (commit `a142bc <https://github.com/mitsuba-renderer/drjit/commit/a142bcdf2143785880cd57c640630abb8b560d9d>`__).

- Added :py:func:`dr.linear_to_srgb() <linear_to_srgb>` and
  :py:func:`dr.srgb_to_linear() <srgb_to_linear>` for color space conversions.
  (commit `a7f138 <https://github.com/mitsuba-renderer/drjit/commit/a7f1380cb2e684056b51ef6d08e6ea33154a5d62>`__).

- Added :py:attr:`JitFlag.ForbidSynchronization` to catch costly
  synchronization operations during development. (
  Dr.Jit PR `#350 <https://github.com/mitsuba-renderer/drjit/pull/350>`__,
  Dr.Jit-Core PR `#128
  <https://github.com/mitsuba-renderer/drjit-core/pull/128>`__).

- Added C++ bindings for thread-local memory arrays through the
  ``dr::Local<Value, Size>`` template, complementing the existing Python
  functionality. This enables efficient scratch space and stack-like data
  structures in GPU kernels from C++ code. (commit `c30ade <https://github.com/mitsuba-renderer/drjit/commit/c30ade7aa596dac838dedece2e73f5a4a3adcec8>`__).

**Notable Bugfixes**

- Fixed ``dr::block_reduce()`` derivative computation for
  arrays not evenly divisible by block size. (commit `df79ed <https://github.com/mitsuba-renderer/drjit/commit/df79ed894a110e2255515e9778032ccac38883a9>`__).

- Fixed potential performance cliffs in :py:func:`dr.gather() <gather>`
  by memoizing expressions and limiting expression growth (Dr.Jit-Core PR `#159
  <https://github.com/mitsuba-renderer/drjit-core/pull/159>`__).

- Fixed :py:func:`dr.rotate() <rotate>` quaternion component ordering to match C++
  implementation. (PR `#416
  <https://github.com/mitsuba-renderer/drjit/pull/416>`__).

- Fixed the derivative of :py:func:`dr.unit_angle() <unit_angle>` at signed zero.
  (commit `9d09a9 <https://github.com/mitsuba-renderer/drjit/commit/9d09a9e9310b29870756faa8b12fa7b1e60c7396>`__).

- Fixed memory leak in Python bindings using dedicated cleanup thread. (PR `#399
  <https://github.com/mitsuba-renderer/drjit/pull/399>`__).

- Preserve tensor shapes in symbolic operations. (commit `74c4d0 <https://github.com/mitsuba-renderer/drjit/commit/74c4d0313a420a22dd9e2fe0cb11205f051cb762>`__).

- Fixed evaluated loop derivative issues with unchanged differentiable state
  variables. (commit `074cfe <https://github.com/mitsuba-renderer/drjit/commit/074cfe9d0c2dc805af00d562a20c6c268477104d>`__).

- Fixed symbolic loop backward derivative compilation for simple loops.
  (commit `01ef10 <https://github.com/mitsuba-renderer/drjit/commit/01ef10ef3b5cb147c1c3116d089438dfcb97e2c8>`__).

- Fixed broadcasting of tensors and handling of unknown objects in
  :py:func:`dr.select() <select>`. (PRs `#339
  <https://github.com/mitsuba-renderer/drjit/issue/339>`__, PRs `#349
  <https://github.com/mitsuba-renderer/drjit/issue/349>`__).

- Fixed :py:func:`dr.abs() <abs>` derivative at x=0 to match PyTorch behavior. (commit `c597de <https://github.com/mitsuba-renderer/drjit/commit/c597de37d98a494e51bd55fc2f40e68d2258691f>`__).

- Fixes for NVIDIA 50-series GPUs and recent driver versions. (Dr.Jit-Core PR
  `#152 <https://github.com/mitsuba-renderer/drjit-core/pull/152>`__).

**Other Improvements**

- Fixed several corner cases in :py:func:`dr.dda.dda() <drjit.dda.dda>` (PR `#311
  <https://github.com/mitsuba-renderer/drjit/pull/311>`__).

- Added support for casting to and from boolean array types in Python. (commit `343d16 <https://github.com/mitsuba-renderer/drjit/commit/343d16e1305d6c51fcfaaa196ce7737a35768af7>`__).

- Enhanced :py:func:`dr.expr_t() <expr_t>` to preserve custom array types when
  compatible. (commit `85d66c <https://github.com/mitsuba-renderer/drjit/commit/85d66c3612190a6b653fc47cd9acbf6be4350e79>`__).

- Improved :py:func:`dr.replace_grad() <replace_grad>` to handle non-differentiable and unknown
  types gracefully. (PR `#364
  <https://github.com/mitsuba-renderer/drjit/pull/364>`__).

- Improved error handling throughout the codebase by replacing ``abort()``
  calls with exceptions for better recovery in interactive environments.
  (commit `27e34c <https://github.com/mitsuba-renderer/drjit/commit/27e34c2170af98a08ff25826a5d49238cc5a29a2>`__).

- Added :py:func:`dr.profile_enable() <profile_enable>` context manager for
  selective CUDA profiling using the NSight tools. (commit `e4dda9 <https://github.com/mitsuba-renderer/drjit/commit/e4dda97b53dba696db40e5a8097310d64fb385f9>`__).

- When compiling Dr.Jit with Clang/Linux, ``libstdc++`` can now also be used.
  Previously, the ``libc++`` standard library was required in this case. (PR
  `#346 <https://github.com/mitsuba-renderer/drjit/pull/346>`__).

DrJit 1.0.5 (February 3, 2025)
------------------------------

- Workaround for OptiX linking issue in driver version R570+. (commit `0c9c54 <https://github.com/mitsuba-renderer/drjit-core/commit/0c9c54ec5c2963dd576c5a16d10fb2d63d67166f>`__).

- Tensors can now be used as condition and state variables of
  ``dr.if_stmt/while_loop``. (commit `4691fe <https://github.com/mitsuba-renderer/drjit/commit/4691fe4421bfd7002cd9c5d998617db0f40cce35>`__).

DrJit 1.0.4 (January 28, 2025)
------------------------------

- Release was retracted

DrJit 1.0.3 (January 16, 2025)
------------------------------

- Fixes to :py:func:`drjit.wrap`. (commit `166be21 <https://github.com/mitsuba-renderer/drjit/pull/326/commits/166be21886e9fc66fe389cbc6f5becec1bfb3417>`__).

DrJit 1.0.2 (January 14, 2025)
------------------------------

- Warning about NVIDIA drivers v565+. (commit `b5fd88 <https://github.com/mitsuba-renderer/drjit-core/commit/b5fd886dcced5b7e5b229e94e2b9e702ae6aba46>`__).
- Support for boolean Python arguments in :py:func:`drjit.select`. (commit `d0c881 <https://github.com/mitsuba-renderer/drjit/commit/d0c881187c9ec0def50ef3f6cde32dacd86a96b4>`__).
- Backend refactoring: vectorized calls are now also isolated per variant. (commit `17bc70 <https://github.com/mitsuba-renderer/drjit/commit/17bc7078918662b06c6e80c3b5f3ac1d5f9f118f>`__).
- Fixes to :cpp:func:`dr::safe_cbrt() <drjit::safe_cbrt>`. (commit `2f8a3a <https://github.com/mitsuba-renderer/drjit/commit/2f8a3ab1acbf8e187a0ef4e248d0f65c00e27e3f>`__).

DrJit 1.0.1 (November 23, 2024)
-------------------------------

- Fixes to various edges cases of :py:func:`drjit.dda.dda` (commit `4ce97d <https://github.com/mitsuba-renderer/drjit/commit/4ce97dc4a5396c74887a6b123e2219e8def680d6>`__).

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
