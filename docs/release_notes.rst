Release notes
=============

Being an experimental research framework, Dr.Jit does not strictly follow the
`Semantic Versioning <https://semver.org/>`_ convention. That said, we will
strive to document breaking API changes in the release notes below.

Dr.Jit 0.3.1
------------

*November 24, 2022*

- Fixed a regression in the C++ implementation of `dr::schedule()`
  `[5e9da6]<https://github.com/mitsuba-renderer/drjit/commit/5e9da65f0e834927349713a5da1ae6e4e207ee16>`_

Dr.Jit 0.3.0
------------

*November 23, 2022*

- Update Dr.Jit Core (various performance and stability improvements)
- Allow functions in ``dr.wrap_ad``to return nested data structures  `[2d4910]<https://github.com/mitsuba-renderer/drjit/commit/2d4910b002baec8b96f80dc37fb4305cd5230c1f>`_
- Fix matrix stride computation `[c7451c]<https://github.com/mitsuba-renderer/drjit/commit/c7451ced5a77d59fb47d90340c49852ada97269d>`_
- Fix ``dr.tile`` and ``dr.repeat`` for Bool types `[c15a71]<https://github.com/mitsuba-renderer/drjit/commit/c15a71d4cf439fe239e1b6713fc426c6d94c45b7>`_
- Support list/dict arguments in ``dr.wrap_ad`` `[9f711c]<https://github.com/mitsuba-renderer/drjit/commit/9f711c5d5efd9ff04a6aa490ea452c51534557cf>`_
- Add 3D variant for ``dr::meshgrid`` in C++ `[ed3d046]<https://github.com/mitsuba-renderer/drjit/commit/ed3d046f4ad6f27090fa9a3106ce310c77edf4b2>`_
- Fix ``dr.unravel`` for Tensor inputs `[578b0dd]<https://github.com/mitsuba-renderer/drjit/commit/578b0dd6258995c95cd9a9213f1d7db39e93c0e9>`_
- Add ``dr::suspend_grad`` and ``dr::resume_grad`` C++ routines `[112c294]<https://github.com/mitsuba-renderer/drjit/commit/112c2940148e8173e5128c962d4dd50d0b9cd579>`_
- Various Python type information (stub generation) improvements `[b102b3c]<https://github.com/mitsuba-renderer/drjit/commit/b102b3ccfe0dac39c580e8112983815dd10da566>`_


Dr.Jit 0.2.2
-------------

*September 12, 2022*

- Add bindings for ``dr.llvm_version()`` `[07e9da8] <https://github.com/mitsuba-renderer/drjit/commit/07e9da811e7284b87fa292472b30ec4465592eef>`_
- Fix ``dr.make_opaque`` for diff TensorXf `[f6bde89] <https://github.com/mitsuba-renderer/drjit/commit/f6bde8920f352f8ea96e652034662e3513a59c45>`_
- Change behavior of ``dr.diff_array_t`` to always return a type `[e0172fc] <https://github.com/mitsuba-renderer/drjit/commit/e0172fcdfcf2a8152d2fe03c1920fe31a0659d93>`_
- Compilation fixes for stub files generation `[bf770d4] <https://github.com/mitsuba-renderer/drjit/commit/bf770d43f6f46f0949067ef81ee3bf061b69a6e6>`_
- Add missing operations on Tensor types `[15d490c] <https://github.com/mitsuba-renderer/drjit/commit/15d490c0f4da2ac9f5f56c249eb2bcb6e6e64da2>`_
- Fix dr.shape for ragged arrays `[a026b56] <https://github.com/mitsuba-renderer/drjit/commit/a026b5695f7abb499e483f5d2cd1523f9084e826>`_
- Add the ``dr.wrap_ad()`` function decorator for interoperability between AD-aware frameworks (e.g. Dr.Jit and PyTorch) `[4a1528e] <https://github.com/mitsuba-renderer/drjit/commit/4a1528ee057c83422316825439b078a7d5277ec4>`_
- ``dr.device`` handles the case where the input was allocated by another framework `[9e993a6] <https://github.com/mitsuba-renderer/drjit/commit/9e993a61870dfab325050368380038e76d95ffa3>`_


Dr.Jit 0.2.1
-------------

*July 20, 2022*

- Fix upsampling of multichannel textures `[53dd605] <https://github.com/mitsuba-renderer/drjit/commit/53dd6058069cbfc98e7bf28cfef6f3f881ebbf5f>`_


Dr.Jit 0.2.0
-------------

*July 19, 2022*

- Change cubic texture gradient and hessian functions to also return the texture values `[1d50efe] <https://github.com/mitsuba-renderer/drjit/commit/1d50efecaad7afac71e32ff5967016a5f816b3bb>`_
- Add support for non-array types in ``dr.CustomOp`` inputs
- Various minor fixes in C++ test suite `[dcaf69a] <https://github.com/mitsuba-renderer/drjit/commit/dcaf69a7a8531692146ef489506cff40b2fab34f>`_


Dr.Jit 0.1.0
-------------

*July 18, 2022*

- Initial release
