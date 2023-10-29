if (SKBUILD)
  # Enable LTO only for release builds targeting PyPI (~5% binary size reduction)
  set(DRJIT_LTO_FLAG "LTO")
endif()

nanobind_add_module(
  drjit-python

  NB_DOMAIN drjit
  NB_STATIC
  STABLE_ABI
  ${DRJIT_LTO_FLAG}

  common.h
  main.cpp
  log.h      log.cpp
  meta.h     meta.cpp
  base.h     base.cpp
  bind.h     bind.cpp
  init.h     init.cpp
  repr.h     repr.cpp
  iter.h     iter.cpp
  shape.h    shape.cpp
  traits.h   traits.cpp
  scalar.h   scalar.cpp
  llvm.h     llvm.cpp llvm_ad.cpp
  cuda.h     cuda.cpp cuda_ad.cpp
  reduce.h   reduce.cpp
  apply.h    apply.cpp
  eval.h     eval.cpp
  memop.h    memop.cpp
  slice.h    slice.cpp
  dlpack.h   dlpack.cpp
  autodiff.h autodiff.cpp
  inspect.h  inspect.cpp
  misc.h     misc.cpp
  switch.h   switch.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

target_compile_definitions(drjit-python PRIVATE -DDRJIT_PYTHON_BUILD)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/config.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/config.py
)

target_link_libraries(drjit-python PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA)
endif()
