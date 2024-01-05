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

  # Entry point
  main.cpp

  # Shared functionality
  common.h
  log.h         log.cpp
  meta.h        meta.cpp
  base.h        base.cpp
  bind.h        bind.cpp
  init.h        init.cpp
  iter.h        iter.cpp
  shape.h       shape.cpp
  traits.h      traits.cpp
  reduce.h      reduce.cpp
  apply.h       apply.cpp
  eval.h        eval.cpp
  memop.h       memop.cpp
  slice.h       slice.cpp
  dlpack.h      dlpack.cpp
  autodiff.h    autodiff.cpp
  inspect.h     inspect.cpp
  detail.h      detail.cpp
  switch.h      switch.cpp
  while_loop.h  while_loop.cpp
  if_stmt.h     if_stmt.cpp
  print.h       print.cpp
  history.h     history.cpp

  # Backends
  scalar.h      scalar.cpp
  llvm.h        llvm.cpp llvm_ad.cpp
  cuda.h        cuda.cpp cuda_ad.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

target_include_directories(drjit-python
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/nanobind/ext/robin_map/include
)

target_compile_definitions(drjit-python PRIVATE -DDRJIT_PYTHON_BUILD)

configure_file(
  ../../resources/config.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/../../drjit/config.py
)

foreach (FNAME IN ITEMS __init__.py ast.py detail.py)
  configure_file(../../drjit/${FNAME} ${CMAKE_CURRENT_BINARY_DIR}/../../drjit/${FNAME} COPYONLY)
endforeach()

target_link_libraries(drjit-python PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA)
endif()
