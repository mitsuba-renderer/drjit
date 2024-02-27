if (SKBUILD)
  # Enable LTO only for release builds targeting PyPI (~5% binary size reduction)
  set(DRJIT_DIST_FLAGS LTO STABLE_ABI)
endif()

# ------------------- Copy Python resources -------------------
configure_file(
  ../../resources/config.py.in
  ${CMAKE_BINARY_DIR}/drjit/config.py
)

set(PY_FILES __init__.py ast.py config.py detail.py interop.py llvm/ad.py cuda/ad.py)
set(PY_FILES_OUT "${CMAKE_BINARY_DIR}/drjit/config.py")
foreach(PY_FILE IN LISTS PY_FILES)
  set(IN_FILE "${CMAKE_SOURCE_DIR}/drjit/${PY_FILE}")
  if (CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    set(OUT_FILE "${IN_FILE}")
  else()
    set(OUT_FILE "${CMAKE_BINARY_DIR}/drjit/${PY_FILE}")
    add_custom_command(
      DEPENDS "${IN_FILE}" OUTPUT "${OUT_FILE}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${IN_FILE} ${OUT_FILE})
  endif()
  list(APPEND PY_FILES_OUT ${OUT_FILE})
endforeach()

# ---------------- Build the main module ----------------

nanobind_add_module(
  drjit-python

  NB_DOMAIN drjit
  NB_STATIC
  ${DRJIT_DIST_FLAGS}

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
  profile.h     profile.cpp

  # Backends
  scalar.h      scalar.cpp
  llvm.h        llvm.cpp llvm_ad.cpp
  cuda.h        cuda.cpp cuda_ad.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME _drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

target_include_directories(drjit-python
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../ext/nanobind/ext/robin_map/include
)

target_compile_definitions(drjit-python PRIVATE -DDRJIT_PYTHON_BUILD)

if (MSVC)
  target_compile_options(drjit-python PRIVATE /wd4714) # function marked as __forceinline not inlined
endif()

target_link_libraries(drjit-python PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA)
endif()

# ------------------- Stub generation -------------------

set(STUB_ARGS
  PYTHON_PATH ${CMAKE_CURRENT_BINARY_DIR}/../..
  PATTERN_FILE  ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/stubs.pat
  DEPENDS drjit-python ${PY_FILES_OUT}
)

nanobind_add_stub(
  drjit-stub-main
  MODULE drjit
  OUTPUT ${CMAKE_BINARY_DIR}/drjit/__init__.pyi
  MARKER_FILE ${CMAKE_BINARY_DIR}/drjit/py.typed
  ${STUB_ARGS}
)

nanobind_add_stub(
  drjit-stub-ast
  MODULE drjit.ast
  OUTPUT ${CMAKE_BINARY_DIR}/drjit/ast.pyi
  ${STUB_ARGS}
)

nanobind_add_stub(
  drjit-stub-interop
  MODULE drjit.interop
  OUTPUT ${CMAKE_BINARY_DIR}/drjit/interop.pyi
  ${STUB_ARGS}
)

nanobind_add_stub(
  drjit-stub-detail
  MODULE drjit.detail
  OUTPUT ${CMAKE_BINARY_DIR}/drjit/detail.pyi
  ${STUB_ARGS}
)

nanobind_add_stub(
  drjit-stub-scalar
  MODULE drjit.scalar
  OUTPUT ${CMAKE_BINARY_DIR}/drjit/scalar/__init__.pyi
  ${STUB_ARGS}
)

if (DRJIT_ENABLE_LLVM)
  nanobind_add_stub(
    drjit-stub-llvm
    MARKER_FILE ${CMAKE_BINARY_DIR}/drjit/llvm/py.typed
    MODULE drjit.llvm
    OUTPUT ${CMAKE_BINARY_DIR}/drjit/llvm/__init__.pyi
    ${STUB_ARGS}
  )

  nanobind_add_stub(
    drjit-stub-llvm-ad
    MODULE drjit.llvm.ad
    OUTPUT ${CMAKE_BINARY_DIR}/drjit/llvm/ad.pyi
    ${STUB_ARGS}
  )
endif()

if (DRJIT_ENABLE_CUDA)
  nanobind_add_stub(
    drjit-stub-cuda
    MARKER_FILE ${CMAKE_BINARY_DIR}/drjit/cuda/py.typed
    MODULE drjit.cuda
    OUTPUT ${CMAKE_BINARY_DIR}/drjit/cuda/__init__.pyi
    ${STUB_ARGS}
  )

  nanobind_add_stub(
    drjit-stub-cuda-ad
    MODULE drjit.cuda.ad
    OUTPUT ${CMAKE_BINARY_DIR}/drjit/cuda/ad.pyi
    ${STUB_ARGS}
  )
endif()
