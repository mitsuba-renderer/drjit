if (SKBUILD)
  # Enable LTO only for release builds targeting PyPI (~5% binary size reduction)
  set(DRJIT_DIST_FLAGS LTO)
endif()

if (SKBUILD AND DRJIT_STABLE_ABI)
    list(APPEND DRJIT_DIST_FLAGS STABLE_ABI)
endif()

# ------------------- Copy Python resources -------------------

set(DRJIT_PYTHON_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../drjit")

if (NOT DRJIT_PYTHON_DST_DIR)
  set(DRJIT_PYTHON_DST_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../drjit")
endif()

configure_file(
  ../../resources/config.py.in
  "${DRJIT_PYTHON_SRC_DIR}/config.py"
)

set(PY_FILES
  config.py __init__.py ast.py detail.py interop.py dda.py opt.py nn.py
  random.py hashgrid.py _sh_eval.py _reduce.py scalar/__init__.py llvm/__init__.py
  llvm/ad.py cuda/__init__.py cuda/ad.py)

set(PY_FILES_OUT "")

foreach(PY_FILE IN LISTS PY_FILES)
  set(IN_FILE "${DRJIT_PYTHON_SRC_DIR}/${PY_FILE}")
  set(OUT_FILE "${DRJIT_PYTHON_DST_DIR}/${PY_FILE}")
  get_filename_component(IN_FILE "${IN_FILE}" ABSOLUTE)
  get_filename_component(OUT_FILE "${OUT_FILE}" REALPATH)

  if (NOT IN_FILE STREQUAL OUT_FILE)
    add_custom_command(
      DEPENDS "${IN_FILE}" OUTPUT "${OUT_FILE}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${IN_FILE} ${OUT_FILE})
  endif()
  list(APPEND PY_FILES_OUT ${OUT_FILE})
endforeach()

add_custom_target(drjit-copy-python-src ALL DEPENDS ${PY_FILES_OUT})

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
  freeze.h      freeze.cpp
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
  tracker.h     tracker.cpp
  local.h       local.cpp
  resample.h    resample.cpp
  coop_vec.h    coop_vec.cpp
  reorder.h     reorder.cpp
  quat.h        quat.cpp

  # Backends
  scalar.h      scalar.cpp
  llvm.h        llvm.cpp llvm_ad.cpp
  cuda.h        cuda.cpp cuda_ad.cpp
)


# Convert the docstrings from a .rst file into a header
# that can be parsed by the compiler
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/docstr.h"
  COMMAND "${Python_EXECUTABLE}"
  ARGS "${CMAKE_CURRENT_SOURCE_DIR}/docstr.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/docstr.rst"
    "${CMAKE_CURRENT_BINARY_DIR}/docstr.h"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/docstr.rst"
)

add_custom_target(
  drjit-python-docstrings
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/docstr.h"
)

target_include_directories(drjit-python PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# This file must be generated before we can run the Python build system
add_dependencies(
  drjit-python
  drjit-python-docstrings
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
else()
  target_compile_options(drjit-python PRIVATE -Wno-redundant-move)
endif()

target_link_libraries(drjit-python PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA)
endif()

if (DRJIT_ENABLE_OPTIX)
    target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_OPTIX)
endif()

# Disable DrJit and nanobind leak warnings (disabled by default in PyPi builds)
if (NOT DRJIT_ENABLE_LEAK_WARNINGS)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_DISABLE_LEAK_WARNINGS)
endif()

# --- MSVC: Copy shared libraries to Python folders -----

set(DRJIT_STUB_DEPENDS drjit-python drjit-copy-python-src)

if (MSVC)
  set(COPY_TARGETS drjit-core drjit-extra nanothread)
  add_custom_target(copy-targets-drjit ALL DEPENDS ${COPY_TARGETS})

  foreach(target ${COPY_TARGETS})
    get_target_property(TARGET_FOLDER ${target} FOLDER)
    set(COPY_TARGET_DESTINATION ${DRJIT_PYTHON_DST_DIR}/$<TARGET_FILE_NAME:${target}>)

    add_custom_command(
      TARGET copy-targets-drjit POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:${target}> ${COPY_TARGET_DESTINATION}
      DEPENDS ${target}
    )
  endforeach(target)

  list(APPEND DRJIT_STUB_DEPENDS copy-targets-drjit)
endif()

# ------------------- Stub generation -------------------

set(STUB_ARGS
  PYTHON_PATH ${DRJIT_PYTHON_DST_DIR}/..
  PATTERN_FILE ${DRJIT_PYTHON_SRC_DIR}/stubs.pat
  DEPENDS ${DRJIT_STUB_DEPENDS}
)

# The sanitizers require preloading shared libraries into Python, and that
# doesn't play well with stub generation. Disable stubs if sanitizers are
# enabled.

if (NOT (DRJIT_SANITIZE_ASAN OR DRJIT_SANITIZE_UBSAN))
  # Expected set of outputs and marker files for all stubs

  set(DRJIT_STUBS
    ${DRJIT_PYTHON_DST_DIR}/__init__.pyi
    ${DRJIT_PYTHON_DST_DIR}/ast.pyi
    ${DRJIT_PYTHON_DST_DIR}/interop.pyi
    ${DRJIT_PYTHON_DST_DIR}/detail.pyi
    ${DRJIT_PYTHON_DST_DIR}/hashgrid.pyi
    ${DRJIT_PYTHON_DST_DIR}/nn.pyi
    ${DRJIT_PYTHON_DST_DIR}/random.pyi
    ${DRJIT_PYTHON_DST_DIR}/scalar/__init__.pyi
  )

  set(DRJIT_MARKERS
    ${DRJIT_PYTHON_DST_DIR}/py.typed)

  if (DRJIT_ENABLE_LLVM)
    list(APPEND DRJIT_STUBS
      ${DRJIT_PYTHON_DST_DIR}/llvm/__init__.pyi
      ${DRJIT_PYTHON_DST_DIR}/llvm/ad.pyi)
    list(APPEND DRJIT_MARKERS
      ${DRJIT_PYTHON_DST_DIR}/llvm/py.typed)
  endif()

  if (DRJIT_ENABLE_CUDA)
    list(APPEND DRJIT_STUBS
      ${DRJIT_PYTHON_DST_DIR}/cuda/__init__.pyi
      ${DRJIT_PYTHON_DST_DIR}/cuda/ad.pyi)
    list(APPEND DRJIT_MARKERS
      ${DRJIT_PYTHON_DST_DIR}/cuda/py.typed)
  endif()

  nanobind_add_stub(
    drjit-stubs
    MODULE drjit
    RECURSIVE
    MARKER_FILE ${DRJIT_MARKERS}
    OUTPUT      ${DRJIT_STUBS}
    OUTPUT_PATH ${DRJIT_PYTHON_DST_DIR}
    ${STUB_ARGS}
  )

  # Generate the 'auto' version of the stubs using string replacement
  add_custom_command(
    OUTPUT ${DRJIT_PYTHON_DST_DIR}/auto/__init__.pyi
    DEPENDS drjit-stubs
    COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/auto-stub.cmake
      ${DRJIT_PYTHON_DST_DIR}/llvm/__init__.pyi
      ${DRJIT_PYTHON_DST_DIR}/auto/__init__.pyi
  )

  add_custom_command(
    OUTPUT ${DRJIT_PYTHON_DST_DIR}/auto/ad.pyi
    DEPENDS drjit-stubs
    COMMAND cmake -P ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/auto-stub.cmake
      ${DRJIT_PYTHON_DST_DIR}/llvm/ad.pyi
      ${DRJIT_PYTHON_DST_DIR}/auto/ad.pyi
  )

  add_custom_command(
    OUTPUT
      ${DRJIT_PYTHON_DST_DIR}/auto/py.typed
      ${DRJIT_PYTHON_DST_DIR}/auto/__init__.py
      ${DRJIT_PYTHON_DST_DIR}/auto/ad.py

    COMMAND cmake -E touch
      ${DRJIT_PYTHON_DST_DIR}/auto/py.typed
      ${DRJIT_PYTHON_DST_DIR}/auto/__init__.py
      ${DRJIT_PYTHON_DST_DIR}/auto/ad.py
  )

  add_custom_target(
    drjit-auto-stub ALL DEPENDS
    ${DRJIT_PYTHON_DST_DIR}/auto/__init__.pyi
    ${DRJIT_PYTHON_DST_DIR}/auto/__init__.py
    ${DRJIT_PYTHON_DST_DIR}/auto/ad.pyi
    ${DRJIT_PYTHON_DST_DIR}/auto/ad.py
    ${DRJIT_PYTHON_DST_DIR}/auto/py.typed
  )
endif()
