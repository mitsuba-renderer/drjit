Python_add_library(
  drjit-python
  SHARED
  ../../include/drjit/python.h
  python.cpp
)

nanobind_cpp17(drjit-python)
nanobind_msvc(drjit-python)
nanobind_headers(drjit-python)
nanobuild_build_library(nanobind SHARED)
target_link_libraries(drjit-python PUBLIC nanobind)
target_include_directories(drjit-python PUBLIC ${Python_INCLUDE_DIRS})
if (APPLE)
  target_link_options(drjit-python PRIVATE -undefined dynamic_lookup)
endif()
nanobind_disable_stack_protector(drjit-python)
nanobind_lto(drjit-python)

nanobind_add_module(
  drjit-python-ext
  random.h
  python.h
  main.cpp
  array_builtin.cpp
  array_math.cpp
  array_misc.cpp
  array_init.cpp
  repr.cpp
  shape.cpp
  traits.cpp
  promote.cpp
  tensor.cpp

  # Specific backends
  scalar.cpp
  cuda.cpp
  cuda_ad.cpp
  llvm.cpp
  llvm_ad.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit-python ${DRJIT_OUTPUT_DIRECTORY})

set_target_properties(nanobind
  PROPERTIES
  OUTPUT_NAME drjit-nanobind ${DRJIT_OUTPUT_DIRECTORY})

set_target_properties(nanothread
  PROPERTIES
  OUTPUT_NAME drjit-nanothread ${DRJIT_OUTPUT_DIRECTORY})

target_link_libraries(drjit-python PRIVATE drjit)
target_compile_definitions(drjit-python PRIVATE -DDRJIT_PYTHON_BUILD)
target_link_libraries(drjit-python-ext PRIVATE drjit-python drjit)

if (MSVC)
  target_compile_options(drjit-python-ext PRIVATE /wd4714) # function marked as __forceinline not inlined
endif()

# macOS: Avoid warnings about loop unrolling being disabled with -Os
target_compile_definitions(drjit-python-ext PRIVATE "-DDRJIT_UNROLL= ")

if (DRJIT_ENABLE_JIT)
  target_link_libraries(drjit-python-ext PRIVATE drjit-core)
  target_compile_definitions(drjit-python-ext PRIVATE -DDRJIT_ENABLE_JIT=1)

  if (NOT APPLE)
    target_compile_definitions(drjit-python-ext PRIVATE -DDRJIT_ENABLE_CUDA=1)
  endif()
endif()

if (DRJIT_ENABLE_AUTODIFF)
  target_link_libraries(drjit-python-ext PRIVATE drjit-autodiff)
  target_compile_definitions(drjit-python-ext PRIVATE -DDRJIT_ENABLE_AUTODIFF=1)
endif()


if (DRJIT_ENABLE_PYTHON_PACKET)
  target_compile_definitions(drjit-python-ext PRIVATE -DDRJIT_ENABLE_PYTHON_PACKET=1)
endif()

set_target_properties(drjit-python-ext PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

target_compile_options(drjit-python-ext PRIVATE ${DRJIT_NATIVE_FLAGS})

if (DRJIT_MASTER_PROJECT)
  # Adapt to variations in rpath handling on Linux and macOS
  if (APPLE)
    set(DRJIT_ORIGIN "@loader_path")
  elseif(UNIX)
    set(DRJIT_ORIGIN "$ORIGIN")
  endif()

  install(TARGETS drjit-python-ext LIBRARY DESTINATION drjit)
  set_target_properties(drjit-python-ext PROPERTIES INSTALL_RPATH "${DRJIT_ORIGIN}")
endif()

if (DRJIT_MASTER_PROJECT AND NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  set(DRJIT_PYTHON_FILES __init__.py)

  set(DRJIT_COPY_FILES "")
  foreach(file ${DRJIT_PYTHON_FILES})
    set(IN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/${file})
    set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/../../drjit/${file})

    add_custom_command(
      OUTPUT ${OUT_FILE} DEPENDS ${IN_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${IN_FILE} ${OUT_FILE})
    list(APPEND DRJIT_COPY_FILES ${OUT_FILE})
    install(FILES ${IN_FILE} DESTINATION drjit)
  endforeach(file)

  add_custom_target(copy-python-files ALL DEPENDS ${DRJIT_COPY_FILES})
endif()
