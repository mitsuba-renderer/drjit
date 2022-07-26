pybind11_add_module(
  drjit-python
  THIN_LTO OPT_SIZE

  bind.h
  common.h
  loop.h
  random.h
  tensor.h
  texture.h
  main.cpp
  implicit.cpp
  reinterpret.cpp
  bind_scalar.cpp
  bind_packet.cpp
  bind_cuda.cpp
  bind_cuda_ad.cpp
  bind_llvm.cpp
  bind_llvm_ad.cpp
  dlpack.cpp
)

target_link_libraries(drjit-python PRIVATE drjit)

if (MSVC)
  target_compile_options(drjit-python PRIVATE /wd4714) # function marked as __forceinline not inlined
endif()

# macOS: Avoid warnings about loop unrolling being disabled with -Os
target_compile_definitions(drjit-python PRIVATE "-DDRJIT_UNROLL= ")

if (DRJIT_ENABLE_JIT)
  target_link_libraries(drjit-python PRIVATE drjit-core)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_JIT=1)

  if (NOT APPLE)
    target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA=1)
  endif()
endif()

if (DRJIT_ENABLE_AUTODIFF)
  target_link_libraries(drjit-python PRIVATE drjit-autodiff)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_AUTODIFF=1)
endif()


if (DRJIT_ENABLE_PYTHON_PACKET)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_PYTHON_PACKET=1)
endif()

set_target_properties(drjit-python PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

target_compile_options(drjit-python PRIVATE ${DRJIT_NATIVE_FLAGS})

if (DRJIT_MASTER_PROJECT)
  install(TARGETS drjit-python LIBRARY DESTINATION drjit)
endif()

if (DRJIT_MASTER_PROJECT AND NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  set(DRJIT_PYTHON_FILES
     __init__.py const.py detail.py generic.py
     matrix.py router.py traits.py tensor.py
  )

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

# ----------------------------------------------------------
#   Generate type information stubs
# ----------------------------------------------------------

if (NOT Python_EXECUTABLE)
    find_package(Python COMPONENTS Interpreter REQUIRED)
endif()

add_custom_target(drjit_stub_file_generation ALL COMMAND
    "${Python_EXECUTABLE}" -Xutf8 ${CMAKE_SOURCE_DIR}/resources/generate_stub_files.py
    ${CMAKE_CURRENT_BINARY_DIR}/../../drjit/ ${CMAKE_CURRENT_BINARY_DIR}/../../drjit/ DEPENDS copy-python-files )
