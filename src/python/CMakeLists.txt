pybind11_add_module(
  enoki-python
  THIN_LTO OPT_SIZE

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

if (ENOKI_ENABLE_JIT)
  target_link_libraries(enoki-python PRIVATE enoki-jit)
endif()

if (ENOKI_ENABLE_AUTODIFF)
  target_link_libraries(enoki-python PRIVATE enoki-autodiff)
endif()

if (MSVC)
  set(ENOKI_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR}/../../enoki)
else()
  set(ENOKI_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../enoki)
endif()

set_target_properties(enoki-python PROPERTIES
  OUTPUT_NAME enoki_ext ${ENOKI_OUTPUT_DIRECTORY})

target_compile_options(enoki-python PRIVATE ${ENOKI_NATIVE_FLAGS})
