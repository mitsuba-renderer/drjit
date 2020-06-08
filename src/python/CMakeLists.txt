pybind11_add_module(
  enoki-python
  THIN_LTO

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

set_target_properties(enoki-python PROPERTIES
  OUTPUT_NAME enoki_ext
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
  target_compile_options(enoki-python PRIVATE ${ENOKI_NATIVE_FLAGS})
  if (NOT (${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES "DEB"))
    target_compile_options(enoki-python PRIVATE -g0 -Os)
  endif()
elseif (MSVC)
  target_compile_options(enoki-python PRIVATE /Os)
endif()
