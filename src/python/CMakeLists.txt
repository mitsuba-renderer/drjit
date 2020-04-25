set(ENOKI_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/pybind11"
  CACHE STRING "Path containing the 'pybind11' library used to compile Enoki.")

add_subdirectory(${ENOKI_PYBIND11_DIR} pybind11)

pybind11_add_module(
  enoki-python
  THIN_LTO

  main.cpp
  implicit.cpp
  bind_scalar.cpp
  bind_packet.cpp
  bind_cuda.cpp
  bind_llvm.cpp

  #common.h
  #bind.h

  #main.cpp

  #promote.cpp promote.h

  #route_basics.cpp
  #route_init.cpp
  #route_math.cpp

  #constants.cpp
  #type_traits.cpp

)

if (ENOKI_ENABLE_JIT)
  target_link_libraries(enoki-python PRIVATE enoki-jit)
endif()


set_target_properties(enoki-python PROPERTIES
  OUTPUT_NAME enoki_ext
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
)

if (CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
  target_compile_options(enoki-python PRIVATE -g0 -Os)
elseif (MSVC)
  target_compile_options(enoki-python PRIVATE /Os)
endif()
