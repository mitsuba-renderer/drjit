set(ENOKI_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ext/pybind11"
  CACHE STRING "Path containing the 'pybind11' library used to compile Enoki.")

add_subdirectory(${ENOKI_PYBIND11_DIR} pybind11)

pybind11_add_module(
  enoki-python
  THIN_LTO

  common.h
  bind.h

  main.cpp

  promote.cpp promote.h

  route_basics.cpp
  route_init.cpp
  route_math.cpp

  constants.cpp
  type_traits.cpp

  bind_scalar.cpp
  bind_packet.cpp
  bind_cuda.cpp
)

if (ENOKI_ENABLE_JIT)
  target_link_libraries(enoki-python PRIVATE enoki-jit)
endif()
