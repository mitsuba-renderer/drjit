add_library(
  drjit-extra
  SHARED

  common.h
  math.cpp
  autodiff.cpp
  call.cpp
)

nanobind_disable_stack_protector(drjit-extra)
target_compile_definitions(drjit-extra PRIVATE -DDRJIT_EXTRA_BUILD)
target_link_libraries(drjit-extra PRIVATE drjit drjit-core)

target_include_directories(drjit-extra PRIVATE
  ../../ext/drjit-core/ext/robin_map/include
  ../../ext/nanobind/include
)
