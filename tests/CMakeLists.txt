nanobind_add_module(
  call_ext
  NB_DOMAIN drjit
  NB_SHARED
  call_ext.cpp
)

target_link_libraries(call_ext PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(call_ext PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(call_ext PRIVATE -DDRJIT_ENABLE_CUDA)
endif()
