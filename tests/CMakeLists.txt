nanobind_add_module(
  vcall_ext
  NB_DOMAIN drjit
  NB_SHARED
  vcall_ext.cpp
)

target_link_libraries(vcall_ext PRIVATE drjit drjit-core drjit-extra)

if (DRJIT_ENABLE_LLVM)
  target_compile_definitions(vcall_ext PRIVATE -DDRJIT_ENABLE_LLVM)
endif()

if (DRJIT_ENABLE_CUDA)
  target_compile_definitions(vcall_ext PRIVATE -DDRJIT_ENABLE_CUDA)
endif()
