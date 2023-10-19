nanobind_add_module(
  vcall_ext
  vcall_ext.cpp
)

target_link_libraries(vcall_ext PRIVATE drjit drjit-extra)
