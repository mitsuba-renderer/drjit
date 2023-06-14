nanobind_add_module(
  drjit-python

  NB_STATIC
  STABLE_ABI
  LTO

  main.cpp
  meta.cpp
  base.cpp
  bind.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/config.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/config.py
)

target_link_libraries(drjit-python PRIVATE drjit)
