nanobind_add_module(
  drjit-python

  NB_STATIC
  STABLE_ABI
  LTO

  common.h
  main.cpp
  meta.h meta.cpp
  base.h base.cpp
  bind.h bind.cpp
  init.h init.cpp
  repr.h repr.cpp
  shape.h shape.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/config.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/config.py
)

target_link_libraries(drjit-python PRIVATE drjit)
