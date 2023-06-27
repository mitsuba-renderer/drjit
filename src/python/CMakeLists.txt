nanobind_add_module(
  drjit-python

  NB_DOMAIN drjit
  NB_STATIC
  STABLE_ABI
  LTO

  common.h
  main.cpp
  log.h    log.cpp
  meta.h   meta.cpp
  base.h   base.cpp
  bind.h   bind.cpp
  init.h   init.cpp
  repr.h   repr.cpp
  iter.h   iter.cpp
  shape.h  shape.cpp
  traits.h traits.cpp
  scalar.h scalar.cpp
  llvm.h   llvm.cpp
  reduce.h reduce.cpp
  apply.h  apply.cpp
  eval.h   eval.cpp
)

set_target_properties(drjit-python
  PROPERTIES
  OUTPUT_NAME drjit_ext ${DRJIT_OUTPUT_DIRECTORY})

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../resources/config.py.in
  ${CMAKE_CURRENT_SOURCE_DIR}/../../drjit/config.py
)

target_link_libraries(drjit-python PRIVATE drjit drjit-core)

if (DRJIT_ENABLE_JIT)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_LLVM)
  target_compile_definitions(drjit-python PRIVATE -DDRJIT_ENABLE_CUDA)
endif()

