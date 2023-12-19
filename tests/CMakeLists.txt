function(add_drjit_test NAME)
  nanobind_add_module(
    ${NAME}
    NB_DOMAIN drjit
    NB_SHARED
    ${ARGV}
  )

  target_link_libraries(${NAME} PRIVATE drjit drjit-core drjit-extra)

  if (DRJIT_ENABLE_LLVM)
    target_compile_definitions(${NAME} PRIVATE -DDRJIT_ENABLE_LLVM)
  endif()

  if (DRJIT_ENABLE_CUDA)
    target_compile_definitions(${NAME} PRIVATE -DDRJIT_ENABLE_CUDA)
  endif()
endfunction()


add_drjit_test(call_ext call_ext.cpp)
add_drjit_test(while_loop_ext while_loop_ext.cpp)
add_drjit_test(if_stmt_ext if_stmt_ext.cpp)
