if (SKBUILD)
  set(DRJIT_DIST_FLAGS STABLE_ABI)
endif()

function(add_drjit_test NAME)
  nanobind_add_module(
    ${NAME}
    NB_DOMAIN drjit
    ${DRJIT_DIST_FLAGS}
    ${ARGV}
  )

  target_link_libraries(${NAME} PRIVATE drjit drjit-core drjit-extra)

  if (DRJIT_ENABLE_LLVM)
    target_compile_definitions(${NAME} PRIVATE -DDRJIT_ENABLE_LLVM)
  endif()

  if (DRJIT_ENABLE_CUDA)
    target_compile_definitions(${NAME} PRIVATE -DDRJIT_ENABLE_CUDA)
  endif()

  if (MSVC)
    message(STATUS ${CMAKE_CURRENT_BINARY_DIR})
    set_target_properties(${NAME} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}
      LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}
      LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}
      LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR})
  endif()
endfunction()


add_drjit_test(call_ext call_ext.cpp)
add_drjit_test(while_loop_ext while_loop_ext.cpp)
add_drjit_test(if_stmt_ext if_stmt_ext.cpp)
add_drjit_test(custom_type_ext custom_type_ext.cpp)
add_drjit_test(py_cpp_consistency_ext py_cpp_consistency_ext.cpp)
add_drjit_test(local_ext local_ext.cpp)

file(GLOB TEST_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.py")

foreach (FNAME IN ITEMS ${TEST_FILES})
  configure_file(${FNAME} ${FNAME} COPYONLY)
endforeach()

