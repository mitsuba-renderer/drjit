pybind11_add_module(
  enoki-python
  THIN_LTO OPT_SIZE

  main.cpp
  implicit.cpp
  reinterpret.cpp
  bind_scalar.cpp
  bind_packet.cpp
  bind_cuda.cpp
  bind_cuda_ad.cpp
  bind_llvm.cpp
  bind_llvm_ad.cpp
  dlpack.cpp
)

target_link_libraries(enoki-python PRIVATE enoki)

if (ENOKI_ENABLE_JIT)
  target_link_libraries(enoki-python PRIVATE enoki-jit)
  target_compile_definitions(enoki-python PRIVATE -DENOKI_ENABLE_JIT=1)
endif()

if (ENOKI_ENABLE_AUTODIFF)
  target_link_libraries(enoki-python PRIVATE enoki-autodiff)
  target_compile_definitions(enoki-python PRIVATE -DENOKI_ENABLE_AUTODIFF=1)
endif()

if (MSVC)
  set(ENOKI_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/../../enoki
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR}/../../enoki)
else()
  set(ENOKI_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../../enoki)
endif()

set_target_properties(enoki-python PROPERTIES
  OUTPUT_NAME enoki_ext ${ENOKI_OUTPUT_DIRECTORY})

target_compile_options(enoki-python PRIVATE ${ENOKI_NATIVE_FLAGS})

if (ENOKI_MASTER_PROJECT)
  install(TARGETS enoki-python LIBRARY DESTINATION enoki)
  set_target_properties(enoki-python PROPERTIES INSTALL_RPATH "${ENOKI_ORIGIN}")
endif()

if (ENOKI_MASTER_PROJECT AND NOT "${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
  set(ENOKI_PYTHON_FILES
     __init__.py const.py detail.py generic.py
     matrix.py router.py traits.py
  )

  set(ENOKI_COPY_FILES "")
  foreach(file ${ENOKI_PYTHON_FILES})
    set(IN_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../../enoki/${file})
    set(OUT_FILE ${CMAKE_CURRENT_BINARY_DIR}/../../enoki/${file})

    add_custom_command(
      OUTPUT ${OUT_FILE} DEPENDS ${IN_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy ${IN_FILE} ${OUT_FILE})
    list(APPEND ENOKI_COPY_FILES ${OUT_FILE})

    if (ENOKI_MASTER_PROJECT)
      install(FILES ${IN_FILE} DESTINATION enoki)
    endif()
  endforeach(file)

  add_custom_target(copy-python-files ALL DEPENDS ${ENOKI_COPY_FILES})
endif()
