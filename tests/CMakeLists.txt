string(TOUPPER "${DRJIT_ARCH_FLAGS}" DRJIT_ARCH_FLAGS)

if (MSVC)
  add_compile_options(/wd4127) # conditional expression is constant
  set(DRJIT_NONE_FLAGS /DDRJIT_DISABLE_VECTORIZATION)
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(DRJIT_SSE42_FLAGS /D__SSE4_2__)
  else()
    set(DRJIT_SSE42_FLAGS /arch:SSE2 /D__SSE4_2__)
  endif()
  set(DRJIT_AVX_FLAGS /arch:AVX)
  set(DRJIT_AVX2_FLAGS /arch:AVX2)
else()
  set(DRJIT_NONE_FLAGS -DDRJIT_DISABLE_VECTORIZATION -ffp-contract=off)
  set(DRJIT_SSE42_FLAGS -msse4.2)
  set(DRJIT_AVX_FLAGS -mavx)
  set(DRJIT_AVX2_FLAGS -mavx2 -mfma -mf16c -mbmi -mbmi2 -mlzcnt)
  if (APPLE AND ${CMAKE_CXX_COMPILER_ID} STREQUAL GNU)
    set(DRJIT_AVX512_FLAGS -march=skylake-avx512 -Wa,-march=skx)
  else()
    set(DRJIT_AVX512_FLAGS -march=skylake-avx512)
  endif()
  set(DRJIT_NEON_FLAGS )
  if (${CMAKE_SYSTEM_PROCESSOR} MATCHES aarch64)
    set(DRJIT_NEON_FLAGS -march=armv8-a+simd -mtune=cortex-a53)
  endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (NOT TARGET check)
  add_custom_target(check
          ${CMAKE_COMMAND} -E echo CWD=${CMAKE_BINARY_DIR}
          COMMAND ${CMAKE_COMMAND} -E echo CMD=${CMAKE_CTEST_COMMAND} -C $<CONFIG>
          COMMAND ${CMAKE_COMMAND} -E echo ----------------------------------
          COMMAND ${CMAKE_COMMAND} -E env CTEST_OUTPUT_ON_FAILURE=1
              ${CMAKE_CTEST_COMMAND} -C $<CONFIG>
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      USES_TERMINAL
  )
endif()

function(drjit_test NAME)
  if (DRJIT_TEST_NAME)
    if (NOT "${DRJIT_TEST_NAME}" STREQUAL "${NAME}")
      return()
    endif()
  endif()
  add_executable(${NAME}_none ${ARGN} ${DRJIT_HEADERS})
  add_test(${NAME}_none_test ${NAME}_none)
  set_tests_properties(${NAME}_none_test PROPERTIES LABELS "none")
  set_target_properties(${NAME}_none PROPERTIES FOLDER ${NAME})
  target_compile_options(${NAME}_none PRIVATE ${DRJIT_NONE_FLAGS})
  target_link_libraries(${NAME}_none drjit)

  if (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    add_executable(${NAME}_neon ${ARGN} ${DRJIT_HEADERS})
    target_compile_options(${NAME}_neon PRIVATE ${DRJIT_NEON_FLAGS})
    set_target_properties(${NAME}_neon PROPERTIES FOLDER ${NAME})
    add_test(${NAME}_neon_test ${NAME}_neon)
    set_tests_properties(${NAME}_neon_test PROPERTIES LABELS "neon")
    target_link_libraries(${NAME}_neon drjit)
  else()
    add_executable(${NAME}_sse42 ${ARGN} ${DRJIT_HEADERS})
    target_compile_options(${NAME}_sse42 PRIVATE ${DRJIT_SSE42_FLAGS})
    set_target_properties(${NAME}_sse42 PROPERTIES FOLDER ${NAME})
    add_test(${NAME}_sse42_test ${NAME}_sse42)
    set_tests_properties(${NAME}_sse42_test PROPERTIES LABELS "sse42")
    target_link_libraries(${NAME}_sse42 drjit)

    add_executable(${NAME}_avx ${ARGN} ${DRJIT_HEADERS})
    target_compile_options(${NAME}_avx PRIVATE ${DRJIT_AVX_FLAGS})
    set_target_properties(${NAME}_avx PROPERTIES FOLDER ${NAME})
    add_test(${NAME}_avx_test ${NAME}_avx)
    set_tests_properties(${NAME}_avx_test PROPERTIES LABELS "avx")
    target_link_libraries(${NAME}_avx drjit)

    add_executable(${NAME}_avx2 ${ARGN} ${DRJIT_HEADERS})
    target_compile_options(${NAME}_avx2 PRIVATE ${DRJIT_AVX2_FLAGS})
    set_target_properties(${NAME}_avx2 PROPERTIES FOLDER ${NAME})
    add_test(${NAME}_avx2_test ${NAME}_avx2)
    set_tests_properties(${NAME}_avx2_test PROPERTIES LABELS "avx2")
    target_link_libraries(${NAME}_avx2 drjit)

    if (NOT MSVC)
      add_executable(${NAME}_avx512 ${ARGN} ${DRJIT_HEADERS})
      target_compile_options(${NAME}_avx512 PRIVATE ${DRJIT_AVX512_FLAGS})
      set_target_properties(${NAME}_avx512 PROPERTIES FOLDER ${NAME})
      add_test(${NAME}_avx512_test ${NAME}_avx512)
      set_tests_properties(${NAME}_avx512_test PROPERTIES LABELS "avx512")
      target_link_libraries(${NAME}_avx512 drjit)
    endif()
  endif()
endfunction()

drjit_test(basic basic.cpp)
# drjit_test(call call.cpp
drjit_test(color color.cpp)
drjit_test(complex complex.cpp)
# drjit_test(conv conv.cpp
# drjit_test(dynamic dynamic.cpp
drjit_test(explog explog.cpp)
drjit_test(float float.cpp)
# drjit_test(histogram histogram.cpp
drjit_test(horiz horiz.cpp)
drjit_test(hyperbolic hyperbolic.cpp)
drjit_test(idiv idiv.cpp)
drjit_test(integer integer.cpp)
drjit_test(matrix matrix.cpp)
drjit_test(memory memory.cpp)
# drjit_test(memory2 memory2.cpp
# drjit_test(morton morton.cpp
drjit_test(nested nested.cpp)
drjit_test(sh sh.cpp)
# drjit_test(special special.cpp
# drjit_test(sphere sphere.cpp
drjit_test(struct struct.cpp)
drjit_test(trig trig.cpp)
# drjit_test(vector vector.cpp

# if (DRJIT_ENABLE_JIT)
#     add_executable(matrix matrix.cpp)
#     target_link_libraries(matrix drjit drjit-core)
#     add_test(matrix_test matrix)
# endif()

if (DRJIT_ENABLE_JIT AND DRJIT_ENABLE_AUTODIFF)
  add_executable(custom custom.cpp)
  target_link_libraries(custom drjit drjit-autodiff drjit-core)
  add_test(custom_test custom)
  set_tests_properties(custom_test PROPERTIES LABELS "jit")

  add_executable(vcall vcall.cpp)
  target_link_libraries(vcall drjit drjit-autodiff drjit-core)
  add_test(vcall_test vcall)
  set_tests_properties(vcall_test PROPERTIES LABELS "jit")

  add_executable(ad_megakernel ad_megakernel.cpp)
  target_link_libraries(ad_megakernel drjit drjit-autodiff drjit-core)
  add_test(ad_megakernel_test ad_megakernel)
  set_tests_properties(ad_megakernel_test PROPERTIES LABELS "jit")

  add_executable(texture texture.cpp)
  target_link_libraries(texture drjit drjit-autodiff drjit-core)
  add_test(texture_test texture)
  set_tests_properties(texture_test PROPERTIES LABELS "jit")

  add_executable(util util.cpp)
  target_link_libraries(util drjit drjit-autodiff drjit-core)
  add_test(util_test util)
  set_tests_properties(util_test PROPERTIES LABELS "jit")
endif()
