set(DRJIT_AUTODIFF_VARIANTS drjit-autodiff-scalar-f32:float drjit-autodiff-scalar-f64:double)

include_directories(
  ../../ext/drjit-core/ext/robin_map/include
  ../../ext/drjit-core/ext/lz4
)

if (DRJIT_ENABLE_JIT)
  set(DRJIT_AUTODIFF_VARIANTS ${DRJIT_AUTODIFF_VARIANTS}
    drjit-autodiff-cuda-f32:CUDAArray<float>
    drjit-autodiff-cuda-f64:CUDAArray<double>
    drjit-autodiff-llvm-f32:LLVMArray<float>
    drjit-autodiff-llvm-f64:LLVMArray<double>)
endif()

foreach(DRJIT_AUTODIFF_VARIANT ${DRJIT_AUTODIFF_VARIANTS})
  string(REPLACE ":" ";" DRJIT_AUTODIFF_VARIANT ${DRJIT_AUTODIFF_VARIANT})
  list(GET DRJIT_AUTODIFF_VARIANT 0 DRJIT_AUTODIFF_VARIANT_NAME)
  list(GET DRJIT_AUTODIFF_VARIANT 1 DRJIT_AUTODIFF_VARIANT_VALUE)
  add_library(${DRJIT_AUTODIFF_VARIANT_NAME} OBJECT
    ../../include/drjit/autodiff.h
    autodiff.cpp
  )
  set_target_properties(${DRJIT_AUTODIFF_VARIANT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  set(DRJIT_AUTODIFF_OBJECTS ${DRJIT_AUTODIFF_OBJECTS} $<TARGET_OBJECTS:${DRJIT_AUTODIFF_VARIANT_NAME}>)
  string(REPLACE "drjit-autodiff-" "" DRJIT_AUTODIFF_VARIANT_NAME_U ${DRJIT_AUTODIFF_VARIANT_NAME})
  string(REPLACE "-" "_" DRJIT_AUTODIFF_VARIANT_NAME_U ${DRJIT_AUTODIFF_VARIANT_NAME_U})
  target_compile_definitions(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE
    -DDRJIT_AUTODIFF_NAME=${DRJIT_AUTODIFF_VARIANT_NAME_U}
    -DDRJIT_AUTODIFF_VALUE=${DRJIT_AUTODIFF_VARIANT_VALUE}
    -DDRJIT_BUILD_AUTODIFF=1
  )
  if (MSVC)
    target_compile_options(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE /wd4714) # marked as __forceinline not inlined
  else()
    target_compile_options(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE -fvisibility=hidden)
  endif()
  target_link_libraries(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE drjit)
  if (DRJIT_ENABLE_JIT)
    target_link_libraries(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE drjit-core)
    target_compile_definitions(${DRJIT_AUTODIFF_VARIANT_NAME} PRIVATE -DDRJIT_ENABLE_JIT=1)
  endif()
endforeach()

add_library(drjit-autodiff SHARED
  ${DRJIT_AUTODIFF_OBJECTS}
  common.h common.cpp
)
target_link_libraries(drjit-autodiff PUBLIC drjit)

if (NOT MSVC)
  target_compile_options(drjit-autodiff PRIVATE -fvisibility=hidden)
endif()

target_compile_definitions(drjit-autodiff PRIVATE -DDRJIT_BUILD_AUTODIFF=1)

if (DRJIT_ENABLE_JIT)
  target_link_libraries(drjit-autodiff PRIVATE drjit-core)
  target_compile_definitions(drjit-autodiff PRIVATE -DDRJIT_ENABLE_JIT=1)
endif()
