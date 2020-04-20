project(enoki)
cmake_minimum_required(VERSION 3.9.0)

option(ENOKI_ENABLE_JIT      "Build Enoki just in time compilation library?" OFF)
option(ENOKI_ENABLE_AUTODIFF "Build Enoki automatic differentation library? (requires ENOKI_AUTODIFF)" OFF)
option(ENOKI_ENABLE_PYTHON   "Build pybind11 interface to Enoki types?" OFF)
option(ENOKI_ENABLE_TESTS    "Build Enoki test suite?" OFF)

if (ENOKI_ENABLE_AUTODIFF AND NOT ENOKI_ENABLE_JIT)
  message(WARNING "ENOKI_ENABLE_AUTODIFF depends on the JIT compiler. Setting ENOKI_ENABLE_JIT=ON.")
  set(ENOKI_ENABLE_JIT ON)
endif()

set(ENOKI_MASTER_PROJECT OFF)
if (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
  set(ENOKI_MASTER_PROJECT ON)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)

set(ENOKI_CXXFLAGS_BACKUP ${CMAKE_CXX_FLAGS})
set(ENOKI_NATIVE_FLAGS "")

if (MSVC)
  set(ENOKI_ARCH_FLAGS_FILE archflags_win32.cpp)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(WARNING "Enoki does not support vectorization on 32-bit Windows due to various")
    message(WARNING "platform limitations (unaligned stack, calling conventions don't allow")
    message(WARNING "passing vector registers, etc.). Switching to scalar mode.")
  endif()
else()
  set(ENOKI_ARCH_FLAGS_FILE archflags_unix.cpp)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
  if (ENOKI_CUDA AND CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DNDEBUG")
  endif()
endif()

# Clang refuses -march=native on ARM machines
if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "armv7")
  set(ENOKI_ARCH_FLAGS -march=armv7-a -mtune=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard -mfp16-format=ieee)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCH_FLAGS})
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(ENOKI_ARCH_FLAGS -march=armv8-a+simd -mtune=cortex-a53)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCH_FLAGS})
elseif (CMAKE_CXX_COMPILER MATCHES "/em\\+\\+(-[a-zA-Z0-9.])?$")
  # Emscripten
else()
  if (UNIX)
    set(ENOKI_ARCH_FLAGS_LINK_LIBRARIES ${CMAKE_EXE_LINKER_FLAGS})
  endif()
  try_run(
    ENOKI_ARCH_FLAGS_RETVAL ENOKI_ARCH_FLAGS_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/resources/${ENOKI_ARCH_FLAGS_FILE}
    COMPILE_OUTPUT_VARIABLE ENOKI_ARCH_FLAGS_MSG
    RUN_OUTPUT_VARIABLE ENOKI_ARCH_FLAGS
    LINK_LIBRARIES ${ENOKI_ARCH_FLAGS_LINK_LIBRARIES}
  )

  if (NOT ${ENOKI_ARCH_FLAGS_COMPILE_RESULT})
    message(FATAL_ERROR "Failed to compile 'archflags' binary: ${ENOKI_ARCH_FLAGS_MSG}")
  endif()
endif()

set(CMAKE_CXX_FLAGS ${ENOKI_CXXFLAGS_BACKUP})

# Optimize for current architecture
if (NOT ENOKI_NATIVE_FLAGS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    # set(ENOKI_NATIVE_FLAGS -march=native)
    set(ENOKI_NATIVE_FLAGS -msse4.2)
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(ENOKI_NATIVE_FLAGS -xHost)
  elseif (MSVC)
    # MSVC doesn't have a flag equivalent to -march=native
    # The 'archflags' executable provides this functionality
    set(ENOKI_NATIVE_FLAGS ${ENOKI_ARCH_FLAGS})
  endif()
endif()

if (NOT ENOKI_MASTER_PROJECT)
  set(ENOKI_ARCH_FLAGS   ${ENOKI_ARCH_FLAGS}   PARENT_SCOPE)
  set(ENOKI_NATIVE_FLAGS ${ENOKI_NATIVE_FLAGS} PARENT_SCOPE)
endif()

set(ENOKI_HOST "INTEL")
set(ENOKI_TEST_SSE42 OFF)
set(ENOKI_TEST_AVX OFF)
set(ENOKI_TEST_AVX2 OFF)
set(ENOKI_TEST_SKX OFF)
set(ENOKI_TEST_ARM OFF)

string(TOUPPER "${ENOKI_ARCH_FLAGS}" ENOKI_ARCH_FLAGS)

if (MSVC OR ENOKI_ARCH_FLAGS MATCHES "SSE")
  set(ENOKI_TEST_SSE42 ON)
endif()

if (ENOKI_ARCH_FLAGS MATCHES "AVX")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
endif()

if (ENOKI_ARCH_FLAGS MATCHES "AVX2")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
  set(ENOKI_TEST_AVX2 ON)
endif()

if (ENOKI_ARCH_FLAGS MATCHES "SKX")
  set(ENOKI_TEST_SSE42 ON)
  set(ENOKI_TEST_AVX ON)
  set(ENOKI_TEST_AVX2 ON)
  set(ENOKI_TEST_SKX ON)
endif()

if (ENOKI_ARCH_FLAGS MATCHES "ARMV")
  set(ENOKI_HOST "ARM")
  set(ENOKI_TEST_ARM ON)
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/resources")

macro(enoki_set_native_flags)
  add_compile_options(${ENOKI_NATIVE_FLAGS})
  string(REPLACE ";" " " ENOKI_NATIVE_FLAGS_STR "${ENOKI_NATIVE_FLAGS}")
  # Some linkers want to know the architecture flags (for LTO)
  if (NOT MSVC)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${ENOKI_NATIVE_FLAGS_STR}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${ENOKI_NATIVE_FLAGS_STR}")
  endif()
endmacro()

macro(enoki_set_compile_flags)
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
      CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
      CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    string(TOUPPER "${CMAKE_BUILD_TYPE}" ENOKI_U_CMAKE_BUILD_TYPE)
    if (NOT (${ENOKI_U_CMAKE_BUILD_TYPE} MATCHES "DEB"))
      if (NOT (${CMAKE_CXX_FLAGS} MATCHES "fsanitize"))
        # Don't use stack security features in release mode
        add_compile_options(-fno-stack-protector)

        # In release mode, don't keep the frame pointer in a dedicated register unless needed
        add_compile_options(-fomit-frame-pointer)
      endif()
    endif()

    # Never update the 'errno' variable due to arithmetic exceptions
    add_compile_options(-fno-math-errno)

    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Intel")
      # Look for opportunities to fuse additions & multiplications into FMA operations
      add_compile_options(-ffp-contract=fast)
    endif()
  endif()

  # Disable overly aggressive FP optimization in the Intel compiler
  if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    add_compile_options(-fp-model precise)
  endif()

  if (MSVC)
    # Disable buffer security check cookie
    set(Configurations RELEASE RELWITHDEBINFO MINSIZEREL)
    foreach(Configuration ${Configurations})
      string(REPLACE "/GS"  "" CMAKE_CXX_FLAGS_${Configuration} "${CMAKE_CXX_FLAGS_${Configuration}}")
    endforeach()
    string(REPLACE "/GS"  "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    add_compile_options("$<$<NOT:$<CONFIG:Debug>>:/GS->")

    # Enable intrinsic functions
    add_compile_options("$<$<CONFIG:Release>:/Oi>")

    # Honor __forceinline statements even in debug mode, needed to avoid internal compiler errors on MSVC
    string(REPLACE "/Ob0" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    add_compile_options("$<$<CONFIG:Debug>:/Ob1>")

    # Don't complain about class/struct X needs to have dll-interface to be used by clients of class/struct Y
    add_compile_options("/wd4251")
  endif()
endmacro()

include_directories(include)

set(ENOKI_HEADERS
    ${PROJECT_SOURCE_DIR}/include/enoki/array.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx2.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_avx512.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_base.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_call.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_enum.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_fallbacks.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_generic.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_idiv.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_intrin.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_kmask.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_masked.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_macro.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_math.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_recursive.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_router.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_sse42.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_static.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_struct.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_traits.h
    ${PROJECT_SOURCE_DIR}/include/enoki/array_utils.h
    ${PROJECT_SOURCE_DIR}/include/enoki/autodiff.h
    ${PROJECT_SOURCE_DIR}/include/enoki/color.h
    ${PROJECT_SOURCE_DIR}/include/enoki/complex.h
    ${PROJECT_SOURCE_DIR}/include/enoki/dynamic.h
    ${PROJECT_SOURCE_DIR}/include/enoki/fwd.h
    ${PROJECT_SOURCE_DIR}/include/enoki/half.h
    ${PROJECT_SOURCE_DIR}/include/enoki/matrix.h
    ${PROJECT_SOURCE_DIR}/include/enoki/morton.h
    ${PROJECT_SOURCE_DIR}/include/enoki/python.h
    ${PROJECT_SOURCE_DIR}/include/enoki/quaternion.h
    ${PROJECT_SOURCE_DIR}/include/enoki/random.h
    ${PROJECT_SOURCE_DIR}/include/enoki/sh.h
    ${PROJECT_SOURCE_DIR}/include/enoki/special.h
    ${PROJECT_SOURCE_DIR}/include/enoki/stl.h
    ${PROJECT_SOURCE_DIR}/include/enoki/transform.h
)

include(CheckCXXSourceRuns)

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  check_cxx_source_runs("#include <iostream>\nint main(int argc, char **argv) { std::cout << \"test\"; return 0; }" ${_RESULT})
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

# Prefer libc++ in conjunction with Clang
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (HAS_LIBCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
    message(STATUS "Enoki: using libc++.")
  else()
    CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP_AND_CPPABI "-stdlib=libc++" "-stdlib=libc++ -lc++abi")
    if (HAS_LIBCPP_AND_CPPABI)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++ -lc++abi")
      message(STATUS "Enoki: using libc++ and libc++abi.")
    else()
      message(FATAL_ERROR "When Clang is used to compile Enoki, libc++ must be available -- GCC's libstdc++ is not supported! (please insteall the libc++ development headers, provided e.g. by the packages 'libc++-dev' and 'libc++abi-dev' on Debian/Ubuntu).")
    endif()
  endif()
endif()

if (ENOKI_ENABLE_JIT)
  message(STATUS "Enoki: building the CUDA & LLVM JIT backend.")
  add_subdirectory(ext/enoki-jit)
endif()

if (ENOKI_ENABLE_AUTODIFF)
  message(STATUS "Enoki: building the autodiff backend.")
  add_subdirectory(src/autodiff)
endif()

if (ENOKI_TEST)
  enable_testing()
  add_subdirectory(tests)
endif()

enoki_set_native_flags()
enoki_set_compile_flags()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)

include_directories(ext/enoki-jit/include)

if (ENOKI_ENABLE_PYTHON)
  message(STATUS "Enoki: building the Python plugin.")
  add_subdirectory(src/python)

  set_target_properties(enoki-python PROPERTIES
    OUTPUT_NAME enoki
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  if (CMAKE_CXX_COMPILER_ID MATCHES "GCC|Clang")
    target_compile_options(enoki-python PRIVATE -g0 -Os)
  elseif (MSVC)
    target_compile_options(enoki-python PRIVATE /Os)
  endif()
else()
  message(STATUS "Enoki: *not* building the Python plugin.")
endif()

# Build the documentation
if (ENOKI_MASTER_PROJECT)
  find_package(Sphinx)

  if (Sphinx_FOUND)
    set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

    add_custom_target(mkdoc
        ${SPHINX_EXECUTABLE} -b html "${SPHINX_INPUT_DIR}" "${SPHINX_OUTPUT_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        USES_TERMINAL)
  endif()
endif()

mark_as_advanced(
  PYBIND11_INSTALL PYBIND11_PYTHON_VERSION PYBIND11_TEST
  PYBIND11_WERROR USE_PYTHON_INCLUDE_DIR
)
