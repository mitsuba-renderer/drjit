project(enoki)
cmake_minimum_required(VERSION 3.10.0)

option(ENOKI_ENABLE_JIT      "Build Enoki just in time compilation library?" OFF)
option(ENOKI_ENABLE_AUTODIFF "Build Enoki automatic differentation library?" OFF)
option(ENOKI_ENABLE_PYTHON   "Build Python extension library?" OFF)
option(ENOKI_ENABLE_TESTS    "Build Enoki test suite? (Warning, this takes *very* long to compile)" OFF)
set(ENOKI_NATIVE_FLAGS "-march=native" CACHE STRING "Compilation flags used to target the host processor architecture.")

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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)

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
  add_definitions(-DENOKI_ENABLE_JIT=1)
  set_target_properties(enoki-jit PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "Enoki: *not* building the CUDA & LLVM JIT backend.")
endif()

include_directories(
  include
  ext/enoki-jit/include
  ext/enoki-jit/ext/robin_map/include
)

if (MSVC)
   add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DNOMINMAX)
   add_compile_options(/wd4251) # 'warning C4251: X needs to have dll-interface to be used by clients of struct Y
endif()

if (ENOKI_ENABLE_AUTODIFF)
  message(STATUS "Enoki: building the autodiff backend.")
  add_definitions(-DENOKI_ENABLE_AUTODIFF=1)
  add_subdirectory(src/autodiff)
  set_target_properties(enoki-autodiff PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "Enoki: *not* building the autodiff backend.")
endif()

if (ENOKI_ENABLE_PYTHON)
  set(ENOKI_PYBIND11_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ext/pybind11"
    CACHE STRING "Path containing the 'pybind11' library used to compile Enoki.")
  add_subdirectory(${ENOKI_PYBIND11_DIR})

  message(STATUS "Enoki: building the Python plugin.")
  add_subdirectory(src/python)
else()
  message(STATUS "Enoki: *not* building the Python plugin.")
endif()

if (ENOKI_ENABLE_TESTS)
  message(STATUS "Enoki: building the test suite (Warning, this takes *very* long to compile).")
  add_subdirectory(tests)
endif()

# Build the documentation
if (ENOKI_MASTER_PROJECT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} resources)
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
  PYBIND11_WERROR USE_PYTHON_INCLUDE_DIR ENOKI_PYBIND11_DIR
  CMAKE_EXECUTABLE_FORMAT CMAKE_OSX_ARCHITECTURES
  ENOKI_JIT_ENABLE_TESTS
)
