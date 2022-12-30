cmake_minimum_required(VERSION 3.13...3.18)

project(drjit
  DESCRIPTION
    "Dr.Jit"
  LANGUAGES
    CXX
)

option(DRJIT_ENABLE_JIT           "Build Dr.Jit just in time compilation library?" ON)
option(DRJIT_ENABLE_AUTODIFF      "Build Dr.Jit automatic differentation library?" ON)
option(DRJIT_ENABLE_PYTHON        "Build Python extension library?" ON)
option(DRJIT_ENABLE_PYTHON_PACKET "Enable packet mode in Python extension library?" OFF)
option(DRJIT_ENABLE_TESTS         "Build Dr.Jit test suite? (Warning, this takes *very* long to compile)" OFF)

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/drjit-core/ext/nanothread/ext/cmake-defaults")
  message(FATAL_ERROR "The Dr.Jit dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()

set(DRJIT_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/drjit/fwd.h")
include(ext/drjit-core/ext/nanothread/ext/cmake-defaults/CMakeLists.txt)

if (MSVC)
  set(DRJIT_OUTPUT_DIRECTORY
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}/drjit
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}/drjit
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/drjit
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR}/drjit
    LIBRARY_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}/drjit
    LIBRARY_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}/drjit
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/drjit
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR}/drjit
  )
else()
  set(DRJIT_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/drjit)
endif()

add_library(drjit INTERFACE)
target_compile_features(drjit INTERFACE cxx_std_17)
target_include_directories(drjit
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/drjit-core/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/drjit-core/ext/nanothread/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  target_compile_options(drjit INTERFACE -fno-strict-aliasing)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 14.0.0 AND
      CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.0.5)
    message(FATAL_ERROR "Clang versions 14.0.0 to 14.0.4 miscompile Dr.Jit. Please use either a newer (14.0.5+) or older (e.g., 13.x) version. On Ubuntu, you can use the LLVM project's package server (https://apt.llvm.org) to get the latest version.")
  endif()
endif()

if (DRJIT_MASTER_PROJECT)
  if (APPLE)
    set(DRJIT_ORIGIN "@loader_path")
  elseif(UNIX)
    set(DRJIT_ORIGIN "$ORIGIN")
  endif()

  set(CMAKE_INSTALL_RPATH "${DRJIT_ORIGIN}")
endif()

if (DRJIT_ENABLE_JIT)
  if (APPLE)
    message(STATUS "Dr.Jit: building the LLVM JIT backend.")
  else()
    message(STATUS "Dr.Jit: building the CUDA & LLVM JIT backend.")
  endif()
  add_subdirectory(ext/drjit-core)
  set_target_properties(drjit-core PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})
  set_target_properties(nanothread PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})

  if (DRJIT_MASTER_PROJECT)
    install(TARGETS drjit-core EXPORT drjitTargets)
    install(TARGETS nanothread EXPORT drjitTargets)
  endif()

  mark_as_advanced(DRJIT_THREAD_ENABLE_TESTS)
else()
  message(STATUS "Dr.Jit: *not* building the CUDA & LLVM JIT backend.")
endif()

if (MSVC)
   add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DNOMINMAX)
   add_compile_options(/wd4251) # 'warning C4251: X needs to have dll-interface to be used by clients of struct Y
endif()

if (DRJIT_ENABLE_AUTODIFF)
  message(STATUS "Dr.Jit: building the autodiff backend.")
  add_subdirectory(src/autodiff)
  set_target_properties(drjit-autodiff PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})

  if (DRJIT_MASTER_PROJECT)
    install(TARGETS drjit-autodiff EXPORT drjitTargets)
  endif()
else()
  message(STATUS "Dr.Jit: *not* building the autodiff backend.")
endif()

if (DRJIT_ENABLE_PYTHON)
  message(STATUS "Dr.Jit: building the Python plugin.")
  add_subdirectory(src/python)
else()
  message(STATUS "Dr.Jit: *not* building the Python plugin.")
endif()

if (DRJIT_ENABLE_TESTS)
  message(STATUS "Dr.Jit: building the test suite (Warning, this takes *very* long to compile).")
  enable_testing()
  add_subdirectory(tests)
endif()

# Build the documentation
if (DRJIT_MASTER_PROJECT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/resources)
  find_package(Sphinx)

  if (Sphinx_FOUND)
    set(SPHINX_INPUT_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/docs")
    set(SPHINX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/html")

    add_custom_target(mkdoc
        ${CMAKE_COMMAND} -E env PYTHONPATH="${CMAKE_CURRENT_BINARY_DIR}" ${SPHINX_EXECUTABLE} -b html "${SPHINX_INPUT_DIR}" "${SPHINX_OUTPUT_DIR}"
        COMMENT "Building HTML documentation with Sphinx"
        USES_TERMINAL)
  endif()
endif()

if (DRJIT_MASTER_PROJECT)
  install(DIRECTORY include/drjit DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if (DRJIT_ENABLE_JIT)
    install(DIRECTORY ext/drjit-core/include/drjit-core DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ext/drjit-core/ext/nanothread/include/nanothread DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif()

  install(TARGETS drjit EXPORT drjitTargets)

  set(DRJIT_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/drjit")

  configure_package_config_file(
    resources/drjitConfig.cmake.in drjitConfig.cmake
    INSTALL_DESTINATION ${DRJIT_CMAKECONFIG_INSTALL_DIR})

  write_basic_package_version_file(
    drjitConfigVersion.cmake
    VERSION ${DRJIT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)

  install(
    EXPORT drjitTargets
    DESTINATION ${DRJIT_CMAKECONFIG_INSTALL_DIR})

  install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/drjitConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/drjitConfig.cmake
    DESTINATION ${DRJIT_CMAKECONFIG_INSTALL_DIR})
endif()

mark_as_advanced(
  CMAKE_EXECUTABLE_FORMAT CMAKE_OSX_ARCHITECTURES
)
