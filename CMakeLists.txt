cmake_minimum_required(VERSION 3.13...3.18)

project(enoki
  DESCRIPTION
    "Enoki 2"
  LANGUAGES
    CXX
)

option(ENOKI_ENABLE_JIT           "Build Enoki just in time compilation library?" OFF)
option(ENOKI_ENABLE_AUTODIFF      "Build Enoki automatic differentation library?" OFF)
option(ENOKI_ENABLE_PYTHON        "Build Python extension library?" OFF)
option(ENOKI_ENABLE_PYTHON_PACKET "Enable packet mode in Python extension library?" OFF)
option(ENOKI_ENABLE_TESTS         "Build Enoki test suite? (Warning, this takes *very* long to compile)" OFF)

# ----------------------------------------------------------
#  Check if submodules have been checked out, or fail early
# ----------------------------------------------------------

if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/ext/enoki-jit/ext/enoki-thread/ext/cmake-defaults")
  message(FATAL_ERROR "The Enoki dependencies are missing! "
    "You probably did not clone the project with --recursive. It is possible to recover "
    "by invoking\n$ git submodule update --init --recursive")
endif()


set(ENOKI_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/include/enoki/fwd.h")
include(ext/enoki-jit/ext/enoki-thread/ext/cmake-defaults/CMakeLists.txt)

if (MSVC)
  set(ENOKI_OUTPUT_DIRECTORY
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     ${CMAKE_CURRENT_BINARY_DIR}
  )
else()
  set(ENOKI_OUTPUT_DIRECTORY
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_library(enoki INTERFACE)
target_compile_features(enoki INTERFACE cxx_std_17)
target_include_directories(enoki
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/enoki-jit/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

if (ENOKI_ENABLE_JIT)
  if (APPLE)
    message(STATUS "Enoki: building the LLVM JIT backend.")
  else()
    message(STATUS "Enoki: building the CUDA & LLVM JIT backend.")
  endif()
  add_subdirectory(ext/enoki-jit)
  set_target_properties(enoki-jit PROPERTIES ${ENOKI_OUTPUT_DIRECTORY})
  set_target_properties(enoki-thread PROPERTIES ${ENOKI_OUTPUT_DIRECTORY})

  if (ENOKI_MASTER_PROJECT)
    install(TARGETS enoki-jit EXPORT enokiTargets)
    install(TARGETS enoki-thread EXPORT enokiTargets)
  endif()
else()
  message(STATUS "Enoki: *not* building the CUDA & LLVM JIT backend.")
endif()

if (MSVC)
   add_definitions(-D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_DEPRECATE -DNOMINMAX)
   add_compile_options(/wd4251) # 'warning C4251: X needs to have dll-interface to be used by clients of struct Y
endif()

if (ENOKI_ENABLE_AUTODIFF)
  message(STATUS "Enoki: building the autodiff backend.")
  add_subdirectory(src/autodiff)
  set_target_properties(enoki-autodiff PROPERTIES ${ENOKI_OUTPUT_DIRECTORY})

  if (ENOKI_MASTER_PROJECT)
    install(TARGETS enoki-autodiff EXPORT enokiTargets)
  endif()
else()
  message(STATUS "Enoki: *not* building the autodiff backend.")
endif()

if (ENOKI_ENABLE_PYTHON)
  message(STATUS "Enoki: building the Python plugin.")
  add_subdirectory(src/python)
else()
  message(STATUS "Enoki: *not* building the Python plugin.")
endif()

if (ENOKI_ENABLE_TESTS)
  message(STATUS "Enoki: building the test suite (Warning, this takes *very* long to compile).")
  enable_testing()
  add_subdirectory(tests)
endif()

# Build the documentation
if (ENOKI_MASTER_PROJECT)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/resources)
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

if (ENOKI_MASTER_PROJECT)
  install(DIRECTORY include/enoki DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if (ENOKI_ENABLE_JIT)
    install(DIRECTORY ext/enoki-jit/include/enoki-jit DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ext/enoki-jit/ext/enoki-thread/include/enoki-thread DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  endif()

  install(TARGETS enoki EXPORT enokiTargets)

  set(ENOKI_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_DATAROOTDIR}/cmake/enoki")

  configure_package_config_file(
    resources/enokiConfig.cmake.in enokiConfig.cmake
    INSTALL_DESTINATION ${ENOKI_CMAKECONFIG_INSTALL_DIR})

  write_basic_package_version_file(
    enokiConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT)

  install(
    EXPORT enokiTargets
    DESTINATION ${ENOKI_CMAKECONFIG_INSTALL_DIR})

  install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/enokiConfigVersion.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/enokiConfig.cmake
    DESTINATION ${ENOKI_CMAKECONFIG_INSTALL_DIR})
endif()

mark_as_advanced(
  CMAKE_EXECUTABLE_FORMAT CMAKE_OSX_ARCHITECTURES
  ENOKI_JIT_ENABLE_TESTS
)
