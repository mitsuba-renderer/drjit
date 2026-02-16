cmake_minimum_required(VERSION 3.13...3.22)

project(drjit
  DESCRIPTION
    "Dr.Jit"
  LANGUAGES
    CXX
)

if (APPLE)
  set(DRJIT_ENABLE_CUDA_DEFAULT OFF)
else()
  set(DRJIT_ENABLE_CUDA_DEFAULT ON)
endif()

if (SKBUILD)
  set(DRJIT_ENABLE_LEAK_WARNINGS_DEFAULT OFF)
else()
  set(DRJIT_ENABLE_LEAK_WARNINGS_DEFAULT ON)
endif()

option(DRJIT_ENABLE_LLVM          "Build the Dr.Jit LLVM backend?" ON)
option(DRJIT_ENABLE_CUDA          "Build the Dr.Jit CUDA backend?" ${DRJIT_ENABLE_CUDA_DEFAULT})
option(DRJIT_ENABLE_AUTODIFF      "Build Dr.Jit automatic differentation library?" ON)
option(DRJIT_ENABLE_PYTHON        "Build Python extension library?" ON)
option(DRJIT_ENABLE_TESTS         "Build Dr.Jit test suite? (Warning, this takes *very* long to compile)" OFF)
option(DRJIT_ENABLE_LEAK_WARNINGS "Emit DrJit and nanobind leak warnings? By default, leak warnings are disabled automatically when building PyPi wheels." ${DRJIT_ENABLE_LEAK_WARNINGS_DEFAULT})

option(DRJIT_STABLE_ABI       "Build Python extension using the CPython stable ABI? (Only relevant when using scikit-build)" OFF)
mark_as_advanced(DRJIT_STABLE_ABI)

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

if (SKBUILD)
  set(CMAKE_INSTALL_DATAROOTDIR drjit)
  set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_DATAROOTDIR}/include)
endif()

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
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/ext/drjit-core/ext/lz4>
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

if (DRJIT_ENABLE_CUDA)
  set(DRJIT_ENABLE_JIT TRUE)
  message(STATUS "Dr.Jit: building the CUDA backend.")
else()
  message(STATUS "Dr.Jit: *not* building the CUDA backend.")
endif()

if (DRJIT_ENABLE_LLVM)
  set(DRJIT_ENABLE_JIT TRUE)
  message(STATUS "Dr.Jit: building the LLVM backend.")
else()
  message(STATUS "Dr.Jit: *not* building the LLVM backend.")
endif()

if (DRJIT_ENABLE_JIT)
  add_subdirectory(ext/drjit-core)
  set_target_properties(drjit-core PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})
  set_target_properties(nanothread PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})
endif()

mark_as_advanced(NANOTHREAD_ENABLE_TESTS NANOTHREAD_STATIC)
mark_as_advanced(DRJIT_CORE_ENABLE_TESTS)
mark_as_advanced(NB_TEST NB_TEST_SHARED_BUILD NB_TEST_STABLE_ABI NB_USE_SUBMODULE_DEPS NB_TEST_SANITZE NB_CREATE_INSTALL_RULES nanobind_DIR)
mark_as_advanced(NB_TEST_CUDA NB_TEST_FREE_THREADED NB_TEST_SANITIZERS_ASAN NB_TEST_SANITIZERS_TSAN NB_TEST_SANITIZERS_UBSAN)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # In case Clang is used, avoid header file bloat from transitive
  # includes. Ignored if Clang is used together with libstdc++.
  add_compile_options(-D_LIBCPP_REMOVE_TRANSITIVE_INCLUDES)
endif()

if (MSVC)
   add_compile_options(/utf-8)  # Ensure source and execution charset is UTF-8
   add_compile_options(/wd4251) # "warning C4251: X needs to have dll-interface to be used by clients of struct Y"
   add_compile_options(/wd4127) # "warning C4127 : conditional expression is constant, consider using 'if constexpr' statement instead"

  # Workaround to avoid a particular crash due to MS VC++ runtime library binary
  # incompatibility. In general the runtime version needs to be at least as new
  # as toolset version but this define provides one such "escape hatch" if that
  # strictly isn't the case to combat against a commonly encountered reported crash
  add_compile_options(-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)
endif()

if (DRJIT_ENABLE_JIT)
  add_subdirectory(src/extra)
  set_target_properties(drjit-extra PROPERTIES ${DRJIT_OUTPUT_DIRECTORY})
endif()

if (DRJIT_ENABLE_PYTHON)
  message(STATUS "Dr.Jit: building the Python plugin.")
  add_subdirectory(src/python)
else()
  message(STATUS "Dr.Jit: *not* building the Python plugin.")
endif()

if (DRJIT_ENABLE_TESTS)
  if (NOT DRJIT_ENABLE_PYTHON)
    message(FATAL_ERROR "The test suite requires the Python plugin to be built.")
  endif()

  add_subdirectory(tests)
endif()

if (DRJIT_SANITIZE_UBSAN)
  list(APPEND DRJIT_SANITIZE "undefined")
endif()

if (DRJIT_SANITIZE_ASAN)
  list(APPEND DRJIT_SANITIZE "address")
endif()

if (DRJIT_SANITIZE_INTENSE)
  target_compile_definitions(drjit-core PRIVATE -DDRJIT_SANITIZE_INTENSE)
  target_compile_definitions(drjit-extra PRIVATE -DDRJIT_SANITIZE_INTENSE)
endif()

if (DRJIT_SANITIZE)
  string (REPLACE ";" "," DRJIT_SANITIZE "${DRJIT_SANITIZE}")
  target_compile_options(drjit INTERFACE -fsanitize=${DRJIT_SANITIZE})
  target_link_options (drjit INTERFACE -fsanitize=${DRJIT_SANITIZE})
  target_compile_options(drjit-core PRIVATE -fsanitize=${DRJIT_SANITIZE})
  target_link_options (drjit-core PRIVATE -fsanitize=${DRJIT_SANITIZE})
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

  install(TARGETS drjit-core EXPORT drjitTargets DESTINATION drjit)
  install(TARGETS drjit-extra EXPORT drjitTargets DESTINATION drjit)
  install(TARGETS nanothread EXPORT drjitTargets DESTINATION drjit)

  if (DRJIT_ENABLE_PYTHON)
    install(TARGETS drjit-python DESTINATION drjit)
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/config.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/__init__.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/__init__.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/ast.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/ast.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/dda.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/detail.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/detail.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/hashgrid.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/hashgrid.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/interop.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/interop.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/nn.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/nn.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/opt.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/random.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/random.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/py.typed
      DESTINATION drjit
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/scalar/__init__.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/scalar/__init__.py
      DESTINATION drjit/scalar
    )
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/llvm/py.typed
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/llvm/__init__.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/llvm/__init__.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/llvm/ad.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/llvm/ad.py
      DESTINATION drjit/llvm
    )
    if (DRJIT_ENABLE_CUDA)
      install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/drjit/cuda/py.typed
        ${CMAKE_CURRENT_BINARY_DIR}/drjit/cuda/__init__.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/drjit/cuda/__init__.py
        ${CMAKE_CURRENT_BINARY_DIR}/drjit/cuda/ad.pyi
        ${CMAKE_CURRENT_BINARY_DIR}/drjit/cuda/ad.py
        DESTINATION drjit/cuda
      )
    endif()
    install(FILES
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/auto/py.typed
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/auto/__init__.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/auto/__init__.py
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/auto/ad.pyi
      ${CMAKE_CURRENT_BINARY_DIR}/drjit/auto/ad.py
      DESTINATION drjit/auto
    )
  endif()

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
