add_library(
  drjit-extra
  SHARED

  common.h
  math.cpp
  autodiff.cpp
  call.cpp
  loop.cpp
  cond.cpp
)

if (NOT MSVC)
  target_compile_options(drjit-extra PRIVATE $<$<OR:$<CONFIG:Release>,$<CONFIG:MinSizeRel>>:-fno-stack-protector>)
else()
  # C24127 conditional expression is constant (a few in robin_hash.h)
  # C24324 structure was padded due to alignment specifier
  # C26294 ill-defined for-loop. Loop body not executed (false-positive)
  target_compile_options(drjit-extra PRIVATE /wd4127 /wd4324 /wd6294)
endif()

target_compile_definitions(drjit-extra PRIVATE -DDRJIT_EXTRA_BUILD)
target_link_libraries(drjit-extra PRIVATE drjit drjit-core)

target_include_directories(drjit-extra PRIVATE
  ../../ext/drjit-core/ext/robin_map/include
  ../../ext/nanobind/include
)
