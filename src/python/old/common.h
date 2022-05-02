#pragma once
#if defined(_WIN32)
#  include <corecrt.h>
#endif

#include <nanobind/nanobind.h>
#include <drjit/array.h>
#include <drjit/packet.h>
#include <drjit/dynamic.h>
#include <drjit-core/traits.h>

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

/// Register an implicit conversion handler for a particular type
extern void register_implicit_conversions(const std::type_info &type);

extern nb::object reinterpret_scalar(const nb::object &source,
                                     VarType vt_source, VarType vt_target);

extern const uint32_t var_type_size[(int) VarType::Count];
extern const bool var_type_is_float[(int) VarType::Count];
extern const bool var_type_is_unsigned[(int) VarType::Count];

extern nb::capsule to_dlpack(const nb::object &owner, uint64_t data,
                             VarType type, int device, const nb::tuple &shape,
                             const nb::tuple &strides);

extern nb::dict from_dlpack(const nb::capsule &o);
