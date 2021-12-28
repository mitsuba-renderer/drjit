#pragma once
#if defined(_WIN32)
#  include <corecrt.h>
#endif

#include <pybind11/pybind11.h>
#include <enoki/array.h>
#include <enoki/packet.h>
#include <enoki/dynamic.h>
#include <enoki-jit/traits.h>

namespace py = pybind11;
namespace ek = enoki;

using namespace py::literals;

/// Register an implicit conversion handler for a particular type
extern void register_implicit_conversions(const std::type_info &type);

extern py::object reinterpret_scalar(const py::object &source,
                                     VarType vt_source, VarType vt_target);

extern const uint32_t var_type_size[(int) VarType::Count];
extern const bool var_type_is_float[(int) VarType::Count];
extern const bool var_type_is_unsigned[(int) VarType::Count];

extern py::capsule to_dlpack(const py::object &owner, uint64_t data,
                             VarType type, int device, const py::tuple &shape,
                             const py::tuple &strides);

extern py::dict from_dlpack(const py::capsule &o);
