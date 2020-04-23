#pragma once

#define ENOKI_TRACK_SCALAR(text)                                               \
    do {                                                                       \
        printf("Scalar operation: %s\n", text);                                \
    } while (0)

#include <pybind11/pybind11.h>
#include <enoki/array.h>
#include <enoki/packet.h>
#include <enoki/dynamic.h>
#include <enoki-jit/traits.h>

namespace ek = enoki;
namespace py = pybind11;

using namespace py::literals;

/// Initialize an Enoki array from a variable-length argument list
extern void array_init(py::handle inst, const py::args &args, size_t size);

/// Determine underlying scalar type of an array
extern VarType var_type(py::handle h, VarType preferred = VarType::Invalid);

/// Is 'o' an Enoki array instance?
extern bool var_is_enoki(py::handle h);

/// Is 'o' an Enoki array instance *or* an Enoki type object?
extern bool var_is_enoki_type(py::handle h);

// Return the generic name associated with an n-dimensional array
extern const char* array_name(VarType vtype, size_t depth, size_t size, bool packet_mode);

/// Register implicit conversion handlers for the given Enoki type
extern void register_implicit_conversions(const std::type_info &type);

template <class Func, typename... Args>
pybind11::object classmethod(Func f, Args... args) {
    pybind11::object cf = pybind11::cpp_function(f, args...);
    return py::reinterpret_steal<py::object>(PyClassMethod_New(cf.ptr()));
}

