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

/// Register an implicit conversion handler for a particular type
extern void register_implicit_conversions(const std::type_info &type);
