#pragma once

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

extern void enoki_free_keepalive(const enoki::ArrayBase *array);

template <typename T> struct EnokiHolder {
    EnokiHolder() : value(nullptr) { }
    EnokiHolder(T *value) : value(value) { }
    ~EnokiHolder() {
        enoki_free_keepalive(value);
        delete value;
    }

    T *get() { return value; }

private:
    T *value;
};

PYBIND11_DECLARE_HOLDER_TYPE(T, EnokiHolder<T>);
