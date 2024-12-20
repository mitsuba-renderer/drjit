/*
    common.h -- Common definitions used by the Dr.Jit Python bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include <drjit/python.h>
#include <nanobind/stl/pair.h>
#include "docstr.h"

namespace nb = nanobind;
namespace dr = drjit;

using namespace nb::literals;

using dr::ArrayMeta;
using dr::ArraySupplement;
using dr::ArrayBinding;
using dr::ArrayOp;
using dr::ArrayBase;
using dr::vector;

inline const ArraySupplement &supp(nb::handle h) {
    return nb::type_supplement<ArraySupplement>(h);
}

inline ArrayBase* inst_ptr(nb::handle h) {
    return nb::inst_ptr<ArrayBase>(h);
}

/// Helper function to perform a tuple-based function call directly using the
/// CPython API. nanobind lacks a nice abstraction for this.
inline nb::object tuple_call(nb::handle callable, nb::handle tuple) {
    nb::object result = nb::steal(PyObject_CallObject(callable.ptr(), tuple.ptr()));
    if (!result.is_valid())
        nb::raise_python_error();
    return result;
}

#define raise_if(expr, ...)                                                    \
    do {                                                                       \
        if (NB_UNLIKELY(expr))                                                 \
            nb::raise(__VA_ARGS__);                                    \
    } while (false)

/// Create interned string for a few very commonly used identifiers
#define DR_STR(x) s_##x
extern nb::handle DR_STR(DRJIT_STRUCT);
extern nb::handle DR_STR(dataclasses);
extern nb::handle DR_STR(__dataclass_fields__);
extern nb::handle DR_STR(name);
extern nb::handle DR_STR(type);
extern nb::handle DR_STR(fields);
extern nb::handle DR_STR(_traverse_write);
extern nb::handle DR_STR(_traverse_read);
extern nb::handle DR_STR(_traverse_1_cb_rw);
extern nb::handle DR_STR(_traverse_1_cb_ro);
extern nb::handle DR_STR(_get_variant);
extern nb::handle DR_STR(typing);
extern nb::handle DR_STR(get_type_hints);

/// Extract the DRJIT_STRUCT element of a custom data structure type, if available
inline nb::dict get_drjit_struct(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(DRJIT_STRUCT), nb::handle());
    if (result.is_valid() && !result.type().is(&PyDict_Type))
        result = nb::object();
    return nb::borrow<nb::dict>(result);
}

/// Extract the dataclass fields element of a custom data structure type, if available
inline nb::object get_dataclass_fields(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(__dataclass_fields__), nb::handle());
    if (result.is_valid()) {
        result = nb::module_::import_(DR_STR(dataclasses)).attr(DR_STR(fields))(tp);

        // Handle postponed type information
        nb::object hints = nb::module_::import_(DR_STR(typing)).attr(DR_STR(get_type_hints))(tp);
        for (auto field : result) {
            nb::object field_type = field.attr(DR_STR(type));
            if (field_type.type().is(&PyUnicode_Type))
                field.attr(DR_STR(type)) = hints[field.attr(DR_STR(name))];
        }
    }
    return result;
}

/// Extract a read-only callback to traverse custom data structures
inline nb::object get_traverse_cb_ro(nb::handle tp) {
    return nb::getattr(tp, DR_STR(_traverse_1_cb_ro), nb::handle());
}

/// Extract a read-write callback to traverse custom data structures
inline nb::object get_traverse_cb_rw(nb::handle tp) {
    return nb::getattr(tp, DR_STR(_traverse_1_cb_rw), nb::handle());
}

inline nb::object get_variant_fn(nb::handle tp) {
    return nb::getattr(tp, DR_STR(_get_variant), nb::handle());
}
