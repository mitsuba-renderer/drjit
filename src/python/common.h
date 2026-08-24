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

using nb::literals::operator""_a;

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

/// Call 'callable' with one positional argument, returning a null object when
/// the call fails. Uses the vector call protocol and works in the limited API.
inline nb::object call_one_arg(nb::handle callable, nb::handle arg) {
    PyObject *args[2] = { nullptr, arg.ptr() };
    return nb::steal(nb::detail::vectorcall(
        callable.ptr(), args + 1, 1 | NB_VECTORCALL_ARGUMENTS_OFFSET, nullptr));
}

#define raise_if(expr, ...)                                                    \
    do {                                                                       \
        if (NB_UNLIKELY(expr))                                                 \
            nb::raise(__VA_ARGS__);                                    \
    } while (false)

/// Create interned string for a few very commonly used identifiers
#define DR_STR(x) s_##x
extern nb::handle DR_STR(DRJIT_STRUCT);
extern nb::handle DR_STR(__dataclass_fields__);
extern nb::handle DR_STR(name);
extern nb::handle DR_STR(type);
extern nb::handle DR_STR(cell_contents);

/// Dr.Jit can lazily import and then cache the following objects to avoid
/// costly nb::module_::import() calls.
enum class LazyImport {
    DataclassesFields,   // dataclasses.fields
    TypingGetTypeHints,  // typing.get_type_hints
    TypingGetArgs,       // typing.get_args
    DataclassesField,    // dataclasses._FIELD (marker of ordinary fields)
    Count
};

/// Fetch one of the object from \ref LazyImport
extern nb::handle lazy_import(LazyImport value);

/// Release every object cached by lazy_import().
extern void lazy_import_shutdown();

/// Extract the DRJIT_STRUCT element of a custom data structure type, if available
inline nb::dict get_drjit_struct(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(DRJIT_STRUCT), nb::handle());
    if (result.is_valid() && !result.type().is(&PyDict_Type))
        result = nb::object();
    return nb::borrow<nb::dict>(result);
}

/// Extract the dataclass fields element of a custom data structure type, if available
inline nb::object dataclass_fields(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(__dataclass_fields__), nb::handle());
    if (result.is_valid()) {
        result = lazy_import(LazyImport::DataclassesFields)(tp);

        // Resolve postponed (string) annotations only if present: the expensive
        // typing.get_type_hints() call is unnecessary for concrete field types.
        bool needs_hints = false;
        for (auto field : result) {
            if (nb::isinstance<nb::str>(field.attr(DR_STR(type)))) {
                needs_hints = true;
                break;
            }
        }

        if (needs_hints) {
            nb::object hints = lazy_import(LazyImport::TypingGetTypeHints)(tp);
            for (auto field : result) {
                if (field.attr(DR_STR(type)).type().is(&PyUnicode_Type))
                    field.attr(DR_STR(type)) = hints[field.attr(DR_STR(name))];
            }
        }
    }
    return result;
}

/// Retrieve the ``__dataclass_fields__`` of a dataclass type.
inline nb::dict dataclass_field_dict(nb::handle tp) {
    nb::object result = nb::getattr(tp, DR_STR(__dataclass_fields__), nb::handle());
    if (result.is_valid() && !result.type().is(&PyDict_Type))
        result = nb::object();
    return nb::borrow<nb::dict>(result);
}

/// Use this function to skip non-field entries returned by \ref dataclass_field_dict
inline bool is_dataclass_field(nb::handle field) {
    return nb::getattr(field, "_field_type", nb::handle())
        .is(lazy_import(LazyImport::DataclassesField));
}

/// Detect builtin scalar types (``float``, ``int``, ``bool``, ``str``, ``None``).
inline bool is_builtin_scalar(nb::handle tp) {
    PyTypeObject *t = (PyTypeObject *) tp.ptr();
    return t == &PyFloat_Type || t == &PyLong_Type || t == &PyBool_Type ||
           t == &PyUnicode_Type || t == Py_TYPE(Py_None);
}

/// Python equality that never raises. A failing comparison counts as unequal.
inline bool py_equal(nb::handle a, nb::handle b) {
    if (a.ptr() == b.ptr())
        return true;
    if (!a.is_valid() || !b.is_valid())
        return false;
    int rv = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
    if (rv == -1) {
        PyErr_Clear();
        return false;
    }
    return rv == 1;
}

/// Python type object of ``drjit::TraversableBase``, set by ``export_detail()``
extern nb::handle traversable_base_type;

/// Return a pointer to the underlying C++ class if the Python object inherits
/// from TraversableBase or null otherwise
inline drjit::TraversableBase *traversable_ptr(nb::handle h) {
    if (!PyType_IsSubtype(Py_TYPE(h.ptr()),
                          (PyTypeObject *) traversable_base_type.ptr()) ||
        !nb::inst_ready(h))
        return nullptr;
    return nb::inst_ptr<drjit::TraversableBase>(h);
}

/**
 * \brief Return the instance dictionary of a traversable object, if available.
 *
 * Given a ``drjit::TraversableBase``-derived instance, this function checks
 * if the type has been subclassed in Python. In that case, it returns the
 * instance's dictionary. Otherwise, it returns an invalid handle.
 */
inline nb::dict traversable_dict(const drjit::TraversableBase *obj) {
    nb::handle self = obj->self_py();
    if (!self.is_valid() || !NB_CALL(nb_inst_python_derived)(self.ptr()))
        return nb::steal<nb::dict>(nb::handle());
    return nb::inst_dict(self);
}

/**
 * \brief Run the traversal callback of a C++ object
 *
 * Calls ``var(index, name, variant, domain)`` for every JIT array held by
 * ``obj`` (the return value is the index that the object holds from now on,
 * see \ref drjit::TraverseVisitor) and ``child(obj, name)`` for every directly
 * held child object. This is the one place that turns the C callback interface
 * of ``traverse_cb()`` into lambdas; all drivers build on it.
 */
template <typename Var, typename Child>
void for_each_member(drjit::TraversableBase *obj, drjit::TraverseRole role,
                     Var &&var, Child &&child) {
    struct Payload {
        Var &var;
        Child &child;
    } p { var, child };

    obj->traverse_cb(
        &p,
        drjit::TraverseVisitor {
            role,
            [](void *p, uint64_t index, const char *name, const char *variant,
               const char *domain) -> uint64_t {
                return ((Payload *) p)->var(index, name, variant, domain);
            },
            [](void *p, drjit::TraversableBase *child, const char *name) {
                ((Payload *) p)->child(child, name);
            } });
}

/// Raise if the nanobind binding of type ``tp`` implements the traversal
/// interface without declaring drjit::TraversableBase as a base class
extern void raise_if_unbound_traversable(nb::handle tp);
