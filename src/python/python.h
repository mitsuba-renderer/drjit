#pragma once

#include <drjit/python.h>

namespace nb = nanobind;
namespace dr = drjit;

using meta = dr::detail::array_metadata;
using ops = dr::detail::array_ops;
using supp = dr::detail::array_supplement;

extern nb::handle array_base;
extern nb::handle array_type;

inline bool is_drjit_array(nb::handle h) {
    return PyType_IsSubtype((PyTypeObject *) h.type().ptr(),
                            (PyTypeObject *) array_base.ptr());
}

extern Py_ssize_t len(PyObject *o) noexcept;
extern nb::object shape(nb::handle_of<dr::ArrayBase> h) noexcept;
