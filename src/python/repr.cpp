/*
    repr.cpp -- implementation of drjit.ArrayBase.__repr__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "repr.h"
#include "shape.h"
#include "../ext/nanobind/src/buffer.h"
#include <nanobind/stl/vector.h>

static nanobind::detail::Buffer buffer;

// Skip output when there are more than 20 elements. Must be
// divisible by 4
static size_t repr_threshold = 20;

void tp_repr_impl(PyObject *self,
                  const std::vector<size_t> &shape,
                  std::vector<size_t> &index,
                  size_t depth) {
    size_t i = index.size() - 1 - depth,
           size = shape.empty() ? 0 : shape[i];

    const ArraySupplement &s =
        nb::type_supplement<ArraySupplement>(Py_TYPE(self));

    if ((s.is_complex || s.is_quaternion) && i == 0) {
        bool prev = false;

        for (size_t j = 0; j < size; ++j) {
            index[i] = j;

            nb::object o = nb::borrow(self);

            for (size_t k = 0; k < index.size(); ++k)
                o = o[index[k]];

            double d = nb::cast<double>(o);
            if (d == 0)
                continue;

            if (prev || d < 0)
                buffer.put(d < 0 ? "-" : "+");
            buffer.fmt("%g", fabs(d));
            prev = true;

            if (s.is_complex && j == 1)
                buffer.put('j');
            else if (s.is_quaternion && j < 3)
                buffer.put("ijk"[j]);
        }
        if (!prev)
            buffer.put("0");
    } else {
        buffer.put('[');
        for (size_t j = 0; j < size; ++j) {
            index[i] = j;

            if (size >= repr_threshold && j * 4 == repr_threshold) {
                buffer.fmt(".. %zu skipped ..", size - repr_threshold / 2);
                j = size - repr_threshold / 4 - 1;
            } else if (i > 0) {
                tp_repr_impl(self, shape, index, depth + 1);
            } else {
                nb::object o = nb::borrow(self);

                for (size_t k = 0; k < index.size(); ++k)
                    o = o[index[k]];

                if (PyFloat_CheckExact(o.ptr())) {
                    double d = nb::cast<double>(o);
                    buffer.fmt("%g", d);
                } else {
                    buffer.put_dstr(nb::str(o).c_str());
                }
            }

            if (j + 1 < size) {
                if (i == 0) {
                    buffer.put(", ");
                } else {
                    buffer.put(",\n");
                    buffer.put(' ', index.size() - 1);
                }
            }
        }
        buffer.put(']');
    }


}

PyObject *tp_repr(PyObject *self) noexcept {
    try {
        buffer.clear();

        nb::object shape_obj = shape(self);
        if (shape_obj.is_none()) {
            buffer.put("[ragged array]");
        } else {
            std::vector<size_t> shape = nb::cast<std::vector<size_t>>(shape_obj),
                                index(shape.size(), 0);
            tp_repr_impl(self, shape, index, 0);
        }

        return PyUnicode_FromString(buffer.get());
    } catch (const std::exception &e) {
        nb::str tp_name = nb::inst_name(self);
        PyErr_Format(PyExc_RuntimeError, "%U.__repr__(): %s", tp_name.ptr(), e.what());
        return nullptr;
    }
}
