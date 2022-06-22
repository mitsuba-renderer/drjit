/*
    repr.cpp -- implementation of drjit.ArrayBase.__repr__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include "../ext/nanobind/src/buffer.h"
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
namespace dr = drjit;

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

            if (o.type() == nb::handle(&PyFloat_Type))
                buffer.fmt("%g", nb::cast<double>(o));
            else
                buffer.put_dstr(nb::str(o).c_str());
        }

        if (j + 1 < size) {
            if (i == 0) {
                buffer.put(", ");
            } else {
                buffer.put(",\n");
                buffer.put(' ', i);
            }
        }
    }

    buffer.put(']');
}

PyObject *tp_repr(PyObject *self) {
    try {
        (void) self;
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
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
}

