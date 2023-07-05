/*
    repr.cpp -- implementation of drjit.ArrayBase.__repr__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "repr.h"
#include "shape.h"
#include "eval.h"
#include "slice.h"
#include "../ext/nanobind/src/buffer.h"

static nanobind::detail::Buffer buffer;

// Skip output when there are more than 20 elements. Must be
// divisible by 4
static size_t repr_threshold = 20;

void tp_repr_impl(PyObject *self,
                  const dr_vector<size_t> &shape,
                  nb::list &index,
                  size_t depth) {
    const ArraySupplement &s = supp(Py_TYPE(self));

    // Reverse the dimensions of non-tensor shapes for convenience
    size_t i = s.is_tensor ? depth : (shape.size() - 1 - depth),
           size = shape.empty() ? 0 : shape[i];

    bool leaf = depth == shape.size() - 1;

    if ((s.is_complex || s.is_quaternion) && leaf) {
        bool prev = false;

        for (size_t j = 0; j < size; ++j) {
            PyObject *jo = PyLong_FromSize_t(j);
            raise_if(!jo, "Index creation failed.");
            raise_if(PyList_SetItem(index.ptr(), (Py_ssize_t) i, jo),
                    "Index assignment failed.");
            nb::tuple index_tuple(index);
            nb::object o = nb::steal(mp_subscript(self, index_tuple.ptr()));

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
        if (s.is_tensor && shape.empty()) {
            nb::object o = nb::steal(s.tensor_array(self))[0];

            if (PyFloat_CheckExact(o.ptr())) {
                double d = nb::cast<double>(o);
                buffer.fmt("%g", d);
            } else {
                buffer.put_dstr(nb::str(o).c_str());
            }

            return;
        }

        buffer.put('[');
        for (size_t j = 0; j < size; ++j) {
            PyObject *jo = PyLong_FromSize_t(j);
            raise_if(!jo, "Index creation failed.");
            raise_if(PyList_SetItem(index.ptr(), (Py_ssize_t) i, jo),
                    "Index assignment failed.");

            if (size >= repr_threshold && j * 4 == repr_threshold) {
                buffer.fmt(".. %zu skipped ..", size - repr_threshold / 2);
                j = size - repr_threshold / 4 - 1;
            } else if (!leaf) {
                tp_repr_impl(self, shape, index, depth + 1);
            } else {
                nb::tuple index_tuple(index);
                nb::object o = nb::steal(mp_subscript(self, index_tuple.ptr()));

                if (s.is_tensor)
                    o = nb::steal(s.tensor_array(o.ptr()))[0];

                if (PyFloat_CheckExact(o.ptr())) {
                    double d = nb::cast<double>(o);
                    buffer.fmt("%g", d);
                } else {
                    buffer.put_dstr(nb::str(o).c_str());
                }
            }

            if (j + 1 < size) {
                if (leaf) {
                    buffer.put(", ");
                } else {
                    buffer.put(",\n");
                    buffer.put(' ', shape.size() - 1);
                }
            }
        }
        buffer.put(']');
    }
}

PyObject *tp_repr(PyObject *self) noexcept {
    try {
        buffer.clear();
        schedule(self);

        dr_vector<size_t> shape;
        if (!shape_impl(self, shape)) {
            buffer.put("[ragged array]");
        } else {
            nb::list index =
                nb::steal<nb::list>(PyList_New((Py_ssize_t) shape.size()));
            if (!index.is_valid())
                nb::detail::raise_python_error();
            tp_repr_impl(self, shape, index, 0);
        }

        return PyUnicode_FromString(buffer.get());
    } catch (nb::python_error &e) {
        nb::str tp_name = nb::inst_name(self);
        e.restore();
        nb::chain_error(PyExc_RuntimeError, "%U.__repr__(): internal error.",
                        tp_name.ptr());
        return nullptr;
    } catch (const std::exception &e) {
        nb::str tp_name = nb::inst_name(self);
        PyErr_Format(PyExc_RuntimeError, "%U.__repr__(): %s", tp_name.ptr(), e.what());
        return nullptr;
    }
}
