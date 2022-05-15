/*
    shape.cpp -- implementation of drjit.shape() and ArrayBase.__len__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/


#include "python.h"

Py_ssize_t len(PyObject *o) noexcept {
    PyTypeObject *tp = Py_TYPE(o);
    const supp &s = nb::type_supplement<supp>(tp);
    Py_ssize_t length = s.meta.shape[0];

    if (length == DRJIT_DYNAMIC)
        length = (Py_ssize_t) s.len(nb::inst_ptr<void>(o));

    return length;
}

bool shape_impl(nb::handle h, int i, Py_ssize_t *shape) noexcept {
    if (i >= 4)
        nb::detail::fail("drjit.shape(): internal error!");
    nb::handle tp = h.type();

    const supp &s = nb::type_supplement<supp>(tp);
    Py_ssize_t size = s.meta.shape[0], cur = shape[i];

    if (size == DRJIT_DYNAMIC)
        size = (Py_ssize_t) s.len(nb::inst_ptr<void>(h));

    Py_ssize_t max_size = size > cur ? size : cur;
    if (max_size != size && size != 1)
        return false;

    shape[i] = max_size;

    if (s.meta.shape[1]) {
        auto sq_item = ((PyTypeObject *) tp.ptr())->tp_as_sequence->sq_item;

        for (Py_ssize_t j = 0; j < size; ++j) {
            PyObject *o = sq_item(h.ptr(), j);

            if (!shape_impl(o, i + 1, shape)) {
                Py_DECREF(o);
                return false;
            }

            Py_DECREF(o);
        }
    }

    return true;
}

nb::object shape(nb::handle_t<dr::ArrayBase> h) noexcept {
    const supp &s = nb::type_supplement<supp>(h.type());
    PyObject *result;

    if (!s.meta.is_tensor) {
        Py_ssize_t shape[4] { -1, -1, -1, -1 };
        if (!shape_impl(h, 0, shape))
            return nb::none();

        result = PyTuple_New(s.meta.ndim);
        for (Py_ssize_t i = 0; i < s.meta.ndim; ++i)
            PyTuple_SET_ITEM(result, i, PyLong_FromSize_t(shape[i] < 0 ? 0 : shape[i]));

        return nb::steal(result);
    } else {
        const drjit::dr_vector<size_t> &shape =
            s.op_tensor_shape(nb::inst_ptr<void>(h));

        result = PyTuple_New((Py_ssize_t) shape.size());
        for (size_t i = 0; i < shape.size(); ++i)
            PyTuple_SET_ITEM(result, i, PyLong_FromSize_t(shape[i]));
    }

    return nb::steal(result);
}
