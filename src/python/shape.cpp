/*
    shape.cpp -- implementation of drjit.shape() and ArrayBase.__len__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "shape.h"

Py_ssize_t sq_length(PyObject *o) noexcept {
    const ArraySupplement &s =
        nb::type_supplement<ArraySupplement>(nb::handle(o).type());

    Py_ssize_t length = s.shape[0];
    if (s.is_tensor) {
        const dr::dr_vector<size_t> &shape = s.tensor_shape(nb::inst_ptr<dr::ArrayBase>(o));
        return shape.size() == 0 ? 0 : (Py_ssize_t) shape[0];
    } else if (length == DRJIT_DYNAMIC) {
        length = (Py_ssize_t) s.len(nb::inst_ptr<dr::ArrayBase>(o));
    }

    return length;
}

static bool shape_impl(nb::handle h, int i, size_t *shape) noexcept {
    if (i >= 4)
        nb::detail::fail("drjit.shape(): internal error!");

    const ArraySupplement &s = nb::type_supplement<ArraySupplement>(h.type());
    size_t size = s.shape[0], cur = shape[i];

    if (size == DRJIT_DYNAMIC)
        size = (Py_ssize_t) s.len(nb::inst_ptr<dr::ArrayBase>(h));

    if (size != cur) {
        if (size == 1 || cur == 1 || cur == 0)
            shape[i] = size > cur ? size : cur;
        else
            return false; // ragged array
    }

    if (s.ndim > 1) {
        for (size_t j = 0; j < size; ++j) {
            if (!shape_impl(h[j], i + 1, shape))
                return false;
        }
    }

    return true;
}

nb::object shape(nb::handle_t<dr::ArrayBase> h) {
    const size_t *shape_ptr = nullptr;
    size_t temp[4] {};
    size_t ndim = 0;

    const ArraySupplement &s = nb::type_supplement<ArraySupplement>(h.type());

    if (s.is_tensor) {
        const dr::dr_vector<size_t> &shape = s.tensor_shape(nb::inst_ptr<dr::ArrayBase>(h));
        shape_ptr = shape.data();
        ndim = shape.size();
    } else {
        shape_ptr = temp;
        ndim = s.ndim;

        if (!shape_impl(h, 0, temp))
            return nb::none();
    }

    nb::object result = nb::steal(PyTuple_New((Py_ssize_t) ndim));
    for (size_t i = 0; i < ndim; ++i)
        NB_TUPLE_SET_ITEM(result.ptr(), i, PyLong_FromSize_t(shape_ptr[i]));
    return result;
}

void export_shape(nb::module_ &m) {
    m.def("shape", &shape, nb::raw_doc(doc_shape));
}
