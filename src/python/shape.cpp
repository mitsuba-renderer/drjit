/*
    shape.cpp -- implementation of drjit.shape() and ArrayBase.__len__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "shape.h"
#include "base.h"
#include "apply.h"

Py_ssize_t sq_length(PyObject *o) noexcept {
    const ArraySupplement &s = supp(Py_TYPE(o));

    Py_ssize_t length = s.shape[0];
    if (s.is_tensor) {
        const vector<size_t> &shape = s.tensor_shape(inst_ptr(o));
        return shape.size() == 0 ? 0 : (Py_ssize_t) shape[0];
    } else if (length == DRJIT_DYNAMIC) {
        length = (Py_ssize_t) s.len(inst_ptr(o));
    }

    return length;
}

Py_ssize_t mp_length(PyObject *o) noexcept {
    return sq_length(o);
}

size_t ndim(nb::handle_t<ArrayBase> h) noexcept {
    const ArraySupplement &s = supp(h.type());
    if (s.is_tensor)
        return s.tensor_shape(inst_ptr(h)).size();
    else
        return s.ndim;
}

static bool shape_traverse(nb::handle h, size_t ndim, size_t *shape) {
    nb::handle tp = h.type();

    size_t size;
    bool recurse = true;

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if (s.ndim == 0)
            nb::raise("drjit.shape(): unsupported type in input!");

        size = s.shape[0];

        if (size == DRJIT_DYNAMIC) {
            size = s.len(inst_ptr(h));

            // Don't recurse down to the scalar level for performance reasons
            if (ndim == 1)
                recurse = false;
        }
    } else {
        Py_ssize_t rv = PyObject_Length(h.ptr());

        if (rv < 0) {
            PyErr_Clear();
            return ndim == 0;
        } else {
            if (ndim == 0)
                return false;
            size = (size_t) rv;
        }
    }

    size_t cur = *shape;
    if (size != cur) {
        if (size == 1 || cur == 1 || cur == 0)
            *shape = size > cur ? size : cur;
        else
            return false; // ragged array
    }

    if (recurse) {
        for (size_t j = 0; j < size; ++j) {
            nb::object o = h[j];
            if (!shape_traverse(o, ndim - 1, shape + 1))
                return false;
        }
    }

    return true;
}

bool shape_impl(nb::handle h, vector<size_t> &result) {
    nb::handle tp = h.type();

    if (is_drjit_type(tp)) {
        const ArraySupplement &s = supp(tp);

        if (s.is_tensor) {
            result = s.tensor_shape(inst_ptr(h));
            return true;
        }
        result = vector<size_t>(s.ndim, 0);
    } else {
        nb::object o = nb::borrow(h);
        size_t ndim = 0;

        while (true) {
            Py_ssize_t rv = PyObject_Length(o.ptr());
            if (rv < 0) {
                PyErr_Clear();
                break;
            }
            ndim++;
            if (rv == 0)
                break;
            o = o[0];
        }

        result = vector<size_t>(ndim, 0);
    }

    return shape_traverse(h, result.size(), result.data());
}

nb::tuple cast_shape(const vector<size_t> &shape) {
    nb::tuple o = nb::steal<nb::tuple>(PyTuple_New((Py_ssize_t) shape.size()));
    if (!o.is_valid())
        nb::raise_python_error();

    for (size_t i = 0; i < shape.size(); ++i) {
        PyObject *l = PyLong_FromSize_t(shape[i]);
        if (!l)
            nb::raise_python_error();
        NB_TUPLE_SET_ITEM(o.ptr(), (Py_ssize_t) i, l);
    }

    return o;
}

nb::object shape(nb::handle h) {
    vector<size_t> result;

    if (!shape_impl(h, result))
        nb::raise("drjit.shape(): the input is ragged (i.e., it does not have a consistent size).");

    return cast_shape(result);
}

size_t width(nb::handle h) {
    struct TraverseOp : TraverseCallback {
        size_t width = 1;
        bool ragged = false;

        void operator()(nb::handle h) override {
            size_t value = len(h);
            if (width != 1 && value != 1 && width != value)
                ragged = true;
            if (value > width)
                width = value;
        }
    };

    TraverseOp to;
    traverse("drjit.width", to, h);
    if (to.ragged)
        nb::raise("drjit.width(): the input is ragged (i.e., it does not have a consistent size).");

    return to.width;
}


/// Return the vectorization width of the given input array or PyTree
extern size_t width(nb::handle h);

void export_shape(nb::module_ &m) {
    m.def("shape", &shape, doc_shape, nb::sig("def shape(arg: object) -> tuple[int, ...]"));
    m.def("width", &width, doc_width)
     .def("width", [](nb::args args) { return width(args); });
}
