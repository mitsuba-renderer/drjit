/*
    shape.cpp -- implementation of drjit.shape() and ArrayBase.__len__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "shape.h"
#include "apply.h"
#include "base.h"
#include "meta.h"
#include <drjit-core/jit.h>

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
        bool ragged = false;
        size_t width = 0, items = 0;

        void operator()(nb::handle h) override {
            size_t value = len(h);
            if (items++ == 0)
                width = value;
            else if (width != 1 && value != 1 && width != value)
                ragged = true;
            if (value > width)
                width = value;
        }

        void traverse_unknown(nb::handle) override {
            if (width == 0)
                width = 1;
            items++;
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


nb::object opaque_width(nb::handle h) {
    struct TraverseOp : TraverseCallback {
        bool ragged = false;
        size_t width = 0, items = 0;
        ArrayMeta meta;
        uint64_t index;

        void operator()(nb::handle h) override {
            nb::handle tp = h.type();
            const ArraySupplement &s = supp(tp);

            if (s.index) {
                index = s.index(inst_ptr(h));
                meta  = supp(tp);
            }

            size_t value = s.len(inst_ptr(h));
            if (items++ == 0)
                width = value;
            else if (width != 1 && value != 1 && width != value)
                ragged = true;
            if (value > width)
                width = value;
        }

        void traverse_unknown(nb::handle) override {
            if (width == 0)
                width = 1;
            items++;
        }
    };

    TraverseOp to;
    traverse("drjit.opaque_width", to, h);
    if (to.ragged)
        nb::raise("drjit.opaque_width(): the input is ragged (i.e., it does not have a consistent size).");

    uint32_t opaque_width = jit_var_opaque_width((uint32_t) to.index);

    ArrayMeta meta = to.meta;
    meta.type      = (uint16_t) VarType::UInt32;

    nb::handle width_tp = meta_get_type(meta);
    const ArraySupplement width_s = supp(width_tp);

    if (!width_s.init_index)
        nb::raise("drjit.opaque_width(): unsupported dtype.");

    nb::object width = nb::inst_alloc(width_tp);
    width_s.init_index(opaque_width, inst_ptr(width));
    nb::inst_mark_ready(width);

    jit_var_dec_ref(opaque_width);

    return width;
}

/// Same as \c width, but returns the width as an opaque array, allowing this
/// relationship to be recorded as part of a frozen function. Used in \c dr::mean.
extern nb::object opaque_width(nb::handle h);


/// Recursively traverses the PyTree of this object to compute the number of
/// elements. If a leaf object is a JIT array, the result will be an opaque
/// array.
nb::object opaque_n_elements(nb::handle h) {
    nb::handle tp = h.type();

    // We use dr::shape() to test for ragged arrays
    auto sh = shape(h);

    if (is_drjit_type(tp)) {

        const ArraySupplement &s = supp(tp);

        if (s.is_tensor)
            return opaque_n_elements(nb::steal(s.tensor_array(h.ptr())));

        if (!s.index)
            jit_raise("opaque_n_lements(): Could not find indexing function");

        uint32_t index = (uint32_t) s.index(inst_ptr(h));

        // Construct the opaque_width python object
        uint32_t opaque_width = jit_var_opaque_width(index);

        ArrayMeta meta = s;
        meta.type = (uint16_t) VarType::UInt32;
        nb::handle width_tp = meta_get_type(meta);
        const ArraySupplement width_s = supp(width_tp);

        nb::object width = nb::inst_alloc(width_tp);
        width_s.init_index(opaque_width, inst_ptr(width));
        nb::inst_mark_ready(width);

        jit_var_dec_ref(opaque_width);

        return width;
    } else {
        Py_ssize_t rv = PyObject_Length(h.ptr());

        return opaque_n_elements(h[0]) * nb::int_(rv);
    }
}

void export_shape(nb::module_ &m) {
    m.def("shape", &shape, doc_shape, nb::sig("def shape(arg: object) -> tuple[int, ...]"));
    m.def("width", &width, doc_width)
     .def("width", [](nb::args args) { return width(args); });
    m.def("opaque_width", &opaque_width);
}
