/*
    iter.cpp -- Iterator implementation for Dr.Jit arrays

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "iter.h"
#include "shape.h"
#include "slice.h"

struct dr_iterator {
    nb::object o;
    const ArraySupplement &s;
    Py_ssize_t index, size;
};

PyObject *tp_iter(PyObject *o) {
    return nb::cast(dr_iterator{ nb::borrow(o), supp(Py_TYPE(o)), 0, sq_length(o) },
                    nb::rv_policy::move)
        .release()
        .ptr();
}

static PyObject *tp_iternext(PyObject *o) {
    dr_iterator &it = *nb::inst_ptr<dr_iterator>(o);
    if (it.index >= it.size)
        return nullptr;

    ssizeargfunc func = it.s.is_tensor ? sq_item_tensor : it.s.item;
    return func(it.o.ptr(), it.index++);
}

static int tp_traverse(PyObject *self, visitproc visit, void *arg) {
    dr_iterator &it = *(dr_iterator *) self;
    Py_VISIT(it.o.ptr());
    return 0;
}


void export_iter(nb::module_ &m) {
    const PyType_Slot iter_slots[] = {
        { Py_tp_traverse, (void *) tp_traverse },
        { Py_tp_iternext, (void *) tp_iternext },
        { 0, nullptr }
    };

    nb::class_<dr_iterator>(m, "dr_iterator", nb::type_slots(iter_slots));
}
