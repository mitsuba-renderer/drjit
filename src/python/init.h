/*
    init.h -- Implementation of <Dr.Jit array>.__init__() and
    other initializion routines like dr.zero(), dr.empty(), etc.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once
#include "common.h"

extern int tp_init_array(PyObject *, PyObject *, PyObject *);
extern int tp_init_tensor(PyObject *, PyObject *, PyObject *);

extern void export_init(nb::module_ &);

extern nb::object arange(const nb::type_object_t<ArrayBase> &dtype,
                         Py_ssize_t start, Py_ssize_t end, Py_ssize_t step);

extern nb::object full(nb::handle dtype, nb::handle value, size_t ndim, const size_t *shape);
extern nb::object full(nb::handle dtype, nb::handle value, size_t size);
extern nb::object full(nb::handle dtype, nb::handle value, const std::vector<size_t> &shape);
