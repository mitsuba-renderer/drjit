/*
    slice.h -- implementation of drjit.slice_index() and
    map subscript operators

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_slice(nb::module_&);

extern std::pair<nb::tuple, nb::object>
slice_index(const nb::type_object_t<ArrayBase> &dtype,
            const nb::tuple &shape, const nb::tuple &indices);

extern PyObject *mp_subscript(PyObject *, PyObject *) noexcept;
extern int mp_ass_subscript(PyObject *, PyObject *, PyObject *) noexcept;
extern PyObject *sq_item_tensor(PyObject *, Py_ssize_t) noexcept;
extern int sq_ass_item_tensor(PyObject *, Py_ssize_t, PyObject *) noexcept;
