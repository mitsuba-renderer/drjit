/*
    shape.h -- implementation of drjit.shape() and ArrayBase.__len__()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern Py_ssize_t sq_length(PyObject *o) noexcept;

extern nb::object shape(nb::handle_t<dr::ArrayBase> h);

/// Publish the drjit.shape() function
extern void export_shape(nb::module_&);
