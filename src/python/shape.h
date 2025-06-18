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
extern Py_ssize_t mp_length(PyObject *o) noexcept;

extern nb::object shape(nb::handle h);
extern bool shape_impl(nb::handle h, vector<size_t> &result);

/// Return the number of dimensions of the given array/tensor
extern size_t ndim(nb::handle_t<ArrayBase> h) noexcept;

/// Return the vectorization width of the given input array or PyTree
extern size_t width(nb::handle h);

/// Same as \c width, but returns the width as an opaque array, allowing this
/// relationship to be recorded as part of a frozen function. Used in \c dr::mean.
extern nb::object opaque_width(nb::handle h);

/// Recursively traverses the PyTree of this object to compute the number of
/// elements. If a leaf object is a JIT array, the result will be an opaque
/// array.
extern nb::object opaque_n_elements(nb::handle h);

/// Convert vector<size_t> into a python tuple
extern nb::tuple cast_shape(const vector<size_t> &shape);

/// Publish the drjit.shape() function
extern void export_shape(nb::module_&);
