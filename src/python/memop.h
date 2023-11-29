/*
    memop.cpp -- Bindings for scatter/gather memory operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_memop(nb::module_ &);

extern nb::object gather(nb::type_object dtype, nb::object source,
                         nb::object index, nb::object active, bool permute);

extern void scatter(nb::object target, nb::object value, nb::object index,
                    nb::object active, bool permute = false);

extern nb::object ravel(nb::handle h, char order,
                        dr_vector<size_t> *shape_out = nullptr,
                        dr_vector<int64_t> *strides_out = nullptr,
                        const VarType *vt_in = nullptr);

extern nb::object unravel(const nb::type_object_t<ArrayBase> &dtype,
                          nb::handle_t<ArrayBase> array, char order);

extern nb::object scatter_inc(nb::handle_t<drjit::ArrayBase> target,
                              nb::object index, nb::object active);
