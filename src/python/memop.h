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
                         nb::object index, nb::object active,
                         ReduceMode mode = ReduceMode::Auto,
                         nb::handle shape = nb::handle());

extern void scatter(nb::object target, nb::object value, nb::object index,
                    nb::object active, ReduceMode mode = ReduceMode::Auto);

extern void scatter_reduce(ReduceOp op, nb::object target, nb::object value,
                           nb::object index, nb::object active,
                           ReduceMode mode);

extern nb::object ravel(nb::handle h, char order,
                        vector<size_t> *shape_out = nullptr,
                        vector<int64_t> *strides_out = nullptr,
                        const VarType *vt_in = nullptr);

extern nb::object unravel(const nb::type_object_t<ArrayBase> &dtype,
                          nb::handle_t<ArrayBase> array, char order,
                          const vector<size_t> *shape_hint = nullptr);

extern nb::object scatter_inc(nb::handle_t<drjit::ArrayBase> target,
                              nb::object index, nb::object active);

extern nb::object slice(nb::handle value, nb::handle index);
