/*
    misc.h -- Bindings for miscellaneous implementation details

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/


#pragma once

#include "common.h"

extern void export_misc(nb::module_ &);

// See misc.cpp for documentation of these functions
extern void collect_indices(nb::handle, dr::dr_vector<uint64_t> &,
                            bool inc_ref = false);
extern dr::dr_vector<uint64_t> collect_indices(nb::handle);
extern nb::object update_indices(nb::handle, const dr::dr_vector<uint64_t> &);
extern void check_compatibility(nb::handle, nb::handle, const char *name);
extern nb::object copy(nb::handle h);
extern nb::object uncopy(nb::handle arg0, nb::handle arg1);
extern nb::object reset(nb::handle h);
extern void enable_py_tracing();
extern void disable_py_tracing();

