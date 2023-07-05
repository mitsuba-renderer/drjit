/*
    reduce.h -- Bindings for horizontal reduction operations

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_reduce(nb::module_ &);

extern nb::object all(nb::handle h);
extern nb::object any(nb::handle h);
extern nb::object all_nested(nb::handle h);
extern nb::object any_nested(nb::handle h);
