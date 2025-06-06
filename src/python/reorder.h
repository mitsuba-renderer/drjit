/*
    reorder.h -- Bindings for drjit.reorder_threads()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern nb::object reorder_threads(nb::handle_t<dr::ArrayBase>, int, nb::handle);
extern void export_reorder(nb::module_ &);
