/*
    bind.h -- Central bind() function used to publish Dr.Jit type bindings

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

/// Publish a Dr.Jit type binding in Python
extern nb::object bind(const ArrayBinding &b);

/// Expose the bind() function in Python as well
extern void export_bind(nb::module_&);
