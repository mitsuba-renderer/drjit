/*
    scalar.cpp -- instantiates the drjit.scalar.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "scalar.h"

void export_scalar() {
    ArrayBinding b;
    dr::bind_all<float>(b);
}
