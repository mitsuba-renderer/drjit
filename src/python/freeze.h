/*
    freeze.h -- Bindings for drjit.freeze()

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"
#include "functional"

struct FrozenFunction;

extern FrozenFunction freeze(nb::callable);
extern void export_freeze(nb::module_ &);
