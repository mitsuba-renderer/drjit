/*
    autodiff.h -- Bindings for autodiff utility functions

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_autodiff(nb::module_ &);

extern nb::object grad(nb::handle h, bool preserve_type_ = true);
extern void set_grad(nb::handle target, nb::handle source);
