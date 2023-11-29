/*
    format.h -- implementation of drjit.format(), drjit.print(),
    and ArrayBase.__repr__().

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_format(nb::module_&);

extern PyObject *tp_repr(PyObject *self) noexcept;
