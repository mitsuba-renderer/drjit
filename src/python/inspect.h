/*
    inspect.h -- operations to label Jit/AD graph nodes and
    visualize their structure using GraphViz.

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2023, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#pragma once

#include "common.h"

extern void export_inspect(nb::module_&);
extern void set_label(nb::handle h, nb::str label);
