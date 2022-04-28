/*
    cuda.cpp -- instantiates the drjit.cuda.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "python.h"
#include "random.h"

void bind_cuda(nb::module_ &m) {
    dr::bind_all_types<dr::CUDAArray<float>>();
    bind_pcg32<dr::CUDAArray<float>>(m);
}
