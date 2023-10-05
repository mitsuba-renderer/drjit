/*
    cuda.cpp -- instantiates the drjit.cuda.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "cuda.h"

#if defined(DRJIT_ENABLE_CUDA)
void export_cuda() {
    ArrayBinding b;
    dr::bind_all<dr::CUDAArray<float>>(b);
}
#endif
