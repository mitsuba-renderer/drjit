/*
    cuda.cpp -- instantiates the drjit.cuda.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "cuda.h"
#include <drjit/autodiff.h>

#if defined(DRJIT_ENABLE_CUDA)
void export_cuda_ad(nb::module_ &m) {
    ArrayBinding b;
    dr::bind_all<dr::CUDADiffArray<float>>(b);
    m.attr("Float32") = m.attr("Float");
    m.attr("Int32") = m.attr("Int");
    m.attr("UInt32") = m.attr("UInt");
}
#endif
