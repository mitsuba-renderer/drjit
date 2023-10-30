/*
    cuda.cpp -- instantiates the drjit.cuda.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "cuda.h"
#include "random.h"

#if defined(DRJIT_ENABLE_CUDA)
void export_cuda(nb::module_ &m) {
    using Guide = dr::CUDAArray<float>;

    ArrayBinding b;
    dr::bind_all<Guide>(b);
    bind_pcg32<Guide>(m);

    m.attr("Float32") = m.attr("Float");
    m.attr("Int32") = m.attr("Int");
    m.attr("UInt32") = m.attr("UInt");
}
#endif
