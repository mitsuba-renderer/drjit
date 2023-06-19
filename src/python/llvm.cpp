/*
    llvm.cpp -- instantiates the drjit.llvm.* namespace

    Dr.Jit: A Just-In-Time-Compiler for Differentiable Rendering
    Copyright 2022, Realistic Graphics Lab, EPFL.

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE.txt file.
*/

#include "llvm.h"

#if defined(DRJIT_ENABLE_LLVM)
void export_llvm() {
    ArrayBinding b;
    dr::bind_all<dr::LLVMArray<float>>(b);
}
#endif
