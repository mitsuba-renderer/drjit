/*
    tests/loop.cpp -- tests loop infrastructure

    Enoki is a C++ template library that enables transparent vectorization
    of numerical kernels using SIMD instruction sets available on current
    processor architectures.

    Copyright (c) 2020 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a BSD-style
    license that can be found in the LICENSE file.
*/

#include "test.h"
#include <enoki/loop.h>
#include <enoki/jit.h>

#define ITERATIONS 1000000

ENOKI_TEST(test01_record_loop) {
    jit_init((uint32_t) JitBackend::LLVM);

    using Mask   = enoki::LLVMArray<bool>;
    using Float  = enoki::LLVMArray<float>;
    using UInt32 = enoki::LLVMArray<uint32_t>;

    // Tests a simple loop evaluated at once, or in parts
    for (uint32_t i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::LoopRecord, i != 0);
        jit_set_flag(JitFlag::LoopOptimize, i == 2);

        for (uint32_t j = 0; j < 2; ++j) {
            UInt32 x = arange<UInt32>(10);
            Float y = zero<Float>(1);
            Float z = 1;

            Loop<Mask> loop("MyLoop", x, y, z);
            loop.init();
            while (loop.cond(x < 5)) {
                y += Float(x);
                x += 1;
                z += 1;
            }

            if (j == 0) {
                jit_var_schedule(x.index());
                jit_var_schedule(y.index());
                jit_var_schedule(z.index());
            }

            assert(strcmp(z.str(), "[6, 5, 4, 3, 2, 1, 1, 1, 1, 1]") == 0);
            assert(strcmp(y.str(), "[10, 10, 9, 7, 4, 0, 0, 0, 0, 0]") == 0);
            assert(strcmp(x.str(), "[5, 5, 5, 5, 5, 5, 6, 7, 8, 9]") == 0);
        }
    }

    jit_shutdown();
}