#include "test.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>
#include <enoki/vcall.h>
#include <enoki/loop.h>

namespace ek = enoki;

using Float  = ek::DiffArray<ek::LLVMArray<float>>;
using UInt32 = ek::uint32_array_t<Float>;
using Mask   = ek::mask_t<Float>;

struct Class {
    Float value;
    Float f(UInt32 i) {
        return ek::sqr(ek::gather<Float>(value, i));
    }

    ENOKI_VCALL_REGISTER(Float, Class)
};

ENOKI_VCALL_BEGIN(Class)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_END(Class)

using ClassPtr = ek::replace_scalar_t<Float, Class *>;

ENOKI_TEST(test01_vcall_reduce_and_record_rev) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 3; ++j) {
        jit_set_flag(JitFlag::VCallOptimize, j == 2);
        jit_set_flag(JitFlag::VCallRecord, j >= 1);

        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 2; ++k) {
                Float x = ek::arange<Float>(10);
                ek::enable_grad(x);
                ek::set_label(x, "x");

                Float y = x;

                if (i == 1)
                    y = ek::gather<Float>(x, 9 - ek::arange<UInt32>(10));

                Class *b1 = new Class();
                Class *b2 = new Class();
                ClassPtr b2p(b2);
                if (k == 1)
                    b2p = ek::opaque<ClassPtr>(b2, 13);

                b1->value = ek::zero<Float>(10);
                b2->value = std::move(y);

                Float z = b2p->f(arange<UInt32>(13) % 10);
                ek::backward(z);

                if (i == 0)
                    assert(ek::grad(x) == Float(0, 4, 8, 6, 8, 10, 12, 14, 16, 18));
                else
                    assert(ek::grad(x) == Float(0, 2, 4, 6, 8, 10, 12, 28, 32, 36));

                delete b1;
                delete b2;
            }
        }
    }
}

ENOKI_TEST(test02_vcall_reduce_and_record_fwd) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 3; ++j) {
        jit_set_flag(JitFlag::VCallOptimize, j == 2);
        jit_set_flag(JitFlag::VCallRecord, j >= 1);

        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 2; ++k) {
                Float x = ek::arange<Float>(10);
                ek::enable_grad(x);
                ek::set_label(x, "x");

                Float y = x;

                if (i == 1)
                    y = ek::gather<Float>(x, 9 - ek::arange<UInt32>(10));

                Class *b1 = new Class();
                Class *b2 = new Class();
                ClassPtr b2p(b2);
                if (k == 1)
                    b2p = ek::opaque<ClassPtr>(b2, 13);

                b1->value = ek::zero<Float>(10);
                b2->value = std::move(y);

                Float z = b2p->f(arange<UInt32>(13) % 10);
                ek::forward(x);

                if (i == 0)
                     assert(ek::grad(z) == Float(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2, 4));
                else
                     assert(ek::grad(z) == Float(18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 18, 16, 14));

                delete b1;
                delete b2;
            }
        }
    }
}

ENOKI_TEST(test03_loop_rev_simple) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 2; ++j) {
        jit_set_flag(JitFlag::LoopRecord, j);

        UInt32 i = ek::arange<UInt32>(10);
        ek::Loop<Float> loop("MyLoop", i);

        Float x = ek::zero<Float>(11);
        ek::enable_grad(x);

        while (loop(i < 10)) {
            Float y = ek::gather<Float>(x, i);
            ek::backward(y);
            ++i;
        }

        assert(ek::grad(x) == Float(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0));
    }
}

ENOKI_TEST(test04_loop_rev_complex) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 2; ++j) {
        jit_set_flag(JitFlag::LoopRecord, j);

        UInt32 i = ek::arange<UInt32>(10);
        ek::Loop<Float> loop("MyLoop", i);

        Float x = ek::zero<Float>(11);
        ek::enable_grad(x);

        Float y = ek::gather<Float>(x, 10 - ek::arange<UInt32>(11));

        while (loop(i < 10)) {
            Float z = ek::gather<Float>(y, i);
            ek::backward(z, true);
            ++i;
        }

        assert(ek::grad(x) == Float(0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    }
}

// 3. loop *and* vcall
// 6. test scatter
//  what if the instance gathers from some type that doesn't even matter?
