#include "test.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>
#include <enoki/vcall.h>
#include <enoki/loop.h>

namespace ek = enoki;

using Float  = ek::DiffArray<ek::CUDAArray<float>>;
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

ENOKI_TEST(test01_vcall_reduce_and_record) {
    jit_init((uint32_t) JitBackend::CUDA);
    // jit_set_flag(JitFlag::VCallOptimize, false);
    // jit_set_flag(JitFlag::VCallRecord, false);
    // jit_set_log_level_stderr(LogLevel::Trace);

    for (int i = 0; i < 2; ++i) {
        Float x = ek::arange<Float>(10);
        ek::enable_grad(x);
        ek::set_label(x, "x");

        Float y = x;

        if (i == 1)
            y = ek::gather<Float>(x, 9 - ek::arange<UInt32>(10));

        Class *b1 = new Class();
        Class *b2 = new Class();
        ClassPtr b2p(b2);

        b1->value = ek::zero<Float>(10);
        b2->value = y;

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
