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
        // a1: x
        // a6: gather
        // a7: mul
        // a8: result
        return ek::sqr(ek::gather<Float>(value, i));
    }

    ENOKI_VCALL_REGISTER(Float, Class)
};

ENOKI_VCALL_BEGIN(Class)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_END(Class)

using ClassPtr = ek::replace_scalar_t<Float, Class *>;

ENOKI_TEST(test01_vcall_reduce_and_record) {
    jit_init((uint32_t) JitBackend::LLVM);

    // a1
    Float x = ek::arange<Float>(10);
    ek::enable_grad(x);
    ek::set_label(x, "x");
    jit_set_log_level_stderr(LogLevel::Trace);

    // Float y = ek::gather<Float>(x, 9 - ek::arange<UInt32>(10));

    Class *b1 = new Class();
    Class *b2 = new Class();
    ClassPtr b2p(b2);

    b1->value = ek::zero<Float>(10);
    b2->value = x;

    // a4
    Float z = b2p->f(arange<UInt32>(10) % 10);
    std::cout << ek::graphviz(z) << std::endl;
    ek::set_label(z, "z");
    ek::backward(z);

    std::cout << ek::grad(x) << std::endl;
    std::cout << ek::grad(b2->value) << std::endl;
    assert(ek::grad(x) == Float(0, 2, 4, 6, 8, 10, 12, 14, 16, 18));

    delete b1;
    delete b2;
}
