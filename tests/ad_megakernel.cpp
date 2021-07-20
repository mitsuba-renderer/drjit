#include "test.h"
#include <enoki/jit.h>
#include <enoki/autodiff.h>
#include <enoki/vcall.h>
#include <enoki/loop.h>

namespace ek = enoki;

using Float  = ek::DiffArray<ek::LLVMArray<float>>;
using UInt32 = ek::uint32_array_t<Float>;

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


struct Base {
    Base() {
        x = 10.f;
        ek::enable_grad(x);
        ek::set_label(x, "Base::x");
    }
    virtual Float f(const Float &m) = 0;
    virtual Float g(const Float &m) = 0;
    ENOKI_VCALL_REGISTER(Float, Base)
    Float x;
};

using BasePtr = ek::replace_scalar_t<Float, Base *>;

struct A : Base {
    Float f(const Float &v) override {
        return x * v;
    }
    Float g(const Float &v) override {
        return v * 2;
    }
};

struct B : Base {
    Float f(const Float &v) override {
        return x * v * 2;
    }
    Float g(const Float &v) override {
        return v;
    }
};

ENOKI_VCALL_BEGIN(Base)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(g)
ENOKI_VCALL_END(Base)


ENOKI_TEST(test05_vcall_symbolic_ad_loop_opt) {
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    int n = 20;
    int max_depth = 5;

    // Compute result values
    float res_a = 0, res_b = 0, va = 1;
    for (size_t i = 0; i < max_depth; i++) {
        res_a += n * va;
        res_b += 2.f * n;
        va *= 2;
    }

    for (int i = 0; i <= 4; ++i) {
        jit_set_flag(JitFlag::VCallRecord,   i>0);
        jit_set_flag(JitFlag::VCallOptimize, i>1);
        jit_set_flag(JitFlag::LoopRecord,    i>2);
        jit_set_flag(JitFlag::LoopOptimize,  i>3);


        A *a = new A();
        B *b = new B();
        ek::mask_t<Float> m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtr arr = ek::select(m, (Base *) a, (Base *) b);
        ek::enable_grad(a->x);
        ek::enable_grad(b->x);

        UInt32 depth = 0;
        ek::mask_t<Float> active = ek::full<ek::mask_t<Float>>(true, n);
        Float unused = 0.f;

        {
            // This variable will be out of scope (only consumed by a side effect)
            Float value = 1.f;

            ek::Loop<Float> loop("MyLoop", active, depth, unused, value);
            while (loop(ek::detach(active))) {
                Float output = arr->f(2.f);

                ek::enqueue(output);
                ek::set_grad(output, value);
                ek::traverse<Float>(true, false);

                value = ek::detach(arr->g(value));

                depth++;
                active &= depth < max_depth;
            }
        }

        assert(ek::all_nested(
            ek::eq(ek::grad(a->x), res_a) &&
            ek::eq(ek::grad(b->x), res_b)));

        delete a;
        delete b;
    }
}


ENOKI_TEST(test06_vcall_symbolic_nested_ad_loop_opt) {
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);
    int n = 20;
    int max_depth = 5;
    jit_set_log_level_stderr(::LogLevel::InfoSym);
    jit_set_flag(JitFlag::VCallRecord,   true);
    jit_set_flag(JitFlag::VCallOptimize, true);
    jit_set_flag(JitFlag::LoopRecord,    true);
    jit_set_flag(JitFlag::LoopOptimize,  true);

    A *a = new A();
    B *b = new B();
    ek::mask_t<Float> m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
    BasePtr arr = ek::select(m, (Base *) a, (Base *) b);

    ek::enable_grad(a->x);
    ek::enable_grad(b->x);

    UInt32 depth = 0;
    ek::mask_t<Float> active = ek::full<ek::mask_t<Float>>(true, n);
    Float unused = 0.f;
    ek::Loop<Float> loop("outer", active, depth, unused);
    while (loop(ek::detach(active))) {
        UInt32 depth2 = 0;
        ek::mask_t<Float> active2 = ek::full<ek::mask_t<Float>>(true, n);
        ek::Loop<Float> loop2("inner", active2, depth2);
        while (loop2(ek::detach(active2))) {
            Float output = arr->f(2.f);
            ek::backward(output);
            depth2++;
            active2 &= depth2 < max_depth;
        }
        depth++;
        active &= depth < max_depth;
    }


    assert(ek::all_nested(
        ek::eq(ek::grad(a->x), max_depth * max_depth * n) &&
        ek::eq(ek::grad(b->x), 2.f * max_depth * max_depth * n)));

    delete a;
    delete b;
}

// 3. loop *and* vcall
// 6. test scatter
//  what if the instance gathers from some type that doesn't even matter?
