#include "test.h"
#include <drjit/jit.h>
#include <drjit/autodiff.h>
#include <drjit/vcall.h>
#include <drjit/loop.h>

namespace dr = drjit;

using Float  = dr::DiffArray<dr::LLVMArray<float>>;
using UInt32 = dr::uint32_array_t<Float>;
using FMask  = dr::mask_t<Float>;

struct Test {
    Float value, value_2;
    Float f(UInt32 i) {
        return dr::sqr(dr::gather<Float>(value, i));
    }
    Float f2(UInt32 i) {
        dr::gather<Float>(value_2, i); // unused
        return f(i);
    }

    DRJIT_VCALL_REGISTER(Float, Test)
};

DRJIT_VCALL_BEGIN(Test)
DRJIT_VCALL_METHOD(f)
DRJIT_VCALL_METHOD(f2)
DRJIT_VCALL_END(Test)

using TestPtr = dr::replace_scalar_t<Float, Test *>;

DRJIT_TEST(test01_vcall_reduce_and_record_bwd) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 3; ++j) {
        jit_set_flag(JitFlag::VCallOptimize, j == 2);
        jit_set_flag(JitFlag::VCallRecord, j >= 1);

        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 2; ++k) {
                Float x = dr::arange<Float>(10);
                dr::enable_grad(x);
                dr::set_label(x, "x");

                Float y = x;

                if (i == 1)
                    y = dr::gather<Float>(x, 9 - dr::arange<UInt32>(10));

                Test *b1 = new Test();
                Test *b2 = new Test();
                TestPtr b2p(b2);
                if (k == 1)
                    b2p = dr::opaque<TestPtr>(b2, 13);

                b1->value = dr::zeros<Float>(10);
                b2->value = std::move(y);

                Float z = b2p->f(arange<UInt32>(13) % 10);
                dr::backward(z);

                if (i == 0)
                    assert(dr::grad(x) == Float(0, 4, 8, 6, 8, 10, 12, 14, 16, 18));
                else
                    assert(dr::grad(x) == Float(0, 2, 4, 6, 8, 10, 12, 28, 32, 36));

                delete b1;
                delete b2;
            }
        }
    }
}

DRJIT_TEST(test02_vcall_reduce_and_record_fwd) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 3; ++j) {
        jit_set_flag(JitFlag::VCallOptimize, j == 2);
        jit_set_flag(JitFlag::VCallRecord, j >= 1);

        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < 2; ++k) {
                Float x = dr::arange<Float>(10);
                dr::enable_grad(x);
                dr::set_label(x, "x");

                Float y = x;

                if (i == 1)
                    y = dr::gather<Float>(x, 9 - dr::arange<UInt32>(10));
                dr::set_label(y, "y");

                Test *b1 = new Test();
                Test *b2 = new Test();
                TestPtr b2p(b2);
                if (k == 1)
                    b2p = dr::opaque<TestPtr>(b2, 13);

                b1->value = dr::zeros<Float>(10);
                b2->value = std::move(y);

                Float z = b2p->f(arange<UInt32>(13) % 10);
                dr::forward(x);

                if (i == 0)
                     assert(dr::grad(z) == Float(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 0, 2, 4));
                else
                     assert(dr::grad(z) == Float(18, 16, 14, 12, 10, 8, 6, 4, 2, 0, 18, 16, 14));

                delete b1;
                delete b2;
            }
        }
    }
}

DRJIT_TEST(test03_loop_bwd_simple) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 2; ++j) {
        jit_set_flag(JitFlag::LoopRecord, j);

        UInt32 i = dr::arange<UInt32>(10);
        dr::Loop<FMask> loop("MyLoop", i);

        Float x = dr::zeros<Float>(11);
        dr::enable_grad(x);

        while (loop(i < 10)) {
            Float y = dr::gather<Float>(x, i);
            dr::backward(y);
            ++i;
        }

        assert(dr::grad(x) == Float(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0));
    }
}

DRJIT_TEST(test04_loop_bwd_complex) {
    jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 0; j < 2; ++j) {
        jit_set_flag(JitFlag::LoopRecord, j);

        UInt32 i = dr::arange<UInt32>(10);

        Float x = dr::zeros<Float>(11);
        dr::enable_grad(x);

        Float y = dr::gather<Float>(x, 10 - dr::arange<UInt32>(11));

        dr::Loop<FMask> loop("MyLoop", i);
        while (loop(i < 10)) {
            Float z = dr::gather<Float>(y, i);
            dr::backward(z, (uint32_t) dr::ADFlag::ClearVertices);
            ++i;
        }

        assert(dr::grad(x) == Float(0, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1));
    }
}

struct Base {
    Base() {
        x = 10.f;
        dr::enable_grad(x);
        dr::set_label(x, "Base::x");
    }
    virtual ~Base() { }
    virtual Float f(const Float &m) = 0;
    virtual Float g(const Float &m) = 0;
    DRJIT_VCALL_REGISTER(Float, Base)
    Float x;
};

using BasePtr = dr::replace_scalar_t<Float, Base *>;

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

DRJIT_VCALL_BEGIN(Base)
DRJIT_VCALL_METHOD(f)
DRJIT_VCALL_METHOD(g)
DRJIT_VCALL_END(Base)


DRJIT_TEST(test05_vcall_symbolic_ad_loop_opt) {
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    int n = 20;
    size_t max_depth = 5;

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
        FMask m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
        BasePtr arr = dr::select(m, (Base *) a, (Base *) b);
        dr::enable_grad(a->x);
        dr::enable_grad(b->x);

        UInt32 depth = 0;
        FMask active = dr::full<FMask>(true, n);
        Float unused = 0.f;

        {
            // This variable will be out of scope (only consumed by a side effect)
            Float value = 1.f;

            dr::Loop<FMask> loop("MyLoop", active, depth, unused, value);
            while (loop(dr::detach(active))) {
                Float output = arr->f(2.f);

                dr::enqueue(ADMode::Backward, output);
                dr::set_grad(output, value);
                dr::traverse<Float>(ADMode::Backward);

                value = dr::detach(arr->g(value));

                depth++;
                active &= depth < max_depth;
            }
        }

        assert(dr::all_nested(
            dr::eq(dr::grad(a->x), res_a) &&
            dr::eq(dr::grad(b->x), res_b)));

        delete a;
        delete b;
    }
}

DRJIT_TEST(test06_vcall_symbolic_nested_ad_loop_opt) {
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);
    int n = 20;
    int max_depth = 5;
    jit_set_flag(JitFlag::VCallRecord,   true);
    jit_set_flag(JitFlag::VCallOptimize, true);
    jit_set_flag(JitFlag::LoopRecord,    true);
    jit_set_flag(JitFlag::LoopOptimize,  true);

    A *a = new A();
    B *b = new B();
    FMask m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
    BasePtr arr = dr::select(m, (Base *) a, (Base *) b);

    dr::enable_grad(a->x);
    dr::enable_grad(b->x);

    UInt32 depth = 0;
    FMask active = dr::full<FMask>(true, n);
    Float unused = 0.f;
    dr::Loop<FMask> loop("outer", active, depth, unused);
    while (loop(dr::detach(active))) {
        UInt32 depth2 = 0;
        FMask active2 = dr::full<FMask>(true, n);
        dr::Loop<FMask> loop2("inner", active2, depth2);
        while (loop2(dr::detach(active2))) {
            Float output = arr->f(2.f);
            dr::backward(output);
            depth2++;
            active2 &= depth2 < max_depth;
        }
        depth++;
        active &= depth < max_depth;
    }


    assert(dr::all_nested(
        dr::eq(dr::grad(a->x), max_depth * max_depth * n) &&
        dr::eq(dr::grad(b->x), 2.f * max_depth * max_depth * n)));

    delete a;
    delete b;
}

DRJIT_TEST(test07_vcall_within_loop_postpone_bwd) {
    /// postponing of AD edges across vcalls/loops, faux dependencies

    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int j = 2; j < 3; ++j) {
        fprintf(stderr, "-------------------------------\nIteration %i\n", j);
        jit_set_flag(JitFlag::VCallOptimize, j == 2);
        jit_set_flag(JitFlag::LoopOptimize, j == 2);
        jit_set_flag(JitFlag::VCallRecord, j >= 1);
        jit_set_flag(JitFlag::LoopRecord, j >= 1);

        Float x = dr::arange<Float>(10);
        dr::enable_grad(x);
        dr::set_label(x, "x");

        Float y = dr::gather<Float>(x, 9 - dr::arange<UInt32>(10));
        dr::set_label(y, "y");

        Float q = x + 1;
        dr::set_label(q, "q");

        Test *b1 = new Test();
        Test *b2 = new Test();

        b1->value = 0;
        b1->value_2 = 0;
        b2->value = std::move(y);
        b2->value_2 = q;

        TestPtr b2p(b2);

        UInt32 i = dr::full<UInt32>(0, 13);
        dr::Loop<FMask> loop("MyLoop", i);
        while (loop(i < 10)) {
            Float z = b2p->f2(arange<UInt32>(13) % 10);
            dr::backward(z, (uint32_t) dr::ADFlag::ClearVertices);
            i++;
        }

        assert(dr::grad(x) == Float(0, 2, 4, 6, 8, 10, 12, 28, 32, 36)*10);

        delete b1;
        delete b2;
    }
}
