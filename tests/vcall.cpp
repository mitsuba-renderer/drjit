#include "test.h"
#include <enoki/vcall.h>
#include <enoki/jit.h>
#include <enoki/autodiff.h>
#include <enoki/struct.h>
#include <enoki/loop.h>

namespace ek = enoki;

template <typename T> struct Struct {
    T a, b;
    Struct(const T &a, const T &b) : a(a), b(b) { }
    ENOKI_STRUCT(Struct, a, b)
};

// using Float = ek::CUDAArray<float>;
using Float = ek::LLVMArray<float>;

using FloatD   = ek::DiffArray<Float>;
using UInt32   = ek::uint32_array_t<Float>;
using UInt32D  = ek::DiffArray<UInt32>;
using Mask     = ek::mask_t<Float>;
using MaskD    = ek::mask_t<FloatD>;
using Array2f  = ek::Array<Float, 2>;
using Array2fD = ek::Array<FloatD, 2>;
using Array3f  = ek::Array<Float, 3>;
using Array3fD = ek::Array<FloatD, 3>;
using StructF  = Struct<Array3f>;
using StructFD = Struct<Array3fD>;

struct Base {
    Base(bool scalar) : x(ek::opaque<Float>(10, scalar ? 1 : 10)) { }
    virtual ~Base() { }

    virtual StructF f(const StructF &m, ::Mask active = true) = 0;

    virtual void side_effect() {
        ek::scatter(x, Float(-10), UInt32(0));
    }

    UInt32 strlen(const std::string &string) {
        return string.length();
    }

    float field() const { return 1.2f; };
    ENOKI_VCALL_REGISTER(Float, Base)

protected:
    Float x;
};

using BasePtr = ek::replace_scalar_t<Float, Base *>;

struct A : Base {
    A(bool scalar) : Base(scalar) { ek::set_attr(this, "field", 2.4f); }
    StructF f(const StructF &m, ::Mask /* active */ = true) override {
        if (x.size() == 1)
            return Struct(m.a * x, m.b * 15);
        else
            return Struct(m.a * ek::gather<Float>(x, UInt32(0)), m.b * 15);
    }
};

struct B : Base {
    B(bool scalar) : Base(scalar) { ek::set_attr(this, "field", 4.8f); }
    StructF f(const StructF &m, ::Mask /* active */ = true) override {
        if (x.size() == 1)
            return Struct(m.b * 20, m.a * x);
        else
            return Struct(m.b * 20, m.a * ek::gather<Float>(x, UInt32(0)));
    }
};

ENOKI_VCALL_BEGIN(Base)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(side_effect)
ENOKI_VCALL_METHOD(strlen)
ENOKI_VCALL_GETTER(field, float)
ENOKI_VCALL_END(Base)

ENOKI_TEST(test01_vcall_reduce_and_record) {
    int n = 9999;

    // jit_set_log_level_stderr(::LogLevel::Error);
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        for (int j = 0; j < 2; ++j) {
            A *a = new A(j != 0);
            B *b = new B(j != 0);

            ::Mask m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
            BasePtr arr = ek::select(m, (Base *) b, (Base *) a);

            StructF result = arr->f(Struct{ Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                            Array3f(4, 5, 6) * ek::full<Float>(1, n)});


            assert(ek::all_nested(
                ek::eq(result.a, ek::select(m, Array3f(80.f, 100.f, 120.f),
                                               Array3f(10.f, 20.f, 30.f))) &&
                ek::eq(result.b, ek::select(m, Array3f(10.f, 20.f, 30.f),
                                               Array3f(60.f, 75.f, 90.f)))));

            UInt32 len = arr->strlen("Hello world");
            assert(len == 11);

            arr->side_effect();

            jit_eval();

            result = arr->f(Struct{ Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                    Array3f(4, 5, 6) * ek::full<Float>(1, n)});

            assert(ek::all_nested(
                ek::eq(result.a, ek::select(m,  Array3f(80.f, 100.f, 120.f),
                                               -Array3f(10.f, 20.f, 30.f))) &&
                ek::eq(result.b, ek::select(m, -Array3f(10.f, 20.f, 30.f),
                                                Array3f(60.f, 75.f, 90.f)))));

            assert(ek::all(ek::eq(arr->field(), ek::select(m, 4.8f, 2.4f))));

            delete a;
            delete b;
        }
    }
}

ENOKI_TEST(test02_vcall_reduce_and_record_masked) {
    int n = 9999;

    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    ::Mask mask = ek::arange<UInt32>(n) > (n / 2);

    jit_set_log_level_stderr(::LogLevel::Error);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        for (int j = 0; j < 2; ++j) {
            A *a = new A(j != 0);
            B *b = new B(j != 0);

            ::Mask m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
            BasePtr arr = ek::select(m, (Base *) b, (Base *) a);

            StructF result = arr->f(Struct(Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                            Array3f(4, 5, 6) * ek::full<Float>(1, n)), mask);

            assert(ek::all_nested(
                ek::eq(result.a,
                        ek::select(mask,
                                    ek::select(m, Array3f(80.f, 100.f, 120.f),
                                                Array3f(10.f, 20.f, 30.f)),
                                    0.f)) &&
                ek::eq(result.b,
                        ek::select(mask,
                                    ek::select(m, Array3f(10.f, 20.f, 30.f),
                                                Array3f(60.f, 75.f, 90.f)),
                                    0.f))));

            // Masked pointer array
            arr[!mask] = nullptr;

            StructF result2 = arr->f(Struct(Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                            Array3f(4, 5, 6) * ek::full<Float>(1, n)));

            assert(ek::all_nested(ek::eq(result.a, result2.a) && ek::eq(result.b, result2.b)));

            delete a;
            delete b;
        }
    }
}

struct BaseD {
    BaseD() {
        x = 10.f;
        ek::enable_grad(x);
        ek::set_label(x, "BaseD::x");
    }
    virtual ~BaseD() { }
    void dummy() { }
    virtual StructFD f(const StructFD &m) = 0;
    virtual StructFD g(const StructFD &m) = 0;
    ENOKI_VCALL_REGISTER(FloatD, BaseD)
    FloatD x;
};

using BasePtrD = ek::replace_scalar_t<FloatD, BaseD *>;

struct AD : BaseD {
    StructFD f(const StructFD &m) override {
        return { m.a * 2, m.b * 3 };
    }

    StructFD g(const StructFD &m) override {
        return { m.a * x, m.b * 3 };
    }
};

struct BD : BaseD {
    StructFD f(const StructFD &m) override {
        return { m.b * 4, m.a * 5 };
    }
    StructFD g(const StructFD &m) override {
        return { m.b * 4, m.a + x };
    }
};

ENOKI_VCALL_BEGIN(BaseD)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(g)
ENOKI_VCALL_METHOD(dummy)
ENOKI_VCALL_END(BaseD)

ENOKI_TEST(test03_vcall_symbolic_ad_fwd) {
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    AD *a = new AD();
    BD *b = new BD();

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = ek::select(m, (BaseD *) a, (BaseD *) b);

        arr->dummy();

        Float o = ek::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) };

        ek::enable_grad(input);

        StructFD output = arr->f(input);
        ek::set_label(input, "input");
        ek::set_label(output, "output");
        ek::set_grad(input, StructF(1, 10));
        ek::enqueue(ADMode::Forward, input);
        ek::traverse<FloatD>();

        StructF grad_out = ek::grad(output);
        ek::eval(output, grad_out);

        assert(ek::all_nested(
            ek::eq(output.a, ek::select(m, input.a * 2, input.b * 4)) &&
            ek::eq(output.b, ek::select(m, input.b * 3, input.a * 5))));

        assert(ek::all_nested(
            ek::eq(grad_out.a, ek::select(ek::detach(m), 2, 40)) &&
            ek::eq(grad_out.b, ek::select(ek::detach(m), 30, 5))));
    }

    delete a;
    delete b;
}

ENOKI_TEST(test04_vcall_symbolic_ad_fwd_accessing_local) {
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        AD *a = new AD();
        BD *b = new BD();

        ek::enable_grad(a->x);
        ek::enable_grad(b->x);
        ek::set_grad(a->x, 100);
        ek::set_grad(b->x, 1000);

        BasePtrD arr = ek::select(m, (BaseD *) a, (BaseD *) b);
        arr->dummy();

        Float o = ek::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) };

        ek::enable_grad(input);

        StructFD output = arr->g(input);
        ek::set_label(input, "input");
        ek::set_label(output, "output");
        ek::set_grad(input, StructF(2, 10));

        ek::enqueue(ADMode::Forward, input, a->x, b->x);

        ek::traverse<FloatD>();

        StructF grad_out = ek::grad(output);
        ek::eval(output, grad_out);

        assert(ek::all_nested(
            ek::eq(output.a, ek::select(m, input.a * 10, input.b * 4)) &&
            ek::eq(output.b, ek::select(m, input.b * 3, input.a + 10))));

        assert(ek::all_nested(
            ek::eq(grad_out.a, ek::detach(ek::select(m, input.a * 100 + 20, 40))) &&
            ek::eq(grad_out.b, ek::detach(ek::select(m, 30, 1002)))));
        delete a;
        delete b;
    }
}

ENOKI_TEST(test05_vcall_symbolic_ad_bwd_accessing_local) {
    if constexpr (ek::is_cuda_array_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        AD *a = new AD();
        BD *b = new BD();

        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = ek::select(m, (BaseD *) a, (BaseD *) b);
        ek::enable_grad(a->x);
        ek::enable_grad(b->x);

        arr->dummy();

        Float o = ek::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) * o };

        ek::enable_grad(input);

        StructFD output = arr->g(input);
        ek::set_label(input, "input");
        ek::set_label(output, "output");
        ek::enqueue(ADMode::Backward, output);

        ek::set_grad(output, StructF(2, 10));
        ek::traverse<FloatD>();

        StructF grad_in = ek::grad(input);
        ek::eval(output, ek::grad(a->x), ek::grad(b->x), grad_in);

        assert(ek::all_nested(
            ek::eq(output.a, ek::select(m, input.a * 10, input.b * 4)) &&
            ek::eq(output.b, ek::select(m, input.b * 3, input.a + 10))));

        assert(ek::all_nested(
            ek::eq(grad_in.a, ek::detach(ek::select(m, 20, 10))) &&
            ek::eq(grad_in.b, ek::detach(ek::select(m, 30, 8)))));

        assert(ek::grad(a->x) == 5*6*2);
        assert(ek::grad(b->x) == 5*3*10);
        delete a;
        delete b;
    }
}
