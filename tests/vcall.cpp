#include "test.h"
#include <drjit/vcall.h>
#include <drjit/jit.h>
#include <drjit/autodiff.h>
#include <drjit/struct.h>

namespace dr = drjit;

template <typename T> struct Struct {
    T a, b;
    Struct(const T &a, const T &b) : a(a), b(b) { }
    DRJIT_STRUCT(Struct, a, b)
};

// using Float = dr::CUDAArray<float>;
using Float = dr::LLVMArray<float>;

using FloatD   = dr::DiffArray<Float>;
using UInt32   = dr::uint32_array_t<Float>;
using UInt32D  = dr::DiffArray<UInt32>;
using Mask     = dr::mask_t<Float>;
using MaskD    = dr::mask_t<FloatD>;
using Array2f  = dr::Array<Float, 2>;
using Array2fD = dr::Array<FloatD, 2>;
using Array3f  = dr::Array<Float, 3>;
using Array3fD = dr::Array<FloatD, 3>;
using StructF  = Struct<Array3f>;
using StructFD = Struct<Array3fD>;

struct Base {
    Base(bool scalar) : x(dr::opaque<Float>(10, scalar ? 1 : 10)) { }
    virtual ~Base() { }

    virtual StructF f(const StructF &m, ::Mask active = true) = 0;

    virtual void side_effect() {
        dr::scatter(x, Float(-10), UInt32(0));
    }

    UInt32 strlen(const std::string &string) {
        return string.length();
    }

    float field() const { return 1.2f; };
    DRJIT_VCALL_REGISTER(Float, Base)

protected:
    Float x;
};

using BasePtr = dr::replace_scalar_t<Float, Base *>;

struct A : Base {
    A(bool scalar) : Base(scalar) { dr::set_attr(this, "field", 2.4f); }
    StructF f(const StructF &m, ::Mask /* active */ = true) override {
        if (x.size() == 1)
            return Struct(m.a * x, m.b * 15);
        else
            return Struct(m.a * dr::gather<Float>(x, UInt32(0)), m.b * 15);
    }
};

struct B : Base {
    B(bool scalar) : Base(scalar) { dr::set_attr(this, "field", 4.8f); }
    StructF f(const StructF &m, ::Mask /* active */ = true) override {
        if (x.size() == 1)
            return Struct(m.b * 20, m.a * x);
        else
            return Struct(m.b * 20, m.a * dr::gather<Float>(x, UInt32(0)));
    }
};

DRJIT_VCALL_BEGIN(Base)
DRJIT_VCALL_METHOD(f)
DRJIT_VCALL_METHOD(side_effect)
DRJIT_VCALL_METHOD(strlen)
DRJIT_VCALL_GETTER(field, float)
DRJIT_VCALL_END(Base)

DRJIT_TEST(test01_vcall_reduce_and_record) {
    int n = 9999;

    // jit_set_log_level_stderr(::LogLevel::Error);
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        for (int j = 0; j < 2; ++j) {
            A *a = new A(j != 0);
            B *b = new B(j != 0);

            ::Mask m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
            BasePtr arr = dr::select(m, (Base *) b, (Base *) a);

            StructF result = arr->f(Struct{ Array3f(1, 2, 3) * dr::full<Float>(1, n),
                                            Array3f(4, 5, 6) * dr::full<Float>(1, n)});


            assert(dr::all_nested(
                dr::eq(result.a, dr::select(m, Array3f(80.f, 100.f, 120.f),
                                               Array3f(10.f, 20.f, 30.f))) &&
                dr::eq(result.b, dr::select(m, Array3f(10.f, 20.f, 30.f),
                                               Array3f(60.f, 75.f, 90.f)))));

            UInt32 len = arr->strlen("Hello world");
            assert(len == 11);

            arr->side_effect();

            jit_eval();

            result = arr->f(Struct{ Array3f(1, 2, 3) * dr::full<Float>(1, n),
                                    Array3f(4, 5, 6) * dr::full<Float>(1, n)});

            assert(dr::all_nested(
                dr::eq(result.a, dr::select(m,  Array3f(80.f, 100.f, 120.f),
                                               -Array3f(10.f, 20.f, 30.f))) &&
                dr::eq(result.b, dr::select(m, -Array3f(10.f, 20.f, 30.f),
                                                Array3f(60.f, 75.f, 90.f)))));

            assert(dr::all(dr::eq(arr->field(), dr::select(m, 4.8f, 2.4f))));

            delete a;
            delete b;
        }
    }
}

DRJIT_TEST(test02_vcall_reduce_and_record_masked) {
    int n = 9999;

    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    ::Mask mask = dr::arange<UInt32>(n) > (n / 2);

    jit_set_log_level_stderr(::LogLevel::Error);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        for (int j = 0; j < 2; ++j) {
            A *a = new A(j != 0);
            B *b = new B(j != 0);

            ::Mask m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
            BasePtr arr = dr::select(m, (Base *) b, (Base *) a);

            StructF result = arr->f(Struct(Array3f(1, 2, 3) * dr::full<Float>(1, n),
                                            Array3f(4, 5, 6) * dr::full<Float>(1, n)), mask);

            assert(dr::all_nested(
                dr::eq(result.a,
                        dr::select(mask,
                                    dr::select(m, Array3f(80.f, 100.f, 120.f),
                                                Array3f(10.f, 20.f, 30.f)),
                                    0.f)) &&
                dr::eq(result.b,
                        dr::select(mask,
                                    dr::select(m, Array3f(10.f, 20.f, 30.f),
                                                Array3f(60.f, 75.f, 90.f)),
                                    0.f))));

            // Masked pointer array
            arr[!mask] = nullptr;

            StructF result2 = arr->f(Struct(Array3f(1, 2, 3) * dr::full<Float>(1, n),
                                            Array3f(4, 5, 6) * dr::full<Float>(1, n)));

            assert(dr::all_nested(dr::eq(result.a, result2.a) && dr::eq(result.b, result2.b)));

            delete a;
            delete b;
        }
    }
}

struct BaseD {
    BaseD() {
        x = 10.f;
        dr::enable_grad(x);
        dr::set_label(x, "BaseD::x");
    }
    virtual ~BaseD() { }
    void dummy() { }
    virtual StructFD f(const StructFD &m) = 0;
    virtual StructFD f_masked(const StructFD &m, MaskD active) = 0;
    virtual StructFD g(const StructFD &m) = 0;
    DRJIT_VCALL_REGISTER(FloatD, BaseD)
    FloatD x;
};

using BasePtrD = dr::replace_scalar_t<FloatD, BaseD *>;

struct AD : BaseD {
    StructFD f(const StructFD &m) override {
        return { m.a * 2, m.b * 3 };
    }

    StructFD f_masked(const StructFD &m, MaskD active) override {
        DRJIT_MARK_USED(active);
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

    StructFD f_masked(const StructFD &m, MaskD active) override {
        DRJIT_MARK_USED(active);
        return { m.b * 4, m.a * 5 };
    }

    StructFD g(const StructFD &m) override {
        return { m.b * 4, m.a + x };
    }
};

DRJIT_VCALL_BEGIN(BaseD)
DRJIT_VCALL_METHOD(f)
DRJIT_VCALL_METHOD(f_masked)
DRJIT_VCALL_METHOD(g)
DRJIT_VCALL_METHOD(dummy)
DRJIT_VCALL_END(BaseD)

DRJIT_TEST(test03_vcall_symbolic_ad_fwd) {
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    AD *a = new AD();
    BD *b = new BD();

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = dr::select(m, (BaseD *) a, (BaseD *) b);

        arr->dummy();

        Float o = dr::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) };
        {
            dr::enable_grad(input);

            StructFD output = arr->f(input);
            dr::set_label(input, "input");
            dr::set_label(output, "output");
            dr::set_grad(input, StructF(1, 10));
            dr::enqueue(ADMode::Forward, input);
            dr::traverse<FloatD>(ADMode::Forward);

            StructF grad_out = dr::grad(output);
            dr::eval(output, grad_out);

            assert(dr::all_nested(
                dr::eq(output.a, dr::select(m, input.a * 2, input.b * 4)) &&
                dr::eq(output.b, dr::select(m, input.b * 3, input.a * 5))));

            assert(dr::all_nested(
                dr::eq(grad_out.a, dr::select(dr::detach(m), 2, 40)) &&
                dr::eq(grad_out.b, dr::select(dr::detach(m), 30, 5))));
        }

        // Masked version
        {
            dr::enable_grad(input);

            MaskD active = dr::arange<UInt32D>(n) < (n / 2);

            StructFD output = arr->f_masked(input, active);
            dr::set_label(input, "input");
            dr::set_label(output, "output");
            dr::set_grad(input, StructF(1, 10));
            dr::enqueue(ADMode::Forward, input);
            dr::traverse<FloatD>(ADMode::Forward);

            StructF grad_out = dr::grad(output);
            dr::eval(output, grad_out);

            assert(dr::all_nested(
                dr::eq(output.a, dr::select(active, dr::select(m, input.a * 2, input.b * 4), 0)) &&
                dr::eq(output.b, dr::select(active, dr::select(m, input.b * 3, input.a * 5), 0))));

            assert(dr::all_nested(
                dr::eq(grad_out.a, dr::select(dr::detach(active), dr::select(dr::detach(m), 2, 40), 0)) &&
                dr::eq(grad_out.b, dr::select(dr::detach(active), dr::select(dr::detach(m), 30, 5), 0))));
        }
    }

    delete a;
    delete b;
}

DRJIT_TEST(test04_vcall_symbolic_ad_fwd_accessing_local) {
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
        AD *a = new AD();
        BD *b = new BD();

        dr::enable_grad(a->x);
        dr::enable_grad(b->x);
        dr::set_grad(a->x, 100);
        dr::set_grad(b->x, 1000);

        BasePtrD arr = dr::select(m, (BaseD *) a, (BaseD *) b);
        arr->dummy();

        Float o = dr::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) };

        dr::enable_grad(input);

        StructFD output = arr->g(input);
        dr::set_label(input, "input");
        dr::set_label(output, "output");
        dr::set_grad(input, StructF(2, 10));

        dr::enqueue(ADMode::Forward, input, a->x, b->x);

        dr::traverse<FloatD>(ADMode::Forward);

        StructF grad_out = dr::grad(output);
        dr::eval(output, grad_out);

        assert(dr::all_nested(
            dr::eq(output.a, dr::select(m, input.a * 10, input.b * 4)) &&
            dr::eq(output.b, dr::select(m, input.b * 3, input.a + 10))));

        assert(dr::all_nested(
            dr::eq(grad_out.a, dr::detach(dr::select(m, input.a * 100 + 20, 40))) &&
            dr::eq(grad_out.b, dr::detach(dr::select(m, 30, 1002)))));
        delete a;
        delete b;
    }
}

DRJIT_TEST(test05_vcall_symbolic_ad_bwd_accessing_local) {
    if constexpr (dr::is_cuda_v<Float>)
        jit_init((uint32_t) JitBackend::CUDA);
    else
        jit_init((uint32_t) JitBackend::LLVM);

    for (int i = 0; i < 3; ++i) {
        AD *a = new AD();
        BD *b = new BD();

        jit_set_flag(JitFlag::VCallRecord, i != 0);
        jit_set_flag(JitFlag::VCallOptimize, i == 2);

        int n = 10;
        MaskD m = dr::neq(dr::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = dr::select(m, (BaseD *) a, (BaseD *) b);
        dr::enable_grad(a->x);
        dr::enable_grad(b->x);

        arr->dummy();

        Float o = dr::full<Float>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o,
                      Array3fD(4, 5, 6) * o };

        dr::enable_grad(input);

        StructFD output = arr->g(input);
        dr::set_label(input, "input");
        dr::set_label(output, "output");
        dr::enqueue(ADMode::Backward, output);

        dr::set_grad(output, StructF(2, 10));
        dr::traverse<FloatD>(ADMode::Backward);

        StructF grad_in = dr::grad(input);
        dr::eval(output, dr::grad(a->x), dr::grad(b->x), grad_in);

        assert(dr::all_nested(
            dr::eq(output.a, dr::select(m, input.a * 10, input.b * 4)) &&
            dr::eq(output.b, dr::select(m, input.b * 3, input.a + 10))));

        assert(dr::all_nested(
            dr::eq(grad_in.a, dr::detach(dr::select(m, 20, 10))) &&
            dr::eq(grad_in.b, dr::detach(dr::select(m, 30, 8)))));

        assert(dr::grad(a->x) == 5*6*2);
        assert(dr::grad(b->x) == 5*3*10);
        delete a;
        delete b;
    }
}
