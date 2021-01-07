#include "test.h"
#include <enoki/vcall.h>
#include <enoki/jit.h>
#include <enoki/autodiff.h>
#include <enoki/struct.h>

namespace ek = enoki;

template <typename T> struct Struct {
    T a;
    T b;
    Struct(const T &a, const T &b) : a(a), b(b) { }
    ENOKI_STRUCT(Struct, a, b)
};

using Float = ek::CUDAArray<float>;
using UInt32 = ek::CUDAArray<uint32_t>;
using Mask = ek::mask_t<Float>;
using Array2f = ek::Array<Float, 2>;
using Array3f = ek::Array<Float, 3>;
using StructF = Struct<Array3f>;

struct Base {
    Base(bool scalar) : x(ek::full<Float>(10, scalar ? 1 : 10, true)) { }

    virtual StructF f(const StructF &m) = 0;

    virtual void side_effect() {
        ek::scatter(x, Float(-10), UInt32(0));
    }

    UInt32 strlen(const std::string &string) {
        return string.length();
    }

    float field() const { return 1.2f; };
    ENOKI_VCALL_REGISTER(Base)

protected:
    Float x;
};

using BasePtr = ek::CUDAArray<Base *>;

struct A : Base {
    A(bool scalar) : Base(scalar) { ek::set_attr(this, "field", 2.4f); }
    StructF f(const StructF &m) override {
        if (x.size() == 1)
            return Struct { m.a * x, m.b * 15 };
        else
            return Struct { m.a * ek::gather<Float>(x, UInt32(0)), m.b * 15 };
    }
};

struct B : Base {
    B(bool scalar) : Base(scalar) { ek::set_attr(this, "field", 4.8f); }
    StructF f(const StructF &m) override {
        if (x.size() == 1)
            return Struct { m.b * 20, m.a * x };
        else
            return Struct { m.b * 20, m.a * ek::gather<Float>(x, UInt32(0)) };
    }
};

ENOKI_VCALL_BEGIN(Base)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(side_effect)
ENOKI_VCALL_METHOD(strlen)
ENOKI_VCALL_GETTER(field, float)
ENOKI_VCALL_END(Base)

ENOKI_TEST(test01_vcall_eager_symbolic) {
    int n = 9999;

    jit_init(0, 1);
    for (int i = 0; i < 2; ++i) {
        jit_set_flags((uint32_t) (i == 0 ? JitFlag::Default : JitFlag::RecordVCalls));
        for (int j = 0; j < 2; ++j) {
            fprintf(stderr, "=============================\n");
            A *a = new A(j != 0);
            B *b = new B(j != 0);

            ::Mask m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
            BasePtr arr = ek::select(m, (Base *) b, (Base *) a);

            StructF result = arr->f(Struct{ Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                            Array3f(4, 5, 6) * ek::full<Float>(1, n)});

            assert(ek::all_nested(
                ek::eq(result.a, Array3f(ek::select(m, 80.f, 10.f),
                                         ek::select(m, 100.f, 20.f),
                                         ek::select(m, 120.f, 30.f))) &&
                ek::eq(result.b, Array3f(ek::select(m, 10.f, 60.f),
                                         ek::select(m, 20.f, 75.f),
                                         ek::select(m, 30.f, 90.f)))));

            UInt32 len = arr->strlen("Hello world");
            assert(len == 11);

            arr->side_effect();

            jit_eval();

            result = arr->f(Struct{ Array3f(1, 2, 3) * ek::full<Float>(1, n),
                                    Array3f(4, 5, 6) * ek::full<Float>(1, n)});

            assert(ek::all_nested(
                ek::eq(result.a, Array3f(ek::select(m, 80.f, -10.f),
                                         ek::select(m, 100.f, -20.f),
                                         ek::select(m, 120.f, -30.f))) &&
                ek::eq(result.b, Array3f(ek::select(m, -10.f, 60.f),
                                         ek::select(m, -20.f, 75.f),
                                         ek::select(m, -30.f, 90.f)))));

            assert(ek::all(ek::eq(arr->field(), ek::select(m, 4.8f, 2.4f))));

            delete a;
            delete b;
        }
    }
}

using FloatD = ek::DiffArray<Float>;
using UInt32D = ek::DiffArray<UInt32>;
using MaskD = ek::mask_t<FloatD>;
using Array2fD = ek::Array<FloatD, 2>;
using Array3fD = ek::Array<FloatD, 3>;
using StructFD = Struct<Array3fD>;

struct BaseD {
    BaseD(bool scalar) {
        if (scalar) {
            x = 10;
            ek::enable_grad(x);
            ek::set_grad(x, ek::full<Float>(0, 1, true));
        }
    }

    void dummy() { }

    virtual StructFD f(const StructFD &m) = 0;
    ENOKI_VCALL_REGISTER(BaseD)

    FloatD x;
};

using BasePtrD = ek::DiffArray<ek::CUDAArray<BaseD *>>;

struct AD : BaseD {
    using BaseD::BaseD;
    StructFD f(const StructFD &m) override {
        if (x.size() > 0)
            return { m.a * x, m.b * 15 };
        else
            return { m.a * 10, m.b * 15 };
    }
};

struct BD : BaseD {
    using BaseD::BaseD;
    StructFD f(const StructFD &m) override {
        if (x.size() > 0)
            return { m.b * 20, m.a * x };
        else
            return { m.b * 20, m.a * 10 };
    }
};

ENOKI_VCALL_BEGIN(BaseD)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(dummy)
ENOKI_VCALL_END(BaseD)

ENOKI_TEST(test02_vcall_eager_symbolic_ad_fwd) {
    int n = 9999;

    jit_init(0, 1);
    for (int i = 0; i < 2; ++i) {
        jit_set_flags((uint32_t) (i == 0 ? JitFlag::Default : JitFlag::RecordVCalls));
        for (int k = 0; k < 2; ++k) {
            fprintf(stderr, "=============================\n");

            AD *a = new AD(k == 1);
            BD *b = new BD(k == 1);

            MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
            BasePtrD arr = ek::select(m, (BaseD *) b, (BaseD *) a);

            arr->dummy();

            FloatD o = ek::full<FloatD>(1, n);

            Struct input{ Array3fD(1, 2, 3) * o, Array3fD(4, 5, 6) * o };

            ek::enable_grad(input);
            ek::set_label(input, "input");

            StructFD result = arr->f(input);

            assert(ek::all_nested(
                ek::eq(result.a, Array3fD(ek::select(m, 80.f, 10.f),
                                          ek::select(m, 100.f, 20.f),
                                          ek::select(m, 120.f, 30.f))) &&
                ek::eq(result.b, Array3fD(ek::select(m, 10.f, 60.f),
                                          ek::select(m, 20.f, 75.f),
                                          ek::select(m, 30.f, 90.f)))));

            ek::set_label(result, "result");
            ek::enqueue(result);
            ek::set_grad(result, StructF(1, 1));
            ek::traverse<FloatD>();

            StructF grad = ek::grad(input);
            ek::eval(grad);

            assert(ek::allclose(grad.a, Array3f(10)));
            assert(ek::allclose(grad.b,
                                Array3f(ek::select(ek::detach(m), 20, 15))));

            if (k == 1) {
                assert(ek::grad(a->x) == 30000);
                assert(ek::grad(b->x) == 29994);
            }

            delete a;
            delete b;
        }
    }
}
