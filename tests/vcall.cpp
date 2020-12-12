#include "test.h"
#include <enoki/vcall.h>
#include <enoki/cuda.h>
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
    Base() : x(ek::full<Float>(10, 10, true)) { }

    virtual StructF f(const StructF &m) = 0;

    virtual void side_effect() {
        ek::scatter(x, Float(-10), UInt32(0));
    }

    float field() const { return 1.2f; };
    ENOKI_VCALL_REGISTER(Base)

protected:
    Float x;
};

using BasePtr = ek::CUDAArray<Base *>;

struct A : Base {
    A() { ek::set_attr(this, "field", 2.4f); }
    StructF f(const StructF &m) override {
        return Struct { m.a * ek::gather<Float>(x, UInt32(0)), m.b * 15};
    }
};

struct B : Base {
    B() { ek::set_attr(this, "field", 4.8f); }
    StructF f(const StructF &m) override {
        return Struct { m.b * 20, m.a * ek::gather<Float>(x, UInt32(0))};
    }
};

ENOKI_VCALL_BEGIN(Base)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_METHOD(side_effect)
ENOKI_VCALL_GETTER(field, float)
ENOKI_VCALL_END(Base)

ENOKI_TEST(test01_vcall_eager_symbolic) {
    int n = 9999;

    jitc_init(0, 1);
    for (int i = 0; i < 2; ++i) {
        jitc_set_mode(i == 0 ? JitMode::Eager : JitMode::SymbolicPreferred);
        printf("=============================\n");

        A *a = new A();
        B *b = new B();

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

        arr->side_effect();

        jitc_eval();

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

using FloatD = ek::DiffArray<Float>;
using UInt32D = ek::DiffArray<UInt32>;
using MaskD = ek::mask_t<FloatD>;
using Array2fD = ek::Array<FloatD, 2>;
using Array3fD = ek::Array<FloatD, 3>;
using StructFD = Struct<Array3fD>;

struct BaseD {
    BaseD() { }
    virtual StructFD f(const StructFD &m) = 0;
    ENOKI_VCALL_REGISTER(BaseD)
};

using BasePtrD = ek::DiffArray<ek::CUDAArray<BaseD *>>;

struct AD : BaseD {
    StructFD f(const StructFD &m) override { return { m.a * 10, m.b * 15 }; }
};

struct BD : BaseD {
    StructFD f(const StructFD &m) override { return { m.b * 20, m.a * 10 }; }
};

ENOKI_VCALL_BEGIN(BaseD)
ENOKI_VCALL_METHOD(f)
ENOKI_VCALL_END(BaseD)

ENOKI_TEST(test02_vcall_eager_symbolic_ad_fwd) {
    int n = 9999;

    jitc_init(0, 1);
    for (int i = 0; i < 2; ++i) {
        jitc_set_mode(i == 0 ? JitMode::Eager : JitMode::SymbolicPreferred);
        printf("=============================\n");

        AD *a = new AD();
        BD *b = new BD();

        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = ek::select(m, (BaseD *) b, (BaseD *) a);

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

        std::cout << grad.a << std::endl;
        std::cout << grad.b << std::endl;

        delete a;
        delete b;
    }
}


/// Functions returning void
/// Functions reading from a private scalar value
/// Autodiffing functions reading from a private scalar value in reverse mode
