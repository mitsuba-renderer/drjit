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
using FloatD = ek::DiffArray<Float>;
using UInt32 = ek::CUDAArray<uint32_t>;
using UInt32D = ek::DiffArray<UInt32>;
using Mask = ek::mask_t<Float>;
using MaskD = ek::mask_t<FloatD>;
using Array2f = ek::Array<Float, 2>;
using Array2fD = ek::Array<FloatD, 2>;
using Array3f = ek::Array<Float, 3>;
using Array3fD = ek::Array<FloatD, 3>;
using StructF = Struct<Array3f>;
using StructFD = Struct<Array3fD>;

#if 0
struct Base {
    Base(bool scalar) : x(ek::opaque<Float>(10, scalar ? 1 : 10)) { }

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

ENOKI_TEST(test01_vcall_reduce_and_record) {
    int n = 9999;

    jit_set_log_level_stderr(::LogLevel::Error);
    jit_init((uint32_t) JitBackend::CUDA);

    for (int i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i);
        for (int j = 0; j < 2; ++j) {
            // fprintf(stderr, "=============================\n");
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
#endif

struct BaseD {
    BaseD() {
        x = 10.f;
        ek::enable_grad(x);
        ek::set_label(x, "BaseD::x");
    }
    void dummy() { }
    virtual StructFD f(const StructFD &m) = 0;
    virtual StructFD g(const StructFD &m) = 0;
    ENOKI_VCALL_REGISTER(BaseD)
    FloatD x;
};

using BasePtrD = ek::DiffArray<ek::CUDAArray<BaseD *>>;

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

ENOKI_TEST(test02_vcall_symbolic_ad_fwd) {
    // jit_set_log_level_stderr(::LogLevel::Error);
    jit_init((uint32_t) JitBackend::CUDA);
    jit_set_flag(JitFlag::VCallRecord, 1);


    AD *a = new AD();
    BD *b = new BD();

    int n = 10;
    MaskD m = ek::neq(ek::arange<UInt32>(n) % 3, 0);
    BasePtrD arr = ek::select(m, (BaseD *) b, (BaseD *) a);

    arr->dummy();

    Float o = ek::full<Float>(1, n);

    Struct input{ Array3fD(1, 2, 3) * o,
                  Array3fD(4, 5, 6) };

    ek::enable_grad(input);

    StructFD output = arr->f(input);
    ek::set_label(input, "input");
    ek::set_label(output, "output");
    std::cout << ek::graphviz(output) << std::endl;
    delete a;
    delete b;
}

#if 0
    int n = 9999;





        ek::enable_grad(input);
        ek::set_label(input, "input");

        StructFD result = arr->f(input);

        assert(ek::all_nested(
            ek::eq(result.a, ek::select(m, Array3fD(80.f, 100.f, 120.f),
                                           Array3fD(10.f, 20.f, 30.f))) &&
            ek::eq(result.b, ek::select(m, Array3fD(10.f, 20.f, 30.f),
                                           Array3fD(60.f, 75.f, 90.f)))));

        ek::set_label(result, "result");


        ek::set_grad(input, StructF(1, 1));
        ek::enqueue(input);
        ek::traverse<FloatD>(false);

        StructF grad = ek::grad(result);
        ek::eval(grad);

        assert(ek::allclose(grad.a, Array3f(ek::select(ek::detach(m), 20, 10))));
        assert(ek::allclose(grad.b, Array3f(ek::select(ek::detach(m), 10, 15))));

        delete a;
        delete b;
    }
}

ENOKI_TEST(test03_vcall_symbolic_ad_rev) {
    int n = 9999;

    jit_set_log_level_stderr(::LogLevel::Error);

    jit_init((uint32_t) JitBackend::CUDA);

    jit_set_flag(JitFlag::VCallBranch, false);
    jit_set_flag(JitFlag::VCallOptimize, false);

    for (int i = 0; i < 2; ++i) {
        jit_set_flag(JitFlag::VCallRecord, i == 0);

        // fprintf(stderr, "=============================\n");

        AD *a = new AD();
        BD *b = new BD();

        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = ek::select(m, (BaseD *) b, (BaseD *) a);

        arr->dummy();

        FloatD o = ek::full<FloatD>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o, Array3fD(4, 5, 6) * o };

        ek::enable_grad(input);
        ek::set_label(input, "input");

        StructFD result = arr->f(input);

        assert(ek::all_nested(
            ek::eq(result.a, ek::select(m, Array3fD(80.f, 100.f, 120.f),
                                           Array3fD(10.f, 20.f, 30.f))) &&
            ek::eq(result.b, ek::select(m, Array3fD(10.f, 20.f, 30.f),
                                           Array3fD(60.f, 75.f, 90.f)))));

        ek::set_label(result, "result");
        ek::enqueue(result);
        ek::set_grad(result, StructF(1, 1));
        ek::traverse<FloatD>();

        StructF grad = ek::grad(input);
        ek::eval(grad);

        assert(ek::allclose(grad.a, Array3f(10)));
        assert(ek::allclose(grad.b, Array3f(ek::select(ek::detach(m), 20, 15))));

        delete a;
        delete b;
    }
}

ENOKI_TEST(test04_vcall_symbolic_ad_field_rev) {
    int n = 1000;

    jit_set_log_level_stderr(::LogLevel::Error);
    jit_init((uint32_t) JitBackend::CUDA);

    jit_set_flag(JitFlag::VCallBranch, false);
    jit_set_flag(JitFlag::VCallOptimize, false);

    for (int i = 0; i < 1; ++i) { // TODO also test vcall reduce
        jit_set_flag(JitFlag::VCallRecord, i == 0);

        // fprintf(stderr, "=============================\n");

        AD *a = new AD();
        BD *b = new BD();

        MaskD m = ek::neq(ek::arange<UInt32>(n) & 1, 0);
        BasePtrD arr = ek::select(m, (BaseD *) b, (BaseD *) a);

        FloatD o = ek::full<FloatD>(1, n);

        Struct input{ Array3fD(1, 2, 3) * o, Array3fD(4, 5, 6) * o };

        StructFD result = arr->g(input);

        assert(ek::all_nested(
            ek::eq(result.a, ek::select(m, Array3fD(80.f, 100.f, 120.f),
                                           Array3fD(10.f, 20.f, 30.f))) &&
            ek::eq(result.b, ek::select(m, Array3fD(10.f, 20.f, 30.f),
                                           Array3fD(60.f, 75.f, 90.f)))));

        ek::set_label(result, "result");

        // std::cout << "ek::graphviz(result): " << ek::graphviz(result) << std::endl;

        ek::enqueue(result);
        ek::set_grad(result, StructF(1, 1));
        ek::traverse<FloatD>();

        // std::cout << "ek::grad(a->x): " << ek::grad(a->x) << std::endl;
        // std::cout << "ek::grad(b->x): " << ek::grad(b->x) << std::endl;

        assert(ek::allclose(ek::grad(a->x), 3000));
        assert(ek::allclose(ek::grad(b->x), 15000));

        delete a;
        delete b;
    }
}
#endif
