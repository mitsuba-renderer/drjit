#include "test.h"
#include <enoki/llvm.h>
#include <enoki/dynamic.h>
#include <enoki/custom.h>

#include <iostream>

namespace ek = enoki;

// Let's define some JIT & differentiable types
using Float     = ek::LLVMArray<float>;
using FloatD    = ek::DiffArray<Float>;

using Vector3f  = ek::Array<Float, 3>;
using Vector3fD = ek::Array<FloatD, 3>;


struct Normalize : ek::CustomOp<FloatD,      // Underlying differentiable type
                                Vector3fD,   // Function output, is allowed to be a std::pair/tuple for multi-valued functions
                                Vector3fD> { // Input argument(s) -- this is a varidic template, just specify a list
    using Base = ek::CustomOp<FloatD, Vector3fD, Vector3fD>;

    // Primal evaluation function, uses non-differentiable input/outputs
    Vector3f eval(const Vector3f &input) override {

        // Can cache values for later use
        m_input = input;
        m_inv_norm = ek::rcp(ek::norm(input));

        return input * m_inv_norm;
    }

    /**
     * Forward-mode AD callback. Should get input gradients via Base::grad_in<..>()
     * and must call Base::set_grad_out(..)
     */
    void forward() override {
        Vector3f grad_in = Base::grad_in(),
                 grad_out = grad_in * m_inv_norm;

        grad_out -= m_input * (ek::dot(m_input, grad_out) * ek::sqr(m_inv_norm));

        Base::set_grad_out(grad_out);
    }

    /**
     * Reverse-mode AD callback. Should get input gradients via Base::grad_out()
     * and must call Base::set_grad_in<..>(..) for each differentiable input
     */
    void backward() override {
        /// Boring example, ek.forward/backward are essentially identical

        Vector3f grad_out = Base::grad_out(),
                 grad_in = grad_out * m_inv_norm;

        grad_in -= m_input * (ek::dot(m_input, grad_in) * ek::sqr(m_inv_norm));

        Base::set_grad_in(grad_in);
    }

    const char *name() override {
        return "normalize";
    }

private:
    Float m_inv_norm;
    Vector3f m_input;
};


ENOKI_TEST(test01_basic) {
    jitc_init(1, 0);

    {
        Vector3fD d(1, 2, 3);
        ek::enable_grad(d);
        Vector3fD d2 = ek::custom<Normalize>(d);
        ek::set_grad(d2, Vector3f(5, 6, 7));
        ek::enqueue(d2);
        ek::traverse<FloatD>(true, false);
        assert(ek::allclose(ek::grad(d), Vector3f(0.610883, 0.152721, -0.305441)));
    }

    {
        Vector3fD d(1, 2, 3);
        ek::enable_grad(d);
        Vector3fD d2 = ek::custom<Normalize>(d);
        ek::set_grad(d, Vector3f(5, 6, 7));
        ek::enqueue(d);
        ek::traverse<FloatD>(false, false);
        assert(ek::allclose(ek::grad(d2), Vector3f(0.610883, 0.152721, -0.305441)));
    }

    jitc_shutdown();
}

struct ScaleAdd2 : ek::CustomOp<FloatD, Vector3fD, Vector3fD, Vector3fD, int> {
    using Base = ek::CustomOp<FloatD, Vector3fD, Vector3fD, Vector3fD, int>;

    // Primal evaluation function, uses non-differentiable input/outputs
    Vector3f eval(const Vector3f &in1, const Vector3f &in2, const int &scale) override {
        m_scale = scale;
        return (in1 + in2) * scale;
    }

    void forward() override {
        fprintf(stderr, "Forward.\n");
        Vector3f grad_in_1 = Base::grad_in<0>(),
                 grad_in_2 = Base::grad_in<1>();

        Base::set_grad_out((grad_in_1 + grad_in_2) * m_scale);
    }

    void backward() override {
        fprintf(stderr, "Reverse.\n");
        Vector3f grad_out = Base::grad_out();

        if (Base::grad_enabled_in<0>())
            Base::set_grad_in<0>(grad_out * m_scale);
        if (Base::grad_enabled_in<1>())
            Base::set_grad_in<1>(grad_out * m_scale);
    }

    const char *name() override {
        return "scale_add2";
    }

private:
    int m_scale;
};

ENOKI_TEST(test02_corner_case) {
    jitc_init(1, 0);

    {
        Vector3fD d1(1, 2, 3);
        Vector3fD d2(4, 5, 6);
        ek::enable_grad(d1.y());
        Vector3fD d3 = ek::custom<ScaleAdd2>(d1, d2, 5);
        ek::set_grad(d3, Vector3f(5, 6, 7));
        ek::enqueue(d3);
        ek::traverse<FloatD>(true, false);
        assert(ek::allclose(ek::grad(d1), Vector3f(0, 30, 0)));
    }

    {
        Vector3fD d1(1, 2, 3);
        Vector3fD d2(4, 5, 6);
        ek::enable_grad(d1.y());
        Vector3fD d3 = ek::custom<ScaleAdd2>(d1, d2, 5);
        ek::set_grad(d1, Vector3f(5, 6, 7));
        ek::enqueue(d1, d2);
        ek::traverse<FloatD>(false, false);
        assert(ek::allclose(ek::grad(d3), Vector3f(0, 30, 0)));
    }


    jitc_shutdown();
}
