#include "test.h"
#include <drjit/jit.h>
#include <drjit/dynamic.h>
#include <drjit/custom.h>

#include <iostream>

namespace dr = drjit;

// Let's define some JIT & differentiable types
using Float    = dr::DiffArray<dr::LLVMArray<float>>;
using Vector3f = dr::Array<Float, 3>;


struct Normalize : dr::CustomOp<Float,      // Underlying differentiable type
                                Vector3f,   // Function output, is allowed to be a std::pair/tuple for multi-valued functions
                                Vector3f> { // Input argument(s) -- this is a varidic template, just specify a list
    using Base = dr::CustomOp<Float, Vector3f, Vector3f>;

    // Primal evaluation function, uses non-differentiable input/outputs
    Vector3f eval(const Vector3f &input) override {

        // Can cache values for later use
        m_input = input;
        m_inv_norm = dr::rcp(dr::norm(input));

        return input * m_inv_norm;
    }

    /**
     * Forward-mode AD callback. Should get input gradients via Base::grad_in<..>()
     * and must call Base::set_grad_out(..)
     */
    void forward() override {
        Vector3f grad_in = Base::grad_in(),
                 grad_out = grad_in * m_inv_norm;

        grad_out -= m_input * (dr::dot(m_input, grad_out) * dr::sqr(m_inv_norm));

        Base::set_grad_out(grad_out);
    }

    /**
     * Backward-mode AD callback. Should get input gradients via Base::grad_out()
     * and must call Base::set_grad_in<..>(..) for each differentiable input
     */
    void backward() override {
        /// Boring example, ek.forward/backward are essentially identical

        Vector3f grad_out = Base::grad_out(),
                 grad_in = grad_out * m_inv_norm;

        grad_in -= m_input * (dr::dot(m_input, grad_in) * dr::sqr(m_inv_norm));

        Base::set_grad_in(grad_in);
    }

    const char *name() const override {
        return "normalize";
    }

private:
    Float m_inv_norm;
    Vector3f m_input;
};


DRJIT_TEST(test01_basic) {
    jit_init((uint32_t) JitBackend::LLVM);

    {
        Vector3f d(1, 2, 3);
        dr::enable_grad(d);
        Vector3f d2 = dr::custom<Normalize>(d);
        dr::set_grad(d2, Vector3f(5, 6, 7));
        dr::enqueue(ADMode::Backward, d2);
        dr::traverse<Float>(ADMode::Backward);
        assert(dr::allclose(dr::grad(d), Vector3f(0.610883, 0.152721, -0.305441)));
    }

    {
        Vector3f d(1, 2, 3);
        dr::enable_grad(d);
        Vector3f d2 = dr::custom<Normalize>(d);
        dr::set_grad(d, Vector3f(5, 6, 7));
        dr::enqueue(ADMode::Forward, d);
        dr::traverse<Float>(ADMode::Forward);
        assert(dr::allclose(dr::grad(d2), Vector3f(0.610883, 0.152721, -0.305441)));
    }

    jit_shutdown(1);
}

struct ScaleAdd2 : dr::CustomOp<Float, Vector3f, Vector3f, Vector3f, int> {
    using Base = dr::CustomOp<Float, Vector3f, Vector3f, Vector3f, int>;

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
        fprintf(stderr, "Backward.\n");
        Vector3f grad_out = Base::grad_out();

        if (Base::grad_enabled_in<0>())
            Base::set_grad_in<0>(grad_out * m_scale);
        if (Base::grad_enabled_in<1>())
            Base::set_grad_in<1>(grad_out * m_scale);
    }

    const char *name() const override {
        return "scale_add2";
    }

private:
    int m_scale;
};

DRJIT_TEST(test02_corner_case) {
    jit_init((uint32_t) JitBackend::LLVM);

    {
        Vector3f d1(1, 2, 3);
        Vector3f d2(4, 5, 6);
        dr::enable_grad(d1.y());
        Vector3f d3 = dr::custom<ScaleAdd2>(d1, d2, 5);
        dr::set_grad(d3, Vector3f(5, 6, 7));
        dr::enqueue(ADMode::Backward, d3);
        dr::traverse<Float>(ADMode::Backward);
        assert(dr::allclose(dr::grad(d1), Vector3f(0, 30, 0)));
    }

    {
        Vector3f d1(1, 2, 3);
        Vector3f d2(4, 5, 6);
        dr::enable_grad(d1.y());
        Vector3f d3 = dr::custom<ScaleAdd2>(d1, d2, 5);
        dr::set_grad(d1, Vector3f(5, 6, 7));
        dr::enqueue(ADMode::Forward, d1, d2);
        dr::traverse<Float>(ADMode::Forward);
        assert(dr::allclose(dr::grad(d3), Vector3f(0, 30, 0)));
    }


    jit_shutdown(1);
}
