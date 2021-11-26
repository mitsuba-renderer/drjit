#include "test.h"
#include <enoki/texture.h>
#include <enoki/random.h>
#include <enoki/autodiff.h>

namespace ek = enoki;

using Float = ek::CUDAArray<float>;
using Array1f = ek::Array<Float, 1>;
using Array2f = ek::Array<Float, 2>;
using Array3f = ek::Array<Float, 3>;
using Array4f = ek::Array<Float, 4>;

ENOKI_TEST(test01_interp_1d) {
    jit_init(JitBackend::CUDA);

    for (int k = 0; k < 2; ++k) {
        jit_set_flag(JitFlag::ForceOptiX, k == 1);

        size_t shape[1] = { 2 };
        ek::Texture<Float, 1> tex(shape, 1);
        tex.set_value(Float(0.f, 1.f));

        size_t N = 11;

        Float ref = ek::linspace<Float>(0, 1, N);
        Array1f pos(ek::linspace<Float>(0.25f, 0.75f, N));

        assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
        assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));

        ref = ek::opaque<Float>(0.f, N);
        pos.x() = ek::linspace<Float>(-0.25f, 0.25f, N);
        assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
        assert(ek::allclose(tex.eval_cuda(pos).x(), ref));

        ref = ek::opaque<Float>(1.f, N);
        pos.x() = ek::linspace<Float>(0.75f, 1.25f, N);
        assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
        assert(ek::allclose(tex.eval_cuda(pos).x(), ref));
    }

    jit_set_flag(JitFlag::ForceOptiX, false);
}

ENOKI_TEST(test02_interp_1d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123 };
        PCG32<Float> rng_1(shape[0] * ch);
        PCG32<Float> rng_2(1024);

        Texture<Float, 1> tex(shape, ch);
        for (int i = 0; i < 4; ++i) {
            tex.set_value(rng_1.next_float32());

            Array1f pos(rng_2.next_float32());
            Array4f result_enoki = tex.eval_enoki(pos);
            ek::eval(result_enoki);
            Array4f result_cuda = tex.eval_cuda(pos);
            ek::eval(result_cuda);

            assert(ek::allclose(result_enoki, result_cuda, 5e-3f, 5e-3f));
        }
    }
}

ENOKI_TEST(test03_interp_2d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123, 456 };
        PCG32<Float> rng_1(shape[0] * shape[1] * ch);
        PCG32<Float> rng_2(1024);

        Texture<Float, 2> tex(shape, ch);
        for (int i = 0; i < 4; ++i) {
            tex.set_value(rng_1.next_float32());

            Array2f pos(rng_2.next_float32(), rng_2.next_float32());
            Array4f result_enoki = tex.eval_enoki(pos);
            ek::eval(result_enoki);
            Array4f result_cuda = tex.eval_cuda(pos);
            ek::eval(result_cuda);
            assert(ek::allclose(result_enoki, result_cuda, 5e-3f, 5e-3f));
        }
    }
}


ENOKI_TEST(test04_interp_3d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123, 456, 12 };
        PCG32<Float> rng_1(shape[0] * shape[1] * shape[2] * ch);
        PCG32<Float> rng_2(1024);

        Texture<Float, 3> tex(shape, ch);
        for (int i = 0; i < 4; ++i) {
            tex.set_value(rng_1.next_float32());

            Array3f pos(rng_2.next_float32(), rng_2.next_float32(),
                        rng_2.next_float32());
            Array4f result_enoki = tex.eval_enoki(pos);
            ek::eval(result_enoki);
            Array4f result_cuda = tex.eval_cuda(pos);
            ek::eval(result_cuda);
            assert(ek::allclose(result_enoki, result_cuda, 5e-3f, 5e-3f));
        }
    }
}


ENOKI_TEST(test05_grad) {
    using DFloat = ek::DiffArray<Float>;
    size_t shape[] = { 3 };
    Texture<DFloat, 1> tex(shape, 1);

    DFloat value(3, 5, 8);
    ek::enable_grad(value);
    tex.set_value(value);

    auto out = tex.eval(ek::Array<DFloat, 1>(1/6.f*0.25f + (1/6.f+1/3.f)*0.75f));
    ek::backward(out.x());

    assert(ek::allclose(ek::grad(value), DFloat(.25f, .75f, 0)));
    assert(ek::allclose(out.x(), DFloat(0.25f * 3 + 0.75f * 5), 5e-3, 5e-3f));
}
