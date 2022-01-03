#include "test.h"
#include <enoki/autodiff.h>
#include <enoki/random.h>
#include <enoki/texture.h>

namespace ek = enoki;

using Float = ek::CUDAArray<float>;
using DFloat = ek::DiffArray<Float>;
using Array1f = ek::Array<Float, 1>;
using Array2f = ek::Array<Float, 2>;
using Array3f = ek::Array<Float, 3>;
using Array4f = ek::Array<Float, 4>;
using ArrayD1f = ek::Array<DFloat, 1>;
using ArrayD2f = ek::Array<DFloat, 2>;
using ArrayD3f = ek::Array<DFloat, 3>;
using ArrayD4f = ek::Array<DFloat, 4>;

void test_interp_1d_wrap(WrapMode wrap_mode) {
    for (int k = 0; k < 2; ++k) {
        jit_set_flag(JitFlag::ForceOptiX, k == 1);

        size_t shape[1] = { 2 };
        ek::Texture<Float, 1> tex(shape, 1, false, FilterMode::Linear,
                                  wrap_mode);
        tex.set_value(Float(0.f, 1.f));

        size_t N = 11;

        Float ref = ek::linspace<Float>(0.f, 1.f, N);
        Array1f pos(ek::linspace<Float>(0.25f, 0.75f, N));

        assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
        assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));

        switch (wrap_mode) {
            case WrapMode::Repeat: {
                pos.x() = ek::linspace<Float>(-0.75f, -0.25f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));

                pos.x() = ek::linspace<Float>(1.25f, 1.75f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));
                break;
            }
            case WrapMode::Clamp: {
                ref     = ek::opaque<Float>(0.f, N);
                pos.x() = ek::linspace<Float>(-0.25f, 0.25f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));

                ref     = ek::opaque<Float>(1.f, N);
                pos.x() = ek::linspace<Float>(0.75f, 1.25f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));
                break;
            }
            case WrapMode::Mirror: {
                pos.x() = ek::linspace<Float>(-0.25f, -0.75f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));

                pos.x() = ek::linspace<Float>(1.75f, 1.25f, N);
                assert(ek::allclose(tex.eval_enoki(pos).x(), ref));
                assert(ek::allclose(tex.eval_cuda(pos).x(), ref, 5e-3f, 5e-3f));
                break;
            }
        }
    }

    jit_set_flag(JitFlag::ForceOptiX, false);
}

ENOKI_TEST(test01_interp_1d) {
    jit_init(JitBackend::CUDA);

    test_interp_1d_wrap(WrapMode::Repeat);
    test_interp_1d_wrap(WrapMode::Clamp);
    test_interp_1d_wrap(WrapMode::Mirror);
}

ENOKI_TEST(test02_interp_1d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123 };
        PCG32<Float> rng_1(shape[0] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 1> tex(shape, ch, false, FilterMode::Linear,
                                  wrap_modes[i]);

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
}

ENOKI_TEST(test03_interp_2d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123, 456 };
        PCG32<Float> rng_1(shape[0] * shape[1] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 2> tex(shape, ch, false, FilterMode::Linear,
                                  wrap_modes[i]);

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
}

ENOKI_TEST(test04_interp_3d) {
    for (int ch = 1; ch <= 4; ++ch) {
        if (ch == 3)
            continue;
        size_t shape[] = { 123, 456, 12 };
        PCG32<Float> rng_1(shape[0] * shape[1] * shape[2] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 3> tex(shape, ch, false, FilterMode::Linear,
                                  wrap_modes[i]);

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
}

void test_grad(bool migrate) {
    size_t shape[] = { 3 };
    Texture<DFloat, 1> tex(shape, 1, migrate);

    DFloat value(3, 5, 8);
    ek::enable_grad(value);
    tex.set_value(value);

    ek::Array<DFloat, 1> pos(1 / 6.f * 0.25f + (1 / 6.f + 1 / 3.f) * 0.75f);
    DFloat expected(0.25f * 3 + 0.75f * 5);
    // check migration
    auto out2 = tex.eval_enoki(pos);
    if (migrate)
        assert(ek::allclose(out2.x(), 0));
    else
        assert(ek::allclose(out2.x(), expected, 5e-3f, 5e-3f));

    auto out = tex.eval(pos);
    ek::backward(out.x());

    assert(ek::allclose(ek::grad(value), DFloat(.25f, .75f, 0)));
    assert(ek::allclose(out.x(), expected, 5e-3f, 5e-3f));
    assert(ek::allclose(tex.value(), value));
}

ENOKI_TEST(test05_grad) {
    test_grad(true);
    test_grad(false);
}

ENOKI_TEST(test06_nearest) {
    size_t shape[1] = { 3 };
    ek::Texture<Float, 1> tex(shape, 1, false, ek::FilterMode::Nearest);
    tex.set_value(Float(0.f, 0.5f, 1.f));

    Float pos = ek::linspace<Float>(0, 1, 80);
    assert(ek::allclose(tex.eval_cuda(pos).x(), tex.eval_enoki(pos).x()));
}

ENOKI_TEST(test07_cubic_analytic) {
    size_t shape[1] = { 4 };
    ek::Texture<DFloat, 1> tex(shape, 1, false, FilterMode::Linear,
                               WrapMode::Clamp);
    tex.set_value(DFloat(0.f, 1.f, 0.f, 0.f));

    ArrayD1f pos(0.5f);
    auto grad_64 = tex.eval_cubic_grad(pos, true);
    ek::enable_grad(pos);
    auto res = tex.eval_cubic(pos, true, true);
    ek::backward(res.x());
    auto grad_ad = ek::grad(pos);
    auto res2 = tex.eval_cubic_helper(pos);

    // 1/6 * (3*a^3 - 6*a^2 + 4) with a=0.5
    Array4f ref_res(0.479167f, 0.0f, 0.0f, 0.0f);
    assert(ek::allclose(res, ref_res, 1e-5f, 1e-5f));
    assert(ek::allclose(res2, ref_res, 1e-5f, 1e-5f));
    // 1/6 * (9*a^2 - 12*a) with a=0.5
    Float ref_grad(-0.625f * 4.0f);
    assert(ek::allclose(grad_64[0][0], ref_grad, 1e-5f, 1e-5f));
    assert(ek::allclose(grad_ad[0], ref_grad, 1e-5f, 1e-5f));
}

void test08_cubic_interp_1d(WrapMode wrap_mode) {
    size_t shape[1] = { 5 };
    ek::Texture<Float, 1> tex(shape, 1, false, FilterMode::Linear, wrap_mode);
    tex.set_value(Float(2.f, 1.f, 3.f, 4.f, 7.f));

    size_t N = 20;

    Array1f pos(ek::linspace<Float>(0.25f, 0.75f, N));
    Float ref = tex.eval_cubic_helper(pos).x();

    assert(ek::allclose(tex.eval_cubic(pos).x(), ref));

    switch (wrap_mode) {
        case WrapMode::Repeat: {
            pos.x() = ek::linspace<Float>(-0.75f, -0.25f, N);
            auto res = tex.eval_cubic(pos, true, true).x();
            auto res2 = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res, ref));
            assert(ek::allclose(res2, ref));

            pos.x() = ek::linspace<Float>(1.25f, 1.75f, N);
            auto res_ = tex.eval_cubic(pos, true, true).x();
            auto res2_ = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res_, ref));
            assert(ek::allclose(res2_, ref));
            break;
        }
        case WrapMode::Clamp: {
            {
                Array1f pos(ek::linspace<Float>(0.f, 1.f, N));
                auto res = tex.eval_cubic(pos, true, true).x();
                auto res2 = tex.eval_cubic_helper(pos).x();

                Float ref = Float(1.9792, 1.9259, 1.8198, 1.6629, 1.5168,
                                  1.4546, 1.5485, 1.8199, 2.2043, 2.6288,
                                  3.0232, 3.3783, 3.7461, 4.1814, 4.7305,
                                  5.3536, 5.9603, 6.4595, 6.7778, 6.9375);
                assert(ek::allclose(res, ref, 5e-4f, 5e-4f));
                assert(ek::allclose(res2, ref, 5e-4f, 5e-4f));
            }

            ref = ek::opaque<Float>(ref[0], N);
            pos.x() = ek::linspace<Float>(-0.25f, 0.25f, N);
            auto res = tex.eval_cubic(pos, true, true).x();
            auto res2 = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res, ref));
            assert(ek::allclose(res2, ref, 5e-3f, 5e-3f));

            ref = ek::opaque<Float>(ref[N - 1], N);
            pos.x() = ek::linspace<Float>(0.75f, 1.25f, N);
            auto res_ = tex.eval_cubic(pos, true, true).x();
            auto res2_ = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res_, ref));
            assert(ek::allclose(res2_, ref));
            break;
        }
        case WrapMode::Mirror: {
            pos.x() = ek::linspace<Float>(-0.25f, -0.75f, N);
            auto res = tex.eval_cubic(pos, true, true).x();
            auto res2 = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res, ref));
            assert(ek::allclose(res2, ref, 5e-3f, 5e-3f));

            pos.x() = ek::linspace<Float>(1.75f, 1.25f, N);
            auto res_ = tex.eval_cubic(pos, true, true).x();
            auto res2_ = tex.eval_cubic_helper(pos).x();
            assert(ek::allclose(res_, ref));
            assert(ek::allclose(res2_, ref));
            break;
        }
    }
}

ENOKI_TEST(test08_cubic_interp_1d) {
    test_interp_1d_wrap(WrapMode::Repeat);
    test_interp_1d_wrap(WrapMode::Clamp);
    test_interp_1d_wrap(WrapMode::Mirror);
}

ENOKI_TEST(test09_cubic_interp_2d) {
    size_t shape[2] = { 5, 4 };
    ek::Texture<Float, 2> tex(shape, 1, false, FilterMode::Linear,
                              WrapMode::Clamp);
    tex.set_value(ek::linspace<Float>(0.f, 20.f, 20));

    Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                  WrapMode::Mirror);

    for (size_t i = 0; i < wrap_modes.size(); ++i) {
        Texture<Float, 2> tex(shape, 1, false, FilterMode::Linear,
                              wrap_modes[i]);
        size_t N = 30;
        Array2f pos(ek::linspace<Float>(0.f, 1.f, N),
                    ek::linspace<Float>(0.f, 1.f, N));
        auto res = tex.eval_cubic(pos, true, true);
        auto res2 = tex.eval_cubic_helper(pos);
        assert(ek::allclose(res, res2, 1e-4f, 1e-4f));

        PCG32<Float> rng(1024);
        Array2f pos_(rng.next_float32(), rng.next_float32());
        auto res_ = tex.eval_cubic(pos, true, true);
        auto res2_ = tex.eval_cubic_helper(pos);
        assert(ek::allclose(res_, res2_, 1e-4f, 1e-4f));
    }
}

ENOKI_TEST(test10_cubic_interp_3d) {
    using TensorXf = Tensor<Float>;
    using UInt32 = ek::uint32_array_t<Float>;
    const int s = 9;
    size_t shape[3] = { s, s, s };
    size_t shape_[] = { s, s, s, 2 };  // 2 channels
    
    auto data = ek::full<Float>((1.0), s*s*s*2);
    TensorXf tensor(data, 4, shape_);
    ek::scatter(tensor.array(), Float(0.0),  UInt32(728));  // tensor[4, 4, 4, 0] = 0.0
    ek::scatter(tensor.array(), Float(2.0),  UInt32(546));  // tensor[3, 3, 3, 0] = 2.0
    ek::scatter(tensor.array(), Float(10.0), UInt32(727));  // tensor[4, 4, 3, 1] = 10.0

    ek::Texture<Float, 3> tex(shape, 2, false, FilterMode::Linear,
                              WrapMode::Clamp);
    tex.set_tensor(tensor);

    Array4f ref(0.71312, 1.86141, 0.0, 0.0);
    Array3f pos(.49f, .5f, .5f);
    auto res = tex.eval_cubic(pos, true, true);
    auto res_ = tex.eval_cubic_helper(pos);
    assert(ek::allclose(res, ref, 1e-4f, 1e-4f));
    assert(ek::allclose(res_, ref, 1e-4f, 1e-4f));

    Array4f ref2(0.800905, 2.60136, 0.0, 0.0);
    Array3f pos2(.45f, .53f, .51f);
    auto res2 = tex.eval_cubic(pos2, true, true);
    auto res2_ = tex.eval_cubic_helper(pos2);
    assert(ek::allclose(res2, ref2, 1e-4f, 1e-4f));
    assert(ek::allclose(res2_, ref2, 1e-4f, 1e-4f));
}

ENOKI_TEST(test11_cubic_grad_pos) {
    using TensorXf = Tensor<Float>;
    using UInt32 = ek::uint32_array_t<Float>;
    size_t shape[3] = { 4, 4, 4 };

    auto data = ek::full<Float>((1.0), 4*4*4);
    size_t shape_[] = { 4, 4, 4, 1 };
    TensorXf tensor(data, 4, shape_);
    ek::scatter(tensor.array(), Float(0.5f), UInt32(21));  // data[1, 1, 1] = 0.5
    ek::scatter(tensor.array(), Float(2.0f), UInt32(25));  // data[1, 2, 1] = 2.0
    ek::scatter(tensor.array(), Float(3.0f), UInt32(41));  // data[2, 2, 1] = 3.0
    ek::scatter(tensor.array(), Float(4.0f), UInt32(22));  // data[1, 1, 2] = 4.0

    ek::Texture<DFloat, 3> tex(shape, 1, false, FilterMode::Linear,
                               WrapMode::Clamp);
    tex.set_tensor(tensor);

    ArrayD3f pos(.5f, .5f, .5f);
    auto grad_64 = tex.eval_cubic_grad(pos, true);
    ek::enable_grad(pos);
    auto res = tex.eval_cubic(pos, true, true);
    ek::backward(res.x());
    auto grad_ad = ek::grad(pos);
    auto res2 = tex.eval_cubic_helper(pos);

    Array4f ref_res(1.60509f, 0.0f, 0.0f, 0.0f);
    assert(ek::allclose(res, ref_res, 1e-5f, 1e-5f));
    assert(ek::allclose(res2, ref_res, 1e-5f, 1e-5f));
    Array3f ref_grad(0.07175f, 0.07175f, -0.21525f);
    ref_grad *= 4.0f;
    assert(ek::allclose(grad_64[0][0], ref_grad[0], 1e-5f, 1e-5f));
    assert(ek::allclose(grad_64[1][0], ref_grad[1], 1e-5f, 1e-5f));
    assert(ek::allclose(grad_64[2][0], ref_grad[2], 1e-5f, 1e-5f));
    assert(ek::allclose(grad_ad, ref_grad, 1e-5f, 1e-5f));
}

ENOKI_TEST(test12_cubic_hessian_pos) {
    using TensorXf = Tensor<Float>;
    using UInt32 = ek::uint32_array_t<Float>;
    size_t shape[3] = { 4, 4, 4 };

    auto data = ek::full<Float>((0.0), 4*4*4);
    size_t shape_[] = { 4, 4, 4, 1 };
    TensorXf tensor(data, 4, shape_);
    ek::scatter(tensor.array(), Float(1.0f), UInt32(21));  // data[1, 1, 1] = 1.0
    ek::scatter(tensor.array(), Float(2.0f), UInt32(37));  // data[2, 1, 1] = 2.0
    // NOTE: Tensor has different index convention with Texture
    //       [2, 1, 1] is equivalent to (x=1, y=1, z=2) in the texture

    ek::Texture<DFloat, 3> tex(shape, 1, false, FilterMode::Linear,
                               WrapMode::Clamp);
    tex.set_tensor(tensor);

    ArrayD3f pos(.5f, .5f, .5f);
    auto grad_64 = tex.eval_cubic_grad(pos, true);
    auto [grad_h, hessian] = tex.eval_cubic_hessian(pos, true);

    assert(ek::allclose(grad_64[0], grad_h[0]));
    assert(ek::allclose(grad_64[1], grad_h[1]));
    assert(ek::allclose(grad_64[2], grad_h[2]));
    // compare with analytical solution
    // note: hessian[grad1][grad2][ch]
    // note: multiply analytical result by 16.0f=4.f*4.f to account for the resolution transformation
    assert(ek::allclose(hessian[0][0][0], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(ek::allclose(hessian[0][1][0],  0.561523f * 16.0f, 1e-5f, 1e-5f));
    assert(ek::allclose(hessian[0][2][0], -0.187174f * 16.0f, 1e-5f, 1e-5f));
    assert(ek::allclose(hessian[1][1][0], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(ek::allclose(hessian[1][2][0], -0.187174f * 16.0f, 1e-5f, 1e-5f));
    assert(ek::allclose(hessian[2][2][0], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(hessian[0][1] == hessian[1][0]);
    assert(hessian[0][2] == hessian[2][0]);
    assert(hessian[1][2] == hessian[2][1]);
}

ENOKI_TEST(test13_move_assignment) {
    size_t shape[1] = { 2 };
    ek::Texture<Float, 1> move_from(shape, 1, false, FilterMode::Nearest,
                              WrapMode::Repeat);
    move_from.set_value(Float(0.f, 1.f));
    const void *from_handle = move_from.handle();

    ek::Texture<Float, 1> move_to;
    move_to = std::move(move_from);

    assert(move_to.ndim() == 2);
    assert(move_to.handle() == from_handle);
    assert(move_from.handle() == nullptr);
    assert(move_to.shape()[0] == shape[0]);
    assert(move_to.wrap_mode() == WrapMode::Repeat);
    assert(move_to.filter_mode() == FilterMode::Nearest);
}

ENOKI_TEST(test14_move_constructor) {
    size_t shape[1] = { 2 };
    ek::Texture<Float, 1> move_from(shape, 1, false, FilterMode::Nearest,
                              WrapMode::Repeat);
    move_from.set_value(Float(0.f, 1.f));
    const void *from_handle = move_from.handle();

    ek::Texture<Float, 1> move_to(std::move(move_from));

    assert(move_to.ndim() == 2);
    assert(move_to.handle() == from_handle);
    assert(move_from.handle() == nullptr);
    assert(move_to.shape()[0] == shape[0]);
    assert(move_to.wrap_mode() == WrapMode::Repeat);
    assert(move_to.filter_mode() == FilterMode::Nearest);
}
