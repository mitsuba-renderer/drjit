#include "test.h"
#include <drjit/autodiff.h>
#include <drjit/random.h>
#include <drjit/texture.h>
#include <drjit/matrix.h>

namespace dr = drjit;

using Float = dr::CUDAArray<float>;
using DFloat = dr::DiffArray<Float>;
using Array1f = dr::Array<Float, 1>;
using Array2f = dr::Array<Float, 2>;
using Array3f = dr::Array<Float, 3>;
using Array4f = dr::Array<Float, 4>;
using ArrayD1f = dr::Array<DFloat, 1>;
using ArrayD2f = dr::Array<DFloat, 2>;
using ArrayD3f = dr::Array<DFloat, 3>;
using ArrayD4f = dr::Array<DFloat, 4>;
using MatrixD3f = dr::Matrix<DFloat, 3>;
using FloatX = dr::DynamicArray<Float>;
using FloatDX = dr::DynamicArray<DFloat>;

#define CHECK_CUDA_AVAILABLE()                     \
    jit_init(JitBackend::CUDA);                    \
    jit_set_log_level_stderr(::LogLevel::Error);   \
    if (!jit_has_backend(JitBackend::CUDA))        \
        return;

void test_interp_1d_wrap(WrapMode wrap_mode) {
    for (int k = 0; k < 2; ++k) {
        jit_set_flag(JitFlag::ForceOptiX, k == 1);

        size_t shape[1] = { 2 };
        dr::Texture<Float, 1> tex(shape, 1, true, FilterMode::Linear, wrap_mode);
        tex.set_value(Float(0.f, 1.f));

        size_t N = 11;

        Float ref = dr::linspace<Float>(0.f, 1.f, N);
        Array1f pos(dr::linspace<Float>(0.25f, 0.75f, N));

        Array1f output = empty<Array1f>(N);

        tex.eval_nonaccel(pos, output.data());
        assert(dr::allclose(output.x(), ref));
        tex.eval_cuda(pos, output.data());
        assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));

        switch (wrap_mode) {
            case WrapMode::Repeat: {
                pos.x() = dr::linspace<Float>(-0.75f, -0.25f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));

                pos.x() = dr::linspace<Float>(1.25f, 1.75f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));
                break;
            }
            case WrapMode::Clamp: {
                ref = dr::opaque<Float>(0.f, N);
                pos.x() = dr::linspace<Float>(-0.25f, 0.25f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));

                ref = dr::opaque<Float>(1.f, N);
                pos.x() = dr::linspace<Float>(0.75f, 1.25f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));
                break;
            }
            case WrapMode::Mirror: {
                pos.x() = dr::linspace<Float>(-0.25f, -0.75f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));

                pos.x() = dr::linspace<Float>(1.75f, 1.25f, N);
                tex.eval_nonaccel(pos, output.data());
                assert(dr::allclose(output.x(), ref));
                tex.eval_cuda(pos, output.data());
                assert(dr::allclose(output.x(), ref, 5e-3f, 5e-3f));
                break;
            }
        }
    }

    jit_set_flag(JitFlag::ForceOptiX, false);
}

DRJIT_TEST(test01_interp_1d) {
    CHECK_CUDA_AVAILABLE()

    jit_init(JitBackend::CUDA);

    test_interp_1d_wrap(WrapMode::Repeat);
    test_interp_1d_wrap(WrapMode::Clamp);
    test_interp_1d_wrap(WrapMode::Mirror);
}

DRJIT_TEST(test02_interp_1d) {
    CHECK_CUDA_AVAILABLE()

    for (int ch = 1; ch <= 8; ++ch) {
        size_t shape[] = { 123 };
        PCG32<Float> rng_1(shape[0] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 1> tex(shape, ch, true, FilterMode::Linear,
                                  wrap_modes[i]);

            for (int j = 0; j < 4; ++j) {
                Float values = rng_1.next_float32();
                tex.set_value(values);
                assert(allclose(tex.value(), values));
                Array1f pos(rng_2.next_float32());
                FloatX result_drjit = empty<FloatX>(ch);
                tex.eval_nonaccel(pos, result_drjit.data());
                dr::eval(result_drjit);
                FloatX result_cuda = empty<FloatX>(ch);
                tex.eval_cuda(pos, result_cuda.data());
                dr::eval(result_cuda);

                assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
            }
        }
    }
}

DRJIT_TEST(test03_interp_2d) {
    CHECK_CUDA_AVAILABLE()

    for (int ch = 1; ch <= 8; ++ch) {
        size_t shape[] = { 123, 456 };
        PCG32<Float> rng_1(shape[0] * shape[1] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 2> tex(shape, ch, true, FilterMode::Linear,
                                  wrap_modes[i]);

            for (int j = 0; j < 4; ++j) {
                tex.set_value(rng_1.next_float32());
                Array2f pos(rng_2.next_float32(), rng_2.next_float32());
                FloatX result_drjit = empty<FloatX>(ch);
                tex.eval_nonaccel(pos, result_drjit.data());
                dr::eval(result_drjit);
                FloatX result_cuda = empty<FloatX>(ch);
                tex.eval_cuda(pos, result_cuda.data());
                dr::eval(result_cuda);

                assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
            }
        }
    }
}

DRJIT_TEST(test04_interp_3d) {
    CHECK_CUDA_AVAILABLE()

    for (int ch = 1; ch <= 8; ++ch) {
        size_t shape[] = { 123, 456, 12 };
        PCG32<Float> rng_1(shape[0] * shape[1] * shape[2] * ch);
        PCG32<Float> rng_2(1024);
        Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                      WrapMode::Mirror);

        for (size_t i = 0; i < wrap_modes.size(); ++i) {
            Texture<Float, 3> tex(shape, ch, true, FilterMode::Linear,
                                  wrap_modes[i]);

            for (int j = 0; j < 4; ++j) {
                tex.set_value(rng_1.next_float32());
                Array3f pos(rng_2.next_float32(), rng_2.next_float32(),
                            rng_2.next_float32());
                FloatX result_drjit = empty<FloatX>(ch);
                tex.eval_nonaccel(pos, result_drjit.data());
                dr::eval(result_drjit);
                FloatX result_cuda = empty<FloatX>(ch);
                tex.eval_cuda(pos, result_cuda.data());
                dr::eval(result_cuda);

                assert(dr::allclose(result_drjit, result_cuda, 6e-3f, 6e-3f));
            }
        }
    }
}

void test_grad(bool migrate) {
    size_t shape[] = { 3 };
    Texture<DFloat, 1> tex(shape, 1, true);

    DFloat value(3, 5, 8);
    dr::enable_grad(value);
    tex.set_value(value, migrate);

    ArrayD1f pos(1 / 6.f * 0.25f + (1 / 6.f + 1 / 3.f) * 0.75f);
    DFloat expected(0.25f * 3 + 0.75f * 5);
    // check migration
    ArrayD1f out2 = empty<ArrayD1f>();
    tex.eval_nonaccel(pos, out2.data());
    if (migrate) {
        assert(dr::allclose(out2.x(), 0));
    } else
        assert(dr::allclose(out2.x(), expected, 5e-3f, 5e-3f));

    ArrayD1f out = empty<ArrayD1f>();
    tex.eval(pos, out.data());
    dr::backward(out.x());

    assert(dr::allclose(dr::grad(value), DFloat(.25f, .75f, 0)));
    assert(dr::allclose(out.x(), expected, 5e-3f, 5e-3f));
    assert(dr::allclose(tex.value(), value));
}

DRJIT_TEST(test05_grad) {
    CHECK_CUDA_AVAILABLE()

    test_grad(true);
    test_grad(false);
}

DRJIT_TEST(test06_nearest) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 3 };
    dr::Texture<Float, 1> tex(shape, 1, true, FilterMode::Nearest);
    tex.set_value(Float(0.f, 0.5f, 1.f));

    Float pos = dr::linspace<Float>(0, 1, 80);
    Array1f out_cuda = empty<Array1f>();
    tex.eval_cuda(pos, out_cuda.data());
    Array1f out_drjit = empty<Array1f>();
    tex.eval_nonaccel(pos, out_drjit.data());
    assert(dr::allclose(out_cuda.x(), out_drjit.x()));
}

DRJIT_TEST(test07_cubic_analytic) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 4 };
    dr::Texture<DFloat, 1> tex(shape, 1, true, FilterMode::Linear, WrapMode::Clamp);
    tex.set_value(DFloat(0.f, 1.f, 0.f, 0.f));

    ArrayD1f pos(0.5f);
    dr::Array<DFloat, 1> val_64 = empty<dr::Array<DFloat, 1>>();
    dr::Array<ArrayD1f, 1> grad_64 = empty<dr::Array<ArrayD1f, 1>>();
    dr::eval(grad_64);
    tex.eval_cubic_grad(pos, val_64.data(), grad_64.data());
    dr::enable_grad(pos);

    ArrayD1f res = empty<ArrayD1f>();
    tex.eval_cubic(pos, res.data(), true, true);

    dr::backward(res.x());
    auto grad_ad = dr::grad(pos);
    ArrayD1f res2 = empty<ArrayD1f>();
    tex.eval_cubic_helper(pos, res2.data());

    // 1/6 * (3*a^3 - 6*a^2 + 4) with a=0.5
    Array1f ref_res(0.479167f);
    assert(dr::allclose(res, ref_res, 1e-5f, 1e-5f));
    assert(dr::allclose(res2, ref_res, 1e-5f, 1e-5f));
    // 1/6 * (9*a^2 - 12*a) with a=0.5
    Float ref_grad(-0.625f * 4.0f);
    assert(dr::allclose(grad_64[0][0], ref_grad, 1e-5f, 1e-5f));
    assert(dr::allclose(grad_ad[0], ref_grad, 1e-5f, 1e-5f));
}

void test_cubic_interp_1d(WrapMode wrap_mode) {
    size_t shape[1] = { 5 };
    dr::Texture<Float, 1> tex(shape, 1, true, FilterMode::Linear, wrap_mode);
    tex.set_value(Float(2.f, 1.f, 3.f, 4.f, 7.f));

    size_t N = 20;

    Array1f pos(dr::linspace<Float>(0.1f, 0.9f, N));
    Array1f out = empty<Array1f>();
    tex.eval_cubic_helper(pos, out.data());
    Float ref = out.x();

    tex.eval_cubic(pos, out.data(), true, true);
    assert(dr::allclose(out.x(), ref));

    switch (wrap_mode) {
        case WrapMode::Repeat: {
            Array1f res = empty<Array1f>();
            Array1f res2 = empty<Array1f>();

            pos.x() = dr::linspace<Float>(-0.9f, -0.1f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));

            pos.x() = dr::linspace<Float>(1.1f, 1.9f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));
            break;
        }
        case WrapMode::Clamp: {
            Array1f res = empty<Array1f>();
            Array1f res2 = empty<Array1f>();

            {
                Array1f pos_2(dr::linspace<Float>(0.f, 1.f, N));
                tex.eval_cubic(pos_2, res.data(), true, true);
                tex.eval_cubic_helper(pos_2, res2.data());

                Float ref_2 = Float(1.9792, 1.9259, 1.8198, 1.6629, 1.5168,
                                    1.4546, 1.5485, 1.8199, 2.2043, 2.6288,
                                    3.0232, 3.3783, 3.7461, 4.1814, 4.7305,
                                    5.3536, 5.9603, 6.4595, 6.7778, 6.9375);
                assert(dr::allclose(res.x(), ref_2, 5e-3f, 5e-3f));
                assert(dr::allclose(res2.x(), ref_2, 5e-3f, 5e-3f));
            }

            ref = dr::opaque<Float>(2.f, N);
            pos.x() = dr::linspace<Float>(-1.f, -0.1f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));

            ref = dr::opaque<Float>(7.f, N);
            pos.x() = dr::linspace<Float>(1.1f, 2.f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));
            break;
        }
        case WrapMode::Mirror: {
            Array1f res = empty<Array1f>();
            Array1f res2 = empty<Array1f>();

            pos.x() = dr::linspace<Float>(-0.1f, -0.9f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));

            pos.x() = dr::linspace<Float>(1.9f, 1.1f, N);
            tex.eval_cubic(pos, res.data(), true, true);
            tex.eval_cubic_helper(pos, res2.data());
            assert(dr::allclose(res.x(), ref));
            assert(dr::allclose(res2.x(), ref));
            break;
        }
    }
}

DRJIT_TEST(test08_cubic_interp_1d) {
    CHECK_CUDA_AVAILABLE()

    test_cubic_interp_1d(WrapMode::Clamp);
    test_cubic_interp_1d(WrapMode::Repeat);
    test_cubic_interp_1d(WrapMode::Mirror);
}

DRJIT_TEST(test09_cubic_interp_2d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[2] = { 5, 4 };
    Array<WrapMode, 3> wrap_modes(WrapMode::Repeat, WrapMode::Clamp,
                                  WrapMode::Mirror);

    for (size_t i = 0; i < wrap_modes.size(); ++i) {
        Texture<Float, 2> tex(shape, 1, true, FilterMode::Linear, wrap_modes[i]);
        PCG32<Float> rng1(shape[0] * shape[1]);
        tex.set_value(rng1.next_float32());

        Array1f res = empty<Array1f>();
        Array1f res2 = empty<Array1f>();

        PCG32<Float> rng2(1024);
        Array2f pos(rng2.next_float32(), rng2.next_float32());
        tex.eval_cubic(pos, res.data(), true, true);
        tex.eval_cubic_helper(pos, res2.data());
        assert(dr::allclose(res, res2));
    }
}

DRJIT_TEST(test10_cubic_interp_3d) {
    CHECK_CUDA_AVAILABLE()

    using TensorXf = Tensor<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    const int s = 9;
    size_t shape[4] = { s, s, s, 2 };  // 2 channels

    auto data = dr::full<Float>((1.0), s*s*s*2);
    TensorXf tensor(data, 4, shape);
    dr::scatter(tensor.array(), Float(0.0),  UInt32(728));  // tensor[4, 4, 4, 0] = 0.0
    dr::scatter(tensor.array(), Float(2.0),  UInt32(546));  // tensor[3, 3, 3, 0] = 2.0
    dr::scatter(tensor.array(), Float(10.0), UInt32(727));  // tensor[4, 4, 3, 1] = 10.0

    dr::Texture<Float, 3> tex(tensor, true, false, FilterMode::Linear, WrapMode::Clamp);

    Array2f res = empty<Array2f>();
    Array2f res2 = empty<Array2f>();

    Array2f ref(0.71312, 1.86141);
    Array3f pos(.49f, .5f, .5f);
    tex.eval_cubic(pos, res.data(), true, true);
    tex.eval_cubic_helper(pos, res2.data());
    assert(dr::allclose(res, ref));
    assert(dr::allclose(res2, ref));

    Array2f ref2(0.800905, 2.60136);
    Array3f pos2(.45f, .53f, .51f);
    tex.eval_cubic(pos2, res.data(), true, true);
    tex.eval_cubic_helper(pos2, res2.data());
    assert(dr::allclose(res, ref2));
    assert(dr::allclose(res2, ref2));
}

DRJIT_TEST(test11_cubic_grad_pos) {
    CHECK_CUDA_AVAILABLE()

    using TensorXf = Tensor<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    size_t shape[4] = { 4, 4, 4, 1 };

    auto data = dr::full<Float>((1.0), 4*4*4);
    TensorXf tensor(data, 4, shape);
    dr::scatter(tensor.array(), Float(0.5f), UInt32(21));  // data[1, 1, 1] = 0.5
    dr::scatter(tensor.array(), Float(2.0f), UInt32(25));  // data[1, 2, 1] = 2.0
    dr::scatter(tensor.array(), Float(3.0f), UInt32(41));  // data[2, 2, 1] = 3.0
    dr::scatter(tensor.array(), Float(4.0f), UInt32(22));  // data[1, 1, 2] = 4.0

    dr::Texture<DFloat, 3> tex(tensor, true, false, FilterMode::Linear,
                               WrapMode::Clamp);

    ArrayD3f pos(.5f, .5f, .5f);
    dr::Array<DFloat, 1> val_64 = empty<dr::Array<DFloat, 1>>();
    dr::Array<ArrayD3f, 1> grad_64 = empty<dr::Array<ArrayD3f, 1>>();
    tex.eval_cubic_grad(pos, val_64.data(), grad_64.data());
    dr::enable_grad(pos);

    ArrayD1f res = empty<ArrayD1f>();
    tex.eval_cubic(pos, res.data(), true, true);
    dr::backward(res.x());

    assert(dr::allclose(res, val_64));
    auto grad_ad = dr::grad(pos);
    ArrayD1f res2 = empty<ArrayD1f>();
    tex.eval_cubic_helper(pos, res2.data());

    Array1f ref_res(1.60509f);
    assert(dr::allclose(res, ref_res));
    assert(dr::allclose(res2, ref_res));
    Array3f ref_grad(0.07175f, 0.07175f, -0.21525f);
    ref_grad *= 4.0f;
    assert(dr::allclose(grad_64[0][0], ref_grad[0]));
    assert(dr::allclose(grad_64[0][1], ref_grad[1]));
    assert(dr::allclose(grad_64[0][2], ref_grad[2]));
    assert(dr::allclose(grad_ad, ref_grad));
}

DRJIT_TEST(test12_cubic_hessian_pos) {
    CHECK_CUDA_AVAILABLE()

    using TensorXf = Tensor<Float>;
    using UInt32 = dr::uint32_array_t<Float>;
    size_t shape[4] = { 4, 4, 4, 1 };

    auto data = dr::full<Float>((0.0), 4*4*4);
    TensorXf tensor(data, 4, shape);
    dr::scatter(tensor.array(), Float(1.0f), UInt32(21));  // data[1, 1, 1] = 1.0
    dr::scatter(tensor.array(), Float(2.0f), UInt32(37));  // data[2, 1, 1] = 2.0
    // NOTE: Tensor has different index convention with Texture
    //       [2, 1, 1] is equivalent to (x=1, y=1, z=2) in the texture

    dr::Texture<DFloat, 3> tex(tensor, true, false, FilterMode::Linear,
                               WrapMode::Clamp);

    ArrayD3f pos(.5f, .5f, .5f);
    dr::Array<DFloat, 1> val_64 = empty<dr::Array<DFloat, 1>>();
    dr::Array<ArrayD3f, 1> grad_64 = empty<dr::Array<ArrayD3f, 1>>();
    tex.eval_cubic_grad(pos, val_64.data(), grad_64.data(), true);

    dr::Array<DFloat, 1> value_h = empty<dr::Array<DFloat, 1>>();
    dr::Array<ArrayD3f, 1> grad_h = empty<dr::Array<ArrayD3f, 1>>();
    dr::Array<MatrixD3f, 1> hessian = empty<dr::Array<MatrixD3f, 1>>();
    tex.eval_cubic_hessian(pos, value_h.data(), grad_h.data(), hessian.data(), true);

    assert(dr::allclose(val_64[0], value_h[0]));

    assert(dr::allclose(grad_64[0][0], grad_h[0][0]));
    assert(dr::allclose(grad_64[0][1], grad_h[0][1]));
    assert(dr::allclose(grad_64[0][2], grad_h[0][2]));
    // compare with analytical solution
    // note: hessian[ch][grad1][grad2]
    // note: multiply analytical result by 16.0f=4.f*4.f to account for the resolution transformation
    assert(dr::allclose(hessian[0][0][0], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(dr::allclose(hessian[0][0][1],  0.561523f * 16.0f, 1e-5f, 1e-5f));
    assert(dr::allclose(hessian[0][0][2], -0.187174f * 16.0f, 1e-5f, 1e-5f));
    assert(dr::allclose(hessian[0][1][1], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(dr::allclose(hessian[0][1][2], -0.187174f * 16.0f, 1e-5f, 1e-5f));
    assert(dr::allclose(hessian[0][2][2], -0.344401f * 16.0f, 1e-5f, 1e-5f));
    assert(hessian[0][0][1] == hessian[0][1][0]);
    assert(hessian[0][0][2] == hessian[0][2][0]);
    assert(hessian[0][1][2] == hessian[0][2][1]);
}

DRJIT_TEST(test13_move_assignment) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 2 };
    dr::Texture<Float, 1> move_from(shape, 1, true, FilterMode::Nearest, WrapMode::Repeat);
    move_from.set_value(Float(0.f, 1.f));
    const void *from_handle = move_from.handle();

    dr::Texture<Float, 1> move_to;
    move_to = std::move(move_from);

    assert(move_to.ndim() == 2);
    assert(move_to.handle() == from_handle);
    assert(move_from.handle() == nullptr);
    assert(move_to.shape()[0] == shape[0]);
    assert(move_to.wrap_mode() == WrapMode::Repeat);
    assert(move_to.filter_mode() == FilterMode::Nearest);
}

DRJIT_TEST(test14_move_constructor) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 2 };
    dr::Texture<Float, 1> move_from(shape, 1, true, FilterMode::Nearest, WrapMode::Repeat);
    move_from.set_value(Float(0.f, 1.f));
    const void *from_handle = move_from.handle();

    dr::Texture<Float, 1> move_to(std::move(move_from));

    assert(move_to.ndim() == 2);
    assert(move_to.handle() == from_handle);
    assert(move_from.handle() == nullptr);
    assert(move_to.shape()[0] == shape[0]);
    assert(move_to.wrap_mode() == WrapMode::Repeat);
    assert(move_to.filter_mode() == FilterMode::Nearest);
}

DRJIT_TEST(test15_tensor_value_1d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 2 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        PCG32<Float> rng(shape[0] * ch);
        dr::Texture<Float, 1> tex(shape, ch, true);

        Float tex_data = rng.next_float32();
        tex.set_value(tex_data);

        assert(allclose(tex.value(), tex_data));
        assert(allclose(tex.tensor().array(), tex_data));
    }
}

DRJIT_TEST(test16_tensor_value_2d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[2] = { 2, 3 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        PCG32<Float> rng(shape[0] * shape[1] * ch);
        dr::Texture<Float, 2> tex(shape, ch, true);

        Float tex_data = rng.next_float32();
        tex.set_value(tex_data);

        assert(allclose(tex.value(), tex_data));
        assert(allclose(tex.tensor().array(), tex_data));
    }
}

DRJIT_TEST(test17_tensor_value_3d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[3] = { 2, 3, 4 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        PCG32<Float> rng(shape[0] * shape[1] * shape[2] * ch);
        dr::Texture<Float, 3> tex(shape, ch, true);

        Float tex_data = rng.next_float32();
        tex.set_value(tex_data);

        assert(allclose(tex.value(), tex_data));
        assert(allclose(tex.tensor().array(), tex_data));
    }
}

DRJIT_TEST(test18_fetch_1d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[1] = { 2 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        dr::Texture<Float, 1> tex(shape, ch, true);
        PCG32<Float> rng(shape[0] * ch);
        Float tex_data = rng.next_float32();
        dr::enable_grad(tex_data);
        tex.set_value(tex_data);

        Array1f pos(0.5f);
        Array<Float *, 2> out_drjit;
        Array<Float *, 2> out_cuda;
        FloatX out0_drjit = empty<FloatX>(ch);
        FloatX out1_drjit = empty<FloatX>(ch);
        FloatX out0_cuda = empty<FloatX>(ch);
        FloatX out1_cuda = empty<FloatX>(ch);
        out_drjit[0] = out0_drjit.data();
        out_drjit[1] = out1_drjit.data();
        out_cuda[0] = out0_cuda.data();
        out_cuda[1] = out1_cuda.data();

        tex.eval_fetch_nonaccel(pos, out_drjit);
        tex.eval_fetch_cuda(pos, out_cuda);
        for (size_t k = 0; k < ch; ++k) {
            assert(allclose(tex_data[k], out0_drjit[k]));
            assert(allclose(tex_data[k], out0_cuda[k]));
            assert(allclose(tex_data[ch + k], out1_drjit[k]));
            assert(allclose(tex_data[ch + k], out1_cuda[k]));
        }
    }
}

DRJIT_TEST(test19_fetch_2d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[2] = { 2, 2 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        dr::Texture<Float, 2> tex(shape, ch, true);
        PCG32<Float> rng(shape[0] * shape[1] * ch);
        rng.next_float32();
        Float tex_data = rng.next_float32();
        tex.set_value(tex_data);

        Array2f pos(0.5f, 0.5f);
        Array<Float *, 4> out_drjit;
        Array<Float *, 4> out_cuda;
        FloatX out00_drjit = empty<FloatX>(ch);
        FloatX out01_drjit = empty<FloatX>(ch);
        FloatX out10_drjit = empty<FloatX>(ch);
        FloatX out11_drjit = empty<FloatX>(ch);
        FloatX out00_cuda = empty<FloatX>(ch);
        FloatX out01_cuda = empty<FloatX>(ch);
        FloatX out10_cuda = empty<FloatX>(ch);
        FloatX out11_cuda = empty<FloatX>(ch);
        out_drjit[0] = out00_drjit.data();
        out_drjit[1] = out01_drjit.data();
        out_drjit[2] = out10_drjit.data();
        out_drjit[3] = out11_drjit.data();
        out_cuda[0] = out00_cuda.data();
        out_cuda[1] = out01_cuda.data();
        out_cuda[2] = out10_cuda.data();
        out_cuda[3] = out11_cuda.data();

        tex.eval_fetch_nonaccel(pos, out_drjit);
        tex.eval_fetch_cuda(pos, out_cuda);
        for (size_t k = 0; k < ch; ++k) {
            assert(allclose(tex_data[k], out00_drjit[k]));
            assert(allclose(tex_data[k], out00_cuda[k]));
            assert(allclose(tex_data[ch + k], out01_drjit[k]));
            assert(allclose(tex_data[ch + k], out01_cuda[k]));
            assert(allclose(tex_data[2 * ch + k], out10_drjit[k]));
            assert(allclose(tex_data[2 * ch + k], out10_cuda[k]));
            assert(allclose(tex_data[3 * ch + k], out11_drjit[k]));
            assert(allclose(tex_data[3 * ch + k], out11_cuda[k]));
        }
    }
}

DRJIT_TEST(test20_fetch_3d) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[3] = { 2, 2, 2 };
    for (size_t ch = 1; ch <= 8; ++ch) {
        dr::Texture<Float, 3> tex(shape, ch, true);
        PCG32<Float> rng(shape[0] * shape[1] * shape[2] * ch);
        Float tex_data = rng.next_float32();
        tex.set_value(tex_data);

        Array3f pos(0.3f, 0.3f, 0.3f);
        Array<Float *, 8> out_drjit;
        Array<Float *, 8> out_cuda;
        FloatX out000_drjit = empty<FloatX>(ch);
        FloatX out001_drjit = empty<FloatX>(ch);
        FloatX out010_drjit = empty<FloatX>(ch);
        FloatX out011_drjit = empty<FloatX>(ch);
        FloatX out100_drjit = empty<FloatX>(ch);
        FloatX out101_drjit = empty<FloatX>(ch);
        FloatX out110_drjit = empty<FloatX>(ch);
        FloatX out111_drjit = empty<FloatX>(ch);

        FloatX out000_cuda = empty<FloatX>(ch);
        FloatX out001_cuda = empty<FloatX>(ch);
        FloatX out010_cuda = empty<FloatX>(ch);
        FloatX out011_cuda = empty<FloatX>(ch);
        FloatX out100_cuda = empty<FloatX>(ch);
        FloatX out101_cuda = empty<FloatX>(ch);
        FloatX out110_cuda = empty<FloatX>(ch);
        FloatX out111_cuda = empty<FloatX>(ch);

        out_drjit[0] = out000_drjit.data();
        out_drjit[1] = out001_drjit.data();
        out_drjit[2] = out010_drjit.data();
        out_drjit[3] = out011_drjit.data();
        out_drjit[4] = out100_drjit.data();
        out_drjit[5] = out101_drjit.data();
        out_drjit[6] = out110_drjit.data();
        out_drjit[7] = out111_drjit.data();

        out_cuda[0] = out000_cuda.data();
        out_cuda[1] = out001_cuda.data();
        out_cuda[2] = out010_cuda.data();
        out_cuda[3] = out011_cuda.data();
        out_cuda[4] = out100_cuda.data();
        out_cuda[5] = out101_cuda.data();
        out_cuda[6] = out110_cuda.data();
        out_cuda[7] = out111_cuda.data();

        tex.eval_fetch_nonaccel(pos, out_drjit);
        tex.eval_fetch_cuda(pos, out_cuda);
        for (size_t k = 0; k < ch; ++k) {
            assert(allclose(tex_data[k], out000_drjit[k]));
            assert(allclose(tex_data[k], out000_cuda[k]));
            assert(allclose(tex_data[ch + k], out001_drjit[k]));
            assert(allclose(tex_data[ch + k], out001_cuda[k]));
            assert(allclose(tex_data[2 * ch + k], out010_drjit[k]));
            assert(allclose(tex_data[2 * ch + k], out010_cuda[k]));
            assert(allclose(tex_data[3 * ch + k], out011_drjit[k]));
            assert(allclose(tex_data[3 * ch + k], out011_cuda[k]));
            assert(allclose(tex_data[4 * ch + k], out100_drjit[k]));
            assert(allclose(tex_data[4 * ch + k], out100_cuda[k]));
            assert(allclose(tex_data[5 * ch + k], out101_drjit[k]));
            assert(allclose(tex_data[5 * ch + k], out101_cuda[k]));
            assert(allclose(tex_data[6 * ch + k], out110_drjit[k]));
            assert(allclose(tex_data[6 * ch + k], out110_cuda[k]));
            assert(allclose(tex_data[7 * ch + k], out111_drjit[k]));
            assert(allclose(tex_data[7 * ch + k], out111_cuda[k]));
        }
    }
}

void test_fetch_migrate(bool migrate) {
    size_t shape[1] = { 2 };
    dr::Texture<Float, 1> tex(shape, 1, true);
    Float tex_data(1.0f, 2.0f);
    tex.set_value(tex_data, migrate);

    Array1f pos(0.5f);
    Array<Float *, 2> out;
    FloatX out0 = empty<FloatX>(1);
    FloatX out1 = empty<FloatX>(1);
    out[0] = out0.data();
    out[1] = out1.data();

    tex.eval_fetch_nonaccel(pos, out);
    if (migrate) {
        assert(allclose(out[0][0], 0));
        assert(allclose(out[1][0], 0));
    } else {
        assert(allclose(out[0][0], 1.0f));
        assert(allclose(out[1][0], 2.0f));
    }
}

DRJIT_TEST(test21_fetch_migrate) {
    CHECK_CUDA_AVAILABLE()

    test_fetch_migrate(true);
    test_fetch_migrate(false);
}

DRJIT_TEST(test22_fetch_grad) {
    CHECK_CUDA_AVAILABLE()

    size_t shape[2] = { 2, 2 };
    Texture<DFloat, 2> tex(shape, 1, true);

    DFloat tex_data(1.f, 2.f, 3.f, 4.f);
    dr::enable_grad(tex_data);
    tex.set_value(tex_data);

    ArrayD2f pos(0.5f, 0.5f);
    Array<DFloat *, 4> out;
    FloatDX out00 = empty<FloatDX>(1);
    FloatDX out01 = empty<FloatDX>(1);
    FloatDX out10 = empty<FloatDX>(1);
    FloatDX out11 = empty<FloatDX>(1);
    out[0] = out00.data();
    out[1] = out01.data();
    out[2] = out10.data();
    out[3] = out11.data();
    tex.eval_fetch_nonaccel(pos, out);
    assert(allclose(1.f, out[0][0]));
    assert(allclose(2.f, out[1][0]));
    assert(allclose(3.f, out[2][0]));
    assert(allclose(4.f, out[3][0]));

    tex.eval_fetch(pos, out);
    assert(allclose(1.f, out[0][0]));
    assert(allclose(2.f, out[1][0]));
    assert(allclose(3.f, out[2][0]));
    assert(allclose(4.f, out[3][0]));

    for (size_t i = 0; i < 4; ++i) {
        dr::backward(out[i][0]);
        Float grad = dr::grad(tex_data);
        Float expected(
                i == 0 ? 1.f : 0.f,
                i == 1 ? 1.f : 0.f,
                i == 2 ? 1.f : 0.f,
                i == 3 ? 1.f : 0.f
        );

        assert(allclose(expected, grad));
        dr::set_grad(tex_data, Float(0, 0, 0, 0));
    }
}

DRJIT_TEST(test23_set_tensor) {
    CHECK_CUDA_AVAILABLE()

    using TensorXf = Tensor<Float>;
    size_t shape[2] = { 2, 2 };
    Texture<Float, 2> tex(shape, 1, true);

    Float tex_data(1.f, 2.f, 3.f, 4.f);
    tex.set_value(tex_data);

    size_t new_shape[3] = { 2, 3, 2 };
    Float new_tex_data(6.5f, 6.f, 5.5f, 5.f, 4.5f, 4.f, 3.5f, 3.f, 2.5f, 2.f,
                       1.5f, 1.f);
    TensorXf new_tensor(new_tex_data, 3, new_shape);
    tex.set_tensor(new_tensor);

    Array2f pos(0.f, 0.f);
    Array2f result_drjit;
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    Array2f result_cuda;
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(6.5, 6.f)));

    pos = Array2f(1.f, 1.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(1.5, 1.f)));

    pos = Array2f(0.f, 1.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(3.5, 3.f)));

    pos = Array2f(1.f, 0.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(4.5, 4.f)));
}

DRJIT_TEST(test24_set_tensor_inplace) {
    CHECK_CUDA_AVAILABLE()

    using TensorXf = Tensor<Float>;
    size_t shape[2] = { 2, 2 };
    Texture<Float, 2> tex(shape, 1, true);

    Float tex_data(1.f, 2.f, 3.f, 4.f);
    tex.set_value(tex_data);
    TensorXf &tex_tensor = tex.tensor();

    size_t new_shape1[3] = { 2, 3, 2 };
    Float new_tex_data1(6.5f, 6.f, 5.5f, 5.f, 4.5f, 4.f, 3.5f, 3.f, 2.5f, 2.f,
                       1.5f, 1.f);
    TensorXf new_tensor(new_tex_data1, 3, new_shape1);

    tex_tensor = std::move(new_tensor);
    tex.set_tensor(tex_tensor);

    Array2f pos(0.f, 0.f);
    Array2f result_drjit;
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    Array2f result_cuda;
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(6.5, 6.f)));

    pos = Array2f(1.f, 1.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(1.5, 1.f)));

    pos = Array2f(0.f, 1.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(3.5, 3.f)));

    pos = Array2f(1.f, 0.f);
    tex.eval_nonaccel(pos, result_drjit.data());
    dr::eval(result_drjit);
    tex.eval_cuda(pos, result_cuda.data());
    dr::eval(result_cuda);
    assert(dr::allclose(result_drjit, result_cuda, 5e-3f, 5e-3f));
    assert(dr::allclose(result_drjit, Array2f(4.5, 4.f)));
}
