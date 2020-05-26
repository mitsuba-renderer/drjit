/*

ENOKI_TEST(test23_csch) {
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    set_requires_gradient(x);
    FloatD y = csch(x*x);
    my_backward(y);
    assert(allclose(y, csch(sqr(detach(x)))));
    assert(allclose(gradient(x), -2*detach(x) * csch(sqr(detach(x))) * coth(sqr(detach(x)))));
}

ENOKI_TEST(test24_sech) {
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    set_requires_gradient(x);
    FloatD y = sech(x*x);
    my_backward(y);
    assert(allclose(y, sech(sqr(detach(x)))));
    assert(allclose(gradient(x), -2.f*detach(x) * sech(sqr(detach(x))) * tanh(sqr(detach(x)))));
}

ENOKI_TEST(test25_coth) {
    FloatD x = linspace<FloatD>(-1.f, 1.f, 10);
    set_requires_gradient(x);
    FloatD y = asinh(x*x);
    my_backward(y);
    assert(allclose(y, asinh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2.f*detach(x) * rsqrt(1.f + sqr(sqr(detach(x))))));
}

ENOKI_TEST(test26_acosh) {
    FloatD x = linspace<FloatD>(1.01f, 2.f, 10);
    set_requires_gradient(x);
    FloatD y = acosh(x*x);
    my_backward(y);
    assert(allclose(y, acosh(sqr(detach(x)))));
    assert(allclose(gradient(x), 2.f*detach(x) * rsqrt(sqr(sqr(detach(x))) - 1.f)));
}

ENOKI_TEST(test27_atanh) {
    FloatD x = linspace<FloatD>(-.99f, .99f, 10);
    set_requires_gradient(x);
    FloatD y = atanh(x*x);
    my_backward(y);
    assert(allclose(y, atanh(sqr(detach(x)))));
    assert(allclose(gradient(x), -2.f*detach(x) * rcp(sqr(sqr(detach(x))) - 1.f)));
}

ENOKI_TEST(test28_linear_to_srgb) {
    FloatD x = linspace<FloatD>(0.f, 1.f, 10);
    set_requires_gradient(x);
    FloatD y = linear_to_srgb(x);

    std::cout << graphviz(y) << std::endl;
    my_backward(y);
    /// from mathematica
    FloatX ref{ 12.92f,   1.58374f,  1.05702f, 0.834376f, 0.705474f,
                0.61937f, 0.556879f, 0.50899f, 0.470847f, 0.439583f };
    assert(hmax(abs(gradient(x) - ref)) < 1e-5f);
}

template <typename Vector2>
Vector2 square_to_uniform_disk_concentric(Vector2 sample) {
    using Value  = value_t<Vector2>;
    using Mask   = mask_t<Value>;
    using Scalar = scalar_t<Value>;

    Value x = fmsub(Scalar(2), sample.x(), Scalar(1)),
          y = fmsub(Scalar(2), sample.y(), Scalar(1));

    Mask is_zero         = eq(x, zero<Value>()) &&
                           eq(y, zero<Value>()),
         quadrant_1_or_3 = abs(x) < abs(y);

    Value r  = select(quadrant_1_or_3, y, x),
          rp = select(quadrant_1_or_3, x, y);

    Value phi = rp / r * Scalar(.25f * Float(M_PI));
    masked(phi, quadrant_1_or_3) = Scalar(.5f * Float(M_PI)) - phi;
    masked(phi, is_zero) = zero<Value>();

    auto [sin_phi, cos_phi] = sincos(phi);

    return Vector2(r * cos_phi, r * sin_phi);
}


/// Warp a uniformly distributed square sample to a Beckmann distribution
template <typename Vector2,
          typename Value   = expr_t<value_t<Vector2>>,
          typename Vector3 = Array<Value, 3>>
Vector3 square_to_beckmann(const Vector2 &sample, const value_t<Vector2> &alpha) {
    Vector2 p = square_to_uniform_disk_concentric(sample);
    Value r2 = squared_norm(p);

    Value tan_theta_m_sqr = -alpha * alpha * log(1 - r2);
    Value cos_theta_m = rsqrt(1 + tan_theta_m_sqr);
    p *= safe_sqrt((1 - cos_theta_m * cos_theta_m) / r2);

    return Vector3(p.x(), p.y(), cos_theta_m);
}

ENOKI_TEST(test32_sample_disk) {
    Vector2f x(.2f, .3f);

    Vector2fD y(x);
    set_requires_gradient(y);

    Vector3fD z = square_to_beckmann(y, .4f);
    Vector3f z_ref (-0.223574f, -0.12908f, 0.966102f);
    assert(allclose(detach(z), z_ref));

    auto sum = hsum(z);
    std::cout << graphviz(sum) << std::endl;
    my_backward(sum);
    Float eps = 1e-3f;
    Float dx = hsum(square_to_beckmann(x + Vector2f(eps, 0.f), .4f) -
                    square_to_beckmann(x - Vector2f(eps, 0.f), .4f)) /
                        (2 * eps);
    Float dy = hsum(square_to_beckmann(x + Vector2f(0.f, eps), .4f) -
                    square_to_beckmann(x - Vector2f(0.f, eps), .4f)) /
                        (2 * eps);

    assert(allclose(gradient(y), Vector2f(dx, dy), 1e-3f, 1e-3f));
}

ENOKI_TEST(test33_bcast) {
    FloatD x(5.f);
    FloatD y = arange<FloatD>(10);

    set_requires_gradient(x);
    set_requires_gradient(y);
    set_label(x, "x");
    set_label(y, "y");
    FloatD z = hsum(sqr(sin(x)*cos(y)));
    my_backward(z);

    assert(allclose(gradient(x), -2.8803, 1e-4f, 1e-4f));
    assert(allclose(gradient(y),
                    FloatX(-0.0000, -0.8361, 0.6959, 0.2569, -0.9098, 0.5002,
                           0.4934, -0.9109, 0.2647, 0.6906), 1e-4f, 1e-4f));
}

ENOKI_TEST(test34_gradient_descent) {
    FloatD x = zero<FloatD>(10);
    set_label(x, "x");
    float loss_f = 0.f;
    for (size_t i = 0; i < 10; ++i) {
        set_requires_gradient(x);
        FloatD loss = norm(x - linspace<FloatD>(0, 1, 10));
        my_backward(loss);
        x = detach(x) - gradient(x)*2e-1f;
        loss_f = detach(loss)[0];
    }
    assert(loss_f < 1e-1f);
}
