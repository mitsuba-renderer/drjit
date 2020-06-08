import enoki as ek
import pytest
import gc


@pytest.fixture(scope="module", params=[ek.cuda.ad, ek.llvm.ad])
def m(request):
    gc.collect()
    gc.collect()
    ek.set_device(0 if 'cuda' in request.param.__name__ else -1)
    yield request.param
    gc.collect()
    gc.collect()


def test01_add_rev(m):
    a, b = m.Float(1), m.Float(2)
    ek.enable_grad(a, b)
    c = 2 * a + b
    ek.backward(c)
    assert ek.grad(a) == 2
    assert ek.grad(b) == 1


def test02_add_fwd(m):
    a, b = m.Float(1), m.Float(2)
    ek.enable_grad(a, b)
    c = 2 * a + b
    ek.forward(a, retain_graph=True)
    assert ek.grad(c) == 2
    ek.set_grad(c, 101)
    ek.forward(b)
    assert ek.grad(c) == 102


def test03_sub_mul(m):
    a, b, c = m.Float(2), m.Float(3), m.Float(4)
    ek.enable_grad(a, b, c)
    d = a * b - c
    ek.backward(d)
    assert ek.grad(a) == ek.detach(b)
    assert ek.grad(b) == ek.detach(a)
    assert ek.grad(c) == -1


def test04_div(m):
    a, b = m.Float(2), m.Float(3)
    ek.enable_grad(a, b)
    d = a / b
    ek.backward(d)
    assert ek.allclose(ek.grad(a),  1.0 / 3.0)
    assert ek.allclose(ek.grad(b), -2.0 / 9.0)


def test05_hsum_0_rev(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.hsum_async(x*x)
    ek.backward(y)
    assert len(y) == 1 and ek.allclose(y, 95.0/27.0)
    assert ek.allclose(ek.grad(x), 2 * ek.detach(x))


def test06_hsum_0_fwd(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.hsum_async(x*x)
    ek.forward(x)
    assert len(y) == 1 and ek.allclose(ek.detach(y), 95.0/27.0)
    assert len(ek.grad(y)) == 1 and ek.allclose(ek.grad(y), 10)


def test07_hsum_1_rev(m):
    x = ek.linspace(m.Float, 0, 1, 11)
    ek.enable_grad(x)
    y = ek.hsum_async(ek.hsum_async(x)*x)
    ek.backward(y)
    assert ek.allclose(ek.grad(x), 11)


def test08_hsum_1_fwd(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.hsum_async(ek.hsum_async(x)*x)
    ek.forward(x)
    assert ek.allclose(ek.grad(y), 100)


def test09_hsum_2_rev(m):
    x = ek.linspace(m.Float, 0, 1, 11)
    ek.enable_grad(x)
    z = ek.hsum_async(ek.hsum_async(x*x)*x*x)
    ek.backward(z)
    assert ek.allclose(ek.grad(x),
                       [0., 1.54, 3.08, 4.62, 6.16, 7.7,
                        9.24, 10.78, 12.32, 13.86, 15.4])


def test10_hsum_2_fwd(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.hsum_async(ek.hsum_async(x*x)*ek.hsum_async(x*x))
    ek.forward(x)
    assert ek.allclose(ek.grad(y), 1900.0 / 27.0)


def test11_hprod(m):
    x = m.Float(1, 2, 5, 8)
    ek.enable_grad(x)
    y = ek.hprod_async(x)
    ek.backward(y)
    assert len(y) == 1 and ek.allclose(y[0], 80)
    assert ek.allclose(ek.grad(x), [80, 40, 16, 10])


def test11_hmax_rev(m):
    x = m.Float(1, 2, 8, 5, 8)
    ek.enable_grad(x)
    y = ek.hmax_async(x)
    ek.backward(y)
    assert len(y) == 1 and ek.allclose(y[0], 8)
    assert ek.allclose(ek.grad(x), [0, 0, 1, 0, 1])


def test12_hmax_fwd(m):
    x = m.Float(1, 2, 8, 5, 8)
    ek.enable_grad(x)
    y = ek.hmax_async(x)
    ek.forward(x)
    assert len(y) == 1 and ek.allclose(y[0], 8)
    assert ek.allclose(ek.grad(y), [2])  # Approximation


def test13_sqrt(m):
    x = m.Float(1, 4, 16)
    ek.enable_grad(x)
    y = ek.sqrt(x)
    ek.backward(y)
    assert ek.allclose(ek.detach(y), [1, 2, 4])
    assert ek.allclose(ek.grad(x), [.5, .25, .125])


def test14_rsqrt(m):
    x = m.Float(1, .25, 0.0625)
    ek.enable_grad(x)
    y = ek.rsqrt(x)
    ek.backward(y)
    assert ek.allclose(ek.detach(y), [1, 2, 4])
    assert ek.allclose(ek.grad(x), [-.5, -4, -32])


def test15_abs(m):
    x = m.Float(-2, 2)
    ek.enable_grad(x)
    y = ek.abs(x)
    ek.backward(y)
    assert ek.allclose(ek.detach(y), [2, 2])
    assert ek.allclose(ek.grad(x), [-1, 1])


def test16_sin(m):
    x = ek.linspace(m.Float, 0, 10, 10)
    ek.enable_grad(x)
    y = ek.sin(x)
    ek.backward(y)
    assert ek.allclose(ek.detach(y), ek.sin(ek.detach(x)))
    assert ek.allclose(ek.grad(x), ek.cos(ek.detach(x)))


def test17_cos(m):
    x = ek.linspace(m.Float, 0.01, 10, 10)
    ek.enable_grad(x)
    y = ek.cos(x)
    ek.backward(y)
    assert ek.allclose(ek.detach(y), ek.cos(ek.detach(x)))
    assert ek.allclose(ek.grad(x), -ek.sin(ek.detach(x)))


def test18_gather(m):
    x = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x)
    y = ek.gather(m.Float, x*x, m.UInt(1, 1, 2, 3))
    z = ek.hsum_async(y)
    ek.backward(z)
    ref = [0, -1.55556*2, -1.11111, -0.666667, 0, 0, 0, 0, 0, 0]
    assert ek.allclose(ek.grad(x), ref)


def test19_gather_fwd(m):
    x = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x)
    y = ek.gather(m.Float, x*x, m.UInt(1, 1, 2, 3))
    ek.forward(x)
    ref = [-1.55556, -1.55556, -1.11111, -0.666667]
    assert ek.allclose(ek.grad(y), ref)


def test20_scatter_add_rev(m):
    for i in range(3):
        idx1 = ek.arange(m.UInt, 5)
        idx2 = ek.arange(m.UInt, 4) + 3

        x = ek.linspace(m.Float, 0, 1, 5)
        y = ek.linspace(m.Float, 1, 2, 4)
        buf = ek.zero(m.Float, 10)

        if i % 2 == 0:
            ek.enable_grad(buf)
        if i // 2 == 0:
            ek.enable_grad(x, y)

        x.label = "x"
        y.label = "y"
        buf.label = "buf"

        buf2 = m.Float(buf)
        ek.scatter_add(buf2, x, idx1)
        ek.scatter_add(buf2, y, idx2)

        ref_buf = m.Float(0.0000, 0.2500, 0.5000, 1.7500, 2.3333,
                          1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        assert ek.allclose(ref_buf, buf2, atol=1e-4)
        assert ek.allclose(ref_buf, buf, atol=1e-4)

        s = ek.dot_async(buf2, buf2)

        print(ek.graphviz_str(s))

        ek.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 3.5000, 4.6667)
        ref_y = m.Float(3.5000, 4.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert ek.allclose(ek.grad(y), ek.detach(ref_y), atol=1e-4)
            assert ek.allclose(ek.grad(x), ek.detach(ref_x), atol=1e-4)
        else:
            assert ek.grad(x) is None
            assert ek.grad(y) is None

        if i % 2 == 0:
            assert ek.allclose(ek.grad(buf), ek.detach(ref_buf) * 2, atol=1e-4)
        else:
            assert ek.grad(buf) is None


def test21_scatter_add_fwd(m):
    for i in range(3):
        idx1 = ek.arange(m.UInt, 5)
        idx2 = ek.arange(m.UInt, 4) + 3

        x = ek.linspace(m.Float, 0, 1, 5)
        y = ek.linspace(m.Float, 1, 2, 4)
        buf = ek.zero(m.Float, 10)

        if i % 2 == 0:
            ek.enable_grad(buf)
            ek.set_grad(buf, 1)
        if i // 2 == 0:
            ek.enable_grad(x, y)
            ek.set_grad(x, 1)
            ek.set_grad(y, 1)

        x.label = "x"
        y.label = "y"
        buf.label = "buf"

        buf2 = m.Float(buf)
        ek.scatter_add(buf2, x, idx1)
        ek.scatter_add(buf2, y, idx2)

        s = ek.dot_async(buf2, buf2)

        if i % 2 == 0:
            ek.enqueue(buf)
        if i // 2 == 0:
            ek.enqueue(x, y)

        ek.traverse(m.Float, reverse=False)

        # Verified against Mathematica
        assert ek.allclose(ek.detach(s), 15.5972)
        assert ek.allclose(ek.grad(s), (25.1667 if i // 2 == 0 else 0)
                           + (17 if i % 2 == 0 else 0))


def test22_scatter_rev(m):
    for i in range(3):
        idx1 = ek.arange(m.UInt, 5)
        idx2 = ek.arange(m.UInt, 4) + 3

        x = ek.linspace(m.Float, 0, 1, 5)
        y = ek.linspace(m.Float, 1, 2, 4)
        buf = ek.zero(m.Float, 10)

        if i % 2 == 0:
            ek.enable_grad(buf)
        if i // 2 == 0:
            ek.enable_grad(x, y)

        x.label = "x"
        y.label = "y"
        buf.label = "buf"

        buf2 = m.Float(buf)
        ek.scatter(buf2, x, idx1)
        ek.eval(buf2)
        ek.scatter(buf2, y, idx2)

        ref_buf = m.Float(0.0000, 0.2500, 0.5000, 1.0000, 1.3333,
                          1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        print('allclose checks')
        assert ek.allclose(ref_buf, buf2, atol=1e-4)
        assert ek.allclose(ref_buf, buf, atol=1e-4)
        print('allclose done')

        s = ek.dot_async(buf2, buf2)

        print(ek.graphviz_str(s))

        ek.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 0.0000, 0.0000)
        ref_y = m.Float(2.0000, 2.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert ek.allclose(ek.grad(y), ek.detach(ref_y), atol=1e-4)
            assert ek.allclose(ek.grad(x), ek.detach(ref_x), atol=1e-4)
        else:
            assert ek.grad(x) is None
            assert ek.grad(y) is None

        if i % 2 == 0:
            assert ek.allclose(ek.grad(buf), 0, atol=1e-4)
        else:
            assert ek.grad(buf) is None


def test23_exp(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.exp(x * x)
    ek.backward(y)
    exp_x = ek.exp(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, exp_x)
    assert ek.allclose(ek.grad(x), 2 * ek.detach(x) * exp_x)


def test24_log(m):
    x = ek.linspace(m.Float, 0.01, 1, 10)
    ek.enable_grad(x)
    y = ek.log(x * x)
    ek.backward(y)
    log_x = ek.log(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, log_x)
    print(ek.grad(x))
    print(2/ek.detach(x))
    assert ek.allclose(ek.grad(x), 2 / ek.detach(x))


def test25_pow(m):
    x = ek.linspace(m.Float, 1, 10, 10)
    y = ek.full(m.Float, 2.0, 10)
    ek.enable_grad(x, y)
    z = x**y
    ek.backward(z)
    assert ek.allclose(ek.grad(x), ek.detach(x)*2)
    assert ek.allclose(ek.grad(y),
                       m.Float(0., 2.77259, 9.88751, 22.1807, 40.2359,
                               64.5033, 95.3496, 133.084, 177.975, 230.259))


def test26_csc(m):
    x = ek.linspace(m.Float, 1, 2, 10)
    ek.enable_grad(x)
    y = ek.csc(x * x)
    ek.backward(y)
    csc_x = ek.csc(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, csc_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(-1.52612, -0.822733, -0.189079, 0.572183,
                               1.88201, 5.34839, 24.6017, 9951.86, 20.1158,
                               4.56495), rtol=5e-5)


def test27_sec(m):
    x = ek.linspace(m.Float, 1, 2, 10)
    ek.enable_grad(x)
    y = ek.sec(x * x)
    ek.backward(y)
    sec_x = ek.sec(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, sec_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(5.76495, 19.2717, 412.208, 61.794, 10.3374,
                               3.64885, 1.35811,  -0.0672242, -1.88437,
                               -7.08534))


def test28_tan(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    ek.enable_grad(x)
    y = ek.tan(x * x)
    ek.backward(y)
    tan_x = ek.tan(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, tan_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(0., 0.222256, 0.44553, 0.674965, 0.924494,
                               1.22406, 1.63572, 2.29919, 3.58948, 6.85104))


def test28_cot(m):
    x = ek.linspace(m.Float, 1, 2, 10)
    ek.enable_grad(x)
    y = ek.cot(x * x)
    ek.backward(y)
    cot_x = ek.cot(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, cot_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(-2.82457, -2.49367, -2.45898, -2.78425,
                               -3.81687, -7.12557, -26.3248, -9953.63,
                               -22.0932, -6.98385), rtol=5e-5)


def test29_asin(m):
    x = ek.linspace(m.Float, -.8, .8, 10)
    ek.enable_grad(x)
    y = ek.asin(x * x)
    ek.backward(y)
    asin_x = ek.asin(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, asin_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(-2.08232, -1.3497, -0.906755, -0.534687,
                               -0.177783, 0.177783, 0.534687, 0.906755,
                               1.3497, 2.08232))


def test30_acos(m):
    x = ek.linspace(m.Float, -.8, .8, 10)
    ek.enable_grad(x)
    y = ek.acos(x * x)
    ek.backward(y)
    acos_x = ek.acos(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, acos_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(2.08232, 1.3497, 0.906755, 0.534687, 0.177783,
                               -0.177783, -0.534687, -0.906755, -1.3497,
                               -2.08232))


def test31_atan(m):
    x = ek.linspace(m.Float, -.8, .8, 10)
    ek.enable_grad(x)
    y = ek.atan(x * x)
    ek.backward(y)
    atan_x = ek.atan(ek.sqr(ek.detach(x)))
    assert ek.allclose(y, atan_x)
    assert ek.allclose(ek.grad(x),
                       m.Float(-1.13507, -1.08223, -0.855508, -0.53065,
                               -0.177767, 0.177767, 0.53065, 0.855508, 1.08223,
                               1.13507))


def test32_atan2(m):
    x = ek.linspace(m.Float, -.8, .8, 10)
    y = m.Float(ek.arange(m.Int, 10) & 1) * 1 - .5
    ek.enable_grad(x, y)
    z = ek.atan2(y, x)
    ek.backward(z)
    assert ek.allclose(z, m.Float(-2.58299, 2.46468, -2.29744, 2.06075,
                                  -1.74674, 1.39486, -1.08084, 0.844154,
                                  -0.676915, 0.558599))
    assert ek.allclose(ek.grad(x),
                       m.Float(0.561798, -0.784732, 1.11724, -1.55709, 1.93873,
                               -1.93873, 1.55709, -1.11724, 0.784732,
                               -0.561798))
    assert ek.allclose(ek.grad(y),
                       m.Float(-0.898876, -0.976555, -0.993103, -0.83045,
                               -0.344663, 0.344663, 0.83045, 0.993103,
                               0.976555, 0.898876))


def test33_cbrt(m):
    x = ek.linspace(m.Float, -.8, .8, 10)
    ek.enable_grad(x)
    y = ek.cbrt(x)
    ek.backward(y)
    assert ek.allclose(y, m.Float(-0.928318, -0.853719, -0.763143, -0.64366,
                                  -0.446289, 0.446289, 0.64366, 0.763143,
                                  0.853719, 0.928318))
    assert ek.allclose(ek.grad(x),
                       m.Float(0.386799, 0.45735, 0.572357, 0.804574, 1.67358,
                               1.67358, 0.804574, 0.572357, 0.45735, 0.386799))


def test34_sinh(m):
    x = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x)
    y = ek.sinh(x)
    ek.backward(y)
    assert ek.allclose(
        y, m.Float(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                   0.11134, 0.339541, 0.584578, 0.858602, 1.1752))
    assert ek.allclose(
        ek.grad(x),
        m.Float(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))


def test35_cosh(m):
    x = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x)
    y = ek.cosh(x)
    ek.backward(y)
    assert ek.allclose(
        y,
        m.Float(1.54308, 1.31803, 1.15833, 1.05607, 1.00618, 1.00618,
                1.05607, 1.15833, 1.31803, 1.54308))
    assert ek.allclose(
        ek.grad(x),
        m.Float(-1.1752, -0.858602, -0.584578, -0.339541, -0.11134,
                0.11134, 0.339541, 0.584578, 0.858602, 1.1752))


def test36_tanh(m):
    x = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x)
    y = ek.tanh(x)
    ek.backward(y)
    assert ek.allclose(
        y,
        m.Float(-0.761594, -0.651429, -0.504672, -0.321513, -0.110656,
                0.110656, 0.321513, 0.504672, 0.651429, 0.761594))
    assert ek.allclose(
        ek.grad(x),
        m.Float(0.419974, 0.57564, 0.745306, 0.89663, 0.987755, 0.987755,
                0.89663, 0.745306, 0.57564, 0.419974)
    )


def test37_asinh(m):
    x = ek.linspace(m.Float, -.9, .9, 10)
    ek.enable_grad(x)
    y = ek.asinh(x)
    ek.backward(y)
    assert ek.allclose(
        y,
        m.Float(-0.808867, -0.652667, -0.481212, -0.295673, -0.0998341,
                0.0998341, 0.295673, 0.481212, 0.652667, 0.808867))
    assert ek.allclose(
        ek.grad(x),
        m.Float(0.743294, 0.819232, 0.894427, 0.957826, 0.995037,
                0.995037, 0.957826, 0.894427, 0.819232, 0.743294)
    )


def test38_acosh(m):
    x = ek.linspace(m.Float, 1.01, 2, 10)
    ek.enable_grad(x)
    y = ek.acosh(x)
    ek.backward(y)
    assert ek.allclose(
        y,
        m.Float(0.141304, 0.485127, 0.665864, 0.802882, 0.916291,
                1.01426, 1.10111, 1.17944, 1.25098, 1.31696))
    assert ek.allclose(
        ek.grad(x),
        m.Float(7.05346, 1.98263, 1.39632, 1.12112, 0.952381,
                0.835191, 0.747665, 0.679095, 0.623528, 0.57735)
    )


def test39_atanh(m):
    x = ek.linspace(m.Float, -.99, .99, 10)
    ek.enable_grad(x)
    y = ek.atanh(x)
    ek.backward(y)
    assert ek.allclose(
        y,
        m.Float(-2.64665, -1.02033, -0.618381, -0.342828, -0.110447, 0.110447,
                0.342828, 0.618381, 1.02033, 2.64665))
    assert ek.allclose(
        ek.grad(x),
        m.Float(50.2513, 2.4564, 1.43369, 1.12221, 1.01225, 1.01225, 1.12221,
                1.43369, 2.4564, 50.2513)
    )
