import enoki as ek
import pytest
import gc


@pytest.fixture(scope="module", params=[ek.cuda.ad, ek.llvm.ad])
def m(request):
    gc.collect()
    gc.collect()
    if 'cuda' in request.param.__name__:
        if not ek.has_backend(ek.JitBackend.CUDA):
            pytest.skip('CUDA mode is unsupported')
    else:
        if not ek.has_backend(ek.JitBackend.LLVM):
            pytest.skip('LLVM mode is unsupported')
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
    if True:
        a, b = m.Float(1), m.Float(2)
        ek.enable_grad(a, b)
        c = 2 * a + b
        ek.forward(a, retain_graph=True)
        assert ek.grad(c) == 2
        ek.set_grad(c, 101)
        ek.forward(b)
        assert ek.grad(c) == 102

    if True:
        a, b = m.Float(1), m.Float(2)
        ek.enable_grad(a, b)
        c = 2 * a + b
        ek.set_grad(a, 1.0)
        ek.enqueue(a)
        ek.traverse(m.Float, retain_graph=True, reverse=False)
        assert ek.grad(c) == 2
        assert ek.grad(a) == 0
        ek.set_grad(a, 1.0)
        ek.enqueue(a)
        ek.traverse(m.Float, retain_graph=False, reverse=False)
        assert ek.grad(c) == 4


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


def test20_scatter_reduce_rev(m):
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
        ek.scatter_reduce(ek.ReduceOp.Add, buf2, x, idx1)
        ek.scatter_reduce(ek.ReduceOp.Add, buf2, y, idx2)

        ref_buf = m.Float(0.0000, 0.2500, 0.5000, 1.7500, 2.3333,
                          1.6667, 2.0000, 0.0000, 0.0000, 0.0000)

        assert ek.allclose(ref_buf, buf2, atol=1e-4)

        s = ek.dot_async(buf2, buf2)

        ek.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 3.5000, 4.6667)
        ref_y = m.Float(3.5000, 4.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert ek.allclose(ek.grad(y), ek.detach(ref_y), atol=1e-4)
            assert ek.allclose(ek.grad(x), ek.detach(ref_x), atol=1e-4)
        else:
            assert ek.grad(x) == 0
            assert ek.grad(y) == 0

        if i % 2 == 0:
            assert ek.allclose(ek.grad(buf), ek.detach(ref_buf) * 2, atol=1e-4)
        else:
            assert ek.grad(buf) == 0


def test21_scatter_reduce_fwd(m):
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
        ek.scatter_reduce(ek.ReduceOp.Add, buf2, x, idx1)
        ek.scatter_reduce(ek.ReduceOp.Add, buf2, y, idx2)

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

        assert ek.allclose(ref_buf, buf2, atol=1e-4)

        s = ek.dot_async(buf2, buf2)

        ek.backward(s)

        ref_x = m.Float(0.0000, 0.5000, 1.0000, 0.0000, 0.0000)
        ref_y = m.Float(2.0000, 2.6667, 3.3333, 4.0000)

        if i // 2 == 0:
            assert ek.allclose(ek.grad(y), ek.detach(ref_y), atol=1e-4)
            assert ek.allclose(ek.grad(x), ek.detach(ref_x), atol=1e-4)
        else:
            assert ek.grad(x) == 0
            assert ek.grad(y) == 0

        if i % 2 == 0:
            assert ek.allclose(ek.grad(buf), 0, atol=1e-4)
        else:
            assert ek.grad(buf) == 0


def test22_scatter_fwd(m):
    x = m.Float(4.0)
    ek.enable_grad(x)

    values = x * x * ek.linspace(m.Float, 1, 4, 4)
    idx = 2 * ek.arange(m.UInt32, 4)

    buf = ek.zero(m.Float, 10)
    ek.scatter(buf, values, idx)

    assert ek.grad_enabled(buf)

    ref = [16.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert ek.allclose(buf, ref)

    ek.forward(x, retain_graph=True)
    grad = ek.grad(buf)

    ref_grad = [8.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert ek.allclose(grad, ref_grad)

    # Overwrite first value with non-diff value, resulting gradient entry should be 0
    y = m.Float(3)
    idx = m.UInt32(0)
    ek.scatter(buf, y, idx)

    ref = [3.0, 0.0, 32.0, 0.0, 48.0, 0.0, 64.0, 0.0, 0.0, 0.0]
    assert ek.allclose(buf, ref)

    ek.forward(x)
    grad = ek.grad(buf)

    ref_grad = [0.0, 0.0, 16.0, 0.0, 24.0, 0.0, 32.0, 0.0, 0.0, 0.0]
    assert ek.allclose(grad, ref_grad)


def test22_scatter_fwd_permute(m):
    x = m.Float(4.0)
    ek.enable_grad(x)

    values_0 = x * ek.linspace(m.Float, 1, 9, 5)
    values_1 = x * ek.linspace(m.Float, 11, 19, 5)

    buf = ek.zero(m.Float, 10)

    idx_0 = ek.arange(m.UInt32, 5)
    idx_1 = ek.arange(m.UInt32, 5) + 5

    ek.scatter(buf, values_0, idx_0, permute=False)
    ek.scatter(buf, values_1, idx_1, permute=False)

    ref = [4.0, 12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0, 68.0, 76.0]
    assert ek.allclose(buf, ref)

    ek.forward(x)
    grad = ek.grad(buf)

    ref_grad = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0]
    assert ek.allclose(grad, ref_grad)


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


def test40_safe_functions(m):
    x = ek.linspace(m.Float, 0, 1, 10)
    y = ek.linspace(m.Float, -1, 1, 10)
    z = ek.linspace(m.Float, -1, 1, 10)
    ek.enable_grad(x, y, z)
    x2 = ek.safe_sqrt(x)
    y2 = ek.safe_acos(y)
    z2 = ek.safe_asin(z)
    ek.backward(x2)
    ek.backward(y2)
    ek.backward(z2)
    assert ek.grad(x)[0] == 0
    assert ek.allclose(ek.grad(x)[1], .5 / ek.sqrt(1 / 9))
    assert x[0] == 0
    assert ek.all(ek.isfinite(ek.grad(x)))
    assert ek.all(ek.isfinite(ek.grad(y)))
    assert ek.all(ek.isfinite(ek.grad(z)))


def test41_replace_grad(m):
    x = m.Array3f(1, 2, 3)
    y = m.Array3f(3, 2, 1)
    ek.enable_grad(x, y)
    x2 = x*x
    y2 = y*y
    z = ek.replace_grad(x2, y2)
    z2 = z*z
    ek.backward(z2)
    assert ek.allclose(z2, [1, 16, 81])
    assert ek.grad(x) == 0
    assert ek.allclose(ek.grad(y), [12, 32, 36])


def test42_suspend_resume(m):
    x = m.Array3f(1, 2, 3)
    y = m.Array3f(3, 2, 1)
    ek.enable_grad(x, y)
    assert ek.grad_enabled(x) and ek.grad_enabled(y)
    assert not ek.grad_suspended(x) and not ek.grad_suspended(y)
    ek.suspend_grad(x, y)
    assert not ek.grad_enabled(x) and not ek.grad_enabled(y)
    assert ek.grad_suspended(x) and ek.grad_suspended(y)
    b = x*y
    ek.resume_grad(x, y)
    assert ek.grad_enabled(x) and ek.grad_enabled(y)
    assert not ek.grad_suspended(x) and not ek.grad_suspended(y)
    c = x*y
    ek.backward(c)
    assert ek.grad(x) == ek.detach(y)
    assert ek.grad(y) == ek.detach(x)
    ek.suspend_grad(x, y) # validate reference counting of suspended variables


class Normalize(ek.CustomOp):
    def eval(self, value):
        self.value = value
        self.inv_norm = ek.rcp(ek.norm(value))
        return value * self.inv_norm

    def forward(self):
        grad_in = self.grad_in('value')
        grad_out = grad_in * self.inv_norm
        grad_out -= self.value * (ek.dot(self.value, grad_out) * ek.sqr(self.inv_norm))
        self.set_grad_out(grad_out)

    def backward(self):
        grad_out = self.grad_out()
        grad_in = grad_out * self.inv_norm
        grad_in -= self.value * (ek.dot(self.value, grad_in) * ek.sqr(self.inv_norm))
        self.set_grad_in('value', grad_in)

    def name(self):
        return "normalize"


def test43_custom_reverse(m):
    d = m.Array3f(1, 2, 3)
    ek.enable_grad(d)
    d2 = ek.custom(Normalize, d)
    ek.set_grad(d2, m.Array3f(5, 6, 7))
    ek.enqueue(d2)
    ek.traverse(m.Float, reverse=True)
    assert ek.allclose(ek.grad(d), m.Array3f(0.610883, 0.152721, -0.305441))


def test44_custom_forward(m):
    d = m.Array3f(1, 2, 3)
    ek.enable_grad(d)
    d2 = ek.custom(Normalize, d)
    ek.set_grad(d, m.Array3f(5, 6, 7))
    ek.enqueue(d)
    ek.traverse(m.Float, reverse=False, retain_graph=True)
    assert ek.grad(d) == 0
    ek.set_grad(d, m.Array3f(5, 6, 7))
    assert ek.allclose(ek.grad(d2), m.Array3f(0.610883, 0.152721, -0.305441))
    ek.enqueue(d)
    ek.traverse(m.Float, reverse=False, retain_graph=False)
    assert ek.allclose(ek.grad(d2), m.Array3f(0.610883, 0.152721, -0.305441)*2)

def test45_diff_loop(m):
    def mcint(a, b, f, sample_count=100000):
        rng = m.PCG32()
        i = m.UInt32(0)
        result = m.Float(0)
        l = m.Loop("test45", i, rng, result)
        while l(i < sample_count):
            result += f(ek.lerp(a, b, rng.next_float32()))
            i += 1
        return result * (b - a) / sample_count

    class EllipticK(ek.CustomOp):
        # --- Internally used utility methods ---

        # Integrand of the 'K' function
        def K(self, x, m_):
            return ek.rsqrt(1 - m_ * ek.sqr(ek.sin(x)))

        # Derivative of the above with respect to 'm'
        def dK(self, x, m_):
            m_ = m.Float(m_) # Convert 'm' to differentiable type
            ek.enable_grad(m_)
            y = self.K(x, m_)
            ek.forward(m_)
            return ek.grad(y)

        # Monte Carlo integral of dK, used in forward/reverse pass
        def eval_grad(self):
            return mcint(a=0, b=ek.Pi/2, f=lambda x: self.dK(x, self.m_))

        # --- CustomOp interface ---

        def eval(self, m_):
            self.m_ = m_ # Stash 'm' for later
            return mcint(a=0, b=ek.Pi/2, f=lambda x: self.K(x, self.m_))

        def forward(self):
            self.set_grad_out(self.grad_in('m_') * self.eval_grad())

    def elliptic_k(m_):
        return ek.custom(EllipticK, m_)

    ek.set_flag(ek.JitFlag.LoopRecord, True)
    x = m.Float(0.5)
    ek.enable_grad(x)
    y = elliptic_k(x)
    ek.forward(x)
    assert ek.allclose(y, 1.85407, rtol=5e-4)
    assert ek.allclose(ek.grad(y), 0.847213, rtol=5e-4)
    ek.set_flag(ek.JitFlag.LoopRecord, False)


def test46_loop_ballistic(m):
    class Ballistic(ek.CustomOp):
        def timestep(self, pos, vel, dt=0.02, mu=.1, g=9.81):
            acc = -mu*vel*ek.norm(vel) - m.Array2f(0, g)
            pos_out = pos + dt * vel
            vel_out = vel + dt * acc
            return pos_out, vel_out

        def eval(self, pos, vel):
            pos, vel = m.Array2f(pos), m.Array2f(vel)

            # Run for 100 iterations
            it, max_it = m.UInt32(0), 100

            # Allocate scratch space
            n = max(ek.width(pos), ek.width(vel))
            self.temp_pos = ek.empty(m.Array2f, n * max_it)
            self.temp_vel = ek.empty(m.Array2f, n * max_it)

            loop = m.Loop("eval", pos, vel, it)
            while loop(it < max_it):
                # Store current loop variables
                index = it * n + ek.arange(m.UInt32, n)
                ek.scatter(self.temp_pos, pos, index)
                ek.scatter(self.temp_vel, vel, index)

                # Update loop variables
                pos_out, vel_out = self.timestep(pos, vel)
                pos.assign(pos_out)
                vel.assign(vel_out)

                it += 1

            # Ensure output and temp. arrays are evaluated at this point
            ek.eval(pos, vel)

            return pos, vel

        def backward(self):
            grad_pos, grad_vel = self.grad_out()

            # Run for 100 iterations
            it = m.UInt32(100)

            loop = m.Loop("backward", it, grad_pos, grad_vel)
            n = ek.width(grad_pos)
            while loop(it > 0):
                # Retrieve loop variables, reverse chronological order
                it -= 1
                index = it * n + ek.arange(m.UInt32, n)
                pos = ek.gather(m.Array2f, self.temp_pos, index)
                vel = ek.gather(m.Array2f, self.temp_vel, index)

                # Differentiate loop body in reverse mode
                ek.enable_grad(pos, vel)
                pos_out, vel_out = self.timestep(pos, vel)
                ek.set_grad(pos_out, grad_pos)
                ek.set_grad(vel_out, grad_vel)
                ek.enqueue(pos_out, vel_out)
                ek.traverse(m.Float, reverse=True)

                # Update loop variables
                grad_pos.assign(ek.grad(pos))
                grad_vel.assign(ek.grad(vel))

            self.set_grad_in('pos', grad_pos)
            self.set_grad_in('vel', grad_vel)

    pos_in = m.Array2f([1, 2, 4], [1, 2, 1])
    vel_in = m.Array2f([10, 9, 4], [5, 3, 6])

    ek.set_flag(ek.JitFlag.LoopRecord, True)
    for i in range(20):
        ek.enable_grad(vel_in)
        ek.eval(vel_in, pos_in)
        pos_out, vel_out = ek.custom(Ballistic, pos_in, vel_in)
        loss = ek.squared_norm(pos_out - m.Array2f(5, 0))
        ek.backward(loss)

        vel_in = m.Array2f(ek.detach(vel_in) - 0.2 * ek.grad(vel_in))

    print(loss)
    assert ek.allclose(loss, 0, atol=1e-4)
    assert ek.allclose(vel_in.x, [3.3516, 2.3789, 0.79156], rtol=1e-3)
    ek.set_flag(ek.JitFlag.LoopRecord, False)


@pytest.mark.skip("TODO bring it back when loop is implemented")
def test46_loop_ballistic_2(m):
    class Ballistic2(ek.CustomOp):
        def timestep(self, pos, vel, dt=0.02, mu=.1, g=9.81):
            acc = -mu*vel*ek.norm(vel) - m.Array2f(0, g)
            pos_out = pos + dt * vel
            vel_out = vel + dt * acc
            return pos_out, vel_out

        def eval(self, pos, vel):
            pos, vel = m.Array2f(pos), m.Array2f(vel)

            # Run for 100 iterations
            it, max_it = m.UInt32(0), 100

            loop = m.Loop(pos, vel, it)
            while loop(it < max_it):
                # Update loop variables
                pos_out, vel_out = self.timestep(pos, vel)
                pos.assign(pos_out)
                vel.assign(vel_out)

                it += 1

            self.pos = pos
            self.vel = vel

            return pos, vel

        def backward(self):
            grad_pos, grad_vel = self.grad_out()
            pos, vel = self.pos, self.vel

            # Run for 100 iterations
            it = m.UInt32(0)

            loop = m.Loop(it, pos, vel, grad_pos, grad_vel)
            while loop(it < 100):
                # Take reverse step in time
                pos_rev, vel_rev = self.timestep(pos, vel, dt=-0.02)
                pos.assign(pos_rev)
                vel.assign(vel_rev)

                # Take a forward step in time, keep track of derivatives
                ek.enable_grad(pos_rev, vel_rev)
                pos_fwd, vel_fwd = self.timestep(pos_rev, vel_rev, dt=0.02)
                ek.set_grad(pos_fwd, grad_pos)
                ek.set_grad(vel_fwd, grad_vel)
                ek.enqueue(pos_fwd, vel_fwd)
                ek.traverse(m.Float, reverse=True)

                grad_pos.assign(ek.grad(pos_rev))
                grad_vel.assign(ek.grad(vel_rev))
                it += 1

            self.set_grad_in('pos', grad_pos)
            self.set_grad_in('vel', grad_vel)

    ek.set_flag(ek.JitFlag.LoopRecord, True)
    pos_in = m.Array2f([1, 2, 4], [1, 2, 1])
    vel_in = m.Array2f([10, 9, 4], [5, 3, 6])

    for i in range(20):
        ek.enable_grad(vel_in)
        ek.eval(vel_in, pos_in)
        pos_out, vel_out = ek.custom(Ballistic2, pos_in, vel_in)
        loss = ek.squared_norm(pos_out - m.Array2f(5, 0))
        ek.backward(loss)

        vel_in = m.Array2f(ek.detach(vel_in) - 0.2 * ek.grad(vel_in))

    assert ek.allclose(loss, 0, atol=1e-4)
    assert ek.allclose(vel_in.x, [3.3516, 2.3789, 0.79156], atol=1e-3)
    ek.set_flag(ek.JitFlag.LoopRecord, False)


def test47_nan_propagation(m):
    for i in range(2):
        x = ek.arange(m.Float, 10)
        ek.enable_grad(x)
        f0 = m.Float(0)
        y = ek.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        ek.backward(y)
        g = ek.grad(x)
        if i == 0:
            assert g == 1
        else:
            assert ek.all(ek.isnan(g))

    for i in range(2):
        x = ek.arange(m.Float, 10)
        ek.enable_grad(x)
        f0 = m.Float(0)
        y = ek.select(x < (20 if i == 0 else 0), x, x * (f0 / f0))
        ek.forward(x)
        g = ek.grad(y)
        if i == 0:
            assert g == 1
        else:
            assert ek.all(ek.isnan(g))
