import drjit as dr
import pytest
import sys

# Work around a refleak in @pytest.mark.parameterize
wrap_modes = [dr.WrapMode.Repeat, dr.WrapMode.Clamp, dr.WrapMode.Mirror]

def _skip_metal_f64(t, texture_type):
    if dr.backend_v(t) == dr.JitBackend.Metal and 'f64' in texture_type:
        pytest.skip("Metal does not support float64")

@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("force_optix", [True, False])
@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test01_interp_1d(t, wrap_mode, force_optix, texture_type):
    _skip_metal_f64(t, texture_type)
    with dr.scoped_set_flag(dr.JitFlag.ForceOptiX, force_optix):
        mod = sys.modules[t.__module__]
        TexType = getattr(mod, texture_type)

        tex = TexType([2], 1, True, dr.FilterMode.Linear, wrap_mode)
        tex.set_value(t(0, 1))

        tex_no_accel = TexType([2], 1, False, dr.FilterMode.Linear, wrap_mode)
        tex_no_accel.set_value(t(0, 1))

        N = 9
        ref = dr.linspace(t, 0, 1, N)
        pos = dr.linspace(t, 0.25, 0.75, N)

        output = tex_no_accel.eval(pos)
        assert dr.allclose(output, ref)

        output = tex.eval(pos)
        assert dr.allclose(output, ref)

        if wrap_mode == dr.WrapMode.Repeat:
            pos = dr.linspace(t, -0.75, -0.25, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

            pos = dr.linspace(t, 1.25, 1.75, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

        elif wrap_mode == dr.WrapMode.Clamp:
            ref = dr.zeros(t, N)
            pos = dr.linspace(t, -0.25, 0.25, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

            ref = dr.ones(t, N)
            pos = dr.linspace(t, 0.75, 1.25, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

        elif wrap_mode == dr.WrapMode.Mirror:
            pos = dr.linspace(t, -0.25, -0.75, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

            pos = dr.linspace(t, 1.75, 1.25, N)
            output = tex_no_accel.eval(pos)
            assert dr.allclose(output, ref)
            output = tex.eval(pos)
            assert dr.allclose(output, ref)

            # Also check that masks are correctly handled
            active = dr.opaque(dr.mask_t(t), False)
            pos = dr.linspace(t, 1.75, 1.25, N)
            output = tex_no_accel.eval(pos, active=active)
            assert dr.allclose(output, 0)
            output = tex.eval(pos, active=active)
            assert dr.allclose(output, 0)


@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test02_interp_1d(t, wrap_mode, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N = 123

    for ch in range(1, 9):
        rng_1 = PCG32(N * ch)
        rng_2 = PCG32(1024)

        tex = TexType([N], ch, True, dr.FilterMode.Linear, wrap_mode)
        tex_no_accel = TexType([N], ch, False, dr.FilterMode.Linear, wrap_mode)

        StorageType = dr.array_t(tex.value())

        values = StorageType(rng_1.next_float32())
        tex.set_value(values)
        tex_no_accel.set_value(values)

        pos = Array1f(rng_2.next_float32())
        result_no_accel = tex_no_accel.eval(pos)
        result_accel = tex.eval(pos)
        dr.eval(result_no_accel, result_accel)

        # Verify the return type of eval
        expected_type = getattr(mod, f'Array{ch}f') if ch <= 4 \
                        else getattr(mod, 'ArrayXf')
        assert type(result_no_accel) is expected_type
        assert type(result_accel) is expected_type

        assert dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3)


@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test03_interp_2d(t, wrap_mode, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    Array2f = getattr(mod, 'Array2f')
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M = 123, 456

    for ch in range(1, 9):
        rng_1 = PCG32(N * M * ch)
        rng_2 = PCG32(1024)

        tex = TexType([N, M], ch, True, dr.FilterMode.Linear, wrap_mode)
        tex_no_accel = TexType([N, M], ch, False, dr.FilterMode.Linear, wrap_mode)

        values = rng_1.next_float32()
        tex.set_value(values)
        tex_no_accel.set_value(values)

        pos = Array2f(rng_2.next_float32(), rng_2.next_float32())
        result_no_accel = tex_no_accel.eval(pos)
        result_accel = tex.eval(pos)
        dr.eval(result_no_accel, result_accel)

        assert(dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3))


@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture3f64', 'Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test04_interp_3d(t, wrap_mode, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    Array3f = getattr(mod, 'Array3f')
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M, L = 123, 456, 12

    for ch in range(1, 9):
        rng_1 = PCG32(N * M * L * ch);
        rng_2 = PCG32(1024);

        tex = TexType([N, M, L], ch, True, dr.FilterMode.Linear, wrap_mode)
        tex_no_accel = TexType([N, M, L], ch, False, dr.FilterMode.Linear, wrap_mode)

        values = rng_1.next_float32()
        tex.set_value(values)
        tex_no_accel.set_value(values)

        pos = Array3f(rng_2.next_float32(), rng_2.next_float32(), rng_2.next_float32())
        result_no_accel = tex_no_accel.eval(pos)
        result_accel = tex.eval(pos)
        dr.eval(result_no_accel, result_accel)

        assert(dr.allclose(result_no_accel, result_accel, 6e-3, 6e-3))


@pytest.mark.parametrize("migrate", [True, False])
@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test05_grad(t, migrate, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    Float = getattr(mod, 'Float')
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)

    N = 3

    tex = TexType([N], 1, True, dr.FilterMode.Linear, dr.WrapMode.Repeat)
    value = t(3, 5, 8)
    dr.enable_grad(value)
    tex.set_value(value, migrate)

    pos = Array1f(1 / 6.0 * 0.25 + (1 / 6.0 + 1 / 3.0) * 0.75)
    expected = t(0.25 * 3 + 0.75 * 5)

    out2 = tex.eval(pos)
    assert dr.allclose(out2, expected, 5e-3, 5e-3)

    out = Array1f(tex.eval(pos))

    dr.backward(out)

    assert dr.allclose(dr.grad(value), Float(.25, .75, 0))
    assert dr.allclose(out, expected, 5e-3, 5e-3)
    assert dr.allclose(tex.value(), value)


@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test_06_nearest(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)

    N = 3
    value = t(0, 0.5, 1)

    tex = TexType([N], 1, True, dr.FilterMode.Nearest, dr.WrapMode.Repeat)
    tex.set_value(value)

    tex_no_accel = TexType([N], 1, False, dr.FilterMode.Nearest, dr.WrapMode.Repeat)
    tex_no_accel.set_value(value)

    pos = dr.linspace(t, 0, 1, 80)
    out_accel = tex.eval(pos)
    out_drjit = tex_no_accel.eval(pos)
    assert dr.allclose(out_accel, out_drjit)


@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f64'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
def test07_cubic_analytic(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)

    N = 4

    tex = TexType([N], 1, True, dr.FilterMode.Linear, dr.WrapMode.Clamp)
    value = t(0, 1, 0, 0)
    tex.set_value(value)

    pos = Array1f(0.5)
    (_, grad_64) = tex.eval_cubic_grad(pos)
    dr.enable_grad(pos)

    res = Array1f(tex.eval_cubic(pos, True, True))

    dr.backward(res)
    grad_ad = dr.grad(pos)
    res2 = tex.eval_cubic_helper(pos)

    # 1/6 * (3*a^3 - 6*a^2 + 4) with a=0.5
    StorageType = dr.array_t(tex.value())
    ref_res = StorageType(0.479167)
    assert dr.allclose(res, ref_res, 1e-5, 1e-5)
    assert dr.allclose(res2, ref_res, 1e-5, 1e-5)
    # 1/6 * (9*a^2 - 12*a) with a=0.5
    ref_grad = StorageType(-0.625 * 4.0)
    assert dr.allclose(grad_64[0][0], ref_grad, 1e-5, 1e-5)
    assert dr.allclose(grad_ad[0], ref_grad, 1e-5, 1e-5)


@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f64'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test08_cubic_interp_1d(t, texture_type, wrap_mode):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)

    tex = TexType([5], 1, True, dr.FilterMode.Linear, wrap_mode)
    tex.set_value(t(2, 1, 3, 4, 7))

    N = 20

    pos = dr.linspace(t, 0.1, 0.9, N)
    out = tex.eval_cubic_helper(pos)
    ref = out

    out = tex.eval_cubic(pos, True, True)
    assert dr.allclose(out, ref)

    if wrap_mode == dr.WrapMode.Repeat:
        pos = dr.linspace(t, -0.9, -0.1, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)

        pos = dr.linspace(t, 1.1, 1.9, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)

    elif wrap_mode == dr.WrapMode.Clamp:
        pos_2 = dr.linspace(t, 0, 1, N)
        res = tex.eval_cubic(pos_2, True, True)
        res2 = tex.eval_cubic_helper(pos_2)

        ref_2 = t(
            1.9792, 1.9259, 1.8198, 1.6629, 1.5168,
            1.4546, 1.5485, 1.8199, 2.2043, 2.6288,
            3.0232, 3.3783, 3.7461, 4.1814, 4.7305,
            5.3536, 5.9603, 6.4595, 6.7778, 6.9375)
        assert dr.allclose(res, ref_2, 5e-3, 5e-3)
        assert dr.allclose(res2, ref_2, 5e-3, 5e-3)

        ref = dr.full(t, 2, N)
        pos = dr.linspace(t, -1, -0.1, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)

        ref = dr.full(t, 7, N)
        pos = dr.linspace(t, 1.1, 2, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)

    elif wrap_mode == dr.WrapMode.Mirror:
        pos = dr.linspace(t, -0.1, -0.9, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)

        pos = dr.linspace(t,1.9, 1.1, N)
        res = tex.eval_cubic(pos, True, True)
        res2 = tex.eval_cubic_helper(pos)
        assert dr.allclose(res, ref)
        assert dr.allclose(res2, ref)


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f64'])
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test09_cubic_interp_2d(t, texture_type, wrap_mode):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M = 5,4

    tex = TexType([N,M], 1, True, dr.FilterMode.Linear, wrap_mode)
    rng1 = PCG32(N*M)
    tex.set_value(rng1.next_float32())

    rng2 = PCG32(1024)
    pos = (rng2.next_float32(), rng2.next_float32())
    res = tex.eval_cubic(pos, True, True)
    res2 = tex.eval_cubic_helper(pos)
    assert dr.allclose(res, res2)


@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f64'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test10_cubic_interp_3d(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    Array3f = getattr(mod, 'Array3f')
    UInt32 = dr.uint32_array_t(t)

    dummy_tex = TexType([1,1,1], 1)

    TensorType = type(dummy_tex.tensor())
    StorageType = dr.array_t(dummy_tex.value())

    s = 9
    tensor = dr.full(TensorType, 1, shape=[s, s, s, 2])
    dr.scatter(tensor.array, StorageType(0.0),  UInt32(728)) # tensor[4, 4, 4, 0] = 0.0
    dr.scatter(tensor.array, StorageType(2.0),  UInt32(546)) # tensor[3, 3, 3, 0] = 2.0
    dr.scatter(tensor.array, StorageType(10.0), UInt32(727)) # tensor[4, 4, 3, 1] = 10.0

    tex = TexType(tensor, True, False, dr.FilterMode.Linear, dr.WrapMode.Clamp)

    ref = Array2f(0.71312, 1.86141)
    pos = Array3f(.49, .5, .5)
    res = tex.eval_cubic(pos, True, True)
    res2 = tex.eval_cubic_helper(pos)
    assert dr.allclose(res, ref, 2e-3, 2e-3)
    assert dr.allclose(res2, ref, 2e-3, 2e-3)

    ref2 = Array2f(0.800905, 2.60136)
    pos2 = Array3f(.45, .53, .51)
    res = tex.eval_cubic(pos2, True, True)
    res2 = tex.eval_cubic_helper(pos2)
    assert dr.allclose(res, ref2, 2e-3, 2e-2)
    assert dr.allclose(res2, ref2, 2e-3, 2e-2)


@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f64'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.mark.skipif(sys.platform == "win32", reason="FIXME: Non-deterministic crashes on Windows")
def test11_cubic_grad_pos(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array3f = getattr(mod, 'Array3f')
    Array1f = getattr(mod, 'Array1f')
    UInt32 = dr.uint32_array_t(t)

    dummy_tex = TexType([1,1,1], 1)

    TensorType = type(dummy_tex.tensor())
    StorageType = dr.array_t(dummy_tex.value())
    tensor = dr.full(TensorType, 1, shape=[4, 4, 4, 1])
    dr.scatter(tensor.array, StorageType(0.5), UInt32(21))  # data[1, 1, 1] = 0.5
    dr.scatter(tensor.array, StorageType(2.0), UInt32(25))  # data[1, 2, 1] = 2.0
    dr.scatter(tensor.array, StorageType(3.0), UInt32(41))  # data[2, 2, 1] = 3.0
    dr.scatter(tensor.array, StorageType(4.0), UInt32(22))  # data[1, 1, 2] = 4.0

    tex = TexType(tensor, True, False, dr.FilterMode.Linear, dr.WrapMode.Clamp)

    pos = Array3f(.5, .5, .5)
    val_64, grad_64 = tex.eval_cubic_grad(pos)
    dr.enable_grad(pos)

    res = Array1f(tex.eval_cubic(pos, True, True))
    dr.backward(res)

    assert dr.allclose(res, val_64)
    grad_ad = dr.grad(pos)
    res2 = tex.eval_cubic_helper(pos)

    ref_res = Array1f(1.60509)
    assert dr.allclose(res, ref_res)
    assert dr.allclose(res2, ref_res)
    ref_grad = Array3f(0.07175, 0.07175, -0.21525)
    ref_grad *= 4.0
    assert dr.allclose(grad_64[0][0], ref_grad[0])
    assert dr.allclose(grad_64[0][1], ref_grad[1])
    assert dr.allclose(grad_64[0][2], ref_grad[2])
    assert dr.allclose(grad_ad, ref_grad)


@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f64'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
def test12_cubic_hessian_pos(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array3f = getattr(mod, 'Array3f')
    UInt32 = dr.uint32_array_t(t)

    dummy_tex = TexType([1,1,1], 1)

    TensorType = type(dummy_tex.tensor())
    StorageType = dr.array_t(dummy_tex.value())

    tensor = dr.full(TensorType, 0, shape=[4, 4, 4, 1])
    dr.scatter(tensor.array, StorageType(1.0), UInt32(21))  # data[1, 1, 1] = 1.0
    dr.scatter(tensor.array, StorageType(2.0), UInt32(37))  # data[2, 1, 1] = 2.0
    # NOTE: Tensor has different index convention with Texture
    #       [2, 1, 1] is equivalent to (x=1, y=1, z=2) in the texture

    tex = TexType(tensor, True, False, dr.FilterMode.Linear, dr.WrapMode.Clamp)

    pos = Array3f(.5, .5, .5)
    val_64, grad_64 = tex.eval_cubic_grad(pos, True)
    value_h, grad_h, hessian = tex.eval_cubic_hessian(pos, True)

    assert dr.allclose(val_64[0], value_h[0])

    assert dr.allclose(grad_64[0][0], grad_h[0][0])
    assert dr.allclose(grad_64[0][1], grad_h[0][1])
    assert dr.allclose(grad_64[0][2], grad_h[0][2])
    # compare with analytical solution
    # note: hessian[ch][grad1][grad2]
    # note: multiply analytical result by 16.0f=4.f*4.f to account for the resolution transformation
    assert dr.allclose(hessian[0][0][0], StorageType(-0.344401 * 16.0), 1e-5, 1e-5)
    assert dr.allclose(hessian[0][0][1], StorageType(0.561523 * 16.0), 1e-5, 1e-5)
    assert dr.allclose(hessian[0][0][2], StorageType(-0.187174 * 16.0), 1e-5, 1e-5)
    assert dr.allclose(hessian[0][1][1], StorageType(-0.344401 * 16.0), 1e-5, 1e-5)
    assert dr.allclose(hessian[0][1][2], StorageType(-0.187174 * 16.0), 1e-5, 1e-5)
    assert dr.allclose(hessian[0][2][2], StorageType(-0.344401 * 16.0), 1e-5, 1e-5)
    assert hessian[0][0][1] == hessian[0][1][0]
    assert hessian[0][0][2] == hessian[0][2][0]
    assert hessian[0][1][2] == hessian[0][2][1]


@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test15_tensor_value_1d(t, texture_type, migrate):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N = 2
    for ch in range(1, 9):
        rng = PCG32(2 * ch)
        tex = TexType([N], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data, migrate=migrate)

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test16_tensor_value_2d(t, texture_type, migrate):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M = 2, 3
    for ch in range(1, 9):
        rng = PCG32(N * M * ch)
        tex = TexType([N, M], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data, migrate=migrate)

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)


@pytest.mark.parametrize("texture_type", ['Texture3f64', 'Texture3f', 'Texture3f16'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test17_tensor_value_3d(t, texture_type, migrate):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M, L = 2, 3, 4
    for ch in range(1, 9):
        rng = PCG32(N * M * L * ch)
        tex = TexType([N, M, L], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data, migrate=migrate)

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)


@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test18_fetch_1d(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array1f = getattr(mod, 'Array1f')
    PCG32 = getattr(mod, 'PCG32')

    N = 2
    for ch in range(1,9):
        tex = TexType([N], ch, True)
        tex_no_accel = TexType([N], ch, False)
        rng = PCG32(N * ch)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())

        tex.set_value(tex_data)
        tex_no_accel.set_value(tex_data)

        pos = Array1f(0.5)
        out_no_accel = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)

        dr.eval(tex_data, out_accel, out_no_accel)

        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_no_accel[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_no_accel[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test19_fetch_2d(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    PCG32 = getattr(mod, 'PCG32')

    N, M = 2, 2
    for ch in range(1, 9):
        tex = TexType([N, M], ch, True)
        tex_no_accel = TexType([N, M], ch, False)
        rng = PCG32(N * M * ch)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())

        tex.set_value(tex_data)
        tex_no_accel.set_value(tex_data)

        pos = Array2f(0.5, 0.5)
        out_no_accel = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)

        dr.eval(tex_data, out_accel, out_no_accel)

        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_no_accel[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_no_accel[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])
            assert dr.allclose(tex_data[2 * ch + k], out_no_accel[2][k])
            assert dr.allclose(tex_data[2 * ch + k], out_accel[2][k])
            assert dr.allclose(tex_data[3 * ch + k], out_no_accel[3][k])
            assert dr.allclose(tex_data[3 * ch + k], out_accel[3][k])


@pytest.mark.parametrize("texture_type", ['Texture3f64', 'Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test20_fetch_3d(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array3f = getattr(mod, 'Array3f')
    PCG32 = getattr(mod, 'PCG32')

    N, M, L = 2, 2, 2
    for ch in range(1, 9):
        tex = TexType([N, M, L], ch, True)
        tex_no_accel = TexType([N, M, L], ch, False)
        rng = PCG32(N * M * L * ch)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())

        tex.set_value(tex_data)
        tex_no_accel.set_value(tex_data)

        pos = Array3f(0.3, 0.3, 0.3)
        out_no_accel = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)

        dr.eval(tex_data, out_accel, out_no_accel)

        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_no_accel[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_no_accel[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])
            assert dr.allclose(tex_data[2 * ch + k], out_no_accel[2][k])
            assert dr.allclose(tex_data[2 * ch + k], out_accel[2][k])
            assert dr.allclose(tex_data[3 * ch + k], out_no_accel[3][k])
            assert dr.allclose(tex_data[3 * ch + k], out_accel[3][k])
            assert dr.allclose(tex_data[4 * ch + k], out_no_accel[4][k])
            assert dr.allclose(tex_data[4 * ch + k], out_accel[4][k])
            assert dr.allclose(tex_data[5 * ch + k], out_no_accel[5][k])
            assert dr.allclose(tex_data[5 * ch + k], out_accel[5][k])
            assert dr.allclose(tex_data[6 * ch + k], out_no_accel[6][k])
            assert dr.allclose(tex_data[6 * ch + k], out_accel[6][k])
            assert dr.allclose(tex_data[7 * ch + k], out_no_accel[7][k])
            assert dr.allclose(tex_data[7 * ch + k], out_accel[7][k])


@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test21_fetch_migrate(t, texture_type, migrate):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array1f = getattr(mod, 'Array1f')
    can_migrate = dr.backend_v(t) in (dr.JitBackend.CUDA, dr.JitBackend.Metal) \
        and texture_type != "Texture1f64"

    N = 2
    tex = TexType([N], 1, True)
    tex_data = t(1.0, 2.0)
    tex.set_value(tex_data, migrate)
    assert tex.migrated() == (migrate and can_migrate)

    pos = Array1f(0.5)
    out = tex.eval_fetch(pos)
    assert tex.migrated() == (migrate and can_migrate)

    assert dr.allclose(out[0][0], 1.0)
    assert dr.allclose(out[1][0], 2.0)


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test22_fetch_grad(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')

    N, M = 2, 2
    tex = TexType([N, M], 1, True)
    tex_no_accel = TexType([N, M], 1, False)

    StorageType = dr.array_t(tex.value())
    tex_data = StorageType(t(1, 2, 3, 4))
    dr.enable_grad(tex_data)
    tex.set_value(tex_data)
    tex_no_accel.set_value(tex_data)

    pos = Array2f(0.5, 0.5)
    out = tex_no_accel.eval_fetch(pos)
    assert dr.allclose(1, out[0][0])
    assert dr.allclose(2, out[1][0])
    assert dr.allclose(3, out[2][0])
    assert dr.allclose(4, out[3][0])

    out = tex.eval_fetch(pos)
    assert dr.allclose(1, out[0][0])
    assert dr.allclose(2, out[1][0])
    assert dr.allclose(3, out[2][0])
    assert dr.allclose(4, out[3][0])

    for i in range(0, 4):
        dr.backward(out[i][0])
        grad = dr.grad(tex_data)
        expected = t(
                1 if i == 0 else 0,
                1 if i == 1 else 0,
                1 if i == 2 else 0,
                1 if i == 3 else 0)

        assert dr.allclose(expected, grad)
        dr.set_grad(tex_data, t(0, 0, 0, 0))


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test23_set_tensor(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')

    tex = TexType([2, 2], 1, True)
    tex_no_accel = TexType([2, 2], 1, False)

    TensorType = type(tex.tensor())
    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))
    assert new_tensor.shape == (2,3,2)

    tex.set_tensor(new_tensor)
    tex_no_accel.set_tensor(new_tensor)
    dr.eval(tex, tex_no_accel)
    assert tex.tensor().shape == (2,3,2)

    pos = Array2f(0, 0)
    result_no_accel = tex_no_accel.eval(pos)
    result_accel = tex.eval(pos)
    dr.eval(result_no_accel, result_accel)
    assert dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_accel, Array2f(6.5, 6))

    pos = Array2f(1, 1)
    result_no_accel = tex_no_accel.eval(pos)
    result_accel = tex.eval(pos)
    dr.eval(result_no_accel, result_accel)
    assert dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_accel, Array2f(1.5, 1))

    pos = Array2f(0, 1)
    result_no_accel = tex_no_accel.eval(pos)
    result_accel = tex.eval(pos)
    dr.eval(result_no_accel, result_accel)
    assert dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_accel, Array2f(3.5, 3))

    pos = Array2f(1, 0)
    result_no_accel = tex_no_accel.eval(pos)
    result_accel = tex.eval(pos)
    dr.eval(result_no_accel, result_accel)
    assert dr.allclose(result_no_accel, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_accel, Array2f(4.5, 4))


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, diff, shape=(*)")
def test24_set_tensor_ad(t, texture_type):
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    UInt32 = dr.uint32_array_t(t)
    dummy = TexType([2, 2], 1, True)
    TensorType = type(dummy.tensor())

    # `set_tensor` (migrate=False) doesn't change index
    tex = TexType([2, 2], 1, True)
    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))
    dr.enable_grad(new_tensor)
    new_tensor_index = new_tensor.array.index
    tex.set_tensor(new_tensor, migrate=False)
    tensor_after = tex.tensor()
    assert tensor_after.array.index == new_tensor_index
    assert tensor_after.array.index_ad > 0

    # `set_tensor` (migrate=True) doesn't change AD index
    tex = TexType([2, 2], 1, True)
    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))
    dr.enable_grad(new_tensor)
    new_tensor_index_ad = new_tensor.array.index_ad
    tex.set_tensor(new_tensor, migrate=True)
    tensor_after = tex.tensor()
    assert tensor_after.array.index_ad == new_tensor_index_ad
    assert dr.allclose(tensor_after, new_tensor)

    # `set_tensor` (migrate=False) inplace doesn't change index
    tex = TexType([2, 3], 2, True)
    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))
    dr.enable_grad(new_tensor)
    current_tensor = tex.tensor()
    dr.scatter(current_tensor.array, new_tensor.array, dr.arange(UInt32, 12))
    new_tensor_index_ad = current_tensor.array.index_ad
    tex.update_inplace(migrate=False) # Signal update
    assert tex.tensor().array.index_ad == new_tensor_index_ad
    assert dr.allclose(tex.tensor(), new_tensor)

    # `set_tensor` (migrate=True) inplace doesn't change index
    tex = TexType([2, 3], 2, True)
    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))
    dr.enable_grad(new_tensor)
    current_tensor = tex.tensor()
    dr.scatter(current_tensor.array, new_tensor.array, dr.arange(UInt32, 12))
    new_tensor_index_ad = current_tensor.array.index_ad
    tex.update_inplace(migrate=True) # Signal update
    assert tex.tensor().array.index_ad == new_tensor_index_ad
    assert dr.allclose(tex.tensor(), new_tensor)


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.mark.parametrize("init", ['constructor', 'set_tensor'])
@pytest.test_arrays("is_jit, float32, diff, shape=(*)")
def test25_eval_ad_migrated(t, texture_type, init):
    _skip_metal_f64(t, texture_type)
    # Test only makes sense for configurations where the texture can be migrated
    can_migrate = dr.backend_v(t) in (dr.JitBackend.CUDA, dr.JitBackend.Metal) \
        and texture_type != "Texture2f64"
    if not can_migrate:
        return

    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    tex = TexType([1,1,1], 1)
    TensorType = type(tex.tensor())

    # Differentiating the texture should not require the texture to be unmigrated
    tex_data = t(1, 2, 3, 4)
    tensor = TensorType(tex_data, shape=(2, 2, 1))
    dr.enable_grad(tensor)
    if init == 'constructor':
        tex = TexType(tensor, use_accel=True, migrate=True)
    elif init == 'set_tensor':
        tex.set_tensor(tensor, migrate=True)
    assert tex.migrated()
    pos = Array2f(0.5, 0)
    result = tex.eval(pos)
    dr.eval(result)
    assert tex.migrated()
    dr.backward(result[0])
    dr.allclose(tensor.grad, [0.5, 0])
    assert tex.migrated()

    # Differentiating the texture lookup position needs the primal texel data,
    # which the readback view provides without undoing the migration
    pos = Array2f(0.5, 0)
    dr.enable_grad(pos)
    result = tex.eval(pos)
    dr.eval(result)
    assert tex.migrated()
    dr.backward(result[0])
    assert dr.allclose(pos.grad, [2, 0])


@pytest.mark.parametrize("texture_type", ['Texture1f64', 'Texture1f', 'Texture1f16'])
@pytest.mark.parametrize("init", ['constructor', 'set_tensor'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, diff, shape=(*)")
def test26_tensor_getter_does_not_drop_gradient_tracking(t, texture_type, init, migrate):
    _skip_metal_f64(t, texture_type)
    # Regression test to insure that `Texture::tensor() doesn't accidentlly drop
    # gradient tracking on its internal members when called in a `suspend_grad`
    # scope.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    tex = TexType([1], 1)
    TensorType = type(tex.tensor())

    # Differentiating the texture should not require the texture to be unmigrated
    tex_data = t(1)
    tensor = TensorType(tex_data, shape=(1, 1))
    dr.enable_grad(tensor)
    if init == 'constructor':
        tex = TexType(tensor, use_accel=True, migrate=migrate)
    elif init == 'set_tensor':
        tex.set_tensor(tensor, migrate=migrate)

    if dr.backend_v(t) in (dr.JitBackend.CUDA, dr.JitBackend.Metal) \
            and texture_type != "Texture1f64":
        assert tex.migrated() == migrate
    else:
        assert tex.migrated() == False

    with dr.suspend_grad():
        tex.tensor() # Might mutate some internal state
    assert dr.grad_enabled(tex.tensor())


@pytest.mark.parametrize("texture_type", ['Texture2f64', 'Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test27_masked(t, texture_type):
    # A masked lookup / fetch must return the unmasked value on active lanes and
    # zero on inactive ones. Covers the accelerated TexLookup (eval) and
    # TexFetchBilerp (eval_fetch) paths with a *mixed* per-lane mask.
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    PCG32 = getattr(mod, 'PCG32')
    UInt32 = dr.uint32_array_t(t)

    N, M, n = 4, 4, 64
    for ch in range(1, 9):
        tex = TexType([N, M], ch, True, dr.FilterMode.Linear, dr.WrapMode.Clamp)
        StorageType = dr.array_t(tex.value())
        tex.set_value(StorageType(PCG32(N * M * ch).next_float32()))

        rng = PCG32(n)
        pos = Array2f(rng.next_float32(), rng.next_float32())

        # Alternating active / inactive lanes; non-literal, so the masked
        # codegen path is exercised.
        active = (dr.arange(UInt32, n) & 1) == 0

        # eval -> hardware TexLookup
        ref = tex.eval(pos)
        out = tex.eval(pos, active=active)
        assert dr.allclose(out, dr.select(active, ref, 0))

        # eval_fetch -> hardware TexFetchBilerp
        ref_f = tex.eval_fetch(pos)
        out_f = tex.eval_fetch(pos, active=active)
        for corner in range(4):
            assert dr.allclose(out_f[corner], dr.select(active, ref_f[corner], 0))


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test28_write_flag(t, texture_type):
    # Writable textures expose write(); a non-writable one rejects it; and a
    # writable texture can be *both* written and sampled.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    UInt32 = dr.uint32_array_t(t)
    Array2u = getattr(mod, 'Array2u')
    Array2f = getattr(mod, 'Array2f')
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    tex = TexType([4, 8], 4, writable=True)
    assert tex.writable()

    ro = TexType([4, 8], 4)
    assert not ro.writable()
    px = dr.arange(UInt32, 8)

    # Writing to a non-writable texture is rejected.
    with pytest.raises(Exception):
        ro.write(Array2u(px, px), [StorageType(1)] * 4)

    # A writable texture can be both written and sampled: store a constant per
    # channel, then sample it back (constant -> independent of filtering/wrap).
    H, W = 4, 8
    idx = dr.arange(UInt32, H * W)
    wx, wy = idx % W, idx // W
    tex.write(Array2u(wx, wy), [dr.full(StorageType, 0.1 * (c + 1), H * W)
                                for c in range(4)])
    dr.eval()
    rng = mod.PCG32(16)
    out = tex.eval(Array2f(rng.next_float32(), rng.next_float32()))
    for c in range(4):
        assert dr.allclose(out[c], 0.1 * (c + 1), 5e-3, 5e-3)


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test29_write_read(t, texture_type):
    # Per-pixel stores round-trip through the texture for every channel
    # count (exercises the 1/2/4-channel sub-texture split and padding). The
    # value written at pixel `idx`, channel `c` is `(idx*ch + c)*0.01`, so the
    # interleaved read-back tensor must equal `arange(H*W*ch)*0.01`.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    UInt32 = dr.uint32_array_t(t)
    Array2u = getattr(mod, 'Array2u')
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    H, W = 4, 8
    for ch in range(1, 9):
        tex = TexType([H, W], ch, writable=True)
        idx = dr.arange(UInt32, H * W)
        px, py = idx % W, idx // W
        vals = [StorageType(idx * ch + c) * 0.01 for c in range(ch)]
        tex.write(Array2u(px, py), vals)
        dr.eval()
        ref = dr.arange(StorageType, H * W * ch) * 0.01
        assert dr.allclose(tex.value(), ref, atol=5e-3)

        # A second round exercises the readback view refresh: the read above
        # materialized the view, which pinned the old contents
        vals = [StorageType(idx * ch + c) * 0.02 for c in range(ch)]
        tex.write(Array2u(px, py), vals)
        dr.eval()
        assert dr.allclose(tex.value(), ref * 2, atol=5e-3)


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test30_write_masked(t, texture_type):
    # A masked store updates only the active lanes.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    UInt32 = dr.uint32_array_t(t)
    Array2u = getattr(mod, 'Array2u')
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    H, W = 4, 8
    tex = TexType([H, W], 1, writable=True)
    idx = dr.arange(UInt32, H * W)
    px, py = idx % W, idx // W

    tex.write(Array2u(px, py), [dr.zeros(StorageType, H * W)])
    dr.eval()
    tex.write(Array2u(px, py), [dr.full(StorageType, 1, H * W)], px < 4)
    dr.eval()

    ref = dr.select(px < 4, StorageType(1), StorageType(0))
    assert dr.allclose(tex.value(), ref, atol=5e-3)


@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test31_write_3d(t, texture_type):
    # 3D stores round-trip.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    UInt32 = dr.uint32_array_t(t)
    Array3u = getattr(mod, 'Array3u')
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    D, H, W, ch = 2, 4, 4, 4
    tex = TexType([D, H, W], ch, writable=True)
    idx = dr.arange(UInt32, D * H * W)
    px = idx % W
    py = (idx // W) % H
    pz = idx // (W * H)
    vals = [StorageType(idx * ch + c) * 0.01 for c in range(ch)]
    tex.write(Array3u(px, py, pz), vals)
    dr.eval()
    ref = dr.arange(StorageType, D * H * W * ch) * 0.01
    assert dr.allclose(tex.value(), ref, atol=5e-3)


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test32_from_native_handle(t, texture_type):
    # Wrap a native texture handle: build a texture, recover its native handle,
    # wrap that as a new texture, and confirm it samples identically. On Metal
    # native_handle() and from_native_handle() share the id<MTLTexture> type, so the
    # round-trip closes; on CUDA native_handle() is a CUtexObject while
    # from_native_handle() takes a GL id, so this is Metal-specific.
    if dr.backend_v(t) != dr.JitBackend.Metal:
        pytest.skip("native_handle/from_native_handle round-trip is Metal-specific")
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    PCG32 = getattr(mod, 'PCG32')
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    H, W, C = 5, 7, 4
    data = mod.TensorXf(StorageType(PCG32(H * W * C).next_float32()),
                        shape=(H, W, C))
    src = TexType(data, migrate=False)

    h = src.native_handle()
    assert h != 0

    wrapped = TexType.from_native_handle(h)
    assert not wrapped.writable()
    assert wrapped.shape == (H, W, C)

    wrapped.map()  # no-op on Metal
    rng = PCG32(64)
    pos = Array2f(rng.next_float32(), rng.next_float32())
    ref = src.eval(pos)
    out = wrapped.eval(pos)
    for ch in range(C):
        assert dr.allclose(ref[ch], out[ch], 5e-3, 5e-3)

    # tensor() reads back the wrapped texture's contents
    assert dr.allclose(wrapped.tensor().array, data.array, 5e-3, 5e-3)
    wrapped.unmap()

    # Dimensionality must match the texture type.
    with pytest.raises(Exception):
        getattr(mod, 'Texture3f').from_native_handle(h)

    # Wrap for *writing* (render into an app texture): write through a wrapped
    # handle, then read it back via the source texture (same native texture).
    UInt = getattr(mod, 'UInt')
    Array2u = getattr(mod, 'Array2u')
    wsrc = TexType([H, W], C, writable=True)
    wtex = TexType.from_native_handle(wsrc.native_handle(), writable=True)
    assert wtex.writable()
    wtex.map()
    idx = dr.arange(UInt, H * W)
    px, py = idx % W, idx // W
    wtex.write(Array2u(px, py),
               [StorageType(idx * C + c) * 0.01 for c in range(C)])
    dr.eval()
    wtex.unmap()
    assert dr.allclose(wsrc.value(),
                       dr.arange(StorageType, H * W * C) * 0.01, 5e-3, 5e-3)

    # A non-writable texture cannot be wrapped for writing.
    with pytest.raises(Exception):
        TexType.from_native_handle(src.native_handle(), writable=True)


def _srgb_to_linear(u):
    x = u / 255.0
    return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4


@pytest.mark.parametrize("srgb", [False, True])
@pytest.mark.parametrize("channels", [1, 3, 5])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test33_uint8(t, channels, srgb):
    # 8-bit textures normalize their 0..255 storage to [0, 1] on lookup,
    # optionally decoding sRGB. Pin the exact normalized value via a nearest
    # lookup, and require the hardware and arithmetic paths to agree.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f8u')
    UInt8 = getattr(mod, 'UInt8')
    Array2f = getattr(mod, 'Array2f')

    H, W = 4, 5
    vals = [(i * 37 + ch * 53) % 256
            for i in range(H * W) for ch in range(channels)]
    data = mod.TensorXu8(UInt8(vals), shape=(H, W, channels))

    tex = TexType(data, use_accel=True, migrate=False, srgb=srgb)
    tex_soft = TexType(data, use_accel=False, srgb=srgb)
    tex_near = TexType(data, use_accel=False, srgb=srgb,
                       filter_mode=dr.FilterMode.Nearest)
    assert tex.srgb() == srgb

    # Nearest lookup at the first texel's center returns its (decoded) value.
    # sRGB decoding (like the hardware) skips each RGBA group's alpha channel.
    out0 = tex_near.eval(Array2f(0.5 / W, 0.5 / H))
    for ch in range(channels):
        ref = (_srgb_to_linear(vals[ch]) if srgb and ch % 4 != 3
               else vals[ch] / 255.0)
        assert dr.allclose(out0[ch], ref, 1e-3, 1e-3)

    # Hardware and arithmetic linear lookups agree
    pos = Array2f([0.2, 0.55, 0.9], [0.3, 0.6, 0.8])
    out_soft, out_accel = tex_soft.eval(pos), tex.eval(pos)
    for ch in range(channels):
        assert dr.allclose(out_soft[ch], out_accel[ch], 5e-3, 5e-3)


@pytest.test_arrays("is_diff, float32, shape=(*)")
def test34_uint8_eval_variants(t):
    # The full lookup surface (fetch / cubic / their derivatives) normalizes
    # 8-bit storage like eval(), and the hardware and arithmetic paths agree.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f8u')
    UInt8 = getattr(mod, 'UInt8')
    Array2f = getattr(mod, 'Array2f')

    H, W, C = 5, 6, 3
    vals = [(i * 29 + ch * 71) % 256
            for i in range(H * W) for ch in range(C)]
    data = mod.TensorXu8(UInt8(vals), shape=(H, W, C))
    tex = TexType(data, use_accel=True, migrate=False)
    tex_soft = TexType(data, use_accel=False)
    pos = Array2f([0.3, 0.6], [0.4, 0.7])

    # eval_fetch: each corner is normalized, and the paths agree
    fa, fs = tex.eval_fetch(pos), tex_soft.eval_fetch(pos)
    for corner in range(4):
        for ch in range(C):
            assert dr.all((fs[corner][ch] >= 0) & (fs[corner][ch] <= 1))
            assert dr.allclose(fa[corner][ch], fs[corner][ch], 5e-3, 5e-3)

    # eval_cubic: hardware and arithmetic agree
    ca, cs = tex.eval_cubic(pos), tex_soft.eval_cubic(pos)
    for ch in range(C):
        assert dr.allclose(ca[ch], cs[ch], 5e-3, 5e-3)

    # eval_cubic_grad / eval_cubic_hessian run and produce finite results
    _, grad = tex_soft.eval_cubic_grad(pos)
    _, _, hess = tex_soft.eval_cubic_hessian(pos)
    assert dr.all(dr.isfinite(dr.ravel(grad[0])), axis=None)
    assert dr.all(dr.isfinite(dr.ravel(hess[0])), axis=None)


@pytest.mark.parametrize("use_accel", [False, True])
@pytest.test_arrays("is_diff, float32, shape=(*)")
def test35_uint8_grad(t, use_accel):
    # An 8-bit lookup is differentiable w.r.t. the (continuous) query position,
    # but its integer storage carries no gradient.
    if use_accel and dr.backend_v(t) not in (dr.JitBackend.CUDA, dr.JitBackend.Metal):
        pytest.skip("hardware textures require the CUDA or Metal backend")
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f8u')
    UInt8 = getattr(mod, 'UInt8')
    Array2f = getattr(mod, 'Array2f')

    H, W, C = 4, 4, 2
    vals = [(i * 17 + ch * 91) % 256
            for i in range(H * W) for ch in range(C)]
    data = mod.TensorXu8(UInt8(vals), shape=(H, W, C))

    # Integer storage cannot be made differentiable
    assert not dr.grad_enabled(data.array)

    tex = TexType(data, use_accel=use_accel, migrate=False)
    pos = Array2f(0.5, 0.5)
    dr.enable_grad(pos)
    out = tex.eval(pos)
    dr.backward(out[0] + out[1])
    g = dr.grad(pos)
    assert dr.all(dr.isfinite(g))


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test36_uint8_write(t):
    # write() to an 8-bit texture quantizes [0, 1] floats to normalized storage,
    # sRGB-encoding when requested (leaving each RGBA group's alpha linear), so a
    # sampled read round-trips the written value. Metal's unorm/sRGB pixel format
    # converts in hardware; CUDA and LLVM do it in software.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f8u')
    UInt32 = dr.uint32_array_t(t)
    Array2u = getattr(mod, 'Array2u')
    Array2f = getattr(mod, 'Array2f')

    H, W = 4, 4
    for srgb in (False, True):
        for ch in (1, 3, 4):
            tex = TexType([H, W], ch, use_accel=True, writable=True,
                          filter_mode=dr.FilterMode.Nearest, srgb=srgb)
            idx = dr.arange(UInt32, H * W)
            vals = [t(idx * ch + c) * (1.0 / (H * W * ch)) for c in range(ch)]
            tex.write(Array2u(idx % W, idx // W), vals)
            dr.eval()
            for i in (0, H * W // 2, H * W - 1):
                out = tex.eval(Array2f((i % W + 0.5) / W, (i // W + 0.5) / H))
                for c in range(ch):
                    v = (i * ch + c) / (H * W * ch)
                    assert dr.allclose(out[c], v, atol=7e-3)

    # The encoding actually happens: mid-gray 0.5 stores 128 linearly but 188 in
    # sRGB, while an RGBA group's alpha channel stays linear either way.
    def stored(srgb):
        tex = TexType([1, 1], 4, use_accel=True, writable=True, srgb=srgb)
        tex.write(Array2u(0, 0), [t(0.5)] * 4)
        dr.eval()
        return [int(x) for x in tex.value()]
    assert stored(False) == [128, 128, 128, 128]
    assert stored(True) == [188, 188, 188, 128]


# -----------------------------------------------------------------------
#                        MIP-mapped texture lookups
# -----------------------------------------------------------------------

def _box_downsample(vals, w, h):
    """Halve a row-major single-channel grid with the pyramid's box filter
    (odd sizes clamp the last tap onto the boundary texel)"""
    w2, h2 = max(w // 2, 1), max(h // 2, 1)
    out = []
    for y in range(h2):
        for x in range(w2):
            acc = 0.0
            for dy in range(2):
                for dx in range(2):
                    sx, sy = min(2 * x + dx, w - 1), min(2 * y + dy, h - 1)
                    acc += vals[sy * w + sx]
            out.append(acc / 4)
    return out, w2, h2


def _bilerp(vals, w, h, u, v, wrap):
    """Bilinear reference lookup of a row-major single-channel grid"""
    import math

    def texel(x, y):
        if wrap == dr.WrapMode.Repeat:
            x, y = x % w, y % h
        elif wrap == dr.WrapMode.Mirror:
            def m(i, n):
                i = i % (2 * n)
                return i if i < n else 2 * n - 1 - i
            x, y = m(x, w), m(y, h)
        else:
            x, y = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
        return vals[y * w + x]

    fx, fy = u * w - 0.5, v * h - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    wx, wy = fx - x0, fy - y0
    return (texel(x0, y0) * (1 - wx) * (1 - wy) +
            texel(x0 + 1, y0) * wx * (1 - wy) +
            texel(x0, y0 + 1) * (1 - wx) * wy +
            texel(x0 + 1, y0 + 1) * wx * wy)


def _test_grid(n, seed=0):
    """Deterministic pseudo-random values in [0, 1)"""
    return [((i * 37 + seed * 51 + 13) % 61) / 61.0 for i in range(n)]


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f64'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test37_mip_lod(t, texture_type):
    # eval_lod() reproduces the box-filtered pyramid levels exactly at their
    # texel centers, blends adjacent levels for fractional inputs, clamps
    # out-of-range levels, and degrades to eval() without a pyramid.
    _skip_metal_f64(t, texture_type)
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')

    vals = _test_grid(16)
    l1, w1, h1 = _box_downsample(vals, 4, 4)
    l2, _, _ = _box_downsample(l1, w1, h1)

    tex = TexType([4, 4], 1, use_accel=False, mip_filter=dr.MipFilter.Linear)
    tex.set_value(t(vals))
    assert tex.mip_levels() == 3
    assert tex.mip_filter() == dr.MipFilter.Linear

    # Pyramid contents at the texel centers of each level
    for y in range(2):
        for x in range(2):
            p = Array2f((x + 0.5) / 2, (y + 0.5) / 2)
            assert dr.allclose(tex.eval_lod(p, 1.0)[0], l1[y * 2 + x])
    p = Array2f(0.77, 0.13)
    assert dr.allclose(tex.eval_lod(p, 2.0)[0], l2[0])

    # A fractional LOD blends the two enclosing levels
    p = Array2f(0.25, 0.25)
    v0 = tex.eval_lod(p, 0.0)[0]
    v1 = tex.eval_lod(p, 1.0)[0]
    vf = tex.eval_lod(p, 0.3)[0]
    assert dr.allclose(vf, dr.fma(v1 - v0, 0.3, v0))

    # Out-of-range LODs clamp to the pyramid
    assert dr.allclose(tex.eval_lod(p, 99.0)[0], l2[0])
    assert dr.allclose(tex.eval_lod(p, -5.0)[0], v0)

    # The nearest MIP filter rounds to the closest level
    tex_n = TexType([4, 4], 1, use_accel=False, mip_filter=dr.MipFilter.Nearest)
    tex_n.set_value(t(vals))
    assert dr.allclose(tex_n.eval_lod(p, 0.4)[0], v0)
    assert dr.allclose(tex_n.eval_lod(p, 0.6)[0], v1)

    # Without a pyramid, eval_lod() degrades to eval()
    tex_d = TexType([4, 4], 1, use_accel=False)
    tex_d.set_value(t(vals))
    assert tex_d.mip_levels() == 1
    assert dr.allclose(tex_d.eval_lod(p, 2.0)[0], tex_d.eval(p)[0])


@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test38_mip_wrap(t, wrap_mode):
    # Lookups within the pyramid apply the wrap mode with each level's own
    # resolution. The 6x4 texture makes the levels (3x2, 1x1) exercise the
    # per-level division constants of the Repeat/Mirror wrap math.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f')
    Array2f = getattr(mod, 'Array2f')

    vals = _test_grid(24, seed=1)
    l1, w1, h1 = _box_downsample(vals, 6, 4)

    tex = TexType([4, 6], 1, use_accel=False, wrap_mode=wrap_mode,
                  mip_filter=dr.MipFilter.Linear)
    tex.set_value(t(vals))
    assert tex.mip_levels() == 3

    for u, v in [(0.05, 0.02), (0.98, 0.5), (-0.3, 1.7), (0.5, -0.01),
                 (0.31, 0.87)]:
        got = tex.eval_lod(Array2f(u, v), 1.0)[0]
        assert dr.allclose(got, _bilerp(l1, w1, h1, u, v, wrap_mode),
                           atol=1e-6)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test39_mip_filtered(t):
    # eval_filtered() implements anisotropic filtering: a footprint averages
    # taps along its major axis at the LOD of the tap extent, and clamping
    # the tap count coarsens the lookup instead.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f')
    Array2f = getattr(mod, 'Array2f')

    vals = _test_grid(64, seed=2)
    tex = TexType([8, 8], 1, use_accel=False, mip_filter=dr.MipFilter.Linear,
                  max_aniso=4)
    tex.set_value(t(vals))
    assert tex.max_aniso() == 4

    # A 4:1 footprint of four texels averages 4 taps at LOD 0
    p, u = (0.4, 0.45), 4.0 / 8.0
    ref = t(0)
    for i in range(4):
        tap = Array2f(p[0] + u * ((i + 0.5) / 4 - 0.5), p[1])
        ref += tex.eval_lod(tap, 0.0)[0]
    ref *= 0.25
    got = tex.eval_filtered(Array2f(*p), Array2f(u, 0), Array2f(0, 1 / 8))[0]
    assert dr.allclose(got, ref, atol=1e-6)

    # With a single allowed tap, the same footprint coarsens to LOD 2
    tex_iso = TexType([8, 8], 1, use_accel=False,
                      mip_filter=dr.MipFilter.Linear, max_aniso=1)
    tex_iso.set_value(t(vals))
    got = tex_iso.eval_filtered(Array2f(*p), Array2f(u, 0),
                                Array2f(0, 1 / 8))[0]
    assert dr.allclose(got, tex_iso.eval_lod(Array2f(*p), 2.0)[0], atol=1e-6)

    # A vanishing footprint reproduces the base-level lookup
    zero = Array2f(0, 0)
    got = tex.eval_filtered(Array2f(*p), zero, zero)[0]
    assert dr.allclose(got, tex.eval(Array2f(*p))[0])

    # Repeat wrap: taps stepping across the texture boundary wrap around
    tex_r = TexType([8, 8], 1, use_accel=False, wrap_mode=dr.WrapMode.Repeat,
                    mip_filter=dr.MipFilter.Linear, max_aniso=4)
    tex_r.set_value(t(vals))
    pr = (0.03, 0.45)
    ref = t(0)
    for i in range(4):
        tap = Array2f(pr[0] + u * ((i + 0.5) / 4 - 0.5), pr[1])
        ref += tex_r.eval_lod(tap, 0.0)[0]
    ref *= 0.25
    got = tex_r.eval_filtered(Array2f(*pr), Array2f(u, 0), Array2f(0, 1 / 8))[0]
    assert dr.allclose(got, ref, atol=1e-6)

    # Masked lanes return zero
    Bool = getattr(mod, 'Bool')
    p2 = Array2f(t(0.4, 0.6), t(0.45, 0.2))
    ddx, ddy = Array2f(t(u, u), t(0, 0)), Array2f(t(0, 0), t(1 / 8, 1 / 8))
    out = tex.eval_filtered(p2, ddx, ddy, Bool(True, False))
    assert out[0][1] == 0


@pytest.test_arrays("is_diff, float32, shape=(*)")
def test40_mip_grad(t):
    # Derivatives flow through the pyramid generation into the base texels:
    # a level-1 texel-center lookup distributes its gradient over the 2x2
    # base quadrant, and the anisotropic tap loop preserves a unit total.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f')
    TensorXf = getattr(mod, 'TensorXf')
    Array2f = getattr(mod, 'Array2f')

    tens = TensorXf(t(_test_grid(64, seed=3)), shape=(8, 8, 1))
    dr.enable_grad(tens)
    tex = TexType(tens, use_accel=False, mip_filter=dr.MipFilter.Linear,
                  max_aniso=4)

    # (1/8, 1/8) is the center of level-1 texel (0, 0)
    out = tex.eval_lod(Array2f(1 / 8, 1 / 8), 1.0)
    dr.backward(out[0])
    g = dr.grad(tens).array
    for i in range(64):
        expected = 0.25 if (i % 8) < 2 and (i // 8) < 2 else 0.0
        assert dr.allclose(g[i], expected)

    # The tap loop of eval_filtered() distributes a unit gradient
    dr.clear_grad(tens)
    out = tex.eval_filtered(Array2f(0.4, 0.45), Array2f(0.5, 0),
                            Array2f(0, 1 / 8))
    dr.backward(out[0])
    assert dr.allclose(dr.sum(dr.grad(tens).array), 1.0)

    # Forward derivatives with respect to the query position
    p = Array2f(0.3, 0.4)
    dr.enable_grad(p)
    out = tex.eval_lod(p, 0.5)
    dr.forward_from(p.x)
    assert dr.all(dr.isfinite(dr.grad(out[0])))

    # eval_filtered() position derivative, checked against finite differences
    ddx, ddy = Array2f(0.25, 0), Array2f(0, 1 / 8)
    p = Array2f(0.3, 0.4)
    dr.enable_grad(p)
    out = tex.eval_filtered(p, ddx, ddy)
    dr.forward_from(p.x)
    eps = 1e-3
    f0 = tex.eval_filtered(Array2f(0.3 - eps, 0.4), ddx, ddy)[0]
    f1 = tex.eval_filtered(Array2f(0.3 + eps, 0.4), ddx, ddy)[0]
    assert dr.allclose(dr.grad(out[0]), (f1 - f0) / (2 * eps), atol=1e-3)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test41_mip_uint8(t):
    # 8-bit sRGB pyramids average in linear space and re-encode each level
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f8u')
    UInt8 = getattr(mod, 'UInt8')
    Array2f = getattr(mod, 'Array2f')

    vals = [(i * 37 + 13) % 256 for i in range(16)]
    data = mod.TensorXu8(UInt8(vals), shape=(4, 4, 1))
    tex = TexType(data, use_accel=False, srgb=True,
                  mip_filter=dr.MipFilter.Linear)

    def linear_to_srgb(x):
        return x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055

    lin = [_srgb_to_linear(v) for v in vals]
    l1, _, _ = _box_downsample(lin, 4, 4)
    ref = [_srgb_to_linear(round(linear_to_srgb(v) * 255)) for v in l1]

    for y in range(2):
        for x in range(2):
            got = tex.eval_lod(Array2f((x + 0.5) / 2, (y + 0.5) / 2), 1.0)[0]
            assert dr.allclose(got, ref[y * 2 + x], atol=2e-3)

    # Four-channel variant: the alpha channel averages without the sRGB
    # transfer function
    vals = [(i * 53 + 7) % 256 for i in range(16)]
    data = mod.TensorXu8(UInt8(vals), shape=(2, 2, 4))
    tex = TexType(data, use_accel=False, srgb=True,
                  mip_filter=dr.MipFilter.Linear)
    got = tex.eval_lod(Array2f(0.5, 0.5), 1.0)
    for ch in range(4):
        chan = vals[ch::4]
        if ch == 3:
            ref_ch = int(sum(chan) / 4 + 0.5) / 255
        else:
            m = sum(_srgb_to_linear(v) for v in chan) / 4
            ref_ch = _srgb_to_linear(int(linear_to_srgb(m) * 255 + 0.5))
        assert dr.allclose(got[ch], ref_ch, atol=2e-3)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test42_mip_1d_3d(t):
    # The pyramid generation and level lookups cover all dimensionalities
    mod = sys.modules[t.__module__]

    Tex1 = getattr(mod, 'Texture1f')
    tex1 = Tex1([4], 1, use_accel=False, mip_filter=dr.MipFilter.Linear)
    tex1.set_value(t(0.0, 0.25, 0.75, 0.5))
    assert tex1.mip_levels() == 3
    assert dr.allclose(tex1.eval_lod(t(0.25), 1.0)[0], 0.125)
    assert dr.allclose(tex1.eval_lod(t(0.75), 1.0)[0], 0.625)
    assert dr.allclose(tex1.eval_lod(t(0.1), 2.0)[0], 0.375)

    Tex3 = getattr(mod, 'Texture3f')
    Array3f = getattr(mod, 'Array3f')
    vals = [float(x + 4 * y + 16 * z)
            for z in range(4) for y in range(4) for x in range(4)]
    tex3 = Tex3([4, 4, 4], 1, use_accel=False, mip_filter=dr.MipFilter.Linear)
    tex3.set_value(t(vals))
    assert tex3.mip_levels() == 3
    # Level-1 texel (0, 0, 0) averages the 8 corner texels
    ref = (0 + 1 + 4 + 5 + 16 + 17 + 20 + 21) / 8.0
    assert dr.allclose(tex3.eval_lod(Array3f(0.25, 0.25, 0.25), 1.0)[0], ref)
    # The top level holds the global mean
    assert dr.allclose(tex3.eval_lod(Array3f(0.9, 0.1, 0.5), 2.0)[0],
                       sum(vals) / 64)


def test43_mip_scalar():
    # The scalar (non-JIT) backend shares the pyramid and lookup code paths
    from drjit.scalar import TensorXf, Texture2f, Array2f

    vals = _test_grid(16, seed=4)
    l1, _, _ = _box_downsample(vals, 4, 4)
    tex = Texture2f(TensorXf(vals, shape=(4, 4, 1)),
                    mip_filter=dr.MipFilter.Linear,
                    wrap_mode=dr.WrapMode.Repeat)
    assert tex.mip_levels() == 3
    assert dr.allclose(tex.eval_lod(Array2f(0.25, 0.25), 1.0)[0], l1[0])
    assert dr.allclose(tex.eval_lod(Array2f(-0.75, 1.25), 1.0)[0], l1[0])
    got = tex.eval_filtered(Array2f(0.4, 0.45), Array2f(0.5, 0),
                            Array2f(0, 0.25))
    assert dr.all(dr.isfinite(got[0]))


@pytest.test_arrays("is_diff, float32, shape=(*)")
def test44_mip_accel(t):
    # The hardware MIP sampling path (CUDA/Metal texture units) agrees with
    # the arithmetic reference within the fixed-point weight precision of the
    # texture units, and derivative tracking splices the hardware primal onto
    # the arithmetic gradient.
    mod = sys.modules[t.__module__]
    if dr.backend_v(t) == dr.JitBackend.LLVM:
        pytest.skip("no hardware texture units")
    TexType = getattr(mod, 'Texture2f')
    TensorXf = getattr(mod, 'TensorXf')
    Array2f = getattr(mod, 'Array2f')

    vals = _test_grid(64, seed=5)
    tens = TensorXf(t(vals), shape=(8, 8, 1))
    hw = TexType(tens, use_accel=True, migrate=False,
                 mip_filter=dr.MipFilter.Linear, max_aniso=4)
    sw = TexType(tens, use_accel=False,
                 mip_filter=dr.MipFilter.Linear, max_aniso=4)

    pos = Array2f(t(0.3, 0.62, 0.85, 0.13), t(0.4, 0.18, 0.77, 0.95))
    for lod in [0.0, 0.7, 1.3, 2.0, 3.0]:
        assert dr.allclose(hw.eval_lod(pos, lod)[0], sw.eval_lod(pos, lod)[0],
                           rtol=5e-3, atol=5e-3)

    # Isotropic footprints select levels the same way on both paths
    for s in [0.5 / 8, 1 / 8, 4 / 8]:
        assert dr.allclose(hw.eval_filtered(pos, Array2f(s, 0), Array2f(0, s))[0],
                           sw.eval_filtered(pos, Array2f(s, 0), Array2f(0, s))[0],
                           rtol=5e-3, atol=5e-3)

    # Anisotropic tap placement is vendor-specific; require the same ballpark
    got = hw.eval_filtered(pos, Array2f(4 / 8, 0), Array2f(0, 1 / 8))[0]
    ref = sw.eval_filtered(pos, Array2f(4 / 8, 0), Array2f(0, 1 / 8))[0]
    assert dr.allclose(got, ref, rtol=0.1, atol=0.05)

    # Fully migrated textures sample the pyramid from texture memory alone
    hw_m = TexType(TensorXf(t(vals), shape=(8, 8, 1)), use_accel=True,
                   migrate=True, mip_filter=dr.MipFilter.Linear)
    ref = (vals[0] + vals[1] + vals[8] + vals[9]) / 4  # level-1 texel (0, 0)
    assert dr.allclose(hw_m.eval_lod(Array2f(1 / 8, 1 / 8), 1.0)[0], ref,
                       rtol=3e-3, atol=3e-3)

    # AD: primal from the hardware, gradient from the arithmetic formulation
    dr.enable_grad(tens)
    hw_ad = TexType(tens, use_accel=True, migrate=False,
                    mip_filter=dr.MipFilter.Linear)
    out = hw_ad.eval_lod(Array2f(1 / 8, 1 / 8), 1.0)
    dr.backward(out[0])
    g = dr.grad(tens).array
    for i in range(64):
        expected = 0.25 if (i % 8) < 2 and (i // 8) < 2 else 0.0
        assert dr.allclose(g[i], expected)


@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test45_migrated_tensor_semantics(t, texture_type):
    # The tensor of a migrated texture is an unevaluated readback expression
    # that reflects the texture contents at the time it is evaluated.
    # Evaluating it pins the contents, so an evaluated tensor is unaffected
    # by later updates of the texture.
    if dr.backend_v(t) not in (dr.JitBackend.CUDA, dr.JitBackend.Metal):
        pytest.skip("requires hardware textures")
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    StorageType = getattr(mod, 'Float16' if texture_type.endswith('f16') else 'Float')

    tex = TexType([2, 2], 1)
    TensorType = type(tex.tensor())

    a = StorageType(1, 2, 3, 4)
    tex.set_tensor(TensorType(a, shape=(2, 2, 1)), migrate=True)
    assert tex.migrated()
    assert tex.tensor().array.state == dr.VarState.Unevaluated

    held = TensorType(tex.tensor())
    dr.eval(held)

    b = StorageType(5, 6, 7, 8)
    tex.set_tensor(TensorType(b, shape=(2, 2, 1)), migrate=True)
    assert dr.all(held.array == a)
    assert dr.all(tex.tensor().array == b)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test46_filtered_rotation_invariance(t):
    # An isotropic footprint must resolve to a single tap at the same LOD
    # regardless of its rotation. Rounding error in the anisotropy ratio
    # used to bump ceil() to two taps at some angles, filtering one level
    # too sharply.
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f')
    TensorXf = getattr(mod, 'TensorXf')
    Array2f = getattr(mod, 'Array2f')
    UInt32 = getattr(mod, 'UInt32')

    res = 256
    tex = TexType(TensorXf(_test_grid(res * res, seed=6), shape=(res, res, 1)),
                  use_accel=False, mip_filter=dr.MipFilter.Linear, max_aniso=16)

    # A few lookup positions, swept over rotation angles in one-degree steps
    n_pos, n_ang = 4, 91
    idx = dr.arange(UInt32, n_pos * n_ang)
    pos = Array2f(dr.gather(t, t(0.3, 0.62, 0.45, 0.71), idx % n_pos),
                  dr.gather(t, t(0.4, 0.35, 0.68, 0.52), idx % n_pos))
    theta = dr.deg2rad(t(idx // n_pos))
    s, c = dr.sincos(theta)

    extent = 8.0 / res  # an 8-texel isotropic footprint, i.e. lod 3
    out = tex.eval_filtered(pos, Array2f(c * extent, s * extent),
                            Array2f(-s * extent, c * extent))
    ref = tex.eval_lod(pos, 3.0)
    assert dr.allclose(out[0], ref[0], rtol=1e-4, atol=1e-5)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test47_filtered_shear_invariance(t):
    # Rotating the footprint parameterization, (ddx, ddy) -> (c*ddx + s*ddy,
    # -s*ddx + c*ddy), sweeps the same ellipse, so the filtered result must
    # not change even though the new axis pair is no longer orthogonal.
    import math
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture2f')
    TensorXf = getattr(mod, 'TensorXf')
    Array2f = getattr(mod, 'Array2f')

    res = 256
    tex = TexType(TensorXf(_test_grid(res * res, seed=7), shape=(res, res, 1)),
                  use_accel=False, mip_filter=dr.MipFilter.Linear, max_aniso=16)
    pos = Array2f(t(0.3, 0.62, 0.45, 0.71), t(0.4, 0.35, 0.68, 0.52))

    th = math.radians(20.0)
    ddx = (math.cos(th) * 16 / res, math.sin(th) * 16 / res)
    ddy = (-math.sin(th) * 4 / res, math.cos(th) * 4 / res)
    ref = tex.eval_filtered(pos, Array2f(*ddx), Array2f(*ddy))

    for alpha in (30.0, 45.0, 75.0):
        c, s = math.cos(math.radians(alpha)), math.sin(math.radians(alpha))
        ddx2 = Array2f(c * ddx[0] + s * ddy[0], c * ddx[1] + s * ddy[1])
        ddy2 = Array2f(c * ddy[0] - s * ddx[0], c * ddy[1] - s * ddx[1])
        out = tex.eval_filtered(pos, ddx2, ddy2)
        assert dr.allclose(out[0], ref[0], rtol=1e-4, atol=1e-4)


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test48_mip_accel_1d(t):
    # 1D MIP-mapped hardware textures used to fail at creation on CUDA
    # because the resource view dimensions disagreed with the mipmapped array
    if dr.backend_v(t) == dr.JitBackend.LLVM:
        pytest.skip("no hardware texture units")
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, 'Texture1f')
    TensorXf = getattr(mod, 'TensorXf')

    vals = _test_grid(64, seed=8)
    tens = TensorXf(t(vals), shape=(64, 1))
    hw = TexType(tens, use_accel=True, migrate=False,
                 mip_filter=dr.MipFilter.Linear, max_aniso=16)
    sw = TexType(tens, use_accel=False,
                 mip_filter=dr.MipFilter.Linear, max_aniso=16)
    pos = t(0.1, 0.33, 0.52, 0.85)
    for lod in [0.0, 1.0, 2.0, 3.0]:
        assert dr.allclose(hw.eval_lod(pos, lod)[0], sw.eval_lod(pos, lod)[0],
                           rtol=5e-3, atol=5e-3)
