import drjit as dr
import pytest
import sys

# Work around a refleak in @pytest.mark.parameterize
wrap_modes = [dr.WrapMode.Repeat, dr.WrapMode.Clamp, dr.WrapMode.Mirror]

@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("force_optix", [True, False])
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.mark.skip()
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test01_interp_1d(t, wrap_mode, force_optix, texture_type):

    with dr.scoped_set_flag(dr.JitFlag.ForceOptiX, force_optix):
        mod = sys.modules[t.__module__]
        Array1f = getattr(mod, 'Array1f')
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

@pytest.mark.skip()
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test02_interp_1d(t, wrap_mode, texture_type):
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

        for j in range(0, 4):
            values = StorageType(rng_1.next_float32())
            tex.set_value(values)
            tex_no_accel.set_value(values)
            assert dr.allclose(tex.value(), values)
            pos = Array1f(rng_2.next_float32())
            result_drjit = tex_no_accel.eval(pos)
            dr.eval(result_drjit)
            result_accel = tex.eval(pos)
            dr.eval(result_accel)

            assert dr.allclose(result_drjit, result_accel, 5e-3, 5e-3)

@pytest.mark.skip()
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test03_interp_2d(t, wrap_mode, texture_type):
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

        for j in range(0, 4):
            values = rng_1.next_float32()
            tex.set_value(values)
            tex_no_accel.set_value(values)
            pos = Array2f(rng_2.next_float32(), rng_2.next_float32())
            result_drjit = tex_no_accel.eval(pos)
            dr.eval(result_drjit)
            result_accel = tex.eval(pos)
            dr.eval(result_accel)

            assert(dr.allclose(result_drjit, result_accel, 5e-3, 5e-3))

@pytest.mark.skip()
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test04_interp_3d(t, wrap_mode, texture_type):
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

        for j in range(0, 4):
            values = rng_1.next_float32()
            tex.set_value(values)
            tex_no_accel.set_value(values)
            pos = Array3f(rng_2.next_float32(), rng_2.next_float32(), rng_2.next_float32())
            result_drjit = tex_no_accel.eval(pos)
            dr.eval(result_drjit)
            result_accel = tex.eval(pos)
            dr.eval(result_accel)

            assert(dr.allclose(result_drjit, result_accel, 6e-3, 6e-3))

@pytest.mark.skip()
@pytest.mark.parametrize("migrate", [True, False])
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test05_grad(t, migrate, texture_type):
    mod = sys.modules[t.__module__]
    Float = getattr(mod, 'Float')
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

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

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test_06_nearest(t, texture_type):
    mod = sys.modules[t.__module__]
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)

    N = 3

    value = t(0, 0.5, 1)
    tex = TexType([N], 1, True, dr.FilterMode.Nearest, dr.WrapMode.Repeat)
    tex_no_accel = TexType([N], 1, True, dr.FilterMode.Nearest, dr.WrapMode.Repeat)
    tex.set_value(value)
    tex_no_accel.set_value(value)

    tex_no_accel = TexType([N], 1, False, dr.FilterMode.Nearest, dr.WrapMode.Repeat)
    tex_no_accel.set_value(value)

    pos = dr.linspace(t, 0, 1, 80)
    out_accel = tex.eval(pos)
    out_drjit = tex_no_accel.eval(pos)
    assert dr.allclose(out_accel, out_drjit)

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture1f'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
def test07_cubic_analytic(t, texture_type):
    mod = sys.modules[t.__module__]
    Array1f = getattr(mod, 'Array1f')
    TexType = getattr(mod, texture_type)

    N = 4

    tex = TexType([N], 1, True, dr.FilterMode.Linear, dr.WrapMode.Clamp)
    value = t(0, 1, 0, 0)
    tex.set_value(value)

    pos = Array1f(0.5)
    (val_64, grad_64) = tex.eval_cubic_grad(pos)
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

@pytest.mark.skip()
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.mark.parametrize("texture_type", ['Texture1f'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test08_cubic_interp_1d(t, texture_type, wrap_mode):
    mod = sys.modules[t.__module__]
    Array1f = getattr(mod, 'Array1f')
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

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture2f'])
@pytest.mark.parametrize("wrap_mode", wrap_modes)
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test09_cubic_interp_2d(t, texture_type, wrap_mode):
    mod = sys.modules[t.__module__]
    Array2f = getattr(mod, 'Array2f')
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

@pytest.mark.parametrize("texture_type", ['Texture3f'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test10_cubic_interp_3d(t, texture_type):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')
    Array3f = getattr(mod, 'Array3f')
    UInt32 = dr.uint32_array_t(t)

    for i in range(2000):
        level = dr.log_level()

        print(f'Iteration {i}')

        print('Post scatter')

        dummy_tex = TexType([1,1,1], 1)

        TensorType = type(dummy_tex.tensor())
        StorageType = dr.array_t(dummy_tex.value())

        s = 9
        tensor = dr.full(TensorType, 1, shape=[s, s, s, 2])
        dr.scatter(tensor.array, StorageType(0.0),  UInt32(728)) # tensor[4, 4, 4, 0] = 0.0
        dr.scatter(tensor.array, StorageType(2.0),  UInt32(546)) # tensor[3, 3, 3, 0] = 2.0
        dr.scatter(tensor.array, StorageType(10.0), UInt32(727)) # tensor[4, 4, 3, 1] = 10.0

        #dr.eval(tensor.array)
        #dr.sync_thread()

        print('Post scatter')

        tex = TexType(tensor, True, False, dr.FilterMode.Linear, dr.WrapMode.Clamp)

        ref = Array2f(0.71312, 1.86141)
        pos = Array3f(.49, .5, .5)
        res = tex.eval_cubic(pos, True, True)
        dr.eval(res)
        dr.sync_thread()
        print('Post cubic 1')
        res2 = tex.eval_cubic_helper(pos)
        #dr.eval(res)
        #dr.sync_thread()
        print('Post cubic helper 1')
        assert dr.allclose(res, ref, 2e-3, 2e-3)
        assert dr.allclose(res2, ref, 2e-3, 2e-3)

        ref2 = Array2f(0.800905, 2.60136)
        pos2 = Array3f(.45, .53, .51)
        res = tex.eval_cubic(pos2, True, True)
        #dr.eval(res)
        #dr.sync_thread()
        print('Post cubic 2')
        res2 = tex.eval_cubic_helper(pos2)
        #dr.eval(res2)
        #dr.sync_thread()
        print('Post cubic helper 2')
        assert dr.allclose(res, ref2, 2e-3, 2e-2)
        assert dr.allclose(res2, ref2, 2e-3, 2e-2)


@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture3f'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.mark.skipif(sys.platform == "win32", reason="FIXME: Non-deterministic crashes on Windows")
def test11_cubic_grad_pos(t, texture_type):
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

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture3f'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
def test12_cubic_hessian_pos(t, texture_type):
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

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test15_tensor_value_1d(t, texture_type):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N = 2
    for ch in range(1, 9):
        rng = PCG32(2 * ch)
        tex = TexType([N], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data)

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test16_tensor_value_2d(t, texture_type):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M = 2, 3
    for ch in range(1, 9):
        rng = PCG32(N * M * ch)
        tex = TexType([N, M], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data);

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test17_tensor_value_3d(t, texture_type):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    PCG32 = getattr(mod, 'PCG32')

    N, M, L = 2, 3, 4
    for ch in range(1, 9):
        rng = PCG32(N * M * L * ch)
        tex = TexType([N, M, L], ch, True)

        StorageType = dr.array_t(tex.value())
        tex_data = StorageType(rng.next_float32())
        tex.set_value(tex_data)

        assert dr.allclose(tex.value(), tex_data)
        assert dr.allclose(tex.tensor().array, tex_data)

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test18_fetch_1d(t, texture_type):
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
        out_drjit = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)
        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_drjit[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_drjit[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test19_fetch_2d(t, texture_type):
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
        out_drjit = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)
        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_drjit[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_drjit[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])
            assert dr.allclose(tex_data[2 * ch + k], out_drjit[2][k])
            assert dr.allclose(tex_data[2 * ch + k], out_accel[2][k])
            assert dr.allclose(tex_data[3 * ch + k], out_drjit[3][k])
            assert dr.allclose(tex_data[3 * ch + k], out_accel[3][k])

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture3f', 'Texture3f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test20_fetch_3d(t, texture_type):
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
        out_drjit = tex_no_accel.eval_fetch(pos)
        out_accel = tex.eval_fetch(pos)
        for k in range(0, ch):
            assert dr.allclose(tex_data[k], out_drjit[0][k])
            assert dr.allclose(tex_data[k], out_accel[0][k])
            assert dr.allclose(tex_data[ch + k], out_drjit[1][k])
            assert dr.allclose(tex_data[ch + k], out_accel[1][k])
            assert dr.allclose(tex_data[2 * ch + k], out_drjit[2][k])
            assert dr.allclose(tex_data[2 * ch + k], out_accel[2][k])
            assert dr.allclose(tex_data[3 * ch + k], out_drjit[3][k])
            assert dr.allclose(tex_data[3 * ch + k], out_accel[3][k])
            assert dr.allclose(tex_data[4 * ch + k], out_drjit[4][k])
            assert dr.allclose(tex_data[4 * ch + k], out_accel[4][k])
            assert dr.allclose(tex_data[5 * ch + k], out_drjit[5][k])
            assert dr.allclose(tex_data[5 * ch + k], out_accel[5][k])
            assert dr.allclose(tex_data[6 * ch + k], out_drjit[6][k])
            assert dr.allclose(tex_data[6 * ch + k], out_accel[6][k])
            assert dr.allclose(tex_data[7 * ch + k], out_drjit[7][k])
            assert dr.allclose(tex_data[7 * ch + k], out_accel[7][k])

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture1f', 'Texture1f16'])
@pytest.mark.parametrize("migrate", [True, False])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test21_fetch_migrate(t, texture_type, migrate):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array1f = getattr(mod, 'Array1f')

    N = 2
    tex = TexType([N], 1, True)
    tex_data = t(1.0, 2.0)
    tex.set_value(tex_data, migrate)

    pos = Array1f(0.5)
    out = tex.eval_fetch(pos)

    assert dr.allclose(out[0][0], 1.0)
    assert dr.allclose(out[1][0], 2.0)

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_diff, float32, shape=(*)")
@pytest.skip_on(RuntimeError, "backend does not support the requested type of atomic reduction")
def test22_fetch_grad(t, texture_type):
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

@pytest.mark.skip()
@pytest.mark.parametrize("texture_type", ['Texture2f', 'Texture2f16'])
@pytest.test_arrays("is_jit, float32, shape=(*)")
def test23_set_tensor(t, texture_type):
    mod = sys.modules[t.__module__]
    TexType = getattr(mod, texture_type)
    Array2f = getattr(mod, 'Array2f')

    tex = TexType([2, 2], 1, True)
    tex_no_accel = TexType([2, 2], 1, False)
    tex_data = t(1, 2, 3, 4)
    tex.set_value(tex_data);
    tex_no_accel.set_value(tex_data);

    TensorType = type(tex.tensor())

    new_tex_data = t(6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1)
    new_tensor = TensorType(new_tex_data, shape=(2, 3, 2))

    assert new_tensor.shape == (2,3,2)

    tex.set_tensor(new_tensor)
    tex_no_accel.set_tensor(new_tensor)

    dr.eval(tex)
    dr.eval(tex_no_accel)

    assert tex.tensor().shape == (2,3,2)

    pos = Array2f(0, 0)
    result_drjit = tex_no_accel.eval(pos)
    dr.eval(result_drjit)
    result_accel = tex.eval(pos)
    dr.eval(result_accel)
    assert dr.allclose(result_drjit, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_drjit, Array2f(6.5, 6))

    pos = Array2f(1, 1)
    result_drjit = tex_no_accel.eval(pos)
    dr.eval(result_drjit)
    result_accel = tex.eval(pos)
    dr.eval(result_accel)
    assert dr.allclose(result_drjit, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_drjit, Array2f(1.5, 1))

    pos = Array2f(0, 1)
    result_drjit = tex_no_accel.eval(pos)
    dr.eval(result_drjit)
    result_accel = tex.eval(pos)
    dr.eval(result_accel)
    assert dr.allclose(result_drjit, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_drjit, Array2f(3.5, 3))

    pos = Array2f(1, 0)
    result_drjit = tex_no_accel.eval(pos)
    dr.eval(result_drjit)
    result_accel = tex.eval(pos)
    dr.eval(result_accel)
    assert dr.allclose(result_drjit, result_accel, 5e-3, 5e-3)
    assert dr.allclose(result_drjit, Array2f(4.5, 4))
