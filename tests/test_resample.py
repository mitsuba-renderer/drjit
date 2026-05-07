import drjit as dr
import pytest

# Test simple upsampling with a box filter (1D array)
@pytest.test_arrays('float, -jit, shape=(*)')
def test01_box_upsample_1d(t):
    r = dr.resample(
        t(1, 2, 3),
        (6,),
        filter='box'
    )
    assert dr.all(r == t(1, 1, 2, 2, 3, 3))

# Test simple upsampling with a box filter (2D tensor)
@pytest.test_arrays('float, -jit, tensor')
def test02_box_upsample_2d(t):
    r = dr.resample(
        t([[1, 2],[3, 4]]),
        (4, 4),
        filter='box'
    )
    assert dr.all(r == t([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]))

# Test simple downsampling with a box filter
@pytest.test_arrays('float, -jit, shape=(*)')
def test03_box_downsample(t):
    r = dr.resample(
        t(1, 2, 10, 11, 100, 101),
        (3,),
        filter='box'
    )
    assert dr.all(r == t(1.5, 10.5, 100.5))

# Validate a range of resampling operations against PIL
@pytest.mark.parametrize('filter', [('box', 'BOX'),
                                    ('linear', 'BILINEAR'),
                                    ('hamming', 'HAMMING'),
                                    ('cubic', 'BICUBIC'),
                                    ('lanczos', 'LANCZOS')])
@pytest.test_arrays('float32, shape=(*)')
def test04_pil_consistency(t, filter):
    # Exhaustively check 400 differently sized resampling operations
    # against PIL (once for each of the 5 builtin filters)

    Image = pytest.importorskip("PIL.Image")
    np = pytest.importorskip("numpy")

    for i in range(1, 20):
        np.random.seed(0)
        x = np.float32(np.random.rand(i))
        for j in range(1, 20):
            filter_pil = getattr(Image, filter[1])
            x_ref = np.array(Image.fromarray(x).resize((1, j), filter_pil)).T
            x_dr = dr.resample(t(x), (j,), filter=filter[0]).array.numpy()
            assert np.allclose(x_ref, x_dr)



@pytest.mark.parametrize('filter', ['box', 'linear', 'cubic', 'lanczos'])
@pytest.test_arrays('float32, shape=(*), is_diff')
def test05_fwd_bwd(t, filter):
    grads = []
    for i in range(3):
        x = t([10, 100, 1000])
        dr.enable_grad(x)
        y = dr.resample(x, (4,), filter=filter)
        x.grad = t([i==0, i==1, i==2])
        dr.forward_to(y)
        grads.append(y.grad)

    x = t([10, 100, 1000])
    dr.enable_grad(x)
    y = dr.resample(x, (4,), filter=filter)
    grad_in = t([1, 10, 100, 1000])
    y.grad = grad_in
    dr.backward_to(x)
    ref = [
        dr.dot(grad_in, grads[0])[0],
        dr.dot(grad_in, grads[1])[0],
        dr.dot(grad_in, grads[2])[0]
    ]
    assert dr.allclose(ref, x.grad)


# Test manually specifying a sampler
@pytest.test_arrays('float, -jit, shape=(*)')
def test06_manual_filter(t):
    r1 = dr.resample(
        t(1, 2, 10, 11, 100, 101),
        (3,),
        filter='linear'
    )

    def filt(x):
        return max(1-abs(x), 0)

    r2 = dr.resample(
        t(1, 2, 10, 11, 100, 101),
        (3,),
        filter=filt,
        filter_radius=1
    )

    assert dr.allclose(r1, r2)

# Test filtering a signal without changing its resolution
@pytest.test_arrays('float, -jit, shape=(*)')
def test07_convolve(t):
    x = t(1, 2, 10, 100)
    y = dr.convolve(x, 'linear', 1)
    assert dr.allclose(x, y)

    y = dr.convolve(x, 'linear', 2)
    z = t((1+2*.5)*2/1.5, 1*.5+2+10*.5, 2*.5+10+100*.5, (100+10*.5)*2/1.5)
    assert dr.allclose(y, z)

    y = dr.convolve(x, 'linear', 2, boundary='zero')
    z = t(1+2*.5, 1*.5+2+10*.5, 2*.5+10+100*.5, 100+10*.5)
    assert dr.allclose(y, z)


# Test convolution with a discrete 1D kernel
@pytest.test_arrays('float, -jit, shape=(*)')
def test08_convolve_discrete_1d(t):
    x = t(1, 2, 10, 100)
    kernel = t(1, 2, 1)

    y = dr.convolve(x, kernel, boundary='zero')
    assert dr.allclose(y, t(4, 15, 122, 210))

    y = dr.convolve(x, kernel)
    assert dr.allclose(y, t(16/3, 15, 122, 280))

    kernel_t = dr.tensor_t(t)([1, 2, 1])
    with pytest.raises(TypeError, match="must be a 1D Dr.Jit array"):
        dr.convolve(x, kernel_t)

    with pytest.raises(TypeError, match="must be a 1D Dr.Jit array"):
        dr.convolve(x, [1, 2, 1])


# Test discrete convolution over selected tensor axes
@pytest.test_arrays('float, -jit, tensor')
def test09_convolve_discrete_tensor_axis(t):
    x = t([[1, 2, 3], [4, 5, 6]])
    kernel = dr.array_t(t)(1, 2, 1)

    y = dr.convolve(x, kernel, axis=1)
    z = t([[16/3, 8, 32/3], [52/3, 20, 68/3]])
    assert dr.all(dr.allclose(y, z), axis=None)

    y = dr.convolve(x, kernel, axis=0, boundary='zero')
    z = t([[6, 9, 12], [9, 12, 15]])
    assert dr.all(dr.allclose(y, z), axis=None)
