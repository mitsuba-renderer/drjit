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
    z = t((1+2*.5)/1.5, (1*.5+2+10*.5)/2, (2*.5+10+100*.5)/2, (100+10*.5)/1.5)
    assert dr.allclose(y, z)


# A discrete kernel with zero padding reproduces np.convolve(..., mode='same')
@pytest.test_arrays('float32, shape=(*)')
def test08_convolve_discrete_numpy(t):
    np = pytest.importorskip("numpy")
    np.random.seed(0)
    x = np.float32(np.random.rand(13))
    for ksize in (1, 3, 5, 8):
        k = np.float32(np.random.rand(ksize) - 0.5)
        ref = np.convolve(x, k, mode='same')
        out = dr.convolve(t(x), list(k), boundary='zero', normalize=False)
        assert np.allclose(ref, out.numpy(), atol=1e-5)


# Each boundary mode matches the corresponding scipy.ndimage convolution
@pytest.mark.parametrize('boundary', [('zero', 'constant'),
                                      ('nearest', 'nearest'),
                                      ('wrap', 'grid-wrap'),
                                      ('reflect', 'reflect'),
                                      ('mirror', 'mirror')])
@pytest.test_arrays('float32, shape=(*)')
def test09_convolve_boundary(t, boundary):
    np = pytest.importorskip("numpy")
    ndimage = pytest.importorskip("scipy.ndimage")
    np.random.seed(1)
    x = np.float32(np.random.rand(11))
    k = np.float32([0.2, -1.0, 0.5, 2.0, 0.3])
    ref = ndimage.convolve(x, k, mode=boundary[1], cval=0.0)
    out = dr.convolve(t(x), list(k), boundary=boundary[0], normalize=False)
    assert np.allclose(ref, out.numpy(), atol=1e-5)


# Normalization preserves a constant signal (DC) for every boundary mode, and
# for boundary modes without masking it is just a global rescaling.
@pytest.mark.parametrize('boundary', ['zero', 'nearest', 'wrap', 'reflect', 'mirror'])
@pytest.test_arrays('float32, shape=(*)')
def test10_convolve_normalize(t, boundary):
    k = [1.0, 2.0, 3.0, 2.0, 1.0]
    c = dr.full(t, 7.0, 10)
    assert dr.allclose(dr.convolve(c, k, boundary=boundary, normalize=True), 7.0)

    if boundary != 'zero':
        x = t(1, 5, 2, 8, 3, 9, 4)
        raw = dr.convolve(x, k, boundary=boundary, normalize=False)
        norm = dr.convolve(x, k, boundary=boundary, normalize=True)
        assert dr.allclose(norm, raw * (1.0 / sum(k)))


# A 1D kernel applied along both axes equals a separable 2D convolution
@pytest.test_arrays('float32, tensor')
def test11_convolve_separable(t):
    np = pytest.importorskip("numpy")
    ndimage = pytest.importorskip("scipy.ndimage")
    np.random.seed(2)
    img = np.float32(np.random.rand(9, 7))
    # Use an odd-length kernel so the numpy/scipy centering conventions agree
    k = np.float32([0.5, 1.0, -0.25, 0.75, 0.2])
    ref = ndimage.convolve(img, np.outer(k, k), mode='reflect')
    out = dr.convolve(t(img), list(k), boundary='reflect', normalize=False)
    assert np.allclose(ref, out.numpy(), atol=1e-5)


# Forward/backward derivatives of a discrete convolution are consistent
@pytest.mark.parametrize('boundary', ['zero', 'wrap', 'reflect'])
@pytest.test_arrays('float32, shape=(*), is_diff')
def test12_convolve_grad(t, boundary):
    k = [0.5, 1.0, -0.3]

    grads = []
    for i in range(4):
        x = t(10, 100, 1000, 10000)
        dr.enable_grad(x)
        y = dr.convolve(x, k, boundary=boundary, normalize=False)
        x.grad = t(i == 0, i == 1, i == 2, i == 3)
        dr.forward_to(y)
        grads.append(y.grad)

    x = t(10, 100, 1000, 10000)
    dr.enable_grad(x)
    y = dr.convolve(x, k, boundary=boundary, normalize=False)
    grad_in = t(1, 10, 100, 1000)
    y.grad = grad_in
    dr.backward_to(x)
    ref = [dr.dot(grad_in, g)[0] for g in grads]
    assert dr.allclose(ref, x.grad)


# The two codegen strategies ('evaluated' / 'symbolic', and the 'None' default
# which selects 'evaluated') must produce identical results, for both the
# uniform-interior fast path and the general/boundary path.
@pytest.mark.parametrize('boundary', ['zero', 'nearest', 'wrap', 'reflect', 'mirror'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.test_arrays('float32, shape=(*)')
def test13_convolve_mode(t, boundary, normalize):
    np = pytest.importorskip("numpy")
    np.random.seed(3)
    x = t(np.float32(np.random.rand(64)))
    for k in ([0.1, 0.2, 0.4, 0.2, 0.1], [0.25, 0.5, 0.25, 0.3]):
        ref = dr.convolve(x, k, boundary=boundary, normalize=normalize,
                          mode='symbolic')
        for m in (None, 'evaluated'):
            out = dr.convolve(x, k, boundary=boundary, normalize=normalize,
                              mode=m)
            assert dr.allclose(out, ref, atol=1e-5)

    with pytest.raises(RuntimeError, match="'mode' must be"):
        dr.convolve(x, [1.0], mode='nonsense')


# Reverse-mode gradients of a convolution must satisfy the linear adjoint
# identity <g, A x> == <A^T g, x>. This guards the *backward* boundary handling
# for continuous filters (e.g. the gather-transpose vs. scatter adjoint), which
# the discrete-kernel gradient test above does not exercise. The array is chosen
# larger than the filter footprint so that the shift-invariant interior (and its
# transpose fast path) is active alongside the boundary.
@pytest.mark.parametrize('boundary', ['zero', 'nearest', 'wrap', 'reflect', 'mirror'])
@pytest.mark.parametrize('normalize', [False, True])
@pytest.test_arrays('float32, shape=(*), is_diff')
def test14_convolve_adjoint(t, boundary, normalize):
    np = pytest.importorskip("numpy")
    np.random.seed(4)
    xv = np.float32(np.random.rand(40))
    gv = np.float32(np.random.rand(40))

    x = t(xv)
    dr.enable_grad(x)
    y = dr.convolve(x, 'cubic', 2, boundary=boundary, normalize=normalize)

    # <g, A x>
    lhs = dr.dot(t(gv), dr.detach(y))[0]

    # backward yields x.grad = A^T g; compare <x, A^T g>
    y.grad = t(gv)
    dr.backward_to(x)
    rhs = dr.dot(t(xv), x.grad)[0]

    assert dr.allclose(lhs, rhs, rtol=1e-3)


# resample() exposes the same boundary modes as convolve(). Check that the
# boundary is actually threaded through (edge values change vs. zero padding)
# and that the resolution-changing forward/backward stay a valid adjoint pair.
@pytest.mark.parametrize('boundary', ['zero', 'nearest', 'wrap', 'reflect', 'mirror'])
@pytest.test_arrays('float32, shape=(*), is_diff')
def test15_resample_boundary(t, boundary):
    np = pytest.importorskip("numpy")
    np.random.seed(5)

    # The boundary must influence the edge samples (vs. zero padding)
    x = t(np.float32(np.random.rand(16)))
    base = dr.resample(x, (6,), filter='cubic', boundary='zero')
    out = dr.resample(x, (6,), filter='cubic', boundary=boundary)
    if boundary != 'zero':
        assert not dr.allclose(base, out)

    # Adjoint identity <g, A x> == <A^T g, x> for both down- and up-sampling
    for n_in, n_out in [(16, 6), (6, 16)]:
        xv = np.float32(np.random.rand(n_in))
        gv = np.float32(np.random.rand(n_out))
        x = t(xv)
        dr.enable_grad(x)
        y = dr.resample(x, (n_out,), filter='cubic', boundary=boundary)
        lhs = dr.dot(t(gv), dr.detach(y))[0]
        y.grad = t(gv)
        dr.backward_to(x)
        rhs = dr.dot(t(xv), x.grad)[0]
        assert dr.allclose(lhs, rhs, rtol=1e-3)

    # The two codegen strategies must agree for resampling as well
    z = t(np.float32(np.random.rand(20)))
    ev = dr.resample(z, (9,), filter='cubic', boundary=boundary, mode='evaluated')
    sy = dr.resample(z, (9,), filter='cubic', boundary=boundary, mode='symbolic')
    assert dr.allclose(ev, sy, atol=1e-5)


# A 1D kernel applied to a single axis matches scipy.ndimage.convolve1d, and
# negative axis indices resolve correctly.
@pytest.test_arrays('float32, tensor')
def test16_convolve_axis(t):
    np = pytest.importorskip("numpy")
    ndimage = pytest.importorskip("scipy.ndimage")
    np.random.seed(6)
    img = np.float32(np.random.rand(8, 6))
    k = np.float32([0.25, 0.6, 0.25])
    for ax in (0, 1):
        ref = ndimage.convolve1d(img, k, axis=ax, mode='reflect')
        out = dr.convolve(t(img), list(k), axis=ax, boundary='reflect',
                          normalize=False)
        assert np.allclose(ref, out.numpy(), atol=1e-5)
    a = dr.convolve(t(img), list(k), axis=-1, boundary='reflect', normalize=False)
    b = dr.convolve(t(img), list(k), axis=1, boundary='reflect', normalize=False)
    assert dr.allclose(a, b)


# Argument validation for the documented error conditions.
@pytest.test_arrays('float32, shape=(*)')
def test17_convolve_errors(t):
    x = t(1, 2, 3, 4, 5)
    # 'filter_radius' is meaningless for a discrete kernel
    with pytest.raises(RuntimeError, match="filter_radius' must be None"):
        dr.convolve(x, [1.0, 2.0, 1.0], filter_radius=1.0)
    # ... but required for a custom continuous filter
    with pytest.raises(RuntimeError, match="must be specified"):
        dr.convolve(x, lambda v: max(1.0 - abs(v), 0.0))
    # an empty kernel is rejected
    with pytest.raises(RuntimeError, match="kernel is empty"):
        dr.convolve(x, [])
    # unknown boundary mode
    with pytest.raises(RuntimeError, match="invalid boundary"):
        dr.convolve(x, [1.0], boundary='nope')
    # a kernel larger than the array is only allowed for the 'zero' boundary
    with pytest.raises(RuntimeError, match="cannot be larger than the array"):
        dr.convolve(t(1, 2, 3), [1.0, 1.0, 1.0, 1.0, 1.0], boundary='wrap')
    # resample shape must match the input rank
    with pytest.raises(RuntimeError, match="same number of axes"):
        dr.resample(x, (4, 4))
