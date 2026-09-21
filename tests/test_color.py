import drjit as dr
import pytest


@pytest.test_arrays('float32, shape=(*)')
def test01_srgb_conversion(t):
    """Spot-check the linear/sRGB conversion routines"""

    assert dr.allclose(dr.linear_to_srgb(0), 0)
    assert dr.allclose(dr.srgb_to_linear(0), 0)
    assert dr.allclose(dr.linear_to_srgb(1), 1)
    assert dr.allclose(dr.srgb_to_linear(1), 1)
    assert dr.allclose(dr.linear_to_srgb(.5), 0.7353569830524495)
    assert dr.allclose(dr.srgb_to_linear(.5), 0.21404114048223244)

    # Out of bounds
    assert dr.allclose(dr.linear_to_srgb(-1), 0)
    assert dr.allclose(dr.srgb_to_linear(-1), 0)
    assert dr.allclose(dr.linear_to_srgb(2), 1)
    assert dr.allclose(dr.srgb_to_linear(2), 1)

    assert dr.allclose(dr.linear_to_srgb(-1, clip=False), -1)
    assert dr.allclose(dr.srgb_to_linear(-1, clip=False), -1)
    assert dr.allclose(dr.linear_to_srgb(2, clip=False), 1.353256046149386)
    assert dr.allclose(dr.srgb_to_linear(2, clip=False), 4.95384575159204)

@pytest.test_arrays('float32, shape=(3, *)')
def test01_oklab_conversion(t):
    """Spot-check the Oklab/sRGB conversion routines"""

    EXAMPLES = [
        # (linear_srgb, oklab)
        # Primary colors
        ([1.0, 0.0, 0.0], [0.62795536, 0.22486305, 0.12584630]),  # Red
        ([0.0, 1.0, 0.0], [0.86643961, -0.23388754, 0.17949847]),  # Green
        ([0.0, 0.0, 1.0], [0.45201372, -0.03245699, -0.31152815]),  # Blue

        # Secondary colors
        ([1.0, 1.0, 0.0], [0.96798272, -0.07136906, 0.19856974]),  # Yellow
        ([0.0, 1.0, 1.0], [0.90539923, -0.14944391, -0.03939817]),  # Cyan
        ([1.0, 0.0, 1.0], [0.70167386, 0.27456628, -0.16915606]),  # Magenta

        # Grayscale (should have a=b≈0)
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),  # Black
        ([1.0, 1.0, 1.0], [0.99999999, 0.0, 0.0]),  # White (using 0 for near-zero)
        ([0.5, 0.5, 0.5], [0.79370052, 0.0, 0.0]),  # 50% Gray (linear)
        ([0.25, 0.25, 0.25], [0.62996052, 0.0, 0.0]),  # 25% Gray (linear)

        # Mixed colors
        ([0.5, 0.25, 0.75], [0.71168106, 0.08614060, -0.10325682]),
        ([0.75, 0.5, 0.25], [0.81430313, 0.02048597, 0.07594870]),
        ([0.3, 0.6, 0.1], [0.78130967, -0.10163317, 0.11875259]),
    ]

    for i, (src, dst) in enumerate(EXAMPLES):
        print(i)
        assert dr.allclose(dr.linear_srgb_to_oklab(t(src)), t(dst))
        assert dr.allclose(dr.oklab_to_linear_srgb(t(dst)), t(src))

@pytest.test_arrays('float32, shape=(3, *)')
def test03_hsv_hsl_conversion(t):
    """Compare the HSV/HSL conversion routines against Python's colorsys module"""
    import colorsys

    EXAMPLES = [
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0],
        [0.5, 0.25, 0.75], [0.75, 0.5, 0.25], [0.3, 0.6, 0.1],
        [0.9, 0.1, 0.2], [0.1, 0.9, 0.8], [0.2, 0.1, 0.9],
        [0.7, 0.7, 0.2], [0.05, 0.9, 0.05], [0.4, 0.2, 0.2],
    ]

    for rgb in EXAMPLES:
        hsv = colorsys.rgb_to_hsv(*rgb)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        hsl = (h, s, l)

        assert dr.allclose(dr.rgb_to_hsv(t(rgb)), t(hsv))
        assert dr.allclose(dr.hsv_to_rgb(t(hsv)), t(rgb))
        assert dr.allclose(dr.rgb_to_hsl(t(rgb)), t(hsl))
        assert dr.allclose(dr.hsl_to_rgb(t(hsl)), t(rgb))

    # Hue wraps around
    assert dr.allclose(dr.hsv_to_rgb(t([1.0, 1.0, 1.0])), t([1.0, 0.0, 0.0]))
    assert dr.allclose(dr.hsv_to_rgb(t([-1 / 3, 1.0, 1.0])), t([0.0, 0.0, 1.0]))
    assert dr.allclose(dr.hsl_to_rgb(t([4 / 3, 1.0, 0.5])), t([0.0, 1.0, 0.0]))


@pytest.test_arrays('float32, shape=(3, *), jit')
def test04_hsv_hsl_batched(t):
    """Round trip a batch of random colors through HSV/HSL, arrays and tensors"""
    Array4f = dr.replace_shape_t(t, (4, -1), 'array')
    Tensor = dr.tensor_t(t)
    n = 1000

    rng = dr.rng(seed=0)
    rgb = rng.random(t, (3, n))

    for fwd, inv in [(dr.rgb_to_hsv, dr.hsv_to_rgb),
                     (dr.rgb_to_hsl, dr.hsl_to_rgb)]:
        out = fwd(rgb)
        assert dr.all((out >= 0) & (out <= 1), axis=None)
        assert dr.allclose(inv(out), rgb, atol=1e-5)

        # Alpha channel passes through unchanged
        rgba = Array4f(rgb.x, rgb.y, rgb.z, rgb.x)
        out4 = fwd(rgba)
        assert dr.allclose(out4.xyz, out)
        assert dr.allclose(out4.w, rgb.x)

        # Tensor with trailing color dimension, in both RGB and RGBA layouts
        tensor = Tensor(rgb, flip_axes=True)
        assert tensor.shape == (n, 3)
        assert dr.allclose(t(fwd(tensor), flip_axes=True), out)
        tensor4 = Tensor(rgba, flip_axes=True)
        assert dr.allclose(Array4f(fwd(tensor4), flip_axes=True), out4)

    with pytest.raises(Exception, match='leading dimension 3 or 4'):
        dr.rgb_to_hsv(dr.replace_shape_t(t, (2, -1), 'array')(rgb.x, rgb.y))
    with pytest.raises(Exception, match='trailing dimension 3 or 4'):
        dr.rgb_to_hsl(Tensor(rgb))
