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
