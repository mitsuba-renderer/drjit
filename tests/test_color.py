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
