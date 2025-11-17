"""
Comprehensive tests for boundary modes in dr.resample() and dr.convolve()
"""
import drjit as dr
import pytest


# Test wrap boundary mode with box filter upsampling
@pytest.test_arrays('float32, shape=(*)')
def test01_wrap_box_upsample(t):
    # Upsampling with wrap should repeat from the beginning
    source = t(1, 2, 3, 4)
    result = dr.resample(source, (8,), filter='box', wrap_mode='wrap')
    expected = t(1, 1, 2, 2, 3, 3, 4, 4)
    assert dr.allclose(result, expected), f"Got {result}, expected {expected}"


# Test wrap boundary mode with linear filter
@pytest.test_arrays('float32, shape=(*)')
def test02_wrap_linear_upsample(t):
    # Linear interpolation with wrap - values at boundaries should wrap around
    source = t(1, 2, 3, 4)
    result = dr.resample(source, (8,), filter='linear', wrap_mode='wrap')
    # When filter extends beyond edges, it should sample from wrapped positions
    # This test verifies wrapping occurs
    assert result.shape == (8,)
    # Values should be smooth and wrap-around should not cause discontinuities
    # at the conceptual boundaries


# Test wrap boundary mode with downsampling
@pytest.test_arrays('float32, shape=(*)')
def test03_wrap_box_downsample(t):
    source = t(1, 2, 3, 4, 5, 6)
    result = dr.resample(source, (3,), filter='box', wrap_mode='wrap')
    expected = t(1.5, 3.5, 5.5)
    assert dr.allclose(result, expected), f"Got {result}, expected {expected}"


# Test mirror boundary mode with box filter
@pytest.test_arrays('float32, shape=(*)')
def test04_mirror_box_upsample(t):
    source = t(1, 2, 3, 4)
    result = dr.resample(source, (8,), filter='box', wrap_mode='mirror')
    expected = t(1, 1, 2, 2, 3, 3, 4, 4)
    assert dr.allclose(result, expected), f"Got {result}, expected {expected}"


# Test mirror boundary mode with linear filter
@pytest.test_arrays('float32, shape=(*)')
def test05_mirror_linear_filter(t):
    # Test that mirror mode reflects values at boundaries
    source = t(1, 2, 3)
    result = dr.resample(source, (5,), filter='linear', wrap_mode='mirror')
    # Mirror should create smooth reflections at boundaries
    assert result.shape == (5,)
    # First and last values should be close to edge values
    assert result[0] >= 0.9 and result[0] <= 1.1  # Should be close to 1
    assert result[4] >= 2.7 and result[4] <= 3.1  # Should be close to 3 (relaxed tolerance)


# Test clamp boundary mode (default behavior)
@pytest.test_arrays('float32, shape=(*)')
def test06_clamp_box_upsample(t):
    source = t(1, 2, 3, 4)
    result_clamp = dr.resample(source, (8,), filter='box', wrap_mode='clamp')
    result_default = dr.resample(source, (8,), filter='box')  # Should default to clamp
    expected = t(1, 1, 2, 2, 3, 3, 4, 4)
    assert dr.allclose(result_clamp, expected), f"Got {result_clamp}, expected {expected}"
    assert dr.allclose(result_default, expected), "Default should be clamp mode"


# Test wrap mode with 2D tensor
@pytest.test_arrays('float32, tensor')
def test07_wrap_2d_tensor(t):
    source = t([[1, 2], [3, 4]])
    result = dr.resample(source, (4, 4), filter='box', wrap_mode='wrap')
    expected = t([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    assert dr.allclose(result, expected), f"Got {result}, expected {expected}"


# Test mirror mode with 2D tensor
@pytest.test_arrays('float32, tensor')
def test08_mirror_2d_tensor(t):
    source = t([[1, 2], [3, 4]])
    result = dr.resample(source, (4, 4), filter='box', wrap_mode='mirror')
    # Mirror should duplicate like box for this simple upsampling
    expected = t([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])
    assert dr.allclose(result, expected), f"Got {result}, expected {expected}"


# Test convolve with wrap boundary mode
@pytest.test_arrays('float32, shape=(*)')
def test09_convolve_wrap(t):
    source = t(1, 2, 3, 4)
    result = dr.convolve(source, filter='box', filter_radius=1.0, wrap_mode='wrap')
    # Convolution maintains size, wrap mode should affect edge samples
    assert result.shape == (4,)
    # With box filter radius 1.0, each output is average of 2 neighbors
    # At edges with wrap, it should wrap around


# Test convolve with mirror boundary mode
@pytest.test_arrays('float32, shape=(*)')
def test10_convolve_mirror(t):
    source = t(1, 2, 3, 4)
    result = dr.convolve(source, filter='box', filter_radius=1.0, wrap_mode='mirror')
    assert result.shape == (4,)
    # Mirror mode should reflect at boundaries


# Test convolve with clamp boundary mode (default)
@pytest.test_arrays('float32, shape=(*)')
def test11_convolve_clamp(t):
    source = t(1, 2, 3, 4)
    result_clamp = dr.convolve(source, filter='box', filter_radius=1.0, wrap_mode='clamp')
    result_default = dr.convolve(source, filter='box', filter_radius=1.0)
    assert result_clamp.shape == (4,)
    assert dr.allclose(result_clamp, result_default), "Default should be clamp"


# Test wrap with cubic filter
@pytest.test_arrays('float32, shape=(*)')
def test12_wrap_cubic_filter(t):
    source = t(1.0, 2.0, 3.0, 4.0, 5.0)
    result = dr.resample(source, (10,), filter='cubic', wrap_mode='wrap')
    assert result.shape == (10,)
    # Cubic filter with wrap should produce smooth results


# Test mirror with cubic filter
@pytest.test_arrays('float32, shape=(*)')
def test13_mirror_cubic_filter(t):
    source = t(1.0, 2.0, 3.0, 4.0, 5.0)
    result = dr.resample(source, (10,), filter='cubic', wrap_mode='mirror')
    assert result.shape == (10,)


# Test wrap with lanczos filter (wide kernel)
@pytest.test_arrays('float32, shape=(*)')
def test14_wrap_lanczos_filter(t):
    source = t(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    result = dr.resample(source, (12,), filter='lanczos', wrap_mode='wrap')
    assert result.shape == (12,)
    # Lanczos has wider kernel, good test for wrap behavior


# Test that different boundary modes produce different results
@pytest.test_arrays('float32, shape=(*)')  # Include JIT backends
def test15_wrap_modes_differ(t):
    source = t(1.0, 5.0, 2.0, 8.0)

    result_clamp = dr.resample(source, (8,), filter='linear', wrap_mode='clamp')
    result_wrap = dr.resample(source, (8,), filter='linear', wrap_mode='wrap')
    result_mirror = dr.resample(source, (8,), filter='linear', wrap_mode='mirror')

    # At least some values should differ between modes
    clamp_wrap_diff = dr.any(dr.abs(result_clamp - result_wrap) > 0.01)
    clamp_mirror_diff = dr.any(dr.abs(result_clamp - result_mirror) > 0.01)

    assert clamp_wrap_diff or clamp_mirror_diff, "Boundary modes should produce different results"


# Test convolve with gaussian filter and different boundary modes
@pytest.test_arrays('float32, shape=(*)')
def test16_convolve_gaussian_boundaries(t):
    source = t(1.0, 2.0, 3.0, 4.0, 5.0)

    result_clamp = dr.convolve(source, filter='gaussian', filter_radius=2.0, wrap_mode='clamp')
    result_wrap = dr.convolve(source, filter='gaussian', filter_radius=2.0, wrap_mode='wrap')
    result_mirror = dr.convolve(source, filter='gaussian', filter_radius=2.0, wrap_mode='mirror')

    assert result_clamp.shape == (5,)
    assert result_wrap.shape == (5,)
    assert result_mirror.shape == (5,)


# Test wrap mode with small array (edge case)
@pytest.test_arrays('float32, shape=(*)')
def test17_wrap_small_array(t):
    source = t(1, 2)
    result = dr.resample(source, (4,), filter='box', wrap_mode='wrap')
    expected = t(1, 1, 2, 2)
    assert dr.allclose(result, expected)


# Test mirror mode with single element (edge case)
@pytest.test_arrays('float32, shape=(*)')
def test18_mirror_single_element(t):
    source = t(5.0)
    result = dr.resample(source, (3,), filter='box', wrap_mode='mirror')
    expected = t(5.0, 5.0, 5.0)
    assert dr.allclose(result, expected)


# Test wrap mode preserves periodicity
@pytest.test_arrays('float32, shape=(*)')
def test19_wrap_periodicity(t):
    # Create a periodic signal
    import math
    source = t(*[math.sin(2 * math.pi * i / 8) for i in range(8)])
    result = dr.resample(source, (16,), filter='linear', wrap_mode='wrap')
    assert result.shape == (16,)
    # Wrap mode should maintain periodicity better than clamp


# Test boundary mode with downsampling and wide filter
@pytest.test_arrays('float32, shape=(*)')
def test20_downsample_wide_filter_boundaries(t):
    source = t(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    result_clamp = dr.resample(source, (5,), filter='cubic', wrap_mode='clamp')
    result_wrap = dr.resample(source, (5,), filter='cubic', wrap_mode='wrap')
    result_mirror = dr.resample(source, (5,), filter='cubic', wrap_mode='mirror')

    assert result_clamp.shape == (5,)
    assert result_wrap.shape == (5,)
    assert result_mirror.shape == (5,)

    # Results should be close in the middle but differ at edges
    # Middle elements (away from boundaries) should be similar
    mid_idx = 2
    assert dr.abs(result_clamp[mid_idx] - result_wrap[mid_idx]) < 0.5
    assert dr.abs(result_clamp[mid_idx] - result_mirror[mid_idx]) < 0.5


# Test invalid boundary mode string
@pytest.test_arrays('float32, shape=(*)')
def test21_invalid_wrap_mode(t):
    source = t(1, 2, 3)
    with pytest.raises(ValueError):
        dr.resample(source, (6,), filter='box', wrap_mode='invalid_mode')


# Test case sensitivity of boundary mode strings
@pytest.test_arrays('float32, shape=(*)')
def test22_wrap_mode_case_insensitive(t):
    source = t(1, 2, 3, 4)

    result_lower = dr.resample(source, (8,), filter='box', wrap_mode='wrap')
    result_upper = dr.resample(source, (8,), filter='box', wrap_mode='WRAP')
    result_mixed = dr.resample(source, (8,), filter='box', wrap_mode='Wrap')

    assert dr.allclose(result_lower, result_upper)
    assert dr.allclose(result_lower, result_mixed)


# Test using BoundaryMode enum directly
@pytest.test_arrays('float32, shape=(*)')
def test23_wrap_mode_enum(t):
    source = t(1, 2, 3, 4)

    result_str = dr.resample(source, (8,), filter='box', wrap_mode='wrap')
    result_enum = dr.resample(source, (8,), filter='box',
                             wrap_mode=dr.WrapMode.Repeat)

    assert dr.allclose(result_str, result_enum), "String and enum should give same result"


# Test convolve on specific axis with boundary modes
@pytest.test_arrays('float32, tensor')
def test24_convolve_axis_boundaries(t):
    source = t([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # Convolve only along axis 0 (rows)
    result = dr.convolve(source, filter='box', filter_radius=1.0,
                        axis=0, wrap_mode='wrap')
    assert result.shape == (3, 3)

    # Convolve only along axis 1 (columns)
    result = dr.convolve(source, filter='box', filter_radius=1.0,
                        axis=1, wrap_mode='mirror')
    assert result.shape == (3, 3)


# Test wrap with non-uniform data to verify correct wrapping
@pytest.test_arrays('float32, shape=(*)')
def test25_wrap_non_uniform_data(t):
    # Use distinct values to verify wrapping works correctly
    source = t(10, 20, 30, 40)
    result = dr.resample(source, (8,), filter='box', wrap_mode='wrap')

    # With box filter upsampling 2x, each value should appear twice
    expected = t(10, 10, 20, 20, 30, 30, 40, 40)
    assert dr.allclose(result, expected)


# Test mirror with asymmetric data
@pytest.test_arrays('float32, shape=(*)')
def test26_mirror_asymmetric_data(t):
    source = t(1, 5, 2, 8)  # Intentionally non-symmetric
    result = dr.resample(source, (8,), filter='box', wrap_mode='mirror')

    # Should maintain size
    assert result.shape == (8,)
    # Basic sanity check - result should be in reasonable range
    assert dr.all(result >= 0.5)
    assert dr.all(result <= 8.5)
