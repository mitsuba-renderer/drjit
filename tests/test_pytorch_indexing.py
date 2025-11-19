"""
Test PyTorch-compatible tensor indexing behavior.

This test suite ensures that Dr.Jit tensor indexing is strictly compatible
with PyTorch, particularly for the critical requirement that integer indexing
returns 0-D tensors (not Python scalars).
"""

import drjit as dr
import pytest
import sys

# Optional PyTorch dependency
try:
    import torch
    import numpy as np

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def skip_if_no_torch():
    """Skip test if PyTorch is not available."""
    if not HAS_TORCH:
        pytest.skip("PyTorch not available")


# Helper functions for conversion
def drjit_to_torch(drjit_tensor):
    """Convert Dr.Jit tensor to PyTorch tensor."""
    if HAS_TORCH:
        np_array = (
            drjit_tensor.numpy()
            if hasattr(drjit_tensor, "numpy")
            else np.array(drjit_tensor)
        )
        return torch.from_numpy(np_array).float()
    return None


def assert_shape_equal(dr_tensor, pt_tensor, msg=""):
    """Assert that Dr.Jit and PyTorch tensors have equal shapes."""
    dr_shape = dr_tensor.shape
    pt_shape = tuple(pt_tensor.shape)
    assert (
        dr_shape == pt_shape
    ), f"{msg}\nDr.Jit shape: {dr_shape}, PyTorch shape: {pt_shape}"


def assert_values_equal(dr_tensor, pt_tensor, rtol=1e-5, atol=1e-7, msg=""):
    """Assert that Dr.Jit and PyTorch tensors have equal values."""
    if HAS_TORCH:
        dr_np = (
            dr_tensor.numpy() if hasattr(dr_tensor, "numpy") else np.array(dr_tensor)
        )
        pt_np = pt_tensor.detach().cpu().numpy()
        np.testing.assert_allclose(dr_np, pt_np, rtol=rtol, atol=atol, err_msg=msg)


# =============================================================================
# Basic Integer Indexing Tests
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test01_single_int_index_1d_returns_0d(t):
    """Test that single integer index on 1D tensor returns 0-D tensor."""
    skip_if_no_torch()

    # Create test data
    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    # Test positive index
    dr_result = dr_tensor[5]
    pt_result = pt_tensor[5]

    # CRITICAL: Should return 0-D tensor, not scalar
    assert dr_result.ndim == 0, f"Expected ndim=0, got {dr_result.ndim}"
    assert dr_result.shape == (), f"Expected shape=(), got {dr_result.shape}"
    assert pt_result.ndim == 0, "PyTorch should also return 0-D tensor"

    assert_shape_equal(dr_result, pt_result, "Single int index shape mismatch")
    assert_values_equal(dr_result, pt_result, msg="Single int index value mismatch")


@pytest.test_arrays("is_tensor, float32, is_jit")
def test02_negative_int_index_1d_returns_0d(t):
    """Test that negative integer index on 1D tensor returns 0-D tensor."""
    skip_if_no_torch()

    data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    # Test negative indices
    for idx in [-1, -5, -10]:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert (
            dr_result.ndim == 0
        ), f"Index {idx}: Expected ndim=0, got {dr_result.ndim}"
        assert (
            dr_result.shape == ()
        ), f"Index {idx}: Expected shape=(), got {dr_result.shape}"
        assert_shape_equal(dr_result, pt_result, f"Negative index {idx} shape mismatch")
        assert_values_equal(
            dr_result, pt_result, msg=f"Negative index {idx} value mismatch"
        )


@pytest.test_arrays("is_tensor, float32, is_jit")
def test03_multi_int_index_2d_returns_0d(t):
    """Test that multiple integer indices on 2D tensor return 0-D tensor."""
    skip_if_no_torch()

    data = list(range(20))
    dr_tensor = t(data, shape=(4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(4, 5)

    # Test various index combinations
    test_cases = [(0, 0), (2, 3), (3, 4), (-1, -1), (-2, 3)]

    for i, j in test_cases:
        dr_result = dr_tensor[i, j]
        pt_result = pt_tensor[i, j]

        assert (
            dr_result.ndim == 0
        ), f"Index ({i}, {j}): Expected ndim=0, got {dr_result.ndim}"
        assert (
            dr_result.shape == ()
        ), f"Index ({i}, {j}): Expected shape=(), got {dr_result.shape}"
        assert_shape_equal(dr_result, pt_result, f"Index ({i}, {j}) shape mismatch")
        assert_values_equal(
            dr_result, pt_result, msg=f"Index ({i}, {j}) value mismatch"
        )


@pytest.test_arrays("is_tensor, float32, is_jit")
def test04_single_int_index_2d_reduces_dim(t):
    """Test that single integer index on 2D tensor reduces dimension."""
    skip_if_no_torch()

    data = list(range(20))
    dr_tensor = t(data, shape=(4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(4, 5)

    # Single index should return 1D tensor
    for idx in [0, 2, -1]:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert (
            dr_result.ndim == 1
        ), f"Index {idx}: Expected ndim=1, got {dr_result.ndim}"
        assert dr_result.shape == (
            5,
        ), f"Index {idx}: Expected shape=(5,), got {dr_result.shape}"
        assert_shape_equal(dr_result, pt_result, f"Single index {idx} shape mismatch")
        assert_values_equal(
            dr_result, pt_result, msg=f"Single index {idx} value mismatch"
        )


@pytest.test_arrays("is_tensor, float32, is_jit")
def test05_multi_int_index_3d_returns_0d(t):
    """Test that full indexing on 3D tensor returns 0-D tensor."""
    skip_if_no_torch()

    data = list(range(60))
    dr_tensor = t(data, shape=(3, 4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(3, 4, 5)

    test_cases = [(0, 0, 0), (1, 2, 3), (2, 3, 4), (-1, -1, -1)]

    for i, j, k in test_cases:
        dr_result = dr_tensor[i, j, k]
        pt_result = pt_tensor[i, j, k]

        assert (
            dr_result.ndim == 0
        ), f"Index ({i}, {j}, {k}): Expected ndim=0, got {dr_result.ndim}"
        assert (
            dr_result.shape == ()
        ), f"Index ({i}, {j}, {k}): Expected shape=(), got {dr_result.shape}"
        assert_shape_equal(
            dr_result, pt_result, f"Index ({i}, {j}, {k}) shape mismatch"
        )
        assert_values_equal(
            dr_result, pt_result, msg=f"Index ({i}, {j}, {k}) value mismatch"
        )


# =============================================================================
# Slicing Tests
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test06_slice_1d(t):
    """Test basic slicing on 1D tensor."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    test_slices = [
        slice(2, 7),  # [2:7]
        slice(None, 5),  # [:5]
        slice(3, None),  # [3:]
        slice(None, None, 2),  # [::2]
        slice(8, 2, -1),  # [8:2:-1]
        slice(None, None, -1),  # [::-1]
    ]

    for s in test_slices:
        dr_result = dr_tensor[s]
        pt_result = pt_tensor[s]

        assert_shape_equal(dr_result, pt_result, f"Slice {s} shape mismatch")
        assert_values_equal(dr_result, pt_result, msg=f"Slice {s} value mismatch")


@pytest.test_arrays("is_tensor, float32, is_jit")
def test07_slice_2d(t):
    """Test slicing on 2D tensor."""
    skip_if_no_torch()

    data = list(range(20))
    dr_tensor = t(data, shape=(4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(4, 5)

    test_cases = [
        (slice(1, 3), slice(None)),  # [1:3, :]
        (slice(None), slice(2, 4)),  # [:, 2:4]
        (slice(1, 3), slice(2, 4)),  # [1:3, 2:4]
        (slice(None, None, -1), slice(None)),  # [::-1, :]
    ]

    for idx in test_cases:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert_shape_equal(dr_result, pt_result, f"Slice {idx} shape mismatch")
        assert_values_equal(dr_result, pt_result, msg=f"Slice {idx} value mismatch")


@pytest.test_arrays("is_tensor, float32, is_jit")
def test08_mixed_int_slice(t):
    """Test mixing integer and slice indices."""
    skip_if_no_torch()

    data = list(range(20))
    dr_tensor = t(data, shape=(4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(4, 5)

    test_cases = [
        (0, slice(None)),  # [0, :]
        (slice(None), 0),  # [:, 0]
        (2, slice(1, 4)),  # [2, 1:4]
        (slice(1, 3), 2),  # [1:3, 2]
    ]

    for idx in test_cases:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert_shape_equal(dr_result, pt_result, f"Mixed index {idx} shape mismatch")
        assert_values_equal(
            dr_result, pt_result, msg=f"Mixed index {idx} value mismatch"
        )


# =============================================================================
# Ellipsis Tests
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test09_ellipsis(t):
    """Test ellipsis (...) indexing."""
    skip_if_no_torch()

    data = list(range(60))
    dr_tensor = t(data, shape=(3, 4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(3, 4, 5)

    test_cases = [
        (Ellipsis,),  # [...]
        (Ellipsis, 0),  # [..., 0]
        (0, Ellipsis),  # [0, ...]
        (1, Ellipsis, 2),  # [1, ..., 2]
    ]

    for idx in test_cases:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert_shape_equal(dr_result, pt_result, f"Ellipsis {idx} shape mismatch")
        assert_values_equal(dr_result, pt_result, msg=f"Ellipsis {idx} value mismatch")


# =============================================================================
# None/newaxis Tests
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test10_newaxis(t):
    """Test None/newaxis indexing."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    test_cases = [
        (None, slice(None)),  # [None, :]
        (slice(None), None),  # [:, None]
        (None, slice(None), None),  # [None, :, None]
    ]

    for idx in test_cases:
        dr_result = dr_tensor[idx]
        pt_result = pt_tensor[idx]

        assert_shape_equal(dr_result, pt_result, f"Newaxis {idx} shape mismatch")
        assert_values_equal(dr_result, pt_result, msg=f"Newaxis {idx} value mismatch")


# =============================================================================
# Edge Cases
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test11_empty_slice(t):
    """Test slicing that produces empty tensor."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    dr_result = dr_tensor[5:5]
    pt_result = pt_tensor[5:5]

    assert_shape_equal(dr_result, pt_result, "Empty slice shape mismatch")
    assert dr_result.shape == (0,), f"Expected shape=(0,), got {dr_result.shape}"


@pytest.test_arrays("is_tensor, float32, is_jit")
def test12_single_element_tensor(t):
    """Test indexing single element tensor."""
    skip_if_no_torch()

    dr_tensor = t([42.0])
    pt_tensor = torch.tensor([42.0], dtype=torch.float32)

    dr_result = dr_tensor[0]
    pt_result = pt_tensor[0]

    assert dr_result.ndim == 0, f"Expected ndim=0, got {dr_result.ndim}"
    assert dr_result.shape == (), f"Expected shape=(), got {dr_result.shape}"
    assert_shape_equal(dr_result, pt_result, "Single element shape mismatch")
    assert_values_equal(dr_result, pt_result, msg="Single element value mismatch")


# =============================================================================
# Array Indexing Tests (using Dr.Jit integer arrays)
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test13_array_index_1d(t):
    """Test array indexing on 1D tensor."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    # Create index array
    indices = [0, 2, 4, 6, 8]
    index_type = dr.uint32_array_t(dr.array_t(t))
    dr_indices = index_type(indices)
    pt_indices = torch.tensor(indices, dtype=torch.long)

    dr_result = dr_tensor[dr_indices]
    pt_result = pt_tensor[pt_indices]

    assert_shape_equal(dr_result, pt_result, "Array index shape mismatch")
    assert_values_equal(dr_result, pt_result, msg="Array index value mismatch")


@pytest.test_arrays("is_tensor, float32, is_jit")
def test14_array_index_2d(t):
    """Test array indexing on 2D tensor (first dimension)."""
    skip_if_no_torch()

    data = list(range(20))
    dr_tensor = t(data, shape=(4, 5))
    pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(4, 5)

    # Create index array
    indices = [0, 2, 3]
    index_type = dr.uint32_array_t(dr.array_t(t))
    dr_indices = index_type(indices)
    pt_indices = torch.tensor(indices, dtype=torch.long)

    dr_result = dr_tensor[dr_indices]
    pt_result = pt_tensor[pt_indices]

    assert_shape_equal(dr_result, pt_result, "Array index 2D shape mismatch")
    assert dr_result.shape == (3, 5), f"Expected shape=(3, 5), got {dr_result.shape}"
    assert_values_equal(dr_result, pt_result, msg="Array index 2D value mismatch")


# =============================================================================
# Assignment Tests
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test15_setitem_single_element(t):
    """Test assigning to single element."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    dr_tensor[5] = 100.0
    pt_tensor[5] = 100.0

    assert_values_equal(dr_tensor, pt_tensor, msg="Single element assignment mismatch")


@pytest.test_arrays("is_tensor, float32, is_jit")
def test16_setitem_slice(t):
    """Test assigning to slice."""
    skip_if_no_torch()

    data = list(range(10))
    dr_tensor = t(data)
    pt_tensor = torch.tensor(data, dtype=torch.float32)

    dr_tensor[2:7] = 100.0
    pt_tensor[2:7] = 100.0

    assert_values_equal(dr_tensor, pt_tensor, msg="Slice assignment mismatch")


# =============================================================================
# Comprehensive Compatibility Test
# =============================================================================


@pytest.test_arrays("is_tensor, float32, is_jit")
def test17_comprehensive_indexing_compatibility(t):
    """Comprehensive test covering multiple indexing scenarios."""
    skip_if_no_torch()

    # Test with different tensor shapes
    test_configs = [
        (10,),  # 1D
        (4, 5),  # 2D
        (2, 3, 4),  # 3D
    ]

    for shape in test_configs:
        size = 1
        for dim in shape:
            size *= dim

        data = list(range(size))
        dr_tensor = t(data, shape=shape) if len(shape) > 1 else t(data)
        pt_tensor = torch.tensor(data, dtype=torch.float32).reshape(shape)

        # Test 1: Verify shapes match
        assert_shape_equal(dr_tensor, pt_tensor, f"Initial shape mismatch for {shape}")

        # Test 2: Full slice should preserve shape
        dr_result = dr_tensor[...]
        pt_result = pt_tensor[...]
        assert_shape_equal(dr_result, pt_result, f"Ellipsis shape mismatch for {shape}")

        # Test 3: Integer index on first dimension
        dr_result = dr_tensor[0]
        pt_result = pt_tensor[0]
        assert_shape_equal(
            dr_result, pt_result, f"First dim index shape mismatch for {shape}"
        )

        # Test 4: If multidimensional, test full integer indexing
        if len(shape) > 1:
            full_idx = tuple([0] * len(shape))
            dr_result = dr_tensor[full_idx]
            pt_result = pt_tensor[full_idx]
            assert dr_result.ndim == 0, f"Full index should return 0-D for {shape}"
            assert_shape_equal(
                dr_result, pt_result, f"Full index shape mismatch for {shape}"
            )

