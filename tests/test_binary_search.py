import drjit as dr
import numpy as np
import pytest


def _ref(start, end, values, thresholds):
    """
    Reference search using NumPy's ``searchsorted`` (binary search).
    Clamps the searchsorted result to the per-lane ``[start, end]`` range.
    """
    idx = np.searchsorted(values, thresholds, side='left')
    return np.clip(idx, start, end)

# Fixed, strictly increasing haystack shared by most tests.
_VALUES = [1, 2, 3, 4, 5, 6, 7, 8]

@pytest.mark.parametrize('case', ["scalar_bounds", "jit_bounds", "jit_end", "jit_start"])
@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test01_various_bound_types(t, case):
    Float = dr.float_array_t(t)

    cases = {
        "scalar_bounds": (0,             len(_VALUES) - 1),
        "jit_bounds":    (t(0, 0, 2, 0), t(7, 5, 7, 7)),
        "jit_end":       (0,             t(7, 7, 3, 7)),
        "jit_start":     (t(0, 1, 2, 3), len(_VALUES) - 1)
    }
    start, end = cases[case]

    data = Float(_VALUES)
    threshold = Float(0.5, 3.5, 6.5, 100.0)

    idx = dr.binary_search(
        start, end,
        lambda i: dr.gather(Float, data, i) < threshold)

    assert dr.all(idx == _ref(start, end, _VALUES, threshold))

@pytest.test_arrays('uint32,is_jit,shape=(*)')
def test02_edge_cases(t):
    # Predicate all-True -> returns 'end'; all-False -> returns 'start';
    # empty range (start == end or start > end) -> returns start;
    # single-element range (end == start + 1);
    # mixed per-lane bounds where some lanes are empty and some are not.
    Float = dr.float_array_t(t)
    data = Float(_VALUES)

    # Predicate always True (every value < 100) -> should return 'end'.
    idx_all_true = dr.binary_search(
        t(0, 0), t(7, 7),
        lambda i: dr.gather(Float, data, i) < Float(100.0))
    assert dr.all(idx_all_true == t(7, 7))

    # Predicate always False (no value < 0) -> should return 'start'.
    idx_all_false = dr.binary_search(
        t(0, 2), t(7, 7),
        lambda i: dr.gather(Float, data, i) < Float(0.0))
    assert dr.all(idx_all_false == t(0, 2))

    # Empty range: start == end -> returns that index unchanged.
    idx_empty = dr.binary_search(
        t(3, 5), t(3, 5),
        lambda i: dr.gather(Float, data, i) < Float(100.0))
    assert dr.all(idx_empty == t(3, 5))

    # Inverted range: start > end -> returns start.
    idx_inverted = dr.binary_search(
        t(10, 5), t(5, 2),
        lambda i: dr.gather(Float, data, i) < Float(100.0))
    assert dr.all(idx_inverted == t(10, 5))

    # Single-element range: end == start + 1.
    # True predicate -> returns end
    idx_single_t = dr.binary_search(
        t(3), t(4),
        lambda i: i < t(4))
    assert dr.all(idx_single_t == t(4))

    # False predicate -> returns start
    idx_single_f = dr.binary_search(
        t(3), t(4),
        lambda i: i > t(100))
    assert dr.all(idx_single_f == t(3))

    # Mixed per-lane: some lanes empty, some non-empty
    start_mix = t(0, 10, 5)
    end_mix = t(9, 10, 5)
    idx_mix = dr.binary_search(start_mix, end_mix, lambda i: t(1) > t(0))
    assert dr.all(idx_mix == t(9, 10, 5))

    array_t = dr.replace_shape_t(t, (3, -1), "array")
    with pytest.raises(ValueError, match="depth <= 1"):
        dr.binary_search(array_t([0], [0], [0]), array_t([10], [10], [10]), lambda i: i < 5)


def test03_pure_scalar_int():
    """Verify binary_search with pure Python int bounds and Python bool predicate."""
    assert dr.binary_search(0, 100, lambda i: i < 42) == 42
    assert dr.binary_search(0, 100, lambda i: i < -10) == 0
    assert dr.binary_search(0, 100, lambda i: i < 200) == 100
    assert dr.binary_search(5, 5, lambda i: i < 10) == 5
    assert dr.binary_search(10, 5, lambda i: i < 10) == 10
    assert dr.binary_search(5, 6, lambda i: i < 6) == 6
    assert dr.binary_search(5, 6, lambda i: i > 100) == 5
