import drjit as dr
import pytest
import sys

@pytest.test_arrays("is_jit, float32, shape=(*)")
def test01_event_basic(t):
    """Test basic event creation, recording, and synchronization"""
    mod = sys.modules[t.__module__]

    # Create and record an event
    event = mod.Event(enable_timing=True)
    event.record()

    # Create some work
    x = dr.arange(t, 1000000)
    y = dr.sqrt(x)
    dr.eval(y)

    # Wait and verify completion
    event.wait()
    assert event.query() == True


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test02_event_timing(t):
    """Test event timing functionality"""
    mod = sys.modules[t.__module__]

    # Create two events for timing
    start = mod.Event(enable_timing=True)
    end = mod.Event(enable_timing=True)

    # Time some computation
    start.record()
    x = dr.arange(t, 10000000)
    y = dr.sin(dr.sqrt(x))
    dr.eval(y)
    end.record()

    # Get elapsed time
    end.wait()
    elapsed = start.elapsed_time(end)
    assert elapsed >= 0


@pytest.test_arrays("is_jit, float32, shape=(*)")
def test03_event_timing_disabled(t):
    """Test that elapsed_time fails when timing is disabled"""
    mod = sys.modules[t.__module__]

    # Create events without timing
    start = mod.Event(enable_timing=False)
    end = mod.Event(enable_timing=False)

    start.record()
    end.record()

    # Should raise an error when trying to get elapsed time
    with pytest.raises(RuntimeError, match="timing"):
        start.elapsed_time(end)
