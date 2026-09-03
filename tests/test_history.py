import drjit as dr
import pytest

@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test01_basic(t):
    for i in range(4):
        dr.eval(dr.arange(t, i + 4))

    # Kernel history should be disabled by default
    assert not dr.flag(dr.JitFlag.KernelHistory)

    with dr.kernel_history() as kh:
        assert dr.flag(dr.JitFlag.KernelHistory)
        for i in range(4):
            dr.eval(dr.arange(t, i + 4))

    assert not dr.flag(dr.JitFlag.KernelHistory)
    assert len(kh) == 4

    for i, k in enumerate(kh):
        assert k.type == dr.KernelType.JIT
        assert k.size == i + 4
        assert k.output_count == 1
        assert len(k.hash) == 32
        assert k.recording_mode == dr.KernelRecordingMode.Inactive
        assert k.execution_time >= 0
        assert k.codegen_time >= 0
        assert isinstance(k.cache_hit, bool)
        assert isinstance(k.cache_disk, bool)

    # All four kernels are re-launches of the same program
    assert kh[1].hash == kh[0].hash
    assert kh[1].cache_hit


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test02_source_and_defaults(t):
    with dr.kernel_history() as kh:
        x = dr.opaque(t, 2, 32)
        dr.eval(dr.sqrt(x))
        dr.eval(dr.sum(dr.arange(t, 1027)))

    jit = [k for k in kh if k.type == dr.KernelType.JIT]
    other = [k for k in kh if k.type != dr.KernelType.JIT]
    assert len(jit) >= 1 and len(other) >= 1

    # Lazily fetched source code of a JIT kernel mentions its hash
    assert jit[0].hash[:16] in jit[0].source

    # Neutral defaults for fields that only JIT kernels provide
    k = other[0]
    assert k.hash is None and k.source is None
    assert k.operation_count == 0 and k.codegen_time == 0
    assert not k.cache_hit and not k.uses_optix

    # The renamed attribute remains accessible under a deprecation warning
    with pytest.warns(DeprecationWarning):
        assert jit[0].ir == jit[0].source


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test03_source_eviction(t):
    with dr.kernel_history() as kh:
        dr.eval(dr.opaque(t, 3, 16) * dr.opaque(t, 5, 16))

    # Flushing the kernel cache makes the lazily fetched source unavailable
    dr.flush_kernel_cache()
    assert kh[0].source is None
    assert kh[0].execution_time >= 0


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test04_nesting_and_live_access(t):
    with dr.kernel_history() as outer:
        dr.eval(dr.arange(t, 10))
        with dr.kernel_history() as inner:
            dr.eval(dr.arange(t, 11))
            # Entries recorded so far are visible inside the region
            assert len(inner) == 1 and len(outer) == 2
        dr.eval(dr.arange(t, 12))

    assert len(outer) == 3 and len(inner) == 1
    assert inner[0].size == 11
    assert [k.size for k in outer] == [10, 11, 12]


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test05_repr(t):
    with dr.kernel_history() as kh:
        dr.eval(dr.arange(t, 123))

    s = repr(kh)
    assert 'Kernel history (1 entry' in s and '123' in s

    with dr.kernel_history() as empty:
        pass
    assert repr(empty) == 'Kernel history (0 entries)'


@pytest.test_arrays('float32,shape=(*),jit,-diff')
def test06_legacy_interface(t):
    # The deprecated interface: manual flag management, a call that returns
    # the accumulated history, and dictionary-style entry access
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory):
        for i in range(4):
            dr.eval(dr.arange(t, i + 4))

        hist = dr.kernel_history((dr.KernelType.JIT,))
        with pytest.warns(DeprecationWarning):
            assert len(hist) == 4
            for i in range(4):
                assert hist[i]['size'] == i + 4
            assert hist[0]['hash'][:16] in hist[0]['ir'].getvalue()

    # Querying the history also clears it
    with pytest.warns(DeprecationWarning):
        assert len(dr.kernel_history()) == 0

    dr.kernel_history_clear()
