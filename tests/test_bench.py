import math
import os
import sys
import time

import drjit as dr
import drjit.bench
import pytest


# Workload that produces a single fused kernel touching `n` elements and
# returns a scalar so dr.eval has something concrete to schedule.
def _workload(t, n=1 << 14):
    x = dr.arange(t, n)
    y = dr.sin(x * 0.001) + dr.cos(x * 0.0007)
    return dr.sum(y)


@pytest.test_arrays('jit,shape=(*),float32,-diff')
def test01_repeat_records_and_call_count(t):
    calls = 0

    records: list = []

    @dr.bench.repeat(label='workload', runs=3, warmup=1,
                     records=records, measure_async=True, clear_cache=False)
    def f(n):
        nonlocal calls
        calls += 1
        return _workload(t, n)

    out = f(1 << 12)

    # warmup + sync runs + async runs
    assert calls == 1 + 3 + 3
    # Function value passes through unchanged.
    assert isinstance(out, t)

    assert len(records) == 1
    r = records[0]
    for key in ('label', 'suffix', 'runs',
                'total_sync_ms', 'total_sync_ms_std',
                'total_async_ms', 'total_async_ms_std',
                'jit_ms', 'jit_ms_std',
                'codegen_ms', 'codegen_ms_std',
                'backend_ms', 'backend_ms_std',
                'execution_ms', 'execution_ms_std'):
        assert key in r, f'missing key {key!r}'

    assert r['label'] == 'workload'
    assert r['suffix'] == ''
    assert r['runs'] == 3
    assert r['total_sync_ms'] > 0
    assert r['total_sync_ms_std'] >= 0
    assert r['total_async_ms'] >= 0
    assert math.isfinite(r['jit_ms'])
    # args=(n,) is non-empty so it must be attached.
    assert r['args'] == (1 << 12,)


@pytest.test_arrays('jit,shape=(*),float32,-diff')
def test02_repeat_no_async_call_count(t):
    calls = 0

    @dr.bench.repeat(label='workload', runs=2, warmup=0,
                     measure_async=False, clear_cache=False)
    def f():
        nonlocal calls
        calls += 1
        return _workload(t, 1 << 12)

    f()
    assert calls == 2


@pytest.test_arrays('jit,shape=(*),float32,-diff')
def test03_repeat_label_kwarg_suffix(t):
    records: list = []

    @dr.bench.repeat(label='workload', runs=1, records=records,
                     measure_async=False, clear_cache=False)
    def f(n):
        return _workload(t, n)

    f(1 << 12, label='small')

    assert records[0]['suffix'] == ' [small]'
    # The 'label' kwarg must be stripped before forwarding, so it must not
    # appear in the recorded kwargs.
    assert 'kwargs' not in records[0] or 'label' not in records[0]['kwargs']


@pytest.test_arrays('jit,shape=(*),float32,-diff')
def test04_measure_appends_record(t):
    records: list = []

    with dr.bench.measure(label='workload', records=records,
                          clear_cache=False):
        s = _workload(t, 1 << 12)
        dr.schedule(s)

    assert len(records) == 1
    r = records[0]
    assert r['label'] == 'workload'
    assert r['runs'] == 1
    assert r['total_ms'] > 0
    for key in ('jit_ms', 'codegen_ms', 'backend_ms', 'execution_ms'):
        assert key in r and math.isfinite(r[key])


@pytest.test_arrays('jit,shape=(*),float32,-diff')
def test05_reset_clears_history_no_disk(t):
    # Build a kernel and harvest history, then reset without touching disk.
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        s = _workload(t, 1 << 12)
        dr.eval(s)
        dr.sync_thread()
        history_before = dr.kernel_history([dr.KernelType.JIT])
        assert len(history_before) >= 1

        # Snapshot cache mtime before reset(clear_cache=False).
        cache_dir = os.path.join(os.path.expanduser('~'), '.drjit')
        before = os.path.getmtime(cache_dir) if os.path.isdir(cache_dir) else None

        dr.bench.reset(clear_cache=False)

        # No disk changes.
        if before is not None:
            assert os.path.getmtime(cache_dir) == before

        # History was cleared.
        assert dr.kernel_history([dr.KernelType.JIT]) == []
