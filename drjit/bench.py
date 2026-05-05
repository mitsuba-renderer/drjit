"""
High-level benchmarking helpers for Dr.Jit.

This module provides three entry points:

- :py:func:`repeat`  — decorator that runs a function many times and records
  averaged timings (total time, JIT, codegen, backend compile, kernel
  execution) into an optional list of records.
- :py:func:`measure` — context manager for one-shot timing of a code block.
- :py:func:`reset`   — drop in-process JIT state (and optionally on-disk
  caches) so the next run starts from a known clean state.

Logging follows Dr.Jit's global :py:func:`drjit.log_level`: at ``Info`` only
the per-benchmark summary is printed, at ``Debug`` each iteration emits a
short progress line, and at ``Warn`` or below the helpers stay silent.
"""

from __future__ import annotations

import gc
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import drjit as dr


# ---------------------------------------------------------------------------
# Logging helpers (gated on dr.log_level())
# ---------------------------------------------------------------------------

def _log_at_least(level: 'dr.LogLevel') -> bool:
    # LogLevel is a nanobind enum; compare via its underlying value.
    return dr.log_level().value >= level.value


def _log_info(msg: str) -> None:
    if _log_at_least(dr.LogLevel.Info):
        print(msg)


def _log_debug(msg: str) -> None:
    if _log_at_least(dr.LogLevel.Debug):
        print(msg)


def _log_warn(msg: str) -> None:
    if _log_at_least(dr.LogLevel.Warn):
        print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / n
    return math.sqrt(max(0.0, var))


def _kernel_history_split() -> Tuple[float, float, float]:
    """Sum codegen / backend / execution times (in ms) from the JIT history."""
    history = dr.kernel_history([dr.KernelType.JIT])
    codegen = sum(k['codegen_time'] for k in history)
    backend = sum(k['backend_time'] for k in history)
    execution = sum(k['execution_time'] for k in history)
    return codegen, backend, execution


def _timed_run(
    func: Callable[..., Any],
    args: tuple,
    kwargs: dict,
    clear_cache: bool,
) -> Tuple[Any, float]:
    """Run ``func(*args, **kwargs)``, force evaluation, and return (value, ms)."""
    reset(clear_cache=clear_cache)
    t0 = time.perf_counter()
    ret = func(*args, **kwargs)
    dr.eval(ret)
    dr.sync_thread()
    return ret, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Cache wiping
# ---------------------------------------------------------------------------

def _purge_cache_dirs(clear_drjit: bool = True, clear_nvidia: bool = True) -> None:
    """
    Remove on-disk kernel cache folders. Destructive — internal use only.

    Cleans ``~/.drjit`` (Linux/macOS) or
    ``%LOCALAPPDATA%/Temp/drjit`` (Windows), and optionally the NVIDIA
    driver's compile cache at ``~/.nv``. Used by :py:func:`reset` to
    guarantee that the backend compile time is actually measured on the next
    run rather than served from disk.
    """
    if os.name != 'nt':
        def _wipe(path: str) -> None:
            if not os.path.exists(path):
                return
            shutil.rmtree(path, ignore_errors=True)
            os.makedirs(path, exist_ok=True)

        if clear_drjit:
            _wipe(os.path.join(os.path.expanduser('~'), '.drjit'))
        if clear_nvidia:
            _wipe(os.path.join(os.path.expanduser('~'), '.nv'))
    else:
        folder = os.path.join(os.path.expanduser('~'),
                              'AppData', 'Local', 'Temp', 'drjit')
        if not os.path.isdir(folder):
            return
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reset(clear_cache: bool = True) -> None:
    """
    Reset Dr.Jit's in-process state to a clean slate.

    Clears the kernel history, runs Python's garbage collector, flushes the
    JIT memory allocator, and resets allocator statistics. When
    ``clear_cache`` is ``True``, also wipes the on-disk kernel cache (so
    that the next run measures backend-compile time from scratch).
    """
    dr.kernel_history_clear()
    gc.collect()
    dr.flush_malloc_cache()
    dr.detail.malloc_clear_statistics()
    if clear_cache:
        _purge_cache_dirs()
        dr.flush_kernel_cache()


def repeat(
    label: str,
    runs: int = 4,
    warmup: int = 0,
    records: Optional[List[Dict[str, Any]]] = None,
    measure_async: bool = True,
    clear_cache: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator: benchmark a function over ``runs`` iterations.

    The wrapped call is executed once per iteration with
    :py:attr:`drjit.JitFlag.LaunchBlocking` enabled and
    :py:attr:`drjit.JitFlag.KernelHistory` enabled, so per-stage timings
    (codegen / backend / execution) can be harvested via
    :py:func:`drjit.kernel_history`. When ``measure_async`` is ``True``, an
    additional ``runs`` iterations are performed with launch-blocking
    disabled to measure the asynchronous-launch speed-up.

    Args:
        label: Name shown in log output and stored in each record.
        runs: Number of measured iterations.
        warmup: Number of un-measured iterations performed first.
        records: Optional list to which one summary :py:class:`dict` is
            appended per decorated call. The dict has the keys
            ``label``, ``suffix``, ``runs``, ``total_sync_ms``,
            ``total_sync_ms_std``, ``jit_ms``, ``jit_ms_std``,
            ``codegen_ms``, ``codegen_ms_std``, ``backend_ms``,
            ``backend_ms_std``, ``execution_ms``, ``execution_ms_std``.
            When ``measure_async=True``, also ``total_async_ms`` and
            ``total_async_ms_std``. Non-empty ``args`` / ``kwargs`` are
            attached too.
        measure_async: When ``True`` (default), a second batch of ``runs``
            iterations runs with asynchronous kernel launches and the
            difference is reported.
        clear_cache: Wipe the kernel cache between iterations (required to
            measure backend-compile time accurately).

    A decorated call may pass an extra ``label=`` keyword argument; it is
    appended as a suffix to the main label and not forwarded to the
    underlying function.

    Example:

    .. code-block:: python

       records = []

       @dr.bench.repeat(label='Render', runs=8, records=records)
       def render(spp):
           return mi.render(scene, spp=spp, seed=0)

       img = render(32, label='@32 spp')
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            suffix = ''
            if 'label' in kwargs:
                suffix = f" [{kwargs.pop('label')}]"

            # Warmup: results discarded.
            for _ in range(warmup):
                ret = func(*args, **kwargs)
                dr.eval(ret)
                dr.sync_thread()

            _log_info(f'Benchmarking: "{label}{suffix}" ...')

            with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
                # --- Synchronous batch: per-stage timings are accurate. ---
                sync_total: List[float] = []
                jit_t: List[float] = []
                codegen_t: List[float] = []
                backend_t: List[float] = []
                exec_t: List[float] = []

                with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
                    for i in range(runs):
                        ret, total_ms = _timed_run(
                            func, args, kwargs, clear_cache)
                        cg, be, ex = _kernel_history_split()
                        codegen_t.append(cg)
                        backend_t.append(be)
                        exec_t.append(ex)
                        jit_t.append(total_ms - (cg + be + ex))
                        sync_total.append(total_ms)
                        _log_debug(f'  run {i + 1}/{runs} (sync): '
                                   f'{total_ms:.2f} ms')

                # --- Optional asynchronous batch: total time only. ---
                async_total: List[float] = []
                if measure_async:
                    with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, False):
                        for i in range(runs):
                            ret, total_ms = _timed_run(
                                func, args, kwargs, clear_cache)
                            async_total.append(total_ms)
                            _log_debug(f'  run {i + 1}/{runs} (async): '
                                       f'{total_ms:.2f} ms')

            # --- Aggregate + log + record. ---
            sync_total_mean, sync_total_std = _mean(sync_total), _std(sync_total)
            jit_mean, jit_std = _mean(jit_t), _std(jit_t)
            cg_mean, cg_std = _mean(codegen_t), _std(codegen_t)
            be_mean, be_std = _mean(backend_t), _std(backend_t)
            ex_mean, ex_std = _mean(exec_t), _std(exec_t)

            _log_info(f'Results (averaged over {runs} runs):')
            if measure_async:
                a_mean, a_std = _mean(async_total), _std(async_total)
                _log_info(f'    - Total time (async): '
                          f'{a_mean:.2f} ms (± {a_std:.2f})')
                _log_info(f'    - Total time (sync):  '
                          f'{sync_total_mean:.2f} ms '
                          f'(± {sync_total_std:.2f}) -> '
                          f'(async gain: {sync_total_mean - a_mean:.2f} ms)')
            else:
                _log_info(f'    - Total time:         '
                          f'{sync_total_mean:.2f} ms '
                          f'(± {sync_total_std:.2f})')
            _log_info(f'    - Jitting time:       '
                      f'{jit_mean:.2f} ms (± {jit_std:.2f})')
            _log_info(f'    - Codegen time:       '
                      f'{cg_mean:.2f} ms (± {cg_std:.2f})')
            _log_info(f'    - Backend time:       '
                      f'{be_mean:.2f} ms (± {be_std:.2f})')
            _log_info(f'    - Execution time:     '
                      f'{ex_mean:.2f} ms (± {ex_std:.2f})')

            if records is not None:
                row: Dict[str, Any] = {
                    'label': label,
                    'suffix': suffix,
                    'runs': runs,
                    'total_sync_ms': sync_total_mean,
                    'total_sync_ms_std': sync_total_std,
                    'jit_ms': jit_mean,
                    'jit_ms_std': jit_std,
                    'codegen_ms': cg_mean,
                    'codegen_ms_std': cg_std,
                    'backend_ms': be_mean,
                    'backend_ms_std': be_std,
                    'execution_ms': ex_mean,
                    'execution_ms_std': ex_std,
                }
                if measure_async:
                    row['total_async_ms'] = _mean(async_total)
                    row['total_async_ms_std'] = _std(async_total)
                if args:
                    row['args'] = args
                if kwargs:
                    row['kwargs'] = kwargs
                records.append(row)

            return ret

        return wrapper

    return decorator


@contextmanager
def measure(
    label: str,
    records: Optional[List[Dict[str, Any]]] = None,
    clear_cache: bool = True,
):
    """
    Context manager: time a single Dr.Jit / Mitsuba code block.

    Be sure to use :py:func:`drjit.schedule` (or evaluate inside the block)
    so that the work you want to time is actually scheduled before the
    closing :py:func:`drjit.eval` / :py:func:`drjit.sync_thread`.

    Args:
        label: Name shown in log output and stored in the record.
        records: Optional list to which one summary dict is appended on
            exit. Keys: ``label``, ``runs`` (always ``1``),
            ``total_ms``, ``jit_ms``, ``codegen_ms``, ``backend_ms``,
            ``execution_ms``.
        clear_cache: Wipe the kernel cache before timing starts.

    Example:

    .. code-block:: python

       with dr.bench.measure(label='Rendering'):
           img = mi.render(scene, spp=512, seed=0)
    """
    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        with dr.scoped_set_flag(dr.JitFlag.LaunchBlocking, True):
            _log_info(f'Benchmarking: "{label}" ...')
            reset(clear_cache=clear_cache)
            t0 = time.perf_counter()
            yield
            dr.eval()
            dr.sync_thread()
            total_ms = (time.perf_counter() - t0) * 1000.0

            cg, be, ex = _kernel_history_split()
            jit_ms = total_ms - (cg + be + ex)

            _log_info(f'{label} benchmark results (single run):')
            _log_info(f'    - Total time (sync):  {total_ms:.2f} ms')
            _log_info(f'    - Jitting time:       {jit_ms:.2f} ms')
            _log_info(f'    - Codegen time:       {cg:.2f} ms')
            _log_info(f'    - Backend time:       {be:.2f} ms')
            _log_info(f'    - Execution time:     {ex:.2f} ms')

            if records is not None:
                records.append({
                    'label': label,
                    'runs': 1,
                    'total_ms': total_ms,
                    'jit_ms': jit_ms,
                    'codegen_ms': cg,
                    'backend_ms': be,
                    'execution_ms': ex,
                })
