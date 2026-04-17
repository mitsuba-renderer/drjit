"""Compare matmul performance on Dr.Jit vs NumPy vs (optionally) PyTorch
across one or more ``size x size`` shapes and all three transpose combos
(``A @ B``, ``A @ B.T``, ``A.T @ B``). Results are emitted as a single
Markdown table with one row per (Dr.Jit backend, dtype, op, size) cell.
The ``Rel.`` column compares Dr.Jit to an appropriate reference (NumPy
for the LLVM backend, PyTorch for the CUDA backend).

Run with
    python tests/bench_gemm.py [sizes...] [--runs N] [--backend LIST]
                               [--dtype LIST]
where ``sizes`` is one or more matrix side lengths (default:
``128 256 512 1024 2048 4096``), ``--runs`` defaults to 10 (the median
is reported), ``--backend`` is a comma-separated list of backends from
``{llvm, cuda}`` (or ``all``, the default), and ``--dtype`` is a
comma-separated list of element types from ``{f16, f32, f64, i32, u32}``
(or ``all``, default ``f32``).
"""
import argparse
import os
import statistics
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import drjit as dr


# The three transpose variants: (label, dr.matmul kwargs, plain '@' form).
OPS = (
    ('A @ B',   {},           lambda A, B: A @ B),
    ('A @ B.T', {'Bt': True}, lambda A, B: A @ B.T),
    ('A.T @ B', {'At': True}, lambda A, B: A.T @ B),
)


# Per-dtype configuration:
#   tensor: Dr.Jit tensor attribute name
#   numpy:  NumPy dtype name
#   torch:  PyTorch dtype name (or None if unsupported by ``torch.matmul``)
#   atol:   tolerance for float cross-check (integers compare exactly)
#   kind:   data-generator selector ('float', 'signed', 'unsigned')
DTYPES = {
    'f16': dict(tensor='TensorXf16', numpy='float16', torch='float16',
                atol=1.0, kind='float'),
    'f32': dict(tensor='TensorXf',   numpy='float32', torch='float32',
                atol=1e-2, kind='float'),
    'f64': dict(tensor='TensorXf64', numpy='float64', torch='float64',
                atol=1e-8, kind='float'),
    'i32': dict(tensor='TensorXi',   numpy='int32',   torch='int32',
                atol=0,    kind='signed'),
    'u32': dict(tensor='TensorXu',   numpy='uint32',  torch=None,
                atol=0,    kind='unsigned'),
}


# Output columns. Row-identity fields (backend/type/op/size) arrive as
# strings; numeric fields go through _fmt_g or _fmt_pct (when `rel` is
# set). Stored numeric values are ``(gflops, rsd)`` pairs; ``_fmt_g``
# appends ``±RSD%`` to its cell when ``--rsd`` is active.
COLUMNS = (
    ('Size',          'size',      None,        'right'),
    ('Backend',       'backend',   None,        'left'),
    ('Type',          'type',      None,        'left'),
    ('Op',            'op',        None,        'left'),
    ('Dr.Jit',        'drjit',     None,        'right'),
    ('PyTorch/NumPy', 'reference', None,        'right'),
    ('Rel.',          'drjit',     'reference', 'right'),
)


def _parse_csv_choices(value, valid, name):
    items = [v.strip() for v in value.split(',') if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"Empty {name} list")
    if 'all' in items:
        return list(valid)
    seen, out = set(), []
    for item in items:
        if item not in valid:
            raise argparse.ArgumentTypeError(
                f"Invalid {name} {item!r}: choose from "
                f"{', '.join(valid)}, all")
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _available_backends(backend_names, tensor_attr):
    """Resolve ``backend_names`` (user order) to ``[(name, TensorT), ...]``."""
    out = []
    for bn in backend_names:
        if bn == 'llvm' and dr.has_backend(dr.JitBackend.LLVM):
            import drjit.llvm  # noqa: F401
            out.append(('llvm', getattr(dr.llvm, tensor_attr)))
        elif bn == 'cuda' and dr.has_backend(dr.JitBackend.CUDA):
            import drjit.cuda  # noqa: F401
            out.append(('cuda', getattr(dr.cuda, tensor_attr)))
    return out


def _setup_torch():
    try:
        import torch
    except ImportError:
        return None, None
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Disable TF32 so PyTorch's f32 matmul is bit-comparable to ours.
        if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
            torch.backends.cuda.matmul.fp32_precision = 'ieee'
            torch.backends.cudnn.conv.fp32_precision = 'ieee'
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    return torch, device


def _bench_cell(size, runs, backends, torch, torch_device, cfg, torch_dtype):
    """Measure one (size, dtype) cell.

    Returns ``{op: {engine: (gflops, rsd)}}`` where ``engine`` is one
    of ``numpy``, ``pytorch``, ``drjit_llvm``, ``drjit_cuda``."""
    import numpy as np

    np_dtype = getattr(np, cfg['numpy'])
    kind = cfg['kind']
    _, GenTensor = backends[0]

    rng = dr.rng(seed=0)
    if kind == 'float':
        A_gen = rng.normal(GenTensor, (size, size))
        B_gen = rng.normal(GenTensor, (size, size))
    else:
        # Keep the value range small so accumulated products stay well
        # within int32/uint32 range for the largest sizes in use.
        low, high = (-8, 8) if kind == 'signed' else (0, 16)
        A_gen = rng.integers(GenTensor, (size, size), low=low, high=high)
        B_gen = rng.integers(GenTensor, (size, size), low=low, high=high)
    dr.eval(A_gen, B_gen)
    A_np = np.ascontiguousarray(A_gen.numpy().astype(np_dtype, copy=False))
    B_np = np.ascontiguousarray(B_gen.numpy().astype(np_dtype, copy=False))
    del A_gen, B_gen

    use_torch = torch_dtype is not None and torch_device == 'cuda'
    if use_torch:
        A_t = torch.from_numpy(A_np).to(torch_device)
        B_t = torch.from_numpy(B_np).to(torch_device)

    flops = 2.0 * size ** 3

    def time_dr(call, count):
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            for _ in range(count):
                call()
            history = dr.kernel_history()
        # Each matmul emits a fixed number of kernel entries (pack-B +
        # row-sweep per K-segment), so the history splits evenly.
        assert len(history) % count == 0, (
            f"kernel_history length {len(history)} is not a multiple of "
            f"count {count}: each call is expected to emit the same "
            f"number of kernel entries")
        k = len(history) // count
        return [sum(history[i * k + j]['execution_time'] for j in range(k))
                for i in range(count)]

    def time_np(call, count):
        samples = []
        for _ in range(count):
            t0 = time.perf_counter()
            call()
            samples.append((time.perf_counter() - t0) * 1e3)
        return samples

    def time_torch(call, count):
        # Enqueue all ``count`` calls on the CUDA stream back-to-back,
        # bracketed by ``count + 1`` events (one boundary event reused
        # as the end of call i and the start of call i+1), and
        # synchronize once at the end.
        events = [torch.cuda.Event(enable_timing=True)
                  for _ in range(count + 1)]
        events[0].record()
        for i in range(count):
            call()
            events[i + 1].record()
        torch.cuda.synchronize()
        return [events[i].elapsed_time(events[i + 1]) for i in range(count)]

    WARMUP = 1

    def bench(timer, call):
        # Take ``runs + WARMUP`` samples and discard the warm-up prefix.
        # Returns a ``(gflops, rsd)`` pair; rsd is ``None`` with one sample.
        samples = timer(call, runs + WARMUP)[WARMUP:]
        med = statistics.median(samples)
        rsd = (100 * statistics.stdev(samples) / statistics.mean(samples)
               if len(samples) >= 2 else None)
        return flops / (med * 1e-3) / 1e9, rsd

    cell = {op: {} for op, _, _ in OPS}

    # Sleep ``IDLE_S`` before each timed bench so the previously-active
    # framework's worker pool fully parks (default OpenBLAS timeout
    # ~200 ms; nanothread's is 20 ms; pick the larger).
    IDLE_S = 0.25

    if any(bn == 'llvm' for bn, _ in backends):
        for op, _, plain in OPS:
            time.sleep(IDLE_S)
            cell[op]['numpy'] = bench(time_np, lambda p=plain: p(A_np, B_np))

    # PyTorch on GPU (independent of Dr.Jit backend).
    if use_torch:
        for op, _, plain in OPS:
            time.sleep(IDLE_S)
            cell[op]['pytorch'] = bench(
                time_torch, lambda p=plain: p(A_t, B_t))

    for bn, TensorT in backends:
        A = TensorT(A_np)
        B = TensorT(B_np)
        dr.eval(A, B)
        for op, kwargs, _ in OPS:
            time.sleep(IDLE_S)
            cell[op][f'drjit_{bn}'] = bench(
                time_dr, lambda kw=kwargs: dr.matmul(A, B, **kw))
        del A, B

    return cell


def _fmt_g(v, show_rsd=False):
    if v is None:
        return '—'
    gf, rsd = v
    out = f"{gf:>5.0f} GFLOP/s"
    if show_rsd:
        out += f" ±{rsd:4.1f}%" if rsd is not None else "       "
    return out


def _fmt_pct(v, ref):
    if v is None or ref is None or not v[0] or not ref[0]:
        return '—'
    return f"{100 * v[0] / ref[0]:.0f}%"


def _render_cell(col, data, show_rsd=False):
    _, key, rel, _ = col
    v = data.get(key)
    if rel is not None:
        return _fmt_pct(v, data.get(rel))
    return v if isinstance(v, str) else _fmt_g(v, show_rsd)


def _print_table(rows, columns):
    headers = [c[0] for c in columns]
    aligns = [c[3] for c in columns]
    widths = [max(len(h), *(len(r[i]) for r in rows))
              for i, h in enumerate(headers)]

    def pad(text, w, align):
        return text.ljust(w) if align == 'left' else text.rjust(w)

    print('| ' + ' | '.join(pad(h, w, 'left')
                            for h, w in zip(headers, widths)) + ' |')
    seps = [(':' + '-' * (w + 1)) if a == 'left' else ('-' * (w + 1) + ':')
            for w, a in zip(widths, aligns)]
    print('|' + '|'.join(seps) + '|')
    for row in rows:
        print('| ' + ' | '.join(pad(c, w, a)
                                for c, w, a in zip(row, widths, aligns)) + ' |')


def _run_all(sizes, runs, backend_names, dtype_names, show_rsd=False):
    # PyTorch is only used as the reference for the CUDA backend, so skip
    # importing it entirely when CUDA isn't in play.
    if 'cuda' in backend_names and dr.has_backend(dr.JitBackend.CUDA):
        torch, torch_device = _setup_torch()
    else:
        torch, torch_device = None, None

    # Resolve per-dtype backends and torch dtype.
    setups = {}
    for name in dtype_names:
        cfg = DTYPES[name]
        backends = _available_backends(backend_names, cfg['tensor'])
        if not backends:
            raise RuntimeError(
                f"No Dr.Jit backend available for dtype={name!r}, "
                f"backends={backend_names!r}.")
        tname = cfg['torch']
        torch_dtype = (getattr(torch, tname, None)
                       if torch is not None and tname is not None else None)
        setups[name] = dict(cfg=cfg, backends=backends, torch_dtype=torch_dtype)

    # Config header.
    print(f"runs           : {runs} (median)")
    print(f"dtypes         : {', '.join(dtype_names)}")
    print(f"sizes          : {', '.join(str(s) for s in sizes)}")
    if torch is None:
        print("PyTorch        : not installed")
    elif torch_device != 'cuda':
        print("PyTorch        : CPU-only install (skipped)")
    else:
        print(f"PyTorch device : {torch_device}")
    print()

    # Measurement phase: one cell per (dtype, size), with a progress line.
    cells = [(d, s) for d in dtype_names for s in sizes]
    n = len(cells)
    results = {}
    for i, (name, size) in enumerate(cells, 1):
        setup = setups[name]
        engines = []
        if any(bn == 'llvm' for bn, _ in setup['backends']):
            engines.append('numpy')
        if setup['torch_dtype'] is not None and torch_device == 'cuda':
            engines.append('pytorch')
        engines.extend(f'drjit_{bn}' for bn, _ in setup['backends'])
        print(f"[{i}/{n}] {size}x{size} {name} "
              f"({', '.join(engines)}) ...", flush=True)
        results[name, size] = _bench_cell(
            size, runs, setup['backends'], torch, torch_device,
            setup['cfg'], setup['torch_dtype'])

    # Output phase. Row order: backend (outer) → type → size → op (inner).
    # Reference is NumPy for LLVM rows, PyTorch for CUDA rows.
    ordered_backends = [bn for bn, _ in setups[dtype_names[0]]['backends']]
    rows = []
    for bn in ordered_backends:
        ref_key = 'numpy' if bn == 'llvm' else 'pytorch'
        for name in dtype_names:
            for s in sizes:
                for op, _, _ in OPS:
                    data = results[name, s].get(op, {})
                    rows.append([_render_cell(c, {
                        'backend':   bn.upper(),
                        'type':      name,
                        'op':        op,
                        'size':      str(s),
                        'drjit':     data.get(f'drjit_{bn}'),
                        'reference': data.get(ref_key),
                    }, show_rsd) for c in COLUMNS])

    print()
    _print_table(rows, COLUMNS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'sizes', nargs='*', type=int,
        default=[128, 256, 512, 1024, 2048, 4096],
        help='One or more matrix side lengths '
             '(default: 128 256 512 1024 2048 4096)')
    parser.add_argument(
        '--runs', '-r', type=int, default=10,
        help='Timed runs per measurement; median is reported (default: 10)')
    parser.add_argument(
        '--backend', '-b',
        type=lambda s: _parse_csv_choices(s, ('llvm', 'cuda'), 'backend'),
        default=['llvm', 'cuda'],
        help="Comma-separated list of Dr.Jit backends from "
             "{llvm, cuda} or 'all' (default: all available)")
    parser.add_argument(
        '--dtype', '-d',
        type=lambda s: _parse_csv_choices(s, tuple(DTYPES), 'dtype'),
        default=['f32'],
        help="Comma-separated list of element types from "
             f"{{{', '.join(DTYPES)}}} or 'all' (default: f32)")
    parser.add_argument(
        '--rsd', action='store_true',
        help='Append the relative standard deviation (stdev / mean of the '
             'per-call timings) as a suffix to each GFLOP/s value')
    args = parser.parse_args()

    _run_all(args.sizes, args.runs, args.backend, args.dtype, args.rsd)
