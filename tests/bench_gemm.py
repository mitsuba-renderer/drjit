"""This benchmark compares three frameworks on a single-precision
``size x size`` matrix multiplication: Dr.Jit's :func:`drjit.matmul`,
NumPy's ``@`` operator, and (optionally) PyTorch's ``@`` operator.
The three supported transpose combos are exercised (``A @ B``,
``A @ B.T``, ``A.T @ B``); for each, the script reports execution
time, GFLOP/s, and the max elementwise error of the first output row
relative to NumPy.

Dr.Jit is exercised against every available backend (``cuda`` and
``llvm``); PyTorch is skipped when not installed.

Run with ``python tests/bench_gemm.py [size] [--runs N]``
(defaults: size=4096, runs=1).
"""
import argparse
import os
import statistics
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import drjit as dr

def _available_backends():
    backends = []
    try:
        from drjit.cuda import TensorXf as CUDATensor  # noqa: F401
        backends.append(('cuda', dr.cuda.TensorXf))
    except ImportError:
        pass
    try:
        from drjit.llvm import TensorXf as LLVMTensor  # noqa: F401
        backends.append(('llvm', dr.llvm.TensorXf))
    except ImportError:
        pass
    return backends


def _benchmark(size=4096, runs=1):
    import time

    import numpy as np

    try:
        import torch
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'


        if torch_device == 'cuda':
            if hasattr(torch.backends.cuda.matmul, 'fp32_precision'):
                torch.backends.cuda.matmul.fp32_precision = 'ieee'
                torch.backends.cudnn.conv.fp32_precision = 'ieee'
            else:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

    except ImportError:
        torch = None
        torch_device = None

    backends = _available_backends()
    if not backends:
        raise RuntimeError("No Dr.Jit backend available (need cuda or llvm).")

    _, GenTensorXf = backends[0]
    rng = dr.rng(seed=0)
    A_gen = rng.normal(GenTensorXf, (size, size))
    B_gen = rng.normal(GenTensorXf, (size, size))
    dr.eval(A_gen, B_gen)
    A_np = np.ascontiguousarray(A_gen.numpy())
    B_np = np.ascontiguousarray(B_gen.numpy())
    del A_gen, B_gen

    if torch is not None:
        A_t = torch.from_numpy(A_np).to(torch_device)
        B_t = torch.from_numpy(B_np).to(torch_device)

    flops = 2.0 * size * size * size
    print(f"size            : {size} x {size}")
    print(f"runs            : {runs} (median)")
    if torch is not None:
        print(f"PyTorch device  : {torch_device}")

    def time_dr(dr_call):
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            C = dr_call()
            dr.eval(C)
            dr.sync_thread()
            history = dr.kernel_history()
        return sum(h['execution_time'] for h in history), C

    def time_np(np_call):
        t0 = time.perf_counter()
        C = np_call()
        return (time.perf_counter() - t0) * 1e3, C

    def time_torch(torch_call):
        if torch_device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            C = torch_call()
            end.record()
            end.synchronize()
            return start.elapsed_time(end), C
        t0 = time.perf_counter()
        C = torch_call()
        return (time.perf_counter() - t0) * 1e3, C

    def run(label, dr_call, np_call, torch_call, ref_np, use_torch):
        # Warm up (JIT compile / BLAS planning)
        _, C = time_dr(dr_call)
        time_np(np_call)
        if use_torch:
            time_torch(torch_call)

        dr_samples = [time_dr(dr_call)[0] for _ in range(runs)]
        np_samples = [time_np(np_call)[0] for _ in range(runs)]
        if use_torch:
            torch_samples = [time_torch(torch_call)[0] for _ in range(runs)]

        dr_ms = statistics.median(dr_samples)
        np_ms = statistics.median(np_samples)
        dr_gflops = flops / (dr_ms * 1e-3) / 1e9
        np_gflops = flops / (np_ms * 1e-3) / 1e9

        if use_torch:
            torch_ms = statistics.median(torch_samples)
            torch_gflops = flops / (torch_ms * 1e-3) / 1e9

        max_err = float(np.max(np.abs(C.numpy()[0] - ref_np[0])))

        print()
        print(f"  [{label}]")
        print(f"    Dr.Jit time     : {dr_ms:7.2f} ms   ({dr_gflops:7.1f} GFLOP/s)")
        if use_torch:
            print(f"    PyTorch time    : {torch_ms:7.2f} ms   ({torch_gflops:7.1f} GFLOP/s)")
            print(f"    PyTorch / Dr.Jit: {dr_ms / torch_ms:.2f}x")
        print(f"    NumPy time      : {np_ms:7.2f} ms   ({np_gflops:7.1f} GFLOP/s)")
        print(f"    Dr.Jit / NumPy  : {np_ms / dr_ms:.2f}x")
        print(f"    max error (row0): {max_err:.3e}")
        assert np.allclose(C.numpy(), ref_np, atol=1e-2), f"{label}: result mismatch vs NumPy"

    for backend_name, TensorXf in backends:
        A = TensorXf(A_np)
        B = TensorXf(B_np)
        dr.eval(A, B)

        # Skip PyTorch for the LLVM backend: the CPU path's baseline is
        # NumPy (BLAS), and PyTorch's CPU BLAS is the same code path, so
        # including it only adds noise.
        use_torch = torch is not None and backend_name != 'llvm'

        print()
        print(f"==== Dr.Jit backend: {backend_name} ====")

        run("A @ B",
            lambda: dr.matmul(A, B),
            lambda: A_np @ B_np,
            lambda: A_t @ B_t if use_torch else None,
            A_np @ B_np, use_torch)
        run("A @ B.T",
            lambda: dr.matmul(A, B, Bt=True),
            lambda: A_np @ B_np.T,
            lambda: A_t @ B_t.T if use_torch else None,
            A_np @ B_np.T, use_torch)
        run("A.T @ B",
            lambda: dr.matmul(A, B, At=True),
            lambda: A_np.T @ B_np,
            lambda: A_t.T @ B_t if use_torch else None,
            A_np.T @ B_np, use_torch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('size', nargs='?', type=int, default=4096,
                        help='Matrix side length (default: 4096)')
    parser.add_argument('--runs', '-r', type=int, default=1,
                        help='Number of timed runs per measurement; the median '
                             'is reported (default: 1)')
    args = parser.parse_args()
    _benchmark(size=args.size, runs=args.runs)
