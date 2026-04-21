"""
Tests for the batched GEMM path behind ``dr.matmul`` on Dr.Jit tensors.

Organized into six sections, each with a specific focus:

- **A** (01-02): baseline 2-D correctness across every supported dtype.
- **B** (03):    empty / degenerate shapes; edge handling in the launchers.
- **C** (04):    large-matrix smoke test -- exercises CUDA tile selection
                 and edge handling at scale.
- **D** (05-06): ``@`` operator binding and input-validation errors.
- **E** (10-13): N-D and broadcast correctness via ``GemmBatch``.
- **F** (20-21): automatic differentiation (forward + reverse).

Each test documents which kernel / dispatch path it exists to cover so
that future edits can be made with a clear sense of what regressions
would go undetected if the test were changed or removed.
"""

import drjit as dr
import pytest


_TOL_BY_TYPE = {
    dr.VarType.Float16: 1e-1,
    dr.VarType.Float32: 1e-4,
    dr.VarType.Float64: 1e-10,
}

_FLOAT_DTYPE = {
    dr.VarType.Float16: 'float16',
    dr.VarType.Float32: 'float32',
    dr.VarType.Float64: 'float64',
}


def _seed(*key):
    return abs(hash(key)) & 0xffff


# =========================================================================
# Section A: baseline 2-D correctness
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float,-is_diff')
@pytest.mark.parametrize('At,Bt', [(False, False), (True, False),
                                   (False, True), (True, True)])
@pytest.mark.parametrize('dims', [
    (3, 5, 7),       # small, all odd:     BM=8, V=1, K-tail only
    (16, 16, 16),    # aligned power-of-2: BM=16, vectorized, aligned K
    (17, 23, 11),    # medium odd:         BM=8 fallback, bulk-K + K-tail
])
def test01_matmul_2d(t, dims, At, Bt):
    """Baseline 2-D correctness. Parameter sweep exercises all three
    float dtypes, all four ``(At, Bt)`` combinations, and three shape
    families chosen to cover distinct tile paths: the unvectorized
    ``BM=8`` path, the vectorized ``BM=16`` path with aligned K, and
    the odd-size alignment fallback where both the bulk-K loop and the
    K-tail run."""
    np = pytest.importorskip("numpy")
    M, K, N = dims
    vt = dr.type_v(t)
    dtype = _FLOAT_DTYPE[vt]
    rng = np.random.default_rng(seed=_seed(dims, At, Bt, vt))

    A_np = rng.standard_normal((K, M) if At else (M, K)).astype(dtype)
    B_np = rng.standard_normal((N, K) if Bt else (K, N)).astype(dtype)

    C = dr.matmul(t(A_np), t(B_np), At=At, Bt=Bt)

    A_eff = A_np.T if At else A_np
    B_eff = B_np.T if Bt else B_np
    ref = A_eff.astype(np.float64) @ B_eff.astype(np.float64)

    assert C.shape == (M, N)
    assert np.allclose(C.numpy(), ref, atol=_TOL_BY_TYPE[vt])


@pytest.test_arrays('is_tensor,jit,int32,-is_diff',
                    'is_tensor,jit,uint32,-is_diff')
def test02_matmul_int(t):
    """Integer matmul. int32 dispatches to the uint32 kernel (identical
    bit pattern under two's-complement multiply/add); bounded operands
    keep the exact result inside the 32-bit range."""
    np = pytest.importorskip("numpy")
    vt = dr.type_v(t)
    dtype = np.int32 if vt == dr.VarType.Int32 else np.uint32
    low, high = (-8, 8) if vt == dr.VarType.Int32 else (0, 16)
    rng = np.random.default_rng(seed=1)

    M, K, N = 12, 7, 9
    A_np = rng.integers(low, high, size=(M, K), dtype=dtype)
    B_np = rng.integers(low, high, size=(K, N), dtype=dtype)

    C = dr.matmul(t(A_np), t(B_np))
    assert C.shape == (M, N)
    assert np.array_equal(C.numpy(), A_np @ B_np)


# =========================================================================
# Section B: empty / degenerate shapes
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('A_shape,B_shape,out_shape', [
    # Size-1 extents: exercise the BM=8 / V=1 fallback in the kernel.
    ((1, 1),    (1, 1),    (1, 1)),
    ((1, 8),    (8, 1),    (1, 1)),     # M = N = 1
    ((8, 1),    (1, 8),    (8, 8)),     # K = 1 (outer-product shape)
    ((1, 5),    (5, 7),    (1, 7)),     # M = 1 only (row-vector @ matrix)
    ((7, 5),    (5, 1),    (7, 1)),     # N = 1 only (matrix @ col-vector)
    # Zero-sized matrix dims and empty batch: Python-layer
    # short-circuits that skip the kernel and return a zero tensor.
    ((0, 4),    (4, 5),    (0, 5)),     # M = 0 -> empty output
    ((3, 4),    (4, 0),    (3, 0)),     # N = 0 -> empty output
    ((3, 0),    (0, 5),    (3, 5)),     # K = 0 -> zero output
    ((0, 3, 4), (0, 4, 5), (0, 3, 5)),  # empty batch
])
def test03_degenerate(t, A_shape, B_shape, out_shape):
    """Shape edge cases: size-1 extents (which exercise the BM=8 /
    V=1 fallback in the CUDA kernel) and every variant of an empty
    operand (empty matrix dim, empty contraction, empty batch). For
    the non-empty-output cases (``K=0``) the result must be all
    zeros -- a sum over zero terms."""
    np = pytest.importorskip("numpy")
    A = dr.zeros(t, A_shape)
    B = dr.zeros(t, B_shape)
    C = dr.matmul(A, B)
    assert C.shape == out_shape
    if 0 not in out_shape:
        assert np.array_equal(C.numpy(),
                              np.zeros(out_shape, dtype=np.float32))


# =========================================================================
# Section C: large-matrix smoke test
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('M,K,N', [
    # Square: spans tile regimes from the odd-size BM=8 fallback up to
    # BM=64 with many occupancy waves.
    (123,  123,  123),
    (512,  512,  512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    # Rectangular: distinct M / K / N reach alignment and grid-coverage
    # combinations that square shapes don't produce (skinny or fat
    # outer dims, long contraction, odd M and N with aligned K).
    (2048, 128,  512),
    (64,   4096, 4096),
    (4096, 1024, 64),
    (127,  2048, 129),
    # One axis = 1 at scale: row-vector @ matrix (M=1), matrix @
    # col-vector (N=1), and outer-product shape (K=1).
    (1,    4096, 4096),
    (4096, 4096, 1),
    (2048, 1,    2048),
])
def test04_matmul_large(t, M, K, N):
    """Large matmul smoke test. Square sizes sweep the tile regimes
    (tiny odd hitting the BM=8 fallback up to BM=64 with many waves);
    rectangular sizes exercise alignment and grid-coverage paths that
    square shapes can't reach."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(M, K, N))
    scale = np.float32(1.0 / np.sqrt(K))
    A_np = (rng.standard_normal((M, K)) * scale).astype(np.float32)
    B_np = (rng.standard_normal((K, N)) * scale).astype(np.float32)

    C = dr.matmul(t(A_np), t(B_np))
    ref = A_np.astype(np.float64) @ B_np.astype(np.float64)
    assert C.shape == (M, N)
    assert np.allclose(C.numpy(), ref, atol=1e-3)


# =========================================================================
# Section D: operator binding and input-validation errors
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    ((7, 5),    (5, 9)),        # 2-D fast path
    ((3, 4, 5), (3, 5, 6)),     # N-D (GemmBatch) path
])
def test05_matmul_operator(t, A_shape, B_shape):
    """``@`` dispatches to ``dr.matmul`` at both ranks (2-D fast path
    and the batched ``GemmBatch`` path). The numerical result is
    already checked elsewhere; this test only guards the binding."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))
    A_np = rng.standard_normal(A_shape).astype(np.float32)
    B_np = rng.standard_normal(B_shape).astype(np.float32)
    C = t(A_np) @ t(B_np)
    assert np.allclose(C.numpy(), A_np @ B_np, atol=1e-4)


@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('A_shape,B_shape,At,Bt', [
    # Matrix-dim K mismatch.
    ((3, 4),            (5, 6),            False, False),
    ((3, 4),            (6, 5),            True,  True),
    # 0-D (scalar) tensor inputs are rejected.
    ((),                (5, 3),            False, False),
    ((3, 5),            (),                False, False),
    # Transpose flag is meaningless on a 1-D operand.
    ((5,),              (5, 3),            True,  False),
    ((3, 5),            (5,),              False, True),
    # Batch rank exceeds DRJIT_GEMM_MAX_BDIMS (= 6).
    ((1,) * 7 + (3, 4), (1,) * 7 + (4, 5), False, False),
    # Broadcast-incompatible batch prefixes.
    ((2, 3, 4),         (3, 4, 5),         False, False),
    ((4, 3, 4),         (5, 4, 6),         False, False),
])
def test06_matmul_errors(t, A_shape, B_shape, At, Bt):
    """Every ``ValueError``-producing input-validation branch: K
    mismatch in either transpose orientation, 0-D operands, transpose
    flag with a 1-D operand, rank above the batch-dim cap, and
    broadcast-incompatible batch prefixes."""
    with pytest.raises(ValueError):
        dr.matmul(dr.zeros(t, A_shape),
                  dr.zeros(t, B_shape),
                  At=At, Bt=Bt)


# =========================================================================
# Section E: N-D / broadcast correctness (GemmBatch path)
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float,-is_diff')
@pytest.mark.parametrize('At,Bt', [(False, False), (True, False),
                                   (False, True), (True, True)])
@pytest.mark.parametrize('batch', [
    (1,),                       # size-1 grid dim (not the 2-D fast path)
    (3,),                       # plain 1-D grid batch
    (2, 3),                     # multi-dim grid batch
    (2, 1, 2, 1, 1, 2),         # max rank = DRJIT_GEMM_MAX_BDIMS
])
def test10_batched_matmul(t, batch, At, Bt):
    """Equal-batch N-D matmul. Sweeps every float dtype, every
    ``(At, Bt)`` combination, and four batch ranks up to and including
    the ``DRJIT_GEMM_MAX_BDIMS = 6`` boundary. The ``(1,)`` case hits
    the ``GemmBatch`` path with a size-1 grid dim (distinct from the
    ``batch=()`` 2-D fast path)."""
    np = pytest.importorskip("numpy")
    vt = dr.type_v(t)
    dtype = _FLOAT_DTYPE[vt]
    M, K, N = 5, 7, 3

    a_mat = (K, M) if At else (M, K)
    b_mat = (N, K) if Bt else (K, N)
    rng = np.random.default_rng(seed=_seed(batch, At, Bt, vt))
    A_np = rng.standard_normal(batch + a_mat).astype(dtype)
    B_np = rng.standard_normal(batch + b_mat).astype(dtype)

    C = dr.matmul(t(A_np), t(B_np), At=At, Bt=Bt)
    A_eff = np.swapaxes(A_np, -1, -2) if At else A_np
    B_eff = np.swapaxes(B_np, -1, -2) if Bt else B_np
    ref = np.matmul(A_eff.astype(np.float64), B_eff.astype(np.float64))

    assert C.shape == batch + (M, N)
    assert np.allclose(C.numpy(), ref, atol=_TOL_BY_TYPE[vt])


@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    # A is a single matrix broadcast over B's batch.
    ((3, 4),       (5, 4, 6)),
    # B is a single matrix broadcast over A's batch.
    ((5, 3, 4),    (4, 6)),
    # A's leading batch dim is 1, broadcasting against B's dim 5.
    ((1, 3, 4),    (5, 4, 6)),
    # Cross-broadcast: each operand is size-1 along a distinct dim.
    ((2, 1, 3, 4), (1, 5, 4, 6)),
    # Rank difference: A is 2-D, B prepends three extra batch axes.
    ((3, 4),       (7, 3, 5, 4, 6)),
])
def test11_broadcast_shapes(t, A_shape, B_shape):
    """Every structurally distinct broadcast topology. Each case
    encodes one or more zero strides in ``GemmBatch``; collectively
    they cover full-operand broadcast, size-1 broadcast, cross-
    broadcast (both operands broadcast along different dims), and
    rank-difference prepending."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))
    A_np = rng.standard_normal(A_shape).astype(np.float32)
    B_np = rng.standard_normal(B_shape).astype(np.float32)

    C = dr.matmul(t(A_np), t(B_np))
    ref = np.matmul(A_np.astype(np.float64), B_np.astype(np.float64))
    assert C.shape == ref.shape
    assert np.allclose(C.numpy(), ref, atol=1e-4)


@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('At,Bt', [(False, False), (True, False),
                                   (False, True), (True, True)])
def test12_broadcast_matmul_transpose(t, At, Bt):
    """Broadcast matmul combined with each ``(At, Bt)`` combination.
    ``A`` is 2-D (broadcast along the batch axis) and ``B`` is 3-D,
    exercising the one-sided broadcast + transpose path through the
    ``GemmBatch`` spec."""
    np = pytest.importorskip("numpy")
    Bat, M, K, N = 4, 3, 5, 6
    rng = np.random.default_rng(seed=_seed(At, Bt))

    A_shape = (K, M) if At else (M, K)
    B_shape = (Bat,) + ((N, K) if Bt else (K, N))
    A_np = rng.standard_normal(A_shape).astype(np.float32)
    B_np = rng.standard_normal(B_shape).astype(np.float32)

    A_eff = A_np.T if At else A_np
    B_eff = np.swapaxes(B_np, -1, -2) if Bt else B_np
    ref = np.matmul(A_eff.astype(np.float64), B_eff.astype(np.float64))

    C = dr.matmul(t(A_np), t(B_np), At=At, Bt=Bt)
    assert C.shape == (Bat, M, N)
    assert np.allclose(C.numpy(), ref, atol=1e-4)


@pytest.test_arrays('is_tensor,jit,int32,-is_diff',
                    'is_tensor,jit,uint32,-is_diff')
def test13_batched_matmul_int(t):
    """Batched integer matmul -- dispatches the uint32 kernel through
    the batched (``GemmBatch``) path rather than the 2-D fast path."""
    np = pytest.importorskip("numpy")
    vt = dr.type_v(t)
    dtype = np.int32 if vt == dr.VarType.Int32 else np.uint32
    low, high = (-8, 8) if vt == dr.VarType.Int32 else (0, 16)
    Bat, M, K, N = 4, 5, 3, 6
    rng = np.random.default_rng(seed=7)
    A_np = rng.integers(low, high, size=(Bat, M, K), dtype=dtype)
    B_np = rng.integers(low, high, size=(Bat, K, N), dtype=dtype)

    C = dr.matmul(t(A_np), t(B_np))
    assert C.shape == (Bat, M, N)
    assert np.array_equal(C.numpy(), A_np @ B_np)


@pytest.test_arrays('is_tensor,jit,float32,-is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    # 1-D x 2-D: A acts as (1, K), the prepended axis is dropped.
    ((5,),       (5, 7)),
    # 2-D x 1-D: B acts as (K, 1), the appended axis is dropped.
    ((4, 5),     (5,)),
    # 1-D x N-D: A broadcasts across B's batch.
    ((5,),       (3, 5, 7)),
    # N-D x 1-D: B broadcasts across A's batch (matrix-vector batch).
    ((3, 4, 5),  (5,)),
    # Both: 1-D vector contracted against a stack of matrices.
    ((5,),       (2, 3, 5, 7)),
    ((2, 3, 4, 5), (5,)),
    # 1-D x 1-D: inner product, returned as a 0-D scalar tensor.
    ((5,),       (5,)),
])
def test14_matmul_1d(t, A_shape, B_shape):
    """1-D operand handling (NumPy ``matmul`` semantics): a 1-D ``A``
    is augmented to ``(1, K)`` and the prepended axis is dropped from
    the output; a 1-D ``B`` is augmented to ``(K, 1)`` and the appended
    axis is dropped. Two 1-D operands collapse to a 0-D inner product.
    Also exercises 1-D x batched-N-D and batched-N-D x 1-D, where the
    vector broadcasts across the batch via zero strides."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))
    A_np = rng.standard_normal(A_shape).astype(np.float32)
    B_np = rng.standard_normal(B_shape).astype(np.float32)

    C = dr.matmul(t(A_np), t(B_np))
    ref = np.matmul(A_np.astype(np.float64), B_np.astype(np.float64))
    assert C.shape == ref.shape
    assert np.allclose(C.numpy(), ref, atol=1e-4)


# =========================================================================
# Section F: automatic differentiation
# =========================================================================

@pytest.test_arrays('is_tensor,jit,float32,is_diff')
@pytest.mark.parametrize('At,Bt', [(False, False), (True, False),
                                   (False, True), (True, True)])
@pytest.mark.parametrize('batch', [(), (3,), (2, 3)])
def test20_matmul_grad(t, batch, At, Bt):
    """Forward- and reverse-mode AD through a matmul. Sweeping ``batch``
    covers the 2-D fast path (``()``), a 1-D grid batch, and a
    multi-dim grid batch. Sweeping all four ``(At, Bt)`` combinations
    exercises the backward-GEMM branches inside
    ``BatchedGemmEdge::backward``. No operand is broadcast here, so
    ``n_rdims = 0`` throughout."""
    np = pytest.importorskip("numpy")
    M, K, N = 4, 5, 6
    rng = np.random.default_rng(seed=_seed(batch, At, Bt))

    a_mat = (K, M) if At else (M, K)
    b_mat = (N, K) if Bt else (K, N)
    A_np  = rng.standard_normal(batch + a_mat).astype(np.float32)
    B_np  = rng.standard_normal(batch + b_mat).astype(np.float32)
    dC_np = rng.standard_normal(batch + (M, N)).astype(np.float32)
    dA_np = rng.standard_normal(A_np.shape).astype(np.float32)

    A_eff = np.swapaxes(A_np, -1, -2) if At else A_np
    B_eff = np.swapaxes(B_np, -1, -2) if Bt else B_np
    grad_A_ref = np.matmul(dC_np, np.swapaxes(B_eff, -1, -2))
    if At:
        grad_A_ref = np.swapaxes(grad_A_ref, -1, -2)
    grad_B_ref = np.matmul(np.swapaxes(A_eff, -1, -2), dC_np)
    if Bt:
        grad_B_ref = np.swapaxes(grad_B_ref, -1, -2)
    dA_eff = np.swapaxes(dA_np, -1, -2) if At else dA_np
    dC_ref = np.matmul(dA_eff, B_eff)

    # Reverse mode
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    C = dr.matmul(A, B, At=At, Bt=Bt)
    assert C.shape == batch + (M, N)
    dr.set_grad(C, t(dC_np))
    dr.backward_to(A, B)
    assert np.allclose(dr.grad(A).numpy(), grad_A_ref, atol=1e-3)
    assert np.allclose(dr.grad(B).numpy(), grad_B_ref, atol=1e-3)

    # Forward mode: tangent along A only (B has zero tangent).
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    dr.set_grad(A, t(dA_np))
    C = dr.matmul(A, B, At=At, Bt=Bt)
    dr.forward_to(C)
    assert np.allclose(dr.grad(C).numpy(), dC_ref, atol=1e-3)


@pytest.test_arrays('is_tensor,jit,float32,is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    # 1-D broadcast: backward for A has n_rdims = 1.
    ((4, 5),       (6, 5, 7)),
    # 2-D broadcast: backward for A has n_rdims = 2.
    ((4, 5),       (2, 3, 5, 7)),
    # Mixed: A broadcasts on one batch axis but not on another.
    # Backward for A has n_bdims = 1 and n_rdims = 1.
    ((2, 1, 4, 5), (2, 3, 5, 7)),
])
def test21_broadcast_matmul_grad(t, A_shape, B_shape):
    """Forward + reverse AD through a broadcast matmul. The reverse
    gradient for a broadcast operand folds its sum-over-batch into the
    backward GEMM's contraction via ``n_rdims > 0``. The three cases
    hit pure 1-D reduce, pure 2-D reduce, and a mixed
    ``(n_bdims, n_rdims) = (1, 1)`` configuration."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))

    A_np  = rng.standard_normal(A_shape).astype(np.float32)
    B_np  = rng.standard_normal(B_shape).astype(np.float32)

    # Output batch = broadcast of the two batch prefixes.
    C_batch = np.broadcast_shapes(A_shape[:-2], B_shape[:-2])
    C_shape = C_batch + (A_shape[-2], B_shape[-1])
    dC_np = rng.standard_normal(C_shape).astype(np.float32)
    dA_np = rng.standard_normal(A_shape).astype(np.float32)

    # Sum ``x`` along every leading / size-1 axis that isn't in
    # ``target_shape``; used to project reverse-broadcast gradients
    # back to their source operand's shape.
    def _sum_to(x, target_shape):
        ndim_extra = x.ndim - len(target_shape)
        if ndim_extra > 0:
            x = x.sum(axis=tuple(range(ndim_extra)))
        for i, ti in enumerate(target_shape):
            if ti == 1 and x.shape[i] != 1:
                x = x.sum(axis=i, keepdims=True)
        return x

    grad_A_ref = _sum_to(np.matmul(dC_np, np.swapaxes(B_np, -1, -2)),
                         A_shape)
    grad_B_ref = _sum_to(np.matmul(np.swapaxes(A_np, -1, -2), dC_np),
                         B_shape)
    dC_ref = np.matmul(dA_np, B_np)

    # Reverse mode
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    C = dr.matmul(A, B)
    assert C.shape == C_shape
    dr.set_grad(C, t(dC_np))
    dr.backward_to(A, B)
    assert np.allclose(dr.grad(A).numpy(), grad_A_ref, atol=1e-3)
    assert np.allclose(dr.grad(B).numpy(), grad_B_ref, atol=1e-3)

    # Forward mode: A's tangent broadcasts across every batch of C.
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    dr.set_grad(A, t(dA_np))
    C = dr.matmul(A, B)
    dr.forward_to(C)
    assert np.allclose(dr.grad(C).numpy(), dC_ref, atol=1e-3)


@pytest.test_arrays('is_tensor,jit,float32,is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    ((5,),       (5, 7)),       # 1-D A dispatches through M=1 GEMM
    ((4, 5),     (5,)),         # 1-D B dispatches through N=1 GEMM
    ((5,),       (5,)),         # 1-D x 1-D dispatches to reduce_dot
    ((3, 4, 5),  (5,)),         # batched matmul, broadcast 1-D B
])
def test22_matmul_1d_grad(t, A_shape, B_shape):
    """AD through each 1-D operand path: vector-matrix, matrix-vector,
    inner product (dispatched to ``reduce_dot``), and batched
    matrix-times-vector (broadcast 1-D B, backward for B sum-reduces
    over the batch via ``n_rdims``)."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))

    A_np  = rng.standard_normal(A_shape).astype(np.float32)
    B_np  = rng.standard_normal(B_shape).astype(np.float32)
    C_shape = np.matmul(A_np, B_np).shape
    dC_np = rng.standard_normal(C_shape).astype(np.float32)
    dA_np = rng.standard_normal(A_shape).astype(np.float32)

    # Reference gradients via the augmented (always 2-D-or-batched)
    # form: matmul gradients are standard, then sum out leading
    # broadcast axes and reshape back to the source shape.
    A2 = A_np[None] if A_np.ndim == 1 else A_np
    B2 = B_np[..., None] if B_np.ndim == 1 else B_np
    dC2 = dC_np.reshape((A2 @ B2).shape)

    def _drop(x, shape, aug_shape):
        extra = x.ndim - len(aug_shape)
        if extra > 0:
            x = x.sum(axis=tuple(range(extra)))
        return x.reshape(shape)
    grad_A_ref = _drop(dC2 @ B2.swapaxes(-1, -2), A_shape, A2.shape)
    grad_B_ref = _drop(A2.swapaxes(-1, -2) @ dC2, B_shape, B2.shape)
    dC_ref     = (dA_np.reshape(A2.shape) @ B2).reshape(C_shape)

    # Reverse mode
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    C = dr.matmul(A, B)
    assert C.shape == C_shape
    dr.set_grad(C, t(dC_np))
    dr.backward_to(A, B)
    assert np.allclose(dr.grad(A).numpy(), grad_A_ref, atol=1e-3)
    assert np.allclose(dr.grad(B).numpy(), grad_B_ref, atol=1e-3)

    # Forward mode
    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    dr.set_grad(A, t(dA_np))
    C = dr.matmul(A, B)
    dr.forward_to(C)
    assert np.allclose(dr.grad(C).numpy(), dC_ref, atol=1e-3)


@pytest.test_arrays('is_tensor,jit,float32,is_diff')
@pytest.mark.parametrize('A_shape,B_shape', [
    ((3, 2),    (2,)),           # matrix-vector, 1-D output
    ((3, 2),    (2, 4)),         # full 2-D GEMM
    ((4, 3, 5), (4, 5, 2)),      # batched 2-D GEMM
    ((3, 5),    (2, 4, 5, 7)),   # broadcast on A: backward has n_rdims > 0
])
def test23_matmul_scalar_loss_grad(t, A_shape, B_shape):
    """Reverse AD with a scalar loss seeded from ``dr.backward``. The
    output's ``grad`` is a size-1 literal, which the backward GEMM must
    broadcast up to ``target->size`` before dispatching the kernel --
    otherwise ``jit_var_batched_gemm`` reads past a 1-element buffer and
    only the first row/column of the operand gradient gets populated.
    Exercises the 1-D output path, the 2-D fast path, a batched GEMM,
    and a broadcast case (``n_rdims > 0`` on the A-edge backward)."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(seed=_seed(A_shape, B_shape))

    A_np = rng.standard_normal(A_shape).astype(np.float32)
    B_np = rng.standard_normal(B_shape).astype(np.float32)

    def _sum_to(x, target_shape):
        ndim_extra = x.ndim - len(target_shape)
        if ndim_extra > 0:
            x = x.sum(axis=tuple(range(ndim_extra)))
        for i, ti in enumerate(target_shape):
            if ti == 1 and x.shape[i] != 1:
                x = x.sum(axis=i, keepdims=True)
        return x

    # Promote 1-D operands to the 2-D form numpy-matmul uses internally,
    # compute the gradient in that form, and demote the inserted axis.
    A2 = A_np[None, :]     if A_np.ndim == 1 else A_np
    B2 = B_np[:, None]     if B_np.ndim == 1 else B_np
    C2 = np.matmul(A2, B2)
    dC2 = np.ones_like(C2)
    grad_A_ref = _sum_to(np.matmul(dC2, np.swapaxes(B2, -1, -2)), A2.shape)
    grad_B_ref = _sum_to(np.matmul(np.swapaxes(A2, -1, -2), dC2), B2.shape)
    if A_np.ndim == 1:
        grad_A_ref = grad_A_ref[0]
    if B_np.ndim == 1:
        grad_B_ref = grad_B_ref[:, 0]

    A, B = t(A_np), t(B_np)
    dr.enable_grad(A, B)
    C = dr.matmul(A, B)
    dr.backward(dr.sum(C))
    assert np.allclose(dr.grad(A).numpy(), grad_A_ref, atol=1e-3)
    assert np.allclose(dr.grad(B).numpy(), grad_B_ref, atol=1e-3)
