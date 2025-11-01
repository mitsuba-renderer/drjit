import pytest

import drjit as dr

pytest.importorskip("drjit.cuda")


def test_green_context_basic():
    if not dr.has_backend(dr.JitBackend.CUDA):
        pytest.skip("CUDA backend unavailable")

    import drjit.cuda as cuda

    size = 128 * 1024

    with dr.cuda.green_context(1) as ctx:
        x = dr.arange(cuda.Float, size)
        y = dr.sqrt(x + 1.0)
        dr.eval(y)

        assert ctx.sm_count >= 1
        assert ctx.requested_sm_count >= 1
        assert ctx.remaining_ctx is not None

    # Re-enter the same context to ensure multiple activations work
    with ctx:
        z = dr.arange(cuda.Float, 16)
        dr.eval(z)
