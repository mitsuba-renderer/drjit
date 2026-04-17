import pytest
import drjit as dr
import drjit.nn as nn
import sys


def skip_if_coopvec_not_supported(t):
    backend = dr.backend_v(t)
    vt = dr.type_v(t)
    if backend == dr.JitBackend.CUDA:
        if dr.detail.cuda_version() < (12, 8):
            pytest.skip(
                "CUDA driver does not support cooperative vectors (Driver R570) or later is required"
            )
        if vt != dr.VarType.Float16:
            pytest.skip(f"CUDA does not support cooperative vectors of type {vt}")
    elif backend == dr.JitBackend.LLVM:
        if dr.detail.llvm_version() < (17, 0):
            pytest.skip(
                f"LLVM version {dr.detail.llvm_version()} does not support"
                " cooperative vectors, LLVM 17.0 or later is required"
            )
        if vt not in [dr.VarType.Float16, dr.VarType.Float32]:
            pytest.skip(f"LLVM does not support cooperative vectors of type {vt}")


@pytest.test_arrays("is_diff,float,shape=(),is_tensor,float")
@pytest.mark.parametrize("layout", ["training", "inference"])
@pytest.mark.parametrize("bias", [True, False])
def test01_simple_linear(t, layout, bias):
    skip_if_coopvec_not_supported(t)
    rng = dr.rng(42)

    # Create simple linear layer
    layer = nn.Linear(-1, 8, bias=bias)
    layer = layer.alloc(t, 4, rng=rng)

    # Pack with both methods
    packed_nn, _ = nn.pack(layer, layout=layout)
    packed_diff = nn.diff_pack(layer, layout=layout)

    assert dr.all(packed_nn == packed_diff)


@pytest.test_arrays("is_diff,float,shape=(),is_tensor,float")
@pytest.mark.parametrize("layout", ["training", "inference"])
@pytest.mark.parametrize("bias", [True, False])
def test02_sequential_networks(t, layout, bias):
    skip_if_coopvec_not_supported(t)
    rng = dr.rng(42)

    mlp = nn.Sequential(
        nn.Linear(-1, 32, bias=bias),
        nn.ReLU(),
        nn.Linear(32, 16, bias=bias),
        nn.ReLU(),
        nn.Linear(16, 3, bias=bias),
    )
    mlp = mlp.alloc(t, 2, rng=rng)

    packed_nn, _ = nn.pack(mlp, layout=layout)
    packed_diff = nn.diff_pack(mlp, layout=layout)

    assert dr.all(packed_nn == packed_diff)


@pytest.test_arrays("is_diff,float,shape=(),is_tensor,float")
@pytest.mark.parametrize("layout", ["training", "inference"])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("n_hidden", [4, 8, 32, 64])
def test03_sequential_large(t, layout, bias, n_hidden):
    skip_if_coopvec_not_supported(t)
    rng = dr.rng(42)

    mlp = []
    mlp.append(nn.Linear(-1, 32, bias=bias))
    mlp.append(nn.ReLU())
    for i in range(n_hidden):
        mlp.append(nn.Linear(-1, -1, bias=bias))
        mlp.append(nn.ReLU())
    mlp.append(nn.Linear(-1, 3, bias=bias))
    mlp = nn.Sequential(*mlp)

    mlp = mlp.alloc(t, 2, rng=rng)

    packed_nn, _ = nn.pack(mlp, layout=layout)
    packed_diff = nn.diff_pack(mlp, layout=layout)

    assert dr.all(packed_nn == packed_diff)


@pytest.test_arrays("is_diff,float,shape=(),is_tensor,float")
def test04_gradient_equality(t):
    """
    Tests that the gradients through the diffpack function are equivalent to
    the gradients when gathering using the indices.
    """
    skip_if_coopvec_not_supported(t)
    rng = dr.rng(42)

    mlp = nn.Sequential(
        nn.Linear(-1, 16),
        nn.ReLU(),
        nn.Linear(-1, 1),
    )
    mlp = mlp.alloc(t, 2, rng=rng)

    key = nn.compute_key(mlp, "training")

    # Enable gradients on unpacked weights
    dr.enable_grad(mlp[0].weights)
    dr.enable_grad(mlp[0].bias)
    dr.enable_grad(mlp[2].weights)
    dr.enable_grad(mlp[2].bias)

    packed, mlp_packed = nn.pack(mlp, layout="training")
    packed[:] = nn.diff_pack(mlp, layout="training")

    m = sys.modules[t.__module__]
    vt = dr.type_v(t)
    if vt == dr.VarType.Float16:
        Array2T = m.Array2f16
    elif vt == dr.VarType.Float32:
        Array2T = m.Array2f
    elif vt == dr.VarType.Float64:
        Array2T = m.Array2f64

    x = rng.random(Array2T, (2, 128))
    (y,) = mlp_packed(nn.CoopVec(x))
    dr.backward(y, dr.ADFlag.ClearNone)

    all_grads = dr.concat(
        [
            mlp[0].weights.grad.array,
            mlp[0].bias.grad.array,
            mlp[2].weights.grad.array,
            mlp[2].bias.grad.array,
        ]
    )

    indices = nn._registry[key]
    grad_repacked = dr.gather(
        type(all_grads), all_grads, indices, active=indices < dr.width(all_grads)
    )

    assert dr.all(packed.grad == grad_repacked)
