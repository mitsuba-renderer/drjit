import drjit as dr
import drjit.hgrid as hgrid
import pytest
import sys


@pytest.test_arrays("jit,shape=(*),float16,diff")
def test01_hash_grid_tcnn(t):
    """
    Tests that the hashgrid implementation produces the same results and gradients
    as the one in tinycudann.
    """
    try:
        import tinycudann as tcnn
    except ImportError:
        pytest.skip("This test requires PyTorch to be installed.")

    try:
        import torch
    except ImportError:
        pytest.skip("This test requires tinycudann to be installed.")

    m = sys.modules[t.__module__]
    Float16 = m.Float16
    Float32 = m.Float32
    ArrayXf = m.ArrayXf
    PCG32 = m.PCG32

    device = t(1, 2).torch().device

    n = 2**10

    config = {
        "hashmap_size": 2**16,
        "n_levels": 16,
        "base_resolution": 16,
        "per_level_scale": 2,
        "n_features_per_level": 2,
    }

    hg = hgrid.HashGridEncoding(
        3,
        **config,
        align_corners=False,
        torchngp_compat=True,
    )
    hg = hg.alloc(Float32)

    config = {
        "otype": "Grid",
        "type": "Hash",
        "n_levels": 16,
        "n_features_per_level": 2,
        "log2_hashmap_size": 16,
        "base_resolution": 16,
        "per_level_scale": 2,
        "interpolation": "Linear",
    }

    hg_ref = tcnn.Encoding(
        3,
        config,
        dtype = torch.float32,
    )
    data = hg_ref.params.data
    data = torch.linspace(0, 1_000, data.shape[0], device = device, dtype = torch.float32)
    hg_ref.params.data = data.cuda()

    for param in hg_ref.parameters():
        param.requires_grad_(True)

    hg.set_params(Float32(data.to(device=device)))
    dr.enable_grad(hg.data)

    sampler = PCG32(n)

    x = [sampler.next_float32(), sampler.next_float32(), sampler.next_float32()]
    dr.make_opaque(x)
    x_torch = torch.stack([xx.torch() for xx in x], dim=1).to(device = "cuda")

    res = hg(x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    ref = hg_ref(x_torch)
    end.record()

    torch.cuda.synchronize()

    ref = ref.to(device)

    res_torch = res.torch().permute(1, 0)

    assert torch.allclose(res_torch, ref, atol=0.00001)

    ## gradients

    res = ArrayXf(res)

    loss_res = dr.mean(dr.square(res - 1), axis=None)

    dr.backward(loss_res)

    loss_ref = torch.mean(torch.square(ref - 1), dim=None)

    loss_ref.backward()

    grad_res = dr.grad(hg.data).torch()
    grad_ref = hg_ref.params.grad.to(device=device)

    assert torch.allclose(grad_res, grad_ref, atol=0.001)

